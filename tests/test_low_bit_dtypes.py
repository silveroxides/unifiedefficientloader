import json
import struct

import pytest
import torch
from safetensors.torch import load_file, save_file

from unifiedefficientloader import IncrementalSafetensorsWriter, UnifiedSafetensorsLoader
from unifiedefficientloader.tensor_utils import get_dtype_size, st_shape_to_torch_shape, st_to_torch_dtype, torch_shape_to_st_shape, torch_to_st_dtype

LOW_BIT_DTYPES = [
    ("float8_e4m3fn", "F8_E4M3"),
    ("float8_e5m2", "F8_E5M2"),
    ("float8_e8m0fnu", "F8_E8M0"),
    ("float4_e2m1fn_x2", "F4"),
]


def _dtype_or_skip(name):
    dtype = getattr(torch, name, None)
    if dtype is None:
        pytest.skip(f"Installed PyTorch has no torch.{name}")
    return dtype


def _payload(path, key="low"):
    with open(path, "rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_size).decode("utf-8"))
        start, end = header[key]["data_offsets"]
        source.seek(8 + header_size + start)
        return header[key], source.read(end - start)


@pytest.mark.parametrize(("torch_name", "storage_code"), LOW_BIT_DTYPES)
def test_low_bit_dtype_mappings(torch_name, storage_code):
    dtype = _dtype_or_skip(torch_name)
    assert torch_to_st_dtype(dtype) == storage_code
    assert st_to_torch_dtype(storage_code) == dtype
    assert get_dtype_size(storage_code) == 1


def test_packed_fp4_shape_conversion():
    assert torch_shape_to_st_shape((2, 3, 4), "F4") == [2, 3, 8]
    assert st_shape_to_torch_shape((2, 3, 8), "F4") == (2, 3, 4)
    with pytest.raises(ValueError, match="at least one dimension"):
        torch_shape_to_st_shape((), "F4")
    with pytest.raises(ValueError, match="Invalid packed F4"):
        st_shape_to_torch_shape((2, 3, 7), "F4")


@pytest.mark.parametrize(("torch_name", "storage_code"), LOW_BIT_DTYPES)
@pytest.mark.parametrize("write_mode", ["single", "batch"])
def test_low_bit_streaming_and_writer_round_trip(tmp_path, torch_name, storage_code, write_mode):
    dtype = _dtype_or_skip(torch_name)
    source_path = tmp_path / f"source_{storage_code}.safetensors"
    output_path = tmp_path / f"output_{storage_code}_{write_mode}.safetensors"
    original = torch.zeros((2, 4), dtype=dtype)
    try:
        save_file({"low": original}, str(source_path))
    except (KeyError, RuntimeError, TypeError, ValueError):
        pytest.skip(f"Installed safetensors cannot write {dtype}")

    with UnifiedSafetensorsLoader(str(source_path), low_memory=True) as loader:
        assert loader.get_dtype("low") == dtype
        assert loader.get_shape("low") == original.shape
        loaded = loader.get_tensor("low")
        assert loaded.dtype == dtype
        assert loaded.shape == original.shape
        with IncrementalSafetensorsWriter(str(output_path)) as writer:
            if write_mode == "single":
                writer.write("low", loaded)
            else:
                writer.write_batch([("low", loaded)])

    source_header, source_payload = _payload(source_path)
    output_header, output_payload = _payload(output_path)
    assert source_header["dtype"] == output_header["dtype"] == storage_code
    assert source_header["shape"] == output_header["shape"]
    assert source_payload == output_payload
    round_tripped = load_file(str(output_path))["low"]
    assert round_tripped.dtype == dtype
    assert round_tripped.shape == original.shape
