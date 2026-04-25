"""
Tests for UnifiedGGUFLoader and related GGUF support.

Coverage:
- File format detection / redirect from UnifiedSafetensorsLoader
- Metadata parsing (strings, ints, floats, bools, arrays)
- Float tensor loading: F32, F16, BF16 (preload + streaming + async_stream)
- Quantized tensor loading: GGMLTensor returned, correct shape / type
- Dequantization: Q4_0, Q8_0 produce numerically correct output
- load_all() returns (state_dict, extra) with arch_str + metadata
- Key / tensor_names API
- Context manager + close()
- Pickling (__getstate__ / __setstate__) for DataLoader multiprocessing
- GGMLTensor properties: shape, tensor_type, patches, to(), clone(), detach()
- is_quantized / is_torch_compatible helpers
- MMAP mode (skipped when uel not available)
- direct_gpu mode (skipped when CUDA not available)
"""

import os
import pickle
import struct
import tempfile
import warnings

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Availability guards
# ---------------------------------------------------------------------------

gguf = pytest.importorskip("gguf", reason="gguf package not installed")


def _uel_available() -> bool:
    try:
        from unifiedefficientloader.uel import control
        control.init()
        return control.lib is not None
    except Exception:
        return False


uel_required = pytest.mark.skipif(
    not _uel_available(), reason="uel native library not available"
)
cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

# ---------------------------------------------------------------------------
# Helpers — write minimal GGUF files
# ---------------------------------------------------------------------------

def _write_gguf(tensors_spec: dict, metadata: dict = None, arch: str = "llama") -> str:
    """Write a temporary GGUF file and return its path.

    tensors_spec: {name: np.ndarray} or
                  {name: (np.ndarray, GGMLQuantizationType)} for quantised.
    metadata:     optional dict of {key: value} to write as KV pairs.
    """
    fd, path = tempfile.mkstemp(suffix=".gguf")
    os.close(fd)

    writer = gguf.GGUFWriter(path, arch)
    if metadata:
        for k, v in metadata.items():
            if isinstance(v, str):
                writer.add_string(k, v)
            elif isinstance(v, bool):
                writer.add_bool(k, v)
            elif isinstance(v, int):
                writer.add_uint32(k, v)
            elif isinstance(v, float):
                writer.add_float32(k, v)

    for name, spec in tensors_spec.items():
        if isinstance(spec, tuple):
            arr, raw_dtype = spec
            writer.add_tensor(name, arr, raw_dtype=raw_dtype)
        else:
            writer.add_tensor(name, spec)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def _make_q4_0_blocks(n_blocks: int, scale: float = 1.0, nibble: int = 8):
    """Build raw Q4_0 bytes: each block has f16 scale + 16 quant bytes."""
    type_size = 18  # Q4_0 type_size
    raw = np.zeros((n_blocks, type_size), dtype=np.uint8)
    packed_scale = struct.pack("<e", scale)
    for i in range(n_blocks):
        raw[i, 0] = packed_scale[0]
        raw[i, 1] = packed_scale[1]
        raw[i, 2:] = (nibble & 0xF) | ((nibble & 0xF) << 4)
    return raw


def _make_q8_0_blocks(n_blocks: int, scale: float = 1.0, quant_val: int = 0):
    """Build raw Q8_0 bytes: each block has f16 scale + 32 int8 quant bytes."""
    type_size = 34  # Q8_0: 2 scale + 32 quant
    raw = np.zeros((n_blocks, type_size), dtype=np.uint8)
    packed_scale = struct.pack("<e", scale)
    for i in range(n_blocks):
        raw[i, 0] = packed_scale[0]
        raw[i, 1] = packed_scale[1]
        raw[i, 2:] = quant_val & 0xFF
    return raw


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def float_gguf():
    """GGUF with F32, F16 and BF16 tensors plus scalar metadata."""
    rng = np.random.default_rng(42)
    data_f32 = rng.standard_normal((4, 8)).astype(np.float32)
    data_f16 = rng.standard_normal((3, 6)).astype(np.float16)
    # BF16 must be written as uint16 raw bytes with raw_dtype=BF16
    data_bf16_f32 = rng.standard_normal((2, 4)).astype(np.float32)
    data_bf16_raw = data_bf16_f32.view(np.uint32) >> 16  # truncate to bf16 bits
    data_bf16_bytes = data_bf16_raw.astype(np.uint16).view(np.uint8).reshape(2, 4 * 2)

    path = _write_gguf(
        {
            "weight_f32": data_f32,
            "weight_f16": data_f16,
            "weight_bf16": (data_bf16_bytes, gguf.GGMLQuantizationType.BF16),
        },
        metadata={
            "test.str_val": "hello",
            "test.int_val": 42,
            "test.float_val": 3.14,
            "test.bool_val": True,
        },
    )
    yield path, {
        "weight_f32": torch.from_numpy(data_f32),
        "weight_f16": torch.from_numpy(data_f16),
    }, {
        "test.str_val": "hello",
        "test.int_val": 42,
        "test.bool_val": True,
    }
    # GGUFReader holds the mmap; on Windows the file may stay locked
    # until GC — tolerate the error.
    try:
        os.unlink(path)
    except PermissionError:
        pass


@pytest.fixture(scope="module")
def quant_gguf():
    """GGUF with Q4_0 and Q8_0 quantised tensors."""
    # Q4_0: 4 blocks x 32 = 128 elements, all zeros after dequant
    q4_blocks = _make_q4_0_blocks(n_blocks=4, scale=1.0, nibble=8)  # nibble 8 -> 8-8=0
    # Q8_0: 2 blocks x 32 = 64 elements, value=5 with scale=2.0 -> 10.0
    q8_blocks = _make_q8_0_blocks(n_blocks=2, scale=2.0, quant_val=5)

    path = _write_gguf({
        "q4_weight": (q4_blocks, gguf.GGMLQuantizationType.Q4_0),
        "q8_weight": (q8_blocks, gguf.GGMLQuantizationType.Q8_0),
    })
    yield path
    try:
        os.unlink(path)
    except PermissionError:
        pass


# ---------------------------------------------------------------------------
# 1. File format detection and redirect
# ---------------------------------------------------------------------------

class TestGGUFRedirect:
    def test_gguf_magic_bytes(self, float_gguf):
        path, _, _ = float_gguf
        with open(path, "rb") as f:
            assert f.read(4) == b"GGUF"

    def test_safetensors_loader_redirects_with_warning(self, float_gguf):
        from unifiedefficientloader import UnifiedSafetensorsLoader, UnifiedGGUFLoader
        path, _, _ = float_gguf
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loader = UnifiedSafetensorsLoader(path, low_memory=True)
        assert len(w) == 1
        assert "GGUF" in str(w[0].message)
        assert isinstance(loader, UnifiedGGUFLoader)
        loader.close()

    def test_safetensors_loader_normal_file_no_warning(self):
        """A genuine .safetensors file must not trigger the GGUF redirect."""
        from safetensors.torch import save_file
        from unifiedefficientloader import UnifiedSafetensorsLoader, UnifiedGGUFLoader
        fd, path = tempfile.mkstemp(suffix=".safetensors")
        os.close(fd)
        try:
            save_file({"w": torch.randn(4)}, path)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                loader = UnifiedSafetensorsLoader(path, low_memory=True)
            gguf_warns = [x for x in w if "GGUF" in str(x.message)]
            assert len(gguf_warns) == 0
            assert isinstance(loader, UnifiedSafetensorsLoader)
            assert not isinstance(loader, UnifiedGGUFLoader)
            loader.close()
        finally:
            try:
                os.unlink(path)
            except PermissionError:
                pass


# ---------------------------------------------------------------------------
# 2. Metadata
# ---------------------------------------------------------------------------

class TestGGUFMetadata:
    def test_metadata_strings(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path, _, expected_meta = float_gguf
        with UnifiedGGUFLoader(path) as loader:
            assert loader.metadata.get("test.str_val") == "hello"

    def test_metadata_int(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path, _, _ = float_gguf
        with UnifiedGGUFLoader(path) as loader:
            assert loader.metadata.get("test.int_val") == 42

    def test_metadata_bool(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path, _, _ = float_gguf
        with UnifiedGGUFLoader(path) as loader:
            assert loader.metadata.get("test.bool_val") is True

    def test_metadata_arch_in_extra(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path, _, _ = float_gguf
        with UnifiedGGUFLoader(path) as loader:
            _, extra = loader.load_all()
            assert extra["arch_str"] == "llama"

    def test_metadata_is_dict(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path, _, _ = float_gguf
        with UnifiedGGUFLoader(path) as loader:
            assert isinstance(loader.metadata, dict)


# ---------------------------------------------------------------------------
# 3. Keys / tensor_names API
# ---------------------------------------------------------------------------

class TestGGUFKeys:
    def test_keys_match_written(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path, tensors, _ = float_gguf
        with UnifiedGGUFLoader(path) as loader:
            assert set(loader.keys()) == set(tensors.keys()) | {"weight_bf16"}

    def test_tensor_names_alias(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path, _, _ = float_gguf
        with UnifiedGGUFLoader(path) as loader:
            assert loader.tensor_names == loader.keys()


# ---------------------------------------------------------------------------
# 4. Float tensor loading — preload mode
# ---------------------------------------------------------------------------

class TestGGUFPreload:
    def test_preload_f32_values(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path, tensors, _ = float_gguf
        with UnifiedGGUFLoader(path, low_memory=False) as loader:
            t = loader.get_tensor("weight_f32")
            assert t.dtype == torch.float32
            assert t.shape == tensors["weight_f32"].shape
            assert torch.allclose(t, tensors["weight_f32"])

    def test_preload_f16_values(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path, tensors, _ = float_gguf
        with UnifiedGGUFLoader(path, low_memory=False) as loader:
            t = loader.get_tensor("weight_f16")
            assert t.dtype == torch.float16
            assert torch.allclose(t, tensors["weight_f16"])

    def test_preload_all_keys_present(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path, tensors, _ = float_gguf
        with UnifiedGGUFLoader(path, low_memory=False) as loader:
            for key in tensors:
                _ = loader.get_tensor(key)  # must not raise


# ---------------------------------------------------------------------------
# 5. Float tensor loading — streaming (low_memory=True)
# ---------------------------------------------------------------------------

class TestGGUFStreaming:
    def test_stream_f32_values(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path, tensors, _ = float_gguf
        with UnifiedGGUFLoader(path, low_memory=True) as loader:
            t = loader.get_tensor("weight_f32")
            assert t.dtype == torch.float32
            assert torch.allclose(t, tensors["weight_f32"])

    def test_stream_f16_values(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path, tensors, _ = float_gguf
        with UnifiedGGUFLoader(path, low_memory=True) as loader:
            t = loader.get_tensor("weight_f16")
            assert torch.allclose(t, tensors["weight_f16"])

    def test_stream_missing_key_raises(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path, _, _ = float_gguf
        with UnifiedGGUFLoader(path, low_memory=False) as loader:
            with pytest.raises(KeyError):
                loader.get_tensor("does_not_exist")

    def test_mark_processed_no_error(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path, tensors, _ = float_gguf
        with UnifiedGGUFLoader(path, low_memory=True) as loader:
            loader.get_tensor("weight_f32")
            loader.mark_processed("weight_f32")  # should not raise


# ---------------------------------------------------------------------------
# 6. load_all
# ---------------------------------------------------------------------------

class TestGGUFLoadAll:
    def test_load_all_keys(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path, tensors, _ = float_gguf
        with UnifiedGGUFLoader(path, low_memory=True) as loader:
            sd, extra = loader.load_all()
        assert set(tensors.keys()).issubset(set(sd.keys()))

    def test_load_all_f32_values(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path, tensors, _ = float_gguf
        with UnifiedGGUFLoader(path, low_memory=True) as loader:
            sd, _ = loader.load_all()
        assert torch.allclose(sd["weight_f32"], tensors["weight_f32"])

    def test_load_all_extra_metadata(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path, _, _ = float_gguf
        with UnifiedGGUFLoader(path, low_memory=True) as loader:
            _, extra = loader.load_all()
        assert "arch_str" in extra
        assert "metadata" in extra
        assert isinstance(extra["metadata"], dict)

    def test_load_all_preload_mode(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path, tensors, _ = float_gguf
        with UnifiedGGUFLoader(path, low_memory=False) as loader:
            sd, _ = loader.load_all()
        assert torch.allclose(sd["weight_f32"], tensors["weight_f32"])


# ---------------------------------------------------------------------------
# 7. Async stream
# ---------------------------------------------------------------------------

class TestGGUFAsyncStream:
    def test_async_stream_yields_all_keys(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path, tensors, _ = float_gguf
        loaded = {}
        with UnifiedGGUFLoader(path, low_memory=True) as loader:
            for batch in loader.async_stream(list(tensors.keys()), batch_size=2):
                for key, t in batch:
                    loaded[key] = t
        assert set(loaded.keys()) == set(tensors.keys())

    def test_async_stream_f32_values(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path, tensors, _ = float_gguf
        loaded = {}
        with UnifiedGGUFLoader(path, low_memory=True) as loader:
            for batch in loader.async_stream(["weight_f32", "weight_f16"], batch_size=1):
                for key, t in batch:
                    loaded[key] = t
        assert torch.allclose(loaded["weight_f32"], tensors["weight_f32"])
        assert torch.allclose(loaded["weight_f16"], tensors["weight_f16"])

    def test_async_stream_batch_size_respected(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path, tensors, _ = float_gguf
        all_keys = list(tensors.keys())
        with UnifiedGGUFLoader(path, low_memory=True) as loader:
            batches = list(loader.async_stream(all_keys, batch_size=1))
        # Each batch has exactly 1 item (except possibly the last)
        for b in batches[:-1]:
            assert len(b) == 1


# ---------------------------------------------------------------------------
# 8. Quantised tensors — type checking
# ---------------------------------------------------------------------------

class TestGGUFQuantisedTypes:
    def test_q4_returns_ggmltensor(self, quant_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        from unifiedefficientloader.gguf_dequant import GGMLTensor
        path = quant_gguf
        with UnifiedGGUFLoader(path, low_memory=True) as loader:
            t = loader.get_tensor("q4_weight")
        assert isinstance(t, GGMLTensor), f"Expected GGMLTensor, got {type(t)}"

    def test_q8_returns_ggmltensor(self, quant_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        from unifiedefficientloader.gguf_dequant import GGMLTensor
        path = quant_gguf
        with UnifiedGGUFLoader(path, low_memory=True) as loader:
            t = loader.get_tensor("q8_weight")
        assert isinstance(t, GGMLTensor)

    def test_q4_tensor_type_set(self, quant_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path = quant_gguf
        with UnifiedGGUFLoader(path, low_memory=True) as loader:
            t = loader.get_tensor("q4_weight")
        assert t.tensor_type == gguf.GGMLQuantizationType.Q4_0

    def test_q4_logical_shape(self, quant_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path = quant_gguf
        with UnifiedGGUFLoader(path, low_memory=True) as loader:
            t = loader.get_tensor("q4_weight")
        # 4 blocks x 32 = 128 logical elements (written as shape [4, 32])
        assert t.shape == torch.Size([4, 32])

    def test_q8_logical_shape(self, quant_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path = quant_gguf
        with UnifiedGGUFLoader(path, low_memory=True) as loader:
            t = loader.get_tensor("q8_weight")
        assert t.shape == torch.Size([2, 32])

    def test_float_tensor_not_ggmltensor(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        from unifiedefficientloader.gguf_dequant import GGMLTensor
        path, _, _ = float_gguf
        with UnifiedGGUFLoader(path, low_memory=True) as loader:
            t = loader.get_tensor("weight_f32")
        assert not isinstance(t, GGMLTensor)
        assert isinstance(t, torch.Tensor)


# ---------------------------------------------------------------------------
# 9. Dequantisation — numerical correctness
# ---------------------------------------------------------------------------

class TestGGUFDequant:
    def test_q4_dequant_all_zeros(self, quant_gguf):
        """nibble=8 → 8-8=0, so all dequantised values should be 0."""
        from unifiedefficientloader import UnifiedGGUFLoader
        from unifiedefficientloader.gguf_dequant import dequantize_tensor
        path = quant_gguf
        with UnifiedGGUFLoader(path, low_memory=True) as loader:
            t = loader.get_tensor("q4_weight")
        deq = dequantize_tensor(t, dtype=torch.float32)
        assert deq.shape == torch.Size([4, 32])
        assert torch.all(deq == 0.0), f"Non-zero values: {deq[deq != 0]}"

    def test_q8_dequant_values(self, quant_gguf):
        """quant_val=5, scale=2.0 → 5*2.0=10.0 for all elements."""
        from unifiedefficientloader import UnifiedGGUFLoader
        from unifiedefficientloader.gguf_dequant import dequantize_tensor
        path = quant_gguf
        with UnifiedGGUFLoader(path, low_memory=True) as loader:
            t = loader.get_tensor("q8_weight")
        deq = dequantize_tensor(t, dtype=torch.float32)
        assert deq.shape == torch.Size([2, 32])
        assert torch.allclose(deq, torch.full_like(deq, 10.0), atol=1e-3), \
            f"Expected 10.0 everywhere, got: min={deq.min()}, max={deq.max()}"

    def test_is_quantized_true_for_ggmltensor(self, quant_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        from unifiedefficientloader.gguf_dequant import is_quantized
        path = quant_gguf
        with UnifiedGGUFLoader(path, low_memory=True) as loader:
            t = loader.get_tensor("q4_weight")
        assert is_quantized(t)

    def test_is_quantized_false_for_float(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        from unifiedefficientloader.gguf_dequant import is_quantized
        path, _, _ = float_gguf
        with UnifiedGGUFLoader(path, low_memory=True) as loader:
            t = loader.get_tensor("weight_f32")
        assert not is_quantized(t)


# ---------------------------------------------------------------------------
# 10. GGMLTensor properties
# ---------------------------------------------------------------------------

class TestGGMLTensor:
    @pytest.fixture
    def gmlt(self):
        from unifiedefficientloader.gguf_dequant import GGMLTensor
        data = torch.zeros(18, dtype=torch.uint8)  # 1 Q4_0 block worth of bytes
        return GGMLTensor(
            data,
            tensor_type=gguf.GGMLQuantizationType.Q4_0,
            tensor_shape=torch.Size([32]),
        )

    def test_shape_returns_logical(self, gmlt):
        assert gmlt.shape == torch.Size([32])

    def test_tensor_type_stored(self, gmlt):
        assert gmlt.tensor_type == gguf.GGMLQuantizationType.Q4_0

    def test_patches_default_empty(self, gmlt):
        assert gmlt.patches == []

    def test_to_preserves_attrs(self, gmlt):
        moved = gmlt.to(torch.float32)
        # After .to() the underlying storage changes dtype but attrs are preserved
        assert moved.tensor_type == gguf.GGMLQuantizationType.Q4_0
        assert moved.tensor_shape == torch.Size([32])

    def test_clone_returns_self(self, gmlt):
        """clone() is a no-op on GGMLTensor (intentional — matches city96/rattus128)."""
        c = gmlt.clone()
        assert c is gmlt

    def test_detach_returns_self(self, gmlt):
        d = gmlt.detach()
        assert d is gmlt

    def test_new_empty_returns_ggmltensor(self, gmlt):
        from unifiedefficientloader.gguf_dequant import GGMLTensor
        ne = gmlt.new_empty((16,))
        assert isinstance(ne, GGMLTensor)
        assert ne.tensor_shape == (16,)


# ---------------------------------------------------------------------------
# 11. Context manager and close
# ---------------------------------------------------------------------------

class TestGGUFContextManager:
    def test_context_manager(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path, tensors, _ = float_gguf
        with UnifiedGGUFLoader(path, low_memory=True) as loader:
            t = loader.get_tensor("weight_f32")
            assert t is not None
        # After __exit__, _tensors is cleared
        assert loader._tensors == {}

    def test_close_clears_state(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path, _, _ = float_gguf
        loader = UnifiedGGUFLoader(path, low_memory=False)
        assert len(loader._tensors) > 0
        loader.close()
        assert loader._tensors == {}


# ---------------------------------------------------------------------------
# 12. Pickling (DataLoader multiprocessing compatibility)
# ---------------------------------------------------------------------------

class TestGGUFPickle:
    def test_pickle_roundtrip(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path, tensors, _ = float_gguf
        loader = UnifiedGGUFLoader(path, low_memory=True)
        blob = pickle.dumps(loader)
        restored = pickle.loads(blob)
        # After unpickling, file handle and reader are None; lazy re-open on use
        assert restored._file is None
        assert restored._reader is None
        # But tensor_index and metadata are preserved
        assert set(restored._tensor_index.keys()) == set(loader._tensor_index.keys())
        # Can still read tensors after unpickling
        t = restored.get_tensor("weight_f32")
        assert torch.allclose(t, tensors["weight_f32"])
        loader.close()
        restored.close()


# ---------------------------------------------------------------------------
# 13. MMAP mode (requires uel)
# ---------------------------------------------------------------------------

class TestGGUFMMAP:
    @uel_required
    def test_mmap_float_values(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path, tensors, _ = float_gguf
        with UnifiedGGUFLoader(path, low_memory=True, use_mmap=True) as loader:
            assert loader.use_mmap is True
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t = loader.get_tensor("weight_f32")
            assert torch.allclose(t, tensors["weight_f32"])

    @uel_required
    def test_mmap_async_stream(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path, tensors, _ = float_gguf
        loaded = {}
        with UnifiedGGUFLoader(path, low_memory=True, use_mmap=True) as loader:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for batch in loader.async_stream(list(tensors.keys()), batch_size=2):
                    for key, t in batch:
                        loaded[key] = t
        assert set(loaded.keys()) == set(tensors.keys())
        assert torch.allclose(loaded["weight_f32"], tensors["weight_f32"])

    def test_mmap_fallback_when_uel_unavailable(self, float_gguf, monkeypatch):
        import unifiedefficientloader.uel.control as ctrl
        from unifiedefficientloader import UnifiedGGUFLoader
        path, tensors, _ = float_gguf
        monkeypatch.setattr(ctrl, "lib", None)
        monkeypatch.setattr(ctrl, "init", lambda: False)
        with UnifiedGGUFLoader(path, low_memory=True, use_mmap=True) as loader:
            assert loader.use_mmap is False
            t = loader.get_tensor("weight_f32")
            assert torch.allclose(t, tensors["weight_f32"])


# ---------------------------------------------------------------------------
# 14. direct_gpu mode (requires CUDA)
# ---------------------------------------------------------------------------

class TestGGUFDirectGPU:
    @cuda_required
    def test_direct_gpu_float_on_gpu(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path, tensors, _ = float_gguf
        loaded = {}
        with UnifiedGGUFLoader(path, low_memory=True, direct_gpu=True) as loader:
            for batch in loader.async_stream(
                ["weight_f32", "weight_f16"],
                batch_size=1,
            ):
                for key, t in batch:
                    loaded[key] = t

        assert loaded["weight_f32"].device.type == "cuda"
        assert torch.allclose(
            loaded["weight_f32"].cpu(), tensors["weight_f32"], atol=1e-5
        )

    @cuda_required
    def test_direct_gpu_quantized_is_ggmltensor_on_gpu(self, quant_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        from unifiedefficientloader.gguf_dequant import GGMLTensor
        path = quant_gguf
        loaded = {}
        with UnifiedGGUFLoader(path, low_memory=True, direct_gpu=True) as loader:
            for batch in loader.async_stream(["q4_weight"], batch_size=1):
                for key, t in batch:
                    loaded[key] = t
        t = loaded["q4_weight"]
        assert isinstance(t, GGMLTensor)
        assert t.device.type == "cuda"

    def test_direct_gpu_forces_low_memory(self, float_gguf):
        from unifiedefficientloader import UnifiedGGUFLoader
        path, _, _ = float_gguf
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            loader = UnifiedGGUFLoader(path, low_memory=False, direct_gpu=True)
        assert loader.low_memory is True
        loader.close()
