"""
Tensor utility functions.

Provides serialization helpers for dictionary/tensor conversion.
Requires `torch`.
"""
import json
from typing import Dict, Tuple

from . import logging_utils

logger = logging_utils.get_logger(__name__)

def _ensure_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError("The 'torch' package is required but not installed. Please install it.")


@logging_utils.log_debug
def dict_to_tensor(data_dict: dict):
    """
    Convert a dictionary to a torch.uint8 tensor containing JSON bytes.

    Args:
        data_dict: Dictionary to serialize

    Returns:
        torch.uint8 tensor containing UTF-8 encoded JSON
    """
    torch = _ensure_torch()
    json_str = json.dumps(data_dict)
    byte_data = json_str.encode("utf-8")
    tensor_data = torch.tensor(list(byte_data), dtype=torch.uint8)
    logging_utils.debug(f"dict_to_tensor: serialized dict to uint8 tensor of shape {tensor_data.shape}")
    return tensor_data

@logging_utils.log_debug
def tensor_to_dict(tensor_data) -> dict:
    """
    Convert a torch.uint8 tensor containing JSON bytes to a dictionary.

    Args:
        tensor_data: Tensor containing UTF-8 encoded JSON bytes

    Returns:
        Parsed dictionary
    """
    if tensor_data.ndim != 1:
        raise ValueError(f"Expected a 1D tensor for dict conversion, got {tensor_data.ndim}D tensor.")
    byte_data = bytes(tensor_data.tolist())
    json_str = byte_data.decode("utf-8")
    data_dict = json.loads(json_str)
    logging_utils.debug(f"tensor_to_dict: deserialized tensor of shape {tensor_data.shape} to dict with keys: {list(data_dict.keys())}")
    return data_dict

def torch_to_st_dtype(dtype) -> str:
    """Map torch dtype to safetensors dtype string."""
    torch = _ensure_torch()
    mapping = {
        torch.float64: "F64",
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.int16: "I16",
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.bool: "BOOL",
        torch.complex64: "C64",
    }
    if hasattr(torch, "float8_e5m2"):
        mapping[torch.float8_e5m2] = "F8_E5M2"
    if hasattr(torch, "float8_e4m3fn"):
        mapping[torch.float8_e4m3fn] = "F8_E4M3"
    if hasattr(torch, "float8_e8m0fnu"):
        mapping[torch.float8_e8m0fnu] = "F8_E8M0"
    if hasattr(torch, "float4_e2m1fn_x2"):
        mapping[torch.float4_e2m1fn_x2] = "F4"
    if hasattr(torch, "uint64"):
        mapping[torch.uint64] = "U64"
    if hasattr(torch, "uint32"):
        mapping[torch.uint32] = "U32"
    if hasattr(torch, "uint16"):
        mapping[torch.uint16] = "U16"

    if dtype in mapping:
        return mapping[dtype]
    raise ValueError(f"Unsupported torch dtype: {dtype}")

def st_to_torch_dtype(dtype_str: str):
    """Map safetensors dtype string to torch dtype."""
    torch = _ensure_torch()
    dtype_map = {
        "F64": torch.float64,
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I64": torch.int64,
        "I32": torch.int32,
        "I16": torch.int16,
        "I8": torch.int8,
        "U8": torch.uint8,
        "BOOL": torch.bool,
        "C64": torch.complex64,
        "F8_E5M2": getattr(torch, "float8_e5m2", None),
        "F8_E4M3": getattr(torch, "float8_e4m3fn", None),
        "F8_E8M0": getattr(torch, "float8_e8m0fnu", None),
        "F4": getattr(torch, "float4_e2m1fn_x2", None),
        "U64": getattr(torch, "uint64", None),
        "U32": getattr(torch, "uint32", None),
        "U16": getattr(torch, "uint16", None),
    }

    dtype = dtype_map.get(dtype_str)
    if dtype is None:
        raise ValueError(f"Unsupported or unavailable dtype: {dtype_str}")
    return dtype

def get_dtype_size(st_dtype: str) -> int:
    """Get the byte size of a safetensors dtype string."""
    sizes = {
        "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
        "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
        "U64": 8, "U32": 4, "U16": 2,
        "BOOL": 1, "C64": 8,
        "F8_E5M2": 1, "F8_E4M3": 1, "F8_E8M0": 1, "F4": 1,
    }
    return sizes[st_dtype]


def torch_shape_to_st_shape(shape, st_dtype: str) -> list:
    """Convert a Torch tensor shape to its safetensors header shape."""
    shape = list(shape)
    if st_dtype != "F4":
        return shape
    if not shape:
        raise ValueError("Packed F4 tensors must have at least one dimension")
    shape[-1] *= 2
    return shape


def st_shape_to_torch_shape(shape, st_dtype: str) -> tuple:
    """Convert a safetensors header shape to its Torch tensor shape."""
    shape = list(shape)
    if st_dtype != "F4":
        return tuple(shape)
    if not shape or shape[-1] % 2:
        raise ValueError(f"Invalid packed F4 safetensors shape: {shape}")
    shape[-1] //= 2
    return tuple(shape)
