"""
Tensor utility functions.

Provides serialization helpers for dictionary/tensor conversion.
Requires `torch`.
"""
import json
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

def _ensure_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError("The 'torch' package is required but not installed. Please install it.")


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
    logger.debug(f"dict_to_tensor: serialized dict to uint8 tensor of shape {tensor_data.shape}")
    return tensor_data

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
    logger.debug(f"tensor_to_dict: deserialized tensor of shape {tensor_data.shape} to dict with keys: {list(data_dict.keys())}")
    return data_dict