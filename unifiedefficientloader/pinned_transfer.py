"""
Pinned memory utilities for faster CPU→GPU tensor transfers.

Pinned (page-locked) memory enables faster DMA transfers to GPU.
Uses PyTorch's native pin_memory() with non_blocking transfers.
"""
from typing import Optional

from . import logging_utils

logger = logging_utils.get_logger(__name__)

def _ensure_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError("The 'torch' package is required but not installed. Please install it.")

# Module-level configuration
_verbose = False
_pinned_transfer_stats = {"pinned": 0, "fallback": 0}

def set_verbose(enabled: bool):
    """
    Enable/disable verbose output for pinned transfers.
    Also adjusts logging level to VERBOSE if enabled.
    """
    global _verbose
    _verbose = enabled
    if enabled:
        logging_utils.setup_logging("VERBOSE")

def get_pinned_transfer_stats():
    """Return pinned transfer statistics for verification."""
    return _pinned_transfer_stats.copy()

def reset_pinned_transfer_stats():
    """Reset transfer statistics."""
    global _pinned_transfer_stats
    _pinned_transfer_stats = {"pinned": 0, "fallback": 0}

@logging_utils.log_debug
def transfer_to_gpu_pinned(
    tensor,
    device: str = 'cuda',
    dtype = None
):
    """Transfer tensor to GPU using pinned memory for faster transfer."""
    torch = _ensure_torch()
    global _pinned_transfer_stats

    # Skip if not a CPU tensor or CUDA unavailable
    if tensor.device.type != 'cpu' or not torch.cuda.is_available():
        if dtype is not None:
            return tensor.to(device=device, dtype=dtype)
        return tensor.to(device=device)

    # Skip if target is not CUDA
    if not str(device).startswith('cuda'):
        if dtype is not None:
            return tensor.to(device=device, dtype=dtype)
        return tensor.to(device=device)

    try:
        pinned = tensor.pin_memory()

        if dtype is not None:
            result = pinned.to(device=device, dtype=dtype, non_blocking=True)
        else:
            result = pinned.to(device=device, non_blocking=True)

        torch.cuda.current_stream().synchronize()

        # One-time confirmation on first success
        if _pinned_transfer_stats["pinned"] == 0:
            logging_utils.verbose("[pinned_transfer] Pinned memory active - faster GPU transfers enabled")

        _pinned_transfer_stats["pinned"] += 1

        msg = f"[pinned_transfer] Pinned: {tensor.shape} ({tensor.numel() * tensor.element_size() / 1024:.1f} KB)"
        if _verbose:
            logging_utils.normal(msg)
        else:
            logging_utils.verbose(msg)

        return result

    except Exception as e:
        _pinned_transfer_stats["fallback"] += 1
        msg = f"[pinned_transfer] Fallback transfer to {device} due to error: {e}"
        if _verbose:
            logging_utils.warning(msg)
        else:
            logging_utils.verbose(msg)

        if dtype is not None:
            return tensor.to(device=device, dtype=dtype)
        return tensor.to(device=device)
