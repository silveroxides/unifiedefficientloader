"""
GPU memory slab allocator for contiguous tensor loading.
"""
from typing import Tuple
from . import logging_utils

logger = logging_utils.get_logger(__name__)

class GpuSlabAllocator:
    """Manages a large contiguous block of GPU memory."""
    def __init__(self, size_bytes: int, device: str = "cuda"):
        import torch
        self.device = device
        self.size_bytes = size_bytes
        self.current_offset = 0

        logging_utils.verbose(f"Allocating GPU slab: {size_bytes / (1024**2):.2f} MB on {device}")

        # Allocate raw bytes
        self.slab = torch.empty(size_bytes, dtype=torch.uint8, device=device)

    def allocate(self, num_bytes: int) -> Tuple[int, 'torch.Tensor']:
        """
        Allocate block. Returns (offset, tensor_view).
        """
        import torch

        # Align to 256 bytes for good memory access patterns
        align = 256
        aligned_bytes = (num_bytes + align - 1) // align * align

        if self.current_offset + aligned_bytes > self.size_bytes:
            raise RuntimeError(f"GPU Slab OOM. Need {aligned_bytes}, have {self.size_bytes - self.current_offset}")

        offset = self.current_offset
        self.current_offset += aligned_bytes

        # Return a view into the slab
        view = self.slab[offset:offset + num_bytes]
        return offset, view

    def reset(self):
        """Reset allocator offset."""
        self.current_offset = 0
