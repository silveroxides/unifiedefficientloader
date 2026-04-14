"""
GPU memory buffer pool for direct-to-GPU streaming.

Maintains a pool of pre-allocated GPU tensors to avoid allocation overhead
and ensure strictly bounded VRAM usage during streaming.
"""
from typing import Tuple, Optional
import torch
from . import logging_utils

logger = logging_utils.get_logger(__name__)

class GpuBufferPool:
    """Manages a pool of fixed-size GPU memory buffers."""
    def __init__(self, size_bytes: int, num_buffers: int, device: str = "cuda"):
        import torch
        import queue
        self.device = device
        self.size_bytes = size_bytes
        self.num_buffers = num_buffers

        logging_utils.verbose(f"Initializing GpuBufferPool: {num_buffers} buffers of {size_bytes / (1024**2):.2f} MB each on {device}.")

        self.buffers = []
        for _ in range(num_buffers):
            buf = torch.empty(size_bytes, dtype=torch.uint8, device=device)
            self.buffers.append(buf)

        self.free_queue = queue.Queue()
        for i in range(num_buffers):
            self.free_queue.put(i)

    def acquire(self) -> Tuple[int, 'torch.Tensor']:
        """Acquire a free buffer. Blocks if empty."""
        idx = self.free_queue.get()
        return idx, self.buffers[idx]

    def release(self, idx: int):
        """Release buffer back to pool."""
        self.free_queue.put(idx)
