"""
Pinned memory pool manager.

Provides pre-allocated pinned buffers to avoid per-tensor allocation overhead.
"""
from typing import Optional
import torch
from . import logging_utils

logger = logging_utils.get_logger(__name__)

class PinnedBufferPool:
    """Manages a pool of pinned memory buffers for fast disk-to-GPU transfer."""
    def __init__(self, size_bytes: int, num_buffers: int):
        import torch
        self.size_bytes = size_bytes
        self.num_buffers = num_buffers

        # Determine whether to use PyTorch pinned memory or uel HostBuffer
        self.use_uel = False
        try:
            from .uel import control, host_buffer, torch as uel_torch
            if control.lib is not None:
                self.use_uel = True
                self._uel_host_buffer = host_buffer
                self._uel_torch = uel_torch
        except ImportError:
            pass

        logging_utils.verbose(f"Initializing PinnedBufferPool: {num_buffers} buffers of {size_bytes / (1024**2):.2f} MB each. Use UEL: {self.use_uel}")

        self.buffers = []
        for _ in range(num_buffers):
            if self.use_uel:
                hbuf = self._uel_host_buffer.HostBuffer(size_bytes)
                buf = self._uel_torch.hostbuf_to_tensor(hbuf)
                # Keep reference to HostBuffer to prevent garbage collection
                setattr(buf, "_uel_hostbuf_ref", hbuf)
                self.buffers.append(buf)
            else:
                buf = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True)
                self.buffers.append(buf)

        import queue
        self.free_queue = queue.Queue()
        for i in range(num_buffers):
            self.free_queue.put(i)

    def acquire(self) -> tuple[int, 'torch.Tensor']:
        """Acquire a free buffer. Blocks if empty."""
        idx = self.free_queue.get()
        return idx, self.buffers[idx]

    def release(self, idx: int):
        """Release buffer back to pool."""
        self.free_queue.put(idx)

