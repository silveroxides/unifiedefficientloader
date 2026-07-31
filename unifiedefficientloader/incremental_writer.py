"""
Incremental safetensors writer for memory-efficient tensor saving.

Provides an I/O-bound, threaded background writer that streams tensor bytes 
to disk dynamically. It reserves a fixed-size header block at the start of the
file and pads the final JSON header to fit perfectly, completely eliminating
the need to calculate offsets upfront or hold massive tensors in RAM.
"""

import os
import json
import struct
import threading
import math
import ctypes
from concurrent.futures import ThreadPoolExecutor

from . import logging_utils
from .tensor_utils import get_dtype_size, torch_shape_to_st_shape, torch_to_st_dtype
logger = logging_utils.get_logger(__name__)

def _ensure_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError("The 'torch' package is required but not installed. Please install it.")

class IncrementalSafetensorsWriter:
    """Memory-efficient safetensors writer supporting dynamic streaming.

    A pure I/O sink that dynamically writes tensors to disk using background threads.
    It works by reserving a large, fixed-size block for the JSON header at the start 
    of the file. As you write tensors, they are appended sequentially to the data section.
    When the writer is closed, the JSON manifest is padded with spaces and written 
    into the reserved block.

    This completely eliminates the need to pre-calculate all tensor shapes and offsets
    before writing.

    Three write modes:

    ``write(name, tensor)``
        Single tensor. One key, one tensor, queued asynchronously.

    ``write_batch(batch)``
        List of ``(name, tensor)`` tuples. Offsets reserved atomically for the
        entire batch under one lock acquisition; workers submitted in parallel.
        Matches the per-batch yield format of ``async_stream``.

    ``write_dict(state_dict)``
        Standard PyTorch state dict ``{name: tensor}``. Converts to ordered
        batch and delegates to ``write_batch``.

    Usage:
        # Single tensor
        writer = IncrementalSafetensorsWriter("output.safetensors")
        with writer:
            for key in loader.keys():
                tensor = process(loader.get_tensor(key))
                writer.write(key, tensor)
                del tensor

        # Streaming batches from async_stream
        with IncrementalSafetensorsWriter("output.safetensors") as writer:
            for batch in loader.async_stream(keys, batch_size=16):
                writer.write_batch(batch)

        # Full state dict
        with IncrementalSafetensorsWriter("output.safetensors") as writer:
            writer.write_dict(model.state_dict())
    """

    @logging_utils.log_debug
    def __init__(self, filename: str, metadata: dict = None, max_header_bytes: int = 1024 * 1024, max_workers: int = 4):
        """
        Initialize the dynamic writer.
        
        Args:
            filename: Path to output safetensors file.
            metadata: Optional dictionary to embed in the safetensors header.
            max_header_bytes: Bytes to reserve at the start of the file for the JSON header.
                              Must be large enough to hold all tensor names, shapes, and offsets.
                              1MB (default) is typically enough for 10,000+ tensors.
            max_workers: Number of background threads to use for writing data.
        """
        self.filename = filename
        self.metadata = metadata or {}
        self.max_header_bytes = max_header_bytes
        
        # Enforce 8-byte alignment for the header block size
        if self.max_header_bytes % 8 != 0:
            self.max_header_bytes += 8 - (self.max_header_bytes % 8)
            
        self._manifest = {}
        if self.metadata:
            self._manifest["__metadata__"] = self.metadata
            
        self._current_data_offset = 0
        self._file = None
        self._max_workers = max_workers
        self._executor = None
        self._futures = []
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(max_workers * 2)

    def __enter__(self):
        os.makedirs(os.path.dirname(self.filename) or ".", exist_ok=True)
        self._file = open(self.filename, "wb")
        
        # Write the 8-byte header size
        self._file.write(struct.pack("<Q", self.max_header_bytes))
        
        # Seek past the reserved header block. The data section starts here.
        self._file.seek(8 + self.max_header_bytes)
        
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

            # Propagate any exceptions from the background threads
            for future in self._futures:
                try:
                    future.result()
                except Exception as e:
                    logging_utils.error(f"Async write failed: {e}")
                    if exc_type is None:
                        raise e

        if self._file and exc_type is None:
            # We are closing successfully. Time to write the header into the reserved space.
            self._finalize_header()
            
        if self._file:
            self._file.close()
            self._file = None

    def _finalize_header(self):
        """Write the collected manifest into the reserved header block with padding."""
        header_json = json.dumps(self._manifest, separators=(",", ":")).encode("utf-8")
        
        if len(header_json) > self.max_header_bytes:
            raise RuntimeError(
                f"Safetensors header size ({len(header_json)} bytes) exceeded reserved space "
                f"({self.max_header_bytes} bytes). Increase `max_header_bytes` during initialization."
            )
            
        # Pad with spaces to perfectly fill the reserved block
        pad_len = self.max_header_bytes - len(header_json)
        header_json += b" " * pad_len
        
        # Seek to byte 8 (just after the uint64 length prefix)
        self._file.seek(8)
        self._file.write(header_json)
        
        logging_utils.normal(
            f"Finalized '{self.filename}' with size "
            f"{(8 + self.max_header_bytes + self._current_data_offset) / (1024**2):.2f} MB "
            f"({len(self._manifest) - (1 if '__metadata__' in self._manifest else 0)} tensors)."
        )

    def _worker_write(self, offset_absolute, tensor):
        """Write a single tensor at a pre-computed absolute offset. Does not touch semaphore."""
        try:
            torch = _ensure_torch()
            # CPU move + contiguous enforcement happens here, off the calling thread
            tensor = tensor.cpu().contiguous()

            # Zero-copy write: build a ctypes array directly over the tensor's raw memory.
            # Mirrors the loader's readinto() trick — no intermediate Python bytes object,
            # no GIL-holding memcpy. OS copies straight from tensor memory to kernel buffer.
            byte_size = tensor.numel() * tensor.element_size()
            c_uint8_array = (ctypes.c_uint8 * byte_size).from_address(tensor.data_ptr())

            with self._lock:
                self._file.seek(offset_absolute)
                self._file.write(c_uint8_array)
        finally:
            del tensor

    def _batch_worker(self, items):
        """Fan out writes for a batch of (offset, tensor) pairs, then release semaphore once."""
        try:
            for offset_absolute, tensor in items:
                self._worker_write(offset_absolute, tensor)
        finally:
            self._semaphore.release()

    @logging_utils.log_debug
    def write_batch(self, batch: list):
        """
        Atomically reserve offsets for a batch of tensors and queue all writes asynchronously.

        Offset reservation for the entire batch is performed under a single lock acquisition,
        guaranteeing contiguous, correctly ordered data regions in the output file.
        Workers are submitted after the lock is released to avoid blocking other threads.

        Args:
            batch: List of (name, tensor) tuples. Matches the per-batch yield format of
                   ``UnifiedSafetensorsLoader.async_stream``.
        """
        if self._file is None:
            raise RuntimeError("Must be used within a context manager.")

        # Extract metadata only — no data movement in calling thread.
        # shape and dtype are available on any tensor regardless of device.
        prepared = []
        for name, tensor in batch:
            st_dtype = torch_to_st_dtype(tensor.dtype)
            tensor_shape = list(tensor.shape)
            shape = torch_shape_to_st_shape(tensor.shape, st_dtype)
            num_elements = math.prod(tensor_shape) if tensor_shape else 1
            byte_size = num_elements * get_dtype_size(st_dtype)
            prepared.append((name, st_dtype, shape, byte_size, tensor))

        # Atomically reserve offsets for all tensors in this batch
        absolute_offsets = []
        with self._lock:
            for name, st_dtype, shape, byte_size, _ in prepared:
                if name in self._manifest:
                    raise ValueError(f"Tensor '{name}' has already been written.")
                offset_start = self._current_data_offset
                offset_end = offset_start + byte_size
                self._manifest[name] = {
                    "dtype": st_dtype,
                    "shape": shape,
                    "data_offsets": [offset_start, offset_end],
                }
                self._current_data_offset += byte_size
                absolute_offsets.append(8 + self.max_header_bytes + offset_start)

        # Single semaphore acquire for the whole batch — no per-tensor blocking in caller.
        # _batch_worker fans out writes internally and releases the semaphore when done.
        items = [(offset, entry[4]) for offset, entry in zip(absolute_offsets, prepared)]
        self._semaphore.acquire()
        future = self._executor.submit(self._batch_worker, items)
        self._futures.append(future)

    @logging_utils.log_debug
    def write_dict(self, state_dict: dict):
        """
        Write all tensors from a state dict as a single batch operation.

        Converts the state dict to an ordered list of (name, tensor) pairs and
        delegates to ``write_batch`` for atomic offset reservation and async writes.

        Args:
            state_dict: Dict mapping tensor names to PyTorch tensors,
                        e.g. the output of ``model.state_dict()``.
        """
        self.write_batch(list(state_dict.items()))

    @logging_utils.log_debug
    def write(self, name: str, tensor):
        """
        Dynamically register and queue a tensor to be written to disk asynchronously.
        
        Args:
            name: The key name for the tensor in the safetensors file.
            tensor: The PyTorch tensor data. Must be contiguous and on CPU.
        """
        if self._file is None:
            raise RuntimeError("Must be used within a context manager.")
            
        if name in self._manifest:
            raise ValueError(f"Tensor '{name}' has already been written.")

        # Extract metadata only — no data movement in calling thread
        st_dtype = torch_to_st_dtype(tensor.dtype)
        tensor_shape = list(tensor.shape)
        shape = torch_shape_to_st_shape(tensor.shape, st_dtype)
        
        num_elements = math.prod(tensor_shape) if tensor_shape else 1
        byte_size = num_elements * get_dtype_size(st_dtype)
        
        # Record manifest entry
        offset_start = self._current_data_offset
        offset_end = offset_start + byte_size
        
        self._manifest[name] = {
            "dtype": st_dtype,
            "shape": shape,
            "data_offsets": [offset_start, offset_end]
        }
        
        # Advance the pointer for the next tensor
        self._current_data_offset += byte_size
        
        # Calculate absolute offset for the worker thread
        absolute_offset = 8 + self.max_header_bytes + offset_start

        self._semaphore.acquire()
        future = self._executor.submit(self._batch_worker, [(absolute_offset, tensor)])
        self._futures.append(future)
