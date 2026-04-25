"""
Incremental safetensors writer for memory-efficient tensor saving.

Provides an I/O-bound, threaded background writer that pre-allocates
the file and streams tensor bytes to avoid RAM overhead.
"""

import os
import json
import struct
import threading
import math
from concurrent.futures import ThreadPoolExecutor

from . import logging_utils
from .tensor_utils import torch_to_st_dtype, get_dtype_size
logger = logging_utils.get_logger(__name__)

def _ensure_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError("The 'torch' package is required but not installed. Please install it.")

class IncrementalSafetensorsWriter:
    """Memory-efficient safetensors writer supporting incremental streaming.

    Features:
        - Structural template registration (clone existing models)
        - Pre-allocates the entire file to avoid fragmentation
        - Background thread pool for async I/O writing
        - Bounded queue to prevent RAM inflation during fast quantization

    Usage:
        writer = IncrementalSafetensorsWriter("output.safetensors")
        for key in loader.keys():
            writer.register_tensor(key, loader.get_shape(key), loader.get_dtype(key))
        writer.preallocate()

        with writer:
            for key in loader.keys():
                q_tensor = quantize(loader.get_tensor(key))
                writer.write_tensor(key, q_tensor)
    """

    @logging_utils.log_debug
    def __init__(self, filename: str, metadata: dict = None, max_workers: int = 4):
        self.filename = filename
        self.metadata = metadata or {}
        self._manifest = {}
        self._finalized = False
        self._header_size = 0
        self._file = None
        self._max_workers = max_workers
        self._executor = None
        self._futures = []
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(max_workers * 2)

    def register_tensor(self, name: str, shape: tuple, dtype):
        """Register a new tensor to the output manifest."""
        if self._finalized:
            raise RuntimeError("Cannot register tensors after preallocate() has been called.")

        torch = _ensure_torch()
        if isinstance(dtype, torch.dtype):
            st_dtype = torch_to_st_dtype(dtype)
        elif isinstance(dtype, str):
            st_dtype = dtype
        else:
            raise ValueError(f"Invalid dtype type: {type(dtype)}")

        self._manifest[name] = {
            "dtype": st_dtype,
            "shape": list(shape),
        }
        logging_utils.debug(f"Registered tensor {name}: {shape} {st_dtype}")

    @logging_utils.log_debug
    def preallocate(self):
        """Finalize the manifest, write the header, and pre-allocate the file."""
        if self._finalized:
            return

        sorted_keys = sorted(self._manifest.keys())
        current_offset = 0
        header_dict = {}

        if self.metadata:
            header_dict["__metadata__"] = self.metadata

        for key in sorted_keys:
            info = self._manifest[key]
            st_dtype = info["dtype"]
            shape = info["shape"]

            num_elements = math.prod(shape) if shape else 1
            byte_size = num_elements * get_dtype_size(st_dtype)

            info["data_offsets"] = [current_offset, current_offset + byte_size]
            header_dict[key] = info
            current_offset += byte_size

        # Serialize header and align to 8 bytes
        header_json = json.dumps(header_dict, separators=(",", ":")).encode("utf-8")

        # padding so that len(header_json) + pad is a multiple of 8
        pad_len = (8 - (len(header_json) % 8)) % 8
        if pad_len > 0:
            header_json += b" " * pad_len

        self._header_size = len(header_json)

        os.makedirs(os.path.dirname(self.filename) or ".", exist_ok=True)
        with open(self.filename, "wb") as f:
            f.write(struct.pack("<Q", self._header_size))
            f.write(header_json)

            total_size = 8 + self._header_size + current_offset
            f.truncate(total_size)

        self._finalized = True
        logging_utils.normal(
            f"Preallocated '{self.filename}' with size {total_size / (1024**2):.2f} MB "
            f"({len(self._manifest)} tensors)."
        )

    def __enter__(self):
        if not self._finalized:
            self.preallocate()
        self._file = open(self.filename, "r+b")
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

        if self._file:
            self._file.close()
            self._file = None

    def _worker_write(self, offset_start, expected_size, tensor):
        try:
            torch = _ensure_torch()
            byte_data = tensor.view(torch.uint8).numpy().tobytes()

            if len(byte_data) != expected_size:
                raise ValueError(
                    f"Byte size mismatch. Expected {expected_size}, got {len(byte_data)}"
                )

            with self._lock:
                self._file.seek(8 + self._header_size + offset_start)
                self._file.write(byte_data)
        finally:
            self._semaphore.release()
            del tensor

    @logging_utils.log_debug
    def write_tensor(self, name: str, tensor):
        """Queue a tensor to be written to disk asynchronously."""
        if not self._finalized or self._file is None:
            raise RuntimeError("Must be used within a context manager after preallocate().")

        if name not in self._manifest:
            raise KeyError(f"Tensor '{name}' not registered in manifest.")

        info = self._manifest[name]
        expected_shape = tuple(info["shape"])

        # Enforce CPU and contiguous
        tensor = tensor.cpu().contiguous()

        if tensor.shape != expected_shape:
            raise ValueError(
                f"Tensor '{name}' shape mismatch. Expected {expected_shape}, got {tensor.shape}"
            )

        st_dtype = torch_to_st_dtype(tensor.dtype)
        if st_dtype != info["dtype"]:
            raise ValueError(
                f"Tensor '{name}' dtype mismatch. Expected {info['dtype']}, got {st_dtype}"
            )

        offset_start, offset_end = info["data_offsets"]
        expected_size = offset_end - offset_start

        self._semaphore.acquire()
        future = self._executor.submit(
            self._worker_write, offset_start, expected_size, tensor
        )
        self._futures.append(future)

    def keys(self):
        """Return the keys registered in the manifest."""
        return list(self._manifest.keys())
