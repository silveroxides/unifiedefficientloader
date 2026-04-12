"""
Unified safetensors loader with optional memory-efficient mode.

Provides a consistent interface for tensor loading regardless of mode.
Requires `torch`, `safetensors`, and optionally `tqdm`.
"""
import gc
import json
import struct
from typing import Dict, Optional, Tuple

from . import logging_utils

logger = logging_utils.get_logger(__name__)

def _ensure_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError("The 'torch' package is required but not installed. Please install it.")

def _ensure_safetensors():
    try:
        import safetensors
        from safetensors import safe_open
        return safe_open
    except ImportError:
        raise ImportError("The 'safetensors' package is required but not installed. Please install it.")

try:
    import torch
except ImportError:
    pass

class UnifiedSafetensorsLoader:
    """Unified safetensors loader supporting both preload and streaming modes.

    In standard mode (low_memory=False):
        - Loads all tensors upfront (fast, uses more RAM)
        - Tensors remain in memory until explicitly deleted

    In low-memory mode (low_memory=True):
        - Loads tensors on-demand via get_tensor()
        - Caller should delete tensors after processing

    Usage:
        with UnifiedSafetensorsLoader("model.safetensors", low_memory=True) as loader:
            for key in loader.keys():
                tensor = loader.get_tensor(key)
                # ... process tensor ...
                loader.mark_processed(key)  # Frees memory in low_memory mode
    """

    @logging_utils.log_debug
    def __init__(self, filename: str, low_memory: bool = False):
        """Initialize the loader.

        Args:
            filename: Path to safetensors file
            low_memory: If True, use streaming mode; if False, preload all tensors
        """
        torch = _ensure_torch()
        safe_open = _ensure_safetensors()

        self.filename = filename
        self.low_memory = low_memory
        self._tensors: Dict[str, 'torch.Tensor'] = {}
        self._all_keys = []
        self._file = None
        self._header = None
        self._header_size = None
        self._metadata: Dict[str, str] = {}

        if low_memory:
            # Streaming mode: read header only
            self._header, self._header_size = self._read_header()
            self._file = None # Opened lazily to support multiprocessing DataLoader
            self._all_keys = [k for k in self._header.keys() if k != "__metadata__"]
            # Extract metadata from header (safetensors stores it under __metadata__ key)
            self._metadata = self._header.get("__metadata__", {})
            logging_utils.verbose(f"Initialized Low-memory mode: parsed header of size {self._header_size} bytes.")
            logging_utils.verbose(f"Found {len(self._all_keys)} tensors (streaming mode)")
        else:
            # Standard mode: preload all tensors
            with safe_open(filename, framework="pt", device="cpu") as f:
                self._metadata = f.metadata() or {}
                self._all_keys = list(f.keys())
                logging_utils.normal(f"Loading {len(self._all_keys)} tensors from source file...")
                try:
                    from tqdm import tqdm
                    iterator = tqdm(self._all_keys, desc="Loading tensors", disable=not logger.isEnabledFor(logging_utils.NORMAL_LEVEL))
                except ImportError:
                    iterator = self._all_keys

                for key in iterator:
                    self._tensors[key] = f.get_tensor(key)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __getstate__(self):
        """Make loader picklable for multiprocessing DataLoaders."""
        state = self.__dict__.copy()
        state['_file'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def close(self):
        """Close file handle and release resources."""
        if self._file:
            self._file.close()
            self._file = None
        self._tensors.clear()

    def keys(self):
        """Return list of all tensor keys."""
        return self._all_keys

    def metadata(self) -> Dict[str, str]:
        """Return file metadata."""
        return self._metadata

    def get_shape(self, key: str) -> tuple:
        """Get tensor shape without loading tensor data.

        In low-memory mode, reads from header.
        In standard mode, returns shape from loaded tensor.
        """
        if self.low_memory:
            if key not in self._header:
                raise KeyError(f"Tensor '{key}' not found in file")
            return tuple(self._header[key]["shape"])
        else:
            return tuple(self._tensors[key].shape)

    def get_ndim(self, key: str) -> int:
        """Get tensor ndim without loading tensor data."""
        return len(self.get_shape(key))

    @logging_utils.log_debug
    def get_tensor(self, key: str) -> 'torch.Tensor':
        """Get a tensor by key.

        In standard mode, returns from cache.
        In low-memory mode, loads from file on-demand.
        """
        if not self.low_memory:
            # Standard mode: return from preloaded cache
            return self._tensors[key]

        # Low-memory mode: load on-demand
        if key not in self._header:
            raise KeyError(f"Tensor '{key}' not found in file")

        if self._file is None:
            self._file = open(self.filename, "rb")

        metadata = self._header[key]
        offset_start, offset_end = metadata["data_offsets"]

        if offset_start != offset_end:
            logging_utils.debug(f"Loading tensor '{key}' from offset {offset_start} to {offset_end} ({(offset_end - offset_start)} bytes)")
            self._file.seek(self._header_size + 8 + offset_start)
            # Use bytearray to create a writable buffer, avoiding PyTorch warning
            # about non-writable tensors from read-only bytes.
            tensor_bytes = bytearray(offset_end - offset_start)
            self._file.readinto(tensor_bytes)
        else:
            tensor_bytes = None

        return self._deserialize_tensor(tensor_bytes, metadata)

    def mark_processed(self, key: str):
        """Mark a tensor as processed, freeing memory if in low-memory mode.

        In standard mode, optionally deletes from cache.
        In low-memory mode, this is a no-op (tensor was never cached).
        """
        if not self.low_memory and key in self._tensors:
            del self._tensors[key]
            gc.collect()

    def _read_header(self):
        """Read and parse the safetensors header."""
        with open(self.filename, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_size).decode("utf-8")
        return json.loads(header_json), header_size

    def _deserialize_tensor(self, tensor_bytes, metadata):
        """Deserialize raw bytes into a torch tensor."""
        torch = _ensure_torch()
        dtype_str = metadata["dtype"]
        shape = metadata["shape"]
        dtype = self._get_torch_dtype(dtype_str)

        if tensor_bytes is None:
            byte_tensor = torch.empty(0, dtype=torch.uint8)
        else:
            byte_tensor = torch.frombuffer(tensor_bytes, dtype=torch.uint8)

        if dtype_str in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, dtype_str, shape)

        return byte_tensor.view(dtype).reshape(shape)

    @staticmethod
    def _get_torch_dtype(dtype_str: str):
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
        }
        if hasattr(torch, "float8_e5m2"):
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn

        dtype = dtype_map.get(dtype_str)
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        return dtype

    @staticmethod
    def _convert_float8(byte_tensor, dtype_str: str, shape: list):
        """Convert bytes to float8 tensor."""
        torch = _ensure_torch()
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        else:
            raise ValueError(f"Unsupported float8 type: {dtype_str}")


    def load_all(self):
        """Load all tensors as a dictionary.

        In low-memory mode: uses async_stream for parallel multi-threaded I/O.
        In standard mode: returns the preloaded tensor cache.

        Returns:
            Dict[str, torch.Tensor]: All tensors keyed by name.
        """
        if not self.low_memory:
            return dict(self._tensors)

        sd = {}
        for batch in self.async_stream(
            keys=self._all_keys,
            batch_size=16,
            prefetch_batches=2,
            pin_memory=False,
        ):
            for key, tensor in batch:
                sd[key] = tensor
        return sd

    def async_stream(self, keys: list, batch_size: int = 1, prefetch_batches: int = 2, pin_memory: bool = False):
        """Asynchronously stream tensors from disk.

        Args:
            keys: List of tensor keys to load
            batch_size: Number of tensors to yield in each batch
            prefetch_batches: Number of batches to pre-fetch in background
            pin_memory: If True, tensors will be pinned in CPU memory (sequentially in main thread)

        Yields:
            List of (key, tensor) tuples
        """
        import threading
        import queue
        from concurrent.futures import ThreadPoolExecutor

        torch = _ensure_torch()
        thread_local = threading.local()

        def get_file_handle():
            if not hasattr(thread_local, 'file'):
                thread_local.file = open(self.filename, "rb")
            return thread_local.file

        def _worker_load(key):
            try:
                # Direct thread-safe read
                metadata = self._header[key]
                offset_start, offset_end = metadata["data_offsets"]
                if offset_start != offset_end:
                    f = get_file_handle()
                    f.seek(self._header_size + 8 + offset_start)
                    tensor_bytes = bytearray(offset_end - offset_start)
                    f.readinto(tensor_bytes)
                else:
                    tensor_bytes = None

                tensor = self._deserialize_tensor(tensor_bytes, metadata)
                return key, tensor, None
            except Exception as e:
                # Fallback info for main thread
                return key, None, e

        # Queue for individual (key, tensor) pairs
        # Size it to hold enough for prefetch_batches
        q = queue.Queue(maxsize=prefetch_batches * batch_size)

        def _producer():
            # Use a reasonable number of workers for I/O bound tasks
            max_workers = min(16, max(4, batch_size))
            # Limit task submission to maintain backpressure on memory
            max_in_flight = max(max_workers, prefetch_batches * batch_size)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                key_iter = iter(keys)

                # Fill the pipeline
                for _ in range(max_in_flight):
                    try:
                        k = next(key_iter)
                        futures.append(executor.submit(_worker_load, k))
                    except StopIteration:
                        break

                while futures:
                    # Maintain order by taking the first future
                    f = futures.pop(0)
                    result = f.result() # Blocks until this specific tensor is loaded
                    q.put(result)       # Blocks if the consumption queue is full

                    # Submit next task if available
                    try:
                        k = next(key_iter)
                        futures.append(executor.submit(_worker_load, k))
                    except StopIteration:
                        pass

            q.put(None) # Sentinel

        producer_thread = threading.Thread(target=_producer, daemon=True)
        producer_thread.start()

        batch = []
        while True:
            res = q.get()
            if res is None:
                if batch:
                    yield batch
                break

            k, t, err = res
            if err is not None:
                logging_utils.warning(f"Async load failed for {k}, falling back to sync: {err}")
                # Fallback synchronous load
                try:
                    t = self.get_tensor(k)
                except Exception as sync_err:
                    logging_utils.error(f"Sync fallback also failed for {k}: {sync_err}")
                    raise sync_err

            # Pin memory sequentially in the main thread to avoid OS-level lock contention
            if pin_memory and t.device.type == 'cpu':
                try:
                    t = t.pin_memory()
                except Exception as e:
                    logging_utils.warning(f"Failed to pin memory for {k}: {e}")

            batch.append((k, t))
            if len(batch) == batch_size:
                yield batch
                batch = []

# Backward compatibility alias
MemoryEfficientSafeOpen = UnifiedSafetensorsLoader
