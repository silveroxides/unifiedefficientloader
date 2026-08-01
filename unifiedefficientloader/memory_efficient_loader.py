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
from .tensor_utils import st_shape_to_torch_shape, st_to_torch_dtype

logger = logging_utils.get_logger(__name__)


def _ensure_torch():
    try:
        import torch

        return torch
    except ImportError:
        raise ImportError(
            "The 'torch' package is required but not installed. Please install it."
        )


def _ensure_safetensors():
    try:
        import safetensors
        from safetensors import safe_open

        return safe_open
    except ImportError:
        raise ImportError(
            "The 'safetensors' package is required but not installed. Please install it."
        )


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
    def __init__(
        self,
        filename: str,
        low_memory: bool = False,
        direct_gpu: bool = False,
        use_mmap: bool = False,
    ):
        """Initialize the loader.

        Args:
            filename: Path to safetensors file
            low_memory: If True, use streaming mode; if False, preload all tensors
            direct_gpu: If True, stream directly to GPU pinned/slab memory (requires low_memory=True)
            use_mmap: If True, map file to virtual memory instead of file reads
        """
        torch = _ensure_torch()
        safe_open = _ensure_safetensors()

        self.filename = filename
        self.low_memory = low_memory
        self.direct_gpu = direct_gpu
        self.use_mmap = use_mmap

        if self.direct_gpu and not self.low_memory:
            logging_utils.warning(
                "direct_gpu=True requires low_memory=True. Forcing low_memory=True."
            )
            self.low_memory = True

        if self.use_mmap:
            try:
                from .uel.model_mmap import ModelMMAP
                from .uel import control
                import os
                import ctypes

                if control.lib is None:
                    init_result = control.init()
                    if not init_result:
                        raise RuntimeError("control.init() returned False — comfy-aimdo library may not be available")

                self._mmap = ModelMMAP(self.filename)
                self._mmap_size = os.path.getsize(self.filename)
                self._mmap_view = memoryview(
                    (ctypes.c_uint8 * self._mmap_size).from_address(self._mmap.get())
                )
                logging_utils.verbose(f"Initialized MMAP mode for {self.filename}")
            except Exception as e:
                logging_utils.warning(
                    f"Failed to initialize MMAP: {e}. Falling back to standard IO."
                )
                logging_utils.debug(f"MMAP initialization exception details: {type(e).__name__}: {e}")
                self.use_mmap = False
                self._mmap = None
                self._mmap_view = None
        else:
            self._mmap = None
            self._mmap_view = None

        self._tensors: Dict[str, "torch.Tensor"] = {}
        self._gpu_buffer_indices: Dict[str, int] = {}
        self._gpu_pool = None

        self._all_keys = []
        self._file = None
        self._header = None
        self._header_size = None
        self._metadata: Dict[str, str] = {}

        if self.low_memory:
            # Streaming mode: read header only
            self._header, self._header_size = self._read_header()
            self._file = None  # Opened lazily to support multiprocessing DataLoader
            self._all_keys = [k for k in self._header.keys() if k != "__metadata__"]
            # Extract metadata from header (safetensors stores it under __metadata__ key)
            self._metadata = self._header.get("__metadata__", {})
            logging_utils.verbose(
                f"Initialized Low-memory mode: parsed header of size {self._header_size} bytes."
            )
            logging_utils.verbose(
                f"Found {len(self._all_keys)} tensors (streaming mode)"
            )
        else:
            # Standard mode: preload all tensors
            with safe_open(self.filename, framework="pt", device="cpu") as f:
                self._metadata = f.metadata() or {}
                self._all_keys = list(f.keys())
                logging_utils.normal(
                    f"Loading {len(self._all_keys)} tensors from source file..."
                )
                try:
                    from tqdm import tqdm

                    iterator = tqdm(
                        self._all_keys,
                        desc="Loading tensors",
                        disable=not logger.isEnabledFor(logging_utils.NORMAL_LEVEL),
                    )
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
        state["_file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def close(self):
        """Close file handle and release resources."""
        if self._file:
            self._file.close()
            self._file = None
        # Release MMAP — critical on Windows where mapped file cannot be deleted
        # while the mapping is alive.
        self._mmap = None
        self._mmap_view = None
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
            metadata = self._header[key]
            return st_shape_to_torch_shape(metadata["shape"], metadata["dtype"])
        else:
            return tuple(self._tensors[key].shape)

    def get_ndim(self, key: str) -> int:
        """Get tensor ndim without loading tensor data."""
        return len(self.get_shape(key))

    def get_dtype(self, key: str):
        """Get tensor torch dtype without loading tensor data."""
        if self.low_memory:
            if key not in self._header:
                raise KeyError(f"Tensor '{key}' not found in file")
            return st_to_torch_dtype(self._header[key]["dtype"])
        else:
            return self._tensors[key].dtype

    @logging_utils.log_debug
    def get_tensor(self, key: str) -> "torch.Tensor":
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
            if self.use_mmap:
                logging_utils.debug(f"Loading tensor '{key}' from MMAP")
                tensor_view = self._mmap_view[
                    self._header_size + 8 + offset_start : self._header_size
                    + 8
                    + offset_end
                ]
                import warnings

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="The given buffer is not writable"
                    )
                    dtype = st_to_torch_dtype(metadata["dtype"])
                    shape = st_shape_to_torch_shape(metadata["shape"], metadata["dtype"])
                    tensor = torch.frombuffer(tensor_view, dtype=dtype).view(shape)
                    storage = tensor.untyped_storage()
                    setattr(storage, "_uel_mmap_ref", self._mmap)
                    return tensor

            logging_utils.debug(
                f"Loading tensor '{key}' from offset {offset_start} to {offset_end} ({(offset_end - offset_start)} bytes)"
            )
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
        In low-memory mode, frees GPU buffer back to pool if direct_gpu.
        """
        if not self.low_memory and key in self._tensors:
            del self._tensors[key]
            gc.collect()

        if self.direct_gpu and key in self._gpu_buffer_indices:
            idx = self._gpu_buffer_indices.pop(key)
            if self._gpu_pool:
                self._gpu_pool.release(idx)

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
        shape = st_shape_to_torch_shape(metadata["shape"], dtype_str)
        dtype = st_to_torch_dtype(dtype_str)

        if tensor_bytes is None:
            byte_tensor = torch.empty(0, dtype=torch.uint8)
        else:
            byte_tensor = torch.frombuffer(tensor_bytes, dtype=torch.uint8)

        return byte_tensor.view(dtype).reshape(shape)

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

    def async_stream(
        self,
        keys: list,
        batch_size: int = 1,
        prefetch_batches: int = 2,
        pin_memory: bool = False,
    ):
        """Asynchronously stream tensors from disk.

        Args:
            keys: List of tensor keys to load
            batch_size: Number of tensors to yield in each batch
            prefetch_batches: Number of batches to pre-fetch in background
            pin_memory: If True, tensors will be pinned in CPU memory (sequentially in main thread)
            direct_gpu: Stream via pinned buffer directly to GPU

        Yields:
            List of (key, tensor) tuples
        """
        import threading
        import queue
        from concurrent.futures import ThreadPoolExecutor
        import os

        torch = _ensure_torch()
        thread_local = threading.local()

        # Initialize GPU slab and Pinned Buffer Pool if direct_gpu
        pinned_pool = None
        cuda_stream = None

        if self.direct_gpu and torch.cuda.is_available():
            try:
                from .gpu_buffer_pool import GpuBufferPool
                from .pinned_buffer_pool import PinnedBufferPool

                # Pre-calculate required slab size
                max_tensor_bytes = 0
                for k in keys:
                    meta = self._header[k]
                    start, end = meta["data_offsets"]
                    sz = end - start
                    max_tensor_bytes = max(max_tensor_bytes, sz)

                # Initialize pools (size of largest tensor)
                # We need a larger pool to allow the GPU to lag behind the CPU without stalling
                max_workers = min(16, max(4, batch_size))
                max_in_flight = max(max_workers, prefetch_batches * batch_size)

                # Double the buffers for a smooth pipeline
                num_buffers = (max_in_flight + max_workers) * 2 + 2

                # Assign pool to instance to survive the generator lifetime
                if not getattr(self, "_gpu_pool", None):
                    self._gpu_pool = GpuBufferPool(max_tensor_bytes, num_buffers)

                pinned_pool = PinnedBufferPool(max_tensor_bytes, num_buffers)
                cuda_stream = torch.cuda.Stream()

                logging_utils.normal(
                    f"Direct GPU pipeline initialized: {num_buffers} buffers, max {max_tensor_bytes / (1024**2):.1f}MB each (Total VRAM: {(num_buffers * max_tensor_bytes) / (1024**2):.1f}MB)"
                )

            except Exception as e:
                logging_utils.warning(
                    f"Failed to initialize direct GPU pipeline: {e}. Falling back."
                )
                self.direct_gpu = False
                pinned_pool = None
        elif self.direct_gpu:
            logging_utils.warning(
                "direct_gpu=True requested but CUDA is not available. Falling back to CPU."
            )
            self.direct_gpu = False

        def get_file_handle():
            if not hasattr(thread_local, "file"):
                thread_local.file = open(self.filename, "rb")
            return thread_local.file

        def _worker_load(key):
            buf_idx = None
            gpu_idx = None
            try:
                metadata = self._header[key]
                offset_start, offset_end = metadata["data_offsets"]
                sz = offset_end - offset_start

                if self.direct_gpu and sz > 0:
                    # Direct GPU Pipeline Path
                    buf_idx, pinned_buf = pinned_pool.acquire()

                    try:
                        # Schedule GPU transfer
                        gpu_idx, gpu_buf = self._gpu_pool.acquire()

                        try:
                            # Read into pinned memory directly (Zero-Copy CPU path)
                            import ctypes

                            view = pinned_buf[:sz]

                            if self.use_mmap:
                                tensor_view = self._mmap_view[
                                    self._header_size
                                    + 8
                                    + offset_start : self._header_size + 8 + offset_end
                                ]
                                import warnings

                                with warnings.catch_warnings():
                                    warnings.filterwarnings(
                                        "ignore",
                                        message="The given buffer is not writable",
                                    )
                                    mmap_tensor = torch.frombuffer(
                                        tensor_view, dtype=torch.uint8
                                    )
                                    view.copy_(mmap_tensor)
                            else:
                                # Create a ctypes c_uint8 array spanning the pinned buffer memory
                                # This allows f.readinto() to write bytes directly to the torch tensor memory
                                c_uint8_array = (ctypes.c_uint8 * sz).from_address(
                                    view.data_ptr()
                                )

                                f = get_file_handle()
                                f.seek(self._header_size + 8 + offset_start)
                                f.readinto(c_uint8_array)

                            gpu_view = gpu_buf[:sz]

                            with torch.cuda.stream(cuda_stream):
                                gpu_view.copy_(view, non_blocking=True)

                                # Create event to track when copy finishes
                                event = torch.cuda.Event()
                                event.record()

                            # Critical: wait for stream before allowing worker to finish
                            # If worker finishes, buffer might be overwritten by next worker
                            # if pool sizing is tight.
                            # In direct_gpu, the tensor is the gpu_view.
                            return key, gpu_view, metadata, buf_idx, gpu_idx, event

                        except Exception as e:
                            # If reading or copying fails, release GPU buffer
                            self._gpu_pool.release(gpu_idx)
                            raise e

                    except Exception as e:
                        # If acquiring GPU buffer fails, release pinned buffer
                        pinned_pool.release(buf_idx)
                        raise e
                else:
                    # Standard CPU Path
                    if offset_start != offset_end:
                        if self.use_mmap:
                            tensor_view = self._mmap_view[
                                self._header_size + 8 + offset_start : self._header_size
                                + 8
                                + offset_end
                            ]
                            tensor_bytes = bytearray(tensor_view)
                        else:
                            f = get_file_handle()
                            f.seek(self._header_size + 8 + offset_start)
                            tensor_bytes = bytearray(offset_end - offset_start)
                            f.readinto(tensor_bytes)
                    else:
                        tensor_bytes = None

                    tensor = self._deserialize_tensor(tensor_bytes, metadata)
                    return key, tensor, None, None, None, None
            except Exception as e:
                return key, None, e, None, None, None

        max_workers = min(16, max(4, batch_size))
        max_in_flight = max(max_workers, prefetch_batches * batch_size)

        # Queue for individual (key, tensor) pairs
        # Size it to hold enough for prefetch_batches PLUS max_workers to prevent stalling
        q = queue.Queue(maxsize=max_in_flight + max_workers)

        def _producer():
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
                    result = f.result()  # Blocks until this specific tensor is loaded
                    q.put(result)  # Blocks if the consumption queue is full

                    # Submit next task if available
                    try:
                        k = next(key_iter)
                        futures.append(executor.submit(_worker_load, k))
                    except StopIteration:
                        pass

            q.put(None)  # Sentinel

        producer_thread = threading.local()
        producer_thread = threading.Thread(target=_producer, daemon=True)
        producer_thread.start()

        batch = []
        pending_pinned = []  # Track (event, buf_idx) to release later

        while True:
            res = q.get()
            if res is None:
                if batch:
                    yield batch
                break

            k, t, err, buf_idx, gpu_idx, event = res
            if err is not None and not isinstance(err, dict):
                logging_utils.warning(
                    f"Async load failed for {k}, falling back to sync: {err}"
                )
                # Fallback synchronous load
                try:
                    t = self.get_tensor(k)
                except Exception as sync_err:
                    logging_utils.error(
                        f"Sync fallback also failed for {k}: {sync_err}"
                    )
                    raise sync_err

            if buf_idx is not None and event is not None:
                # Synchronize before reshaping — GPU copy must complete before
                # we hand the tensor to the caller or values will be garbage.
                event.synchronize()

                # Release the pinned staging buffer now that copy is complete
                pinned_pool.release(buf_idx)

                # Register GPU index for cleanup
                self._gpu_buffer_indices[k] = gpu_idx

                # Reshape GPU view to tensor
                meta = err  # we reused err for metadata in direct_gpu path
                dtype = st_to_torch_dtype(meta["dtype"])
                shape = meta["shape"]

                t = t.view(dtype).reshape(shape)

            # Pin memory sequentially in the main thread to avoid OS-level lock contention
            elif pin_memory and t.device.type == "cpu":
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
