"""
UnifiedGGUFLoader — GGUF file loader with the same four loading modes as
UnifiedSafetensorsLoader (preload, low-memory stream, async stream, direct GPU,
memory-mapped).

Tensors are returned as GGMLTensor instances that carry the raw quantized bytes
and the logical shape.  Float/integer types (F32, F16, BF16, I8 …) are returned
as plain torch.Tensor views of the same shape.

The uel subsystem (ModelMMAP, VBAR, HostBuffer) is used exactly as in the
safetensors loader — it operates on raw byte buffers and is format-agnostic.

Dequantization is lazy: callers (GGMLLayer / quant_ops in ComfyUI-GGUF) call
dequantize_tensor() at compute time.  This keeps the hot path identical to the
rattus128/dynamic-vram approach.
"""

import gc
import logging
import struct
import threading
import warnings
from typing import Dict, List, Optional, Tuple

from . import logging_utils
from .gguf_dequant import GGMLTensor, dequantize_tensor, is_quantized

logger = logging_utils.get_logger(__name__)

# GGUF magic bytes: 'G','G','U','F'
_GGUF_MAGIC = b"GGUF"

# ggml types that map directly to a torch dtype (no dequant needed)
_DIRECT_TORCH_GGML_TYPES = None  # built lazily after gguf import


def _ensure_gguf():
    try:
        import gguf
        return gguf
    except ImportError:
        raise ImportError(
            "The 'gguf' package is required for GGUF support but is not installed. "
            "Please install it: pip install gguf"
        )


def _ensure_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "The 'torch' package is required but is not installed. "
            "Please install it."
        )


def _direct_torch_types(gguf):
    """Return the set of GGMLQuantizationTypes that map to native torch dtypes."""
    return {
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
        gguf.GGMLQuantizationType.BF16,
        # Integer types
        gguf.GGMLQuantizationType.I8,
        gguf.GGMLQuantizationType.I16,
        gguf.GGMLQuantizationType.I32,
        gguf.GGMLQuantizationType.I64,
        gguf.GGMLQuantizationType.F64,
    }


# Map from GGMLQuantizationType to torch dtype for the directly-mappable types
def _ggml_to_torch_dtype(gguf, qtype, torch):
    mapping = {
        gguf.GGMLQuantizationType.F32:  torch.float32,
        gguf.GGMLQuantizationType.F16:  torch.float16,
        gguf.GGMLQuantizationType.BF16: torch.bfloat16,
        gguf.GGMLQuantizationType.F64:  torch.float64,
        gguf.GGMLQuantizationType.I8:   torch.int8,
        gguf.GGMLQuantizationType.I16:  torch.int16,
        gguf.GGMLQuantizationType.I32:  torch.int32,
        gguf.GGMLQuantizationType.I64:  torch.int64,
    }
    return mapping.get(qtype)


def _parse_gguf_metadata(reader, gguf_mod) -> dict:
    """Extract scalar metadata KV pairs from a GGUFReader into a plain dict.

    Arrays and nested types are included when they can be serialised cheaply;
    unreadable fields are silently skipped (same behaviour as get_gguf_metadata
    in rattus128/loader.py).
    """
    metadata = {}
    for field_name in reader.fields:
        try:
            field = reader.get_field(field_name)
            if len(field.types) == 1:
                t = field.types[0]
                if t == gguf_mod.GGUFValueType.STRING:
                    metadata[field_name] = str(field.parts[field.data[-1]], "utf-8")
                elif t == gguf_mod.GGUFValueType.INT8:
                    metadata[field_name] = int(field.parts[field.data[-1]][0])
                elif t == gguf_mod.GGUFValueType.INT16:
                    metadata[field_name] = int(field.parts[field.data[-1]][0])
                elif t == gguf_mod.GGUFValueType.INT32:
                    metadata[field_name] = int(field.parts[field.data[-1]][0])
                elif t == gguf_mod.GGUFValueType.INT64:
                    metadata[field_name] = int(field.parts[field.data[-1]][0])
                elif t == gguf_mod.GGUFValueType.UINT8:
                    metadata[field_name] = int(field.parts[field.data[-1]][0])
                elif t == gguf_mod.GGUFValueType.UINT16:
                    metadata[field_name] = int(field.parts[field.data[-1]][0])
                elif t == gguf_mod.GGUFValueType.UINT32:
                    metadata[field_name] = int(field.parts[field.data[-1]][0])
                elif t == gguf_mod.GGUFValueType.UINT64:
                    metadata[field_name] = int(field.parts[field.data[-1]][0])
                elif t == gguf_mod.GGUFValueType.FLOAT32:
                    metadata[field_name] = float(field.parts[field.data[-1]][0])
                elif t == gguf_mod.GGUFValueType.FLOAT64:
                    metadata[field_name] = float(field.parts[field.data[-1]][0])
                elif t == gguf_mod.GGUFValueType.BOOL:
                    metadata[field_name] = bool(field.parts[field.data[-1]][0])
            elif (
                len(field.types) == 2
                and field.types[0] == gguf_mod.GGUFValueType.ARRAY
            ):
                # Flat arrays of scalars
                inner = field.types[1]
                if inner == gguf_mod.GGUFValueType.STRING:
                    metadata[field_name] = tuple(
                        str(field.parts[idx], "utf-8") for idx in field.data
                    )
                elif inner in (
                    gguf_mod.GGUFValueType.INT8,  gguf_mod.GGUFValueType.INT16,
                    gguf_mod.GGUFValueType.INT32, gguf_mod.GGUFValueType.INT64,
                    gguf_mod.GGUFValueType.UINT8, gguf_mod.GGUFValueType.UINT16,
                    gguf_mod.GGUFValueType.UINT32,gguf_mod.GGUFValueType.UINT64,
                ):
                    metadata[field_name] = tuple(
                        int(field.parts[idx][0]) for idx in field.data
                    )
                elif inner in (
                    gguf_mod.GGUFValueType.FLOAT32, gguf_mod.GGUFValueType.FLOAT64
                ):
                    metadata[field_name] = tuple(
                        float(field.parts[idx][0]) for idx in field.data
                    )
        except Exception:
            continue
    return metadata


def _build_tensor_index(reader, gguf_mod) -> Dict[str, dict]:
    """Build a name→info dict from GGUFReader.tensors.

    Info dict keys:
        tensor_type  — GGMLQuantizationType
        shape        — torch.Size (logical, reversed from GGUF column-major)
        byte_offset  — absolute byte offset from start of file (tensor.data_offset)
        byte_size    — number of raw bytes for this tensor (tensor.data.nbytes)
        orig_shape   — comfy.gguf.orig_shape override if present (may be None)
    """
    import torch
    index = {}
    for tensor in reader.tensors:
        name = tensor.name

        # GGUF stores dims in column-major order; reverse for row-major (PyTorch)
        shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))

        # Check for ComfyUI orig_shape override (rattus128/city96 convention)
        orig_shape = None
        field_key = f"comfy.gguf.orig_shape.{name}"
        field = reader.get_field(field_key)
        if field is not None:
            try:
                if (
                    len(field.types) == 2
                    and field.types[0] == gguf_mod.GGUFValueType.ARRAY
                    and field.types[1] == gguf_mod.GGUFValueType.INT32
                ):
                    orig_shape = torch.Size(
                        tuple(int(field.parts[idx][0]) for idx in field.data)
                    )
            except Exception:
                pass

        logical_shape = orig_shape if orig_shape is not None else shape

        # tensor.data_offset is the absolute byte offset from start of file
        byte_offset = tensor.data_offset
        # byte_size derived from the numpy array underlying the tensor
        byte_size = tensor.data.nbytes

        index[name] = {
            "tensor_type":  tensor.tensor_type,
            "shape":        logical_shape,
            "byte_offset":  byte_offset,
            "byte_size":    byte_size,
        }
    return index


class UnifiedGGUFLoader:
    """Unified GGUF loader supporting the same four loading modes as
    UnifiedSafetensorsLoader.

    Tensors are returned as GGMLTensor (quantized) or plain torch.Tensor
    (float/int types).  Callers that use GGMLLayer / quant_ops will handle
    dequantization transparently at compute time.

    Usage (mirrors safetensors loader):

        with UnifiedGGUFLoader("model.gguf", low_memory=True) as loader:
            for key in loader.tensor_names:
                tensor = loader.get_tensor(key)
                # ... process ...
                loader.mark_processed(key)

        # Or load everything at once:
        state_dict, extra = loader.load_all()
        # extra = {"arch_str": "...", "metadata": {...}}
    """

    @logging_utils.log_debug
    def __init__(
        self,
        filename: str,
        low_memory: bool = False,
        direct_gpu: bool = False,
        use_mmap: bool = False,
    ):
        """Initialise the GGUF loader.

        Args:
            filename:   Path to .gguf file.
            low_memory: Stream tensors on-demand instead of preloading all.
            direct_gpu: Copy tensors directly to GPU slab via pinned buffers
                        (forces low_memory=True).
            use_mmap:   Map the whole file via uel ModelMMAP for zero-copy reads.
        """
        self._gguf = _ensure_gguf()
        self._torch = _ensure_torch()

        self.filename = filename
        self.low_memory = low_memory
        self.direct_gpu = direct_gpu
        self.use_mmap = use_mmap

        if self.direct_gpu and not self.low_memory:
            logging_utils.warning(
                "direct_gpu=True requires low_memory=True. Forcing low_memory=True."
            )
            self.low_memory = True

        # -------------------------------------------------------------------
        # MMAP initialisation (same pattern as safetensors loader)
        # -------------------------------------------------------------------
        if self.use_mmap:
            try:
                import ctypes
                import os
                from .uel import control
                from .uel.model_mmap import ModelMMAP

                if control.lib is None:
                    control.init()

                self._mmap = ModelMMAP(self.filename)
                self._mmap_size = os.path.getsize(self.filename)
                self._mmap_view = memoryview(
                    (ctypes.c_uint8 * self._mmap_size).from_address(
                        self._mmap.get()
                    )
                )
                logging_utils.verbose(f"GGUF: MMAP initialised for {self.filename}")
            except Exception as e:
                logging_utils.warning(
                    f"GGUF: Failed to init MMAP: {e}. Falling back to standard IO."
                )
                self.use_mmap = False
                self._mmap = None
                self._mmap_view = None
        else:
            self._mmap = None
            self._mmap_view = None

        # -------------------------------------------------------------------
        # Parse header via GGUFReader
        # GGUFReader mmaps the file internally via numpy; this is independent
        # of (and compatible with) our uel ModelMMAP above.
        # -------------------------------------------------------------------
        self._reader = self._gguf.GGUFReader(self.filename)

        # Tensor index: name → {tensor_type, shape, byte_offset, byte_size}
        self._tensor_index: Dict[str, dict] = _build_tensor_index(
            self._reader, self._gguf
        )

        # GGUF metadata KV store
        self._metadata: dict = _parse_gguf_metadata(self._reader, self._gguf)

        self._all_keys: List[str] = list(self._tensor_index.keys())

        # Preload cache (used when low_memory=False)
        self._tensors: Dict[str, object] = {}

        # GPU buffer tracking (mirrors safetensors loader)
        self._gpu_buffer_indices: Dict[str, int] = {}
        self._gpu_pool = None

        # Lazy file handle (streaming mode, multiprocessing-safe)
        self._file = None

        # -------------------------------------------------------------------
        # Preload all tensors if not in low-memory / streaming mode
        # -------------------------------------------------------------------
        if not self.low_memory:
            logging_utils.verbose(
                f"GGUF: Preloading {len(self._all_keys)} tensors from {self.filename}"
            )
            for key in self._all_keys:
                self._tensors[key] = self._read_tensor(key)
            logging_utils.verbose("GGUF: Preload complete.")
        else:
            logging_utils.verbose(
                f"GGUF: Low-memory mode — {len(self._all_keys)} tensors "
                f"(streaming) from {self.filename}"
            )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
        self._tensors.clear()
        self._gpu_pool = None
        gc.collect()

    # Pickling support (DataLoader multiprocessing)
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        # GGUFReader holds file handles/numpy mmaps — drop it; will re-open lazily
        state["_reader"] = None
        state["_mmap"] = None
        state["_mmap_view"] = None
        # Module objects cannot be pickled; re-import on restore
        state["_gguf"] = None
        state["_torch"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore module references
        self._gguf = _ensure_gguf()
        self._torch = _ensure_torch()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def tensor_names(self) -> List[str]:
        """All tensor names in the file."""
        return self._all_keys

    def keys(self) -> List[str]:
        """Alias for tensor_names — matches safetensors loader interface."""
        return self._all_keys

    @property
    def metadata(self) -> dict:
        """All scalar GGUF metadata KV pairs as a Python dict."""
        return self._metadata

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_file_handle(self):
        if self._file is None:
            self._file = open(self.filename, "rb")
        return self._file

    def _raw_bytes_from_file(self, byte_offset: int, byte_size: int) -> bytes:
        f = self._get_file_handle()
        f.seek(byte_offset)
        buf = bytearray(byte_size)
        f.readinto(buf)
        return buf

    def _raw_bytes_from_mmap(self, byte_offset: int, byte_size: int):
        """Return a memoryview slice (zero-copy) from the mmap."""
        return self._mmap_view[byte_offset: byte_offset + byte_size]

    def _bytes_to_tensor(self, raw, info: dict):
        """Convert raw bytes (bytearray / memoryview / bytes) to tensor.

        Float/int types → plain torch.Tensor (reshaped to logical shape).
        Quantized types → GGMLTensor (raw uint8 storage, logical shape stored).
        """
        torch = self._torch
        gguf = self._gguf
        qtype = info["tensor_type"]
        shape = info["shape"]
        direct_types = _direct_torch_types(gguf)

        if qtype in direct_types:
            torch_dtype = _ggml_to_torch_dtype(gguf, qtype, torch)
            t = torch.frombuffer(
                bytearray(raw) if isinstance(raw, memoryview) else raw,
                dtype=torch.uint8,
            ).view(torch_dtype)
            # BF16 stored as raw bytes needs special handling
            if qtype == gguf.GGMLQuantizationType.BF16:
                # frombuffer with uint8 + view int16 << 16 done in dequant;
                # for direct use just view as bfloat16
                t = torch.frombuffer(
                    bytearray(raw) if isinstance(raw, memoryview) else raw,
                    dtype=torch.bfloat16,
                )
            return t.reshape(shape)
        else:
            # Quantized — keep as uint8 blob wrapped in GGMLTensor
            raw_bytes = bytearray(raw) if isinstance(raw, memoryview) else raw
            t = torch.frombuffer(raw_bytes, dtype=torch.uint8)
            tensor = GGMLTensor(t, tensor_type=qtype, tensor_shape=shape)

            # 1-D BF16 tensors should be dequantised immediately
            # (norm layers etc. — same workaround as rattus128)
            if len(shape) <= 1 and qtype == gguf.GGMLQuantizationType.BF16:
                tensor = dequantize_tensor(tensor, dtype=torch.float32)

            return tensor

    def _read_tensor(self, key: str):
        """Read a single tensor from disk (or mmap) and deserialise it."""
        info = self._tensor_index[key]
        byte_offset = info["byte_offset"]
        byte_size = info["byte_size"]

        if byte_size == 0:
            # Zero-element tensor
            return self._torch.empty(0, dtype=self._torch.uint8)

        if self.use_mmap and self._mmap_view is not None:
            raw = self._raw_bytes_from_mmap(byte_offset, byte_size)
        else:
            raw = self._raw_bytes_from_file(byte_offset, byte_size)

        return self._bytes_to_tensor(raw, info)

    # ------------------------------------------------------------------
    # Public API (matches UnifiedSafetensorsLoader)
    # ------------------------------------------------------------------

    def get_tensor(self, key: str):
        """Return the tensor for `key`.

        In preload mode: returns from cache.
        In streaming mode: reads from disk on every call.
        """
        if not self.low_memory:
            if key not in self._tensors:
                raise KeyError(f"Tensor '{key}' not found in GGUF file.")
            return self._tensors[key]

        # Streaming: check direct_gpu cache first
        if self.direct_gpu and key in self._gpu_buffer_indices:
            raise RuntimeError(
                f"Tensor '{key}' is still held in a GPU buffer slot. "
                "Call mark_processed() before re-fetching."
            )

        return self._read_tensor(key)

    def mark_processed(self, key: str):
        """Release resources held for `key` (mirrors safetensors loader)."""
        if not self.low_memory:
            if key in self._tensors:
                del self._tensors[key]
        if self.direct_gpu and key in self._gpu_buffer_indices:
            idx = self._gpu_buffer_indices.pop(key)
            if self._gpu_pool:
                self._gpu_pool.release(idx)

    def load_all(self) -> Tuple[dict, dict]:
        """Load all tensors as a state dict plus an extras dict.

        Returns:
            (state_dict, extra) where
            state_dict — {name: tensor/GGMLTensor}
            extra      — {"arch_str": str|None, "metadata": dict}
        """
        if not self.low_memory:
            sd = dict(self._tensors)
        else:
            sd = {}
            for batch in self.async_stream(
                keys=self._all_keys,
                batch_size=16,
                prefetch_batches=2,
                pin_memory=False,
            ):
                for key, tensor in batch:
                    sd[key] = tensor

        # Mark largest quantised tensor for VRAM estimation
        # (mirrors city96/rattus128 loader.py)
        qsd = {k: v for k, v in sd.items() if is_quantized(v)}
        if qsd:
            max_key = max(qsd, key=lambda k: qsd[k].numel())
            try:
                qsd[max_key].is_largest_weight = True
            except Exception:
                pass

        arch_str = self._metadata.get("general.architecture", None)
        extra = {
            "arch_str": arch_str,
            "metadata": self._metadata,
        }
        return sd, extra

    # ------------------------------------------------------------------
    # Async streaming (mirrors async_stream in safetensors loader exactly,
    # except offset arithmetic uses byte_offset/byte_size from tensor index)
    # ------------------------------------------------------------------

    def async_stream(
        self,
        keys: list,
        batch_size: int = 1,
        prefetch_batches: int = 2,
        pin_memory: bool = False,
    ):
        """Asynchronously stream tensors from disk.

        Args:
            keys:             Tensor names to stream.
            batch_size:       Tensors per yielded batch.
            prefetch_batches: Number of batches to prefetch.
            pin_memory:       Pin CPU tensors (sequentially, main thread).

        Yields:
            List of (key, tensor) tuples.
        """
        import ctypes
        import queue
        from concurrent.futures import ThreadPoolExecutor

        torch = self._torch
        thread_local = threading.local()

        pinned_pool = None
        cuda_stream = None

        # ---- direct_gpu pool init ----------------------------------------
        if self.direct_gpu and torch.cuda.is_available():
            try:
                from .gpu_buffer_pool import GpuBufferPool
                from .pinned_buffer_pool import PinnedBufferPool

                max_tensor_bytes = max(
                    (self._tensor_index[k]["byte_size"] for k in keys),
                    default=0,
                )
                max_workers = min(16, max(4, batch_size))
                max_in_flight = max(max_workers, prefetch_batches * batch_size)
                num_buffers = (max_in_flight + max_workers) * 2 + 2

                if not getattr(self, "_gpu_pool", None):
                    self._gpu_pool = GpuBufferPool(max_tensor_bytes, num_buffers)

                pinned_pool = PinnedBufferPool(max_tensor_bytes, num_buffers)
                cuda_stream = torch.cuda.Stream()

                logging_utils.normal(
                    f"GGUF direct GPU pipeline: {num_buffers} buffers, "
                    f"max {max_tensor_bytes / (1024**2):.1f} MB each "
                    f"(total VRAM: {num_buffers * max_tensor_bytes / (1024**2):.1f} MB)"
                )
            except Exception as e:
                logging_utils.warning(
                    f"GGUF: Failed to init direct GPU pipeline: {e}. Falling back."
                )
                self.direct_gpu = False
                pinned_pool = None
        elif self.direct_gpu:
            logging_utils.warning(
                "GGUF: direct_gpu=True but CUDA not available. Falling back to CPU."
            )
            self.direct_gpu = False

        # ---- per-worker file handle (avoids seek contention) -------------
        def get_file_handle():
            if not hasattr(thread_local, "file"):
                thread_local.file = open(self.filename, "rb")
            return thread_local.file

        # ---- worker ------------------------------------------------------
        def _worker_load(key):
            try:
                info = self._tensor_index[key]
                byte_offset = info["byte_offset"]
                byte_size = info["byte_size"]

                if self.direct_gpu and byte_size > 0:
                    # ---- direct GPU path (mirrors safetensors loader) -----
                    buf_idx, pinned_buf = pinned_pool.acquire()
                    try:
                        gpu_idx, gpu_buf = self._gpu_pool.acquire()
                        try:
                            view = pinned_buf[:byte_size]

                            if self.use_mmap and self._mmap_view is not None:
                                tensor_view = self._mmap_view[
                                    byte_offset: byte_offset + byte_size
                                ]
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
                                c_arr = (ctypes.c_uint8 * byte_size).from_address(
                                    view.data_ptr()
                                )
                                f = get_file_handle()
                                f.seek(byte_offset)
                                f.readinto(c_arr)

                            gpu_view = gpu_buf[:byte_size]
                            with torch.cuda.stream(cuda_stream):
                                gpu_view.copy_(view, non_blocking=True)
                                event = torch.cuda.Event()
                                event.record()

                            # reuse `info` slot (same trick as safetensors loader
                            # which reuses `err` for metadata in direct_gpu path)
                            return key, gpu_view, info, buf_idx, gpu_idx, event

                        except Exception as e:
                            self._gpu_pool.release(gpu_idx)
                            raise e
                    except Exception as e:
                        pinned_pool.release(buf_idx)
                        raise e

                else:
                    # ---- CPU / mmap path ---------------------------------
                    if byte_size > 0:
                        if self.use_mmap and self._mmap_view is not None:
                            raw = bytearray(
                                self._mmap_view[byte_offset: byte_offset + byte_size]
                            )
                        else:
                            f = get_file_handle()
                            f.seek(byte_offset)
                            raw = bytearray(byte_size)
                            f.readinto(raw)
                    else:
                        raw = None

                    tensor = (
                        self._bytes_to_tensor(raw, info)
                        if raw is not None
                        else torch.empty(0, dtype=torch.uint8)
                    )
                    return key, tensor, None, None, None, None

            except Exception as e:
                return key, None, e, None, None, None

        # ---- queue + producer thread (identical structure to safetensors) -
        max_workers = min(16, max(4, batch_size))
        max_in_flight = max(max_workers, prefetch_batches * batch_size)
        q = queue.Queue(maxsize=max_in_flight + max_workers)

        def _producer():
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                key_iter = iter(keys)

                for _ in range(max_in_flight):
                    try:
                        k = next(key_iter)
                        futures.append(executor.submit(_worker_load, k))
                    except StopIteration:
                        break

                while futures:
                    fut = futures.pop(0)
                    result = fut.result()
                    q.put(result)
                    try:
                        k = next(key_iter)
                        futures.append(executor.submit(_worker_load, k))
                    except StopIteration:
                        pass

            q.put(None)  # sentinel

        producer_thread = threading.Thread(target=_producer, daemon=True)
        producer_thread.start()

        batch = []

        while True:
            res = q.get()
            if res is None:
                if batch:
                    yield batch
                break

            k, t, err_or_info, buf_idx, gpu_idx, event = res

            if err_or_info is not None and not isinstance(err_or_info, dict):
                # err_or_info holds an Exception
                logging_utils.warning(
                    f"GGUF async load failed for '{k}', falling back to sync: "
                    f"{err_or_info}"
                )
                try:
                    t = self.get_tensor(k)
                    err_or_info = None
                except Exception as sync_err:
                    logging_utils.error(
                        f"GGUF sync fallback also failed for '{k}': {sync_err}"
                    )
                    raise sync_err

            if buf_idx is not None and event is not None:
                # direct_gpu path: synchronise then hand off
                event.synchronize()
                pinned_pool.release(buf_idx)
                self._gpu_buffer_indices[k] = gpu_idx

                # Deserialise from the GPU uint8 view
                info = err_or_info
                t = self._gpu_bytes_to_tensor(t, info)

            elif pin_memory and t is not None and t.device.type == "cpu":
                try:
                    t = t.pin_memory()
                except Exception as e:
                    logging_utils.warning(f"GGUF: pin_memory failed for '{k}': {e}")

            batch.append((k, t))
            if len(batch) == batch_size:
                yield batch
                batch = []

    def _gpu_bytes_to_tensor(self, gpu_view, info: dict):
        """Convert a uint8 GPU slab view to a typed tensor or GGMLTensor.

        For quantised types the uint8 blob lives on GPU — GGMLTensor wraps it
        and GGMLLayer.get_weight() will dequantise on the same device.
        """
        torch = self._torch
        gguf = self._gguf
        qtype = info["tensor_type"]
        shape = info["shape"]
        byte_size = info["byte_size"]
        direct_types = _direct_torch_types(gguf)

        raw_view = gpu_view[:byte_size]

        if qtype in direct_types:
            torch_dtype = _ggml_to_torch_dtype(gguf, qtype, torch)
            if qtype == gguf.GGMLQuantizationType.BF16:
                return raw_view.view(torch.bfloat16).reshape(shape)
            return raw_view.view(torch_dtype).reshape(shape)
        else:
            return GGMLTensor(raw_view, tensor_type=qtype, tensor_shape=shape)
