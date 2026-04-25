"""
Unified data loader.

Fast thread-based replacement for torch.utils.data.DataLoader.

Design mirrors the async_stream pattern in UnifiedSafetensorsLoader:
- One ThreadPoolExecutor worker call per dataset item (not per batch).
- Pipeline pre-fills max_in_flight individual item futures, then slides
  one-in-one-out as results are consumed — identical to async_stream.
- Queue holds individual (idx, item) results; consumer assembles into
  batches and collates. This gives true per-item I/O parallelism.
- RAM bounded by max_in_flight + max_workers items in flight at once.
- Direct GPU pipeline: collated batch flattened into pinned buffer,
  async copied to GPU slab via CUDA stream. Pool sized to match pipeline.
"""

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Optional

from . import logging_utils
from .gpu_buffer_pool import GpuBufferPool
from .pinned_buffer_pool import PinnedBufferPool

logger = logging_utils.get_logger(__name__)

def _ensure_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "The 'torch' package is required but not installed. Please install it."
        )


def _default_collate(batch):
    """Fallback minimal collate function that handles basic torch types."""
    torch = _ensure_torch()
    if len(batch) == 0:
        return []

    elem = batch[0]
    elem_type = type(elem)
    
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, we should stack into a shared memory tensor.
            # But here we are in a thread, so torch.stack is fine.
            pass
        return torch.stack(batch, 0)
    elif isinstance(elem, dict):
        return {key: _default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, (tuple, list)):
        transposed = zip(*batch)
        return elem_type([_default_collate(samples) for samples in transposed])
    elif isinstance(elem, (int, float)):
        return torch.tensor(batch)
    
    # If it's something else, return as list
    return batch


def _bytes_of_collated(obj) -> int:
    """Return total byte size of all tensors in a collated batch structure."""
    torch = _ensure_torch()
    if isinstance(obj, torch.Tensor):
        return obj.element_size() * obj.numel()
    elif isinstance(obj, (tuple, list)):
        return sum(_bytes_of_collated(x) for x in obj)
    elif isinstance(obj, dict):
        return sum(_bytes_of_collated(v) for v in obj.values())
    return 0


def _flatten_tensors_to_buf(obj, buf, offset: int) -> int:
    """Copy all tensors from obj into uint8 buffer starting at offset.
    Returns new offset after writing."""
    torch = _ensure_torch()
    if isinstance(obj, torch.Tensor):
        sz = obj.element_size() * obj.numel()
        buf[offset:offset + sz].copy_(obj.contiguous().view(torch.uint8).flatten())
        return offset + sz
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            offset = _flatten_tensors_to_buf(item, buf, offset)
        return offset
    elif isinstance(obj, dict):
        for v in obj.values():
            offset = _flatten_tensors_to_buf(v, buf, offset)
        return offset
    return offset


def _restore_tensors_from_buf(obj, buf, offset: int):
    """Rebuild batch structure with tensors sourced from gpu buf.
    Returns (restored_obj, new_offset)."""
    torch = _ensure_torch()
    if isinstance(obj, torch.Tensor):
        sz = obj.element_size() * obj.numel()
        restored = buf[offset:offset + sz].view(obj.dtype).reshape(obj.shape)
        return restored, offset + sz
    elif isinstance(obj, tuple):
        parts = []
        for item in obj:
            r, offset = _restore_tensors_from_buf(item, buf, offset)
            parts.append(r)
        return tuple(parts), offset
    elif isinstance(obj, list):
        parts = []
        for item in obj:
            r, offset = _restore_tensors_from_buf(item, buf, offset)
            parts.append(r)
        return parts, offset
    elif isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            r, offset = _restore_tensors_from_buf(v, buf, offset)
            out[k] = r
        return out, offset
    return obj, offset


def _pin_collated(obj):
    """Recursively pin all CPU tensors in a collated structure."""
    torch = _ensure_torch()
    if isinstance(obj, torch.Tensor):
        return obj.pin_memory() if obj.device.type == "cpu" else obj
    elif isinstance(obj, tuple):
        return tuple(_pin_collated(x) for x in obj)
    elif isinstance(obj, list):
        return [_pin_collated(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: _pin_collated(v) for k, v in obj.items()}
    return obj


class UnifiedDataLoader:
    """Fast thread-based data loader with per-item parallel loading.

    Mirrors the async_stream pattern from UnifiedSafetensorsLoader:
    one worker call per dataset item, pipeline pre-fill with sliding window,
    consumer assembles batches from individual item results.

    Usage::

        loader = UnifiedDataLoader(dataset, batch_size=32, direct_gpu=True)
        for batch in loader:
            pass
    """

    def __init__(
        self,
        dataset=None,
        *,
        load_fn=None,
        length: Optional[int] = None,
        keys: Optional[list] = None,
        collate_fn=None,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        prefetch_batches: int = 2,
        pin_memory: bool = False,
        direct_gpu: bool = False,
        drop_last: bool = False,
        device: str = "cuda",
    ):
        if dataset is not None:
            # Backwards compatibility for torch.utils.data.Dataset
            # Check if it has a fast path async_stream (like UnifiedSafetensorsLoader)
            if hasattr(dataset, "async_stream") and callable(dataset.async_stream):
                self.load_fn = dataset.get_tensor if hasattr(dataset, "get_tensor") else None
                self.keys = list(dataset.keys()) if hasattr(dataset, "keys") else None
                self.length = len(self.keys) if self.keys is not None else None
            else:
                self.load_fn = lambda idx: dataset[idx]
                self.length = len(dataset)
                self.keys = None
            self.dataset = dataset
        else:
            if load_fn is None:
                raise ValueError("Must provide either 'dataset' or 'load_fn'")
            self.load_fn = load_fn
            self.length = length
            self.keys = keys
            self.dataset = None

        if self.length is None and self.keys is None and self.dataset is None:
             raise ValueError("Must provide either 'length' or 'keys' if not using 'dataset'")

        self.collate_fn = collate_fn or _default_collate
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = max(1, num_workers)
        self.prefetch_batches = max(1, prefetch_batches)
        self.pin_memory = pin_memory
        self.direct_gpu = direct_gpu
        self.drop_last = drop_last
        self.device = device

        torch = _ensure_torch()

        if self.direct_gpu and not torch.cuda.is_available():
            logging_utils.warning(
                "direct_gpu=True requested but CUDA is not available. Falling back to CPU."
            )
            self.direct_gpu = False

    def __len__(self) -> int:
        dataset_len = self.length if self.length is not None else len(self.keys)
        n_batches = dataset_len // self.batch_size
        if dataset_len % self.batch_size != 0 and not self.drop_last:
            n_batches += 1
        return n_batches

    def __iter__(self) -> Iterator:
        torch = _ensure_torch()
        import random

        if self.dataset is not None and hasattr(self.dataset, "async_stream") and callable(self.dataset.async_stream):
            # Fast path for UnifiedSafetensorsLoader passed as dataset
            # (or any object with async_stream). We directly yield from it.
            # We assume it handles its own shuffling and dropping last if needed
            # for its own items, but right now async_stream takes keys.
            
            keys = list(self.dataset.keys())
            if self.shuffle:
                 random.shuffle(keys)
                 
            if self.drop_last:
                 dataset_len = len(keys)
                 trim = dataset_len - (dataset_len % self.batch_size)
                 keys = keys[:trim]
                 
            if not keys:
                 return

            # Yield batches directly from async_stream
            for batch_items in self.dataset.async_stream(
                keys=keys,
                batch_size=self.batch_size,
                prefetch_batches=self.prefetch_batches,
                pin_memory=self.pin_memory,
            ):
                 # async_stream returns list of (key, tensor)
                 # We want to yield collated batch. If direct_gpu is on, 
                 # async_stream might return tensor directly, or it might just be the list.
                 # Let's unwrap (key, tensor) and collate.
                 
                 # actually async stream returns `batch` which is `list[(key, tensor)]`
                 tensors = [t for _, t in batch_items]
                 yield self.collate_fn(tensors)

            return
            
        dataset_len = self.length if self.length is not None else len(self.keys)
        if self.keys is not None:
             indices_or_keys = list(self.keys)
        else:
             indices_or_keys = list(range(dataset_len))
             
        if self.shuffle:
            random.shuffle(indices_or_keys)

        # Drop last incomplete batch if requested
        if self.drop_last:
            trim = dataset_len - (dataset_len % self.batch_size)
            indices_or_keys = indices_or_keys[:trim]

        if not indices_or_keys:
            return

        # Single-threaded fast path
        if self.num_workers <= 1 and not self.direct_gpu:
            for i in range(0, len(indices_or_keys), self.batch_size):
                batch_indices = indices_or_keys[i:i + self.batch_size]
                items = [self.load_fn(idx) for idx in batch_indices]
                yield self.collate_fn(items)
            return

        yield from self._threaded_iter(torch, indices_or_keys)

    def _threaded_iter(self, torch, indices_or_keys) -> Iterator:
        # Mirror async_stream sizing exactly:
        # max_workers scales with batch_size, max_in_flight covers prefetch depth
        max_workers = min(16, max(4, self.num_workers))
        max_in_flight = max(max_workers, self.prefetch_batches * self.batch_size)

        # Queue holds individual (position, item) results
        # Sized to hold max_in_flight + max_workers items without stalling
        q = queue.Queue(maxsize=max_in_flight + max_workers)

        # Direct GPU pool setup
        pinned_pool: Optional[PinnedBufferPool] = None
        gpu_pool: Optional[GpuBufferPool] = None
        cuda_stream = None
        direct_gpu = self.direct_gpu

        if direct_gpu:
            try:
                sample = self.collate_fn([self.load_fn(indices_or_keys[0])])
                batch_bytes = _bytes_of_collated(sample) * self.batch_size
                if batch_bytes == 0:
                    raise ValueError("Batch byte size is 0")
                # Pool sizing mirrors async_stream: (max_in_flight + max_workers) * 2 + 2
                num_buffers = (max_in_flight // self.batch_size + max_workers) * 2 + 2
                pinned_pool = PinnedBufferPool(batch_bytes, num_buffers)
                gpu_pool = GpuBufferPool(batch_bytes, num_buffers, device=self.device)
                cuda_stream = torch.cuda.Stream(device=self.device)
                logging_utils.normal(
                    f"Direct GPU pool: {num_buffers} buffers x "
                    f"{batch_bytes / (1024**2):.1f} MB "
                    f"(VRAM: {num_buffers * batch_bytes / (1024**2):.1f} MB)"
                )
            except Exception as e:
                logging_utils.warning(
                    f"Failed to allocate GPU pools: {e}. Falling back to CPU."
                )
                direct_gpu = False
                pinned_pool = None
                gpu_pool = None
                cuda_stream = None

        def _worker_load(pos, idx):
            """Load one item. Mirrors _worker_load(key) in async_stream."""
            try:
                item = self.load_fn(idx)
                return pos, item, None
            except Exception as e:
                return pos, None, e

        def _producer():
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                idx_iter = iter(enumerate(indices_or_keys))

                # Pre-fill pipeline — identical to async_stream fill loop
                for _ in range(max_in_flight):
                    try:
                        pos, idx = next(idx_iter)
                        futures.append(executor.submit(_worker_load, pos, idx))
                    except StopIteration:
                        break

                while futures:
                    # Take oldest future first — preserves order
                    result = futures.pop(0).result()
                    q.put(result)  # blocks if consumer is slow — backpressure

                    # Slide one more in
                    try:
                        pos, idx = next(idx_iter)
                        futures.append(executor.submit(_worker_load, pos, idx))
                    except StopIteration:
                        pass

            q.put(None)  # sentinel

        producer_thread = threading.Thread(target=_producer, daemon=True)
        producer_thread.start()

        # Consumer: collect batch_size items, collate, optionally push to GPU
        batch_items = []

        try:
            while True:
                res = q.get()
                if res is None:
                    # Yield final partial batch if not drop_last
                    if batch_items and not self.drop_last:
                        collated = self.collate_fn(batch_items)
                        if self.pin_memory:
                            collated = _pin_collated(collated)
                        yield collated
                    break

                pos, item, err = res

                if err is not None:
                    logging_utils.warning(f"Item load failed at position {pos}: {err}")
                    raise err

                batch_items.append(item)

                if len(batch_items) == self.batch_size:
                    collated = self.collate_fn(batch_items)
                    batch_items = []

                    if direct_gpu and pinned_pool is not None:
                        buf_idx, pinned_buf = pinned_pool.acquire()
                        try:
                            gpu_idx, gpu_buf = gpu_pool.acquire()
                            try:
                                total_bytes = _flatten_tensors_to_buf(collated, pinned_buf, 0)
                                with torch.cuda.stream(cuda_stream):
                                    gpu_buf[:total_bytes].copy_(
                                        pinned_buf[:total_bytes], non_blocking=True
                                    )
                                    event = torch.cuda.Event()
                                    event.record(cuda_stream)
                                event.synchronize()
                                pinned_pool.release(buf_idx)
                                restored, _ = _restore_tensors_from_buf(collated, gpu_buf, 0)
                                yield restored
                                gpu_pool.release(gpu_idx)
                            except Exception:
                                gpu_pool.release(gpu_idx)
                                raise
                        except Exception:
                            pinned_pool.release(buf_idx)
                            raise
                    elif self.pin_memory:
                        yield _pin_collated(collated)
                    else:
                        yield collated

        finally:
            # Drain so producer can unblock and exit
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
