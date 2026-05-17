"""
Unified data loader.

Fast thread-based replacement for torch.utils.data.DataLoader.
Supports zero-copy direct-to-GPU streaming via uel and pinned memory.
"""

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

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


class UnifiedDataLoader:
    """Fast data loader using threads and custom buffer pools.

    Usage:
        loader = UnifiedDataLoader(dataset, batch_size=32, direct_gpu=True)
        for batch in loader:
            # batch is on GPU
            pass
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        prefetch_batches: int = 2,
        pin_memory: bool = False,
        direct_gpu: bool = False,
        drop_last: bool = False,
        device: str = "cuda",
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = max(1, num_workers)
        self.prefetch_batches = prefetch_batches
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

    def __iter__(self) -> Iterator:
        torch = _ensure_torch()
        import random

        dataset_len = len(self.dataset)
        indices = list(range(dataset_len))
        if self.shuffle:
            random.shuffle(indices)

        # Batch the indices
        batches = []
        for i in range(0, dataset_len, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            if len(batch_indices) < self.batch_size and self.drop_last:
                continue
            batches.append(batch_indices)

        if not batches:
            return iter([])

        if self.num_workers <= 1 and not self.direct_gpu:
            # Fast path for no background workers and no GPU pipeline
            for batch_indices in batches:
                items = [self.dataset[idx] for idx in batch_indices]
                yield torch.utils.data.default_collate(items)
            return

        # Setup pools for direct GPU
        pinned_pool = None
        gpu_pool = None
        cuda_stream = None

        if self.direct_gpu:
            # Auto-detect tensor sizes by pulling one item
            sample_item = self.dataset[indices[0]]
            sample_batch = torch.utils.data.default_collate([sample_item])

            if isinstance(sample_batch, torch.Tensor):
                max_tensor_bytes = sample_batch.element_size() * sample_batch.numel()
            elif isinstance(sample_batch, (tuple, list)):
                # Handle tuple of tensors (e.g. image, label)
                max_tensor_bytes = sum(
                    t.element_size() * t.numel()
                    if isinstance(t, torch.Tensor)
                    else 0
                    for t in sample_batch
                )
            elif isinstance(sample_batch, dict):
                max_tensor_bytes = sum(
                    t.element_size() * t.numel()
                    if isinstance(t, torch.Tensor)
                    else 0
                    for t in sample_batch.values()
                )
            else:
                logging_utils.warning(
                    "Unsupported dataset item type for direct_gpu. Falling back."
                )
                self.direct_gpu = False
                max_tensor_bytes = 0

            if self.direct_gpu:
                # Need batch_size * max_tensor_bytes per buffer
                max_tensor_bytes = max_tensor_bytes * self.batch_size
                max_in_flight = max(self.num_workers, self.prefetch_batches)
                num_buffers = (max_in_flight + self.num_workers) * 2 + 2

                gpu_pool = GpuBufferPool(max_tensor_bytes, num_buffers, device=self.device)
                pinned_pool = PinnedBufferPool(max_tensor_bytes, num_buffers)
                cuda_stream = torch.cuda.Stream(device=self.device)

        max_in_flight = max(self.num_workers, self.prefetch_batches)
        q = queue.Queue(maxsize=max_in_flight + self.num_workers)

        def _worker_load(batch_indices):
            try:
                items = [self.dataset[idx] for idx in batch_indices]
                collated = torch.utils.data.default_collate(items)

                if self.direct_gpu:
                    buf_idx, pinned_buf = pinned_pool.acquire()
                    try:
                        gpu_idx, gpu_buf = gpu_pool.acquire()
                        try:
                            import ctypes

                            # Flatten collated into pinned buffer
                            def _flatten_to_pinned(batch_data, offset=0):
                                if isinstance(batch_data, torch.Tensor):
                                    sz = batch_data.element_size() * batch_data.numel()
                                    view = pinned_buf[offset : offset + sz]
                                    byte_data = batch_data.contiguous().view(torch.uint8).flatten()
                                    view.copy_(byte_data)
                                    return offset + sz
                                elif isinstance(batch_data, (tuple, list)):
                                    for t in batch_data:
                                        offset = _flatten_to_pinned(t, offset)
                                    return offset
                                elif isinstance(batch_data, dict):
                                    for t in batch_data.values():
                                        offset = _flatten_to_pinned(t, offset)
                                    return offset
                                return offset

                            total_bytes = _flatten_to_pinned(collated)

                            pinned_view = pinned_buf[:total_bytes]
                            gpu_view = gpu_buf[:total_bytes]

                            with torch.cuda.stream(cuda_stream):
                                gpu_view.copy_(pinned_view, non_blocking=True)
                                event = torch.cuda.Event()
                                event.record(cuda_stream)

                            return collated, buf_idx, gpu_idx, event, None

                        except Exception as e:
                            gpu_pool.release(gpu_idx)
                            raise e
                    except Exception as e:
                        pinned_pool.release(buf_idx)
                        raise e
                else:
                    return collated, None, None, None, None

            except Exception as e:
                return None, None, None, None, e

        def _producer():
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                batch_iter = iter(batches)

                for _ in range(max_in_flight):
                    try:
                        b = next(batch_iter)
                        futures.append(executor.submit(_worker_load, b))
                    except StopIteration:
                        break

                while futures:
                    f = futures.pop(0)
                    result = f.result()
                    q.put(result)

                    try:
                        b = next(batch_iter)
                        futures.append(executor.submit(_worker_load, b))
                    except StopIteration:
                        pass

            q.put(None)

        producer_thread = threading.Thread(target=_producer, daemon=True)
        producer_thread.start()

        # Generator
        try:
            while True:
                res = q.get()
                if res is None:
                    break

                collated, buf_idx, gpu_idx, event, err = res

                if err is not None:
                    logging_utils.warning(f"Worker load failed: {err}")
                    raise err

                if buf_idx is not None and event is not None:
                    event.synchronize()
                    pinned_pool.release(buf_idx)

                    # Unflatten from gpu_buf back to collated structure
                    def _unflatten_from_gpu(batch_data, gpu_buffer, offset=0):
                        if isinstance(batch_data, torch.Tensor):
                            sz = batch_data.element_size() * batch_data.numel()
                            gpu_view = gpu_buffer[offset : offset + sz]
                            restored = gpu_view.view(batch_data.dtype).reshape(batch_data.shape)
                            return restored, offset + sz
                        elif isinstance(batch_data, tuple):
                            restored_list = []
                            for t in batch_data:
                                restored_item, offset = _unflatten_from_gpu(t, gpu_buffer, offset)
                                restored_list.append(restored_item)
                            return tuple(restored_list), offset
                        elif isinstance(batch_data, list):
                            restored_list = []
                            for t in batch_data:
                                restored_item, offset = _unflatten_from_gpu(t, gpu_buffer, offset)
                                restored_list.append(restored_item)
                            return restored_list, offset
                        elif isinstance(batch_data, dict):
                            restored_dict = {}
                            for k, t in batch_data.items():
                                restored_item, offset = _unflatten_from_gpu(t, gpu_buffer, offset)
                                restored_dict[k] = restored_item
                            return restored_dict, offset
                        return batch_data, offset

                    gpu_buf = gpu_pool.buffers[gpu_idx]
                    restored_batch, _ = _unflatten_from_gpu(collated, gpu_buf)

                    yield restored_batch

                    gpu_pool.release(gpu_idx)

                elif self.pin_memory:
                    def _pin(data):
                        if isinstance(data, torch.Tensor) and data.device.type == "cpu":
                            return data.pin_memory()
                        elif isinstance(data, tuple):
                            return tuple(_pin(d) for d in data)
                        elif isinstance(data, list):
                            return [_pin(d) for d in data]
                        elif isinstance(data, dict):
                            return {k: _pin(v) for k, v in data.items()}
                        return data

                    yield _pin(collated)
                else:
                    yield collated
        finally:
            # No cleanup needed for daemon threads
            pass
