# Direct-to-GPU Streaming

`direct_gpu=True` activates a zero-copy pipeline that transfers tensors from disk to GPU without intermediate Python allocations. Data flows: disk → pinned CPU buffer → GPU slab via async CUDA DMA.

## Requirements

- `low_memory=True` (forced automatically if not set).
- CUDA-capable GPU.
- `torch.cuda.is_available()` must return `True`.

If CUDA is unavailable at runtime, `direct_gpu` silently falls back to standard CPU streaming with a warning.

## How it works

1. `GpuBufferPool` pre-allocates a slab of GPU memory large enough for the largest tensor in the file.
2. `PinnedBufferPool` pre-allocates page-locked CPU staging buffers of the same size.
3. Worker threads read tensor bytes from disk directly into pinned buffers using `readinto()` on a ctypes view of the pinned tensor memory (zero Python copy).
4. Each worker issues an async `copy_()` to the GPU via a shared CUDA stream.
5. The CUDA event is synchronized before the tensor is yielded, guaranteeing correct values.
6. After yield, the GPU buffer is released back to the pool via `mark_processed()`.

## Usage

### Via `load_all()`

```python
from unifiedefficientloader import UnifiedSafetensorsLoader

with UnifiedSafetensorsLoader("model.safetensors", low_memory=True, direct_gpu=True) as loader:
    state_dict = loader.load_all()

for key, tensor in state_dict.items():
    assert tensor.device.type == "cuda"
```

### Via `async_stream()`

```python
from unifiedefficientloader import UnifiedSafetensorsLoader

with UnifiedSafetensorsLoader("model.safetensors", low_memory=True, direct_gpu=True) as loader:
    stream = loader.async_stream(
        loader.keys(),
        batch_size=8,
        prefetch_batches=2,
    )
    for batch in stream:
        for key, gpu_tensor in batch:
            assert gpu_tensor.device.type == "cuda"
            # process gpu_tensor ...
            loader.mark_processed(key)  # releases GPU buffer back to pool
```

## Buffer pool sizing

Pool sizes are calculated automatically from `prefetch_batches` and `num_workers`:

```
num_buffers = prefetch_batches + num_workers + 1
buffer_size = max_tensor_bytes * batch_size
```

Total VRAM consumed = `num_buffers × buffer_size`. Logged at startup:

```
Direct GPU pool: 6 slots x 512.0 MB each (VRAM budget: 3072.0 MB)
```

The pool is intentionally bounded — it holds only the in-flight pipeline window,
not the full dataset. Ensure your GPU has sufficient headroom before model weights
are loaded.

## Memory cleanup

GPU buffers are returned to the pool only when `mark_processed(key)` is called. Always call it after consuming each tensor:

```python
for batch in stream:
    for key, tensor in batch:
        process(tensor)
        loader.mark_processed(key)  # required — returns GPU slab to pool
```

Forgetting `mark_processed` will exhaust the pool and stall the pipeline.

## Fallback behaviour

| Condition | Behaviour |
|---|---|
| `direct_gpu=True`, CUDA unavailable | Falls back to CPU streaming, warning logged |
| Pool init failure | Falls back to CPU streaming, warning logged |
| Worker read failure | Falls back to synchronous `get_tensor()` for that key |

## Performance characteristics

Observed throughput order from fastest to slowest when `uel` native extension
is present:

```
async_stream (use_mmap=True)  >  async_stream  >  direct_gpu
```

`async_stream + use_mmap=True` eliminates disk read overhead entirely — the OS
page cache serves tensors directly from mapped virtual memory with no file IO
after the first pass. `async_stream` without mmap is the next fastest because
it keeps pure CPU-side copies minimal. `direct_gpu=True` adds PCIe transfer
overhead per batch on top of disk IO, which only pays off when tensors are large
enough that the async DMA overlap meaningfully hides PCIe latency.

| Mode | RAM used | VRAM used | GPU transfer | Notes |
|---|---|---|---|---|
| Standard IO + `transfer_to_gpu_pinned` | Full model | Full model | Explicit per-tensor | Baseline; no streaming |
| `async_stream(pin_memory=True)` | Pool only | Full model | Pinned DMA per tensor | Good general-purpose choice |
| `async_stream` + `use_mmap=True` | OS page cache | Full model | Pinned DMA per tensor | Fastest when uel available |
| `direct_gpu=True` | Pool only | Pool + model | Async DMA from pinned buf | Best for very large tensors on PCIe-bottlenecked systems |
| `direct_gpu=True` + `use_mmap=True` | OS page cache | Pool + model | Async DMA from mapped pages | Maximum pipeline overlap when uel available |
