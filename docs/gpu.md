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

Pool sizes are calculated automatically based on the largest tensor in the key list:

```
num_buffers = (max_in_flight + max_workers) * 2 + 2
```

Where:
- `max_workers = min(16, max(4, batch_size))`
- `max_in_flight = max(max_workers, prefetch_batches * batch_size)`

Total VRAM consumed = `num_buffers × max_tensor_bytes`. Logged at startup:

```
Direct GPU pipeline initialized: 98 buffers, max 512.0MB each (Total VRAM: 50176.0MB)
```

Ensure your GPU has sufficient headroom before the model weights are loaded into it.

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

## Comparison

| Mode | RAM used | VRAM used | GPU transfer |
|---|---|---|---|
| Standard IO + `transfer_to_gpu_pinned` | Full model | Full model | Explicit per-tensor |
| `async_stream(pin_memory=True)` | Streaming | Full model | Pinned DMA per tensor |
| `direct_gpu=True` | Pool only | Pool + model | Async DMA, overlapped with disk I/O |
