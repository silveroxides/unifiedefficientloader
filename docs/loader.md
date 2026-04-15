# UnifiedSafetensorsLoader

Core loader class. Supports three loading strategies selectable at construction time.

## Constructor

```python
UnifiedSafetensorsLoader(
    filename: str,
    low_memory: bool = False,
    direct_gpu: bool = False,
    use_mmap: bool = False,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filename` | `str` | — | Path to `.safetensors` file |
| `low_memory` | `bool` | `False` | Stream tensors on-demand instead of preloading all |
| `direct_gpu` | `bool` | `False` | Enable pinned-buffer → GPU DMA pipeline (forces `low_memory=True`) |
| `use_mmap` | `bool` | `False` | Map file into virtual memory for zero-copy access (requires UEL native lib) |

Backward-compatibility alias: `MemoryEfficientSafeOpen`.

---

## Loading Strategies

### Standard (preload)

`low_memory=False`. All tensors loaded into RAM at construction. Fast random access, higher peak RAM.

```python
from unifiedefficientloader import UnifiedSafetensorsLoader

with UnifiedSafetensorsLoader("model.safetensors") as loader:
    tensor = loader.get_tensor("weight_name")
    print(tensor.shape)
```

---

### Low-memory streaming

`low_memory=True`. Only the file header is read at construction. Each `get_tensor()` call reads that tensor's bytes from disk on-demand.

```python
from unifiedefficientloader import UnifiedSafetensorsLoader

with UnifiedSafetensorsLoader("model.safetensors", low_memory=True) as loader:
    for key in loader.keys():
        tensor = loader.get_tensor(key)
        # process tensor ...
        loader.mark_processed(key)  # frees memory
```

---

### Async streaming

`async_stream()` uses a `ThreadPoolExecutor` for parallel disk reads with a bounded queue for memory backpressure. Best throughput for bulk loading on `low_memory=True`.

```python
from unifiedefficientloader import UnifiedSafetensorsLoader, transfer_to_gpu_pinned

with UnifiedSafetensorsLoader("model.safetensors", low_memory=True) as loader:
    stream = loader.async_stream(
        loader.keys(),
        batch_size=8,
        prefetch_batches=2,
        pin_memory=True,
    )
    for batch in stream:
        for key, tensor in batch:
            gpu_tensor = transfer_to_gpu_pinned(tensor, device="cuda")
            # process gpu_tensor ...
            loader.mark_processed(key)
```

---

### Header analysis — selective tensor loading

Read only specific tensors by inspecting the header first.

```python
from unifiedefficientloader import UnifiedSafetensorsLoader, tensor_to_dict

with UnifiedSafetensorsLoader("model.safetensors", low_memory=True) as loader:
    # Find all 1D uint8 tensors (commonly used to store embedded JSON)
    dict_keys = [
        k for k, v in loader._header.items()
        if isinstance(v, dict) and v.get("dtype") == "U8" and len(v.get("shape", [])) == 1
    ]
    for key in dict_keys:
        tensor = loader.get_tensor(key)
        data = tensor_to_dict(tensor)
        print(f"{key}:", data)
```

---

## API Reference

### `keys() -> list[str]`

Returns list of all tensor keys in the file.

```python
keys = loader.keys()
```

---

### `metadata() -> dict[str, str]`

Returns the file-level metadata dict stored in the safetensors header.

```python
meta = loader.metadata()
print(meta.get("format"))
```

---

### `get_tensor(key: str) -> torch.Tensor`

Returns tensor by key.

- Standard mode: returns from preloaded cache.
- Low-memory mode: reads from file on-demand.
- MMAP mode: returns zero-copy view into mapped memory.

```python
tensor = loader.get_tensor("layer1.weight")
```

---

### `get_shape(key: str) -> tuple`

Returns tensor shape without loading data. In `low_memory` mode reads from header only.

```python
shape = loader.get_shape("layer1.weight")  # e.g. (512, 256)
```

---

### `get_ndim(key: str) -> int`

Returns number of dimensions without loading data.

```python
ndim = loader.get_ndim("layer1.weight")  # e.g. 2
```

---

### `mark_processed(key: str)`

Signals that a tensor has been consumed.

- Standard mode: deletes tensor from cache and calls `gc.collect()`.
- Low-memory mode: releases GPU buffer back to pool if `direct_gpu=True`.

```python
loader.mark_processed("layer1.weight")
```

---

### `load_all() -> dict[str, torch.Tensor]`

Loads all tensors and returns as a dict.

- Standard mode: returns shallow copy of preloaded cache.
- Low-memory mode: uses `async_stream` internally for parallel I/O.
- MMAP mode: returns zero-copy views from mapped memory.

```python
state_dict = loader.load_all()
model.load_state_dict(state_dict)
```

---

### `async_stream(keys, batch_size, prefetch_batches, pin_memory) -> Generator`

Streams tensors asynchronously using a background thread pool.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `keys` | `list[str]` | — | Ordered list of keys to stream |
| `batch_size` | `int` | `1` | Tensors per yielded batch |
| `prefetch_batches` | `int` | `2` | Batches to buffer ahead in the queue |
| `pin_memory` | `bool` | `False` | Pin tensors to page-locked memory in main thread |

Yields `list[tuple[str, torch.Tensor]]` batches in key order.

```python
for batch in loader.async_stream(loader.keys(), batch_size=16, prefetch_batches=2):
    for key, tensor in batch:
        # tensor is ready
        pass
```

---

### `close()`

Closes file handles and releases all resources including MMAP mappings.
Called automatically when used as a context manager.

```python
loader.close()
```

---

## Supported dtypes

All safetensors dtype strings are supported:

| Safetensors dtype | PyTorch dtype |
|---|---|
| `F64` | `torch.float64` |
| `F32` | `torch.float32` |
| `F16` | `torch.float16` |
| `BF16` | `torch.bfloat16` |
| `F8_E5M2` | `torch.float8_e5m2` (if available) |
| `F8_E4M3` | `torch.float8_e4m3fn` (if available) |
| `I64` | `torch.int64` |
| `I32` | `torch.int32` |
| `I16` | `torch.int16` |
| `I8` | `torch.int8` |
| `U64` | `torch.uint64` (if available) |
| `U32` | `torch.uint32` (if available) |
| `U16` | `torch.uint16` (if available) |
| `U8` | `torch.uint8` |
| `BOOL` | `torch.bool` |
| `C64` | `torch.complex64` |
