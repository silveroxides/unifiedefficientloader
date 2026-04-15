# Zero-Copy MMAP Loading

`use_mmap=True` activates OS memory-mapped file loading via the `uel` native C extension. The file is mapped into virtual memory — no data is copied into RAM at load time. The OS pages in data on demand when tensors are actually accessed.

## Requirements

- The `uel` native extension must be compiled and installed. See [building.md](building.md).
- Windows and Linux only. macOS not supported.
- NVIDIA GPU required for CUDA-backed features within the extension.

## How it works

| Mode | What happens at `get_tensor()` |
|---|---|
| Standard IO | `f.readinto()` copies bytes from disk into a new `bytearray` |
| MMAP | Returns a `torch.frombuffer` view directly into the mapped file pages |

In MMAP mode no data moves — PyTorch holds a pointer into the OS page cache. The `ModelMMAP` handle is attached to the tensor's storage as `_uel_mmap_ref` to keep the mapping alive as long as the tensor is alive.

## Usage

### Basic MMAP load

```python
from unifiedefficientloader import UnifiedSafetensorsLoader

with UnifiedSafetensorsLoader("model.safetensors", low_memory=True, use_mmap=True) as loader:
    tensor = loader.get_tensor("layer1.weight")
    # tensor is a read-only view into mapped memory
    # clone() to get a writable copy
    writable = tensor.clone()
```

### Load full state dict via MMAP

```python
from unifiedefficientloader import UnifiedSafetensorsLoader

with UnifiedSafetensorsLoader("model.safetensors", low_memory=True, use_mmap=True) as loader:
    state_dict = loader.load_all()
    # All tensors are zero-copy views — clone before modifying
```

### MMAP with async_stream (parallel page-fault prefetch)

Workers read from the MMAP pointer into buffers in parallel, forcing OS page faults concurrently. This prefetches file pages while the main thread processes earlier batches.

```python
from unifiedefficientloader import UnifiedSafetensorsLoader, transfer_to_gpu_pinned

with UnifiedSafetensorsLoader("model.safetensors", low_memory=True, use_mmap=True) as loader:
    stream = loader.async_stream(
        loader.keys(),
        batch_size=16,
        prefetch_batches=2,
        pin_memory=True,
    )
    for batch in stream:
        for key, tensor in batch:
            gpu_tensor = transfer_to_gpu_pinned(tensor, device="cuda")
            # process ...
            loader.mark_processed(key)
```

## Graceful fallback

If the `uel` native library is not available (not compiled or not installed), `use_mmap=True` silently falls back to standard async IO. No exception is raised. A warning is logged:

```
[WARNING] Failed to initialize MMAP: unifiedefficientloader-uel is not initialized. Falling back to standard IO.
```

Check whether MMAP is active after construction:

```python
loader = UnifiedSafetensorsLoader("model.safetensors", low_memory=True, use_mmap=True)
print(loader.use_mmap)  # True if native lib loaded, False if fell back
```

## Read-only tensors

MMAP tensors are backed by read-only OS pages. PyTorch will warn if you attempt an in-place operation:

```
UserWarning: The given buffer is not writable
```

Call `.clone()` to obtain a writable copy before any in-place modification:

```python
tensor = loader.get_tensor("weight")
tensor = tensor.clone()  # now writable
tensor.mul_(0.5)
```

## Memory lifecycle

The `ModelMMAP` mapping stays alive as long as any tensor view referencing it is alive. When the loader is closed (via `close()` or context manager exit), the mapping reference is released. Existing tensor views remain valid as long as the Python tensor object is alive — the mapping is ref-counted via `_uel_mmap_ref` on the tensor storage.

```python
loader = UnifiedSafetensorsLoader("model.safetensors", low_memory=True, use_mmap=True)
tensor = loader.get_tensor("weight")
loader.close()  # mapping ref released from loader
# tensor still valid — storage holds its own ref to the mapping
del tensor      # mapping fully released here
```

## UEL Native Extension internals

The `uel` module (`unifiedefficientloader/uel/`) wraps a C shared library (`uel.dll` / `uel.so`) that provides:

| Component | Source | Description |
|---|---|---|
| `ModelMMAP` | `src-win/model-mmap.c`, `src-posix/model-mmap.c` | File memory-mapping |
| CUDA detour | `src-win/cuda-detour.c` | Hooks PyTorch CUDA allocator |
| VRAM budget | `src/model-vbar.c`, `src/vrambuf.c` | VRAM slab allocation |
| Host buffer | `src/hostbuf.c` | Pinned CPU buffer management |
| Control | `src/control.c` | Init, device selection, VRAM query |

Python wrappers live in `unifiedefficientloader/uel/`:

| Module | Description |
|---|---|
| `control.py` | `init()`, `deinit()`, log level, VRAM query |
| `model_mmap.py` | `ModelMMAP` class |
| `model_vbar.py` | VRAM budget ring-buffer |
| `host_buffer.py` | Pinned host buffer allocation |
| `torch.py` | PyTorch allocator integration |
