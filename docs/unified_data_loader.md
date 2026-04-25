# UnifiedDataLoader

A drop-in, high-performance replacement for `torch.utils.data.DataLoader`. Designed to eliminate the IPC overhead of Python multiprocessing and provide a direct, zero-copy pipeline to the GPU.

## Why use UnifiedDataLoader?

The standard PyTorch `DataLoader` uses multiprocessing for background loading. This introduces significant Inter-Process Communication (IPC) overhead, especially when moving large image batches or tensors across process boundaries. Additionally, using `.to(device)` blocks the main CPU thread while waiting for the GPU transfer to complete.

`UnifiedDataLoader` solves this by using:
1. **Thread Pools:** `ThreadPoolExecutor` handles background loading in threads instead of processes, eliminating IPC serialization costs.
2. **Pinned Memory Pools:** Reuses pre-allocated pinned memory buffers (`PinnedBufferPool`) to avoid per-batch allocation overhead.
3. **Direct-to-GPU Streaming:** When `direct_gpu=True` is enabled, batches are copied asynchronously from pinned memory directly into pre-allocated GPU slabs (`GpuBufferPool`) using a dedicated CUDA stream. This hides PCIe transfer latency completely behind disk I/O.
4. **UEL Zero-Copy:** If the `uel` C extension is present, the pinned memory pools are allocated natively, avoiding PyTorch overhead entirely on the CPU side.

## Basic Usage

```python
from unifiedefficientloader import UnifiedDataLoader
from torchvision import datasets, transforms

dataset = datasets.FakeData(transform=transforms.ToTensor())

# Standard CPU threading replacement
loader = UnifiedDataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

for batch_image, batch_label in loader:
    # batch is on CPU
    pass
```

## Direct GPU Pipeline

To maximize throughput, stream batches directly into VRAM.

```python
loader = UnifiedDataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    direct_gpu=True, # Enables the zero-copy pipeline
    prefetch_batches=2 # Number of batches to pre-load into VRAM
)

for batch_image, batch_label in loader:
    # batch is already on the GPU (device="cuda")
    # No need to call .to(device)
    pass
```

## Configuration Options

`UnifiedDataLoader(dataset, **kwargs)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `batch_size` | `int` | `1` | Number of samples per batch. |
| `shuffle` | `bool` | `False` | Randomize indices each epoch. |
| `num_workers` | `int` | `0` | Number of background threads. `0` means main-thread loading. |
| `prefetch_batches` | `int` | `2` | Number of batches to keep ahead in the pipeline. |
| `pin_memory` | `bool` | `False` | Pin CPU tensors for faster GPU transfer (ignored if `direct_gpu=True`). |
| `direct_gpu` | `bool` | `False` | Enable the direct-to-GPU streaming pipeline using pre-allocated pools. |
| `drop_last` | `bool` | `False` | Drop the last incomplete batch if dataset size isn't divisible by `batch_size`. |
| `device` | `str` | `"cuda"` | Target device for `direct_gpu=True`. |

## Fallbacks

`UnifiedDataLoader` is designed to be highly defensive.
* If `direct_gpu=True` is requested but CUDA is unavailable, it warns and falls back to standard CPU loading.
* If the `uel` C extension is missing, it falls back to PyTorch's `torch.empty(..., pin_memory=True)` for buffer pools.
* If the dataset yields an unsupported datatype (e.g. nested lists of custom objects), `direct_gpu` will safely fall back to standard collate.
