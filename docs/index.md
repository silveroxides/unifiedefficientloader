# unifiedefficientloader

A unified interface for memory-efficient loading of safetensors files, CPU/GPU pinned transfers, and tensor/dict serialization.

## Contents

| Document | Description |
|---|---|
| [loader.md](loader.md) | `UnifiedSafetensorsLoader` — all modes and API reference |
| [mmap.md](mmap.md) | Zero-copy MMAP loading via the `uel` native extension |
| [gpu.md](gpu.md) | Direct-to-GPU streaming pipeline (`direct_gpu`) |
| [utils.md](utils.md) | Tensor/dict utilities, pinned transfers, logging |
| [building.md](building.md) | Compiling the native extension and producing wheels |

## Quick Install

```bash
pip install unifiedefficientloader
pip install torch safetensors tqdm
```

## Feature Overview

| Feature | Flag | Description |
|---|---|---|
| Preload all | `low_memory=False` | Loads all tensors upfront into RAM |
| Streaming | `low_memory=True` | Loads tensors on-demand from disk |
| Async stream | `async_stream()` | Parallel I/O via `ThreadPoolExecutor` |
| Direct GPU | `direct_gpu=True` | Pinned buffer → GPU DMA pipeline |
| Zero-copy MMAP | `use_mmap=True` | OS memory-maps the file, no disk reads |
