# Utilities

## Tensor / Dict Conversion

Serialize Python dicts into `torch.uint8` tensors and back. Used to embed metadata or config dicts directly inside a safetensors file alongside model weights.

### `dict_to_tensor(data_dict: dict) -> torch.Tensor`

Serializes a dict to a 1D `torch.uint8` tensor containing UTF-8 JSON bytes.

```python
from unifiedefficientloader import dict_to_tensor

config = {"model": "MyModel", "version": 2, "quantized": False}
tensor = dict_to_tensor(config)
# tensor.dtype == torch.uint8
# tensor.ndim == 1
```

### `tensor_to_dict(tensor_data) -> dict`

Deserializes a 1D `torch.uint8` tensor back to a Python dict.

```python
from unifiedefficientloader import tensor_to_dict

recovered = tensor_to_dict(tensor)
assert recovered == config
```

Raises `ValueError` if tensor is not 1D.

### Round-trip example

```python
from unifiedefficientloader import dict_to_tensor, tensor_to_dict
from safetensors.torch import save_file
import torch

weights = {"layer.weight": torch.randn(256, 128)}
metadata = dict_to_tensor({"arch": "mlp", "layers": 4})

save_file({**weights, "metadata": metadata}, "model.safetensors")
```

```python
from unifiedefficientloader import UnifiedSafetensorsLoader, tensor_to_dict

with UnifiedSafetensorsLoader("model.safetensors", low_memory=True) as loader:
    dict_keys = [
        k for k, v in loader._header.items()
        if isinstance(v, dict) and v.get("dtype") == "U8" and len(v.get("shape", [])) == 1
    ]
    for key in dict_keys:
        data = tensor_to_dict(loader.get_tensor(key))
        print(data)  # {"arch": "mlp", "layers": 4}
```

---

## Pinned Memory Transfer

### `transfer_to_gpu_pinned(tensor, device, dtype, non_blocking) -> torch.Tensor`

Transfers a CPU tensor to GPU using page-locked (pinned) memory for faster DMA throughput.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tensor` | `torch.Tensor` | — | Source tensor (CPU) |
| `device` | `str` | `"cuda"` | Target device string |
| `dtype` | `torch.dtype \| None` | `None` | Optional dtype cast during transfer |
| `non_blocking` | `bool` | `True` | Use non-blocking CUDA transfer |

Automatically falls back to a standard `.to(device)` call if:
- Tensor is already on GPU
- CUDA is unavailable
- Target device is not CUDA
- `pin_memory()` raises (e.g. insufficient page-locked memory)

```python
import torch
from unifiedefficientloader import transfer_to_gpu_pinned

tensor = torch.randn(1024, 1024)
gpu_tensor = transfer_to_gpu_pinned(tensor, device="cuda:0")
assert gpu_tensor.device.type == "cuda"
```

With dtype cast:

```python
gpu_f16 = transfer_to_gpu_pinned(tensor, device="cuda", dtype=torch.float16)
```

### Transfer statistics

```python
from unifiedefficientloader import get_pinned_transfer_stats, reset_pinned_transfer_stats

reset_pinned_transfer_stats()

# ... run transfers ...

stats = get_pinned_transfer_stats()
print(stats)  # {"pinned": 42, "fallback": 0}
```

`pinned` — number of transfers that used pinned memory.
`fallback` — number of transfers that fell back to standard copy.

### `set_verbose(enabled: bool)`

Enables verbose logging for each pinned transfer (logs shape and size).

```python
from unifiedefficientloader import set_verbose
set_verbose(True)
```

---

## Logging

`unifiedefficientloader` uses a custom logger with four levels.

| Level | Value | When |
|---|---|---|
| `MINIMAL` | 30 | Warnings and errors only |
| `NORMAL` | 20 | Standard operational messages (default) |
| `VERBOSE` | 15 | Detailed per-tensor and pipeline messages |
| `DEBUG` | 10 | Every function call with args and return values |

### `setup_logging(verbose_arg: str)`

Configure log level by name.

```python
from unifiedefficientloader import setup_logging

setup_logging("NORMAL")    # default
setup_logging("VERBOSE")   # pipeline details
setup_logging("DEBUG")     # full trace
setup_logging("MINIMAL")   # warnings/errors only
```

### Convenience log functions

All write to the `unifiedefficientloader` logger.

```python
from unifiedefficientloader import debug, verbose, normal, info, warning, error, minimal

debug("low-level trace")
verbose("pipeline detail")
normal("operational message")
info("alias for normal")
warning("something unexpected")
error("something failed")
minimal("critical only")
```

### Integration with Python `logging`

The logger name is `"unifiedefficientloader"`. Configure it via standard `logging`:

```python
import logging
logging.getLogger("unifiedefficientloader").setLevel(logging.WARNING)
```
