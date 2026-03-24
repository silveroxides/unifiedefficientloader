# unifiedefficientloader

A unified interface for loading safetensors, handling CPU/GPU pinned transfers, and converting between tensors and dicts.

## Installation

You can install this package via pip. Since it heavily relies on `torch` and `safetensors` but doesn't strictly force them as hard dependencies for package building/installation, make sure you have them installed in your environment:

```bash
pip install unifiedefficientloader
pip install torch safetensors tqdm
```

## Usage

### Unified Safetensors Loader

```python
from unifiedefficientloader import UnifiedSafetensorsLoader

# Standard mode (preload all)
with UnifiedSafetensorsLoader("model.safetensors", low_memory=False) as loader:
    tensor = loader.get_tensor("weight_name")

# Low memory mode (streaming)
with UnifiedSafetensorsLoader("model.safetensors", low_memory=True) as loader:
    for key in loader.keys():
        tensor = loader.get_tensor(key)
        # Process tensor...
        loader.mark_processed(key) # Frees memory
```

### Loading Specific Tensors Dynamically (Header Analysis)

You can analyze the file's header without loading the entire multi-gigabyte safetensors file into memory. This allows you to locate specific data (like embedded JSON dictionaries stored as `uint8` tensors) and load *only* those specific tensors directly from their file offsets.

```python
from unifiedefficientloader import UnifiedSafetensorsLoader, tensor_to_dict

with UnifiedSafetensorsLoader("model.safetensors", low_memory=True) as loader:
    # 1. Analyze the header metadata without loading any tensors
    # loader._header contains the full safetensors header directory
    uint8_tensor_keys = [
        key for key, info in loader._header.items()
        if isinstance(info, dict) and info.get("dtype") == "U8"
    ]
    
    # 2. Load ONLY those specific tensors using their keys
    for key in uint8_tensor_keys:
        # get_tensor dynamically reads only the bytes for this tensor 
        # based on the offsets found in the header
        loaded_tensor = loader.get_tensor(key)
        
        # 3. Decode the uint8 tensor back into a Python dictionary
        extracted_dict = tensor_to_dict(loaded_tensor)
        print(f"Decoded {key}:", extracted_dict)
```

### Optimized Asynchronous Streaming via ThreadPoolExecutor

For maximum I/O throughput while maintaining strict memory backpressure, use `async_stream`. This utilizes a `ThreadPoolExecutor` for background disk reading and a bounded queue to prevent memory exhaustion. By setting `pin_memory=True`, memory pinning is performed sequentially in the main thread to avoid OS-level lock contention and preserve high DMA transfer speeds.

```python
from unifiedefficientloader import UnifiedSafetensorsLoader, transfer_to_gpu_pinned

with UnifiedSafetensorsLoader("model.safetensors", low_memory=True) as loader:
    keys_to_load = loader.keys()
    
    # Create the continuous streaming generator
    # prefetch_batches controls how many batches to buffer in memory
    stream = loader.async_stream(
        keys_to_load, 
        batch_size=8, 
        prefetch_batches=2, 
        pin_memory=True
    )
    
    # Iterate directly over the generator
    for batch in stream:
        for key, pinned_tensor in batch:
            # Transfer directly to GPU via DMA (pinning is already done)
            gpu_tensor = transfer_to_gpu_pinned(pinned_tensor, device="cuda")
            
            # ... process gpu_tensor ...
            loader.mark_processed(key)
```

### Tensor/Dict Conversion

```python
from unifiedefficientloader import dict_to_tensor, tensor_to_dict

my_dict = {"param": 1.0, "name": "test"}
tensor = dict_to_tensor(my_dict)
recovered_dict = tensor_to_dict(tensor)
```

### Pinned Memory Transfers

```python
import torch
from unifiedefficientloader import transfer_to_gpu_pinned

tensor = torch.randn(100, 100)
# Transfers using pinned memory if CUDA is available, otherwise falls back gracefully
gpu_tensor = transfer_to_gpu_pinned(tensor, device="cuda:0")