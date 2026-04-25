# IncrementalSafetensorsWriter

The `IncrementalSafetensorsWriter` is a memory-efficient safetensors writer that supports incremental streaming. It is designed to handle saving tensors to disk without holding the entire model in RAM, which is crucial for processing very large models (e.g., 50GB+) on hardware with limited memory.

## Core Concepts

Unlike standard `safetensors.torch.save_file` which requires all tensors to be passed in a single dictionary (resulting in massive RAM usage), `IncrementalSafetensorsWriter` uses a **three-phase** approach:

1. **Manifest Registration**: Register the exact structure of the output file (tensor names, shapes, and dtypes) without providing the actual data yet. You can even clone the structure of an existing file using `register_template`.
2. **Preallocation**: Generates the full Safetensors header, pads it to enforce strict 8-byte alignment, and creates a "dummy" file on disk of the exact final size using OS-level file truncation.
3. **Background Streaming**: Submit tensors one by one using a background `ThreadPoolExecutor`. The tensor data is written directly to its pre-calculated offset on disk. Since the writing happens in the background, your main thread can immediately `del` the tensor to free RAM.

## Basic Usage

### Saving a Model Incrementally

```python
import torch
from unifiedefficientloader import IncrementalSafetensorsWriter

# 1. Initialize with your desired output filename
writer = IncrementalSafetensorsWriter("output.safetensors", metadata={"version": "1.0"})

# 2. Register your tensors (the Manifest)
writer.register_tensor("layer.0.weight", shape=(1024, 1024), dtype=torch.float16)
writer.register_tensor("layer.0.bias", shape=(1024,), dtype=torch.float16)

# 3. Preallocate the dummy file on disk
writer.preallocate()

# 4. Stream tensors into the file
with writer:
    # Generate or process your tensor...
    w = torch.randn((1024, 1024), dtype=torch.float16)
    
    # Hand off to the background thread pool
    writer.write_tensor("layer.0.weight", w)
    
    # Immediately release from RAM!
    del w 
    
    # Continue streaming the rest...
    b = torch.zeros((1024,), dtype=torch.float16)
    writer.write_tensor("layer.0.bias", b)
    del b
```

## Cloning an Existing Model Structure

If you are processing a model (e.g., quantizing or merging), you can instantly clone the structure of the input model to serve as the template for your output file.

```python
from unifiedefficientloader import UnifiedSafetensorsLoader, IncrementalSafetensorsWriter

loader = UnifiedSafetensorsLoader("source.safetensors")
writer = IncrementalSafetensorsWriter("quantized.safetensors", metadata=loader.metadata())

# Clone the entire structure of the source model in milliseconds
writer.register_template(loader)

# Optional: Add any NEW tensors to the manifest (e.g., quantization scales)
writer.register_tensor("layer.0.scale", shape=(128,), dtype=torch.float32)

writer.preallocate()

with writer:
    for key in loader.keys():
        tensor = loader.get_tensor(key)
        
        # ... perform arbitrary logic ...
        quantized_tensor = my_quantize_func(tensor)
        
        # Write back to disk
        writer.write_tensor(key, quantized_tensor)
        
        # Important: Free memory immediately
        del tensor, quantized_tensor
```

## API Reference

### `IncrementalSafetensorsWriter(filename: str, metadata: dict = None, max_workers: int = 4)`
- `filename`: Target path for the `.safetensors` file.
- `metadata`: Optional dictionary to be serialized into the `__metadata__` header property.
- `max_workers`: Number of background threads to use for writing data. Defaults to 4.

### `register_tensor(name: str, shape: tuple, dtype)`
Registers a single tensor.
- `name`: Tensor key name in the safetensors file.
- `shape`: Tuple representing the tensor shape.
- `dtype`: The PyTorch dtype (e.g., `torch.float16` or `torch.float8_e4m3fn`).

### `register_template(loader: UnifiedSafetensorsLoader)`
Extracts the manifest (header shapes and dtypes) from an existing `UnifiedSafetensorsLoader` and adds it to the writer. Automatically copies the loader's metadata if the writer was not initialized with any.

### `preallocate()`
Calculates all offsets, constructs the JSON header, aligns the data block to an 8-byte boundary, and creates the full-size "dummy" file on disk using `os.truncate()`. Must be called before entering the context manager or writing tensors.

### `write_tensor(name: str, tensor: torch.Tensor)`
Dispatches a background write task.
- `name`: The registered tensor name.
- `tensor`: The PyTorch tensor data. Must be contiguous, on CPU, and strictly match the registered shape and dtype.

**Important**: The `write_tensor` method blocks via a semaphore if the background threads are overwhelmed. This acts as backpressure to prevent your quantization/generation loop from out-pacing disk speeds and causing an Out-Of-Memory (OOM) error.