# IncrementalSafetensorsWriter

The `IncrementalSafetensorsWriter` is a memory-efficient safetensors writer that supports dynamic, incremental streaming. It acts as a "Dumb Sink" that writes tensors to disk dynamically using background threads. 

By reserving a fixed-size block for the JSON header at the start of the file, it completely eliminates the need to pre-calculate all tensor shapes and offsets before writing. This allows you to process massive models (e.g., 50GB+) on hardware with limited memory using a simple stream-and-forget loop.

## Core Concepts

Unlike standard `safetensors.torch.save_file` which requires all tensors to be passed in a single dictionary (resulting in massive RAM usage), incremental saving uses a **dynamic streaming** approach:

1. **Header Reservation**: When the writer opens, it reserves a large, fixed-size block (default 1MB) at the beginning of the file. 
2. **Dynamic Data Append**: As you call `write(name, tensor)`, the writer calculates the necessary offsets on the fly and immediately dispatches the bytes to a background thread to be written to the data section of the file.
3. **Strict Memory Discipline**: The background thread explicitly calls `del tensor` internally immediately after the file write completes. If you drop your reference to the tensor, it will be garbage collected the moment it hits the disk.
4. **Header Finalization**: When the writer is closed, it converts the collected manifest into JSON, pads it with spaces to exactly match the reserved block size, and writes it to the start of the file.

## Basic Usage

### Saving a Model Incrementally

```python
import torch
from unifiedefficientloader import IncrementalSafetensorsWriter

# 1. Initialize with an optional metadata dictionary
# max_header_bytes defaults to 1MB, which is plenty for >10,000 tensors.
writer = IncrementalSafetensorsWriter("output.safetensors", metadata={"version": "1.0"})

# 2. Stream tensors into the file
with writer:
    # Generate or process your tensor...
    w = torch.randn((1024, 1024), dtype=torch.float16)
    
    # Hand off to the background thread pool
    writer.write("layer.0.weight", w)
    
    # Immediately release from RAM!
    del w 
    
    # Continue streaming the rest...
    b = torch.zeros((1024,), dtype=torch.float16)
    writer.write("layer.0.bias", b)
    del b
```

## Transforming an Existing Model Structure

If you are processing a model (e.g., quantizing or merging), you simply iterate through the source model and write the transformed tensors directly to the writer.

```python
from unifiedefficientloader import UnifiedSafetensorsLoader, IncrementalSafetensorsWriter

loader = UnifiedSafetensorsLoader("source.safetensors", low_memory=True)

# Preserve metadata from the source
writer = IncrementalSafetensorsWriter("quantized.safetensors", metadata=loader.metadata())

# The Streaming Lifecycle Loop
with writer:
    for key in loader.keys():
        src_t = loader.get_tensor(key)                # 1. Loader -> Memory
        gpu_t = src_t.to("cuda")                      # 2. Memory -> GPU
        del src_t                                     # <--- Clean up Step 1
        
        out_gpu_t = my_quantize_func(gpu_t)           # 3. Process on GPU
        out_t = out_gpu_t.cpu()                       # 4. GPU -> Memory
        del gpu_t, out_gpu_t                          # <--- Clean up Steps 2 & 3
        
        writer.write(key, out_t)                      # 5. Memory -> Writer Queue
        del out_t                                     # <--- Clean up Step 4
        
    # Write any new tensors we generated (e.g., quantization scales)
    scale_tensor = torch.ones((128,), dtype=torch.float32)
    writer.write("layer.0.scale", scale_tensor)
    del scale_tensor
```

## API Reference

### `IncrementalSafetensorsWriter(filename: str, metadata: dict = None, max_header_bytes: int = 1048576, max_workers: int = 4)`
- `filename`: Target path for the `.safetensors` file.
- `metadata`: Optional dictionary to be serialized into the `__metadata__` header property.
- `max_header_bytes`: Bytes to reserve at the start of the file for the JSON header. It must be large enough to hold all tensor names, shapes, and offsets. 1MB (1048576) is typically enough for 10,000+ tensors. The writer will automatically enforce 8-byte alignment.
- `max_workers`: Number of background threads to use for writing data. Defaults to 4.

### `write(name: str, tensor: torch.Tensor)`
Dynamically registers and dispatches a background write task.
- `name`: The key name for the tensor in the safetensors file.
- `tensor`: The PyTorch tensor data. Must be contiguous and on CPU.

**Important**: The `write` method blocks via a semaphore if the background threads are overwhelmed. This acts as backpressure to prevent your quantization/generation loop from out-pacing disk speeds and causing an Out-Of-Memory (OOM) error. The background worker explicitly calls `del tensor` internally immediately after the file write completes.