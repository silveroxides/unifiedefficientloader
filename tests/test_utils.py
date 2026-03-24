import pytest
import os
import tempfile
import json
import torch
import warnings
from safetensors.torch import save_file
from unifiedefficientloader import (
    UnifiedSafetensorsLoader,
    dict_to_tensor,
    tensor_to_dict,
    transfer_to_gpu_pinned,
    get_pinned_transfer_stats,
    reset_pinned_transfer_stats
)

# Silence internal PyTorch dataloader deprecation warnings
warnings.filterwarnings("ignore", message=".*The argument 'device' of Tensor.*")

@pytest.fixture(scope="module")
def sample_safetensors_file():
    """Create a temporary safetensors file with various tensor types for testing."""
    # Create some dummy tensors
    tensors = {
        "layer1.weight": torch.randn(10, 10, dtype=torch.float32),
        "layer1.bias": torch.randn(10, dtype=torch.float32),
        "layer2.weight": torch.randn(5, 5, dtype=torch.float16),
    }
    
    # Add a uint8 tensor created from a dictionary
    metadata_dict = {"model_name": "test_model", "version": 1.0, "is_quantized": True}
    tensors["metadata_tensor"] = dict_to_tensor(metadata_dict)

    # Save to a temporary file
    fd, path = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    
    save_file(tensors, path, metadata={"format": "pt"})
    
    yield path, tensors, metadata_dict
    
    # Cleanup
    if os.path.exists(path):
        os.remove(path)

def test_unified_loader_header_and_low_memory(sample_safetensors_file):
    filepath, original_tensors, original_metadata_dict = sample_safetensors_file
    
    # Load using low_memory=True
    with UnifiedSafetensorsLoader(filepath, low_memory=True) as loader:
        # Check that we can read keys and header correctly without loading everything
        keys = loader.keys()
        for expected_key in original_tensors.keys():
            assert expected_key in keys
            
        # Check metadata
        assert loader.metadata().get("format") == "pt"
        
        # Check get_shape and get_ndim using header info
        assert loader.get_shape("layer1.weight") == (10, 10)
        assert loader.get_ndim("layer2.weight") == 2
        
        # Load a specific tensor from offset and check values
        loaded_weight = loader.get_tensor("layer1.weight")
        assert torch.equal(loaded_weight, original_tensors["layer1.weight"])
        assert loaded_weight.dtype == torch.float32
        
        loaded_bias = loader.get_tensor("layer1.bias")
        assert torch.equal(loaded_bias, original_tensors["layer1.bias"])
        
        # Dynamically locate any torch.uint8 tensor in the file header
        uint8_tensor_keys = [
            k for k, v in loader._header.items()
            if isinstance(v, dict) and v.get("dtype") == "U8"
        ]
        
        # Verify we found at least one
        assert len(uint8_tensor_keys) > 0, "No U8 tensors found in the safetensors header!"
        
        # Load those specific uint8 tensors and receive the dict from them
        for dict_tensor_key in uint8_tensor_keys:
            # We specifically only load this tensor from the file
            loaded_metadata_tensor = loader.get_tensor(dict_tensor_key)
            assert loaded_metadata_tensor.dtype == torch.uint8
            
            # Verify it converts back to dict correctly
            extracted_dict = tensor_to_dict(loaded_metadata_tensor)
            
            if dict_tensor_key == "metadata_tensor":
                assert extracted_dict == original_metadata_dict

def test_pinned_transfer_with_loaded_tensors(sample_safetensors_file):
    filepath, original_tensors, _ = sample_safetensors_file
    
    with UnifiedSafetensorsLoader(filepath, low_memory=True) as loader:
        tensor = loader.get_tensor("layer1.weight")
        
        # Test CPU transfer (which should just return the tensor or a copy if dtype changes)
        reset_pinned_transfer_stats()
        cpu_result = transfer_to_gpu_pinned(tensor, device="cpu")
        assert cpu_result.device.type == "cpu"
        stats = get_pinned_transfer_stats()
        assert stats["pinned"] == 0
        
        # If CUDA is available, test the pinned transfer mechanism
        if torch.cuda.is_available():
            reset_pinned_transfer_stats()
            gpu_result = transfer_to_gpu_pinned(tensor, device="cuda")
            
            assert gpu_result.device.type == "cuda"
            stats = get_pinned_transfer_stats()
            assert stats["pinned"] == 1
            assert stats["fallback"] == 0

def test_async_stream_loading(sample_safetensors_file):
    filepath, original_tensors, _ = sample_safetensors_file
    
    with UnifiedSafetensorsLoader(filepath, low_memory=True) as loader:
        keys_to_load = list(original_tensors.keys())
        # Use async_stream directly with pin_memory=True
        stream = loader.async_stream(keys_to_load, batch_size=2, pin_memory=True if torch.cuda.is_available() else False)
        
        loaded_count = 0
        
        for batch in stream:
            for k, tensor in batch:
                assert k in original_tensors
                # Ensure the loaded tensor matches the original
                assert torch.equal(tensor, original_tensors[k])
                
                # Verify that it is properly pinned if requested and CUDA is available
                if torch.cuda.is_available():
                    assert tensor.is_pinned()
                    
                loaded_count += 1
                
        assert loaded_count == len(keys_to_load)