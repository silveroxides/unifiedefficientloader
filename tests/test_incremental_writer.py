import pytest
import os
import tempfile
import torch
import numpy as np
from safetensors.torch import save_file, load_file
from unifiedefficientloader import (
    UnifiedSafetensorsLoader,
    IncrementalSafetensorsWriter
)

@pytest.fixture
def temp_output_file():
    """Create a temporary path for the output safetensors file."""
    fd, path = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    if os.path.exists(path):
        os.remove(path) # Ensure it doesn't exist initially
    yield path
    if os.path.exists(path):
        os.remove(path)

@pytest.fixture
def source_safetensors_file():
    """Create a source safetensors file for testing streaming loops."""
    tensors = {
        "weight1": torch.randn(16, 16, dtype=torch.float32),
        "bias1": torch.randn(16, dtype=torch.float32),
    }
    metadata = {"test_key": "test_value"}

    fd, path = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    save_file(tensors, path, metadata=metadata)

    yield path, tensors, metadata

    if os.path.exists(path):
        os.remove(path)

def test_incremental_writer_basic(temp_output_file):
    filename = temp_output_file
    metadata = {"version": "1.0"}

    # Define some dummy tensors
    w1_shape = (10, 20)
    b1_shape = (10,)
    w1_data = torch.randn(w1_shape, dtype=torch.float32)
    b1_data = torch.randn(b1_shape, dtype=torch.float16)

    # Use a small max header for testing
    writer = IncrementalSafetensorsWriter(filename, metadata=metadata, max_header_bytes=1024)

    with writer:
        writer.write("w1", w1_data)
        writer.write("b1", b1_data)

    # Check file exists and has size
    assert os.path.exists(filename)
    assert os.path.getsize(filename) > 1024

    # Verify content using standard safetensors
    loaded = load_file(filename)
    assert torch.allclose(loaded["w1"], w1_data)
    assert torch.allclose(loaded["b1"], b1_data)
    assert loaded["w1"].dtype == torch.float32
    assert loaded["b1"].dtype == torch.float16

def test_incremental_writer_dynamic_streaming(source_safetensors_file, temp_output_file):
    source_path, original_tensors, original_metadata = source_safetensors_file
    output_path = temp_output_file

    extra_shape = (5,)
    extra_data = torch.ones(extra_shape, dtype=torch.float32)

    with UnifiedSafetensorsLoader(source_path, low_memory=True) as loader:
        writer = IncrementalSafetensorsWriter(output_path, metadata=loader.metadata())

        with writer:
            for key in loader.keys():
                src_t = loader.get_tensor(key)
                writer.write(key, src_t)
                del src_t # explicitly drop ref
                
            writer.write("extra", extra_data)

    # Verify
    loaded = load_file(output_path)
    for key in original_tensors:
        assert torch.allclose(loaded[key], original_tensors[key])
    assert torch.allclose(loaded["extra"], extra_data)

    # Check metadata was cloned
    with UnifiedSafetensorsLoader(output_path, low_memory=True) as verify_loader:
        assert verify_loader.metadata() == original_metadata

def test_incremental_writer_errors(temp_output_file):
    writer = IncrementalSafetensorsWriter(temp_output_file)

    # Error: write without context manager
    with pytest.raises(RuntimeError, match="Must be used within a context manager"):
        writer.write("test", torch.randn(2, 2))
        
    with writer:
        writer.write("test1", torch.randn(2, 2))
        
        # Error: Duplicate key
        with pytest.raises(ValueError, match="already been written"):
            writer.write("test1", torch.randn(2, 2))

def test_incremental_writer_header_overflow(temp_output_file):
    # Purposefully allocate a tiny header block
    writer = IncrementalSafetensorsWriter(temp_output_file, max_header_bytes=16)

    # The exception happens on exit when it tries to finalize the header
    with pytest.raises(RuntimeError, match="exceeded reserved space"):
        with writer:
            writer.write("a_very_long_tensor_name_that_exceeds_16_bytes", torch.randn(2, 2))

def test_incremental_writer_large_data_sim(temp_output_file):
    # Test with multiple small tensors to ensure async ordering and semaphore work
    filename = temp_output_file

    num_tensors = 20
    shape = (100, 100)
    data_list = []

    for i in range(num_tensors):
        data_list.append(torch.randn(shape) * i)

    writer = IncrementalSafetensorsWriter(filename, max_workers=2)

    with writer:
        for i in range(num_tensors):
            writer.write(f"tensor_{i}", data_list[i])

    loaded = load_file(filename)
    for i in range(num_tensors):
        assert torch.allclose(loaded[f"tensor_{i}"], data_list[i])