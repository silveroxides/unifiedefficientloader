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
    """Create a source safetensors file for template testing."""
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

    writer = IncrementalSafetensorsWriter(filename, metadata=metadata)

    # Register tensors
    w1_shape = (10, 20)
    b1_shape = (10,)
    writer.register_tensor("w1", w1_shape, torch.float32)
    writer.register_tensor("b1", b1_shape, torch.float16)

    # Preallocate
    writer.preallocate()
    assert os.path.exists(filename)

    # Check initial size (8 bytes header size + header + data)
    # Data size: (10*20*4) + (10*2) = 800 + 20 = 820
    # Header size is dynamic but file size should be > 820
    initial_size = os.path.getsize(filename)
    assert initial_size > 820

    # Write data
    w1_data = torch.randn(w1_shape, dtype=torch.float32)
    b1_data = torch.randn(b1_shape, dtype=torch.float16)

    with writer:
        writer.write_tensor("w1", w1_data)
        writer.write_tensor("b1", b1_data)

    # Verify content using standard safetensors
    loaded = load_file(filename)
    assert torch.allclose(loaded["w1"], w1_data)
    assert torch.allclose(loaded["b1"], b1_data)
    assert loaded["w1"].dtype == torch.float32
    assert loaded["b1"].dtype == torch.float16

def test_incremental_writer_template_orchestration(source_safetensors_file, temp_output_file):
    source_path, original_tensors, original_metadata = source_safetensors_file
    output_path = temp_output_file

    with UnifiedSafetensorsLoader(source_path) as loader:
        writer = IncrementalSafetensorsWriter(output_path, metadata=loader.metadata())

        # Manual orchestration instead of internal template
        for key in loader.keys():
            writer.register_tensor(key, loader.get_shape(key), loader.get_dtype(key))

        # Test registering an additional tensor after template
        extra_shape = (5,)
        writer.register_tensor("extra", extra_shape, torch.float32)

        writer.preallocate()

        extra_data = torch.ones(extra_shape, dtype=torch.float32)

        with writer:
            for key in loader.keys():
                writer.write_tensor(key, loader.get_tensor(key))
            writer.write_tensor("extra", extra_data)

    # Verify
    loaded = load_file(output_path)
    for key in original_tensors:
        assert torch.allclose(loaded[key], original_tensors[key])
    assert torch.allclose(loaded["extra"], extra_data)

    # Check metadata was cloned
    with UnifiedSafetensorsLoader(output_path) as verify_loader:
        assert verify_loader.metadata() == original_metadata

def test_incremental_writer_errors(temp_output_file):
    writer = IncrementalSafetensorsWriter(temp_output_file)
    writer.register_tensor("test", (2, 2), torch.float32)
    writer.preallocate()

    # Error: registering after preallocate
    with pytest.raises(RuntimeError, match="Cannot register tensors after preallocate"):
        writer.register_tensor("fail", (1,), torch.float32)

    # Error: write_tensor without context manager
    with pytest.raises(RuntimeError, match="Must be used within a context manager"):
        writer.write_tensor("test", torch.randn(2, 2))

    with writer:
        # Error: unknown tensor
        with pytest.raises(KeyError):
            writer.write_tensor("unknown", torch.randn(2, 2))

        # Error: shape mismatch
        with pytest.raises(ValueError, match="shape mismatch"):
            writer.write_tensor("test", torch.randn(3, 3))

        # Error: dtype mismatch
        with pytest.raises(ValueError, match="dtype mismatch"):
            writer.write_tensor("test", torch.randn(2, 2, dtype=torch.float16))

def test_incremental_writer_large_data_sim(temp_output_file):
    # Test with multiple small tensors to ensure async ordering and semaphore work
    filename = temp_output_file
    writer = IncrementalSafetensorsWriter(filename, max_workers=2)

    num_tensors = 20
    shape = (100, 100)
    data_list = []

    for i in range(num_tensors):
        name = f"tensor_{i}"
        writer.register_tensor(name, shape, torch.float32)
        data_list.append(torch.randn(shape) * i)

    writer.preallocate()

    with writer:
        for i in range(num_tensors):
            writer.write_tensor(f"tensor_{i}", data_list[i])

    loaded = load_file(filename)
    for i in range(num_tensors):
        assert torch.allclose(loaded[f"tensor_{i}"], data_list[i])
