import os
import tempfile
import pytest

try:
    import torch
    from safetensors.torch import save_file

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from unifiedefficientloader import MemoryEfficientSafeOpen


@pytest.fixture
def sample_safetensors():
    if not HAS_TORCH:
        pytest.skip("Requires torch and safetensors")

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        path = f.name

    tensors = {
        "weight1": torch.randn(10, 10),
        "weight2": torch.randn(20, 20),
        "bias": torch.zeros(10),
    }
    save_file(tensors, path)

    yield path, tensors

    if os.path.exists(path):
        os.remove(path)


@pytest.mark.skipif(
    not HAS_TORCH or not torch.cuda.is_available(), reason="Requires CUDA"
)
def test_direct_gpu_streaming(sample_safetensors):
    path, original_tensors = sample_safetensors

    loader = MemoryEfficientSafeOpen(path, low_memory=True, direct_gpu=True)

    # Test load_all which uses async_stream under the hood
    loaded_tensors = loader.load_all()

    for key, orig_tensor in original_tensors.items():
        assert key in loaded_tensors
        loaded_tensor = loaded_tensors[key]

        # Verify it's on GPU
        assert loaded_tensor.device.type == "cuda"

        # Verify data matches
        torch.testing.assert_close(loaded_tensor.cpu(), orig_tensor)

    loader.close()


@pytest.mark.skipif(
    not HAS_TORCH or not torch.cuda.is_available(), reason="Requires CUDA"
)
def test_direct_gpu_async_stream(sample_safetensors):
    path, original_tensors = sample_safetensors

    loader = MemoryEfficientSafeOpen(path, low_memory=True, direct_gpu=True)

    stream = loader.async_stream(
        keys=list(original_tensors.keys()),
        batch_size=2,
        prefetch_batches=1,
    )

    loaded_count = 0
    for batch in stream:
        for key, tensor in batch:
            assert tensor.device.type == "cuda"
            torch.testing.assert_close(tensor.cpu(), original_tensors[key])
            loaded_count += 1

    assert loaded_count == len(original_tensors)
    loader.close()


@pytest.mark.skipif(not HAS_TORCH, reason="Requires torch")
def test_direct_gpu_fallback_no_cuda(sample_safetensors, monkeypatch):
    # Force cuda to be unavailable
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    path, original_tensors = sample_safetensors

    # Should fallback to CPU silently
    loader = MemoryEfficientSafeOpen(path, low_memory=True, direct_gpu=True)

    loaded_tensors = loader.load_all()

    for key, orig_tensor in original_tensors.items():
        loaded_tensor = loaded_tensors[key]
        assert loaded_tensor.device.type == "cpu"
        torch.testing.assert_close(loaded_tensor, orig_tensor)

    loader.close()
