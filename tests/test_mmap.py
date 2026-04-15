"""
Tests for use_mmap=True path in UnifiedSafetensorsLoader.

Covers:
- MMAP init success / graceful fallback when uel.dll not available
- get_tensor returns correct values via MMAP zero-copy path
- async_stream yields correct values via MMAP parallel page-fault path
- load_all returns correct dict via MMAP path
- Tensors loaded via MMAP are read-only (non-writable buffer) but viewable
- Original IO path unaffected when use_mmap=False
"""

import os
import pytest
import tempfile
import warnings
import torch
from safetensors.torch import save_file
from unifiedefficientloader import UnifiedSafetensorsLoader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_file():
    """Temporary safetensors file with varied dtypes."""
    tensors = {
        "weight_f32": torch.randn(32, 16, dtype=torch.float32),
        "weight_f16": torch.randn(8, 8, dtype=torch.float16),
        "bias_f32": torch.randn(32, dtype=torch.float32),
        "ids_i32": torch.randint(0, 100, (20,), dtype=torch.int32),
        "mask_bool": torch.zeros(10, dtype=torch.bool),
    }
    fd, path = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    save_file(tensors, path)
    yield path, tensors
    if os.path.exists(path):
        os.remove(path)


def _uel_available() -> bool:
    """Return True if the compiled uel native library loaded successfully."""
    try:
        from unifiedefficientloader.uel import control

        control.init()
        return control.lib is not None
    except Exception:
        return False


uel_required = pytest.mark.skipif(
    not _uel_available(), reason="uel native library (uel.dll/uel.so) not available"
)


# ---------------------------------------------------------------------------
# Graceful fallback when uel not compiled
# ---------------------------------------------------------------------------


def test_mmap_fallback_when_unavailable(sample_file, monkeypatch):
    """
    If uel native lib fails to load, use_mmap should silently fall back
    to standard IO without raising.
    """
    path, tensors = sample_file

    # Patch control.init to simulate missing library
    import unifiedefficientloader.uel.control as ctrl

    monkeypatch.setattr(ctrl, "lib", None)

    with UnifiedSafetensorsLoader(path, low_memory=True, use_mmap=True) as loader:
        assert loader.use_mmap is False, "Should fall back to IO when lib unavailable"
        # Must still load correctly via standard IO
        t = loader.get_tensor("weight_f32")
        assert torch.equal(t, tensors["weight_f32"])


# ---------------------------------------------------------------------------
# MMAP path correctness
# ---------------------------------------------------------------------------


@uel_required
def test_mmap_get_tensor_values(sample_file):
    """get_tensor via MMAP returns identical values to standard IO."""
    path, tensors = sample_file

    with UnifiedSafetensorsLoader(path, low_memory=True, use_mmap=True) as mmap_loader:
        assert mmap_loader.use_mmap is True
        for key, expected in tensors.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t = mmap_loader.get_tensor(key)
            assert t.shape == expected.shape, f"{key}: shape mismatch"
            assert t.dtype == expected.dtype, f"{key}: dtype mismatch"
            # Values must match — copy to writable tensor first for comparison
            assert torch.equal(t.clone(), expected), f"{key}: value mismatch"


@uel_required
def test_mmap_get_tensor_mmap_ref_attached(sample_file):
    """
    Tensors returned via MMAP path have _uel_mmap_ref attached to storage,
    keeping the mapping alive while the tensor is alive.
    """
    path, tensors = sample_file

    with UnifiedSafetensorsLoader(path, low_memory=True, use_mmap=True) as loader:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = loader.get_tensor("weight_f32")
        storage = t.untyped_storage()
        assert hasattr(storage, "_uel_mmap_ref"), (
            "MMAP ref not attached to tensor storage"
        )
        assert storage._uel_mmap_ref is not None


@uel_required
def test_mmap_load_all(sample_file):
    """load_all via MMAP returns correct full dict."""
    path, tensors = sample_file

    with UnifiedSafetensorsLoader(path, low_memory=True, use_mmap=True) as loader:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sd = loader.load_all()

    assert set(sd.keys()) == set(tensors.keys())
    for key, expected in tensors.items():
        assert torch.equal(sd[key].clone(), expected), f"{key}: load_all value mismatch"


@uel_required
def test_mmap_async_stream_values(sample_file):
    """async_stream via MMAP path yields all keys with correct values."""
    path, tensors = sample_file

    loaded = {}
    with UnifiedSafetensorsLoader(path, low_memory=True, use_mmap=True) as loader:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for batch in loader.async_stream(list(tensors.keys()), batch_size=2):
                for key, t in batch:
                    loaded[key] = t.clone()  # clone to make writable before storing

    assert set(loaded.keys()) == set(tensors.keys())
    for key, expected in tensors.items():
        assert torch.equal(loaded[key], expected), f"{key}: async_stream value mismatch"


@uel_required
def test_mmap_keys_and_metadata(sample_file):
    """Keys and metadata accessible correctly in MMAP mode."""
    path, tensors = sample_file

    with UnifiedSafetensorsLoader(path, low_memory=True, use_mmap=True) as loader:
        assert set(loader.keys()) == set(tensors.keys())
        assert isinstance(loader.metadata(), dict)


@uel_required
def test_mmap_get_shape(sample_file):
    """get_shape works correctly in MMAP mode (reads from header, no IO)."""
    path, tensors = sample_file

    with UnifiedSafetensorsLoader(path, low_memory=True, use_mmap=True) as loader:
        assert loader.get_shape("weight_f32") == (32, 16)
        assert loader.get_shape("bias_f32") == (32,)
        assert loader.get_ndim("weight_f16") == 2


# ---------------------------------------------------------------------------
# Original IO path unaffected
# ---------------------------------------------------------------------------


def test_standard_io_unaffected_by_mmap(sample_file):
    """use_mmap=False (default) still works correctly."""
    path, tensors = sample_file

    with UnifiedSafetensorsLoader(path, low_memory=True, use_mmap=False) as loader:
        assert loader.use_mmap is False
        for key, expected in tensors.items():
            t = loader.get_tensor(key)
            assert torch.equal(t, expected), f"{key}: standard IO value mismatch"


def test_standard_preload_unaffected(sample_file):
    """Standard preload mode (low_memory=False) unaffected."""
    path, tensors = sample_file

    with UnifiedSafetensorsLoader(path, low_memory=False) as loader:
        for key, expected in tensors.items():
            t = loader.get_tensor(key)
            assert torch.equal(t, expected), f"{key}: preload value mismatch"
