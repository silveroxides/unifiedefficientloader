"""
Diagnostics for MMAP initialization and comfy-aimdo availability.

Run with: pytest tests/test_mmap_diagnostics.py -v -s
Or use check_mmap_available() in other tests to detect capabilities.
"""

import pytest
import tempfile
import os
import sys

try:
    import torch
    from safetensors.torch import save_file
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def check_mmap_available() -> tuple[bool, str]:
    """
    Check if MMAP is available and initialized.
    
    Returns:
        (is_available, status_message)
    """
    try:
        import comfy_aimdo
    except ImportError:
        return False, "comfy_aimdo package not installed"
    
    try:
        from unifiedefficientloader.uel import control
        
        if control.lib is None:
            init_result = control.init()
            if not init_result:
                return False, "control.init() returned False"
        
        # Try importing ModelMMAP
        from unifiedefficientloader.uel.model_mmap import ModelMMAP
        
        return True, "MMAP available"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


@pytest.fixture
def mmap_status():
    """Get current MMAP availability status."""
    available, message = check_mmap_available()
    return {"available": available, "message": message}


def test_mmap_availability(mmap_status):
    """
    Diagnostic test: Report MMAP availability status.
    
    This test always passes but provides visibility into whether MMAP
    is available in the current environment.
    """
    print(f"\nMMAP Status: {mmap_status['available']}")
    print(f"Details: {mmap_status['message']}")
    
    if not mmap_status["available"]:
        pytest.skip(f"MMAP not available: {mmap_status['message']}")


@pytest.mark.skipif(
    not HAS_TORCH, reason="Requires torch and safetensors"
)
def test_mmap_initialization_diagnostics():
    """
    Diagnostic test: Verify full MMAP initialization pipeline.
    
    Tests each step and reports where/why it might fail.
    """
    import logging
    from unifiedefficientloader import logging_utils
    
    # Enable verbose logging for diagnostics
    logging_utils.setup_logging("VERBOSE")
    
    print("\n" + "=" * 70)
    print("MMAP INITIALIZATION DIAGNOSTICS")
    print("=" * 70)
    
    # Step 1: Check comfy-aimdo
    print("\n[1] Checking comfy-aimdo package...")
    try:
        import comfy_aimdo
        print("    ✓ comfy_aimdo imported")
    except ImportError as e:
        print(f"    ✗ FAILED: {e}")
        pytest.skip("comfy_aimdo not installed")
    
    # Step 2: Import and initialize control
    print("\n[2] Initializing control module...")
    try:
        from unifiedefficientloader.uel import control
        print(f"    ✓ control.lib before init: {control.lib}")
        
        result = control.init()
        print(f"    ✓ control.init() returned: {result}")
        print(f"    ✓ control.lib after init: {control.lib}")
        
        if control.lib is None:
            print("    ✗ FAILED: control.lib is still None")
            pytest.skip("control.init() did not load library")
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        pytest.skip(f"control initialization failed: {e}")
    
    # Step 3: Import ModelMMAP
    print("\n[3] Importing ModelMMAP...")
    try:
        from unifiedefficientloader.uel.model_mmap import ModelMMAP
        print("    ✓ ModelMMAP imported")
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        pytest.skip(f"ModelMMAP import failed: {e}")
    
    # Step 4: Try creating MMAP with test file
    print("\n[4] Creating test MMAP...")
    try:
        tensors = {"test_tensor": torch.randn(10, 10, dtype=torch.float32)}
        fd, test_file = tempfile.mkstemp(suffix=".safetensors")
        os.close(fd)
        save_file(tensors, test_file)
        print(f"    ✓ Created test file: {test_file}")
        
        try:
            mmap = ModelMMAP(test_file)
            print(f"    ✓ ModelMMAP created successfully")
            print(f"    ✓ mmap.state: {mmap.state}")
            print(f"    ✓ mmap.get(): {mmap.get()}")
            mmap = None  # Release
            print("    ✓ MMAP released")
        finally:
            # Windows locks MMAP files, so we can't always delete immediately
            try:
                os.remove(test_file)
            except PermissionError:
                print(f"    ⚠ Could not delete test file (locked by MMAP): {test_file}")
    
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"MMAP creation failed: {e}")
    
    # Step 5: Try UnifiedSafetensorsLoader with MMAP
    print("\n[5] Testing UnifiedSafetensorsLoader with use_mmap=True...")
    try:
        from unifiedefficientloader import UnifiedSafetensorsLoader
        
        tensors = {
            "weight": torch.randn(5, 5, dtype=torch.float32),
            "bias": torch.randn(5, dtype=torch.float32),
        }
        fd, test_file = tempfile.mkstemp(suffix=".safetensors")
        os.close(fd)
        save_file(tensors, test_file)
        print(f"    ✓ Created test file: {test_file}")
        
        try:
            with UnifiedSafetensorsLoader(test_file, low_memory=True, use_mmap=True) as loader:
                print(f"    ✓ Loader created")
                print(f"    ✓ loader.use_mmap: {loader.use_mmap}")
                
                if not loader.use_mmap:
                    print("    ✗ MMAP not active (fell back to standard IO)")
                    pytest.fail("MMAP initialization failed in UnifiedSafetensorsLoader")
                
                # Load a tensor via MMAP
                tensor = loader.get_tensor("weight")
                print(f"    ✓ Loaded tensor via MMAP: {tensor.shape} {tensor.dtype}")
                
                # Verify _uel_mmap_ref is attached
                storage = tensor.untyped_storage()
                if hasattr(storage, "_uel_mmap_ref"):
                    print(f"    ✓ MMAP reference attached to tensor storage")
                else:
                    print(f"    ⚠ MMAP reference NOT attached (tensor may not keep mapping alive)")
        finally:
            try:
                os.remove(test_file)
            except PermissionError:
                print(f"    ⚠ Could not delete test file: {test_file}")
    
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"UnifiedSafetensorsLoader MMAP test failed: {e}")
    
    print("\n" + "=" * 70)
    print("✓ ALL DIAGNOSTICS PASSED - MMAP IS WORKING")
    print("=" * 70)


@pytest.mark.skipif(
    not HAS_TORCH, reason="Requires torch and safetensors"
)
def test_mmap_not_available_graceful_fallback(monkeypatch):
    """
    Test that when MMAP is not available, loader falls back gracefully.
    """
    from unifiedefficientloader import UnifiedSafetensorsLoader
    from unifiedefficientloader.uel import control
    
    # Patch control to simulate missing library
    monkeypatch.setattr(control, "lib", None)
    monkeypatch.setattr(control, "init", lambda: False)
    
    # Create test file
    tensors = {"weight": torch.randn(5, 5)}
    fd, test_file = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    save_file(tensors, test_file)
    
    try:
        # Try with MMAP=True, should fall back silently
        with UnifiedSafetensorsLoader(test_file, low_memory=True, use_mmap=True) as loader:
            assert loader.use_mmap is False, "Should have fallen back when MMAP unavailable"
            
            # Verify it still loads correctly via standard IO
            tensor = loader.get_tensor("weight")
            assert tensor.shape == (5, 5)
            assert torch.equal(tensor, tensors["weight"])
            
            print("\n✓ Graceful fallback works: Loader reverted to standard IO")
    finally:
        try:
            os.remove(test_file)
        except PermissionError:
            pass
