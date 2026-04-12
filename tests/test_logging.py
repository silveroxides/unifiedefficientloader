import os
import torch
from unifiedefficientloader import (
    UnifiedSafetensorsLoader,
    setup_logging,
    VERBOSE_LEVEL,
    DEBUG_LEVEL
)
from safetensors.torch import save_file

def test_logging():
    # 1. Create a dummy safetensors file
    dummy_file = "test_logging.safetensors"
    tensors = {
        "weight1": torch.randn(10, 10),
        "weight2": torch.randn(5, 5)
    }
    save_file(tensors, dummy_file)

    try:
        # 2. Test NORMAL logging (default)
        print("\n--- Testing NORMAL Logging (Default) ---")
        setup_logging("NORMAL")
        with UnifiedSafetensorsLoader(dummy_file, low_memory=False) as loader:
            _ = loader.get_tensor("weight1")

        # 3. Test VERBOSE logging
        print("\n--- Testing VERBOSE Logging ---")
        setup_logging("VERBOSE")
        with UnifiedSafetensorsLoader(dummy_file, low_memory=True) as loader:
            _ = loader.get_tensor("weight1")
            _ = loader.get_tensor("weight2")

        # 4. Test DEBUG logging (includes function traces)
        print("\n--- Testing DEBUG Logging ---")
        setup_logging("DEBUG")
        with UnifiedSafetensorsLoader(dummy_file, low_memory=True) as loader:
            _ = loader.get_tensor("weight1")

        # 5. Test MINIMAL logging
        print("\n--- Testing MINIMAL Logging ---")
        setup_logging("MINIMAL")
        with UnifiedSafetensorsLoader(dummy_file, low_memory=False) as loader:
            _ = loader.get_tensor("weight1")

    finally:
        if os.path.exists(dummy_file):
            os.remove(dummy_file)

if __name__ == "__main__":
    test_logging()
