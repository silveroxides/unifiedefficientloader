import importlib.util

def check_dependencies(*packages):
    """
    Check if required packages are installed.
    Throws a descriptive error if not.
    """
    missing = []
    for pkg in packages:
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)
    
    if missing:
        missing_str = ", ".join(missing)
        raise ImportError(
            f"Missing required packages for unifiedefficientloader: {missing_str}. "
            f"Please install them using: pip install {missing_str}"
        )

# Pre-check torch as it is the foundation of most of these tools
check_dependencies("torch")

from .memory_efficient_loader import UnifiedSafetensorsLoader, MemoryEfficientSafeOpen
from .tensor_utils import dict_to_tensor, tensor_to_dict
from .pinned_transfer import transfer_to_gpu_pinned, set_verbose, get_pinned_transfer_stats, reset_pinned_transfer_stats
from .logging_utils import (
    setup_logging,
    MINIMAL_LEVEL, 
    NORMAL_LEVEL, 
    VERBOSE_LEVEL, 
    DEBUG_LEVEL,
    debug,
    verbose,
    normal,
    info,
    minimal,
    warning,
    error
)

__all__ = [
    "UnifiedSafetensorsLoader",
    "MemoryEfficientSafeOpen",
    "dict_to_tensor",
    "tensor_to_dict",
    "transfer_to_gpu_pinned",
    "set_verbose",
    "get_pinned_transfer_stats",
    "reset_pinned_transfer_stats",
    "setup_logging",
    "MINIMAL_LEVEL",
    "NORMAL_LEVEL",
    "VERBOSE_LEVEL",
    "DEBUG_LEVEL",
    "debug",
    "verbose",
    "normal",
    "info",
    "minimal",
    "warning",
    "error",
]
