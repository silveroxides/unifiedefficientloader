"""
Redirect: unifiedefficientloader.uel.control → comfy_aimdo.control

Wraps init()/init_devices() to:
1. Patch module-level 'lib' references in comfy_aimdo submodules
2. Auto-initialize the default CUDA device for backward compatibility
   (old uel.dll didn't require explicit init_device before hostbuf usage)
"""
import sys
import comfy_aimdo.control as _ctrl

_original_init = _ctrl.init
_original_init_devices = _ctrl.init_devices
_original_init_device = _ctrl.init_device


def _patch_lib_refs():
    """Propagate control.lib to submodules that captured it at import time."""
    for mod_name in ("comfy_aimdo.model_mmap", "comfy_aimdo.host_buffer", "comfy_aimdo.model_vbar", "comfy_aimdo.vram_buffer"):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "lib"):
            mod.lib = _ctrl.lib


def _auto_init_device():
    """Auto-initialize default GPU device for backward compat with old uel behavior."""
    if _ctrl.devctxs:
        return  # Already initialized
    try:
        import torch
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            _original_init_device(device_id)
            _patch_lib_refs()
    except Exception:
        pass


def _patched_init(*args, **kwargs):
    result = _original_init(*args, **kwargs)
    if result and _ctrl.lib is not None:
        _patch_lib_refs()
        _auto_init_device()
    return result


def _patched_init_devices(*args, **kwargs):
    result = _original_init_devices(*args, **kwargs)
    if result and _ctrl.lib is not None:
        _patch_lib_refs()
    return result


_ctrl.init = _patched_init
_ctrl.init_devices = _patched_init_devices

sys.modules[__name__] = _ctrl
