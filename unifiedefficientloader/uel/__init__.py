"""
UEL compatibility layer — delegates to comfy_aimdo package.

Submodules are loaded lazily via __getattr__ to avoid capturing
control.lib as None before init() is called.

Individual .py stubs handle sys.modules registration so that direct
imports like `from unifiedefficientloader.uel.model_mmap import ModelMMAP`
also work correctly.
"""
import importlib as _importlib

# Eagerly import control (the wrapper patches init for lib propagation)
from . import control  # noqa: F401  — triggers control.py stub

_SUBMODULE_MAP = {
    "host_buffer": "comfy_aimdo.host_buffer",
    "model_vbar": "comfy_aimdo.model_vbar",
    "torch": "comfy_aimdo.torch",
    "vram_buffer": "comfy_aimdo.vram_buffer",
}


def __getattr__(name):
    if name in _SUBMODULE_MAP:
        mod = _importlib.import_module(_SUBMODULE_MAP[name])
        globals()[name] = mod
        return mod
    if name == "model_mmap":
        from . import model_mmap as _mm
        globals()["model_mmap"] = _mm
        return _mm
    if name == "ModelMMAP":
        from .model_mmap import ModelMMAP
        globals()["ModelMMAP"] = ModelMMAP
        return ModelMMAP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "control",
    "host_buffer",
    "model_mmap",
    "model_vbar",
    "torch",
    "vram_buffer",
    "ModelMMAP",
]
