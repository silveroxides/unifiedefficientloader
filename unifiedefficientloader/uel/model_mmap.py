import ctypes
import os

from . import control


def _get_lib():
    """Return the loaded ctypes library, calling init() if not yet done."""
    if control.lib is None:
        control.init()
    return control.lib


def _setup_argtypes():
    lib = control.lib
    if lib is None:
        return
    lib.model_mmap_allocate.argtypes = [ctypes.c_char_p]
    lib.model_mmap_allocate.restype = ctypes.c_void_p

    lib.model_mmap_get.argtypes = [ctypes.c_void_p]
    lib.model_mmap_get.restype = ctypes.c_void_p

    lib.model_mmap_bounce.argtypes = [ctypes.c_void_p]
    lib.model_mmap_bounce.restype = ctypes.c_bool

    lib.model_mmap_deallocate.argtypes = [ctypes.c_void_p]


class ModelMMAP:
    def __init__(self, filepath):
        lib = _get_lib()
        if lib is None:
            raise RuntimeError("unifiedefficientloader-uel is not initialized")

        _setup_argtypes()

        normalized_path = os.fspath(filepath)
        if isinstance(normalized_path, bytes):
            filepath_bytes = normalized_path
        elif os.name == "nt":
            filepath_bytes = normalized_path.encode("utf-8")
        else:
            filepath_bytes = os.fsencode(normalized_path)

        self.state = lib.model_mmap_allocate(filepath_bytes)
        if not self.state:
            raise RuntimeError(f"ModelMMAP allocation failed for {filepath}")

    def get(self):
        return control.lib.model_mmap_get(self.state)

    def bounce(self):
        return bool(control.lib.model_mmap_bounce(self.state))

    def __del__(self):
        state = getattr(self, "state", None)
        if state and control.lib is not None:
            control.lib.model_mmap_deallocate(state)
