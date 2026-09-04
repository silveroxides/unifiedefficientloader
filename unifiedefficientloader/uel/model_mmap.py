"""
ModelMMAP wrapper that accesses comfy_aimdo.control.lib dynamically.

Unlike comfy_aimdo.model_mmap which captures `lib = control.lib` at import time
(breaking if init() hasn't been called yet or lib is temporarily None),
this wrapper always accesses control.lib at call time.
"""
import ctypes
import os

import comfy_aimdo.control as _ctrl

_argtypes_set = False


def _setup_argtypes(lib):
    global _argtypes_set
    if _argtypes_set:
        return
    lib.model_mmap_allocate.argtypes = [ctypes.c_char_p]
    lib.model_mmap_allocate.restype = ctypes.c_void_p

    lib.model_mmap_get.argtypes = [ctypes.c_void_p]
    lib.model_mmap_get.restype = ctypes.c_void_p

    lib.model_mmap_get_file_handle.argtypes = [ctypes.c_void_p]
    lib.model_mmap_get_file_handle.restype = ctypes.c_uint64

    lib.model_mmap_bounce.argtypes = [ctypes.c_void_p]
    lib.model_mmap_bounce.restype = ctypes.c_bool

    lib.model_mmap_deallocate.argtypes = [ctypes.c_void_p]
    _argtypes_set = True


class ModelMMAP:
    def __init__(self, filepath):
        lib = _ctrl.lib
        if lib is None:
            raise RuntimeError("comfy-aimdo is not initialized")

        _setup_argtypes(lib)

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
        return _ctrl.lib.model_mmap_get(self.state)

    def get_file_handle(self):
        return int(_ctrl.lib.model_mmap_get_file_handle(self.state))

    def bounce(self):
        return bool(_ctrl.lib.model_mmap_bounce(self.state))

    def __del__(self):
        state = getattr(self, "state", None)
        if state and _ctrl.lib is not None:
            _ctrl.lib.model_mmap_deallocate(state)
