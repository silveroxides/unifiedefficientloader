import ctypes

from . import control


def _bind_argtypes(lib):
    """Bind hostbuf argtypes against a live lib handle."""
    lib.hostbuf_allocate.argtypes = [ctypes.c_uint64]
    lib.hostbuf_allocate.restype = ctypes.c_void_p
    lib.hostbuf_free.argtypes = [ctypes.c_void_p]


class HostBuffer:
    def __init__(self, size):
        lib = control.lib
        if lib is None:
            raise RuntimeError("uel lib not loaded — call control.init() first")
        _bind_argtypes(lib)
        self._lib = lib
        self.size = int(size)
        self._ptr = lib.hostbuf_allocate(self.size)
        if not self._ptr:
            raise RuntimeError("CUDA host buffer allocation failed")

    def get_raw_address(self):
        return int(self._ptr)

    def __del__(self):
        ptr = getattr(self, "_ptr", None)
        lib = getattr(self, "_lib", None)
        if ptr and lib is not None:
            lib.hostbuf_free(ptr)
            self._ptr = None
