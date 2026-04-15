import ctypes

from . import control

lib = control.lib

if lib is not None:
    lib.hostbuf_allocate.argtypes = [ctypes.c_uint64]
    lib.hostbuf_allocate.restype = ctypes.c_void_p

    lib.hostbuf_free.argtypes = [ctypes.c_void_p]


class HostBuffer:
    def __init__(self, size):
        self.size = int(size)
        self._ptr = lib.hostbuf_allocate(self.size)
        if not self._ptr:
            raise RuntimeError("CUDA host buffer allocation failed")

    def get_raw_address(self):
        return int(self._ptr)

    def __del__(self):
        ptr = getattr(self, "_ptr", None)
        if ptr:
            lib.hostbuf_free(ptr)
            self._ptr = None
