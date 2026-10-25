"""ctypes runtime for native tests, selected by the loaded P2P binary."""

import ctypes
import os


def backend_info():
    from uccl import p2p

    return p2p.backend_info()


class Rt:
    def __init__(self):
        info = backend_info()
        self.kind = info["device"]
        # Optional library path override for test installations; the compiled
        # binary remains the authority for API names and memcpy enum values.
        path = os.environ.get("UCCL_GPU_RT_LIB") or info["runtime_library"]
        self.lib = ctypes.CDLL(path)
        prefix = info["runtime_prefix"]
        self._h2d, self._d2h = info["memcpy_h2d"], info["memcpy_d2h"]
        self._malloc = getattr(self.lib, prefix + "Malloc")
        self._free = getattr(self.lib, prefix + "Free")
        self._memcpy = getattr(self.lib, prefix + "Memcpy")
        self._set = getattr(self.lib, prefix + "SetDevice")
        self._malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self._free.argtypes = [ctypes.c_void_p]
        self._memcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self._set.argtypes = [ctypes.c_int]
        for fn in (self._malloc, self._free, self._memcpy, self._set):
            fn.restype = ctypes.c_int
        if info["runtime_init"]:
            init = getattr(self.lib, info["runtime_init"])
            init.argtypes = [ctypes.c_uint]
            init.restype = ctypes.c_int
            self._chk(init(0), info["runtime_init"])

    @staticmethod
    def _chk(err, name):
        if err != 0:
            raise RuntimeError(f"{name} failed err={err}")

    def set_device(self, idx):
        self._chk(self._set(idx), f"set_device({idx})")

    def malloc(self, nbytes):
        ptr = ctypes.c_void_p()
        self._chk(self._malloc(ctypes.byref(ptr), nbytes), "malloc")
        return ptr

    def free(self, ptr):
        self._chk(self._free(ptr), "free")

    def memcpy_h2d(self, dst, host, nbytes):
        self._chk(self._memcpy(dst, host, nbytes, self._h2d), "memcpy H2D")

    def memcpy_d2h(self, host, src, nbytes):
        self._chk(self._memcpy(host, src, nbytes, self._d2h), "memcpy D2H")
