# Preload torch's C library with RTLD_GLOBAL *before* importing torch so its
# symbols are visible when our ep .so loads (THPVariable_Wrap, etc.).
import ctypes
import importlib.util
import os
import sys

_torch_spec = importlib.util.find_spec("torch")
if _torch_spec is None or not _torch_spec.submodule_search_locations:
    raise ImportError("torch package not found")
_torch_dir = _torch_spec.submodule_search_locations[0]
_torch_python = os.path.join(_torch_dir, "lib", "libtorch_python.so")
if os.path.isfile(_torch_python):
    try:
        ctypes.CDLL(_torch_python, mode=getattr(os, "RTLD_GLOBAL", 0x00100))
    except OSError:
        pass
del _torch_spec, _torch_dir, _torch_python

import torch  # noqa: F401

_ep_so = os.path.join(os.path.dirname(__file__), "_ep_native.abi3.so")
if not os.path.isfile(_ep_so):
    raise ImportError(f"uccl ep extension not found: {_ep_so}")

_spec = importlib.util.spec_from_file_location("uccl._ep_native", _ep_so)
_native = importlib.util.module_from_spec(_spec)
sys.modules["uccl._ep_native"] = _native
_spec.loader.exec_module(_native)

# Re-export the native module's public contents so "from uccl import ep" and ep.xxx work.
for _name in dir(_native):
    if not _name.startswith("_"):
        setattr(sys.modules[__name__], _name, getattr(_native, _name))

del _ep_so, _spec, _native, _name
