import torch
from typing import Dict, Tuple, Optional, List

try:
    from . import p2p
except ImportError:
    import p2p

class P2PTensor(torch.Tensor):
    """
    ipc_name = "ipc_name"
    """
    def __new__(cls, data, *args, **kwargs):
        ipc_name = kwargs.pop("ipc_name", None)
        obj = torch.as_tensor(data, *args, **kwargs).as_subclass(cls)
        if not obj.is_contiguous():
            raise ValueError("Tensor must be contiguous")
        ptr = obj.data_ptr()
        size = obj.numel() * obj.element_size()
        obj._ipc_name = ipc_name

        assert ipc_name != "", "ipc_name must not be empty"
        if ipc_name:
            ok = p2p.reg_ipc_with_name(ptr, size, ipc_name)
            if not ok:
                raise RuntimeError(f"Failed to register IPC handle for {ipc_name}")
        return obj

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __del__(self):
        ipc_name = getattr(self, "_ipc_name", None)
        assert ipc_name != "", "ipc_name must not be empty"
        if ipc_name:
            p2p.dereg_ipc_with_name(ipc_name)
            # # we don't need care this
            # ok = p2p.dereg_ipc_with_name(ipc_name)
            # if not ok:
            #     print(f"[P2PTensor] Warning: failed to dereg {ipc_name}")

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if not all(issubclass(t, cls) for t in types):
            return NotImplemented
        return super().__torch_function__(func, types, args, kwargs)
