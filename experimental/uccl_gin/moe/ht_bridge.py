"""Forward inference through the frozen NCCL-EP tensor-array ABI."""
import ctypes
from dataclasses import dataclass
from pathlib import Path

import torch
from expert_mma import KernelResources


class ScheduleResources(ctypes.Structure):
    _fields_ = [('dispatch', KernelResources), ('combine', KernelResources)]


@dataclass
class Dispatched:
    hidden: torch.Tensor
    weights: torch.Tensor
    experts: torch.Tensor


class HtBridge:
    def __init__(self, library: Path, stream: torch.cuda.Stream, maximum: int,
                 hidden: int = 2048, experts: int = 128, rank: int = 0, world: int = 1,
                 local_ranks: int = 1, unique: bytes | None = None, stages: int = 0, blocks: int = 16):
        self.library = ctypes.CDLL(str(library.resolve()))
        self.library.ht_bridge_abi_version.argtypes = []
        self.library.ht_bridge_abi_version.restype = ctypes.c_int
        assert self.library.ht_bridge_abi_version() == 2
        self.library.ht_bridge_active_allocations.argtypes = []
        self.library.ht_bridge_active_allocations.restype = ctypes.c_uint64
        self.library.ht_bridge_unique_id.argtypes = [ctypes.c_void_p]
        self.library.ht_bridge_unique_id.restype = ctypes.c_uint
        self.library.ht_bridge_create.argtypes = [ctypes.c_int] * 9 + [ctypes.c_void_p] * 2
        self.library.ht_bridge_create.restype = ctypes.c_void_p
        self.library.ht_bridge_destroy.argtypes = [ctypes.c_void_p]
        self.library.ht_bridge_destroy.restype = None
        self.library.ht_bridge_prepare.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int] * 2
        self.library.ht_bridge_prepare.restype = ctypes.c_uint
        self.library.ht_bridge_dispatch.argtypes = [ctypes.c_void_p] * 6
        self.library.ht_bridge_dispatch.restype = None
        self.library.ht_bridge_combine.argtypes = [ctypes.c_void_p] * 3
        self.library.ht_bridge_combine.restype = None
        self.library.ht_bridge_resources.argtypes = [ctypes.c_void_p, ctypes.POINTER(ScheduleResources)]
        self.library.ht_bridge_resources.restype = None
        self.library.ht_bridge_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
        self.library.ht_bridge_stats.restype = None
        self.library.ht_bridge_versions.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.library.ht_bridge_versions.restype = None
        identifier = ctypes.create_string_buffer(128)
        if unique is None:
            assert world == 1
            assert self.library.ht_bridge_unique_id(identifier) == 128
        else:
            assert len(unique) == 128
            ctypes.memmove(identifier, unique, 128)
        self.stream = stream
        self.hidden = hidden
        self.rank, self.world, self.experts = rank, world, experts
        self.tokens = 0
        self.routing: torch.Tensor | None = None
        self.context = self.library.ht_bridge_create(stream.device.index, rank, world, local_ranks,
                                                     hidden, experts, maximum, stages, blocks,
                                                     identifier, stream.cuda_stream)
        assert self.context

    def dispatch(self, hidden: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> Dispatched:
        assert self.routing is None and torch.cuda.current_stream(hidden.device) == self.stream
        assert hidden.is_cuda and hidden.dtype == torch.bfloat16 and hidden.is_contiguous()
        assert hidden.ndim == 2 and hidden.shape[1] == self.hidden
        assert indices.dtype == torch.int64 and indices.is_contiguous()
        assert weights.dtype == torch.float32 and weights.is_contiguous()
        assert indices.shape == weights.shape and indices.shape[0] == hidden.shape[0]
        assert indices.device == weights.device == hidden.device
        self.tokens, topk = indices.shape
        self.routing = indices
        rows = self.library.ht_bridge_prepare(self.context, indices.data_ptr(), self.tokens, topk)
        result = Dispatched(torch.empty((rows, self.hidden), device=hidden.device, dtype=hidden.dtype),
                            torch.empty((rows, topk), device=hidden.device, dtype=torch.float32),
                            torch.empty((rows, topk), device=hidden.device, dtype=torch.int64))
        self.library.ht_bridge_dispatch(self.context, hidden.data_ptr(), weights.data_ptr(), result.hidden.data_ptr(),
                                        result.weights.data_ptr(), result.experts.data_ptr())
        return result

    def combine(self, expert_output: torch.Tensor) -> torch.Tensor:
        assert self.routing is not None and expert_output.is_contiguous()
        assert expert_output.dtype == torch.bfloat16 and expert_output.shape[1] == self.hidden
        assert torch.cuda.current_stream(expert_output.device) == self.stream
        output = torch.empty((self.tokens, self.hidden), device=expert_output.device, dtype=expert_output.dtype)
        self.library.ht_bridge_combine(self.context, expert_output.data_ptr(), output.data_ptr())
        self.routing = None
        return output

    def resources(self) -> dict[str, dict[str, int]]:
        value = ScheduleResources()
        self.library.ht_bridge_resources(self.context, ctypes.byref(value))
        return {'dispatch': value.dispatch.report(), 'combine': value.combine.report()}

    def stats(self) -> dict[str, int]:
        values = (ctypes.c_uint64 * 6)()
        self.library.ht_bridge_stats(self.context, values)
        return dict(zip(('dispatch_calls', 'combine_calls', 'live_handle', 'dispatched',
                         'callback_active_allocations', 'callback_total_allocations'), values, strict=True))

    def versions(self) -> tuple[int, int]:
        values = (ctypes.c_int * 2)()
        self.library.ht_bridge_versions(values)
        return values[0], values[1]

    def close(self) -> None:
        assert self.routing is None
        self.library.ht_bridge_destroy(self.context)

    def active_allocations(self) -> int:
        return self.library.ht_bridge_active_allocations()
