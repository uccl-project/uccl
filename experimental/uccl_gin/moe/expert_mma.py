"""Explicit C ABI for BF16 CUTLASS expert GEMMs; one workspace per stream."""
import ctypes
from pathlib import Path

import torch


class KernelResources(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int) for name in
                ('device', 'sms', 'registers', 'threads', 'blocks', 'resident_blocks')] + [
                    (name, ctypes.c_size_t) for name in
                    ('static_smem', 'dynamic_smem', 'smem_per_sm', 'smem_per_block')]

    def report(self) -> dict[str, int]:
        return {'device': self.device, 'sms': self.sms, 'registers': self.registers,
                'threads': self.threads, 'blocks': self.blocks, 'resident_blocks_api': self.resident_blocks,
                'static_smem': self.static_smem, 'dynamic_smem': self.dynamic_smem,
                'smem_per_sm': self.smem_per_sm, 'smem_per_block': self.smem_per_block}


class ExpertMma:
    def __init__(self, library: Path, ctas: int):
        assert ctas > 0
        self.library = ctypes.CDLL(str(library.resolve()))
        self.library.expert_mma_create.argtypes = []
        self.library.expert_mma_create.restype = ctypes.c_void_p
        self.library.expert_mma_destroy.argtypes = [ctypes.c_void_p]
        self.library.expert_mma_destroy.restype = None
        matrix_args = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4
        self.library.expert_mma_run.argtypes = matrix_args + [ctypes.c_void_p]
        self.library.expert_mma_run.restype = None
        self.library.expert_mma_query.argtypes = matrix_args + [ctypes.POINTER(KernelResources)]
        self.library.expert_mma_query.restype = None
        self.library.expert_mma_swiglu.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        self.library.expert_mma_swiglu.restype = None
        self.stream = torch.cuda.current_stream()
        self.handle = self.library.expert_mma_create()
        self.ctas = ctas
        assert self.handle

    def linear(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        assert value.is_cuda and weight.is_cuda and value.device == weight.device
        assert value.dtype == weight.dtype == torch.bfloat16
        assert value.is_contiguous() and weight.is_contiguous()
        assert value.ndim == weight.ndim == 2 and value.shape[1] == weight.shape[1]
        assert torch.cuda.current_stream(value.device) == self.stream
        assert weight.shape[0] % 8 == weight.shape[1] % 8 == 0
        output = torch.empty((value.shape[0], weight.shape[0]), device=value.device, dtype=value.dtype)
        self.library.expert_mma_run(self.handle, value.data_ptr(), weight.data_ptr(), output.data_ptr(),
                                    value.shape[0], weight.shape[0], value.shape[1], self.ctas, self.stream.cuda_stream)
        return output

    def swiglu(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        assert gate.shape == up.shape and gate.dtype == up.dtype == torch.bfloat16
        assert gate.is_contiguous() and up.is_contiguous() and gate.device == up.device
        assert torch.cuda.current_stream(gate.device) == self.stream
        output = torch.empty_like(gate)
        self.library.expert_mma_swiglu(gate.data_ptr(), up.data_ptr(), output.data_ptr(), gate.numel(),
                                       self.ctas, self.stream.cuda_stream)
        return output

    def query(self, value: torch.Tensor, weight: torch.Tensor) -> dict[str, int]:
        output = torch.empty((value.shape[0], weight.shape[0]), device=value.device, dtype=value.dtype)
        resource = KernelResources()
        self.library.expert_mma_query(self.handle, value.data_ptr(), weight.data_ptr(), output.data_ptr(),
                                      value.shape[0], weight.shape[0], value.shape[1], self.ctas, ctypes.byref(resource))
        return resource.report()

    def close(self) -> None:
        self.stream.synchronize()
        self.library.expert_mma_destroy(self.handle)
