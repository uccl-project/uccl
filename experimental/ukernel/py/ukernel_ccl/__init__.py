"""UKernel CCL — collective communication (allreduce / alltoall / alltoallv).

Python wrapper over the SprayExecutor-based CCL engine. Collectives run
**in place** on CUDA tensors, which must be contiguous (call
``tensor.contiguous()`` yourself first if needed — the engine cannot
silently copy back into a non-contiguous view).
"""

import os
from enum import IntEnum
from typing import Optional

import torch

from ._C import ProcessGroup as _ProcessGroup


class ReduceOp(IntEnum):
    SUM = 1
    PRODUCT = 2
    MAX = 3
    MIN = 4
    BAND = 5


_ARITHMETIC_DTYPES = {
    torch.int8, torch.int32, torch.int64,
    torch.float16, torch.float32, torch.float64, torch.bfloat16,
}


def _validate_dtype(tensor: torch.Tensor, what: str) -> None:
    if tensor.dtype not in _ARITHMETIC_DTYPES:
        raise ValueError(f"{what} supports int8/int32/int64/fp16/fp32/fp64/bf16")


def _cuda_check(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous (call .contiguous() first — "
            "collectives run in place and cannot copy back)")


class ProcessGroup:
    """A ukernel process group over one GPU.

    prepare() (peer setup, MR registration, buffer resolution) is cached
    per collective shape + pointer set, so steady-state calls only pay
    submit + wait.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        gpu_id: int,
        exchanger_ip: str = "127.0.0.1",
        exchanger_port: int = 16998,
        threads_per_block: int = 64,
        blocks_per_worker: int = 1,
        smem_size: int = 4096,
    ) -> None:
        self._impl = _ProcessGroup(
            rank, world_size, gpu_id,
            exchanger_ip, exchanger_port,
            threads_per_block, blocks_per_worker, smem_size,
        )

    @property
    def rank(self) -> int:
        return self._impl.rank

    @property
    def world_size(self) -> int:
        return self._impl.world_size

    @property
    def gpu_id(self) -> int:
        return self._impl.gpu_id

    def allreduce(self, tensor, op=ReduceOp.SUM, tile_bytes=64 << 10,
                  signal_group_tiles=1):
        """In-place allreduce over the ring algorithm.

        signal_group_tiles: one signal per this many tiles per chunk pair
        (1 = per tile; 2-4 usually best for small messages).
        """
        _cuda_check(tensor, "tensor")
        _validate_dtype(tensor, "allreduce")
        self._impl.allreduce(tensor, int(op), tile_bytes, signal_group_tiles)

    def alltoall(self, tensor, tile_bytes=64 << 10, signal_group_tiles=1):
        """In-place equal-split alltoall."""
        _cuda_check(tensor, "tensor")
        _validate_dtype(tensor, "alltoall")
        self._impl.alltoall(tensor, tile_bytes, signal_group_tiles)

    def alltoallv(self, output, input, output_split_sizes, input_split_sizes,
                  tile_bytes=64 << 10, signal_group_tiles=1):
        """Variable-split alltoall (out of place)."""
        _cuda_check(output, "output")
        _cuda_check(input, "input")
        _validate_dtype(output, "alltoallv")
        self._impl.alltoallv(
            output, input,
            list(output_split_sizes), list(input_split_sizes),
            tile_bytes, signal_group_tiles)

    def barrier(self):
        self._impl.barrier()


_DEFAULT_GROUP: Optional[ProcessGroup] = None


def init_process_group(
    backend: str = "ukernel",
    *,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    gpu_id: Optional[int] = None,
    exchanger_ip: Optional[str] = None,
    exchanger_port: Optional[int] = None,
    threads_per_block: int = 64,
    blocks_per_worker: int = 1,
    smem_size: int = 4096,
) -> ProcessGroup:
    global _DEFAULT_GROUP
    if backend not in ("ukernel", "ucc", "ccl"):
        raise ValueError(f"unsupported backend: {backend}")
    rank = int(os.getenv("RANK", "0")) if rank is None else rank
    world_size = int(os.getenv("WORLD_SIZE", "1")) if world_size is None else world_size
    gpu_id = int(os.getenv("LOCAL_RANK", str(rank))) if gpu_id is None else gpu_id
    exchanger_ip = os.getenv("MASTER_ADDR", "127.0.0.1") if exchanger_ip is None else exchanger_ip
    exchanger_port = int(os.getenv("MASTER_PORT", "16998")) if exchanger_port is None else exchanger_port
    _DEFAULT_GROUP = ProcessGroup(
        rank=rank, world_size=world_size, gpu_id=gpu_id,
        exchanger_ip=exchanger_ip, exchanger_port=exchanger_port,
        threads_per_block=threads_per_block,
        blocks_per_worker=blocks_per_worker,
        smem_size=smem_size,
    )
    return _DEFAULT_GROUP


def is_initialized() -> bool:
    return _DEFAULT_GROUP is not None


def get_rank(group=None) -> int:
    pg = _DEFAULT_GROUP if group is None else group
    if pg is None:
        raise RuntimeError("process group not initialized")
    return pg.rank


def get_world_size(group=None) -> int:
    pg = _DEFAULT_GROUP if group is None else group
    if pg is None:
        raise RuntimeError("process group not initialized")
    return pg.world_size


def barrier(group=None):
    pg = _DEFAULT_GROUP if group is None else group
    if pg is None:
        raise RuntimeError("process group not initialized")
    pg.barrier()


def allreduce(tensor, op=ReduceOp.SUM, group=None, tile_bytes=64 << 10,
              signal_group_tiles=1):
    pg = _DEFAULT_GROUP if group is None else group
    if pg is None:
        raise RuntimeError("process group not initialized")
    pg.allreduce(tensor, op=op, tile_bytes=tile_bytes,
                 signal_group_tiles=signal_group_tiles)


def alltoall(tensor, group=None, tile_bytes=64 << 10, signal_group_tiles=1):
    pg = _DEFAULT_GROUP if group is None else group
    if pg is None:
        raise RuntimeError("process group not initialized")
    pg.alltoall(tensor, tile_bytes=tile_bytes,
                signal_group_tiles=signal_group_tiles)


def alltoallv(output, input, output_split_sizes, input_split_sizes, group=None,
              tile_bytes=64 << 10, signal_group_tiles=1):
    pg = _DEFAULT_GROUP if group is None else group
    if pg is None:
        raise RuntimeError("process group not initialized")
    pg.alltoallv(output, input, output_split_sizes, input_split_sizes,
                 tile_bytes=tile_bytes, signal_group_tiles=signal_group_tiles)
