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


_REDUCE_NAME_MAP = {
    "SUM": ReduceOp.SUM,
    "PRODUCT": ReduceOp.PRODUCT,
    "PROD": ReduceOp.PRODUCT,
    "MAX": ReduceOp.MAX,
    "MIN": ReduceOp.MIN,
    "BAND": ReduceOp.BAND,
}

_ARITHMETIC_DTYPES = {
    torch.int8,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
}

_BITWISE_DTYPES = {
    torch.int8,
    torch.int32,
    torch.int64,
}


def _canonical_reduce_op(op) -> ReduceOp:
    if isinstance(op, ReduceOp):
        return op
    if isinstance(op, int):
        return ReduceOp(op)
    name = getattr(op, "name", None)
    if isinstance(name, str):
        mapped = _REDUCE_NAME_MAP.get(name.upper())
        if mapped is not None:
            return mapped
    raise ValueError(f"unsupported reduce op: {op!r}")


def _validate_reduce_dtype(op: ReduceOp, tensor: torch.Tensor) -> None:
    if op == ReduceOp.BAND:
        if tensor.dtype not in _BITWISE_DTYPES:
            raise ValueError(
                "ReduceOp.BAND currently supports int8/int32/int64 tensors"
            )
        return
    if tensor.dtype not in _ARITHMETIC_DTYPES:
        raise ValueError(
            "all_reduce currently supports int8/int32/int64/fp16/fp32/fp64/bf16 tensors"
        )


def _validate_alltoall_dtype(tensor: torch.Tensor) -> None:
    if tensor.dtype not in _ARITHMETIC_DTYPES:
        raise ValueError(
            "all_to_all currently supports int8/int32/int64/fp16/fp32/fp64/bf16 tensors"
        )


def _ensure_cuda_tensor(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")


def _ensure_contiguous_tensor(tensor: torch.Tensor, name: str) -> None:
    if not tensor.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous; ukernel does not perform implicit payload copies"
        )


class Work:
    def __init__(self, result=None):
        self._result = result

    def wait(self):
        return self._result

    def is_completed(self) -> bool:
        return True


class ProcessGroup:
    def __init__(
        self,
        rank: int,
        world_size: int,
        gpu_id: int,
        exchanger_ip: str = "127.0.0.1",
        exchanger_port: int = 6979,
        transport: str = "auto",
        device_task_capacity: int = 4096,
        max_device_fifos: int = 8,
        threads_per_block: int = 256,
        fifo_capacity: int = 64,
        smem_size: int = 0,
    ) -> None:
        self._impl = _ProcessGroup(
            rank,
            world_size,
            gpu_id,
            exchanger_ip,
            exchanger_port,
            transport,
            device_task_capacity,
            max_device_fifos,
            threads_per_block,
            fifo_capacity,
            smem_size,
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

    @property
    def backend(self) -> str:
        return "ukernel"

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        async_op: bool = False,
        tile_bytes: int = 64 << 10,
        num_flows: int = 2,
    ):
        op = _canonical_reduce_op(op)
        _ensure_cuda_tensor(tensor, "tensor")
        _ensure_contiguous_tensor(tensor, "tensor")
        _validate_reduce_dtype(op, tensor)
        self._impl.allreduce(
            tensor,
            tile_bytes=tile_bytes,
            num_flows=num_flows,
        )
        if async_op:
            return Work(tensor)
        return None

    def all_to_all_single(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        output_split_sizes=None,
        input_split_sizes=None,
        async_op: bool = False,
        tile_bytes: int = 64 << 10,
        num_flows: int = 2,
    ):
        _ensure_cuda_tensor(output, "output tensor")
        _ensure_cuda_tensor(input, "input tensor")
        _ensure_contiguous_tensor(output, "output tensor")
        _ensure_contiguous_tensor(input, "input tensor")
        _validate_alltoall_dtype(input)
        if output_split_sizes is None and input_split_sizes is None:
            self._impl.alltoall_out(
                output, input,
                tile_bytes=tile_bytes, num_flows=num_flows,
            )
        else:
            self._impl.alltoallv_out(
                output, input,
                output_split_sizes or [], input_split_sizes or [],
                tile_bytes=tile_bytes, num_flows=num_flows,
            )
        if async_op:
            return _CompletedWork(output)
        return None

    def barrier(self, async_op: bool = False):
        self._impl.barrier()
        if async_op:
            return Work()
        return None

    def same_host(self, peer_rank: int) -> bool:
        return self._impl.same_host(peer_rank)

    def peer_transport(self, peer_rank: int) -> str:
        return self._impl.peer_transport(peer_rank)


_DEFAULT_GROUP: Optional[ProcessGroup] = None


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def init_process_group(
    backend: str = "ukernel",
    *,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    gpu_id: Optional[int] = None,
    exchanger_ip: Optional[str] = None,
    exchanger_port: Optional[int] = None,
    transport: str = "auto",
    device_task_capacity: int = 4096,
    max_device_fifos: int = 8,
    threads_per_block: int = 256,
    fifo_capacity: int = 64,
    smem_size: int = 0,
) -> ProcessGroup:
    global _DEFAULT_GROUP
    if backend not in ("ukernel", "ucc", "ccl"):
        raise ValueError(f"unsupported backend: {backend}")

    rank = _env_int("RANK", 0) if rank is None else rank
    world_size = _env_int("WORLD_SIZE", 1) if world_size is None else world_size
    gpu_id = _env_int("LOCAL_RANK", rank) if gpu_id is None else gpu_id
    exchanger_ip = (
        os.getenv("MASTER_ADDR", "127.0.0.1") if exchanger_ip is None else exchanger_ip
    )
    exchanger_port = (
        _env_int("MASTER_PORT", 29500) if exchanger_port is None else exchanger_port
    )

    _DEFAULT_GROUP = ProcessGroup(
        rank=rank,
        world_size=world_size,
        gpu_id=gpu_id,
        exchanger_ip=exchanger_ip,
        exchanger_port=exchanger_port,
        transport=transport,
        device_task_capacity=device_task_capacity,
        max_device_fifos=max_device_fifos,
        threads_per_block=threads_per_block,
        fifo_capacity=fifo_capacity,
        smem_size=smem_size,
    )
    return _DEFAULT_GROUP


def destroy_process_group(group: Optional[ProcessGroup] = None) -> None:
    global _DEFAULT_GROUP
    if group is None or group is _DEFAULT_GROUP:
        _DEFAULT_GROUP = None


def is_initialized() -> bool:
    return _DEFAULT_GROUP is not None


def get_rank(group: Optional[ProcessGroup] = None) -> int:
    pg = _DEFAULT_GROUP if group is None else group
    if pg is None:
        raise RuntimeError("process group is not initialized")
    return pg.rank


def get_world_size(group: Optional[ProcessGroup] = None) -> int:
    pg = _DEFAULT_GROUP if group is None else group
    if pg is None:
        raise RuntimeError("process group is not initialized")
    return pg.world_size


def barrier(group: Optional[ProcessGroup] = None, async_op: bool = False):
    pg = _DEFAULT_GROUP if group is None else group
    if pg is None:
        raise RuntimeError("process group is not initialized")
    return pg.barrier(async_op=async_op)


def all_reduce(
    tensor: torch.Tensor,
    op: ReduceOp = ReduceOp.SUM,
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
    *,
    tile_bytes: int = 64 << 10,
    num_flows: int = 2,
):
    pg = _DEFAULT_GROUP if group is None else group
    if pg is None:
        raise RuntimeError("process group is not initialized")
    return pg.all_reduce(
        tensor,
        op=op,
        async_op=async_op,
        tile_bytes=tile_bytes,
        num_flows=num_flows,
    )


def all_to_all_single(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes=None,
    input_split_sizes=None,
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
    *,
    tile_bytes: int = 64 << 10,
    num_flows: int = 2,
):
    pg = _DEFAULT_GROUP if group is None else group
    if pg is None:
        raise RuntimeError("process group is not initialized")
    return pg.all_to_all_single(
        output,
        input,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        async_op=async_op,
        tile_bytes=tile_bytes,
        num_flows=num_flows,
    )


__all__ = [
    "ProcessGroup",
    "ReduceOp",
    "Work",
    "init_process_group",
    "destroy_process_group",
    "is_initialized",
    "get_rank",
    "get_world_size",
    "barrier",
    "all_reduce",
    "all_to_all_single",
]
