import os
from enum import IntEnum
from typing import Callable, Optional, Sequence

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


def _require_contiguous_flat(tensor: torch.Tensor, name: str) -> torch.Tensor:
    _ensure_contiguous_tensor(tensor, name)
    return tensor.view(-1)


def _normalize_split_sizes(
    splits: Optional[Sequence[int]], total_elems: int, world_size: int, which: str
) -> list[int]:
    if splits is None:
        if total_elems % world_size != 0:
            raise ValueError(
                f"{which} tensor numel must be divisible by world_size when split sizes are omitted"
            )
        each = total_elems // world_size
        return [each] * world_size
    normalized = [int(v) for v in splits]
    if len(normalized) != world_size:
        raise ValueError(f"{which}_split_sizes must have length world_size")
    if any(v < 0 for v in normalized):
        raise ValueError(f"{which}_split_sizes must be non-negative")
    if sum(normalized) != total_elems:
        raise ValueError(f"sum({which}_split_sizes) must equal {which}.numel()")
    return normalized


def _is_equal_split(splits: Sequence[int]) -> bool:
    return all(v == splits[0] for v in splits)


class _WorkRunner:
    def wait(self):
        raise NotImplementedError

    def is_completed(self) -> bool:
        raise NotImplementedError


class _NativeCollectiveRunner(_WorkRunner):
    def __init__(
        self,
        group: "ProcessGroup",
        handle: int,
        *,
        result=None,
        on_complete: Optional[Callable[[], None]] = None,
    ) -> None:
        self._group = group
        self._handle = handle
        self._result = result
        self._on_complete = on_complete
        self._done = False

    def _finish(self):
        if self._done:
            return self._result
        if self._on_complete is not None:
            self._on_complete()
            self._on_complete = None
        self._done = True
        return self._result

    def wait(self):
        if self._done:
            return self._result
        self._group._impl.wait_handle(self._handle)
        return self._finish()

    def is_completed(self) -> bool:
        if self._done:
            return True
        if not self._group._impl.poll_handle(self._handle):
            return False
        self._finish()
        return True


class Work:
    def __init__(self, runner: _WorkRunner):
        self._runner = runner

    def wait(self):
        return self._runner.wait()

    def is_completed(self) -> bool:
        return self._runner.is_completed()


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
        handle = self._impl.submit_allreduce(
            tensor,
            int(op),
            tile_bytes=tile_bytes,
            num_flows=num_flows,
        )
        work = Work(_NativeCollectiveRunner(self, handle, result=tensor))
        if async_op:
            return work
        work.wait()
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
        runner = self._create_all_to_all_single_runner(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            tile_bytes=tile_bytes,
            num_flows=num_flows,
        )
        work = Work(runner)
        if async_op:
            return work
        work.wait()
        return None

    def barrier(self, async_op: bool = False):
        work = Work(_NativeCollectiveRunner(self, self._impl.submit_barrier()))
        if async_op:
            return work
        work.wait()
        return None

    def same_host(self, peer_rank: int) -> bool:
        return self._impl.same_host(peer_rank)

    def peer_transport(self, peer_rank: int) -> str:
        return self._impl.peer_transport(peer_rank)

    def _prepare_output_flat(
        self, output: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[Callable[[], None]]]:
        if not output.is_contiguous():
            raise ValueError(
                "output must be contiguous; ukernel does not perform implicit payload copies"
            )
        return output.view(-1), None

    def _create_all_to_all_single_runner(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        *,
        output_split_sizes,
        input_split_sizes,
        tile_bytes: int,
        num_flows: int,
    ) -> _WorkRunner:
        _ensure_cuda_tensor(input, "input")
        _ensure_cuda_tensor(output, "output")
        if output.dtype != input.dtype:
            raise ValueError("output and input must have the same dtype")
        if output.device != input.device:
            raise ValueError("output and input must be on the same device")

        input_splits = _normalize_split_sizes(
            input_split_sizes, input.numel(), self.world_size, "input"
        )
        output_splits = _normalize_split_sizes(
            output_split_sizes, output.numel(), self.world_size, "output"
        )

        if _is_equal_split(input_splits) and _is_equal_split(output_splits):
            if output.numel() != input.numel():
                raise ValueError(
                    "equal-split all_to_all_single requires output.numel() == input.numel()"
                )
            input_flat = _require_contiguous_flat(input, "input")
            output_flat, copy_back = self._prepare_output_flat(output)
            handle = self._impl.submit_alltoall_out(
                output_flat,
                input_flat,
                tile_bytes=tile_bytes,
                num_flows=num_flows,
            )
            return _NativeCollectiveRunner(
                self, handle, result=output, on_complete=copy_back
            )

        input_flat = _require_contiguous_flat(input, "input")
        output_flat, copy_back = self._prepare_output_flat(output)
        handle = self._impl.submit_alltoallv_out(
            output_flat,
            input_flat,
            output_splits,
            input_splits,
            tile_bytes=tile_bytes,
            num_flows=num_flows,
        )
        return _NativeCollectiveRunner(
            self, handle, result=output, on_complete=copy_back
        )


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
