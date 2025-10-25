"""
Standalone utilities for All-to-All benchmarking.
Contains all necessary data structures and distributed utilities.

Note: The distributed launching utilities (_worker_parallel_launch,
parallel_launch, parallel_launch_from_env) are adapted from the original
implementation at:
https://github.com/perplexity-ai-labs/pplx-kernels

These functions provide robust multi-node and single-node distributed
process management with proper device initialization and synchronization.

"""

import dataclasses
import logging
import os
from collections.abc import Callable
from typing import Any, Concatenate, Optional, ParamSpec

import torch
import torch.distributed as dist
from torch.multiprocessing import spawn

import nvshmem.core as nvshmem

P = ParamSpec("P")

logger = logging.getLogger(__name__)


# ============================================================================
# MoE Configuration and Test Data
# ============================================================================


@dataclasses.dataclass
class MoEConfig:
    """Configuration for Mixture of Experts model."""

    num_experts: int
    experts_per_token: int
    hidden_dim: int
    max_num_tokens: int
    in_dtype: torch.dtype = torch.bfloat16
    out_dtype: torch.dtype = torch.bfloat16
    block_size: int = 128


class RankTestData:
    """Test data generator for each rank."""

    def __init__(
        self,
        cfg: MoEConfig,
        rng: torch.Generator,
        use_max_tokens: bool,
    ) -> None:
        # Determine number of tokens
        self.num_tokens = (
            int(torch.randint(1, cfg.max_num_tokens, [1], generator=rng).item())
            if not use_max_tokens
            else cfg.max_num_tokens
        )

        # Generate expert indices for each token
        self.indices = torch.empty(
            self.num_tokens,
            cfg.experts_per_token,
            dtype=torch.int32,
        )
        for i in range(self.num_tokens):
            perm = torch.randperm(cfg.num_experts, generator=rng)
            self.indices[i] = perm[: cfg.experts_per_token]

        # Generate routing weights
        self.weights = torch.rand(
            self.num_tokens, cfg.experts_per_token, dtype=torch.float32, generator=rng
        )

        # Generate input data with optional FP8 scaling
        self.x_scale: torch.Tensor | None = None
        if cfg.in_dtype.itemsize == 1:  # FP8
            x_fp32 = torch.rand(
                self.num_tokens, cfg.hidden_dim, dtype=torch.float32, generator=rng
            )
            self.x = ((x_fp32 - 0.5) * 400).to(cfg.in_dtype)
            self.x_scale = torch.rand(
                self.num_tokens,
                (cfg.hidden_dim + cfg.block_size - 1) // cfg.block_size,
                dtype=torch.float32,
                generator=rng,
            )
        else:
            self.x = torch.randn(
                self.num_tokens, cfg.hidden_dim, dtype=cfg.in_dtype, generator=rng
            )


# ============================================================================
# Process Group Info
# ============================================================================


@dataclasses.dataclass
class ProcessGroupInfo:
    """Information about the distributed process group."""

    world_size: int
    world_local_size: int
    rank: int
    node_rank: int
    local_rank: int
    device: torch.device


# ============================================================================
# Distributed Initialization Worker
# ============================================================================


def _worker_parallel_launch(
    local_rank: int,
    world_size: int,
    world_local_size: int,
    node_rank: int,
    init_method: str,
    worker: Callable[Concatenate[ProcessGroupInfo, P], None],
    *args: P.args,
    **kwargs: P.kwargs,
) -> None:
    """Internal worker function for parallel launch."""
    rank = node_rank * world_local_size + local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Initialize process group with device_id to avoid warnings
    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        device_id=device,
    )

    world_group = torch.distributed.group.WORLD
    assert world_group is not None
    torch._C._distributed_c10d._register_process_group("default", world_group)

    # Barrier to ensure all processes are ready
    barrier = torch.tensor([rank], device=device)
    torch.distributed.all_reduce(barrier)

    # Setup logging with rank prefix
    setup_logging(f"[rank{rank:{len(str(world_size - 1))}d}] ")

    try:
        worker(
            ProcessGroupInfo(
                world_size=world_size,
                world_local_size=world_local_size,
                rank=rank,
                node_rank=node_rank,
                local_rank=local_rank,
                device=device,
            ),
            *args,
            **kwargs,
        )
    except Exception:
        logger.exception("Error in worker function of parallel_launch")
        raise
    finally:
        torch.distributed.destroy_process_group()


# ============================================================================
# Parallel Launch Functions
# ============================================================================


def parallel_launch(
    world_size: int,
    worker: Callable[Concatenate[ProcessGroupInfo, P], None],
    *args: P.args,
    **kwargs: P.kwargs,
) -> None:
    """
    Launch distributed processes using torch.multiprocessing on a single node.

    Args:
        world_size: Number of processes to spawn (should equal number of GPUs)
        worker: Function to execute on each process
        *args: Positional arguments to pass to worker
        **kwargs: Keyword arguments to pass to worker
    """
    assert not kwargs, "Keyword arguments not supported in parallel_launch"

    spawn(
        _worker_parallel_launch,
        args=(
            world_size,
            world_size,
            0,  # node_rank = 0 for single node
            "tcp://localhost:29500",
            worker,
        )
        + args,
        nprocs=world_size,
        join=True,
    )


def parallel_launch_from_env(
    worker: Callable[Concatenate[ProcessGroupInfo, P], None],
    *args: P.args,
    **kwargs: P.kwargs,
) -> None:
    """
    Launch worker function in parallel across all processes in the current environment.

    The environment must have the following variables set:
    - WORLD_SIZE: The total number of processes
    - WORLD_LOCAL_SIZE: The number of processes on the current node
    - NODE_RANK: The rank of the current node
    - MASTER_ADDR: The address of the master process
    - MASTER_PORT: The port of the master process

    Args:
        worker: Function to execute on each process
        *args: Positional arguments to pass to worker
        **kwargs: Keyword arguments to pass to worker
    """
    assert not kwargs, "Keyword arguments not supported in parallel_launch_from_env"

    world_size = int(os.environ["WORLD_SIZE"])
    world_local_size = int(os.environ["WORLD_LOCAL_SIZE"])
    node_rank = int(os.environ["NODE_RANK"])

    assert "MASTER_ADDR" in os.environ, "MASTER_ADDR not set in environment"
    assert "MASTER_PORT" in os.environ, "MASTER_PORT not set in environment"

    spawn(
        _worker_parallel_launch,
        args=(
            world_size,
            world_local_size,
            node_rank,
            "env://",
            worker,
        )
        + args,
        nprocs=world_local_size,
        join=True,
    )


# ============================================================================
# Utility Functions
# ============================================================================


def setup_logging(prefix: str = "") -> None:
    """
    Setup basic logging configuration.

    Args:
        prefix: Prefix to add to log messages
    """
    logging.basicConfig(
        level="DEBUG",
        format=prefix
        + "[%(asctime)s] [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s",
        datefmt="%H:%M:%S",
    )


def get_available_gpus() -> int:
    """Get the number of available CUDA GPUs."""
    return torch.cuda.device_count()


def check_distributed_env() -> bool:
    """Check if running in a distributed environment."""
    required_vars = ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE"]
    return all(var in os.environ for var in required_vars)


###### NVSHMEM ######
def nvshmem_init(
    global_rank: int,
    local_rank: int,
    world_size: int,
    device: Any,
    uid: Optional[Any] = None,
) -> None:
    uniqueid = nvshmem.get_unique_id(empty=True)
    if local_rank == 0:
        uniqueid = nvshmem.get_unique_id()
        broadcast_objects = [uniqueid]
    else:
        broadcast_objects = [None]

    dist.broadcast_object_list(broadcast_objects, src=0)
    dist.barrier()

    nvshmem.init(
        device=device,
        uid=broadcast_objects[0],
        rank=global_rank,
        nranks=world_size,
        initializer_method="uid",
    )


# This stream wrapper returns the format required by CUDA Python. This workaround will be removed when nvshmem4py supports Torch stream interoperability.
# For more information see: https://nvidia.github.io/cuda-python/cuda-core/latest/interoperability.html#cuda-stream-protocol
class PyTorchStreamWrapper:
    def __init__(self, pt_stream: Any) -> None:
        self.pt_stream = pt_stream
        self.handle = pt_stream.cuda_stream

    def __cuda_stream__(self) -> tuple[int, int]:
        stream_id = self.pt_stream.cuda_stream
        return (0, stream_id)
