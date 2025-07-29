import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import nvtx
import torch
import torch.distributed as dist

from pynccl import PyNcclCommunicator


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def warmup_all2all(
    iters: int = 5,
    chunk: int = 4 * 1024,
    dtype: torch.dtype = torch.float16,
    device: torch.device = torch.device("cuda:0"),
    comm: PyNcclCommunicator = None,
):
    assert comm is not None

    world_size = comm.world_size
    sendbuf = torch.randn(world_size * chunk, dtype=dtype, device=device)
    recvbuf = torch.empty_like(sendbuf)

    for _ in range(iters):
        comm.all_to_all_single(sendbuf, recvbuf)

    comm.barrier()
    torch.cuda.synchronize(device)


@dataclass
class Metrics:
    avg_time: float
    total_flops: float
    mem_buckets: np.ndarray
    flops_buckets: np.ndarray
    seq_lens: np.ndarray


def get_fcp_comm_plan(num_gpus: int):
    perm = np.random.permutation(num_gpus)
    P = np.zeros((num_gpus, num_gpus))
    P[np.arange(num_gpus), perm] = 1

    assert np.all(np.sum(P, axis=1) == 1)
    assert np.all(np.sum(P, axis=0) == 1)

    return P


def get_fcp_comm_plans(num_gpus: int, num_iters: int):
    plans = []
    for _ in range(num_iters):
        plan = get_fcp_comm_plan(num_gpus)
        plans.append(
            (
                np.argmax(plan, axis=1),
                np.argmax(plan, axis=0),
            )
        )
    return plans


def run_ring_p2p(
    block_size: int,
    num_qo_heads: int,
    gqa_group_size: int,
    head_dim: int,
    num_iters: int,
    comm: PyNcclCommunicator,
    device: torch.device,
) -> Metrics:
    global_rank = dist.get_rank(comm.group)
    world_size = dist.get_world_size(comm.group)

    # init tensors
    num_kv_heads = num_qo_heads // gqa_group_size
    send_tensor = torch.randn(
        block_size,
        2,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    recv_tensor = torch.empty_like(send_tensor, device=device)

    send_rank = (global_rank + 1) % world_size
    recv_rank = (global_rank - 1) % world_size

    comm.barrier()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    elapsed_time = []

    start_event.record()
    for idx in range(num_iters):
        with nvtx.annotate(f"iter {idx}"):
            comm.group_start()
            comm.send(send_tensor, send_rank)
            comm.recv(recv_tensor, recv_rank)
            comm.group_end()

    end_event.record()
    torch.cuda.synchronize()
    elapsed_time.append(start_event.elapsed_time(end_event))

    data = Metrics(
        avg_time=np.mean(elapsed_time) / num_iters,
        total_flops=0,
        mem_buckets=np.zeros(world_size),
        flops_buckets=np.zeros(world_size),
        seq_lens=np.zeros(world_size),
    )

    return data


def run_fcp_p2p(
    block_size: int,
    num_qo_heads: int,
    gqa_group_size: int,
    head_dim: int,
    num_iters: int,
    comm: PyNcclCommunicator,
    device: torch.device,
) -> Metrics:
    global_rank = dist.get_rank(comm.group)
    world_size = dist.get_world_size(comm.group)

    # init tensors
    num_kv_heads = num_qo_heads // gqa_group_size
    send_tensor = torch.randn(
        block_size,
        2,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    recv_tensor = torch.empty_like(send_tensor, device=device)

    comm.barrier()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    elapsed_time = []

    plans = get_fcp_comm_plans(world_size, num_iters)
    comm.barrier()

    start_event.record()
    for idx in range(num_iters):
        send_rank = plans[idx][0][global_rank]
        recv_rank = plans[idx][1][global_rank]

        with nvtx.annotate(f"iter {idx}"):
            comm.group_start()
            comm.send(send_tensor, send_rank)
            comm.recv(recv_tensor, recv_rank)
            comm.group_end()

    end_event.record()
    torch.cuda.synchronize()
    elapsed_time.append(start_event.elapsed_time(end_event))

    data = Metrics(
        avg_time=np.mean(elapsed_time) / num_iters,
        total_flops=0,
        mem_buckets=np.zeros(world_size),
        flops_buckets=np.zeros(world_size),
        seq_lens=np.zeros(world_size),
    )

    return data


if __name__ == "__main__":
    """
    Usage:
    NCCL_MAX_CTAS=8 NCCL_CGA_CLUSTER_SIZE=0 \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ENABLE_INTRA_NODE_COMM=1 PYTHONOPTIMIZE=1 \
    nsys profile --trace=cuda,nvtx,osrt --cuda-event-trace=false --delay=5 --duration=10 --force-overwrite true -o pynccl-all2all-profile \
    torchrun --standalone --nproc_per_node=8 test-pynccl-all2all.py --block-size 4096 --num-qo-heads 32 --gqa-group-size 4 --head-dim 128 --num-iters 10000
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-size", type=int, default=4 * 1024)
    parser.add_argument("--num-qo-heads", type=int, default=32)
    parser.add_argument("--gqa-group-size", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-iters", type=int, default=10)
    args = parser.parse_args()

    # setup random seed
    setup_seed(330)

    # set default device
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)

    dist.init_process_group(backend="gloo", device_id=device)
    communicator = PyNcclCommunicator(group=dist.group.WORLD, device=device)

    global_rank = dist.get_rank(group=dist.group.WORLD)
    world_size = dist.get_world_size(group=dist.group.WORLD)

    # warmup
    warmup_all2all(
        iters=10,
        chunk=args.block_size
        * 2
        * args.num_qo_heads
        // args.gqa_group_size
        * args.head_dim,
        dtype=torch.float16,
        device=device,
        comm=communicator,
    )
    run_fcp_p2p(
        block_size=args.block_size,
        num_qo_heads=args.num_qo_heads,
        gqa_group_size=args.gqa_group_size,
        head_dim=args.head_dim,
        num_iters=100,
        comm=communicator,
        device=device,
    )

    data = run_ring_p2p(
        block_size=args.block_size,
        num_qo_heads=args.num_qo_heads,
        gqa_group_size=args.gqa_group_size,
        head_dim=args.head_dim,
        num_iters=args.num_iters,
        comm=communicator,
        device=device,
    )

    if global_rank == 0:
        # print data
        msg_sz = (
            args.block_size
            * 2
            * args.num_qo_heads
            // args.gqa_group_size
            * args.head_dim
            * 2
            / 1024
            / 1024
        )
        avg_bw = msg_sz / data.avg_time

        print(
            (
                f"num_gpus={world_size},block_size={args.block_size//1024}K,num_qo_heads={args.num_qo_heads},"
                f"gqa_group_size={args.gqa_group_size},head_dim={args.head_dim},num_iters={args.num_iters},"
                f"msg_sz={msg_sz:.2f}MB,avg_bw={avg_bw:.2f}GB/s",
            )
        )

    communicator.barrier()
    dist.destroy_process_group()
