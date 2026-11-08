"""
Simple internode test for DeepEP low-latency kernels.
This test avoids the IPC handle issues by focusing only on low-latency functionality.
On first node:
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
  --master_addr=10.1.227.34 --master_port=12356 \
  bench/test_internode_simple.py

On second node:
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \
  --master_addr=10.1.227.34 --master_port=12356 \
  bench/test_internode_simple.py
"""

import torch
import torch.distributed as dist
from buffer import Buffer
import os

from utils import (
    init_dist_under_torchrun,
    detect_ib_hca,
)


def test_simple_internode(rank: int, num_ranks: int, group: dist.ProcessGroup):
    num_tokens = 512
    hidden = 2048
    num_experts = 8 * num_ranks
    num_topk = 4
    device_index = int(os.environ["LOCAL_RANK"])
    print(f"[simple-test] Running on device {device_index}", flush=True)

    torch.manual_seed(rank)
    x = torch.randn(
        (num_tokens, hidden),
        dtype=torch.bfloat16,
        device=f"cuda:{device_index}",
    )
    # Each token routes to distinct experts, as required by dispatch/combine.
    topk_idx = (
        torch.rand((num_tokens, num_experts), device=f"cuda:{device_index}")
        .topk(num_topk, dim=-1)
        .indices
    )
    scratch_nbytes = Buffer.get_low_latency_rdma_size_hint(
        num_tokens, hidden, num_ranks, num_experts
    )
    buffer = None

    try:
        buffer = Buffer(
            group=group,
            num_nvl_bytes=0,
            num_rdma_bytes=int(scratch_nbytes),
            low_latency_mode=True,
            num_qps_per_rank=num_experts // num_ranks,
            allow_nvlink_for_low_latency_mode=True,
            allow_mnnvl=False,
            explicitly_destroy=True,
        )

        if rank == 0:
            print("[simple-test] ✓ Buffer created successfully", flush=True)

        cumulative_local_expert_recv_stats = torch.zeros(
            (num_experts // num_ranks,), dtype=torch.int, device="cuda"
        )
        recv_x, recv_count, handle, event, dispatch_hook = buffer.low_latency_dispatch(
            x=x,
            topk_idx=topk_idx,
            num_max_dispatch_tokens_per_rank=num_tokens,
            num_experts=num_experts,
            use_fp8=False,
            round_scale=False,
            use_ue8m0=False,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
            async_finish=False,
            return_recv_hook=True,
        )
        dispatch_hook()

        print("[simple-test] ✓ Low-latency dispatch completed", flush=True)
        print(f"[simple-test] Received tensor shape: {recv_x.shape}", flush=True)

        topk_weights = torch.ones(
            (num_tokens, num_topk), dtype=torch.float32, device="cuda"
        )
        combined_x, combine_event, combine_hook = buffer.low_latency_combine(
            x=recv_x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            handle=handle,
            use_logfmt=False,
            zero_copy=False,
            async_finish=False,
            return_recv_hook=True,
        )
        combine_hook()

        print("[simple-test] ✓ Low-latency combine completed", flush=True)
        print(
            f"[simple-test] Combined tensor shape: {combined_x.shape}",
            flush=True,
        )
        torch.testing.assert_close(
            combined_x.float(), x.float() * num_topk, rtol=0.02, atol=0.02
        )
        print("[simple-test] ✓ Dispatch/combine values verified", flush=True)
    finally:
        if buffer is not None:
            buffer.destroy()


def test_worker(local_rank: int, num_local_ranks: int):
    rank, num_ranks, group = init_dist_under_torchrun(local_rank, num_local_ranks)

    try:
        test_simple_internode(rank, num_ranks, group)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    ib_dev = detect_ib_hca()
    if ib_dev and ib_dev.startswith("mlx"):  # Mellanox IB devices show up like mlx5_0
        os.environ["NCCL_IB_HCA"] = ib_dev
        print(f"Set NCCL_IB_HCA={ib_dev}")
    print("Simple internode test starting...")
    local_rank = int(os.environ["LOCAL_RANK"])
    num_local_ranks = int(os.environ["LOCAL_WORLD_SIZE"])
    test_worker(local_rank, num_local_ranks)
