"""
Single-node test
torchrun --nproc_per_node=4 bench/test_single_node.py --num-tokens 1024 --hidden 4096 --num-topk 4
"""

import torch
import torch.distributed as dist
import os
import argparse
from buffer import Buffer
from utils import init_dist, detect_ib_hca, initialize_uccl, destroy_uccl


def test_single(rank, num_ranks, group, args):
    device_index = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(device_index)

    num_tokens, hidden, num_topk = args.num_tokens, args.hidden, args.num_topk
    num_experts = args.num_experts or 4 * num_ranks

    x = torch.randn(
        (num_tokens, hidden), dtype=torch.bfloat16, device=f"cuda:{device_index}"
    )
    topk_idx = torch.randint(
        0, num_experts, (num_tokens, num_topk), device=f"cuda:{device_index}"
    )

    scratch_nbytes = int(args.buffer_size * 1e9)
    scratch = torch.empty(
        scratch_nbytes, dtype=torch.uint8, device=f"cuda:{device_index}"
    )

    proxies, workers, bench = initialize_uccl(
        scratch, scratch_nbytes, rank, num_ranks, group
    )

    buffer = Buffer(
        group=group,
        rdma_buffer_ptr=scratch.data_ptr(),
        num_nvl_bytes=scratch_nbytes,
        num_rdma_bytes=scratch_nbytes,
        low_latency_mode=True,
        num_qps_per_rank=torch.cuda.get_device_properties(
            device_index
        ).multi_processor_count,
        allow_nvlink_for_low_latency_mode=(num_ranks > 1),
        allow_mnnvl=False,
        explicitly_destroy=True,
    )

    buffer.connect_atomic_buffer(proxies[0])
    for proxy in proxies:
        proxy.calculate_and_set_dispatch_recv_data_offset(
            num_tokens, hidden, num_experts
        )
        proxy.set_atomic_buffer_ptr(proxies[0].get_atomic_buffer_ptr())

    # One dispatch + combine
    recv_x, recv_count, handle, _, dispatch_hook = buffer.low_latency_dispatch(
        x=x,
        topk_idx=topk_idx,
        num_max_dispatch_tokens_per_rank=num_tokens,
        num_experts=num_experts,
        use_fp8=False,
        round_scale=False,
        use_ue8m0=False,
        cumulative_local_expert_recv_stats=torch.zeros(
            num_experts // num_ranks, device=f"cuda:{device_index}"
        ),
        async_finish=False,
        return_recv_hook=True,
    )
    dispatch_hook()

    topk_weights = torch.ones((num_tokens, num_topk), device=f"cuda:{device_index}")
    combined_x, _, combine_hook = buffer.low_latency_combine(
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
    torch.cuda.synchronize()

    if rank == 0:
        print(f"[single-node] ✓ Dispatch+Combine done. Output shape={combined_x.shape}")

    buffer.destroy()
    destroy_uccl(proxies, workers, bench)
    dist.barrier()


def test_worker(local_rank, num_local_ranks, args):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    test_single(rank, num_ranks, group, args)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument("--num-topk", type=int, default=4)
    parser.add_argument("--buffer-size", type=float, default=1.0)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    num_local_ranks = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    ib_dev = detect_ib_hca()
    if ib_dev and ib_dev.startswith("mlx"):
        os.environ["NCCL_IB_HCA"] = ib_dev

    test_worker(local_rank, num_local_ranks, args)
