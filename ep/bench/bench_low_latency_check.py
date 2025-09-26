"""
This is the same test_low_latency.py test in DeepEP's repo.
On first node:
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
  --master_addr=10.1.1.171 --master_port=12355 \
  bench/bench_low_latency_check.py

On second node:
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \
  --master_addr=10.1.1.171 --master_port=12355 \
  bench/bench_low_latency_check.py
"""

import argparse
import random
import os
import torch
import torch.distributed as dist
from functools import partial

from buffer import Buffer
from utils import (
    init_dist_under_torchrun,
    bench,
    initialize_uccl,
    destroy_uccl,
)

def test_main(
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
    rank: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    buffer: Buffer,
    use_logfmt: bool = False,
    seed: int = 0,
):
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    # NOTES: the last one is for performance testing
    # Most of the values in the perf case is lower than the threshold, casting most channels
    current_x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * 0.1
    
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs()
        + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    cumulative_local_expert_recv_stats = torch.zeros(
        (num_local_experts,), dtype=torch.int, device="cuda"
    )

    # noinspection PyShadowingNames
    def large_gemm_with_hook(hook):
        mat_0 = torch.randn((8192, 8192), dtype=torch.float)
        mat_1 = torch.randn((8192, 8192), dtype=torch.float)
        mat_0 @ mat_1
        hook()

    # noinspection PyShadowingNames
    def test_func(return_recv_hook: bool):
        recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
            current_x,
            topk_idx,
            num_tokens,
            num_experts,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
            use_fp8=True,
            async_finish=False,
            return_recv_hook=return_recv_hook,
        )
        large_gemm_with_hook(hook) if return_recv_hook else None

    # Calculate bandwidth
    num_fp8_bytes = (hidden + hidden / 128 * 4 + 16)
    num_dispatch_comm_bytes = 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += num_fp8_bytes * num_selections

    # Dispatch + combine testing
    avg_t, min_t, max_t = bench(partial(test_func, return_recv_hook=False))
    print(
        f"[rank {rank}] Dispatchbandwidth: {(num_dispatch_comm_bytes) / 1e6 / avg_t:.2f} MB/s, "
        f"avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us",
        flush=True,
    )
    return


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist_under_torchrun(local_rank, num_local_ranks)
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts
    num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
        num_tokens, hidden, num_ranks, num_experts
    )

    # UCCL new code for initialization
    device_index = int(os.environ["LOCAL_RANK"])
    scratch = torch.zeros(
        num_rdma_bytes, dtype=torch.uint8, device=f"cuda:{device_index}"
    )
    proxies, workers = initialize_uccl(
        scratch, num_rdma_bytes, rank, num_ranks, group, args.num_experts
    )

    buffer = Buffer(
        group,
        rdma_buffer_ptr=scratch.data_ptr(),
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=num_experts // num_ranks,
        allow_nvlink_for_low_latency_mode=not args.disable_nvlink,
        explicitly_destroy=True,
        allow_mnnvl=args.allow_mnnvl,
    )

    buffer.connect_atomic_buffer(proxies[0])

    for proxy in proxies:
        proxy.calculate_and_set_dispatch_recv_data_offset(
            num_tokens, hidden, num_experts
        )
        proxy.set_atomic_buffer_ptr(proxies[0].get_atomic_buffer_ptr())

    test_main(
        num_tokens,
        hidden,
        num_experts,
        num_topk,
        rank,
        num_ranks,
        group,
        buffer,
        use_logfmt=args.use_logfmt,
        seed=1,
    )

    # Destroy the buffer runtime and communication group
    group.barrier()
    buffer.destroy()
    dist.barrier()
    destroy_uccl(proxies, workers)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    # TODO: you may modify NUMA binding for less CPU overhead
    # TODO: buggy with `num_tokens=512`

    parser = argparse.ArgumentParser(description="Test low-latency EP kernels")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of processes to spawn (default: 8)",
    )
    parser.add_argument(
        "--num-tokens", type=int, default=128, help="Number of tokens (default: 128)"
    )
    parser.add_argument(
        "--hidden", type=int, default=7168, help="Hidden dimension size (default: 7168)"
    )
    parser.add_argument(
        "--num-topk", type=int, default=8, help="Number of top-k experts (default: 8)"
    )
    parser.add_argument(
        "--num-experts", type=int, default=288, help="Number of experts (default: 288)"
    )
    parser.add_argument(
        "--allow-mnnvl", action="store_true", help="Allow MNNVL for communication"
    )
    parser.add_argument(
        "--disable-nvlink",
        action="store_true",
        help="Whether to disable NVLink for testing",
    )
    parser.add_argument(
        "--use-logfmt", action="store_true", help="Whether to test LogFMT combine"
    )
    args = parser.parse_args()

    num_processes = args.num_processes
    # NOTE: modified from deep_ep
    local_rank = int(os.environ["LOCAL_RANK"])
    num_local_ranks = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    test_loop(local_rank, num_local_ranks, args)
