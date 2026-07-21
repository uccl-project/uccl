"""Performance benchmark for the unified dual-mode Buffer.

Runs the two canonical benchmarks back-to-back on ONE Buffer + one dual proxy
pool, so the numbers are directly comparable with the dedicated-mode baselines:

- Phase 1: ``test_internode.test_main`` (high-throughput dispatch/combine
  correctness + full tuning sweep, reports best BF16/FP8 dispatch and BF16
  combine bandwidth in GB/s RDMA + NVL)
- ``clean_low_latency_buffer`` (mode switch)
- Phase 2: ``test_low_latency.test_main`` (LL dispatch/combine correctness +
  per-rank latency and bandwidth)

Launch (2 nodes x 8 ranks):

    ./run_bench.sh <node_rank> bench_dual_mode.py \
        --num-tokens=4096 --hidden=7168 --num-topk=8 --num-experts=256 \
        --ll-num-tokens=128
"""

import argparse
import os

import torch
import torch.distributed as dist

from buffer import Buffer
from utils import init_dist_under_torchrun
import test_internode as ti
import test_low_latency as tll


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank, num_ranks, group = init_dist_under_torchrun(local_rank, num_local_ranks)
    num_nodes = num_ranks // num_local_ranks
    assert num_nodes > 1, "dual-mode benchmark is internode-only"

    if torch.version.cuda:
        num_sms = 24
    else:
        num_sms = 64 if num_nodes < 4 else 32

    # Size for the union of the HT layout (canonical internode sizing) and the
    # LL layout.
    num_nvl_bytes, ht_rdma_bytes = ti.compute_buffer_sizes(
        num_sms, args.hidden, num_ranks
    )
    ll_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
        args.ll_num_tokens, args.hidden, num_ranks, args.num_experts
    )
    num_rdma_bytes = max(ht_rdma_bytes, ll_rdma_bytes)

    if local_rank == 0:
        print(
            f"[dual-bench] world={num_ranks} nodes={num_nodes} "
            f"nvl={num_nvl_bytes / 1e9:.2f}GB "
            f"rdma={num_rdma_bytes / 1e9:.2f}GB (ht={ht_rdma_bytes / 1e9:.2f} "
            f"ll={ll_rdma_bytes / 1e9:.2f})",
            flush=True,
        )

    buffer = Buffer(
        group,
        num_nvl_bytes=num_nvl_bytes,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=args.num_experts // num_ranks,
        allow_nvlink_for_low_latency_mode=True,
        explicitly_destroy=True,
        dual_mode=True,
    )
    dist.barrier(group)

    # Phase 1: canonical high-throughput benchmark (correctness + tuning).
    if args.phase in ("both", "ht"):
        if local_rank == 0:
            print(
                "[dual-bench] ===== HT phase (test_internode.test_main) =====",
                flush=True,
            )
        ti.test_main(
            args,
            num_sms,
            local_rank,
            num_local_ranks,
            num_ranks,
            num_nodes,
            rank,
            buffer,
            group,
            False,
        )
        dist.barrier(group)

    if args.phase in ("both", "ll"):
        # Mode switch: HT residue -> clean LL layout + shared atomic buffer.
        if local_rank == 0:
            print(
                "[dual-bench] ===== clean_low_latency_buffer (mode switch) =====",
                flush=True,
            )
        buffer.clean_low_latency_buffer(
            args.ll_num_tokens, args.hidden, args.num_experts
        )
        torch.cuda.synchronize()
        dist.barrier(group)

        # Phase 2: canonical low-latency benchmark (correctness + latency/BW).
        if local_rank == 0:
            print(
                "[dual-bench] ===== LL phase (test_low_latency.test_main) =====",
                flush=True,
            )
        tll.test_main(
            args.ll_num_tokens,
            args.hidden,
            args.num_experts,
            args.num_topk,
            rank,
            num_ranks,
            group,
            buffer,
            use_logfmt=False,
            dispatch_use_fp8=True,
            seed=1,
        )
        dist.barrier(group)

    buffer.destroy()
    dist.barrier(group)
    if local_rank == 0:
        print("[dual-bench] DONE", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-mode buffer benchmark")
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk-groups", type=int, default=None)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--ll-num-tokens", type=int, default=128)
    parser.add_argument("--phase", choices=["both", "ht", "ll"], default="both")
    args = parser.parse_args()

    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if args.num_topk_groups is None:
        args.num_topk_groups = min(world_size // local_world_size, 4)

    local_rank = int(os.environ["LOCAL_RANK"])
    main(local_rank, local_world_size, args)
