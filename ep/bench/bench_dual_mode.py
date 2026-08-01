"""
Correctness + performance test for the unified dual-mode Buffer: ONE Buffer
(``low_latency_mode=True, dual_mode=True``) backed by ONE dual proxy pool
serves both the high-throughput kernels (internode.cu) and the low-latency
kernels (internode_ll.cu), the way an application toggling on sequence length
would use it.

Build:
export OMP_NUM_THREADS=6
make clean && make -j install

On first node:
NCCL_SOCKET_IFNAME=enp71s0 NCCL_IB_DISABLE=1 \
UCCL_SOCKET_IFNAME=enp71s0 GLOO_SOCKET_IFNAME=enp71s0 \
torchrun --nnodes=2 --nproc_per_node=<local_gpu_count> --node_rank=0 \
  --master_addr=<first_node_ip> --master_port=12368 \
  bench/bench_dual_mode.py --num-tokens=4096 \
  --hidden=7168 --num-topk=8 --num-experts=256 --ll-num-tokens=128

On second node:
NCCL_SOCKET_IFNAME=enp71s0 NCCL_IB_DISABLE=1 \
UCCL_SOCKET_IFNAME=enp71s0 GLOO_SOCKET_IFNAME=enp71s0 \
torchrun --nnodes=2 --nproc_per_node=<local_gpu_count> --node_rank=1 \
  --master_addr=<first_node_ip> --master_port=12368 \
  bench/bench_dual_mode.py --num-tokens=4096 \
  --hidden=7168 --num-topk=8 --num-experts=256 --ll-num-tokens=128

This benchmark verifies, all on a single dual-mode Buffer:
  * HT <-> LL mode toggling: ``--toggle-rounds`` correctness-only rounds of
    HT -> clean_low_latency_buffer -> LL before the measured round (the
    final round alternates once more, so the benchmarked kernels run on a
    buffer that has already switched modes)
  * HT dispatch/combine correctness for BF16/FP8 and the full NVL/RDMA chunk
    tuning sweep (``test_internode.test_main`` verbatim)
  * LL dispatch/combine correctness and per-rank latency/bandwidth
    (``test_low_latency.test_main`` verbatim)

The numbers are directly comparable with the dedicated-pool baselines:
run test_internode.py / test_low_latency.py with the same arguments.
"""

import argparse
import os

import torch
import torch.distributed as dist

from buffer import Buffer
from utils import init_dist_under_torchrun
import test_internode as ti
import test_low_latency as tll


def run_ht_phase(
    args,
    num_sms,
    local_rank,
    num_local_ranks,
    num_ranks,
    num_nodes,
    rank,
    buffer,
    group,
    skip_benchmark,
):
    if local_rank == 0:
        print(
            f"[dual-bench] ===== HT phase (test_internode.test_main, "
            f"benchmark={'off' if skip_benchmark else 'on'}) =====",
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
        skip_benchmark,
    )
    dist.barrier(group)


def switch_to_ll(args, buffer, group, local_rank):
    # HT kernels dirty the LL RDMA layout and the shared atomic buffer; the
    # clean is required on every HT -> LL switch (LL -> HT needs nothing:
    # LL self-resets its signaling).
    if local_rank == 0:
        print(
            "[dual-bench] ===== clean_low_latency_buffer (mode switch) =====",
            flush=True,
        )
    buffer.clean_low_latency_buffer(args.ll_num_tokens, args.hidden, args.num_experts)
    torch.cuda.synchronize()
    dist.barrier(group)


def run_ll_phase(
    args, rank, num_ranks, group, buffer, seed, skip_benchmark, local_rank
):
    if local_rank == 0:
        print(
            f"[dual-bench] ===== LL phase (test_low_latency.test_main, "
            f"benchmark={'off' if skip_benchmark else 'on'}) =====",
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
        seed=seed,
        skip_benchmark=skip_benchmark,
    )
    dist.barrier(group)


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

    # Correctness-only toggle rounds: prove the buffer survives repeated
    # HT -> clean -> LL alternation before anything is measured.
    if args.phase == "both":
        for round_idx in range(args.toggle_rounds):
            if local_rank == 0:
                print(f"[dual-bench] toggle round {round_idx}", flush=True)
            run_ht_phase(
                args,
                num_sms,
                local_rank,
                num_local_ranks,
                num_ranks,
                num_nodes,
                rank,
                buffer,
                group,
                skip_benchmark=True,
            )
            switch_to_ll(args, buffer, group, local_rank)
            run_ll_phase(
                args,
                rank,
                num_ranks,
                group,
                buffer,
                seed=round_idx,
                skip_benchmark=True,
                local_rank=local_rank,
            )

    # Measured round (correctness runs again inside each test_main).
    if args.phase in ("both", "ht"):
        run_ht_phase(
            args,
            num_sms,
            local_rank,
            num_local_ranks,
            num_ranks,
            num_nodes,
            rank,
            buffer,
            group,
            skip_benchmark=False,
        )
    if args.phase in ("both", "ll"):
        switch_to_ll(args, buffer, group, local_rank)
        run_ll_phase(
            args,
            rank,
            num_ranks,
            group,
            buffer,
            seed=args.toggle_rounds,
            skip_benchmark=False,
            local_rank=local_rank,
        )

    buffer.destroy()
    dist.barrier(group)
    if local_rank == 0:
        print("[dual-bench] DONE", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dual-mode buffer correctness + performance benchmark"
    )
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk-groups", type=int, default=None)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--ll-num-tokens", type=int, default=128)
    parser.add_argument(
        "--toggle-rounds",
        type=int,
        default=1,
        help="Correctness-only HT->clean->LL rounds before the measured round",
    )
    parser.add_argument(
        "--phase",
        choices=["both", "ht", "ll"],
        default="both",
        help="Restrict to one phase (skips the toggle rounds; for iteration)",
    )
    args = parser.parse_args()

    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if args.num_topk_groups is None:
        args.num_topk_groups = min(world_size // local_world_size, 4)

    local_rank = int(os.environ["LOCAL_RANK"])
    main(local_rank, local_world_size, args)
