"""Smoke test: a single process can hold one high-throughput-mode Buffer and
one low-latency-mode Buffer simultaneously without crashing on the shared
shm-barrier / proxy registry that previously segfaulted.

Pre-fix behavior:
- ``register_proxies`` aborts on the second call for the same device.
- The two proxy thread pools share ``/uccl_barrier_<ip>_uid<uid>_th<idx>``
  shm names; the second Buffer's proxy threads attach to the first
  Buffer's LocalBarrier and counters interleave, leading to undefined
  behaviour and segfaults during further RDMA setup.

Post-fix behavior:
- Proxy registry is keyed by ``(device_index, low_latency_mode)``.
- shm names include a ``_ht_`` (high-throughput) / ``_ll_`` (low-latency)
  mode token.
- LL-mode proxies pin to a CPU range disjoint from HT-mode proxies.

The test is a construction-only smoke test: build both Buffers in the same
process, do a process-wide barrier between them, and tear them down. If we
get past both constructions without an abort/segfault the regression is
fixed.

Run on a single node, 8 ranks (intranode):

    torchrun --standalone --nproc_per_node=8 \\
        bench/test_dual_mode.py

Or across two nodes, 8 ranks each:

    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=<R> \\
        --master_addr=<head_ip> --master_port=12366 \\
        bench/test_dual_mode.py
"""

import argparse
import os

import torch.distributed as dist

from buffer import Buffer
from utils import init_dist_under_torchrun


def make_low_latency_buffer(group, num_ranks, num_tokens, hidden, num_experts):
    rdma_size = Buffer.get_low_latency_rdma_size_hint(
        num_tokens, hidden, num_ranks, num_experts
    )
    return Buffer(
        group,
        num_rdma_bytes=rdma_size,
        low_latency_mode=True,
        num_qps_per_rank=max(1, num_experts // num_ranks),
        allow_nvlink_for_low_latency_mode=True,
        allow_mnnvl=False,
        explicitly_destroy=True,
    )


def make_high_throughput_buffer(group, num_ranks, hidden):
    from uccl.ep import Config as EpConfig  # type: ignore

    cfg = EpConfig(
        num_sms=24,
        num_max_nvl_chunked_send_tokens=8,
        num_max_nvl_chunked_recv_tokens=72,
        num_max_rdma_chunked_send_tokens=8,
        num_max_rdma_chunked_recv_tokens=128,
    )
    hidden_bytes = hidden * 2  # bf16
    nvl_bytes = cfg.get_nvl_buffer_size_hint(hidden_bytes, num_ranks)
    rdma_bytes = cfg.get_rdma_buffer_size_hint(hidden_bytes, num_ranks)
    return Buffer(
        group,
        num_nvl_bytes=nvl_bytes,
        num_rdma_bytes=rdma_bytes,
        low_latency_mode=False,
        num_qps_per_rank=1,
        explicitly_destroy=True,
    )


def main(local_rank: int, num_local_ranks: int, args):
    rank, num_ranks, group = init_dist_under_torchrun(local_rank, num_local_ranks)

    # Number of experts must divide num_ranks for LL mode and
    # num_ranks * NUM_MAX_NVL_PEERS for high-throughput mode. Pick a safe
    # small multiple.
    num_experts = args.num_experts or max(num_ranks * 2, 16)
    if num_experts % num_ranks:
        num_experts = ((num_experts // num_ranks) + 1) * num_ranks

    if rank == 0:
        print(
            f"[dual-mode] starting: world={num_ranks} hidden={args.hidden} "
            f"tokens/rank={args.num_tokens} experts={num_experts}",
            flush=True,
        )

    # 1) low-latency Buffer first
    if rank == 0:
        print("[dual-mode] creating low-latency Buffer ...", flush=True)
    ll_buf = make_low_latency_buffer(
        group, num_ranks, args.num_tokens, args.hidden, num_experts
    )
    dist.barrier(group)
    if rank == 0:
        print("[dual-mode] low-latency Buffer ready", flush=True)

    # 2) high-throughput Buffer in the SAME process. Pre-fix: aborts in
    # register_proxies, or segfaults on shared LocalBarrier shm.
    if rank == 0:
        print("[dual-mode] creating high-throughput Buffer ...", flush=True)
    ht_buf = make_high_throughput_buffer(group, num_ranks, args.hidden)
    dist.barrier(group)
    if rank == 0:
        print(
            "[dual-mode] high-throughput Buffer ready -- both modes coexisting",
            flush=True,
        )

    # Tear down in reverse order.
    if rank == 0:
        print("[dual-mode] destroying buffers ...", flush=True)
    ht_buf.destroy()
    ll_buf.destroy()
    dist.barrier(group)
    if rank == 0:
        print("[dual-mode] PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument(
        "--num-experts",
        type=int,
        default=0,
        help="0 = pick a small multiple of world size",
    )
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    num_local_ranks = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    main(local_rank, num_local_ranks, args)
