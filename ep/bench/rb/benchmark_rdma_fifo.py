#!/usr/bin/env python3
"""Run the two-node GPU-driven RDMA FIFO benchmark with torchrun."""

import argparse
import os

import torch
import torch.distributed as dist

from uccl import ep
from utils import detect_ib_hca, get_cpu_proxies_meta, init_dist


def make_fifo_proxies(bench, buf_addr, total_size, rank, node_idx, local_rank):
    """Construct listeners before exchanging their actual ports with peers."""
    proxies = []
    for i in range(int(bench.env_info().blocks)):
        proxy = ep.FifoProxy(
            thread_idx=i,
            gpu_buffer_addr=buf_addr,
            total_size=total_size,
            rank=rank,
            node_idx=node_idx,
            local_rank=local_rank,
            is_intranode=False,
        )
        proxy.set_fifo(bench.get_fifo(i))
        proxies.append(proxy)
    return proxies


def run_rank0_sender(bench, proxies):
    env = bench.env_info()
    print(
        f"[rank 0] blocks={int(env.blocks)} "
        f"tpb={int(env.threads_per_block)} iters={int(env.iterations)} (FIFO mode)",
        flush=True,
    )
    bench.launch_gpu_issue_batched_commands()
    # A timeout or GPU failure must fail the benchmark rather than print success.
    bench.sync_stream_interruptible(poll_ms=5, timeout_ms=5000)
    for proxy in proxies:
        count = proxy.processed_count()
        print(f"FifoProxy {proxy.thread_idx} processed_count: {count}", flush=True)
        if count != int(env.iterations):
            raise RuntimeError(
                f"Proxy {proxy.thread_idx} acknowledged {count} commands; "
                f"expected {int(env.iterations)}"
            )
        print(
            f"FifoProxy {proxy.thread_idx} avg_wr_latency_us: "
            f"{proxy.avg_wr_latency_us()} µs"
        )
    bench.print_block_latencies()
    bench.print_summary(bench.compute_stats())
    print("Benchmark elapsed_ms:", bench.last_elapsed_ms())


def main():
    parser = argparse.ArgumentParser(
        description="UCCL GPU-driven benchmark (remote-only, FIFO mode, torchrun)"
    )
    parser.add_argument(
        "--size-mb", type=int, default=256, help="Total buffer size in MiB"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    try:
        ib_dev = detect_ib_hca()
        if ib_dev:
            os.environ["NCCL_IB_HCA"] = ib_dev
            print(f"Set NCCL_IB_HCA={ib_dev}", flush=True)
    except Exception:
        pass

    local_rank = int(os.environ["LOCAL_RANK"])
    num_local_ranks = int(os.environ["LOCAL_WORLD_SIZE"])
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    if num_ranks != 2 or num_local_ranks != 1:
        raise ValueError(
            "RDMA FIFO benchmark requires one process on each of two nodes"
        )
    ep.set_device(local_rank)
    scratch_nbytes = int(args.size_mb) << 20
    scratch = torch.empty(
        scratch_nbytes, dtype=torch.uint8, device=f"cuda:{local_rank}"
    )
    bench = ep.BenchFifo()
    proxies = make_fifo_proxies(
        bench, scratch.data_ptr(), scratch_nbytes, rank, rank, local_rank
    )
    try:
        rank2meta = get_cpu_proxies_meta(
            proxies, rank, scratch.data_ptr(), scratch_nbytes, num_ranks, group
        )
        peers = [rank2meta[r] for r in range(num_ranks)]
        for proxy in proxies:
            proxy.set_peers_meta(peers)
        for proxy in proxies:
            if rank == 0:
                proxy.start_sender()
            else:
                proxy.start_remote()
        dist.barrier(group)
        if rank == 0:
            run_rank0_sender(bench, proxies)
        # Keep receiver progress alive until the sender has completed all WRs.
        dist.barrier(group)
    finally:
        for proxy in proxies:
            proxy.stop()
        dist.destroy_process_group()
    print("FIFO benchmark completed.", flush=True)


if __name__ == "__main__":
    main()
