#!/usr/bin/env python3
"""
UCCL GPU-driven benchmark (Python) — Remote-only (torchrun)

Usage
-----
# Node 0
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
  --master_addr=10.1.227.34 --master_port=12356 \
  bench/benchmark_remote.py

# Node 1
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \
  --master_addr=10.1.227.34 --master_port=12356 \
  bench/benchmark_remote.py
"""

import argparse
import os
import sys
from typing import List
import time
import torch
import torch.distributed as dist

try:
    from uccl import ep
except ImportError as exc:
    sys.stderr.write("Failed to import ep\n")
    raise

from utils import init_dist, get_peer_ip, detect_ib_hca, get_cpu_proxies_meta


def make_proxies(
    bench: ep.Bench,
    buf_addr: int,
    total_size: int,
    rank: int,
    peer_ip: str,
    mode: str,
    peers_meta_list=None,
) -> List[ep.Proxy]:
    env = bench.env_info()
    num_blocks = int(env.blocks)
    proxies: List[ep.Proxy] = []
    for i in range(num_blocks):
        rb_i = bench.ring_addr(i)
        p = ep.Proxy(
            rb_addr=rb_i,
            thread_idx=i,
            gpu_buffer_addr=buf_addr,
            total_size=total_size,
            rank=rank,
            peer_ip=peer_ip or "",
        )
        if peers_meta_list is not None:
            p.set_peers_meta(peers_meta_list)
        proxies.append(p)
    for p in proxies:
        if mode == "sender":
            p.start_sender()
        elif mode == "remote":
            p.start_remote()
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return proxies


def run_rank0_sender(args, peer_ip: str, peers_meta_list: list, nbytes: int, buf_addr):
    dev = torch.cuda.current_device()
    ep.set_device(dev)
    print(
        f"[rank 0] Using CUDA device {dev}: {torch.cuda.get_device_name(dev)}",
        flush=True,
    )

    bench = ep.Bench()
    env = bench.env_info()
    print(
        f"[rank 0] peer={peer_ip} blocks={int(env.blocks)} "
        f"tpb={int(env.threads_per_block)} iters={int(env.iterations)}",
        flush=True,
    )
    proxies = make_proxies(
        bench,
        buf_addr,
        nbytes,
        rank=0,
        peer_ip=peer_ip,
        mode="sender",
        peers_meta_list=peers_meta_list,
    )
    bench.launch_gpu_issue_batched_commands()
    try:
        bench.sync_stream_interruptible(poll_ms=5, timeout_ms=2000)
    except KeyboardInterrupt:
        print("[rank 0] Interrupted during wait.")
    except RuntimeError as e:
        print(f"[rank 0] sync failed: {e}")

    try:
        for p in proxies:
            print(f"Proxy {p.thread_idx} avg_wr_latency_us: {p.avg_wr_latency_us()} µs")
            p.stop()
    except Exception:
        pass

    bench.print_block_latencies()
    stats = bench.compute_stats()
    bench.print_summary(stats)
    print("Benchmark elapsed_ms:", bench.last_elapsed_ms())


def run_rank1_remote(args, peer_ip: str, peers_meta_list: list, nbytes: int, buf_addr):
    dev = torch.cuda.current_device()
    ep.set_device(dev)
    bench = ep.Bench()
    env = bench.env_info()
    proxies = make_proxies(
        bench,
        buf_addr,
        nbytes,
        rank=1,
        peer_ip=peer_ip,
        mode="remote",
        peers_meta_list=peers_meta_list,
    )
    device_index = int(os.environ.get("LOCAL_RANK", "0"))
    workers = ep.PeerCopyManager(src_device=device_index)
    workers.start_for_proxies(proxies)
    print("[rank 1] PeerCopyManager started.", flush=True)
    time.sleep(2)
    try:
        workers.stop()
    except Exception:
        pass
    try:
        for p in proxies:
            p.stop()
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser(
        description="UCCL GPU-driven benchmark (remote-only, torchrun)"
    )
    p.add_argument("--size-mb", type=int, default=256, help="Total buffer size in MiB")
    return p.parse_args()


def main():
    try:
        ib_dev = detect_ib_hca()
        if ib_dev:
            os.environ["NCCL_IB_HCA"] = ib_dev
            print(f"Set NCCL_IB_HCA={ib_dev}", flush=True)
    except Exception:
        pass

    if not torch.cuda.is_available():
        print("CUDA is not available.", file=sys.stderr)
        sys.exit(1)

    args = parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    num_local_ranks = int(os.environ["LOCAL_WORLD_SIZE"])
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    peer_ip = get_peer_ip(rank, num_ranks, group)

    scratch_nbytes = int(args.size_mb) << 20
    scratch = torch.empty(
        scratch_nbytes, dtype=torch.uint8, device=f"cuda:{local_rank}"
    )
    scratch_ptr = scratch.data_ptr()
    scratch_bytes = scratch.numel() * scratch.element_size()

    rank2meta = get_cpu_proxies_meta(rank, scratch_ptr, scratch_bytes, num_ranks, group)
    peers_meta_list = [rank2meta[r] for r in range(num_ranks)]
    dist.barrier(group)

    if rank == 0:
        run_rank0_sender(args, peer_ip, peers_meta_list, scratch_nbytes, scratch_ptr)
    else:
        run_rank1_remote(args, peer_ip, peers_meta_list, scratch_nbytes, scratch_ptr)

    try:
        dist.barrier(group)
    except Exception:
        pass
    try:
        dist.destroy_process_group()
    except Exception:
        pass


if __name__ == "__main__":
    main()
