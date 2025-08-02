#!/usr/bin/env python3
"""
UCCL GPU-driven benchmark (Python) — Remote-only

Usage
-----
# Receiver on node B
python benchmark_remote.py --rank 1 --peer-ip <nodeA_ip>

# Sender on node A (waits 2s before issuing)
python benchmark_remote.py --rank 0 --peer-ip <nodeB_ip>
"""
import argparse
import signal
import sys
import time
from typing import List

import torch
import pyproxy


def make_proxies(
    bench: pyproxy.Bench,
    gpu_addr: int,
    total_size: int,
    rank: int,
    peer_ip: str,
    mode: str,
) -> List[pyproxy.Proxy]:
    """Create one proxy per GPU block and start them in the requested mode."""
    env = bench.env_info()
    num_blocks = int(env["blocks"])
    proxies: List[pyproxy.Proxy] = []

    for i in range(num_blocks):
        rb_i = bench.ring_addr(i)
        p = pyproxy.Proxy(
            rb_addr=rb_i,
            block_idx=i,
            gpu_buffer_addr=gpu_addr,
            total_size=total_size,
            rank=rank,
            peer_ip=peer_ip or "",
        )
        if mode == "sender":
            p.start_sender()
        elif mode == "remote":
            p.start_remote()
        else:
            raise ValueError(f"Unknown mode: {mode}")
        proxies.append(p)
    return proxies


def stop_proxies(proxies: List[pyproxy.Proxy]):
    for p in proxies:
        try:
            p.stop()
        except Exception:
            pass


def run_rank0_sender(args):
    bench = pyproxy.Bench()
    env = bench.env_info()

    nbytes = int(args.size_mb) << 20
    gpu_buf = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    gpu_addr = gpu_buf.data_ptr()

    print(
        f"[rank 0] peer={args.peer_ip} blocks={int(env['blocks'])} "
        f"tpb={int(env['threads_per_block'])} iters={int(env['iterations'])} "
        f"size={args.size_mb} MiB"
    )

    proxies = make_proxies(
        bench, gpu_addr, nbytes, rank=0, peer_ip=args.peer_ip, mode="sender"
    )
    try:
        if args.wait_sec > 0:
            print(f"[rank 0] Waiting {args.wait_sec} s before issuing commands...")
            time.sleep(args.wait_sec)

        bench.launch_gpu_issue_batched_commands()
        try:
            bench.sync_stream_interruptible(poll_ms=5, timeout_ms=120_000)
        except KeyboardInterrupt:
            print("Interrupted by user; stopping proxies and resetting device...")
            stop_proxies(proxies)
            try:
                pyproxy.device_reset()  # optional if you added it
            except Exception as e:
                print("device_reset failed:", e)
            raise
        bench.sync_stream()

        stop_proxies(proxies)

        bench.print_block_latencies()
        stats = bench.compute_stats()
        bench.print_summary(stats)
        print("elapsed_ms:", bench.last_elapsed_ms())
    finally:
        stop_proxies(proxies)


def run_rank1_remote(args):
    bench = pyproxy.Bench()
    env = bench.env_info()

    nbytes = int(args.size_mb) << 20
    gpu_buf = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    gpu_addr = gpu_buf.data_ptr()

    print(
        f"[rank 1] peer={args.peer_ip} blocks={int(env['blocks'])} "
        f"tpb={int(env['threads_per_block'])} iters={int(env['iterations'])} "
        f"size={args.size_mb} MiB"
    )

    proxies = make_proxies(
        bench, gpu_addr, nbytes, rank=1, peer_ip=args.peer_ip, mode="remote"
    )

    stop_flag = {"stop": False}

    def _sigint(_sig, _frm):
        stop_flag["stop"] = True
        print("\n[rank 1] Caught SIGINT, shutting down...")

    signal.signal(signal.SIGINT, _sigint)

    try:
        while not stop_flag["stop"]:
            print("[rank 1] waiting...")
            time.sleep(1.0)
    finally:
        stop_proxies(proxies)


def parse_args():
    p = argparse.ArgumentParser(description="UCCL GPU-driven benchmark (remote-only)")
    p.add_argument(
        "--rank",
        type=int,
        choices=[0, 1],
        required=True,
        help="0=sender/issuer, 1=remote/receiver",
    )
    p.add_argument("--peer-ip", type=str, required=True, help="Peer IP address")
    p.add_argument("--size-mb", type=int, default=256, help="Total buffer size in MiB")
    p.add_argument(
        "--wait-sec",
        type=int,
        default=2,
        help="Sender delay before issuing commands (rank 0)",
    )
    return p.parse_args()


def main():
    if not torch.cuda.is_available():
        print(
            "CUDA is not available. Please ensure a CUDA-capable device is present.",
            file=sys.stderr,
        )
        sys.exit(1)

    args = parse_args()

    dev = torch.cuda.current_device()
    pyproxy.set_device(dev)
    print(f"[py] Using CUDA device {dev}: {torch.cuda.get_device_name(dev)}")
    if args.rank == 0:
        run_rank0_sender(args)
    else:
        run_rank1_remote(args)


if __name__ == "__main__":
    main()
