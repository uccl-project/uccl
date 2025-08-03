#!/usr/bin/env python3
"""
UCCL GPU-driven benchmark (Python) — Remote-only

Usage
-----
# Receiver on node B
python benchmark_remote.py --rank 1 --peer-ip <nodeA_ip> --size-mb 256

# Sender on node A (waits 2s before issuing)
python benchmark_remote.py --rank 0 --peer-ip <nodeB_ip> --size-mb 256 --wait-sec 2
"""

import argparse
import os
import signal
import sys
import time
from typing import List

import torch
import pyproxy


def make_proxies(
    bench: pyproxy.Bench,
    buf_addr: int,
    total_size: int,
    rank: int,
    peer_ip: str,
    mode: str,
) -> List[pyproxy.Proxy]:
    env = bench.env_info()
    num_blocks = int(env["blocks"])
    proxies: List[pyproxy.Proxy] = []
    for i in range(num_blocks):
        rb_i = bench.ring_addr(i)
        p = pyproxy.Proxy(
            rb_addr=rb_i,
            block_idx=i,
            gpu_buffer_addr=buf_addr,
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


def stop_proxies_gracefully(proxies: List[pyproxy.Proxy], join_ms: int = 500) -> None:
    for p in proxies:
        try:
            if hasattr(p, "request_stop"):
                p.request_stop()
        except Exception:
            pass
    deadline = time.time() + (join_ms / 1000.0)
    for p in proxies:
        try:
            if hasattr(p, "try_join"):
                remaining = max(1, int(1000 * (deadline - time.time())))
                p.try_join(remaining)
            else:
                p.stop()
        except Exception:
            pass


def run_rank0_sender(args):
    dev = torch.cuda.current_device()
    pyproxy.set_device(dev)
    print(f"[py] Using CUDA device {dev}: {torch.cuda.get_device_name(dev)}")

    bench = pyproxy.Bench()
    env = bench.env_info()

    nbytes = int(args.size_mb) << 20
    buf = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    buf_addr = buf.data_ptr()

    print(
        f"[rank 0] peer={args.peer_ip} blocks={int(env['blocks'])} "
        f"tpb={int(env['threads_per_block'])} iters={int(env['iterations'])} "
        f"size={args.size_mb} MiB"
    )

    proxies = make_proxies(
        bench, buf_addr, nbytes, rank=0, peer_ip=args.peer_ip, mode="sender"
    )

    stop_flag = {"stop": False}

    def _sigint(_sig, _frm):
        stop_flag["stop"] = True
        print("\n[rank 0] Ctrl-C received; shutting down...")

    signal.signal(signal.SIGINT, _sigint)

    try:
        for _ in range(max(0, args.wait_sec)):
            if stop_flag["stop"]:
                break
            time.sleep(1)

        if stop_flag["stop"]:
            stop_proxies_gracefully(proxies)
            return

        bench.launch_gpu_issue_batched_commands()
        try:
            to_ms = (args.timeout_sec * 1000) if args.timeout_sec > 0 else -1
            bench.sync_stream_interruptible(poll_ms=5, timeout_ms=to_ms)
        except KeyboardInterrupt:
            print("[rank 0] Interrupted during wait.")
        except RuntimeError as e:
            print(f"[rank 0] sync failed: {e}")

    finally:
        stop_proxies_gracefully(proxies)
    try:
        bench.print_block_latencies()
        stats = bench.compute_stats()
        bench.print_summary(stats)
        print("elapsed_ms:", bench.last_elapsed_ms())
    except Exception:
        pass

    if stop_flag["stop"]:
        os._exit(130)


def run_rank1_remote(args):
    dev = torch.cuda.current_device()
    pyproxy.set_device(dev)
    print(f"Using CUDA device {dev}: {torch.cuda.get_device_name(dev)}")

    bench = pyproxy.Bench()
    env = bench.env_info()

    nbytes = int(args.size_mb) << 20
    buf = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    buf_addr = buf.data_ptr()

    print(
        f"[rank 1] peer={args.peer_ip} blocks={int(env['blocks'])} "
        f"tpb={int(env['threads_per_block'])} iters={int(env['iterations'])} "
        f"size={args.size_mb} MiB"
    )
    proxies = make_proxies(
        bench, buf_addr, nbytes, rank=1, peer_ip=args.peer_ip, mode="remote"
    )
    workers = None
    if hasattr(pyproxy, "PeerCopyManager"):
        try:
            workers = pyproxy.PeerCopyManager(src_device=0)
            workers.start_for_proxies(proxies)
            print("[rank 1] PeerCopyManager started.")
        except Exception as e:
            print(f"[rank 1] PeerCopyManager unavailable: {e}")

    stop_flag = {"stop": False}

    def _sigint(_sig, _frm):
        stop_flag["stop"] = True
        print("\n[rank 1] Ctrl-C received; shutting down...")

    signal.signal(signal.SIGINT, _sigint)

    try:
        while not stop_flag["stop"]:
            print("[rank 1] waiting...")
            time.sleep(1.0)
    finally:
        if workers is not None:
            try:
                workers.stop()
            except Exception:
                pass
        stop_proxies_gracefully(proxies)
        if stop_flag["stop"]:
            os._exit(130)


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
    p.add_argument(
        "--timeout-sec",
        type=int,
        default=120,
        help="Abort sender wait after this many seconds (0 disables).",
    )
    return p.parse_args()


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available.", file=sys.stderr)
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
