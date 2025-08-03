"""
Benchmark UCCL Collective API

This benchmark demonstrates the high-level collective API for UCCL P2P engine.
It provides an interface similar to NCCL but uses UCCL P2P underneath.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List

import torch
import torch.distributed as dist

import os

os.environ["UCCL_RCMODE"] = "1"

from uccl import collective


def _make_buffer(size_bytes: int):
    """Allocate a contiguous GPU tensor of *size_bytes* and return it."""
    n_elems = size_bytes // 4  # float32
    tensor = torch.ones(n_elems, dtype=torch.float32)
    return tensor


def _pretty_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    val = float(num_bytes)
    for u in units:
        if val < 1024 or u == units[-1]:
            return f"{val:.0f} {u}" if u == "B" else f"{val:.1f} {u}"
        val /= 1024
    return f"{num_bytes} B"


################################################################################
# Benchmark roles
################################################################################


def _run_server(args):
    peer = 0  # client rank
    for size in args.sizes:
        tensor = _make_buffer(size)

        # Register tensor for efficient memory access
        collective.register_tensor(tensor)

        # Warm-up receive
        collective.recv(tensor, src=peer)

        start = time.perf_counter()
        total = 0
        for _ in range(args.iters):
            collective.recv(tensor, src=peer)
            total += size

        elapsed = time.perf_counter() - start

        gbps = (total * 8) / elapsed / 1e9
        gb_sec = total / elapsed / 1e9
        print(
            f"[Server] {_pretty_size(size):>9} : {gbps:7.2f} Gbps | {gb_sec:7.2f} GB/s"
        )
    print("[Server] Benchmark complete")


def _run_client(args):
    peer = 1  # server rank
    for size in args.sizes:
        tensor = _make_buffer(size)

        # Register tensor for efficient memory access
        collective.register_tensor(tensor)

        # Warm-up send
        collective.send(tensor, dst=peer)

        start = time.perf_counter()
        total = 0
        for _ in range(args.iters):
            collective.send(tensor, dst=peer)
            total += size

        elapsed = time.perf_counter() - start

        gbps = (total * 8) / elapsed / 1e9
        gb_sec = total / elapsed / 1e9
        print(
            f"[Client] {_pretty_size(size):>9} : {gbps:7.2f} Gbps | {gb_sec:7.2f} GB/s"
        )
    print("[Client] Benchmark complete")


def _run_async_server(args):
    """Demonstrate async API usage."""
    peer = 0  # client rank
    for size in args.sizes:
        tensor = _make_buffer(size)

        # Register tensor for efficient memory access
        collective.register_tensor(tensor)

        # Warm-up
        req = collective.irecv(tensor, src=peer)
        collective.wait(req)

        start = time.perf_counter()
        total = 0
        for _ in range(args.iters):
            req = collective.irecv(tensor, src=peer)
            collective.wait(req)
            total += size

        elapsed = time.perf_counter() - start

        gbps = (total * 8) / elapsed / 1e9
        gb_sec = total / elapsed / 1e9
        print(
            f"[Server Async] {_pretty_size(size):>9} : {gbps:7.2f} Gbps | {gb_sec:7.2f} GB/s"
        )
    print("[Server Async] Benchmark complete")


def _run_async_client(args):
    """Demonstrate async API usage."""
    peer = 1  # server rank
    for size in args.sizes:
        tensor = _make_buffer(size)

        # Register tensor for efficient memory access
        collective.register_tensor(tensor)

        # Warm-up
        req = collective.isend(tensor, dst=peer)
        collective.wait(req)

        start = time.perf_counter()
        total = 0
        for _ in range(args.iters):
            req = collective.isend(tensor, dst=peer)
            collective.wait(req)
            total += size

        elapsed = time.perf_counter() - start

        gbps = (total * 8) / elapsed / 1e9
        gb_sec = total / elapsed / 1e9
        print(
            f"[Client Async] {_pretty_size(size):>9} : {gbps:7.2f} Gbps | {gb_sec:7.2f} GB/s"
        )
    print("[Client Async] Benchmark complete")


def parse_size_list(val: str) -> List[int]:
    try:
        return [int(s) for s in val.split(",") if s]
    except ValueError:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers")


def main():
    p = argparse.ArgumentParser(
        description="Benchmark UCCL Collective API bandwidth (GPU only)"
    )
    p.add_argument("--num-cpus", type=int, default=4, help="#CPU threads for RDMA ops")
    p.add_argument(
        "--sizes",
        type=parse_size_list,
        default=[
            256,
            1024,
            4096,
            16384,
            65536,
            262144,
            1048576,
            10485760,
            104857600,
        ],
    )
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument(
        "--async-api",
        action="store_true",
        help="Use async API (isend/irecv/wait)",
    )
    args = p.parse_args()

    # Initialize torch.distributed with gloo backend for coordination
    dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, "This benchmark only supports 2 processes"

    try:
        # Initialize UCCL collective context (local_gpu_idx auto-detected from torch.distributed)
        collective.init_collective(args.num_cpus)

        # Get the actual GPU index used by the collective
        ctx = collective.get_collective()
        local_gpu_idx = ctx.local_gpu_idx
        torch.cuda.set_device(local_gpu_idx)

        print(
            "UCCL Collective Benchmark — role:",
            "client" if rank == 0 else "server",
        )
        print("Message sizes:", ", ".join(_pretty_size(s) for s in args.sizes))
        print(f"Device: GPU | Local GPU idx: {local_gpu_idx} | Iters: {args.iters}")
        if args.async_api:
            print("Using async API (isend/irecv/wait)")

        if args.async_api:
            if rank == 0:
                _run_async_client(args)
            else:
                _run_async_server(args)
        else:
            if rank == 0:
                _run_client(args)
            else:
                _run_server(args)
    finally:
        collective.finalize_collective()
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted] Benchmark aborted by user.")
        sys.exit(1)
