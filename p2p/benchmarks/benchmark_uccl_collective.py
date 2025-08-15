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

from uccl import collective


def _make_buffer(size_bytes: int):
    """Allocate a contiguous GPU tensor of *size_bytes* and return it."""
    n_elems = size_bytes // 4  # float32
    tensor = torch.ones(n_elems, dtype=torch.float32).cuda()
    assert tensor.is_contiguous()
    assert tensor.device.type == "cuda"
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

        # check if tensor is filled with size
        if not tensor.allclose(torch.tensor(size, dtype=torch.int32).cuda()):
            print(f"[Server] WARNING: Tensor is not filled with {size}")
            print(f"[Server] Tensor: {tensor}")
            print(f"[Server] Tensor size: {tensor.size()}")
            print(f"[Server] Tensor dtype: {tensor.dtype}")
            print(f"[Server] Tensor device: {tensor.device}")

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
        tensor.fill_(size)

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


def _run_dual_benchmark(args):
    """Demonstrate dual-direction async communication (both isend and irecv simultaneously)."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size != 2:
        raise RuntimeError("Dual benchmark only supports exactly 2 processes")

    peer = 1 - rank  # peer rank (0 <-> 1)

    for size in args.sizes:
        send_tensor = _make_buffer(size)
        recv_tensor = _make_buffer(size)

        # Register tensors for efficient memory access
        collective.register_tensor(send_tensor)
        collective.register_tensor(recv_tensor)

        # Warm-up: simultaneous send and receive
        send_req = collective.isend(send_tensor, dst=peer)
        recv_req = collective.irecv(recv_tensor, src=peer)
        collective.wait_all([send_req, recv_req])

        start = time.perf_counter()
        total_sent = 0
        total_recv = 0

        for _ in range(args.iters):
            # Start both operations simultaneously
            send_req = collective.isend(send_tensor, dst=peer)
            recv_req = collective.irecv(recv_tensor, src=peer)

            # Wait for both to complete
            collective.wait_all([send_req, recv_req])

            total_sent += size
            total_recv += size

        elapsed = time.perf_counter() - start

        role_name = "Client" if rank == 0 else "Server"
        # Calculate individual send/recv throughput
        send_gbps = (total_sent * 8) / elapsed / 1e9
        send_gb_sec = total_sent / elapsed / 1e9
        recv_gbps = (total_recv * 8) / elapsed / 1e9
        recv_gb_sec = total_recv / elapsed / 1e9

        print(
            f"[{role_name} Dual Send Recv] {_pretty_size(size):>9} : {(send_gbps + recv_gbps) / 2:6.2f} Gbps | {(send_gb_sec + recv_gb_sec) / 2:5.2f} GB/s"
        )

    role_name = "Client" if rank == 0 else "Server"
    print(f"[{role_name} Dual] Benchmark complete")


def _run_ring_benchmark(args):
    """Ring communication: each rank sends to next rank in a ring pattern."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Ring pattern: rank i sends to rank (i+1) % world_size
    dst_rank = (rank + 1) % world_size
    src_rank = (rank - 1 + world_size) % world_size

    print(
        f"[Rank {rank}] Ring pattern: receiving from rank {src_rank}, sending to rank {dst_rank}"
    )

    for size in args.sizes:
        send_tensor = _make_buffer(size)
        recv_tensor = _make_buffer(size)

        # Fill send tensor with rank-specific data for verification
        send_tensor.fill_(size)

        # Register tensors for efficient memory access
        collective.register_tensor(send_tensor)
        collective.register_tensor(recv_tensor)

        # Warm-up
        send_req = collective.isend(send_tensor, dst=dst_rank)
        recv_req = collective.irecv(recv_tensor, src=src_rank)
        collective.wait_all([send_req, recv_req])

        # Verify received data
        expected_value = float(size)
        received_value = recv_tensor[0].item()

        if abs(received_value - expected_value) > 1e-6:
            print(f"[Rank {rank}] WARNING: Data verification failed for warm-up")
            print(f"  Expected: {expected_value}, Received: {received_value}")
            print(f"  Source rank: {src_rank}, Current rank: {rank}")

        # Fill tensor once before benchmark loop for best performance
        benchmark_send_value = float(size)
        send_tensor.fill_(benchmark_send_value)

        start = time.perf_counter()
        total_bytes = 0

        for iteration in range(args.iters):
            # Use async operations for better performance (no fill overhead)
            send_req = collective.isend(send_tensor, dst=dst_rank)
            recv_req = collective.irecv(recv_tensor, src=src_rank)
            collective.wait_all([send_req, recv_req])
            total_bytes += size

        elapsed = time.perf_counter() - start

        # Perform final data verification after all iterations complete
        final_send_value = float(size)
        send_tensor.fill_(final_send_value)

        # Final verification round
        send_req = collective.isend(send_tensor, dst=dst_rank)
        recv_req = collective.irecv(recv_tensor, src=src_rank)
        collective.wait_all([send_req, recv_req])

        # Synchronize all ranks before verification
        dist.barrier()

        # Expected value should be from the source rank's benchmark value
        expected_value = float(final_send_value)
        received_value = recv_tensor[0].item()  # Check first element

        if abs(received_value - expected_value) > 1e-6:
            print(f"[Rank {rank}] WARNING: Final data verification failed")
            print(f"  Expected: {expected_value}, Received: {received_value}")
            print(f"  Source rank: {src_rank}, Current rank: {rank}")
            print(f"  Final send value was: {final_send_value}")
            print(f"  Benchmark send value was: {benchmark_send_value}")

        gbps = (total_bytes * 8) / elapsed / 1e9
        gb_sec = total_bytes / elapsed / 1e9
        print(
            f"[Rank {rank}] Ring Async {_pretty_size(size):>9} : {gbps:7.2f} Gbps | {gb_sec:7.2f} GB/s"
        )

    print(f"[Rank {rank}] Ring async benchmark complete")


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
            16777216,
            104857600,
        ],
    )
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument(
        "--async-api",
        action="store_true",
        help="Use async API (isend/irecv/wait)",
    )
    p.add_argument(
        "--dual",
        action="store_true",
        help="Test bidirectional communication (simultaneous isend and irecv).",
    )
    p.add_argument(
        "--ring",
        action="store_true",
        help="Test ring communication pattern (rank i sends to rank (i+1) % world_size).",
    )
    args = p.parse_args()

    # Initialize torch.distributed with gloo backend for coordination
    dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Check for incompatible options
    if args.dual and args.ring:
        print("ERROR: --dual and --ring options are mutually exclusive")
        sys.exit(1)

    # Validate world size for specific benchmarks
    if args.dual and world_size != 2:
        print("ERROR: --dual benchmark requires exactly 2 processes")
        sys.exit(1)

    if args.ring and world_size < 2:
        print("ERROR: --ring benchmark requires at least 2 processes")
        sys.exit(1)

    # Default client-server benchmark still requires exactly 2 processes
    if not args.dual and not args.ring and world_size != 2:
        print(
            "ERROR: Default client-server benchmark requires exactly 2 processes. Use --ring for multi-process scenarios."
        )
        sys.exit(1)

    try:
        # Initialize UCCL collective context (local_gpu_idx auto-detected from torch.distributed)
        collective.init_collective(args.num_cpus)

        # Get the actual GPU index used by the collective
        ctx = collective.get_collective()
        local_gpu_idx = ctx.local_gpu_idx
        torch.cuda.set_device(local_gpu_idx)

        if args.ring:
            print(f"[Rank {rank}/{world_size}] UCCL Collective Ring Benchmark")
        else:
            print(
                "UCCL Collective Benchmark — role:",
                "client" if rank == 0 else "server",
            )
        print("Message sizes:", ", ".join(_pretty_size(s) for s in args.sizes))
        print(f"Device: GPU | Local GPU idx: {local_gpu_idx} | Iters: {args.iters}")
        if args.ring:
            print(f"[Rank {rank}] Using async ring communication pattern (isend/irecv)")
        elif args.dual:
            print("Using dual-direction mode (simultaneous isend/irecv)")
        elif args.async_api:
            print("Using async API (isend/irecv/wait)")
        else:
            print("Using synchronous API (send/recv)")

        # Synchronize all ranks before starting benchmark
        dist.barrier()

        if args.ring:
            _run_ring_benchmark(args)
        elif args.dual:
            _run_dual_benchmark(args)
        elif args.async_api:
            if rank == 0:
                _run_async_client(args)
            else:
                _run_async_server(args)
        else:
            if rank == 0:
                _run_client(args)
            else:
                _run_server(args)

        # Synchronize all ranks before finishing
        dist.barrier()
        if args.ring:
            print(f"[Rank {rank}] Ring benchmark completed successfully!")
        else:
            print("Benchmark completed successfully!")
    finally:
        collective.finalize_collective()
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted] Benchmark aborted by user.")
        sys.exit(1)
