"""
Benchmark UCCL Collective API

This benchmark demonstrates the high-level collective API for UCCL P2P engine.
It provides an interface similar to NCCL but uses UCCL P2P underneath.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import List

import torch
import torch.distributed as dist

from uccl import collective


@dataclass
class BenchmarkResult:
    """Data class to store benchmark results."""
    role: str
    mode: str
    size_bytes: int
    size_pretty: str
    gbps: float
    gb_sec: float
    iters: int
    rank: int
    world_size: int


def _get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch.dtype."""
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Supported: {list(dtype_map.keys())}")
    return dtype_map[dtype_str]


def _get_dtype_size(dtype: torch.dtype) -> int:
    """Get the size in bytes for a given dtype."""
    if dtype == torch.float32:
        return 4
    elif dtype == torch.bfloat16:
        return 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def _make_buffer(size_bytes: int, dtype: torch.dtype = torch.float32):
    """Allocate a contiguous GPU tensor of *size_bytes* and return it."""
    elem_size = _get_dtype_size(dtype)
    n_elems = size_bytes // elem_size
    tensor = torch.rand(n_elems, dtype=dtype, device="cuda")
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


def _run_server(args) -> List[BenchmarkResult]:
    results = []
    peer = 0  # client rank
    for size in args.sizes:
        tensor = _make_buffer(size, args.tensor_dtype)

        # Register tensor for efficient memory access
        collective.register_tensor(tensor)

        # Warm-up receive
        collective.recv(tensor, src=peer)
        print(tensor)
        start = time.perf_counter()
        total = 0
        for _ in range(args.iters):
            collective.recv(tensor, src=peer)
            total += size

        elapsed = time.perf_counter() - start

        # check if tensor is filled with size
        if not tensor.allclose(torch.tensor(size, dtype=args.tensor_dtype).cuda()):
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
        results.append(BenchmarkResult(
            role="Server",
            mode="sync",
            size_bytes=size,
            size_pretty=_pretty_size(size),
            gbps=gbps,
            gb_sec=gb_sec,
            iters=args.iters,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
        ))
    print("[Server] Benchmark complete")
    return results


def _run_client(args) -> List[BenchmarkResult]:
    results = []
    peer = 1  # server rank
    for size in args.sizes:
        tensor = _make_buffer(size, args.tensor_dtype)
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
        results.append(BenchmarkResult(
            role="Client",
            mode="sync",
            size_bytes=size,
            size_pretty=_pretty_size(size),
            gbps=gbps,
            gb_sec=gb_sec,
            iters=args.iters,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
        ))
    print("[Client] Benchmark complete")
    return results


def _run_async_server(args) -> List[BenchmarkResult]:
    """Demonstrate async API usage."""
    results = []
    peer = 0  # client rank
    for size in args.sizes:
        tensor = _make_buffer(size, args.tensor_dtype)

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
        results.append(BenchmarkResult(
            role="Server",
            mode="async",
            size_bytes=size,
            size_pretty=_pretty_size(size),
            gbps=gbps,
            gb_sec=gb_sec,
            iters=args.iters,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
        ))
    print("[Server Async] Benchmark complete")
    return results


def _run_async_client(args) -> List[BenchmarkResult]:
    """Demonstrate async API usage."""
    results = []
    peer = 1  # server rank
    for size in args.sizes:
        tensor = _make_buffer(size, args.tensor_dtype)

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
        results.append(BenchmarkResult(
            role="Client",
            mode="async",
            size_bytes=size,
            size_pretty=_pretty_size(size),
            gbps=gbps,
            gb_sec=gb_sec,
            iters=args.iters,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
        ))
    print("[Client Async] Benchmark complete")
    return results


def _run_dual_benchmark(args) -> List[BenchmarkResult]:
    """Demonstrate dual-direction async communication (both isend and irecv simultaneously)."""
    results = []
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size != 2:
        raise RuntimeError("Dual benchmark only supports exactly 2 processes")

    peer = 1 - rank  # peer rank (0 <-> 1)

    for size in args.sizes:
        send_tensor = _make_buffer(size, args.tensor_dtype)
        recv_tensor = _make_buffer(size, args.tensor_dtype)

        # Register tensors for efficient memory access
        collective.register_tensor(send_tensor)
        collective.register_tensor(recv_tensor)

        # Warm-up: simultaneous send and receive
        handles = collective.batch_isend_irecv(
            [
                collective.P2POp(collective.isend, send_tensor, peer),
                collective.P2POp(collective.irecv, recv_tensor, peer),
            ]
        )
        collective.wait_all(handles)

        start = time.perf_counter()
        total_sent = 0
        total_recv = 0

        for _ in range(args.iters):
            # Start both operations simultaneously with batching
            handles = collective.batch_isend_irecv(
                [
                    collective.P2POp(collective.isend, send_tensor, peer),
                    collective.P2POp(collective.irecv, recv_tensor, peer),
                ]
            )
            collective.wait_all(handles)

            total_sent += size
            total_recv += size

        elapsed = time.perf_counter() - start

        role_name = "Client" if rank == 0 else "Server"
        # Calculate individual send/recv throughput
        send_gbps = (total_sent * 8) / elapsed / 1e9
        send_gb_sec = total_sent / elapsed / 1e9
        recv_gbps = (total_recv * 8) / elapsed / 1e9
        recv_gb_sec = total_recv / elapsed / 1e9

        avg_gbps = (send_gbps + recv_gbps) / 2
        avg_gb_sec = (send_gb_sec + recv_gb_sec) / 2

        print(
            f"[{role_name} Dual Send Recv] {_pretty_size(size):>9} : {avg_gbps:6.2f} Gbps | {avg_gb_sec:5.2f} GB/s"
        )
        results.append(BenchmarkResult(
            role=role_name,
            mode="dual",
            size_bytes=size,
            size_pretty=_pretty_size(size),
            gbps=avg_gbps,
            gb_sec=avg_gb_sec,
            iters=args.iters,
            rank=rank,
            world_size=world_size,
        ))

    role_name = "Client" if rank == 0 else "Server"
    print(f"[{role_name} Dual] Benchmark complete")
    return results


def _run_ring_benchmark(args) -> List[BenchmarkResult]:
    """Ring communication: each rank sends to next rank in a ring pattern."""
    results = []
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Ring pattern: rank i sends to rank (i+1) % world_size
    dst_rank = (rank + 1) % world_size
    src_rank = (rank - 1 + world_size) % world_size

    print(
        f"[Rank {rank}] Ring pattern: receiving from rank {src_rank}, sending to rank {dst_rank}"
    )

    for size in args.sizes:
        send_tensor = _make_buffer(size, args.tensor_dtype)
        recv_tensor = _make_buffer(size, args.tensor_dtype)

        # Fill send tensor with rank-specific data for verification
        send_tensor.fill_(size)

        # Register tensors for efficient memory access
        collective.register_tensor(send_tensor)
        collective.register_tensor(recv_tensor)

        # Warm-up
        handles = collective.batch_isend_irecv(
            [
                collective.P2POp(collective.isend, send_tensor, dst_rank),
                collective.P2POp(collective.irecv, recv_tensor, src_rank),
            ]
        )
        collective.wait_all(handles)

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
            # Use batched operations for better performance
            handles = collective.batch_isend_irecv(
                [
                    collective.P2POp(collective.isend, send_tensor, dst_rank),
                    collective.P2POp(collective.irecv, recv_tensor, src_rank),
                ]
            )
            collective.wait_all(handles)
            total_bytes += size

        elapsed = time.perf_counter() - start

        # Perform final data verification after all iterations complete
        final_send_value = float(size)
        send_tensor.fill_(final_send_value)

        # Final verification round
        handles = collective.batch_isend_irecv(
            [
                collective.P2POp(collective.isend, send_tensor, dst_rank),
                collective.P2POp(collective.irecv, recv_tensor, src_rank),
            ]
        )
        collective.wait_all(handles)

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
        results.append(BenchmarkResult(
            role=f"Rank{rank}",
            mode="ring",
            size_bytes=size,
            size_pretty=_pretty_size(size),
            gbps=gbps,
            gb_sec=gb_sec,
            iters=args.iters,
            rank=rank,
            world_size=world_size,
        ))

    print(f"[Rank {rank}] Ring async benchmark complete")
    return results


def parse_size_list(val: str) -> List[int]:
    try:
        return [int(s) for s in val.split(",") if s]
    except ValueError:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers")


def save_results_to_csv(results: List[BenchmarkResult], filename: str):
    """Save benchmark results to a CSV file."""
    if not results:
        print(f"No results to save.")
        return

    fieldnames = [
        "role", "mode", "size_bytes", "size_pretty", "gbps", "gb_sec",
        "iters", "rank", "world_size"
    ]

    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "role": result.role,
                "mode": result.mode,
                "size_bytes": result.size_bytes,
                "size_pretty": result.size_pretty,
                "gbps": f"{result.gbps:.4f}",
                "gb_sec": f"{result.gb_sec:.4f}",
                "iters": result.iters,
                "rank": result.rank,
                "world_size": result.world_size,
            })
    print(f"Results saved to {filename}")


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
            104857600*2,
            104857600*3,
            104857600*4,
        ],
    )
    p.add_argument("--iters", type=int, default=10)
    p.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "fp32", "bfloat16", "bf16"],
        help="Data type for tensors (default: float32)",
    )
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
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="Output CSV file path. If not specified, results are not saved to file.",
    )
    args = p.parse_args()

    # Convert dtype string to torch.dtype
    args.tensor_dtype = _get_dtype(args.dtype)

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
        print(f"Device: GPU | Local GPU idx: {local_gpu_idx} | Iters: {args.iters} | Dtype: {args.dtype}")
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

        results: List[BenchmarkResult] = []
        if args.ring:
            results = _run_ring_benchmark(args)
        elif args.dual:
            results = _run_dual_benchmark(args)
        elif args.async_api:
            if rank == 0:
                results = _run_async_client(args)
            else:
                results = _run_async_server(args)
        else:
            if rank == 0:
                results = _run_client(args)
            else:
                results = _run_server(args)

        # Synchronize all ranks before finishing
        dist.barrier()

        # Save results to CSV if output file is specified
        if args.output:
            # For multi-rank scenarios, each rank saves to a separate file
            if world_size > 1:
                base, ext = os.path.splitext(args.output)
                output_file = f"{base}_rank{rank}{ext}" if ext else f"{args.output}_rank{rank}.csv"
            else:
                output_file = args.output
            save_results_to_csv(results, output_file)

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
