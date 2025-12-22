"""
Unified AllToAll Benchmark - Compares NCCL vs UCCL with identical methodology.

This benchmark ensures fair comparison by:
1. Using the same tensor sizes and data types
2. Using the same warmup and iteration counts
3. Using identical timing methodology (CUDA events)
4. Running both benchmarks in the same process

Usage:
    torchrun --nnodes=2 --nproc_per_node=8 --node-rank=0 --master_addr=<IP> --master_port=29950 \
        benchmark_alltoall_comparison.py --sizes 1024,4096,16384,65536,262144,1048576 --num-iters 100

    # Single node multi-GPU:
    torchrun --standalone --nproc_per_node=4 benchmark_alltoall_comparison.py
"""

import argparse
import os
import sys
from typing import List

import torch
import torch.distributed as dist

from util import setup_seed
from pynccl import PyNcclCommunicator
from uccl import collective


def _pretty_size(num_bytes: int) -> str:
    """Format bytes into human-readable string."""
    units = ["B", "KB", "MB", "GB"]
    val = float(num_bytes)
    for u in units:
        if val < 1024 or u == units[-1]:
            return f"{val:.0f} {u}" if u == "B" else f"{val:.1f} {u}"
        val /= 1024
    return f"{num_bytes} B"


def sync_all(device):
    """Synchronize all processes and CUDA."""
    dist.barrier()
    torch.cuda.synchronize(device)


def run_nccl_alltoall(
    comm: PyNcclCommunicator,
    send_tensor: torch.Tensor,
    recv_tensor: torch.Tensor,
    num_iters: int,
    warmup_iters: int,
    device: torch.device,
) -> float:
    """
    Run NCCL alltoall benchmark.
    Returns average time in milliseconds.
    """
    # Warmup
    for _ in range(warmup_iters):
        comm.all_to_all_single(send_tensor, recv_tensor)
    torch.cuda.synchronize(device)

    # Timed iterations
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        comm.all_to_all_single(send_tensor, recv_tensor)
    end_event.record()

    torch.cuda.synchronize(device)
    elapsed_ms = start_event.elapsed_time(end_event)
    return elapsed_ms / num_iters


def run_uccl_alltoall(
    send_tensor: torch.Tensor,
    recv_tensor: torch.Tensor,
    num_iters: int,
    warmup_iters: int,
    device: torch.device,
) -> float:
    """
    Run UCCL alltoall benchmark.
    Returns average time in milliseconds.
    """
    # Warmup
    for _ in range(warmup_iters):
        collective.alltoall(send_tensor, recv_tensor)
    torch.cuda.synchronize(device)

    # Timed iterations
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        collective.alltoall(send_tensor, recv_tensor)
    end_event.record()

    torch.cuda.synchronize(device)
    elapsed_ms = start_event.elapsed_time(end_event)
    return elapsed_ms / num_iters


def calculate_bandwidth(
    avg_time_ms: float,
    count_per_rank: int,
    element_size: int,
    world_size: int,
) -> tuple:
    """Calculate algorithm and bus bandwidth."""
    chunk_bytes = count_per_rank * element_size
    # AllToAll: each rank sends (n-1) chunks and receives (n-1) chunks
    data_moved = 2 * (world_size - 1) * chunk_bytes
    algbw_gbps = (data_moved / (avg_time_ms / 1000)) / 1e9
    busbw_gbps = algbw_gbps * (world_size - 1) / world_size
    return algbw_gbps, busbw_gbps


def parse_size_list(val: str) -> List[int]:
    """Parse comma-separated list of sizes."""
    try:
        return [int(s.strip()) for s in val.split(",") if s.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers")


def main():
    parser = argparse.ArgumentParser(
        description="Unified AllToAll Benchmark - NCCL vs UCCL comparison"
    )
    parser.add_argument(
        "--num-cpus", type=int, default=4, help="Number of CPU threads for UCCL RDMA"
    )
    parser.add_argument(
        "--sizes",
        type=parse_size_list,
        default=[1024, 4096, 16384, 65536, 262144, 1048576, 4194304],
        help="Comma-separated list of chunk sizes in bytes (per rank)",
    )
    parser.add_argument(
        "--num-iters", type=int, default=100, help="Number of timed iterations"
    )
    parser.add_argument(
        "--warmup-iters", type=int, default=10, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type for tensors",
    )
    parser.add_argument(
        "--nccl-only", action="store_true", help="Run only NCCL benchmark"
    )
    parser.add_argument(
        "--uccl-only", action="store_true", help="Run only UCCL benchmark"
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default="alltoall_comparison.csv",
        help="CSV output file name",
    )
    args = parser.parse_args()

    setup_seed(42)

    # Set device from LOCAL_RANK
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Initialize distributed with gloo for coordination
    dist.init_process_group(backend="gloo", device_id=device)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    dtype = dtype_map[args.dtype]
    element_size = torch.tensor([], dtype=dtype).element_size()

    run_nccl = not args.uccl_only
    run_uccl = not args.nccl_only

    # Initialize communicators
    nccl_comm = None
    if run_nccl:
        nccl_comm = PyNcclCommunicator(group=dist.group.WORLD, device=device)

    if run_uccl:
        collective.init_collective(args.num_cpus)

    try:
        if rank == 0:
            print("=" * 90)
            print("Unified AllToAll Benchmark - NCCL vs UCCL Comparison")
            print("=" * 90)
            print(f"World size: {world_size}")
            print(f"Device: cuda:{local_rank}")
            print(f"Data type: {args.dtype} ({element_size} bytes)")
            print(f"Warmup iterations: {args.warmup_iters}")
            print(f"Timed iterations: {args.num_iters}")
            print(f"Backends: {'NCCL' if run_nccl else ''} {'UCCL' if run_uccl else ''}")
            print("=" * 90)

        sync_all(device)

        results = []

        for size_bytes in args.sizes:
            count_per_rank = size_bytes // element_size
            total_count = world_size * count_per_rank
            chunk_bytes = count_per_rank * element_size
            total_bytes = total_count * element_size

            if rank == 0:
                print(
                    f"\n🚀 Chunk: {_pretty_size(chunk_bytes)}, "
                    f"Total: {_pretty_size(total_bytes)}"
                )

            # Create tensors - same for both
            send_tensor = torch.randn(total_count, dtype=dtype, device=device)
            recv_tensor = torch.empty(total_count, dtype=dtype, device=device)

            # Register tensors for UCCL (required for RDMA)
            if run_uccl:
                collective.register_tensor(send_tensor)
                collective.register_tensor(recv_tensor)

            sync_all(device)

            result = {
                "chunk_bytes": chunk_bytes,
                "total_bytes": total_bytes,
            }

            # Run NCCL benchmark
            if run_nccl:
                nccl_time = run_nccl_alltoall(
                    nccl_comm,
                    send_tensor,
                    recv_tensor,
                    args.num_iters,
                    args.warmup_iters,
                    device,
                )
                nccl_algbw, nccl_busbw = calculate_bandwidth(
                    nccl_time, count_per_rank, element_size, world_size
                )
                result["nccl_time_ms"] = nccl_time
                result["nccl_algbw"] = nccl_algbw
                result["nccl_busbw"] = nccl_busbw

                if rank == 0:
                    print(
                        f"   NCCL: {nccl_time:.4f} ms | "
                        f"AlgBW: {nccl_algbw:.2f} GB/s | "
                        f"BusBW: {nccl_busbw:.2f} GB/s"
                    )

            sync_all(device)

            # Run UCCL benchmark
            if run_uccl:
                uccl_time = run_uccl_alltoall(
                    send_tensor,
                    recv_tensor,
                    args.num_iters,
                    args.warmup_iters,
                    device,
                )
                uccl_algbw, uccl_busbw = calculate_bandwidth(
                    uccl_time, count_per_rank, element_size, world_size
                )
                result["uccl_time_ms"] = uccl_time
                result["uccl_algbw"] = uccl_algbw
                result["uccl_busbw"] = uccl_busbw

                if rank == 0:
                    print(
                        f"   UCCL: {uccl_time:.4f} ms | "
                        f"AlgBW: {uccl_algbw:.2f} GB/s | "
                        f"BusBW: {uccl_busbw:.2f} GB/s"
                    )

            # Calculate speedup if both ran
            if run_nccl and run_uccl and rank == 0:
                speedup = nccl_time / uccl_time if uccl_time > 0 else 0
                result["speedup"] = speedup
                if speedup > 1:
                    print(f"   📈 UCCL is {speedup:.2f}x faster than NCCL")
                else:
                    print(f"   📉 NCCL is {1/speedup:.2f}x faster than UCCL")

            results.append(result)

            # Cleanup UCCL registration
            if run_uccl:
                collective.deregister_tensor(send_tensor)
                collective.deregister_tensor(recv_tensor)

            sync_all(device)

        # Print summary
        if rank == 0:
            print("\n" + "=" * 100)
            print("Summary")
            print("=" * 100)

            if run_nccl and run_uccl:
                print(
                    f"{'Chunk':>10} | {'Total':>10} | "
                    f"{'NCCL(ms)':>10} | {'NCCL BW':>10} | "
                    f"{'UCCL(ms)':>10} | {'UCCL BW':>10} | {'Speedup':>8}"
                )
                print("-" * 100)
                for r in results:
                    speedup_str = (
                        f"{r.get('speedup', 0):.2f}x"
                        if r.get("speedup", 0) >= 1
                        else f"{1/r.get('speedup', 1):.2f}x (NCCL)"
                    )
                    print(
                        f"{_pretty_size(r['chunk_bytes']):>10} | "
                        f"{_pretty_size(r['total_bytes']):>10} | "
                        f"{r.get('nccl_time_ms', 0):>10.4f} | "
                        f"{r.get('nccl_busbw', 0):>9.2f}G | "
                        f"{r.get('uccl_time_ms', 0):>10.4f} | "
                        f"{r.get('uccl_busbw', 0):>9.2f}G | "
                        f"{speedup_str:>8}"
                    )
            else:
                backend = "NCCL" if run_nccl else "UCCL"
                print(
                    f"{'Chunk':>10} | {'Total':>10} | "
                    f"{'Time(ms)':>10} | {'AlgBW(GB/s)':>12} | {'BusBW(GB/s)':>12}"
                )
                print("-" * 70)
                for r in results:
                    time_key = "nccl_time_ms" if run_nccl else "uccl_time_ms"
                    algbw_key = "nccl_algbw" if run_nccl else "uccl_algbw"
                    busbw_key = "nccl_busbw" if run_nccl else "uccl_busbw"
                    print(
                        f"{_pretty_size(r['chunk_bytes']):>10} | "
                        f"{_pretty_size(r['total_bytes']):>10} | "
                        f"{r.get(time_key, 0):>10.4f} | "
                        f"{r.get(algbw_key, 0):>12.2f} | "
                        f"{r.get(busbw_key, 0):>12.2f}"
                    )

            print("=" * 100)

            # Save to CSV
            csv_file = args.csv_output
            write_header = not os.path.exists(csv_file)

            with open(csv_file, mode="a", newline="") as f:
                if write_header:
                    headers = [
                        "world_size",
                        "dtype",
                        "num_iters",
                        "chunk_bytes",
                        "total_bytes",
                    ]
                    if run_nccl:
                        headers.extend(["nccl_time_ms", "nccl_algbw", "nccl_busbw"])
                    if run_uccl:
                        headers.extend(["uccl_time_ms", "uccl_algbw", "uccl_busbw"])
                    if run_nccl and run_uccl:
                        headers.append("speedup")
                    f.write(",".join(headers) + "\n")

                for r in results:
                    row = [
                        str(world_size),
                        args.dtype,
                        str(args.num_iters),
                        str(r["chunk_bytes"]),
                        str(r["total_bytes"]),
                    ]
                    if run_nccl:
                        row.extend(
                            [
                                f"{r.get('nccl_time_ms', 0):.4f}",
                                f"{r.get('nccl_algbw', 0):.4f}",
                                f"{r.get('nccl_busbw', 0):.4f}",
                            ]
                        )
                    if run_uccl:
                        row.extend(
                            [
                                f"{r.get('uccl_time_ms', 0):.4f}",
                                f"{r.get('uccl_algbw', 0):.4f}",
                                f"{r.get('uccl_busbw', 0):.4f}",
                            ]
                        )
                    if run_nccl and run_uccl:
                        row.append(f"{r.get('speedup', 0):.4f}")
                    f.write(",".join(row) + "\n")

            print(f"✅ Results saved to {csv_file}")

    finally:
        if run_uccl:
            collective.finalize_collective()
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted] Benchmark aborted by user.")
        sys.exit(1)

