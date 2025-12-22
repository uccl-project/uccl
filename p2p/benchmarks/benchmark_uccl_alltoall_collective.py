"""
Benchmark UCCL AllToAll Collective

Usage:
    torchrun --nnodes=2 --nproc_per_node=8 --node-rank=0 --master_addr=<IP> --master_port=19999 \
        benchmark_uccl_alltoall_collective.py --sizes 1024,4096,16384,65536,262144,1048576 --num-iters 100

    # Single node multi-GPU:
    torchrun --standalone --nproc_per_node=4 benchmark_uccl_alltoall_collective.py --sizes 65536,262144,1048576 --num-iters 100

Environment Variables:
    UCCL_CHUNK_SIZE_KB: Chunk size for UCCL transport (default: 64)
    UCCL_ENTROPY: Entropy setting for UCCL (default: 2)
"""

import argparse
import os
import sys
from typing import List

import torch
import torch.distributed as dist

from util import setup_seed, sync_all
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


def warmup_alltoall_check(
    count_per_rank: int = 1024,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
):
    """
    Warmup and verification for UCCL alltoall operation.
    """
    print("\n🔧 Running UCCL warmup_alltoall_check...")

    if device is None:
        device = torch.cuda.current_device()

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Create send tensor: chunk i is sent to rank i
    send_tensor = torch.zeros(
        (world_size * count_per_rank,),
        dtype=dtype,
        device=device,
    )
    for i in range(world_size):
        send_tensor[i * count_per_rank : (i + 1) * count_per_rank].fill_(
            float(rank * world_size + i + 1)
        )

    # Create recv tensor
    recv_tensor = torch.zeros(
        (world_size * count_per_rank,),
        dtype=dtype,
        device=device,
    )

    # Register tensors for UCCL
    collective.register_tensor(send_tensor)
    collective.register_tensor(recv_tensor)

    sync_all()

    print(
        f"[Rank {rank}] send_tensor: shape={send_tensor.shape}, "
        f"first chunk value={rank * world_size + 1}"
    )

    # Perform UCCL alltoall
    collective.alltoall(send_tensor, recv_tensor)
    torch.cuda.synchronize(device)

    sync_all()

    # Verify results
    all_passed = True
    for src_rank in range(world_size):
        expected_val = float(src_rank * world_size + rank + 1)
        chunk = recv_tensor[src_rank * count_per_rank : (src_rank + 1) * count_per_rank]

        if torch.allclose(chunk, torch.full_like(chunk, expected_val)):
            if rank == 0:
                print(f"[Rank {rank}] chunk from rank {src_rank} PASSED ✓")
        else:
            print(
                f"[Rank {rank}] ERROR: chunk from rank {src_rank} expected {expected_val}, "
                f"got {chunk[:5].tolist()}..."
            )
            all_passed = False

    if rank == 0:
        if all_passed:
            print(f"[Rank {rank}] ✅ UCCL AllToAll verification PASSED")
        else:
            print(f"[Rank {rank}] ❌ UCCL AllToAll verification FAILED")

    # Deregister tensors
    collective.deregister_tensor(send_tensor)
    collective.deregister_tensor(recv_tensor)

    return all_passed


def run_alltoall_benchmark(
    count_per_rank: int,
    num_iters: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> dict:
    """
    Run UCCL alltoall benchmark.

    Args:
        count_per_rank: Number of elements per rank
        num_iters: Number of iterations
        dtype: Data type
        device: CUDA device

    Returns:
        dict with timing and bandwidth metrics
    """
    if device is None:
        device = torch.cuda.current_device()

    world_size = dist.get_world_size()

    # Create tensors
    total_count = world_size * count_per_rank
    send_tensor = torch.randn(total_count, dtype=dtype, device=device)
    recv_tensor = torch.empty(total_count, dtype=dtype, device=device)

    # Register tensors for UCCL
    collective.register_tensor(send_tensor)
    collective.register_tensor(recv_tensor)

    sync_all()

    # Warmup iterations
    for _ in range(5):
        collective.alltoall(send_tensor, recv_tensor)
    torch.cuda.synchronize(device)

    sync_all()

    # Timed iterations using CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        collective.alltoall(send_tensor, recv_tensor)
    end_event.record()

    torch.cuda.synchronize(device)

    # Calculate elapsed time in milliseconds
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_ms / num_iters

    # Calculate data sizes
    element_size = send_tensor.element_size()
    total_bytes = total_count * element_size
    chunk_bytes = count_per_rank * element_size

    # Algorithm bandwidth: total data movement = 2 * (world_size - 1) * chunk_size
    # Each rank sends (world_size - 1) chunks and receives (world_size - 1) chunks
    data_moved = 2 * (world_size - 1) * chunk_bytes
    algbw_gbps = (data_moved / (avg_time_ms / 1000)) / 1e9

    # Bus bandwidth: accounts for network topology efficiency
    busbw_gbps = algbw_gbps * (world_size - 1) / world_size

    # Deregister tensors
    collective.deregister_tensor(send_tensor)
    collective.deregister_tensor(recv_tensor)

    return {
        "count_per_rank": count_per_rank,
        "total_bytes": total_bytes,
        "chunk_bytes": chunk_bytes,
        "avg_time_ms": avg_time_ms,
        "algbw_gbps": algbw_gbps,
        "busbw_gbps": busbw_gbps,
    }


def parse_size_list(val: str) -> List[int]:
    """Parse comma-separated list of sizes."""
    try:
        return [int(s.strip()) for s in val.split(",") if s.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark UCCL AllToAll Collective"
    )
    parser.add_argument(
        "--sizes",
        type=parse_size_list,
        default=[
            1024,
            4096,
            16384,
            65536,
            262144,
            1048576,
            4194304,
            16777216,
        ],
        help="Comma-separated list of chunk sizes in bytes (per rank)",
    )
    parser.add_argument(
        "--num-iters", type=int, default=100, help="Number of iterations"
    )
    parser.add_argument(
        "--num-cpus", type=int, default=4, help="Number of CPU threads for RDMA ops"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type for tensors",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip verification warmup",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default="uccl_alltoall_results.csv",
        help="CSV output file name",
    )
    args = parser.parse_args()

    setup_seed(42)

    # Set device from LOCAL_RANK
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Initialize distributed with gloo for coordination
    dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    dtype = dtype_map[args.dtype]
    element_size = torch.tensor([], dtype=dtype).element_size()

    try:
        # Initialize UCCL collective
        collective.init_collective(args.num_cpus)
        print(f"[Rank {rank}] UCCL Collective initialized successfully")

        dist.barrier()

        if rank == 0:
            print("=" * 70)
            print("UCCL AllToAll Collective Benchmark")
            print("=" * 70)
            print(f"World size: {world_size}")
            print(f"Device: cuda:{local_rank}")
            print(f"Data type: {args.dtype} ({element_size} bytes)")
            print(f"Iterations: {args.num_iters}")
            print(f"CPU threads: {args.num_cpus}")
            print("=" * 70)

        sync_all()

        # Run verification warmup
        if not args.skip_verify:
            verify_count = args.sizes[0] // element_size
            verify_passed = warmup_alltoall_check(
                count_per_rank=verify_count,
                dtype=dtype,
                device=device,
            )
            if not verify_passed:
                print(f"[Rank {rank}] Verification failed, aborting benchmark")
                sys.exit(1)

        sync_all()

        results = []

        # Run benchmarks for each size
        for size_bytes in args.sizes:
            count_per_rank = size_bytes // element_size

            if rank == 0:
                print(
                    f"\n🚀 Running benchmark: chunk_size={_pretty_size(size_bytes)}, "
                    f"total_size={_pretty_size(size_bytes * world_size)}"
                )

            data = run_alltoall_benchmark(
                count_per_rank=count_per_rank,
                num_iters=args.num_iters,
                dtype=dtype,
                device=device,
            )

            results.append(data)

            if rank == 0:
                print(
                    f"   Chunk: {_pretty_size(data['chunk_bytes']):>10} | "
                    f"Total: {_pretty_size(data['total_bytes']):>10} | "
                    f"Time: {data['avg_time_ms']:.3f} ms | "
                    f"AlgBW: {data['algbw_gbps']:.2f} GB/s | "
                    f"BusBW: {data['busbw_gbps']:.2f} GB/s"
                )

            sync_all()

        # Print summary table
        if rank == 0:
            print("\n" + "=" * 85)
            print("Summary (UCCL AllToAll Collective)")
            print("=" * 85)
            print(
                f"{'Chunk Size':>12} | {'Total Size':>12} | "
                f"{'Time (ms)':>10} | {'AlgBW (GB/s)':>12} | {'BusBW (GB/s)':>12}"
            )
            print("-" * 85)
            for r in results:
                print(
                    f"{_pretty_size(r['chunk_bytes']):>12} | "
                    f"{_pretty_size(r['total_bytes']):>12} | "
                    f"{r['avg_time_ms']:>10.3f} | "
                    f"{r['algbw_gbps']:>12.2f} | "
                    f"{r['busbw_gbps']:>12.2f}"
                )
            print("=" * 85)

            # Save to CSV with aligned columns
            csv_file = args.csv_output
            write_header = not os.path.exists(csv_file)

            col_widths = {
                "world_size": 10,
                "dtype": 8,
                "num_iters": 9,
                "backend": 8,
                "chunk_bytes": 12,
                "total_bytes": 12,
                "avg_time_ms": 12,
                "algbw_gbps": 12,
                "busbw_gbps": 12,
            }

            with open(csv_file, mode="a", newline="") as f:
                if write_header:
                    header = ",".join(
                        f"{col:>{col_widths[col]}}" for col in col_widths.keys()
                    )
                    f.write(header + "\n")

                for r in results:
                    row_data = [
                        f"{world_size:>{col_widths['world_size']}}",
                        f"{args.dtype:>{col_widths['dtype']}}",
                        f"{args.num_iters:>{col_widths['num_iters']}}",
                        f"{'uccl':>{col_widths['backend']}}",
                        f"{r['chunk_bytes']:>{col_widths['chunk_bytes']}}",
                        f"{r['total_bytes']:>{col_widths['total_bytes']}}",
                        f"{r['avg_time_ms']:>{col_widths['avg_time_ms']}.4f}",
                        f"{r['algbw_gbps']:>{col_widths['algbw_gbps']}.4f}",
                        f"{r['busbw_gbps']:>{col_widths['busbw_gbps']}.4f}",
                    ]
                    f.write(",".join(row_data) + "\n")
            print(f"✅ Results saved to {csv_file}")

        sync_all()

    finally:
        collective.finalize_collective()
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted] Benchmark aborted by user.")
        sys.exit(1)

