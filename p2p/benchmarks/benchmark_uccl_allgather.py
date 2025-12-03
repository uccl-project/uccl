"""
Benchmark UCCL Collective for AllGather

Usage:
    torchrun --nnodes=2 --nproc_per_node=1 --node-rank=0 --master_addr=<IP> --master_port=19999 \
        benchmark_uccl_allgather.py --sizes 1024,4096,16384,65536,262144,1048576 --num-iters 100

    # Single node multi-GPU:
    torchrun --standalone --nproc_per_node=4 benchmark_uccl_allgather.py --sizes 65536,262144,1048576 --num-iters 100
"""

import argparse
import csv
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


def warmup_allgather_check(
    count: int = 4 * 1024,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
):
    """
    Warmup and verification for allgather operation.
    Each rank fills its send buffer with (rank + 1) and verifies
    the gathered result contains all ranks' data.
    """
    print("\n🔧 Running warmup_allgather_check...")

    if device is None:
        device = torch.cuda.current_device()

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Create send tensor filled with rank-specific value
    send_tensor = torch.full(
        (count,),
        fill_value=float(rank + 1),
        dtype=dtype,
        device=device,
    )

    # Create recv tensor (world_size * count elements)
    recv_tensor = torch.zeros(
        (world_size * count,),
        dtype=dtype,
        device=device,
    )

    sync_all()

    # Register tensors for RDMA
    collective.register_tensor(send_tensor)
    collective.register_tensor(recv_tensor)

    print(f"[Rank {rank}] send_tensor: shape={send_tensor.shape}, value={rank + 1}")

    # Perform allgather
    collective.allgather(send_tensor, recv_tensor)

    sync_all()

    # Verify results
    all_passed = True
    for src_rank in range(world_size):
        expected_val = float(src_rank + 1)
        chunk = recv_tensor[src_rank * count : (src_rank + 1) * count]

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
            print(f"[Rank {rank}] ✅ AllGather verification PASSED")
        else:
            print(f"[Rank {rank}] ❌ AllGather verification FAILED")

    # Cleanup
    collective.deregister_tensor(send_tensor)
    collective.deregister_tensor(recv_tensor)

    return all_passed


def run_allgather_sync_benchmark(
    count: int,
    num_iters: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> dict:
    """
    Run synchronous allgather benchmark.

    Args:
        count: Number of elements per rank to gather
        num_iters: Number of iterations
        dtype: Data type
        device: CUDA device

    Returns:
        dict with timing and bandwidth metrics
    """
    if device is None:
        device = torch.cuda.current_device()

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Create tensors
    send_tensor = torch.randn(count, dtype=dtype, device=device)
    recv_tensor = torch.empty(world_size * count, dtype=dtype, device=device)

    # Register tensors
    collective.register_tensor(send_tensor)
    collective.register_tensor(recv_tensor)

    sync_all()

    # Warmup iterations
    for _ in range(5):
        collective.allgather(send_tensor, recv_tensor)

    sync_all()

    # Timed iterations using CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        collective.allgather(send_tensor, recv_tensor)
    end_event.record()

    sync_all()

    # Calculate elapsed time in milliseconds
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_ms / num_iters

    # Calculate data sizes
    element_size = send_tensor.element_size()
    send_bytes = count * element_size
    total_recv_bytes = world_size * send_bytes

    # Algorithm bandwidth: total data in output / time
    algbw_gbps = (total_recv_bytes / (avg_time_ms / 1000)) / 1e9

    # Bus bandwidth for allgather: algbw * (n-1)/n
    # This accounts for the ring algorithm efficiency
    busbw_gbps = algbw_gbps * (world_size - 1) / world_size

    # Cleanup
    collective.deregister_tensor(send_tensor)
    collective.deregister_tensor(recv_tensor)

    return {
        "count": count,
        "send_bytes": send_bytes,
        "recv_bytes": total_recv_bytes,
        "avg_time_ms": avg_time_ms,
        "algbw_gbps": algbw_gbps,
        "busbw_gbps": busbw_gbps,
    }


def run_allgather_async_benchmark(
    count: int,
    num_iters: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> dict:
    """
    Run asynchronous allgather benchmark using iallgather.

    Args:
        count: Number of elements per rank to gather
        num_iters: Number of iterations
        dtype: Data type
        device: CUDA device

    Returns:
        dict with timing and bandwidth metrics
    """
    if device is None:
        device = torch.cuda.current_device()

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Create tensors
    send_tensor = torch.randn(count, dtype=dtype, device=device)
    recv_tensor = torch.empty(world_size * count, dtype=dtype, device=device)

    # Register tensors
    collective.register_tensor(send_tensor)
    collective.register_tensor(recv_tensor)

    sync_all()

    # Warmup iterations
    for _ in range(5):
        transfer_ids = collective.iallgather(send_tensor, recv_tensor)
        collective.wait_all(transfer_ids)

    sync_all()

    # Timed iterations using CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        transfer_ids = collective.iallgather(send_tensor, recv_tensor)
        collective.wait_all(transfer_ids)
    end_event.record()

    sync_all()

    # Calculate elapsed time in milliseconds
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_ms / num_iters

    # Calculate data sizes
    element_size = send_tensor.element_size()
    send_bytes = count * element_size
    total_recv_bytes = world_size * send_bytes

    # Algorithm bandwidth: total data in output / time
    algbw_gbps = (total_recv_bytes / (avg_time_ms / 1000)) / 1e9

    # Bus bandwidth for allgather: algbw * (n-1)/n
    busbw_gbps = algbw_gbps * (world_size - 1) / world_size

    # Cleanup
    collective.deregister_tensor(send_tensor)
    collective.deregister_tensor(recv_tensor)

    return {
        "count": count,
        "send_bytes": send_bytes,
        "recv_bytes": total_recv_bytes,
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
        description="Benchmark UCCL Collective AllGather API"
    )
    parser.add_argument(
        "--num-cpus", type=int, default=4, help="Number of CPU threads for RDMA ops"
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
        help="Comma-separated list of message sizes in bytes",
    )
    parser.add_argument(
        "--num-iters", type=int, default=100, help="Number of iterations"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type for tensors",
    )
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Use async iallgather API instead of sync allgather",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip verification warmup",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default="uccl_allgather_results.csv",
        help="CSV output file name",
    )
    args = parser.parse_args()

    setup_seed(42)

    # Set device from LOCAL_RANK
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Initialize distributed
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

    try:
        # Initialize UCCL collective
        collective.init_collective(args.num_cpus)

        if rank == 0:
            print("=" * 60)
            print("UCCL AllGather Benchmark")
            print("=" * 60)
            print(f"World size: {world_size}")
            print(f"Device: cuda:{local_rank}")
            print(f"Data type: {args.dtype} ({element_size} bytes)")
            print(f"Iterations: {args.num_iters}")
            print(
                f"Mode: {'async (iallgather)' if args.use_async else 'sync (allgather)'}"
            )
            print("=" * 60)

        dist.barrier()

        # Run verification warmup
        if not args.skip_verify:
            # Convert first size to element count for verification
            verify_count = args.sizes[0] // element_size
            verify_passed = warmup_allgather_check(
                count=verify_count,
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
            count = size_bytes // element_size

            if rank == 0:
                print(
                    f"\n🚀 Running benchmark: size={_pretty_size(size_bytes)}, count={count}"
                )

            if args.use_async:
                data = run_allgather_async_benchmark(
                    count=count,
                    num_iters=args.num_iters,
                    dtype=dtype,
                    device=device,
                )
            else:
                data = run_allgather_sync_benchmark(
                    count=count,
                    num_iters=args.num_iters,
                    dtype=dtype,
                    device=device,
                )

            results.append(data)

            if rank == 0:
                print(
                    f"   Send: {_pretty_size(data['send_bytes']):>10} | "
                    f"Recv: {_pretty_size(data['recv_bytes']):>10} | "
                    f"Time: {data['avg_time_ms']:.3f} ms | "
                    f"AlgBW: {data['algbw_gbps']:.2f} GB/s | "
                    f"BusBW: {data['busbw_gbps']:.2f} GB/s"
                )

            sync_all()

        # Print summary table
        if rank == 0:
            print("\n" + "=" * 80)
            print("Summary")
            print("=" * 80)
            print(
                f"{'Size':>12} | {'Send':>10} | {'Recv':>10} | "
                f"{'Time (ms)':>10} | {'AlgBW (GB/s)':>12} | {'BusBW (GB/s)':>12}"
            )
            print("-" * 80)
            for r in results:
                print(
                    f"{_pretty_size(r['send_bytes']):>12} | "
                    f"{_pretty_size(r['send_bytes']):>10} | "
                    f"{_pretty_size(r['recv_bytes']):>10} | "
                    f"{r['avg_time_ms']:>10.3f} | "
                    f"{r['algbw_gbps']:>12.2f} | "
                    f"{r['busbw_gbps']:>12.2f}"
                )
            print("=" * 80)

            # Save to CSV
            csv_file = args.csv_output
            write_header = not os.path.exists(csv_file)

            with open(csv_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(
                        [
                            "world_size",
                            "dtype",
                            "num_iters",
                            "mode",
                            "send_bytes",
                            "recv_bytes",
                            "avg_time_ms",
                            "algbw_gbps",
                            "busbw_gbps",
                        ]
                    )

                for r in results:
                    writer.writerow(
                        [
                            world_size,
                            args.dtype,
                            args.num_iters,
                            "async" if args.use_async else "sync",
                            r["send_bytes"],
                            r["recv_bytes"],
                            f"{r['avg_time_ms']:.4f}",
                            f"{r['algbw_gbps']:.4f}",
                            f"{r['busbw_gbps']:.4f}",
                        ]
                    )
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
