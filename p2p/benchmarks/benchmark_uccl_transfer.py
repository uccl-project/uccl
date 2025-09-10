"""
Benchmark UCCL Transfer API

This benchmark demonstrates the transfer API for UCCL P2P engine.
It tests point-to-point transfers between rank 0 (sender) and rank 1 (receiver).
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List

import torch
import torch.distributed as dist

from uccl import transfer


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


def parse_sizes(v: str) -> List[int]:
    """Parse comma-separated sizes like '1MB,4MB,16MB'."""
    try:
        sizes = []
        for s in v.split(","):
            s = s.strip().upper()
            if s.endswith("B"):
                s = s[:-1]

            multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3}

            if s[-1] in multipliers:
                sizes.append(int(s[:-1]) * multipliers[s[-1]])
            else:
                sizes.append(int(s))
        return sizes
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError("bad --sizes")


################################################################################
# Benchmark functions
################################################################################


def _run_receiver(args):
    """Run as receiver (rank 1) - receives data from sender."""
    sender_rank = 0
    local_gpu_idx = dist.get_rank()

    # Create transfer manager as receiver
    gpu_idx_list = [local_gpu_idx]
    zmq_port = 5556  # Receiver port
    manager = transfer.TransferManager(gpu_idx_list, num_cpus=4, zmq_port=zmq_port)

    # Accept connection from sender
    success = manager.accept()
    if not success:
        raise RuntimeError("Failed to accept connection from sender")

    for size in args.sizes:
        # Create receive buffer
        recv_tensor = _make_buffer(size)

        # Register transfer
        transfer_id = manager.register_transfer(local_gpu_idx, sender_rank, recv_tensor)

        # Warm-up receive
        manager.post_transfer_metadata(transfer_id)

        # Benchmark receives
        start = time.perf_counter()
        total = 0

        for _ in range(args.iters):
            manager.post_transfer_metadata(transfer_id)
            total += size

        elapsed = time.perf_counter() - start

        # Check if tensor received data correctly
        if not recv_tensor.allclose(torch.tensor(size, dtype=torch.float32).cuda()):
            print(f"[Receiver] WARNING: Tensor not filled correctly for size {size}")

        gbps = (total * 8) / elapsed / 1e9
        gb_sec = total / elapsed / 1e9
        print(
            f"[Receiver] {_pretty_size(size):>9} : {gbps:7.2f} Gbps | {gb_sec:7.2f} GB/s"
        )

        # Cleanup transfer
        manager.cleanup_transfer(transfer_id)

    print("[Receiver] Benchmark complete")


def _run_sender(args):
    """Run as sender (rank 0) - sends data to receiver."""
    receiver_rank = 1
    local_gpu_idx = dist.get_rank()

    # Create transfer manager as sender
    gpu_idx_list = [local_gpu_idx]
    manager = transfer.TransferManager(gpu_idx_list, num_cpus=4, zmq_port=5555)

    # Connect to receiver
    receiver_ip = "127.0.0.1"  # For local testing
    receiver_zmq_port = 5556
    success = manager.connect(receiver_ip, receiver_zmq_port)
    if not success:
        raise RuntimeError("Failed to connect to receiver")

    for size in args.sizes:
        # Create send buffer with known pattern
        send_tensor = _make_buffer(size)
        send_tensor.fill_(size)  # Fill with size value for verification

        # Register transfer
        transfer_id = manager.register_transfer(
            local_gpu_idx, receiver_rank, send_tensor
        )

        # Warm-up send
        transfer_metadata = manager.fetch_transfer_metadata(transfer_id)
        manager.transfer_tensor(transfer_id, transfer_metadata)

        # Benchmark sends
        start = time.perf_counter()
        total = 0

        for _ in range(args.iters):
            transfer_metadata = manager.fetch_transfer_metadata(transfer_id)
            manager.transfer_tensor(transfer_id, transfer_metadata)
            total += size

        elapsed = time.perf_counter() - start

        gbps = (total * 8) / elapsed / 1e9
        gb_sec = total / elapsed / 1e9
        print(
            f"[Sender] {_pretty_size(size):>9} : {gbps:7.2f} Gbps | {gb_sec:7.2f} GB/s"
        )

        # Cleanup transfer
        manager.cleanup_transfer(transfer_id)

    print("[Sender] Benchmark complete")


def main():
    parser = argparse.ArgumentParser(description="UCCL Transfer API Benchmark")
    parser.add_argument(
        "--sizes",
        type=parse_sizes,
        default="1MB,4MB,16MB,64MB,256MB",
        help="Comma-separated list of message sizes (e.g., 1MB,4MB,16MB)",
    )
    parser.add_argument(
        "--iters", type=int, default=100, help="Number of iterations per size"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        help="PyTorch distributed backend (gloo recommended for metadata exchange)",
    )

    args = parser.parse_args()

    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend=args.backend)

    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if world_size != 2:
            raise RuntimeError(
                "Transfer benchmark requires exactly 2 processes (sender and receiver)"
            )

        local_gpu_idx = rank  # Assume each rank uses its own GPU

        # Print benchmark info
        if rank == 0:
            print("=== UCCL Transfer Benchmark ===")
            print("Message sizes:", ", ".join(_pretty_size(s) for s in args.sizes))
            print(f"Device: GPU | Local GPU idx: {local_gpu_idx} | Iters: {args.iters}")
            print("Transfer pattern: Rank 0 (sender) -> Rank 1 (receiver)")
            print()

        # Synchronize before starting
        dist.barrier()

        # Run benchmark based on rank
        if rank == 0:
            _run_sender(args)
        else:
            _run_receiver(args)

        # Synchronize before finishing
        dist.barrier()
        print("Transfer benchmark completed successfully!")

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted] Benchmark aborted by user.")
        sys.exit(1)
