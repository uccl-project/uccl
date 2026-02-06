from __future__ import annotations
import argparse
import sys
import time
import torch.distributed as dist
import torch
from typing import List
import os

try:
    from uccl import p2p
except ImportError:
    sys.stderr.write("Failed to import p2p\n")
    raise


def _send_bytes(payload: bytes, dst: int):
    """Send bytes via PyTorch distributed."""
    n = len(payload)
    t_size = torch.tensor([n], dtype=torch.int64)
    dist.send(t_size, dst=dst)
    if n == 0:
        return
    buf = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    dist.send(buf, dst=dst)


def _recv_bytes(src: int) -> bytes:
    """Receive bytes via PyTorch distributed."""
    t_size = torch.empty(1, dtype=torch.int64)
    dist.recv(t_size, src=src)
    n = int(t_size.item())
    if n == 0:
        return b""
    buf = torch.empty(n, dtype=torch.uint8)
    dist.recv(buf, src=src)
    return buf.numpy().tobytes()


def _make_buffer(n_bytes: int, device: str, gpu: int = 0):
    assert n_bytes % 4 == 0, "n_bytes must be multiple of 4 for float32"
    n = n_bytes // 4

    if device in ("gpu", "cuda"):
        dev = torch.device(f"cuda:{gpu}")
        return torch.ones(n, dtype=torch.float32, device=dev)
    else:
        pin = torch.cuda.is_available()
        return torch.ones(n, dtype=torch.float32, pin_memory=pin)


def _pretty(num: int):
    """Format bytes to human-readable format."""
    units, val = ["B", "KB", "MB", "GB"], float(num)
    for u in units:
        if val < 1024 or u == units[-1]:
            return f"{val:.0f} {u}" if u == "B" else f"{val:.1f} {u}"
        val /= 1024


def _run_server(args, ep):
    """Server side: receives data via WRITE/READ operations."""
    peer = 0

    # Get local metadata first
    local_metadata = ep.get_metadata()
    print(f"[Server] Got local metadata, size={len(local_metadata)}")

    # Exchange metadata with client (server receives first, then sends)
    remote_metadata = _recv_bytes(src=peer)
    _send_bytes(local_metadata, dst=peer)
    print(f"[Server] Exchanged metadata with client")

    for sz in args.sizes:
        if args.raw:
            # Raw mode: register memory and send descriptors only once
            size_per_block = sz // args.num_iovs
            buf_v = []
            for _ in range(args.num_iovs):
                buf = _make_buffer(size_per_block, args.device, args.local_gpu_idx)
                buf_v.append(buf)

            remote_descs = ep.register_memory(buf_v)

            remote_descs_serialized = ep.get_serialized_descs(remote_descs)
            _send_bytes(remote_descs_serialized, dst=peer)

            # Wait for all iterations to complete
            dist.barrier()
        else:
            # Normal mode: register memory and send descriptors for each iteration
            for iter_idx in range(args.iters + 1):  # +1 for warmup
                # Create new memory buffer for each iteration
                size_per_block = sz // args.num_iovs
                buf_v = []
                for _ in range(args.num_iovs):
                    buf = _make_buffer(size_per_block, args.device, args.local_gpu_idx)
                    buf_v.append(buf)

                # Register remote memory (new memory for each iteration)
                remote_descs = ep.register_memory(buf_v)

                # Serialize and send remote descriptors to client
                remote_descs_serialized = ep.get_serialized_descs(remote_descs)
                _send_bytes(remote_descs_serialized, dst=peer)

                # Wait for transfer to complete
                dist.barrier()

        print(f"[Server] Completed {args.iters} iterations for size {_pretty(sz)}")

    print("[Server] Benchmark complete")


def _run_client(args, ep):
    """Client side: sends data via WRITE/READ operations."""
    peer = 1

    # Exchange metadata with server (only once at the beginning)
    local_metadata = ep.get_metadata()
    _send_bytes(local_metadata, dst=peer)
    remote_metadata = _recv_bytes(src=peer)
    print(f"[Client] Exchanged metadata with server")

    for sz in args.sizes:
        size_per_block = sz // args.num_iovs

        if args.raw:
            # Raw mode: setup once, reuse for all iterations
            buf_v = []
            for _ in range(args.num_iovs):
                buf = _make_buffer(size_per_block, args.device, args.local_gpu_idx)
                buf_v.append(buf)

            local_descs = ep.register_memory(buf_v)
            success, conn_id = ep.add_remote_endpoint(remote_metadata)
            assert success, "Failed to add remote endpoint"

            remote_descs_serialized = _recv_bytes(src=peer)
            remote_descs = ep.deserialize_descs(remote_descs_serialized)

            # Warmup transfer
            success, transfer_id = ep.transfer(
                conn_id, args.mode, local_descs, remote_descs
            )
            assert success, "Failed to start warmup transfer"
            is_done = False
            while not is_done:
                _, is_done = ep.poll_async(transfer_id)

            # Benchmark iterations - reuse everything
            start = time.perf_counter()
            total = 0
            for _ in range(args.iters):
                success, transfer_id = ep.transfer(
                    conn_id, args.mode, local_descs, remote_descs
                )
                assert success, "Failed to start transfer"
                is_done = False
                while not is_done:
                    _, is_done = ep.poll_async(transfer_id)
                total += sz

            elapsed = time.perf_counter() - start
            dist.barrier()
        else:
            # Normal mode: setup for each iteration
            # Warmup iteration
            buf_v = []
            for _ in range(args.num_iovs):
                buf = _make_buffer(size_per_block, args.device, args.local_gpu_idx)
                buf_v.append(buf)

            local_descs = ep.register_memory(buf_v)
            success, conn_id = ep.add_remote_endpoint(remote_metadata)
            assert success, "Failed to add remote endpoint"

            remote_descs_serialized = _recv_bytes(src=peer)
            remote_descs = ep.deserialize_descs(remote_descs_serialized)

            # Warmup transfer
            success, transfer_id = ep.transfer(
                conn_id, args.mode, local_descs, remote_descs
            )
            assert success, "Failed to start warmup transfer"
            is_done = False
            while not is_done:
                _, is_done = ep.poll_async(transfer_id)
            dist.barrier()

            # Benchmark iterations
            start = time.perf_counter()
            total = 0
            for _ in range(args.iters):
                buf_v = []
                for _ in range(args.num_iovs):
                    buf = _make_buffer(size_per_block, args.device, args.local_gpu_idx)
                    buf_v.append(buf)

                local_descs = ep.register_memory(buf_v)
                success, conn_id = ep.add_remote_endpoint(remote_metadata)
                assert success, "Failed to add remote endpoint"

                remote_descs_serialized = _recv_bytes(src=peer)
                remote_descs = ep.deserialize_descs(remote_descs_serialized)

                success, transfer_id = ep.transfer(
                    conn_id, args.mode, local_descs, remote_descs
                )
                assert success, "Failed to start transfer"

                is_done = False
                while not is_done:
                    _, is_done = ep.poll_async(transfer_id)

                total += sz
                dist.barrier()

            elapsed = time.perf_counter() - start

        print(
            f"[Client] {_pretty(sz):>8} : "
            f"{(total * 8) / elapsed / 1e9:6.2f} Gbps | "
            f"{total / elapsed / 1e9:6.2f} GB/s | "
            f"{elapsed / args.iters:6.6f} s"
        )
    print("[Client] Benchmark complete")


def parse_sizes(v: str) -> List[int]:
    """Parse comma-separated size list."""
    try:
        return [int(x) for x in v.split(",") if x]
    except ValueError:
        raise argparse.ArgumentTypeError("bad --sizes")


def main():
    p = argparse.ArgumentParser("UCCL Ray P2P benchmark using transfer API")
    p.add_argument("--local-gpu-idx", type=int, default=0)
    p.add_argument("--num-cpus", type=int, default=4)
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--mode", choices=["read", "write"], default="write")
    p.add_argument(
        "--perf", action="store_true", help="Measure pure transfer performance"
    )
    p.add_argument(
        "--raw",
        action="store_true",
        help="Raw mode: exchange metadata only once, reuse for all iterations to measure pure transfer performance",
    )
    p.add_argument(
        "--sizes",
        type=parse_sizes,
        default=[
            256,
            1024,
            4096,
            16384,
            65536,
            262144,
            1048576,
            10485760,
            67108864,
            104857600,
        ],
        help="Comma-separated list of message sizes in bytes",
    )
    p.add_argument("--iters", type=int, default=10, help="Number of iterations")
    p.add_argument(
        "--num-iovs",
        type=int,
        default=1,
        help="Number of iovs to transfer in a single call",
    )
    args = p.parse_args()

    print("UCCL Ray P2P Benchmark")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Raw mode: {args.raw}")
    print("Sizes:", ", ".join(_pretty(s) for s in args.sizes))
    print(f"Iterations: {args.iters}")
    print(f"Number of IOVs: {args.num_iovs}")
    print("=" * 60)

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, "This benchmark only supports 2 processes"

    ep = p2p.Endpoint(args.local_gpu_idx, args.num_cpus)

    ep.start_passive_accept()

    if rank == 0:
        _run_client(args, ep)
    elif rank == 1:
        _run_server(args, ep)

    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Ctrl-C] Aborted.")
        sys.exit(1)
