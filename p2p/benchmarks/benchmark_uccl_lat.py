from __future__ import annotations

import argparse
import sys
import time
import os
from typing import List

import numpy as np
import torch
import torch.distributed as dist

# Only RC mode is supported for read/write operations.
os.environ["UCCL_RCMODE"] = "1"

try:
    from uccl import p2p
except ImportError:
    sys.stderr.write("Failed to import p2p\n")
    raise

FIFO_BLOB_SIZE = 64  # bytes


def _make_buffer(n_bytes: int, device: str, gpu: int):
    """Allocate a contiguous buffer of n_bytes and return (buffer, ptr)."""
    n = max(n_bytes // 4, 1)
    if device == "gpu":
        buf = torch.ones(n, dtype=torch.float32, device=f"cuda:{gpu}")
        ptr = buf.data_ptr()
    else:
        buf = torch.ones(n, dtype=torch.float32, pin_memory=True)
        ptr = buf.data_ptr()
    return buf, ptr


def _pretty_size(num_bytes: int) -> str:
    """Format byte size in human-readable form."""
    units = ["B", "KB", "MB", "GB"]
    val = float(num_bytes)
    for u in units:
        if val < 1024 or u == units[-1]:
            return f"{val:.0f} {u}" if u == "B" else f"{val:.1f} {u}"
        val /= 1024
    return f"{num_bytes} B"


def _compute_percentiles(latencies: List[float]) -> dict:
    """Compute P50 and P99 latencies from a list of latency measurements."""
    arr = np.array(latencies)
    return {
        "p50": np.percentile(arr, 50),
        "p99": np.percentile(arr, 99),
        "avg": np.mean(arr),
        "min": np.min(arr),
        "max": np.max(arr),
    }


def _print_latency_stats(role: str, size: int, stats: dict, mode: str):
    """Print latency statistics in a formatted way."""
    p50_us = stats["p50"] * 1e6  # Convert to microseconds
    p99_us = stats["p99"] * 1e6
    avg_us = stats["avg"] * 1e6
    min_us = stats["min"] * 1e6
    max_us = stats["max"] * 1e6
    print(
        f"[{role}] {_pretty_size(size):>8} ({mode:>5}): "
        f"P50={p50_us:8.2f} us | P99={p99_us:8.2f} us | "
        f"Avg={avg_us:8.2f} us | Min={min_us:8.2f} us | Max={max_us:8.2f} us"
    )


# ============================================================================
# Send/Recv Latency Benchmark
# ============================================================================


def _run_sendrecv_server(args, ep, remote_metadata):
    """Server side for send/recv latency benchmark - receives data."""
    ok, r_ip, r_gpu, conn_id = ep.accept()
    assert ok, "[Server] Failed to accept RDMA connection"
    print(f"[Server] Accept from {r_ip} (GPU {r_gpu}) conn_id={conn_id}")

    for size in args.sizes:
        buf, ptr = _make_buffer(size, args.device, args.local_gpu_idx)
        ok, mr_id = ep.reg(ptr, size)
        assert ok, "[Server] register failed"

        # Warmup
        if args.async_api:
            ok, transfer_id = ep.recv_async(conn_id, mr_id, ptr, size)
            assert ok, "[Server] recv_async error"
            is_done = False
            while not is_done:
                ok, is_done = ep.poll_async(transfer_id)
                assert ok, "[Server] poll_async error"
        else:
            ok = ep.recv(conn_id, mr_id, ptr, size)
            assert ok, "[Server] recv error"

        # Timed iterations
        latencies = []
        for _ in range(args.iters):
            start = time.perf_counter()
            if args.async_api:
                ok, transfer_id = ep.recv_async(conn_id, mr_id, ptr, size)
                assert ok, "[Server] recv_async error"
                is_done = False
                while not is_done:
                    ok, is_done = ep.poll_async(transfer_id)
                    assert ok, "[Server] poll_async error"
            else:
                ok = ep.recv(conn_id, mr_id, ptr, size)
                assert ok, "[Server] recv error"
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

        stats = _compute_percentiles(latencies)
        _print_latency_stats("Server", size, stats, "recv")

    print("[Server] Send/Recv benchmark complete")


def _run_sendrecv_client(args, ep, remote_metadata):
    """Client side for send/recv latency benchmark - sends data."""
    ip, port, r_gpu = p2p.Endpoint.parse_metadata(remote_metadata)
    ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
    assert ok, "[Client] Failed to connect to server"
    print(f"[Client] Connected to {ip}:{port} (GPU {r_gpu}) conn_id={conn_id}")

    for size in args.sizes:
        buf, ptr = _make_buffer(size, args.device, args.local_gpu_idx)
        ok, mr_id = ep.reg(ptr, size)
        assert ok, "[Client] register failed"

        # Warmup
        if args.async_api:
            ok, transfer_id = ep.send_async(conn_id, mr_id, ptr, size)
            assert ok, "[Client] send_async error"
            is_done = False
            while not is_done:
                ok, is_done = ep.poll_async(transfer_id)
                assert ok, "[Client] poll_async error"
        else:
            ok = ep.send(conn_id, mr_id, ptr, size)
            assert ok, "[Client] send error"

        # Timed iterations
        latencies = []
        for _ in range(args.iters):
            start = time.perf_counter()
            if args.async_api:
                ok, transfer_id = ep.send_async(conn_id, mr_id, ptr, size)
                assert ok, "[Client] send_async error"
                is_done = False
                while not is_done:
                    ok, is_done = ep.poll_async(transfer_id)
                    assert ok, "[Client] poll_async error"
            else:
                ok = ep.send(conn_id, mr_id, ptr, size)
                assert ok, "[Client] send error"
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

        stats = _compute_percentiles(latencies)
        _print_latency_stats("Client", size, stats, "send")

    print("[Client] Send/Recv benchmark complete")


# ============================================================================
# Read Latency Benchmark
# ============================================================================


def _run_read_server(args, ep, remote_metadata):
    """Server side for read latency benchmark - advertises memory."""
    peer = 0
    ok, r_ip, r_gpu, conn_id = ep.accept()
    assert ok, "[Server] Failed to accept RDMA connection"
    print(f"[Server] Accept from {r_ip} (GPU {r_gpu}) conn_id={conn_id}")

    for size in args.sizes:
        buf, ptr = _make_buffer(size, args.device, args.local_gpu_idx)
        ok, mr_id = ep.reg(ptr, size)
        assert ok, "[Server] register failed"

        # Advertise memory for read (use advertisev with single element)
        ok, fifo_blob_v = ep.advertisev(conn_id, [mr_id], [ptr], [size], 1)
        assert ok, "[Server] advertisev failed"
        assert len(fifo_blob_v) == 1
        assert len(fifo_blob_v[0]) == FIFO_BLOB_SIZE

        # Send fifo_blob to client
        dist.send(torch.ByteTensor(list(fifo_blob_v[0])), dst=peer)
        dist.barrier()

    print("[Server] Read benchmark complete")


def _run_read_client(args, ep, remote_metadata):
    """Client side for read latency benchmark - performs RDMA reads."""
    peer = 1
    ip, port, r_gpu = p2p.Endpoint.parse_metadata(remote_metadata)
    ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
    assert ok, "[Client] Failed to connect to server"
    print(f"[Client] Connected to {ip}:{port} (GPU {r_gpu}) conn_id={conn_id}")

    for size in args.sizes:
        buf, ptr = _make_buffer(size, args.device, args.local_gpu_idx)
        ok, mr_id = ep.reg(ptr, size)
        assert ok, "[Client] register failed"

        # Receive fifo_blob from server
        fifo_blob_tensor = torch.zeros(FIFO_BLOB_SIZE, dtype=torch.uint8)
        dist.recv(fifo_blob_tensor, src=peer)
        fifo_blob = bytes(fifo_blob_tensor.tolist())

        # Wrap in lists for API calls
        mr_id_v = [mr_id]
        ptr_v = [ptr]
        size_v = [size]
        fifo_blob_v = [fifo_blob]

        # Warmup
        if args.async_api:
            ok, transfer_id = ep.read_async(conn_id, mr_id, ptr, size, fifo_blob)
            assert ok, "[Client] read_async error"
            is_done = False
            while not is_done:
                ok, is_done = ep.poll_async(transfer_id)
                assert ok, "[Client] poll_async error"
        else:
            ep.readv(conn_id, mr_id_v, ptr_v, size_v, fifo_blob_v, 1)

        # Timed iterations
        latencies = []
        for _ in range(args.iters):
            start = time.perf_counter()
            if args.async_api:
                ok, transfer_id = ep.read_async(conn_id, mr_id, ptr, size, fifo_blob)
                assert ok, "[Client] read_async error"
                is_done = False
                while not is_done:
                    ok, is_done = ep.poll_async(transfer_id)
                    assert ok, "[Client] poll_async error"
            else:
                ep.readv(conn_id, mr_id_v, ptr_v, size_v, fifo_blob_v, 1)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

        stats = _compute_percentiles(latencies)
        _print_latency_stats("Client", size, stats, "read")
        dist.barrier()

    print("[Client] Read benchmark complete")


# ============================================================================
# Write Latency Benchmark
# ============================================================================


def _run_write_server(args, ep, remote_metadata):
    """Server side for write latency benchmark - advertises memory."""
    peer = 0
    ok, r_ip, r_gpu, conn_id = ep.accept()
    assert ok, "[Server] Failed to accept RDMA connection"
    print(f"[Server] Accept from {r_ip} (GPU {r_gpu}) conn_id={conn_id}")

    for size in args.sizes:
        buf, ptr = _make_buffer(size, args.device, args.local_gpu_idx)
        ok, mr_id = ep.reg(ptr, size)
        assert ok, "[Server] register failed"

        # Advertise memory for write (use advertisev with single element)
        ok, fifo_blob_v = ep.advertisev(conn_id, [mr_id], [ptr], [size], 1)
        assert ok, "[Server] advertisev failed"
        assert len(fifo_blob_v) == 1
        assert len(fifo_blob_v[0]) == FIFO_BLOB_SIZE

        # Send fifo_blob to client
        dist.send(torch.ByteTensor(list(fifo_blob_v[0])), dst=peer)
        dist.barrier()

    print("[Server] Write benchmark complete")


def _run_write_client(args, ep, remote_metadata):
    """Client side for write latency benchmark - performs RDMA writes."""
    peer = 1
    ip, port, r_gpu = p2p.Endpoint.parse_metadata(remote_metadata)
    ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
    assert ok, "[Client] Failed to connect to server"
    print(f"[Client] Connected to {ip}:{port} (GPU {r_gpu}) conn_id={conn_id}")

    for size in args.sizes:
        buf, ptr = _make_buffer(size, args.device, args.local_gpu_idx)
        ok, mr_id = ep.reg(ptr, size)
        assert ok, "[Client] register failed"

        # Receive fifo_blob from server
        fifo_blob_tensor = torch.zeros(FIFO_BLOB_SIZE, dtype=torch.uint8)
        dist.recv(fifo_blob_tensor, src=peer)
        fifo_blob = bytes(fifo_blob_tensor.tolist())

        # Wrap in lists for API calls
        mr_id_v = [mr_id]
        ptr_v = [ptr]
        size_v = [size]
        fifo_blob_v = [fifo_blob]

        # Warmup
        if args.async_api:
            ok, transfer_id = ep.write_async(conn_id, mr_id, ptr, size, fifo_blob)
            assert ok, "[Client] write_async error"
            is_done = False
            while not is_done:
                ok, is_done = ep.poll_async(transfer_id)
                assert ok, "[Client] poll_async error"
        else:
            ep.writev(conn_id, mr_id_v, ptr_v, size_v, fifo_blob_v, 1)

        # Timed iterations
        latencies = []
        for _ in range(args.iters):
            start = time.perf_counter()
            if args.async_api:
                ok, transfer_id = ep.write_async(conn_id, mr_id, ptr, size, fifo_blob)
                assert ok, "[Client] write_async error"
                is_done = False
                while not is_done:
                    ok, is_done = ep.poll_async(transfer_id)
                    assert ok, "[Client] poll_async error"
            else:
                ep.writev(conn_id, mr_id_v, ptr_v, size_v, fifo_blob_v, 1)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

        stats = _compute_percentiles(latencies)
        _print_latency_stats("Client", size, stats, "write")
        dist.barrier()

    print("[Client] Write benchmark complete")


# ============================================================================
# Main
# ============================================================================


def parse_size_list(val: str) -> List[int]:
    try:
        return [int(s) for s in val.split(",") if s]
    except ValueError:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers")


def main():
    p = argparse.ArgumentParser(
        description="UCCL P2P Latency Benchmark - measures P50/P99 latencies"
    )
    p.add_argument(
        "--mode",
        choices=["sendrecv", "read", "write", "all"],
        default="all",
        help="Benchmark mode: sendrecv, read, write, or all (default: all)",
    )
    p.add_argument(
        "--local-gpu-idx",
        type=int,
        default=0,
        help="Local GPU index to bind buffers",
    )
    p.add_argument("--num-cpus", type=int, default=4, help="#CPU threads for RDMA ops")
    p.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Buffer location (cpu or gpu)",
    )
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
        ],
        help="Comma separated list of message sizes in bytes",
    )
    p.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Iterations per message size (excluding 1 warm-up)",
    )
    p.add_argument(
        "--async-api",
        action="store_true",
        help="Use asynchronous transfers",
    )
    args = p.parse_args()

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, "This benchmark only supports 2 processes"

    api_type = "Async" if args.async_api else "Sync"
    role = "client" if rank == 0 else "server"
    print(
        f"UCCL P2P Latency Benchmark — mode: {args.mode} | API: {api_type} | role: {role}"
    )
    print("Message sizes:", ", ".join(_pretty_size(s) for s in args.sizes))
    print(
        f"Device: {args.device} | Local GPU idx: {args.local_gpu_idx} | Iterations: {args.iters}"
    )

    # Note: Do NOT call torch.cuda.set_device() here - it can interfere with
    # RDMA read operations. The GPU index is passed to Endpoint and _make_buffer instead.

    ep = p2p.Endpoint(args.local_gpu_idx, args.num_cpus)
    local_metadata = ep.get_metadata()

    # Exchange metadata between processes
    if rank == 0:
        dist.send(torch.ByteTensor(list(local_metadata)), dst=1)
        remote_metadata_tensor = torch.zeros(len(local_metadata), dtype=torch.uint8)
        dist.recv(remote_metadata_tensor, src=1)
        remote_metadata = bytes(remote_metadata_tensor.tolist())
    else:
        remote_metadata_tensor = torch.zeros(len(local_metadata), dtype=torch.uint8)
        dist.recv(remote_metadata_tensor, src=0)
        dist.send(torch.ByteTensor(list(local_metadata)), dst=0)
        remote_metadata = bytes(remote_metadata_tensor.tolist())

    modes_to_run = ["sendrecv", "read", "write"] if args.mode == "all" else [args.mode]

    for mode in modes_to_run:
        if mode != modes_to_run[0]:
            # Synchronize before tearing down old connection
            dist.barrier()

            # Delete old endpoint and create new one
            del ep
            ep = p2p.Endpoint(args.local_gpu_idx, args.num_cpus)
            local_metadata = ep.get_metadata()

            # Re-exchange metadata
            if rank == 0:
                dist.send(torch.ByteTensor(list(local_metadata)), dst=1)
                remote_metadata_tensor = torch.zeros(
                    len(local_metadata), dtype=torch.uint8
                )
                dist.recv(remote_metadata_tensor, src=1)
                remote_metadata = bytes(remote_metadata_tensor.tolist())
            else:
                remote_metadata_tensor = torch.zeros(
                    len(local_metadata), dtype=torch.uint8
                )
                dist.recv(remote_metadata_tensor, src=0)
                dist.send(torch.ByteTensor(list(local_metadata)), dst=0)
                remote_metadata = bytes(remote_metadata_tensor.tolist())

            # Synchronize after metadata exchange before connecting
            dist.barrier()

        print(f"\n{'='*60}")
        print(f"Running {mode.upper()} latency benchmark")
        print(f"{'='*60}")

        if mode == "sendrecv":
            if rank == 0:
                _run_sendrecv_client(args, ep, remote_metadata)
            else:
                _run_sendrecv_server(args, ep, remote_metadata)
        elif mode == "read":
            if rank == 0:
                _run_read_client(args, ep, remote_metadata)
            else:
                _run_read_server(args, ep, remote_metadata)
        elif mode == "write":
            if rank == 0:
                _run_write_client(args, ep, remote_metadata)
            else:
                _run_write_server(args, ep, remote_metadata)

        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted] Benchmark aborted by user.")
        sys.exit(1)
