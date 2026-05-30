from __future__ import annotations

import argparse
import sys
import time
from typing import List
import os
import socket
import struct
import torch
import torch.distributed as dist
import numpy as np

try:
    from uccl import p2p
except ImportError as exc:
    sys.stderr.write("Failed to import p2p\n")
    raise


# parse_metadata is now provided by the C++ layer via p2p.Endpoint.parse_metadata()


_FLOAT_TYPE_TORCH = {
    "float16": (torch.float16, 2),
    "bfloat16": (torch.bfloat16, 2),
    "float32": (torch.float32, 4),
}


def _str_to_float_type(name: str) -> "p2p.FloatType":
    return {
        "float16": p2p.FloatType.kFloat16,
        "bfloat16": p2p.FloatType.kBFloat16,
        "float32": p2p.FloatType.kFloat32,
        "none": p2p.FloatType.kUndefined,
    }.get(name, p2p.FloatType.kUndefined)


def _make_buffer(
    size_bytes: int,
    device: str,
    gpu_idx: int,
    pinned: bool = False,
    float_type_str: str = "none",
):
    """Allocate a contiguous buffer and return (buffer, ptr).

    If *float_type_str* is set, the GPU tensor uses the matching dtype so
    bytes are valid float elements for the DietGPU compressor.
    """
    if size_bytes <= 0:
        raise ValueError(f"buffer size must be positive, got {size_bytes}")
    if device == "gpu":
        if float_type_str in _FLOAT_TYPE_TORCH:
            dtype, esize = _FLOAT_TYPE_TORCH[float_type_str]
            assert (
                size_bytes % esize == 0
            ), f"size {size_bytes} not divisible by {esize} for {float_type_str}"
            buf = torch.ones(size_bytes // esize, dtype=dtype, device=f"cuda:{gpu_idx}")
        else:
            buf = torch.ones(size_bytes, dtype=torch.uint8, device=f"cuda:{gpu_idx}")
        assert buf.device.type == "cuda"
        assert buf.is_contiguous()
        ptr = buf.data_ptr()
    elif device == "cpu" and pinned:
        buf = torch.ones(size_bytes, dtype=torch.uint8).pin_memory()
        assert buf.is_pinned()
        assert buf.is_contiguous()
        ptr = buf.data_ptr()
    else:  # cpu (pageable)
        buf = np.ones(size_bytes, dtype=np.uint8)
        ptr = buf.ctypes.data
    return buf, ptr


def _pretty_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    val = float(num_bytes)
    for u in units:
        if val < 1024 or u == units[-1]:
            return f"{val:.0f} {u}" if u == "B" else f"{val:.1f} {u}"
        val /= 1024
    return f"{num_bytes} B"  # fallback


def _run_server(args, ep, remote_metadata):
    peer = 0
    ok, r_ip, r_gpu, conn_id = ep.accept()
    assert ok, "[Server] Failed to accept RDMA connection"
    print(f"[Server] Accept from {r_ip} (GPU {r_gpu}) conn_id={conn_id}")

    ft = _str_to_float_type(args.float_type)
    for size in args.sizes:
        size_per_block = size
        buf_v = []
        mr_id_v = []
        data_ptr_v = []
        size_v = []
        for _ in range(args.num_iovs):
            buf, ptr = _make_buffer(
                size_per_block,
                args.device,
                args.local_gpu_idx,
                args.pinned,
                args.float_type,
            )
            ok, mr_id = ep.reg(ptr, size_per_block, ft)
            assert ok, "[Server] register failed"
            buf_v.append(buf)
            mr_id_v.append(mr_id)
            data_ptr_v.append(ptr)
            size_v.append(size_per_block)

        ok, fifo_blob_v = ep.advertisev(
            conn_id, mr_id_v, data_ptr_v, size_v, args.num_iovs
        )
        assert ok, "[Server] advertisev failed"
        for fifo_blob in fifo_blob_v:
            dist.send(torch.ByteTensor(list(fifo_blob)), dst=peer)
        dist.barrier()
    print("[Server] Benchmark complete")


def _run_client(args, ep, remote_metadata):
    peer = 1
    ip, port, r_gpu = p2p.Endpoint.parse_metadata(remote_metadata)
    ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
    assert ok, "[Client] Failed to connect to server"
    print(f"[Client] Connected to {ip}:{port} (GPU {r_gpu}) conn_id={conn_id}")

    ft = _str_to_float_type(args.float_type)
    for size in args.sizes:
        size_per_block = size
        buf_v = []
        mr_id_v = []
        data_ptr_v = []
        size_v = []
        for _ in range(args.num_iovs):
            buf, ptr = _make_buffer(
                size_per_block,
                args.device,
                args.local_gpu_idx,
                args.pinned,
                args.float_type,
            )
            ok, mr_id = ep.reg(ptr, size_per_block, ft)
            assert ok, "[Client] register failed"
            buf_v.append(buf)
            mr_id_v.append(mr_id)
            data_ptr_v.append(ptr)
            size_v.append(size_per_block)

        fifo_blob_v = []
        for _ in range(args.num_iovs):
            fifo_blob = torch.zeros(64, dtype=torch.uint8)
            dist.recv(fifo_blob, src=peer)
            fifo_blob_v.append(bytes(fifo_blob.tolist()))

        def submit_once():
            if args.async_api:
                if args.num_iovs == 1:
                    if args.mode == "write":
                        ok, transfer_id = ep.write_async(
                            conn_id,
                            mr_id_v[0],
                            data_ptr_v[0],
                            size_v[0],
                            fifo_blob_v[0],
                        )
                    else:
                        ok, transfer_id = ep.read_async(
                            conn_id,
                            mr_id_v[0],
                            data_ptr_v[0],
                            size_v[0],
                            fifo_blob_v[0],
                        )
                else:
                    if args.mode == "write":
                        ok, transfer_id = ep.writev_async(
                            conn_id,
                            mr_id_v,
                            data_ptr_v,
                            size_v,
                            fifo_blob_v,
                            args.num_iovs,
                        )
                    else:
                        ok, transfer_id = ep.readv_async(
                            conn_id,
                            mr_id_v,
                            data_ptr_v,
                            size_v,
                            fifo_blob_v,
                            args.num_iovs,
                        )
                assert ok, f"[Client] {args.mode}_async error"
                is_done = False
                while not is_done:
                    ok, is_done = ep.poll_async(transfer_id)
                    assert ok, "[Client] poll_async error"
            else:
                if args.mode == "write":
                    ok = ep.writev(
                        conn_id, mr_id_v, data_ptr_v, size_v, fifo_blob_v, args.num_iovs
                    )
                else:
                    ok = ep.readv(
                        conn_id, mr_id_v, data_ptr_v, size_v, fifo_blob_v, args.num_iovs
                    )
                assert ok, f"[Client] {args.mode} error"

        submit_once()
        start = time.perf_counter()
        total = 0
        for _ in range(args.iters):
            submit_once()
            total += sum(size_v)
        elapsed = time.perf_counter() - start
        gbps = (total * 8) / elapsed / 1e9
        gb_sec = total / elapsed / 1e9
        lat = elapsed / args.iters

        print(
            f"[Client] {_pretty_size(size):>8} : {gbps:6.2f} Gbps | {gb_sec:6.2f} GB/s  | {lat * 1e6:8.2f} us"
        )
        dist.barrier()
    print("[Client] Benchmark complete")


def _send_bytes_dist(payload: bytes, dst: int):
    """Send a byte blob to another rank via torch.distributed."""
    n = len(payload)
    dist.send(torch.tensor([n], dtype=torch.int64), dst=dst)
    if n == 0:
        return
    buf = torch.frombuffer(memoryview(payload), dtype=torch.uint8)
    dist.send(buf, dst=dst)


def _recv_bytes_dist(src: int) -> bytes:
    """Receive a byte blob from another rank via torch.distributed."""
    n_tensor = torch.empty(1, dtype=torch.int64)
    dist.recv(n_tensor, src=src)
    n = int(n_tensor.item())
    if n == 0:
        return b""
    buf = torch.empty(n, dtype=torch.uint8)
    dist.recv(buf, src=src)
    return buf.numpy().tobytes()


def _run_server_write_ipc(args, ep):
    """Server for write_ipc benchmark: advertises buffer(s), client writes into them."""
    ok, remote_gpu_idx, conn_id = ep.accept_local()
    assert ok, "[Server] Failed to accept local IPC connection"

    num_iovs = args.num_iovs
    for size in args.sizes:
        size_per_block = size

        if num_iovs == 1:
            buf, ptr = _make_buffer(size_per_block, "gpu", args.local_gpu_idx)
            ok, info_blob = ep.advertise_ipc(conn_id, ptr, size_per_block)
            assert ok, "[Server] advertise_ipc failed"
            _send_bytes_dist(bytes(info_blob), dst=0)
        else:
            bufs_ptrs = [
                _make_buffer(size_per_block, "gpu", args.local_gpu_idx)
                for _ in range(num_iovs)
            ]
            ptrs = [p for _, p in bufs_ptrs]
            ok, info_blobs = ep.advertisev_ipc(
                conn_id, ptrs, [size_per_block] * num_iovs
            )
            assert ok, "[Server] advertisev_ipc failed"
            packed = struct.pack("I", num_iovs) + b"".join(bytes(b) for b in info_blobs)
            _send_bytes_dist(packed, dst=0)

        _recv_bytes_dist(src=0)

    print("[Server] write_ipc benchmark complete")


def _run_client_write_ipc(args, ep, remote_gpu_idx):
    """Client for write_ipc benchmark: writes local buffer(s) into server's advertised buffer(s)."""
    ok, conn_id = ep.connect_local(remote_gpu_idx)
    assert ok, "[Client] Failed to connect to local server via IPC"

    num_iovs = args.num_iovs
    for size in args.sizes:
        size_per_block = size

        if num_iovs == 1:
            buf, ptr = _make_buffer(
                size_per_block, args.device, args.local_gpu_idx, args.pinned
            )
            info_blob = _recv_bytes_dist(src=1)

            # Warm-up
            if args.async_api:
                ok, transfer_id = ep.write_ipc_async(
                    conn_id, ptr, size_per_block, info_blob
                )
                assert ok, "[Client] write_ipc_async warm-up error"
                is_done = False
                while not is_done:
                    ok, is_done = ep.poll_async(transfer_id)
                    assert ok, "[Client] poll_async error"
            else:
                ok = ep.write_ipc(conn_id, ptr, size_per_block, info_blob)
                assert ok, "[Client] write_ipc warm-up error"

            start = time.perf_counter()
            total = 0
            for _ in range(args.iters):
                if args.async_api:
                    ok, transfer_id = ep.write_ipc_async(
                        conn_id, ptr, size_per_block, info_blob
                    )
                    assert ok, "[Client] write_ipc_async error"
                    is_done = False
                    while not is_done:
                        ok, is_done = ep.poll_async(transfer_id)
                        assert ok, "[Client] poll_async error"
                else:
                    ok = ep.write_ipc(conn_id, ptr, size_per_block, info_blob)
                    assert ok, "[Client] write_ipc error"
                total += size_per_block
        else:
            bufs_ptrs = [
                _make_buffer(
                    size_per_block, args.device, args.local_gpu_idx, args.pinned
                )
                for _ in range(num_iovs)
            ]
            ptrs = [p for _, p in bufs_ptrs]
            packed = _recv_bytes_dist(src=1)
            blob_size = (len(packed) - 4) // num_iovs
            info_blobs = [
                packed[4 + i * blob_size : 4 + (i + 1) * blob_size]
                for i in range(num_iovs)
            ]
            size_v = [size_per_block] * num_iovs

            # Warm-up
            if args.async_api:
                ok, transfer_id = ep.writev_ipc_async(conn_id, ptrs, size_v, info_blobs)
                assert ok, "[Client] writev_ipc_async warm-up error"
                is_done = False
                while not is_done:
                    ok, is_done = ep.poll_async(transfer_id)
                    assert ok, "[Client] poll_async error"
            else:
                ok = ep.writev_ipc(conn_id, ptrs, size_v, info_blobs)
                assert ok, "[Client] writev_ipc warm-up error"

            start = time.perf_counter()
            total = 0
            for _ in range(args.iters):
                if args.async_api:
                    ok, transfer_id = ep.writev_ipc_async(
                        conn_id, ptrs, size_v, info_blobs
                    )
                    assert ok, "[Client] writev_ipc_async error"
                    is_done = False
                    while not is_done:
                        ok, is_done = ep.poll_async(transfer_id)
                        assert ok, "[Client] poll_async error"
                else:
                    ok = ep.writev_ipc(conn_id, ptrs, size_v, info_blobs)
                    assert ok, "[Client] writev_ipc error"
                total += size_per_block * num_iovs

        elapsed = time.perf_counter() - start
        gbps = (total * 8) / elapsed / 1e9
        gb_sec = total / elapsed / 1e9
        lat = elapsed / args.iters

        print(
            f"[Client] {_pretty_size(size):>8} : {gbps:6.2f} Gbps | {gb_sec:6.2f} GB/s  | {lat * 1e6:8.2f} us"
        )

        _send_bytes_dist(b"\x01", dst=1)

    print("[Client] write_ipc benchmark complete")


def _run_server_read_ipc(args, ep):
    """Server for read_ipc benchmark: advertises buffer(s), client reads from them."""
    ok, remote_gpu_idx, conn_id = ep.accept_local()
    assert ok, "[Server] Failed to accept local IPC connection"

    num_iovs = args.num_iovs
    for size in args.sizes:
        size_per_block = size

        if num_iovs == 1:
            buf, ptr = _make_buffer(size_per_block, "gpu", args.local_gpu_idx)
            ok, info_blob = ep.advertise_ipc(conn_id, ptr, size_per_block)
            assert ok, "[Server] advertise_ipc failed"
            _send_bytes_dist(bytes(info_blob), dst=0)
        else:
            bufs_ptrs = [
                _make_buffer(size_per_block, "gpu", args.local_gpu_idx)
                for _ in range(num_iovs)
            ]
            ptrs = [p for _, p in bufs_ptrs]
            ok, info_blobs = ep.advertisev_ipc(
                conn_id, ptrs, [size_per_block] * num_iovs
            )
            assert ok, "[Server] advertisev_ipc failed"
            packed = struct.pack("I", num_iovs) + b"".join(bytes(b) for b in info_blobs)
            _send_bytes_dist(packed, dst=0)

        _recv_bytes_dist(src=0)

    print("[Server] read_ipc benchmark complete")


def _run_client_read_ipc(args, ep, remote_gpu_idx):
    """Client for read_ipc benchmark: reads from server's advertised buffer(s) into local buffer(s)."""
    ok, conn_id = ep.connect_local(remote_gpu_idx)
    assert ok, "[Client] Failed to connect to local server via IPC"

    num_iovs = args.num_iovs
    for size in args.sizes:
        size_per_block = size

        if num_iovs == 1:
            buf, ptr = _make_buffer(
                size_per_block, args.device, args.local_gpu_idx, args.pinned
            )
            info_blob = _recv_bytes_dist(src=1)

            # Warm-up
            if args.async_api:
                ok, transfer_id = ep.read_ipc_async(
                    conn_id, ptr, size_per_block, info_blob
                )
                assert ok, "[Client] read_ipc_async warm-up error"
                is_done = False
                while not is_done:
                    ok, is_done = ep.poll_async(transfer_id)
                    assert ok, "[Client] poll_async error"
            else:
                ok = ep.read_ipc(conn_id, ptr, size_per_block, info_blob)
                assert ok, "[Client] read_ipc warm-up error"

            start = time.perf_counter()
            total = 0
            for _ in range(args.iters):
                if args.async_api:
                    ok, transfer_id = ep.read_ipc_async(
                        conn_id, ptr, size_per_block, info_blob
                    )
                    assert ok, "[Client] read_ipc_async error"
                    is_done = False
                    while not is_done:
                        ok, is_done = ep.poll_async(transfer_id)
                        assert ok, "[Client] poll_async error"
                else:
                    ok = ep.read_ipc(conn_id, ptr, size_per_block, info_blob)
                    assert ok, "[Client] read_ipc error"
                total += size_per_block
        else:
            bufs_ptrs = [
                _make_buffer(
                    size_per_block, args.device, args.local_gpu_idx, args.pinned
                )
                for _ in range(num_iovs)
            ]
            ptrs = [p for _, p in bufs_ptrs]
            packed = _recv_bytes_dist(src=1)
            blob_size = (len(packed) - 4) // num_iovs
            info_blobs = [
                packed[4 + i * blob_size : 4 + (i + 1) * blob_size]
                for i in range(num_iovs)
            ]
            size_v = [size_per_block] * num_iovs

            # Warm-up
            if args.async_api:
                ok, transfer_id = ep.readv_ipc_async(conn_id, ptrs, size_v, info_blobs)
                assert ok, "[Client] readv_ipc_async warm-up error"
                is_done = False
                while not is_done:
                    ok, is_done = ep.poll_async(transfer_id)
                    assert ok, "[Client] poll_async error"
            else:
                ok = ep.readv_ipc(conn_id, ptrs, size_v, info_blobs)
                assert ok, "[Client] readv_ipc warm-up error"

            start = time.perf_counter()
            total = 0
            for _ in range(args.iters):
                if args.async_api:
                    ok, transfer_id = ep.readv_ipc_async(
                        conn_id, ptrs, size_v, info_blobs
                    )
                    assert ok, "[Client] readv_ipc_async error"
                    is_done = False
                    while not is_done:
                        ok, is_done = ep.poll_async(transfer_id)
                        assert ok, "[Client] poll_async error"
                else:
                    ok = ep.readv_ipc(conn_id, ptrs, size_v, info_blobs)
                    assert ok, "[Client] readv_ipc error"
                total += size_per_block * num_iovs

        elapsed = time.perf_counter() - start
        gbps = (total * 8) / elapsed / 1e9
        gb_sec = total / elapsed / 1e9
        lat = elapsed / args.iters

        print(
            f"[Client] {_pretty_size(size):>8} : {gbps:6.2f} Gbps | {gb_sec:6.2f} GB/s  | {lat * 1e6:8.2f} us"
        )

        _send_bytes_dist(b"\x01", dst=1)

    print("[Client] read_ipc benchmark complete")


def parse_size_list(val: str) -> List[int]:
    try:
        return [int(s) for s in val.split(",") if s]
    except ValueError:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers")


def main():
    p = argparse.ArgumentParser(description="Benchmark UCCL P2P Engine bandwidth")
    p.add_argument(
        "--local-gpu-idx",
        type=int,
        default=0,
        help="Local GPU index to bind buffers",
    )
    p.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Buffer location (cpu or gpu)",
    )
    p.add_argument(
        "--pinned",
        action="store_true",
        help="Use pinned (page-locked) memory for CPU buffers",
    )
    p.add_argument(
        "--sizes",
        type=parse_size_list,
        default=[
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
            131072,
            262144,
            524288,
            1048576,
        ],
        help="Comma separated list of message sizes in bytes",
    )
    p.add_argument(
        "--iters",
        type=int,
        default=128,
        help="Iterations per message size (excluding 1 warm-up)",
    )
    p.add_argument(
        "--float-type",
        choices=["float16", "bfloat16", "float32", "none"],
        default="float16",
        help=(
            "Float element type passed to ep.reg for compression context. "
            "Use 'none' to disable compression (kUndefined). "
            "Default: float16."
        ),
    )
    p.add_argument(
        "--num-iovs",
        type=int,
        default=1,
        help="Number of iovs to read/write in a single call",
    )
    p.add_argument(
        "--mode",
        choices=["write", "read"],
        default="write",
        help="One-sided network benchmark mode",
    )
    p.add_argument(
        "--async-api",
        action="store_true",
        help="Use asynchronous transfers",
    )
    p.add_argument(
        "--sender-device",
        choices=["cpu", "gpu"],
        default=None,
        help="Buffer location for sender (IPC mode). Defaults to --device value.",
    )
    p.add_argument(
        "--receiver-device",
        choices=["cpu", "gpu"],
        default=None,
        help="Buffer location for receiver (IPC mode). Defaults to --device value.",
    )
    p.add_argument(
        "--write-ipc",
        action="store_true",
        help="Benchmark one-sided write_ipc (client writes into server's advertised buffer)",
    )
    p.add_argument(
        "--read-ipc",
        action="store_true",
        help="Benchmark one-sided read_ipc (client reads from server's advertised buffer)",
    )
    args = p.parse_args()

    # Default sender/receiver device to --device if not specified
    if args.sender_device is None:
        args.sender_device = args.device
    if args.receiver_device is None:
        args.receiver_device = args.device

    # Check for incompatible options
    mode_flags = sum([args.write_ipc, args.read_ipc])
    if mode_flags > 1:
        print("Error: --write-ipc and --read-ipc are mutually exclusive")
        sys.exit(1)

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, "This benchmark only supports 2 processes"

    is_ipc_mode = args.write_ipc or args.read_ipc
    if args.write_ipc:
        mode = "write_ipc"
    elif args.read_ipc:
        mode = "read_ipc"
    else:
        mode = "Standard"
    api_type = "Async" if args.async_api else "Sync"
    print(
        f"UCCL P2P Benchmark — mode: {mode} | API: {api_type} | role:",
        "client" if rank == 0 else "server",
    )
    if not is_ipc_mode:
        print("Number of IOVs per message:", args.num_iovs)
    else:
        # Use the rank as the local GPU index for IPC
        print(f"Using rank {rank} as local GPU index for IPC")
        args.local_gpu_idx = rank

    print("Message sizes:", ", ".join(_pretty_size(s) for s in args.sizes))
    if not is_ipc_mode:
        print("Float type:", args.float_type)
    pinned_str = " (pinned)" if args.pinned else ""
    if is_ipc_mode:
        iovs_str = (
            f" | IOVs per call: {args.num_iovs}"
            if (args.write_ipc or args.read_ipc)
            else ""
        )
        print(
            f"Sender device: {args.sender_device}{pinned_str} | Receiver device: {args.receiver_device}{pinned_str} | Local GPU idx: {args.local_gpu_idx} | Iterations: {args.iters}{iovs_str}"
        )
    else:
        print(
            f"Device: {args.device}{pinned_str} | Local GPU idx: {args.local_gpu_idx} | Iterations: {args.iters}"
        )
    torch.cuda.set_device(f"cuda:{args.local_gpu_idx}")

    ep = p2p.Endpoint(args.local_gpu_idx)
    local_metadata = ep.get_metadata()

    # This also serves as a barrier to guarantee both processes have created the endpoint
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

    if args.write_ipc:
        _, _, remote_gpu_idx = p2p.Endpoint.parse_metadata(remote_metadata)
        if rank == 0:
            _run_client_write_ipc(args, ep, remote_gpu_idx)
        else:
            _run_server_write_ipc(args, ep)
    elif args.read_ipc:
        _, _, remote_gpu_idx = p2p.Endpoint.parse_metadata(remote_metadata)
        if rank == 0:
            _run_client_read_ipc(args, ep, remote_gpu_idx)
        else:
            _run_server_read_ipc(args, ep)
    else:
        if rank == 0:
            _run_client(args, ep, remote_metadata)
        else:
            _run_server(args, ep, remote_metadata)

    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted] Benchmark aborted by user.")
        sys.exit(1)
