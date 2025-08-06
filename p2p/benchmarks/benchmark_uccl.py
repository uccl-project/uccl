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


def parse_metadata(metadata: bytes):
    if len(metadata) == 10:
        # IPv4: 4 bytes IP, 2 bytes port, 4 bytes GPU idx
        ip_bytes = metadata[:4]
        port_bytes = metadata[4:6]
        gpu_idx_bytes = metadata[6:10]
        ip = socket.inet_ntop(socket.AF_INET, ip_bytes)
    elif len(metadata) == 22:
        # IPv6: 16 bytes IP, 2 bytes port, 4 bytes GPU idx
        ip_bytes = metadata[:16]
        port_bytes = metadata[16:18]
        gpu_idx_bytes = metadata[18:22]
        ip = socket.inet_ntop(socket.AF_INET6, ip_bytes)
    else:
        raise ValueError(f"Unexpected metadata length: {len(metadata)}")

    port = struct.unpack("!H", port_bytes)[0]
    remote_gpu_idx = struct.unpack("i", gpu_idx_bytes)[0]  # host byte order
    return ip, port, remote_gpu_idx


def _make_buffer(size_bytes: int, device: str, gpu_idx: int):
    """Allocate a contiguous buffer of *size_bytes* and return (buffer, ptr)."""
    if device == "gpu":
        dtype = torch.float32 if size_bytes >= 4 else torch.uint8
        n_elems = size_bytes // dtype.itemsize
        buf = torch.ones(n_elems, dtype=dtype).cuda()
        assert buf.device.type == "cuda"
        assert buf.is_contiguous()
        ptr = buf.data_ptr()
    else:  # cpu
        dtype = np.float32 if size_bytes >= 4 else np.uint8
        n_elems = size_bytes // dtype.itemsize
        buf = np.ones(n_elems, dtype=dtype)
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
    ok, r_ip, r_gpu, conn_id = ep.accept()
    assert ok, "[Server] Failed to accept RDMA connection"
    print(f"[Server] Accept from {r_ip} (GPU {r_gpu}) conn_id={conn_id}")

    for size in args.sizes:
        size_per_block = size // args.num_kvblocks
        buf_v = []
        mr_id_v = []
        data_ptr_v = []
        size_v = []
        for _ in range(args.num_kvblocks):
            buf, ptr = _make_buffer(size_per_block, args.device, args.local_gpu_idx)
            ok, mr_id = ep.reg(ptr, size_per_block)
            assert ok, "[Server] register failed"
            buf_v.append(buf)
            mr_id_v.append(mr_id)
            data_ptr_v.append(ptr)
            size_v.append(size_per_block)

        if args.num_kvblocks == 1:
            if args.async_api:
                ok, transfer_id = ep.recv_async(
                    conn_id, mr_id_v[0], data_ptr_v[0], size_v[0]
                )
                assert ok, "[Server] recv_async error"
                is_done = False
                while not is_done:
                    ok, is_done = ep.poll_async(transfer_id)
                    assert ok, "[Server] poll_async error"
            else:
                ok = ep.recv(conn_id, mr_id_v[0], data_ptr_v[0], size_v[0])
                assert ok, "[Server] recv error"

            start = time.perf_counter()
            total_recv = 0
            for _ in range(args.iters):
                if args.async_api:
                    ok, transfer_id = ep.recv_async(
                        conn_id, mr_id_v[0], data_ptr_v[0], size_v[0]
                    )
                    assert ok, "[Server] recv_async error"
                    is_done = False
                    while not is_done:
                        ok, is_done = ep.poll_async(transfer_id)
                        assert ok, "[Server] poll_async error"
                        # Now, we assume async recv knows the to-receive size in advance.
                    total_recv += size_v[0]
                else:
                    ok = ep.recv(conn_id, mr_id_v[0], data_ptr_v[0], size_v[0])
                    assert ok, "[Server] recv error"
                    total_recv += size_v[0]
            elapsed = time.perf_counter() - start

            gbps = (total_recv * 8) / elapsed / 1e9  # bits per second → Gbps
            gb_sec = total_recv / elapsed / 1e9  # bytes per second → GB/s
            lat = elapsed / args.iters
        else:
            ep.recvv(conn_id, mr_id_v, data_ptr_v, size_v, args.num_kvblocks)
            start = time.perf_counter()
            total_recv = 0
            for _ in range(args.iters):
                ok = ep.recvv(conn_id, mr_id_v, data_ptr_v, size_v, args.num_kvblocks)
                assert ok, "[Server] recv error"
                total_recv += sum(size_v)
            elapsed = time.perf_counter() - start
            gbps = (total_recv * 8) / elapsed / 1e9  # bits per second → Gbps
            gb_sec = total_recv / elapsed / 1e9  # bytes per second → GB/s
            lat = elapsed / args.iters

        print(
            f"[Server] {_pretty_size(size):>8} : {gbps:6.2f} Gbps | {gb_sec:6.2f} GB/s  | {lat:6.6f} s"
        )
    print("[Server] Benchmark complete")


def _run_client(args, ep, remote_metadata):
    ip, port, r_gpu = parse_metadata(remote_metadata)
    ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
    assert ok, "[Client] Failed to connect to server"
    print(f"[Client] Connected to {ip}:{port} (GPU {r_gpu}) conn_id={conn_id}")

    for size in args.sizes:
        size_per_block = size // args.num_kvblocks
        buf_v = []
        mr_id_v = []
        data_ptr_v = []
        size_v = []
        for _ in range(args.num_kvblocks):
            buf, ptr = _make_buffer(size_per_block, args.device, args.local_gpu_idx)
            ok, mr_id = ep.reg(ptr, size_per_block)
            assert ok, "[Client] register failed"
            buf_v.append(buf)
            mr_id_v.append(mr_id)
            data_ptr_v.append(ptr)
            size_v.append(size_per_block)

        if args.num_kvblocks == 1:
            if args.async_api:
                ok, transfer_id = ep.send_async(
                    conn_id, mr_id_v[0], data_ptr_v[0], size_v[0]
                )
                assert ok, "[Client] send_async error"
                is_done = False
                while not is_done:
                    ok, is_done = ep.poll_async(transfer_id)
                    assert ok, "[Client] poll_async error"
            else:
                ep.send(conn_id, mr_id_v[0], data_ptr_v[0], size_v[0])

            start = time.perf_counter()
            total_sent = 0
            for _ in range(args.iters):
                if args.async_api:
                    ok, transfer_id = ep.send_async(
                        conn_id, mr_id_v[0], data_ptr_v[0], size_v[0]
                    )
                    assert ok, "[Client] send_async error"
                    is_done = False
                    while not is_done:
                        ok, is_done = ep.poll_async(transfer_id)
                        assert ok, "[Client] poll_async error"
                else:
                    ok = ep.send(conn_id, mr_id_v[0], data_ptr_v[0], size_v[0])
                    assert ok, "[Client] send error"
                total_sent += size_v[0]
            elapsed = time.perf_counter() - start

            gbps = (total_sent * 8) / elapsed / 1e9
            gb_sec = total_sent / elapsed / 1e9
            lat = elapsed / args.iters
        else:
            ep.sendv(conn_id, mr_id_v, data_ptr_v, size_v, args.num_kvblocks)
            start = time.perf_counter()
            total_sent = 0
            for _ in range(args.iters):
                ok = ep.sendv(conn_id, mr_id_v, data_ptr_v, size_v, args.num_kvblocks)
                assert ok, "[Client] send error"
                total_sent += sum(size_v)
            elapsed = time.perf_counter() - start
            gbps = (total_sent * 8) / elapsed / 1e9
            gb_sec = total_sent / elapsed / 1e9
            lat = elapsed / args.iters

        print(
            f"[Client] {_pretty_size(size):>8} : {gbps:6.2f} Gbps | {gb_sec:6.2f} GB/s  | {lat:6.6f} s"
        )
    print("[Client] Benchmark complete")


def _run_server_dual(args, ep, remote_metadata):
    ip, port, r_gpu = parse_metadata(remote_metadata)

    print("[Server] Waiting for connection …", flush=True)
    ok, r_ip, r_gpu2, conn_id = ep.accept()
    assert ok, "[Server] Failed to accept RDMA connection"
    print(f"[Server] Accept from {r_ip} (GPU {r_gpu2}) conn_id={conn_id}")

    ok, conn_id2 = ep.connect(ip, r_gpu, remote_port=port)
    assert ok, "[Server] Failed to connect to client"
    print(f"[Server] Connected to {ip}:{port} (GPU {r_gpu}) conn_id={conn_id2}")

    for size in args.sizes:
        buf, ptr = _make_buffer(size, args.device, args.local_gpu_idx)
        ok, mr_id = ep.reg(ptr, size)
        assert ok, "[Server] register failed"
        # ep.recv(conn_id, mr_id, ptr, size)

        buf2, ptr2 = _make_buffer(size, args.device, args.local_gpu_idx)
        ok, mr_id2 = ep.reg(ptr2, size)
        assert ok, "[Server] register failed"
        # ep.send(conn_id, mr_id2, ptr2, size)

        start = time.perf_counter()
        total_recv = 0
        for _ in range(args.iters):
            transfer_ids = []

            ok, transfer_id = ep.recv_async(conn_id, mr_id, ptr, size)
            assert ok, "[Server] recv error"
            transfer_ids.append(transfer_id)

            ok, transfer_id2 = ep.send_async(conn_id2, mr_id2, ptr2, size)
            assert ok, "[Server] send error"
            transfer_ids.append(transfer_id2)

            for transfer_id in transfer_ids:
                is_done = False
                while not is_done:
                    ok, is_done = ep.poll_async(transfer_id)
                    assert ok, "[Server] poll error"

            # ok = ep.send(conn_id, mr_id2, ptr2, size)
            # assert ok, "[Server] send error"

            total_recv += size
        elapsed = time.perf_counter() - start

        gbps = (total_recv * 8) / elapsed / 1e9  # bits per second → Gbps
        gb_sec = total_recv / elapsed / 1e9  # bytes per second → GB/s
        lat = elapsed / args.iters

        print(
            f"[Server] {_pretty_size(size):>8} : {gbps:6.2f} Gbps | {gb_sec:6.2f} GB/s  | {lat:6.6f} s"
        )
    print("[Server] Benchmark complete")


def _run_client_dual(args, ep, remote_metadata):
    ip, port, r_gpu = parse_metadata(remote_metadata)

    ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
    assert ok, "[Client] Failed to connect to server"
    print(f"[Client] Connected to {ip}:{port} (GPU {r_gpu}) conn_id={conn_id}")

    ok, r_ip, r_gpu2, conn_id2 = ep.accept()
    assert ok, "[Client] Failed to accept RDMA connection"
    print(f"[Client] Accept from {r_ip} (GPU {r_gpu2}) conn_id={conn_id2}")

    for size in args.sizes:
        buf, ptr = _make_buffer(size, args.device, args.local_gpu_idx)
        ok, mr_id = ep.reg(ptr, size)
        assert ok, "[Client] register failed"
        # ep.send(conn_id, mr_id, ptr, size)

        buf2, ptr2 = _make_buffer(size, args.device, args.local_gpu_idx)
        ok, mr_id2 = ep.reg(ptr2, size)
        assert ok, "[Client] register failed"
        # ep.recv(conn_id, mr_id2, ptr2, size)

        start = time.perf_counter()
        total_sent = 0
        for _ in range(args.iters):
            transfer_ids = []

            ok, transfer_id = ep.send_async(conn_id, mr_id, ptr, size)
            assert ok, "[Client] send error"
            transfer_ids.append(transfer_id)

            ok, transfer_id2 = ep.recv_async(conn_id2, mr_id2, ptr2, size)
            assert ok, "[Client] recv error"
            transfer_ids.append(transfer_id2)

            for transfer_id in transfer_ids:
                is_done = False
                while not is_done:
                    ok, is_done = ep.poll_async(transfer_id)
                    assert ok, "[Client] poll error"

            # ok = ep.recv(conn_id, mr_id2, ptr2, size)
            # assert ok, "[Client] recv error"

            total_sent += size
        elapsed = time.perf_counter() - start

        gbps = (total_sent * 8) / elapsed / 1e9
        gb_sec = total_sent / elapsed / 1e9
        lat = elapsed / args.iters

        print(
            f"[Client] {_pretty_size(size):>8} : {gbps:6.2f} Gbps | {gb_sec:6.2f} GB/s  | {lat:6.6f} s"
        )
    print("[Client] Benchmark complete")


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
            10485760,
            16 * 1024 * 1024,
            104857600,
        ],
        help="Comma separated list of message sizes in bytes",
    )
    p.add_argument(
        "--iters",
        type=int,
        default=1000,
        help="Iterations per message size (excluding 1 warm-up)",
    )
    p.add_argument(
        "--num-kvblocks",
        type=int,
        default=1,
        help="Number of key-value blocks to send/recv in a single call",
    )
    p.add_argument(
        "--async-api",
        action="store_true",
        help="Use asynchronous transfers",
    )
    p.add_argument(
        "--dual",
        action="store_true",
        help="Run dual benchmark",
    )
    args = p.parse_args()

    if args.async_api:
        assert args.num_kvblocks == 1, "Async transfers only support one block"

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, "This benchmark only supports 2 processes"

    print("UCCL P2P Benchmark — role:", "client" if rank == 0 else "server")
    print("Number of key-value blocks per message:", args.num_kvblocks)
    print("Message sizes:", ", ".join(_pretty_size(s) for s in args.sizes))
    print(
        f"Device: {args.device} | Local GPU idx: {args.local_gpu_idx} | Iterations: {args.iters}"
    )
    torch.cuda.set_device(f"cuda:{args.local_gpu_idx}")

    ep = p2p.Endpoint(args.local_gpu_idx, args.num_cpus)
    local_metadata = ep.get_endpoint_metadata()

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

    if args.dual:
        if rank == 0:
            _run_client_dual(args, ep, remote_metadata)
        else:
            _run_server_dual(args, ep, remote_metadata)
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
