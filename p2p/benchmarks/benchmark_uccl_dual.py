from __future__ import annotations

import argparse
import sys
import time
from typing import List
import os
import socket
import struct

try:
    from uccl import p2p
except ImportError as exc:
    sys.stderr.write("Failed to import p2p\n")
    raise

_HAS_TORCH = False
try:
    import torch

    print("Torch imported")

    _HAS_TORCH = True
except ModuleNotFoundError:
    pass

import numpy as np


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
    n_elems = size_bytes // 4  # float32 elements
    if device == "gpu":
        if not _HAS_TORCH:
            raise RuntimeError(
                "PyTorch is required for GPU buffers (install torch)"
            )

        buf = torch.ones(n_elems, dtype=torch.float32, device=f"cuda:{gpu_idx}")
        assert buf.is_contiguous()
        ptr = buf.data_ptr()
    else:  # cpu
        buf = np.ones(n_elems, dtype=np.float32)
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


def _run_server(args):
    ep = p2p.Endpoint(args.local_gpu_idx, args.num_cpus)
    s = listen_oob(args)
    send_oob(ep.get_endpoint_metadata(), s)
    meta = recv_oob(args.remote_ip, args)
    ip, port, r_gpu = parse_metadata(meta)

    print("[Server] Waiting for connection …", flush=True)
    ok, r_ip, r_gpu2, conn_id = ep.accept()
    assert ok, "[Server] Failed to accept RDMA connection"
    print(f"[Server] Accept from {r_ip} (GPU {r_gpu2}) conn_id={conn_id}")

    ok, conn_id2 = ep.connect(ip, r_gpu, remote_port=port)
    assert ok, "[Server] Failed to connect to client"
    print(f"[Server] Connected to {args.remote_ip} conn_id={conn_id2}")

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


def _run_client(args):
    if args.remote_ip is None:
        sys.exit("[Client] --remote-ip is required")

    ep = p2p.Endpoint(args.local_gpu_idx, args.num_cpus)
    s = listen_oob(args)
    meta = recv_oob(args.remote_ip, args)
    ip, port, r_gpu = parse_metadata(meta)
    send_oob(ep.get_endpoint_metadata(), s)

    ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
    assert ok, "[Client] Failed to connect to server"
    print(f"[Client] Connected to {args.remote_ip} conn_id={conn_id}")

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
        raise argparse.ArgumentTypeError(
            "sizes must be comma-separated integers"
        )


def listen_oob(args):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", args.oob_port))
    s.listen(1)
    print(
        f"[Server] OOB channel listening on port {args.oob_port}",
        flush=True,
    )
    return s


def send_oob(metadata: bytes, s):
    conn, _ = s.accept()
    with conn:
        conn.sendall(metadata)
        print("[Server] OOB metadata sent", flush=True)


def recv_oob(server_ip: str, args) -> bytes:
    with socket.create_connection((server_ip, args.oob_port), timeout=10) as s:
        data = s.recv(32)
        if not data:
            sys.exit("[Client] Empty OOB metadata")
        return data


def main():
    p = argparse.ArgumentParser(
        description="Benchmark UCCL P2P Engine bandwidth"
    )
    p.add_argument(
        "--role",
        choices=["server", "client"],
        required=True,
        help="Run as server (receiver) or client (sender)",
    )
    p.add_argument("--remote-ip", help="Server IP address (client only)")
    p.add_argument(
        "--local-gpu-idx",
        type=int,
        default=0,
        help="Local GPU index to bind buffers",
    )
    p.add_argument(
        "--num-cpus", type=int, default=4, help="#CPU threads for RDMA ops"
    )
    p.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
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
        "--async-transfer",
        action="store_true",
        help="Use asynchronous transfers",
    )
    p.add_argument(
        "--oob-port",
        type=int,
        default=19999,
        help="TCP port used to ship metadata (server listens, client fetches)",
    )
    args = p.parse_args()

    print("UCCL P2P Benchmark — role:", args.role)
    print("Message sizes:", ", ".join(_pretty_size(s) for s in args.sizes))
    print(
        f"Device: {args.device} | Local GPU idx: {args.local_gpu_idx} | Iterations: {args.iters}"
    )
    if args.role == "server":
        _run_server(args)
    else:
        _run_client(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted] Benchmark aborted by user.")
        sys.exit(1)
