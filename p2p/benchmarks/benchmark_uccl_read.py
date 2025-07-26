from __future__ import annotations
import argparse, sys, time, socket, struct
from typing import List

try:
    from uccl import p2p
except ImportError:
    sys.stderr.write("Failed to import p2p\n")
    raise

_HAS_TORCH = False
try:
    import torch

    _HAS_TORCH = True
except ModuleNotFoundError:
    pass
import numpy as np
import os

os.environ["UCCL_RCMODE"] = "1"


def parse_metadata(meta: bytes):
    if len(meta) == 10:  # IPv4
        ip_b, port_b, gpu_b = meta[:4], meta[4:6], meta[6:10]
        ip = socket.inet_ntop(socket.AF_INET, ip_b)
    elif len(meta) == 22:  # IPv6
        ip_b, port_b, gpu_b = meta[:16], meta[16:18], meta[18:22]
        ip = socket.inet_ntop(socket.AF_INET6, ip_b)
    else:
        raise ValueError(f"Unexpected metadata length {len(meta)}")
    return ip, struct.unpack("!H", port_b)[0], struct.unpack("i", gpu_b)[0]


def _make_buffer(n_bytes: int, device: str, gpu: int):
    n = n_bytes // 4
    if device == "gpu":
        if not _HAS_TORCH:
            raise RuntimeError("Install torch for GPU buffers")
        buf = torch.ones(n, dtype=torch.float32, device=f"cuda:{gpu}")
        ptr = buf.data_ptr()
    else:
        if not _HAS_TORCH:
            raise RuntimeError("Install torch for pinned host buffers")
        buf = torch.ones(n, dtype=torch.float32, pin_memory=True)
        ptr = buf.data_ptr()
    return buf, ptr


def _pretty(num: int):
    units, val = ["B", "KB", "MB", "GB"], float(num)
    for u in units:
        if val < 1024 or u == units[-1]:
            return f"{val:.0f} {u}" if u == "B" else f"{val:.1f} {u}"
        val /= 1024


def send_oob(payload: bytes, port: int):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", port))
        s.listen(1)
        conn, _ = s.accept()
        with conn:
            conn.sendall(payload)


def recv_oob(ip: str, port: int) -> bytes:
    for i in range(20):
        try:
            with socket.create_connection((ip, port), timeout=2) as s:
                data = s.recv(64)
                if not data:
                    sys.exit("[Client] Empty OOB metadata")
                return data
        except (ConnectionRefusedError, socket.timeout):
            time.sleep(0.1)
    raise RuntimeError(f"Failed to connect to OOB server at {ip}:{port}")


def _run_server_read(args):
    ep = p2p.Endpoint(args.local_gpu_idx, args.num_cpus)
    send_oob(ep.get_endpoint_metadata(), args.oob_port)

    print("[Server] Waiting for connection …")
    ok, r_ip, r_gpu, conn_id = ep.accept()
    assert ok
    print(f"[Server] Connected to {r_ip} (GPU {r_gpu}) id={conn_id}")
    for sz in args.sizes:
        buf, ptr = _make_buffer(sz, args.device, args.local_gpu_idx)
        ok, mr_id = ep.reg(ptr, sz)
        assert ok
        ok, fifo_blob = ep.advertise(conn_id, mr_id, ptr, sz)
        assert ok and len(fifo_blob) == 64
        send_oob(fifo_blob, args.oob_port + 1)
    print("[Server] Benchmark complete")


def _run_client_recv(args):
    if args.remote_ip is None:
        sys.exit("--remote-ip required")

    ep_meta = recv_oob(args.remote_ip, args.oob_port)
    ip, port, r_gpu = parse_metadata(ep_meta)

    ep = p2p.Endpoint(args.local_gpu_idx, args.num_cpus)
    ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
    assert ok
    print(f"[Client] Connected to {ip}:{port} id={conn_id}")

    for sz in args.sizes:
        buf, ptr = _make_buffer(sz, args.device, args.local_gpu_idx)
        ok, mr_id = ep.reg(ptr, sz)
        assert ok
        fifo_blob = recv_oob(args.remote_ip, args.oob_port + 1)
        start = time.perf_counter()
        total = 0
        ep.read(conn_id, mr_id, ptr, sz, fifo_blob)
        start = time.perf_counter()
        total = 0
        for _ in range(args.iters):
            ep.read(conn_id, mr_id, ptr, sz, fifo_blob)
            total += sz
        elapsed = time.perf_counter() - start
        print(
            f"[Client] {_pretty(sz):>8} : "
            f"{(total*8)/elapsed/1e9:6.2f} Gbps | "
            f"{total/elapsed/1e9:6.2f} GB/s | "
            f"{elapsed/args.iters:6.6f} s"
        )
    print("[Client] Benchmark complete")


def parse_sizes(v: str) -> List[int]:
    try:
        return [int(x) for x in v.split(",") if x]
    except ValueError:
        raise argparse.ArgumentTypeError("bad --sizes")


def main():
    p = argparse.ArgumentParser("UCCL READ benchmark (one-sided)")
    p.add_argument("--role", choices=["server", "client"], required=True)
    p.add_argument("--remote-ip")
    p.add_argument("--local-gpu-idx", type=int, default=0)
    p.add_argument("--num-cpus", type=int, default=4)
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument(
        "--sizes",
        type=parse_sizes,
        default=[256, 1024, 4096, 16384, 65536, 262144, 1048576, 10485760],
    )
    p.add_argument("--iters", type=int, default=1)
    p.add_argument("--async-transfer", action="store_true")
    p.add_argument("--oob-port", type=int, default=20000)
    args = p.parse_args()

    print("Sizes:", ", ".join(_pretty(s) for s in args.sizes))
    if args.async_transfer:
        print("Async path enabled")

    if args.role == "server":
        _run_server_read(args)
    else:
        _run_client_recv(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Ctrl-C] Aborted.")
        sys.exit(1)
