from __future__ import annotations

import argparse
import sys
import time
from typing import List
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import kvtrans_engine as ke
except ImportError as exc:
    sys.stderr.write("Failed to import kvtrans_engine — did you run `make`?\n")
    raise

_HAS_TORCH = False
try:
    import torch
    print("Torch imported")

    _HAS_TORCH = True
except ModuleNotFoundError:
    pass

import numpy as np

def _make_buffer(size_bytes: int, device: str, gpu_idx: int):
    """Allocate a contiguous buffer of *size_bytes* and return (buffer, ptr)."""
    n_elems = size_bytes // 4  # float32 elements
    if device == "gpu":
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required for GPU buffers (install torch)")
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
    ep = ke.Endpoint(args.local_gpu_idx, args.num_cpus)
    print("[Server] Waiting for connection …", flush=True)
    ok, r_ip, r_gpu, conn_id = ep.accept()
    if not ok:
        sys.exit("[Server] Failed to accept RDMA connection")
    print(f"[Server] Connected to {r_ip} (GPU {r_gpu}) conn_id={conn_id}")

    for size in args.sizes:
        buf, ptr = _make_buffer(size, args.device, args.local_gpu_idx)
        ok, mr_id = ep.reg_kv(ptr, size)
        if not ok:
            sys.exit("[Server] reg_kv failed")

        ep.recv_kv(conn_id, mr_id, ptr, size)
        start = time.perf_counter()
        total_recv = 0
        for _ in range(args.iters):
            ok, recv_sz = ep.recv_kv(conn_id, mr_id, ptr, size)
            if not ok or recv_sz != size:
                sys.exit("[Server] recv_kv error")
            total_recv += recv_sz
        elapsed = time.perf_counter() - start
        gbps = (total_recv * 8) / elapsed / 1e9  # bits per second → Gbps
        gb_sec = total_recv / elapsed / 1e9  # bytes per second → GB/s
        print(f"[Server] {_pretty_size(size):>8} : {gbps:6.2f} Gbps | {gb_sec:6.2f} GB/s")
    print("[Server] Benchmark complete")


def _run_client(args):
    if args.remote_ip is None:
        sys.exit("[Client] --remote-ip is required")
    ep = ke.Endpoint(args.local_gpu_idx, args.num_cpus)
    ok, conn_id = ep.connect(args.remote_ip, args.remote_gpu_idx)
    if not ok:
        sys.exit("[Client] Failed to connect to server")
    print(f"[Client] Connected to {args.remote_ip} conn_id={conn_id}")

    for size in args.sizes:
        buf, ptr = _make_buffer(size, args.device, args.local_gpu_idx)
        ok, mr_id = ep.reg_kv(ptr, size)
        if not ok:
            sys.exit("[Client] reg_kv failed")

        ep.send_kv(conn_id, mr_id, ptr, size)
        start = time.perf_counter()
        total_sent = 0
        for _ in range(args.iters):
            ok = ep.send_kv(conn_id, mr_id, ptr, size)
            if not ok:
                sys.exit("[Client] send_kv error")
            total_sent += size
        elapsed = time.perf_counter() - start
        gbps = (total_sent * 8) / elapsed / 1e9
        gb_sec = total_sent / elapsed / 1e9
        print(f"[Client] {_pretty_size(size):>8} : {gbps:6.2f} Gbps | {gb_sec:6.2f} GB/s")
    print("[Client] Benchmark complete")

def parse_size_list(val: str) -> List[int]:
    try:
        return [int(s) for s in val.split(",") if s]
    except ValueError:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers")


def main():
    p = argparse.ArgumentParser(description="Benchmark KVTrans Engine bandwidth")
    p.add_argument("--role", choices=["server", "client"], required=True,
                   help="Run as server (receiver) or client (sender)")
    p.add_argument("--remote-ip", help="Server IP address (client only)")
    p.add_argument("--remote-gpu-idx", type=int, default=0,
                   help="Server GPU index (client only)")
    p.add_argument("--local-gpu-idx", type=int, default=0,
                   help="Local GPU index to bind buffers")
    p.add_argument("--num-cpus", type=int, default=4,
                   help="#CPU threads for RDMA ops")
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                   help="Buffer location (cpu or gpu)")
    p.add_argument("--sizes", type=parse_size_list,
                   default=[256, 1024, 4096, 16384, 65536, 262144, 1048576, 10485760, 104857600],
                   help="Comma separated list of message sizes in bytes")
    p.add_argument("--iters", type=int, default=1000,
                   help="Iterations per message size (excluding 1 warm-up)")
    args = p.parse_args()

    print("KVTrans Benchmark — role:", args.role)
    print("Message sizes:", ", ".join(_pretty_size(s) for s in args.sizes))
    print(f"Device: {args.device} | Local GPU idx: {args.local_gpu_idx} | Iterations: {args.iters}")
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