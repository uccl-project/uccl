from __future__ import annotations

import argparse
import sys
import time
from typing import List
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import uccl_p2p
except ImportError as exc:
    sys.stderr.write("Failed to import uccl_p2p — did you run `make`?\n")
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
    return f"{num_bytes} B"


def _run_server(args, ep: uccl_p2p.Endpoint, conn_id: int):
    print(f"[Server] Starting benchmark on conn_id={conn_id}\n", flush=True)
    for size in args.sizes:
        buf, ptr = _make_buffer(size, args.device, args.local_gpu_idx)
        ok, mr_id = ep.reg(ptr, size)
        if not ok:
            sys.exit("[Server] register failed")

        # Warm-up
        print("[Server] Warm-up send ...", flush=True)
        ep.recv(conn_id, mr_id, ptr, size)

        # Timed loop
        start = time.perf_counter()
        total_recv = 0
        for _ in range(args.iters):
            ok, recv_sz = ep.recv(conn_id, mr_id, ptr, size)
            if not ok or recv_sz != size:
                sys.exit("[Server] recv error")
            total_recv += recv_sz
        elapsed = time.perf_counter() - start

        gbps  = (total_recv * 8) / elapsed / 1e9
        gbsec = total_recv / elapsed / 1e9
        print(f"[Server] {_pretty_size(size):>8} : {gbps:6.2f} Gbps | {gbsec:6.2f} GB/s")
    print("[Server] Benchmark complete")


def _run_client(args, ep: uccl_p2p.Endpoint, conn_id: int):
    print(f"[Client] Starting benchmark on conn_id={conn_id}\n", flush=True)
    for size in args.sizes:
        buf, ptr = _make_buffer(size, args.device, args.local_gpu_idx)
        ok, mr_id = ep.reg(ptr, size)
        if not ok:
            sys.exit("[Client] register failed")

        # Warm-up
        print("[Client] Warm-up send ...", flush=True)
        ep.send(conn_id, mr_id, ptr, size)

        # Timed loop
        start = time.perf_counter()
        total_sent = 0
        for _ in range(args.iters):
            ok = ep.send(conn_id, mr_id, ptr, size)
            if not ok:
                sys.exit("[Client] send error")
            total_sent += size
        elapsed = time.perf_counter() - start

        gbps  = (total_sent * 8) / elapsed / 1e9
        gbsec = total_sent / elapsed / 1e9
        print(f"[Client] {_pretty_size(size):>8} : {gbps:6.2f} Gbps | {gbsec:6.2f} GB/s")
    print("[Client] Benchmark complete")


def parse_size_list(val: str) -> List[int]:
    try:
        return [int(s) for s in val.split(",") if s]
    except ValueError:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers")


def main():
    p = argparse.ArgumentParser(
        description="Benchmark UCCL P2P Engine bandwidth"
    )
    p.add_argument("--role", choices=["server", "client"], required=True,
                   help="Run as server (receiver) or client (sender)")
    # Legacy args for direct connect/accept:
    p.add_argument("--remote-ip", help="Server IP address (client only)")
    # Rendezvous args:
    p.add_argument("--discovery-uri",
                   help="Discovery URI for rendezvous (e.g. redis://127.0.0.1:6379)")
    p.add_argument("--group-name", help="Logical group name for rendezvous")
    p.add_argument("--world-size", type=int,
                   help="Total number of ranks in rendezvous")
    p.add_argument("--my-rank", type=int,
                   help="This process’s rank (0-based) for rendezvous")
    p.add_argument("--listen-port", type=int, default=12345,
                   help="TCP port bound by this process (passed to join_group)")
    p.add_argument("--local-gpu-idx", type=int, default=0,
                   help="Local GPU index to bind buffers")
    p.add_argument("--num-cpus", type=int, default=4,
                   help="#CPU threads for RDMA ops")
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                   help="Buffer location (cpu or gpu)")
    p.add_argument("--sizes", type=parse_size_list,
                   default=[256, 1024, 4096, 16384, 65536,
                            262144, 1048576, 10485760, 104857600],
                   help="Comma separated list of message sizes in bytes")
    p.add_argument("--iters", type=int, default=1000,
                   help="Iterations per message size (excluding warm-up)")
    args = p.parse_args()

    print("UCCL P2P Benchmark — role:", args.role)
    print("Message sizes:", ", ".join(_pretty_size(s) for s in args.sizes))
    print(f"Device: {args.device} | Local GPU idx: {args.local_gpu_idx} | Iterations: {args.iters}")

    # 1) Create & connect via rendezvous, or fallback to legacy
    if args.discovery_uri:
        # Rendezvous: static CreateAndJoin already publishes, waits, and connects
        ep = uccl_p2p.Endpoint.CreateAndJoin(
            args.discovery_uri,
            args.group_name,
            args.world_size,
            args.my_rank,
            args.local_gpu_idx,
            args.num_cpus,
            args.remote_gpu_idx
        )
        # In a 2-node world, peer rank = the one ≠ my_rank
        peers = [r for r in range(args.world_size) if r != args.my_rank]
        if len(peers) != 1:
            sys.exit("[Error] rendezvous script only supports world_size=2")
        conn_id = ep.conn_id_of_rank(peers[0])
        print(f"[{args.role.title()}] Rendezvous complete → conn_id={conn_id}")
    else:
        # Legacy direct connect/accept
        ep = uccl_p2p.Endpoint(args.local_gpu_idx, args.num_cpus)
        if args.role == "server":
            print("[Server] Waiting for connection …", flush=True)
            ok, r_ip, r_gpu, conn_id = ep.accept()
            if not ok:
                sys.exit("[Server] Failed to accept RDMA connection")
            print(f"[Server] Connected to {r_ip} (GPU {r_gpu}) conn_id={conn_id}")
        else:
            if not args.remote_ip:
                sys.exit("[Client] --remote-ip is required")
            ok, conn_id = ep.connect(args.remote_ip, args.remote_gpu_idx)
            if not ok:
                sys.exit("[Client] Failed to connect to server")
            print(f"[Client] Connected to {args.remote_ip} conn_id={conn_id}")

    # 2) Run the appropriate benchmark
    if args.role == "server":
        _run_server(args, ep, conn_id)
    else:
        _run_client(args, ep, conn_id)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted] Benchmark aborted by user.")
        sys.exit(1)