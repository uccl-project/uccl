from __future__ import annotations

import argparse
import csv
import os
import socket
import sys
import time
from typing import List

import numpy as np
import torch
import torch.distributed as dist


def _make_buffer(size_bytes: int, device: str, gpu_idx: int):
    """Allocate a contiguous tensor/array of *size_bytes* and return it."""
    n_elems = size_bytes // 4  # float32
    if device == "gpu":
        tensor = torch.ones(n_elems, dtype=torch.float32, device=f"cuda:{gpu_idx}")
        return tensor
    arr = np.ones(n_elems, dtype=np.float32)
    return arr


def _pretty_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    val = float(num_bytes)
    for u in units:
        if val < 1024 or u == units[-1]:
            return f"{val:.0f} {u}" if u == "B" else f"{val:.1f} {u}"
        val /= 1024
    return f"{num_bytes} B"


def _send(tensor, dst, async_op=False):
    if isinstance(tensor, torch.Tensor):
        if async_op:
            return dist.isend(tensor, dst=dst)
        else:
            dist.send(tensor, dst=dst)
    else:  # numpy array (gloo backend)
        t = torch.from_numpy(tensor)
        if async_op:
            return dist.isend(t, dst=dst)
        else:
            dist.send(t, dst=dst)


def _recv(tensor, src, async_op=False):
    if isinstance(tensor, torch.Tensor):
        if async_op:
            return dist.irecv(tensor, src=src)
        else:
            dist.recv(tensor, src=src)
    else:
        t = torch.from_numpy(tensor)
        if async_op:
            return dist.irecv(t, src=src)
        else:
            dist.recv(t, src=src)
        tensor[:] = t.cpu().numpy()


################################################################################
# Benchmark roles
################################################################################


def _run_server(args):
    peer = 0  # client rank
    rows = []
    for size in args.sizes:
        tensor = _make_buffer(size, args.device, args.local_gpu_idx)
        # Warm-up receive
        _recv(tensor, src=peer)
        torch.cuda.synchronize() if isinstance(tensor, torch.Tensor) else None

        start = time.perf_counter()
        total = 0
        for _ in range(args.iters):
            _recv(tensor, src=peer)
            total += size
        torch.cuda.synchronize() if isinstance(tensor, torch.Tensor) else None
        elapsed = time.perf_counter() - start
        gbps = (total * 8) / elapsed / 1e9
        gb_sec = total / elapsed / 1e9
        print(
            f"[Server] {_pretty_size(size):>9} : {gbps:7.2f} Gbps | {gb_sec:7.2f} GB/s"
        )
        rows.append(
            {
                "role": "server",
                "size_bytes": size,
                "size_pretty": _pretty_size(size),
                "gbps": gbps,
                "gb_s": gb_sec,
            }
        )
    print("[Server] Benchmark complete")
    return rows


def _run_client(args):
    peer = 1  # server rank
    rows = []
    for size in args.sizes:
        tensor = _make_buffer(size, args.device, args.local_gpu_idx)
        # Warm-up send
        _send(tensor, dst=peer)
        torch.cuda.synchronize() if isinstance(tensor, torch.Tensor) else None

        start = time.perf_counter()
        total = 0
        for _ in range(args.iters):
            _send(tensor, dst=peer)
            total += size
        torch.cuda.synchronize() if isinstance(tensor, torch.Tensor) else None
        elapsed = time.perf_counter() - start
        gbps = (total * 8) / elapsed / 1e9
        gb_sec = total / elapsed / 1e9
        print(
            f"[Client] {_pretty_size(size):>9} : {gbps:7.2f} Gbps | {gb_sec:7.2f} GB/s"
        )
        rows.append(
            {
                "role": "client",
                "size_bytes": size,
                "size_pretty": _pretty_size(size),
                "gbps": gbps,
                "gb_s": gb_sec,
            }
        )
    print("[Client] Benchmark complete")
    return rows


def _run_server_dual(args):
    peer = 0  # client rank
    rows = []
    for size in args.sizes:
        tensor = _make_buffer(size, args.device, args.local_gpu_idx)
        tensor2 = _make_buffer(size, args.device, args.local_gpu_idx)
        # Warm-up receive
        _recv(tensor, src=peer)
        torch.cuda.synchronize() if isinstance(tensor, torch.Tensor) else None

        start = time.perf_counter()
        total = 0
        for _ in range(args.iters):
            recv_op = dist.P2POp(dist.irecv, tensor, peer)
            send_op = dist.P2POp(dist.isend, tensor2, peer)
            reqs = dist.batch_isend_irecv([send_op, recv_op])
            for req in reqs:
                req.wait()
            total += size
        torch.cuda.synchronize() if isinstance(tensor, torch.Tensor) else None
        elapsed = time.perf_counter() - start
        gbps = (total * 8) / elapsed / 1e9
        gb_sec = total / elapsed / 1e9
        print(
            f"[Server] {_pretty_size(size):>9} : {gbps:7.2f} Gbps | {gb_sec:7.2f} GB/s"
        )
        rows.append(
            {
                "role": "server",
                "size_bytes": size,
                "size_pretty": _pretty_size(size),
                "gbps": gbps,
                "gb_s": gb_sec,
            }
        )
    print("[Server] Benchmark complete")
    return rows


def _run_client_dual(args):

    peer = 1  # server rank
    rows = []
    for size in args.sizes:
        tensor = _make_buffer(size, args.device, args.local_gpu_idx)
        tensor2 = _make_buffer(size, args.device, args.local_gpu_idx)
        # Warm-up send
        _send(tensor, dst=peer)
        torch.cuda.synchronize() if isinstance(tensor, torch.Tensor) else None

        start = time.perf_counter()
        total = 0
        for _ in range(args.iters):
            send_op = dist.P2POp(dist.isend, tensor, peer)
            recv_op = dist.P2POp(dist.irecv, tensor2, peer)
            reqs = dist.batch_isend_irecv([send_op, recv_op])
            for req in reqs:
                req.wait()
            total += size
        torch.cuda.synchronize() if isinstance(tensor, torch.Tensor) else None
        elapsed = time.perf_counter() - start
        gbps = (total * 8) / elapsed / 1e9
        gb_sec = total / elapsed / 1e9
        print(
            f"[Client] {_pretty_size(size):>9} : {gbps:7.2f} Gbps | {gb_sec:7.2f} GB/s"
        )
        rows.append(
            {
                "role": "client",
                "size_bytes": size,
                "size_pretty": _pretty_size(size),
                "gbps": gbps,
                "gb_s": gb_sec,
            }
        )
    print("[Client] Benchmark complete")
    return rows


def parse_size_list(val: str) -> List[int]:
    try:
        return [int(s) for s in val.split(",") if s]
    except ValueError:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers")


def main():
    p = argparse.ArgumentParser(
        description="Benchmark NCCL (torch.distributed) bandwidth"
    )
    p.add_argument("--local-gpu-idx", type=int, default=0)
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument(
        "--sizes",
        type=parse_size_list,
        default=[
            256,  # 256 B
            1024,  # 1 KB
            4096,  # 4 KB
            16384,  # 16 KB
            65536,  # 64 KB
            262144,  # 256 KB
            1048576,  # 1 MB
            1048576 * 4,  # 4 MB
            1048576 * 8,  # 8 MB
            1048576 * 10,  # 10 MB
            16777216,  # 16 MB
            33554432,  # 32 MB
            67108864,  # 64 MB
            1048576 * 100,  # 100 MB
            134217728,  # 128 MB
            268435456,  # 256 MB
            536870912,  # 512 MB
            1073741824,  # 1 GB
        ],
    )
    p.add_argument("--iters", type=int, default=10)
    p.add_argument(
        "--dual",
        action="store_true",
        help="Run dual benchmark",
    )
    p.add_argument("--csv", type=str, default=None, help="Path to output CSV file")
    args = p.parse_args()

    backend = "nccl" if args.device == "gpu" else "gloo"
    dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, "This benchmark only supports 2 processes"

    print("NCCL Benchmark — role:", "client" if rank == 0 else "server")
    print("Message sizes:", ", ".join(_pretty_size(s) for s in args.sizes))
    print(
        f"Device: {args.device} | Local GPU idx: {args.local_gpu_idx} | Iters: {args.iters}"
    )

    try:
        if args.dual:
            if rank == 0:
                rows = _run_client_dual(args)
            else:
                rows = _run_server_dual(args)
        else:
            if rank == 0:
                rows = _run_client(args)
            else:
                rows = _run_server(args)
    finally:
        dist.destroy_process_group()

    if args.csv and rows:
        fieldnames = ["role", "size_bytes", "size_pretty", "gbps", "gb_s"]
        write_header = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        print(f"Results saved to {args.csv}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted] Benchmark aborted by user.")
        sys.exit(1)
