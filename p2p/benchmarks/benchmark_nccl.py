from __future__ import annotations

import argparse
import os
import socket
import sys
import time
import statistics
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


def _run_role(args, sending):
    peer = 1 if sending else 0
    role = "Client" if sending else "Server"
    operation = _send if sending else _recv
    for size in args.sizes:
        tensor = _make_buffer(size, args.device, args.local_gpu_idx)
        # Warm up communicator kernels and clocks before collecting samples.
        for _ in range(args.warmup_iters):
            operation(tensor, peer)
        if args.device == "gpu":
            torch.cuda.synchronize()
        samples = []
        iterations = args.iters
        for _ in range(args.repeats):
            while True:
                dist.barrier()
                if args.device == "gpu":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(iterations):
                    operation(tensor, peer)
                if args.device == "gpu":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                # Both ranks must agree on the next iteration count. Use the
                # slower participant so this measures completed transfers.
                duration = torch.tensor(
                    [elapsed],
                    dtype=torch.float64,
                    device=tensor.device if args.device == "gpu" else "cpu",
                )
                dist.all_reduce(duration, op=dist.ReduceOp.MAX)
                elapsed = duration.item()
                if elapsed >= args.min_seconds:
                    break
                iterations = max(
                    iterations + 1, int(iterations * args.min_seconds / elapsed * 1.1)
                )
            samples.append(size * iterations * 8 / elapsed / 1e9)
        gbps = statistics.median(samples)
        print(
            f"[{role}] {_pretty_size(size):>9} : {gbps:7.2f} Gbps | {gbps / 8:7.2f} GB/s"
        )
        print(
            f"[{role}] {args.repeats} trials: min={min(samples):.2f}, max={max(samples):.2f} Gbps, last_iters={iterations}"
        )
    print(f"[{role}] Benchmark complete")


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
    )
    p.add_argument("--iters", type=int, default=128)
    p.add_argument("--warmup-iters", type=int, default=64)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument(
        "--min-seconds",
        type=float,
        default=0.2,
        help="Minimum duration per trial; 0 keeps the exact --iters count",
    )
    args = p.parse_args()
    if (
        args.iters <= 0
        or args.repeats <= 0
        or args.warmup_iters < 0
        or args.min_seconds < 0
    ):
        p.error(
            "iters/repeats must be positive; warmup-iters/min-seconds must be nonnegative"
        )
    if args.device == "gpu":
        torch.cuda.set_device(args.local_gpu_idx)

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
        if rank == 0:
            _run_role(args, sending=True)
        else:
            _run_role(args, sending=False)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted] Benchmark aborted by user.")
        sys.exit(1)
