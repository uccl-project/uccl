"""Benchmark UCCL P2P write bandwidth with optional compression."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import List, Tuple

import torch
import torch.distributed as dist

from uccl import p2p

_DTYPE_ITEM_SIZE = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float8_e4m3fn: 1,
    torch.float8_e5m2: 1,
}

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}


_P2P_FLOAT_TYPE = {
    "float16": p2p.FloatType.kFloat16,
    "bfloat16": p2p.FloatType.kBFloat16,
    "float32": p2p.FloatType.kFloat32,
    # The P2P compression type enum does not expose fp8 today.
    "float8_e4m3fn": p2p.FloatType.kUndefined,
    "float8_e5m2": p2p.FloatType.kUndefined,
}


def _make_buffer(size_bytes: int, dtype: torch.dtype = torch.float32):
    """Allocate a contiguous GPU tensor of *size_bytes* filled with values
    sampled from a standard normal distribution clipped to [0, 1]."""
    item_size = _DTYPE_ITEM_SIZE[dtype]
    n_elems = size_bytes // item_size
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # fp8 types don't support uniform_ directly; create in float32 then convert
        tensor = (
            torch.empty(n_elems, device="cuda", dtype=torch.float32)
            .uniform_(-1, 1)
            .to(dtype)
        )
    else:
        tensor = torch.empty(n_elems, device="cuda", dtype=dtype).uniform_(-1, 1)
    assert tensor.is_contiguous()
    assert tensor.device.type == "cuda"
    return tensor


def _pretty_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    val = float(num_bytes)
    for u in units:
        if val < 1024 or u == units[-1]:
            return f"{val:.0f} {u}" if u == "B" else f"{val:.1f} {u}"
        val /= 1024
    return f"{num_bytes} B"


################################################################################
# Benchmark roles
################################################################################


def _exchange_metadata(local_metadata: bytes, rank: int) -> bytes:
    if rank == 0:
        dist.send(torch.ByteTensor(list(local_metadata)), dst=1)
        remote = torch.zeros(len(local_metadata), dtype=torch.uint8)
        dist.recv(remote, src=1)
    else:
        remote = torch.zeros(len(local_metadata), dtype=torch.uint8)
        dist.recv(remote, src=0)
        dist.send(torch.ByteTensor(list(local_metadata)), dst=0)
    return bytes(remote.tolist())


def _send_bytes(payload: bytes, dst: int) -> None:
    size = torch.tensor([len(payload)], dtype=torch.int64)
    dist.send(size, dst=dst)
    if payload:
        dist.send(torch.ByteTensor(list(payload)), dst=dst)


def _recv_bytes(src: int) -> bytes:
    size = torch.zeros(1, dtype=torch.int64)
    dist.recv(size, src=src)
    n = int(size.item())
    if n == 0:
        return b""
    payload = torch.zeros(n, dtype=torch.uint8)
    dist.recv(payload, src=src)
    return bytes(payload.tolist())


def _run_server(args, ep: "p2p.Endpoint") -> List[Tuple]:
    peer = 0  # client rank
    results = []
    ok, r_ip, r_gpu, conn_id = ep.accept()
    assert ok, "[Server] Failed to accept RDMA connection"
    print(f"[Server] Accept from {r_ip} (GPU {r_gpu}) conn_id={conn_id}")

    ft = _P2P_FLOAT_TYPE[args.dtype]
    for size in args.sizes:
        tensor = _make_buffer(size, args.torch_dtype)
        ok, mr_id = ep.reg(tensor.data_ptr(), size, ft)
        assert ok, "[Server] register failed"

        ok, fifo_blob = ep.advertise(conn_id, mr_id, tensor.data_ptr(), size)
        assert ok, "[Server] advertise failed"
        _send_bytes(bytes(fifo_blob), dst=peer)

        # The initiator measures write completion; the target waits at the
        # barrier before reusing the advertised buffer for the next size.
        dist.barrier()
        ep.dereg(mr_id)
        print(f"[Server] {_pretty_size(size):>9} : advertised write target")
        results.append(("Server", size, _pretty_size(size), 0.0, 0.0))
    print("[Server] Benchmark complete")
    return results


def _run_client(args, ep: "p2p.Endpoint", remote_metadata: bytes) -> List[Tuple]:
    peer = 1  # server rank
    results = []
    ip, port, r_gpu = p2p.Endpoint.parse_metadata(remote_metadata)
    ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
    assert ok, "[Client] Failed to connect to server"
    print(f"[Client] Connected to {ip}:{port} (GPU {r_gpu}) conn_id={conn_id}")

    ft = _P2P_FLOAT_TYPE[args.dtype]
    for size in args.sizes:
        tensor = _make_buffer(size, args.torch_dtype)
        ok, mr_id = ep.reg(tensor.data_ptr(), size, ft)
        assert ok, "[Client] register failed"
        fifo_blob = _recv_bytes(src=peer)

        def write_once() -> None:
            if args.async_api:
                ok, transfer_id = ep.write_async(
                    conn_id, mr_id, tensor.data_ptr(), size, fifo_blob
                )
                assert ok, "[Client] write_async error"
                is_done = False
                while not is_done:
                    ok, is_done = ep.poll_async(transfer_id)
                    assert ok, "[Client] poll_async error"
            else:
                ok = ep.write(conn_id, mr_id, tensor.data_ptr(), size, fifo_blob)
                assert ok, "[Client] write error"

        write_once()

        start = time.perf_counter()
        total = 0
        for _ in range(args.iters):
            write_once()
            total += size

        elapsed = time.perf_counter() - start

        gbps = (total * 8) / elapsed / 1e9
        gb_sec = total / elapsed / 1e9
        dist.barrier()
        ep.dereg(mr_id)
        print(
            f"[Client] {_pretty_size(size):>9} : {gbps:7.2f} Gbps | {gb_sec:7.2f} GB/s"
        )
        results.append(("Client", size, _pretty_size(size), gbps, gb_sec))
    print("[Client] Benchmark complete")
    return results


def parse_size_list(val: str) -> List[int]:
    try:
        return [int(s) for s in val.split(",") if s]
    except ValueError:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers")


def main():
    p = argparse.ArgumentParser(description="Benchmark UCCL P2P write bandwidth")
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
    p.add_argument(
        "--async-api",
        action="store_true",
        help="Use asynchronous write transfers",
    )
    p.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32", "float8_e4m3fn", "float8_e5m2"],
        default="float32",
        help="Float dtype for tensors (default: float32)",
    )
    p.add_argument(
        "--csv",
        metavar="FILE",
        default=None,
        help="Path to CSV file to save benchmark results; when world_size > 1, "
        "_rank{N} is inserted before the extension for each rank",
    )
    args = p.parse_args()

    # Convert dtype string to torch dtype
    args.torch_dtype = _DTYPE_MAP[args.dtype]

    # Initialize torch.distributed with gloo backend for coordination
    dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size != 2:
        print("ERROR: Default client-server benchmark requires exactly 2 processes.")
        sys.exit(1)

    try:
        local_gpu_idx = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_gpu_idx)
        ep = p2p.Endpoint(local_gpu_idx)
        remote_metadata = _exchange_metadata(bytes(ep.get_metadata()), rank)

        print(
            "UCCL P2P Compression Write Benchmark — role:",
            "client" if rank == 0 else "server",
        )
        print("Message sizes:", ", ".join(_pretty_size(s) for s in args.sizes))
        print(
            f"Device: GPU | Local GPU idx: {local_gpu_idx} | Iters: {args.iters} | Dtype: {args.dtype}"
        )
        print(
            "Using async write API" if args.async_api else "Using synchronous write API"
        )

        # Synchronize all ranks before starting benchmark
        dist.barrier()

        if rank == 0:
            results = _run_client(args, ep, remote_metadata)
        else:
            results = _run_server(args, ep)

        # Write results to CSV if requested
        if args.csv:
            if world_size > 1:
                base, ext = os.path.splitext(args.csv)
                csv_path = f"{base}_rank{rank}{ext}"
            else:
                csv_path = args.csv
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["role", "size_bytes", "size_pretty", "gbps", "gb_sec"])
                writer.writerows(results)
            print(f"Results saved to {csv_path}")

        # Synchronize all ranks before finishing
        dist.barrier()
        print("Benchmark completed successfully!")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted] Benchmark aborted by user.")
        sys.exit(1)
