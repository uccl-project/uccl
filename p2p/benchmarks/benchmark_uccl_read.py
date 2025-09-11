from __future__ import annotations
import argparse, sys, time, socket, struct
from typing import List
import torch.distributed as dist
import torch
import numpy as np
import os

try:
    from uccl import p2p
except ImportError:
    sys.stderr.write("Failed to import p2p\n")
    raise


# parse_metadata is now provided by the C++ layer via p2p.Endpoint.parse_metadata()


def _make_buffer(n_bytes: int, device: str, gpu: int):
    n = n_bytes // 4
    if device == "gpu":
        buf = torch.ones(n, dtype=torch.float32, device=f"cuda:{gpu}")
        ptr = buf.data_ptr()
    else:
        buf = torch.ones(n, dtype=torch.float32, pin_memory=True)
        ptr = buf.data_ptr()
    return buf, ptr


def _pretty(num: int):
    units, val = ["B", "KB", "MB", "GB"], float(num)
    for u in units:
        if val < 1024 or u == units[-1]:
            return f"{val:.0f} {u}" if u == "B" else f"{val:.1f} {u}"
        val /= 1024


def _run_server_read(args, ep, remote_metadata):
    peer = 0
    print("[Server] Waiting for connection …")
    ok, r_ip, r_gpu, conn_id = ep.accept()
    assert ok
    print(f"[Server] Connected to {r_ip} (GPU {r_gpu}) id={conn_id}")
    for sz in args.sizes:
        size_per_block = sz // args.num_iovs
        buf_v = []
        ptr_v = []
        mr_id_v = []
        size_v = []
        for _ in range(args.num_iovs):
            buf, ptr = _make_buffer(size_per_block, args.device, args.local_gpu_idx)
            ok, mr_id = ep.reg(ptr, size_per_block)
            assert ok
            buf_v.append(buf)
            ptr_v.append(ptr)
            mr_id_v.append(mr_id)
            size_v.append(size_per_block)
        # Use advertisev to advertise all blocks at once
        ok, fifo_blob_v = ep.advertisev(conn_id, mr_id_v, ptr_v, size_v, args.num_iovs)
        assert ok and all(len(fifo_blob) == 64 for fifo_blob in fifo_blob_v)
        # Send all fifo_blobs to peer
        for fifo_blob in fifo_blob_v:
            dist.send(torch.ByteTensor(list(fifo_blob)), dst=peer)
    print("[Server] Benchmark complete")


def _run_client_recv(args, ep, remote_metadata):
    peer = 1
    ip, port, r_gpu = p2p.Endpoint.parse_metadata(remote_metadata)
    ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
    assert ok
    print(f"[Client] Connected to {ip}:{port} id={conn_id}")

    for sz in args.sizes:
        size_per_block = sz // args.num_iovs
        buf_v = []
        ptr_v = []
        mr_id_v = []
        size_v = []
        fifo_blob_v = []
        for _ in range(args.num_iovs):
            buf, ptr = _make_buffer(size_per_block, args.device, args.local_gpu_idx)
            ok, mr_id = ep.reg(ptr, size_per_block)
            assert ok
            buf_v.append(buf)
            ptr_v.append(ptr)
            mr_id_v.append(mr_id)
            size_v.append(size_per_block)
        for _ in range(args.num_iovs):
            fifo_blob = torch.zeros(64, dtype=torch.uint8)
            dist.recv(fifo_blob, src=peer)
            fifo_blob_v.append(bytes(fifo_blob.tolist()))
        start = time.perf_counter()
        total = 0
        if args.async_api:
            ok, transfer_id = ep.read_async(
                conn_id, mr_id_v[0], ptr_v[0], size_v[0], fifo_blob_v[0]
            )
            assert ok
            is_done = False
            while not is_done:
                ok, is_done = ep.poll_async(transfer_id)
                assert ok
        else:
            ep.readv(conn_id, mr_id_v, ptr_v, size_v, fifo_blob_v, args.num_iovs)
        start = time.perf_counter()
        total = 0
        for _ in range(args.iters):
            if args.async_api:
                ok, transfer_id = ep.read_async(
                    conn_id, mr_id_v[0], ptr_v[0], size_v[0], fifo_blob_v[0]
                )
                assert ok
                is_done = False
                while not is_done:
                    ok, is_done = ep.poll_async(transfer_id)
                    assert ok
                total += size_v[0]
            else:
                ep.readv(conn_id, mr_id_v, ptr_v, size_v, fifo_blob_v, args.num_iovs)
                total += sum(size_v)
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
    p.add_argument("--local-gpu-idx", type=int, default=0)
    p.add_argument("--num-cpus", type=int, default=4)
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument(
        "--sizes",
        type=parse_sizes,
        default=[
            256,
            1024,
            4096,
            16384,
            65536,
            262144,
            1048576,
            10485760,
            67108864,
            104857600,
        ],
    )
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--async-api", action="store_true")
    p.add_argument(
        "--num-iovs",
        type=int,
        default=1,
        help="Number of iovs to read in a single call",
    )
    args = p.parse_args()

    if args.async_api:
        assert args.num_iovs == 1, "Async transfers only support one iov"

    print("Sizes:", ", ".join(_pretty(s) for s in args.sizes))
    if args.async_api:
        print("Async path enabled")

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, "This benchmark only supports 2 processes"

    ep = p2p.Endpoint(args.local_gpu_idx, args.num_cpus)
    local_metadata = ep.get_metadata()

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

    if rank == 0:
        _run_client_recv(args, ep, remote_metadata)
    elif rank == 1:
        _run_server_read(args, ep, remote_metadata)

    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Ctrl-C] Aborted.")
        sys.exit(1)
