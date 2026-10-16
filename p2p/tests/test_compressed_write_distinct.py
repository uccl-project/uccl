"""
Regression test for compressed P2P writes (#1071): a completed write must
leave exactly the written bytes in the receiver's buffer.

The writer (rank 0) issues several back-to-back synchronous writes with
distinct BF16 payloads into distinct receiver buffers, then tells the
receiver (rank 1) that all writes have completed. The receiver checks every
buffer once, without waiting: a completed write must already have been
decompressed into place. Before the fix the receiver acked a compressed write
when its meta arrived, so the sender could reuse the same DecompressArena
range for the next write while the previous decompression was still reading
it, and buffers ended up with wrong contents.

An arena exactly one message long makes every write reuse offset 0, the
setting that exposes the race most often.

Usage (two ranks, one per node or two local GPUs):
  UCCL_P2P_COMPRESS_STRATEGY=split \\
  UCCL_P2P_MIN_COMPRESS_BYTES=65536 \\
  UCCL_P2P_COMPRESS_BUFFER_BYTES=16777216 \\
  torchrun --nnodes=2 --nproc_per_node=1 ... \\
    tests/test_compressed_write_distinct.py --sizes 16777216
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
from uccl import p2p


def digest(t: torch.Tensor) -> list[int]:
    x = t.view(torch.int16).to(torch.int64)
    idx = torch.arange(x.numel(), device=x.device, dtype=torch.int64)
    w = (idx * 2654435761) % 1000003 + 1
    return [int(x.sum().item()), int((x * w).sum().item())]


def payload(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    t = torch.empty(n, device="cuda", dtype=torch.bfloat16)
    return t.uniform_(-1, 1, generator=g)


def exchange_metadata(ep, rank: int) -> bytes:
    md = bytes(ep.get_metadata())
    other = torch.zeros(len(md), dtype=torch.uint8)
    if rank == 0:
        dist.send(torch.ByteTensor(list(md)), dst=1)
        dist.recv(other, src=1)
    else:
        dist.recv(other, src=0)
        dist.send(torch.ByteTensor(list(md)), dst=0)
    return bytes(other.tolist())


def receiver_round(ep, n: int, size: int, buffers: int) -> int:
    ft = p2p.FloatType.kBFloat16
    bufs, mrs = [], []
    for _ in range(buffers):
        b = torch.empty(n, device="cuda", dtype=torch.bfloat16)
        b.view(torch.int16).fill_(-1)  # NaN pattern: never a valid payload
        ok, mr = ep.reg(b.data_ptr(), size, ft)
        assert ok, "reg failed"
        bufs.append(b)
        mrs.append(mr)
    torch.cuda.synchronize()
    for b, mr in zip(bufs, mrs):
        ok, blob = ep.advertise(mr, b.data_ptr(), size)
        assert ok, "advertise failed"
        blob = bytes(blob)
        dist.send(torch.tensor([len(blob)], dtype=torch.int64), dst=0)
        dist.send(torch.ByteTensor(list(blob)), dst=0)
    expected = torch.zeros(buffers, 2, dtype=torch.int64)
    dist.recv(expected, src=0)  # sent only after every write completed
    bad = [j for j, b in enumerate(bufs) if digest(b) != expected[j].tolist()]
    dist.barrier()
    for mr in mrs:
        ep.dereg(mr)
    return len(bad)


def writer_round(ep, conn_id, n: int, size: int, seeds: list[int]) -> None:
    ft = p2p.FloatType.kBFloat16
    srcs, mrs, digs = [], [], []
    for seed in seeds:
        s = payload(n, seed)
        ok, mr = ep.reg(s.data_ptr(), size, ft)
        assert ok, "reg failed"
        srcs.append(s)
        mrs.append(mr)
        digs.append(digest(s))
    torch.cuda.synchronize()
    blobs = []
    for _ in seeds:
        ln = torch.zeros(1, dtype=torch.int64)
        dist.recv(ln, src=1)
        bb = torch.zeros(int(ln.item()), dtype=torch.uint8)
        dist.recv(bb, src=1)
        blobs.append(bytes(bb.tolist()))
    for s, mr, blob in zip(srcs, mrs, blobs):
        assert ep.write(conn_id, mr, s.data_ptr(), size, blob), "write failed"
    dist.send(torch.tensor(digs, dtype=torch.int64), dst=1)
    dist.barrier()
    for mr in mrs:
        ep.dereg(mr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="16777216")
    ap.add_argument("--buffers", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261014)
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",") if s]

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    assert dist.get_world_size() == 2, "run with exactly two ranks"
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    ep = p2p.Endpoint(torch.cuda.current_device())
    remote_md = exchange_metadata(ep, rank)
    if rank == 0:
        ip, port, r_gpu = p2p.Endpoint.parse_metadata(remote_md)
        ok, conn_id = ep.connect(ip, r_gpu, remote_port=port)
    else:
        ok, _, _, conn_id = ep.accept()
    assert ok, "connect/accept failed"

    failures = 0
    for size in sizes:
        n = size // 2
        for rnd in range(args.rounds):
            seeds = [
                args.seed + (size % 100003) * 1000 + rnd * args.buffers + j
                for j in range(args.buffers)
            ]
            if rank == 0:
                writer_round(ep, conn_id, n, size, seeds)
            else:
                bad = receiver_round(ep, n, size, args.buffers)
                print(f"size {size} round {rnd}: {bad}/{args.buffers} wrong")
                failures += bad

    result = torch.tensor([failures], dtype=torch.int64)
    dist.broadcast(result, src=1)
    dist.barrier()
    dist.destroy_process_group()
    assert result.item() == 0, f"{result.item()} buffers had wrong contents"
    if rank == 1:
        print("PASS")


if __name__ == "__main__":
    main()
