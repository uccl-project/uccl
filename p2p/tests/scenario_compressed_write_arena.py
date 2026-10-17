"""Exercise compressed writes with distinct payloads across arena wraparound.

Run on two RDMA-connected GPU processes with USE_DIETGPU=1:
  UCCL_P2P_COMPRESS_STRATEGY=split_only \
  UCCL_P2P_COMPRESS_BUFFER_BYTES=16777216 \
  torchrun --nnodes=2 --nproc_per_node=1 --node_rank=... --master_addr=... \
      tests/scenario_compressed_write_arena.py --bytes=16777216

The receiver retains every destination until the whole batch is verified.
For dispatch diagnostics, set UCCL_P2P_LOG_LEVEL=INFO and UCCL_DEBUG_SUBSYS=RDMA;
compressed reception logs include handle_compressed_write_arrival. A raw-path
run alone does not establish compression coverage.
Reported bandwidth includes the receiver's exact-byte checks and Gloo reply.
Use an arena of at least --bytes * --batch for a valid pre-fix timing baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from uccl import p2p


def payload(nbytes: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    count = nbytes // torch.empty((), dtype=dtype).element_size()
    return (
        torch.empty(count, dtype=torch.float32)
        .uniform_(-1, 1, generator=generator)
        .to(dtype=dtype, device="cuda")
    )


def write_batch(ep, conn, ids, tensors, blobs, api, timeout):
    if api == "write":
        for mr_id, tensor, blob in zip(ids, tensors, blobs):
            if not ep.write(conn, mr_id, tensor.data_ptr(), tensor.nbytes, blob):
                raise RuntimeError("compressed write failed")
        return

    handles = []
    for mr_id, tensor, blob in zip(ids, tensors, blobs):
        ok, handle = ep.write_async(conn, mr_id, tensor.data_ptr(), tensor.nbytes, blob)
        if not ok:
            raise RuntimeError("compressed write_async submission failed")
        handles.append(handle)
    deadline = time.monotonic() + timeout
    for handle in handles:
        while True:
            ok, done = ep.poll_async(handle)
            if not ok:
                raise RuntimeError("compressed write_async completion failed")
            if done:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("compressed write_async completion timed out")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--api", choices=("write", "write_async"), default="write")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    itemsize = torch.empty((), dtype=dtype).element_size()
    if args.bytes <= 0 or args.bytes % itemsize:
        parser.error("--bytes must be a positive multiple of the dtype size")
    if args.batch < 2 or args.rounds < 1 or args.warmup < 0 or args.timeout <= 0:
        parser.error("batch >= 2, rounds >= 1, warmup >= 0 and timeout > 0 required")
    if os.environ.get("UCCL_P2P_TRANSPORT", "rdma") != "rdma":
        parser.error("this scenario requires UCCL_P2P_TRANSPORT=rdma")
    strategy = os.environ.get("UCCL_P2P_COMPRESS_STRATEGY", "")
    if strategy not in ("split", "split_only"):
        parser.error("set UCCL_P2P_COMPRESS_STRATEGY=split_only")
    minimum = int(os.environ.get("UCCL_P2P_MIN_COMPRESS_BYTES", 16 * 1024 * 1024))
    arena = int(os.environ.get("UCCL_P2P_COMPRESS_BUFFER_BYTES", 2 * 1024**3))
    if args.bytes < minimum or args.bytes > arena:
        parser.error(
            "message size must reach the compression threshold and fit the arena"
        )

    linkage = subprocess.run(
        ["ldd", p2p.__file__], capture_output=True, text=True, check=True
    )
    if "libdietgpu_float.so =>" not in linkage.stdout or "not found" in linkage.stdout:
        parser.error("the loaded P2P extension requires a native USE_DIETGPU=1 build")

    dist.init_process_group("gloo", timeout=timedelta(seconds=args.timeout + 120))
    if dist.get_world_size() != 2:
        raise RuntimeError("this scenario requires exactly two processes")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    float_type = getattr(
        p2p.FloatType,
        {"bfloat16": "kBFloat16", "float16": "kFloat16", "float32": "kFloat32"}[
            args.dtype
        ],
    )
    ep = p2p.Endpoint(local_gpu_idx=torch.cuda.current_device())
    tensors = [
        torch.empty(args.bytes // itemsize, dtype=dtype, device="cuda")
        for _ in range(args.batch)
    ]
    ids = []
    for tensor in tensors:
        ok, mr_id = ep.reg(tensor.data_ptr(), tensor.nbytes, float_type)
        if not ok:
            raise RuntimeError("memory registration failed")
        ids.append(mr_id)

    metadata = [None, None]
    dist.all_gather_object(metadata, bytes(ep.get_metadata()))
    if rank == 0:
        ip, port, gpu = p2p.Endpoint.parse_metadata(metadata[1])
        ok, conn = ep.connect(ip, gpu, remote_port=port)
    else:
        ok, _, _, conn = ep.accept()
    if not ok:
        raise RuntimeError("RDMA connection failed")
    blobs = []
    if rank == 1:
        for mr_id, tensor in zip(ids, tensors):
            ok, blob = ep.advertise(mr_id, tensor.data_ptr(), tensor.nbytes)
            if not ok:
                raise RuntimeError("destination advertisement failed")
            blobs.append(bytes(blob))
    advertised = [blobs]
    dist.broadcast_object_list(advertised, src=1)
    blobs = advertised[0]

    latencies = []
    for round_id in range(args.warmup + args.rounds):
        expected = [
            payload(args.bytes, dtype, 1009 + round_id * args.batch + index)
            for index in range(args.batch)
        ]
        for index, tensor in enumerate(tensors):
            if rank == 0:
                tensor.copy_(expected[index])
            else:
                tensor.view(torch.uint8).fill_(255)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            started = time.perf_counter()
            write_batch(ep, conn, ids, tensors, blobs, args.api, args.timeout)
            submitted_ms = 1000 * (time.perf_counter() - started)
            dist.send(torch.tensor([round_id], dtype=torch.int64), dst=1)
            reply = torch.empty(1, dtype=torch.int64)
            dist.recv(reply, src=1)
            elapsed_ms = 1000 * (time.perf_counter() - started)
            valid = reply.item() == 0
            row = {
                "round": round_id,
                "warmup": round_id < args.warmup,
                "api": args.api,
                "dtype": args.dtype,
                "bytes": args.bytes,
                "batch": args.batch,
                "arena_bytes": arena,
                "correct": valid,
                "submitted_ms": submitted_ms,
            }
            if valid:
                row["verified_roundtrip_ms"] = elapsed_ms
                row["verified_payload_GBps"] = (
                    args.bytes * args.batch / (elapsed_ms * 1e6)
                )
                if round_id >= args.warmup:
                    latencies.append(elapsed_ms)
            print(json.dumps(row), flush=True)
        else:
            dist.recv(torch.empty(1, dtype=torch.int64), src=0)
            pending = list(range(args.batch))
            deadline = time.monotonic() + args.timeout
            while pending and time.monotonic() < deadline:
                pending = [
                    index
                    for index in pending
                    if not torch.equal(
                        tensors[index].view(torch.uint8),
                        expected[index].view(torch.uint8),
                    )
                ]
            torch.cuda.synchronize()
            valid = not pending
            if pending:
                print(
                    json.dumps({"round": round_id, "incorrect_buffers": pending}),
                    flush=True,
                )
            dist.send(torch.tensor([len(pending)], dtype=torch.int64), dst=0)
        if not valid:
            raise RuntimeError(f"round {round_id} produced incorrect compressed writes")

    dist.barrier()
    torch.cuda.synchronize()
    for mr_id in ids:
        if not ep.dereg(mr_id):
            raise RuntimeError("memory deregistration failed")
    if rank == 0:
        print(
            json.dumps(
                {
                    "correct_rounds": args.rounds,
                    "median_verified_roundtrip_ms": statistics.median(latencies),
                    "median_verified_payload_GBps": args.bytes
                    * args.batch
                    / (statistics.median(latencies) * 1e6),
                }
            ),
            flush=True,
        )
    del ep
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
