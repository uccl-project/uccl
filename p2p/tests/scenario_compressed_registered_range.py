"""Exercise full-buffer and subrange writes through a typed registration.

Run two RDMA-connected GPU processes with a USE_DIETGPU=1 build:
  UCCL_P2P_TRANSPORT=rdma UCCL_P2P_COMPRESS_STRATEGY=split_only \
  torchrun --standalone --nproc-per-node=2 \
      tests/scenario_compressed_registered_range.py --api=write_async

Repeat with write, writev and writev_async, then with the encode strategy.
Set UCCL_P2P_LOG_LEVEL=INFO and UCCL_DEBUG_SUBSYS=RDMA to confirm that
split_only compresses full-buffer writes and sends subranges uncompressed.
The encode strategy retains its existing uncompressed one-sided write path.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from uccl import p2p

from scenario_compressed_write_arena import payload


def submit(ep, conn, mr_id, source, blob, api, timeout):
    if api in ("writev", "writev_async"):
        args = (
            conn,
            [mr_id],
            [source.data_ptr()],
            [source.nbytes],
            [blob],
            1,
        )
    else:
        args = (conn, mr_id, source.data_ptr(), source.nbytes, blob)
    result = getattr(ep, api)(*args)
    if not api.endswith("_async"):
        if not result:
            raise RuntimeError(f"{api} failed")
        return
    ok, handle = result
    if not ok:
        raise RuntimeError(f"{api} submission failed")
    deadline = time.monotonic() + timeout
    while True:
        ok, done = ep.poll_async(handle)
        if not ok:
            raise RuntimeError(f"{api} completion failed")
        if done:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{api} completion timed out")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument(
        "--api",
        choices=("write", "write_async", "writev", "writev_async"),
        default="write",
    )
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    itemsize = torch.empty((), dtype=dtype).element_size()
    minimum = int(os.environ.get("UCCL_P2P_MIN_COMPRESS_BYTES", 16 * 1024 * 1024))
    arena = int(os.environ.get("UCCL_P2P_COMPRESS_BUFFER_BYTES", 2 * 1024**3))
    if args.bytes < minimum or args.bytes % itemsize or args.bytes * 4 > arena:
        parser.error(
            "--bytes must reach the threshold and four blocks must fit the arena"
        )
    if args.rounds < 1 or args.timeout <= 0 or minimum <= itemsize:
        parser.error(
            "rounds >= 1, timeout > 0 and compression threshold > itemsize required"
        )
    if os.environ.get("UCCL_P2P_TRANSPORT", "rdma") not in ("rdma", "ib"):
        parser.error("this scenario requires the RDMA transport")
    strategy = os.environ.get("UCCL_P2P_COMPRESS_STRATEGY", "")
    if strategy not in ("split", "split_only", "encode", "split_encode", "full", "1"):
        parser.error("set UCCL_P2P_COMPRESS_STRATEGY=split_only or encode")
    linkage = subprocess.run(
        ["ldd", p2p.__file__], capture_output=True, text=True, check=True
    )
    if "libdietgpu_float.so =>" not in linkage.stdout or "not found" in linkage.stdout:
        parser.error("the loaded extension requires a native USE_DIETGPU=1 build")

    dist.init_process_group("gloo", timeout=timedelta(seconds=args.timeout + 120))
    if dist.get_world_size() != 2:
        raise RuntimeError("this scenario requires exactly two processes")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    ep = p2p.Endpoint(local_gpu_idx=torch.cuda.current_device())
    count = args.bytes // itemsize
    guard = 4096 // itemsize
    tensor = torch.empty(
        count * 4 + (2 * guard if rank else 0), dtype=dtype, device="cuda"
    )
    float_type = getattr(
        p2p.FloatType,
        {"bfloat16": "kBFloat16", "float16": "kFloat16", "float32": "kFloat32"}[
            args.dtype
        ],
    )
    ok, mr_id = ep.reg(
        tensor.data_ptr(),
        tensor.nbytes,
        float_type if rank == 0 else p2p.FloatType.kUndefined,
    )
    if not ok:
        raise RuntimeError("memory registration failed")
    metadata = [None, None]
    dist.all_gather_object(metadata, bytes(ep.get_metadata()))
    if rank == 0:
        ip, port, gpu = p2p.Endpoint.parse_metadata(metadata[1])
        ok, conn = ep.connect(ip, gpu, remote_port=port)
    else:
        ok, _, _, conn = ep.accept()
    if not ok:
        raise RuntimeError("RDMA connection failed")

    ranges = (
        ("full", 0, 4 * count),
        ("prefix", 0, count),
        ("middle", count, count),
        ("suffix", 3 * count, count),
        ("below_threshold", 0, max(1, minimum // (2 * itemsize))),
    )
    for round_id in range(args.rounds):
        expected_source = payload(4 * args.bytes, dtype, 1201 + round_id)
        if rank == 0:
            tensor.copy_(expected_source)
        for name, start, length in ranges:
            advertised = [None]
            if rank == 1:
                tensor.view(torch.uint8).fill_(255)
                expected = tensor.clone()
                expected[guard : guard + length].copy_(
                    expected_source[start : start + length]
                )
                ok, blob = ep.advertise(
                    mr_id, tensor[guard:].data_ptr(), length * itemsize
                )
                if not ok:
                    raise RuntimeError("destination advertisement failed")
                advertised[0] = bytes(blob)
            dist.broadcast_object_list(advertised, src=1)
            torch.cuda.synchronize()
            dist.barrier()
            if rank == 0:
                submit(
                    ep,
                    conn,
                    mr_id,
                    tensor[start : start + length],
                    advertised[0],
                    args.api,
                    args.timeout,
                )
                dist.send(torch.tensor([1], dtype=torch.int64), dst=1)
                reply = torch.empty(1, dtype=torch.int64)
                dist.recv(reply, src=1)
                valid = reply.item() == 0
            else:
                dist.recv(torch.empty(1, dtype=torch.int64), src=0)
                deadline = time.monotonic() + args.timeout
                valid = False
                while time.monotonic() < deadline:
                    if torch.equal(
                        tensor.view(torch.uint8), expected.view(torch.uint8)
                    ):
                        valid = True
                        break
                dist.send(torch.tensor([0 if valid else 1], dtype=torch.int64), dst=0)
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "round": round_id,
                        "range": name,
                        "api": args.api,
                        "strategy": strategy,
                        "bytes": length * itemsize,
                        "payload_and_guards_correct": valid,
                    }
                ),
                flush=True,
            )
            if not valid:
                raise RuntimeError(f"{name} write changed the payload or its guards")

    dist.barrier()
    torch.cuda.synchronize()
    if not ep.dereg(mr_id):
        raise RuntimeError("memory deregistration failed")
    del ep
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
