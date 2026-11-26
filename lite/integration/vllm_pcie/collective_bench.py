#!/usr/bin/env python3
"""Measure the pinned eight-rank c03 extension and an out-of-place NCCL control."""

import argparse
from datetime import timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 8
C03_SOURCE = "ce751fa67339d5163171721a350c09d8edc14876a2e89cb808349ba984f75175"
C03_BINARY = "89ebe0c1de86d93f6d92e709fcc2dbf753b46579e5fd588c8e35ae5bd1dc243b"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(rank, port, args):
    torch.set_num_threads(1)
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD,
                            timeout=timedelta(minutes=10))
    sys.path.insert(0, str(Path(args.reference).resolve()))
    import _uccl_lite_nccl_reference as nccl
    if args.backend == "c03":
        spec = importlib.util.spec_from_file_location("_pcieccl_C", args.extension)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        unique_id = [module.create_unique_id() if rank == 0 else None]
    else:
        unique_id = [nccl.create_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(unique_id, src=0)
    if args.backend == "c03":
        comm = module.Communicator(rank, WORLD, unique_id[0], rank, 1 << args.max_power)
    else:
        comm = nccl.Communicator(rank, WORLD, unique_id[0])
    stream = torch.cuda.Stream()
    powers = list(range(args.min_power, args.max_power + 1))
    if args.reverse:
        powers.reverse()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    dtype_code = 0 if args.dtype == "bf16" else 1
    guard = 128 // dtype.itemsize
    for power in powers:
        size = 1 << power
        count = size // dtype.itemsize
        source_storage = torch.full((count + 2 * guard,), 17, dtype=dtype, device=rank)
        output_storage = torch.full_like(source_storage, -19)
        source = source_storage[guard:guard + count]
        output = output_storage[guard:guard + count]
        # Dyadic values keep this corpus exactly representable in BF16/FP32
        # through all six phases. Position, rank and generation all vary.
        base = (torch.arange(count, device=rank, dtype=torch.int32) % 17 - 8).to(dtype)
        expected_input = torch.empty_like(source)
        expected_output = torch.empty_like(source)
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        source.copy_(base).add_(2 * (rank - 3)).div_(8)
        torch.cuda.synchronize()
        dist.barrier()
        with torch.cuda.stream(stream):
            for _ in range(args.warmups):
                comm.all_reduce(source.data_ptr(), output.data_ptr(), count,
                                dtype_code, stream.cuda_stream)
        stream.synchronize()
        rows = []
        for repeat in range(args.repeats):
            for iteration in range(args.iterations):
                generation = (repeat * args.iterations + iteration + args.seed) % 3
                expected_input.copy_(base).add_(2 * (rank - 3) + generation).div_(8)
                expected_output.copy_(base).add_(1 + generation)
                source.copy_(expected_input)
                output.fill_(-19)
                torch.cuda.synchronize()
                dist.barrier()
                with torch.cuda.stream(stream):
                    start.record()
                    comm.all_reduce(source.data_ptr(), output.data_ptr(), count,
                                    dtype_code, stream.cuda_stream)
                    stop.record()
                stop.synchronize()
                us = start.elapsed_time(stop) * 1000
                checked = torch.equal(output, expected_output) and torch.equal(source, expected_input)
                for tensor, sentinel in ((source_storage, 17), (output_storage, -19)):
                    checked = checked and bool((tensor[:guard] == sentinel).all())
                    checked = checked and bool((tensor[-guard:] == sentinel).all())
                if not checked:
                    raise RuntimeError(f"full-output/input/guard check failed rank={rank} bytes={size}")
                rows.append({"rank": rank, "bytes": size, "repeat": repeat,
                             "iteration": iteration, "us": us, "checked": True})
        gathered = [None] * WORLD if rank == 0 else None
        dist.gather_object(rows, gathered, dst=0)
        if rank == 0:
            with Path(args.output).open("a") as out:
                for rank_rows in gathered:
                    for row in rank_rows:
                        out.write(json.dumps(row, sort_keys=True) + "\n")
            print(f"{args.backend} {args.label} bytes={size} checked={len(rows) * WORLD}", flush=True)
        del source, output, source_storage, output_storage, expected_input, expected_output, base
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    dist.barrier()
    comm.close()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("c03", "nccl"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--extension", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--min-power", type=int, default=10)
    parser.add_argument("--max-power", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    if torch.cuda.device_count() != WORLD:
        raise RuntimeError("Expose exactly eight GPUs in physical order")
    if sha(args.extension) != C03_BINARY or sha(args.source) != C03_SOURCE:
        raise RuntimeError("Expected the frozen EAT c03 source and extension")
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    metadata = {"arguments": vars(args), "source_sha256": sha(args.source),
                "extension_sha256": sha(args.extension), "harness_sha256": sha(__file__),
                "reference_sha256": sha(Path(args.reference) / "_uccl_lite_nccl_reference.so"),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "nccl": torch.cuda.nccl.version(),
                "environment": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output + ".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(port, args), nprocs=WORLD, join=True)
    metadata["status"] = "complete"
    Path(args.output + ".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
