#!/usr/bin/env python3
"""Checked two-host comparison of AllGather download layouts."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--binary", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--hosts", default="10.10.55.2:2,10.10.55.1:2")
parser.add_argument("--rounds", type=int, default=3)
parser.add_argument("--warmups", type=int, default=50)
parser.add_argument("--iterations", type=int, default=100)
parser.add_argument("--begin", type=int, default=1024)
parser.add_argument("--end", type=int, default=262144)
parser.add_argument("--delay-consumer", type=int, default=-1)
parser.add_argument("--chunk-bytes", type=int, default=262144)
parser.add_argument("--size-order", choices=["ascending", "alternating"], default="ascending")
parser.add_argument("--modes", nargs="+", default=["packet_sm", "packet_dma", "contiguous_dma", "contiguous_sm"],
                    choices=["packet_sm", "packet_dma", "contiguous_dma", "contiguous_sm",
                             "packed_dma", "packed_sm", "packed_dma_early", "packed_sm_early",
                             "packed_dma_self", "packed_dma_shared", "packed_dma_fixed",
                             "packed_sm_fixed", "packed_dma_fixed_unsplit", "packed_sm_self", "combined"])
parser.add_argument("--timeout", type=int, default=180)
args = parser.parse_args()
binary = args.binary.resolve()
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=False)
configs = {
    "packet_sm": ["--download", "sm", "--local-input", "self",
                  "--local-exchange", "shared", "--early-consume", "1"],
    "packet_dma": ["--download", "dma", "--local-input", "self",
                   "--local-exchange", "shared", "--early-consume", "1"],
    "contiguous_dma": ["--download", "contiguous-dma"],
    "contiguous_sm": ["--download", "contiguous-sm"],
    "packed_dma": ["--download", "packed-dma"],
    "packed_sm": ["--download", "packed-sm"],
    "packed_dma_early": ["--download", "packed-dma", "--early-consume", "1"],
    "packed_sm_early": ["--download", "packed-sm", "--early-consume", "1"],
    "packed_dma_self": ["--download", "packed-dma", "--early-consume", "1",
                        "--local-input", "self"],
    "packed_dma_shared": ["--download", "packed-dma", "--early-consume", "1",
                          "--local-exchange", "shared"],
    "packed_dma_fixed": ["--download", "packed-dma", "--early-consume", "1",
                         "--local-input", "self", "--local-exchange", "shared"],
    "packed_sm_fixed": ["--download", "packed-sm", "--early-consume", "1",
                        "--local-input", "self", "--local-exchange", "shared"],
    "packed_dma_fixed_unsplit": ["--download", "packed-dma",
                                 "--local-input", "self", "--local-exchange", "shared"],
    "packed_sm_self": ["--download", "packed-sm", "--local-input", "self"],
    "combined": ["--download", "combined"],
}
configs = {mode: configs[mode] for mode in args.modes}
env = {k: v for k, v in os.environ.items()
       if not k.startswith(("NCCL_", "UCX_", "MSCCLPP_", "CUDA_VISIBLE_DEVICES"))}
env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
sizes = []
n = args.begin
while n <= args.end:
    sizes.append(n)
    n *= 2
records = []
manifest = {"arguments": vars(args) | {"binary": str(binary), "out": str(out)},
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "runs": []}
for repeat in range(1, args.rounds + 1):
    order = list(configs)
    if repeat % 2 == 0:
        order.reverse()
    for mode in order:
        cmd = ["/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun", "--host", args.hosts,
               "-np", "4", "--map-by", "ppr:2:node", "--bind-to", "none",
               "--mca", "pml", "ucx", "--mca", "coll", "^hcoll",
               "-x", "CUDA_VISIBLE_DEVICES", str(binary),
               "-b", str(args.begin), "-e", str(args.end),
               "-w", str(args.warmups), "-n", str(args.iterations), "-r", "1",
               "-k", str(args.chunk_bytes), "--upload", "dma", "--gpu-workers", "0",
               "--progress", "ready", "--nccl", "0", "--delay-consumer-rank",
               str(args.delay_consumer), "--size-order", args.size_order, *configs[mode]]
        name = f"{repeat}_{mode}"
        print(f"{name} starting", flush=True)
        entry = {"name": name, "command": cmd,
                 "started": datetime.now(timezone.utc).isoformat()}
        with (out / f"{name}.log").open("w") as log:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=args.timeout)
        entry.update(returncode=result.returncode,
                     ended=datetime.now(timezone.utc).isoformat())
        manifest["runs"].append(entry)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"{name} failed: see its log")
        values = {n: [] for n in sizes}
        for line in (out / f"{name}.log").read_text().splitlines():
            fields = line.split(",")
            if len(fields) == 8 and fields[:2] == ["# sample", "allgather"]:
                n, value, error = int(fields[2]), float(fields[6]), int(fields[7])
                assert error == 0 and math.isfinite(value) and value > 0
                values[n].append(value)
        for n, v in values.items():
            assert len(v) == args.iterations, (name, n, len(v))
            records.append(dict(mode=mode, round=repeat, bytes=n,
                                mean_us=statistics.mean(v)))
        print(f"{name} PASS", flush=True)
with (out / "run_means.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=["mode", "round", "bytes", "mean_us"])
    writer.writeheader()
    writer.writerows(records)
with (out / "summary.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=["mode", "bytes", "median_us", "min_us", "max_us"])
    writer.writeheader()
    for mode in configs:
        for n in sizes:
            v = [r["mean_us"] for r in records if r["mode"] == mode and r["bytes"] == n]
            writer.writerow(dict(mode=mode, bytes=n, median_us=statistics.median(v),
                                 min_us=min(v), max_us=max(v)))
