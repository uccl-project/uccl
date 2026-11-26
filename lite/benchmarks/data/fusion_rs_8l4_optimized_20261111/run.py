#!/usr/bin/env python3
"""Run the checked eight-L4 GEMM+RS sweep after prefill scheduling fixes."""

import argparse
import csv
import os
import subprocess
from collections import Counter
from pathlib import Path


DATA = Path(__file__).resolve().parent
LITE = DATA.parents[2]
BENCH = LITE / "build/uccl_lite_router_fusion_8l4"
MODELS = (
    ("qwen3.5-9b", 4096, 1536),
    ("qwen3.5-27b", 5120, 2176),
    ("gemma-4-31b-it", 5376, 2688),
    ("qwen3-32b", 5120, 3200),
)
PREFILL_M = (2048, 4096, 8192)
DECODE_M = {
    "gemm-rs": (1, 32, 64),
}
METHODS = ("cublas+nccl", "cublas+router", "fused+router")
SAMPLES_PER_METHOD = 50


def completed_shapes(path, op):
    if not path.exists():
        return set()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    if any(row["op"] != op for row in rows):
        raise RuntimeError(f"unexpected operation in {path}")
    counts = Counter(
        (int(row["m"]), int(row["n"]), int(row["k"]), row["method"])
        for row in rows
    )
    for key, count in counts.items():
        if count > SAMPLES_PER_METHOD:
            raise RuntimeError(f"duplicate sample set {key}: {count}")
    done = {
        (m, n, k)
        for m, n, k, _ in counts
        if all(counts[(m, n, k, method)] == SAMPLES_PER_METHOD
               for method in METHODS)
    }
    partial = {key[:3] for key in counts} - done
    if partial:
        print(f"discarding partial samples for {sorted(partial)}", flush=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields,
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(row for row in rows
                             if (int(row["m"]), int(row["n"]),
                                 int(row["k"])) not in partial)
        tmp.replace(path)
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=tuple(DECODE_M), required=True)
    parser.add_argument("--run", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    name = args.op.replace("-", "_")
    samples = DATA / f"samples_{name}_run{args.run}.csv"
    log = DATA / f"{name}_run{args.run}.log"
    done = completed_shapes(samples, args.op)
    shapes = [(model, m, n, k) for model, n, k in MODELS
              for m in (*DECODE_M[args.op], *PREFILL_M)]
    if args.run == 2:
        shapes.reverse()
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        f"{LITE / 'build/nccl-2.28.9/lib'}:/usr/local/cuda-12.9/lib64:"
        + env.get("LD_LIBRARY_PATH", "")
    )
    env["NCCL_NET"] = "Socket"
    with log.open("a", buffering=1) as output:
        for index, (model, m, n, k) in enumerate(shapes, 1):
            if (m, n, k) in done:
                print(f"[{index}/{len(shapes)}] skip {model} {m}x{n}x{k}",
                      flush=True)
                continue
            cmd = [str(BENCH), "--op", args.op, "--m", str(m),
                   "--n", str(n), "--k", str(k), "--warmup", "2",
                   "--iterations", "10", "--repeats", "5",
                   "--samples", str(samples)]
            print(f"[{index}/{len(shapes)}] {model} {m}x{n}x{k}", flush=True)
            output.write("$ " + " ".join(cmd) + "\n")
            output.flush()
            result = subprocess.run(cmd, cwd=LITE, env=env,
                                    stdout=output, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"failed {model} {m}x{n}x{k}: "
                                   f"exit {result.returncode}; see {log}")
            done = completed_shapes(samples, args.op)
            if (m, n, k) not in done:
                raise RuntimeError(f"missing samples for {model} {m}x{n}x{k}")
    print(f"{args.op} run {args.run} complete: {len(done)} shapes", flush=True)


if __name__ == "__main__":
    main()
