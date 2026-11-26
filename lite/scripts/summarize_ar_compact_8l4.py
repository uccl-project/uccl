#!/usr/bin/env python3
"""Summarize paired compact-router AllReduce runs on eight L4 GPUs."""

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks/data/ar_compact_8l4_20261111"
SIZES = (128 << 10, 256 << 10)


def load(baseline, run):
    stem = f"{baseline}_run{run}"
    log = (DATA / f"{stem}.log").read_text()
    if "# correctness: PASS (every invocation, all ranks)" not in log:
        raise ValueError(f"missing output verification: {stem}")
    profile = (DATA / f"{stem}.profile").read_text()
    if "small-ar-128-256k-workers-v1=4,16" not in profile:
        raise ValueError(f"wrong router geometry: {stem}")
    groups = defaultdict(list)
    with (DATA / f"{stem}.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["collective"] != "all_reduce" or row["in_place"] != "1":
                raise ValueError(f"wrong workload: {stem}")
            key = int(row["bytes"]), row["backend"], int(row["repeat"])
            groups[key].append(float(row["slowest_rank_us"]))
    control = "uccl-lite-local" if baseline == "local" else "nccl"
    expected = {(size, backend, repeat) for size in SIZES
                for backend in (control, "cpu-router") for repeat in range(3)}
    if set(groups) != expected:
        raise ValueError(f"incomplete run: {stem}")
    result = {}
    for size in SIZES:
        for backend in (control, "cpu-router"):
            repetitions = []
            for repeat in range(3):
                samples = groups[size, backend, repeat]
                if (len(samples) != 10 or
                        any(not math.isfinite(value) or value <= 0
                            for value in samples)):
                    raise ValueError(f"invalid samples: {stem} {size} {backend}")
                repetitions.append(statistics.median(samples))
            result[size, backend] = statistics.median(repetitions)
    return result


def main():
    runs = {(baseline, run): load(baseline, run)
            for baseline in ("local", "nccl") for run in (1, 2)}
    output = DATA / "summary.csv"
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("bytes", "uccl_run1_us", "uccl_run2_us",
                         "uccl_median_us", "local_median_us",
                         "nccl_median_us", "speedup_vs_nccl"))
        for size in SIZES:
            u1 = runs["nccl", 1][size, "cpu-router"]
            u2 = runs["nccl", 2][size, "cpu-router"]
            local = statistics.median(runs["local", run][size, "uccl-lite-local"]
                                      for run in (1, 2))
            nccl = statistics.median(runs["nccl", run][size, "nccl"]
                                     for run in (1, 2))
            uccl = statistics.median((u1, u2))
            writer.writerow((size, *(f"{v:.3f}" for v in
                                     (u1, u2, uccl, local, nccl)),
                             f"{nccl / uccl:.4f}"))
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
