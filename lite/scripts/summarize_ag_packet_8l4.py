#!/usr/bin/env python3
"""Summarize paired 8xL4 AllGather runs with size-specific router packets."""

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks/data/ag_packet_8l4_20261111"
SIZES = tuple(1 << power for power in range(10, 31))


def load(run):
    log = (DATA / f"run{run}.log").read_text()
    if "# correctness: PASS (every invocation, all ranks)" not in log:
        raise ValueError(f"missing output verification in run {run}")
    if "packet=524288" not in log or "packet=2097152" not in log:
        raise ValueError(f"unexpected packet geometry in run {run}")
    groups = defaultdict(list)
    with (DATA / f"run{run}.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["collective"] != "all_gather" or row["in_place"] != "0":
                raise ValueError(f"wrong AllGather workload in run {run}")
            key = int(row["bytes"]), row["backend"], int(row["repeat"])
            groups[key].append(float(row["slowest_rank_us"]))
    if len(groups) != len(SIZES) * 2 * 3:
        raise ValueError(f"incomplete run {run}")
    medians = {}
    for size in SIZES:
        for backend in ("nccl", "cpu-router"):
            repetitions = []
            for repeat in range(3):
                samples = groups[size, backend, repeat]
                if (len(samples) != 10 or
                        any(not math.isfinite(value) or value <= 0
                            for value in samples)):
                    raise ValueError(f"incomplete samples: {run} {size} {backend}")
                repetitions.append(statistics.median(samples))
            medians[size, backend] = statistics.median(repetitions)
    return medians


def main():
    first, second = load(1), load(2)
    output = DATA / "summary.csv"
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("bytes", "packet_bytes", "uccl_run1_us",
                         "nccl_run1_us", "uccl_run2_us", "nccl_run2_us",
                         "uccl_median_us", "nccl_median_us", "speedup"))
        for size in SIZES:
            u1, n1 = first[size, "cpu-router"], first[size, "nccl"]
            u2, n2 = second[size, "cpu-router"], second[size, "nccl"]
            u, n = statistics.median((u1, u2)), statistics.median((n1, n2))
            packet = (2 << 20) if size >= (4 << 20) else (512 << 10)
            writer.writerow((size, packet, *(f"{v:.3f}" for v in
                                                    (u1, n1, u2, n2, u, n)),
                             f"{n / u:.4f}"))
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
