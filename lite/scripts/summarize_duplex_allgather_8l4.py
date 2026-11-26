#!/usr/bin/env python3
"""Summarize paired eight-L4 AllGather samples by run-level medians."""

import csv
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks/data/duplex_allgather_8l4_20261111"
METHODS = ("nccl", "packet-router", "duplex-stage")
SIZES = tuple(1 << power for power in range(20, 31))


def load(run):
    groups = defaultdict(list)
    with (DATA / f"samples_run{run}.csv").open() as stream:
        for row in csv.DictReader(stream):
            key = int(row["bytes"]), row["method"], int(row["repeat"])
            groups[key].append(float(row["slowest_rank_us"]))
    expected = {(size, method, repeat) for size in SIZES
                for method in METHODS for repeat in range(3)}
    if set(groups) != expected or any(len(values) != 10 for values in groups.values()):
        raise ValueError(f"incomplete run {run}")
    return {(size, method): statistics.median(
        statistics.median(groups[size, method, repeat]) for repeat in range(3))
        for size in SIZES for method in METHODS}


def main():
    runs = {run: load(run) for run in (1, 2)}
    output = DATA / "summary.csv"
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("bytes", "nccl_run1_us", "packet_run1_us",
                         "duplex_run1_us", "nccl_run2_us", "packet_run2_us",
                         "duplex_run2_us", "nccl_median_us",
                         "packet_median_us", "duplex_median_us",
                         "duplex_vs_packet"))
        for size in SIZES:
            first = [runs[1][size, method] for method in METHODS]
            second = [runs[2][size, method] for method in METHODS]
            medians = [statistics.median((a, b))
                       for a, b in zip(first, second)]
            writer.writerow((size, *(f"{v:.3f}" for v in first),
                             *(f"{v:.3f}" for v in second),
                             *(f"{v:.3f}" for v in medians),
                             f"{medians[1] / medians[2]:.4f}"))
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
