#!/usr/bin/env python3
"""Summarize paired eight-L4 SHM collective samples."""

import csv
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks/data/shm_8l4_20261111"
OPERATIONS = (("ar", "all_reduce", "in_place"),
              ("ag", "all_gather", "out_of_place"),
              ("rs", "reduce_scatter", "out_of_place"))
SIZES = tuple(1 << power for power in range(10, 31))


def load(prefix, collective, run):
    groups = defaultdict(list)
    with (DATA / f"{prefix}_run{run}.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["collective"] != collective:
                raise ValueError(f"wrong collective in {prefix} run {run}")
            key = int(row["bytes"]), row["backend"], int(row["repeat"])
            groups[key].append(float(row["slowest_rank_us"]))
    result = {}
    for size in SIZES:
        paths = {backend for message, backend, _ in groups
                 if message == size and backend != "nccl"}
        if len(paths) != 1:
            raise ValueError(f"missing or duplicate project path: {prefix} {size}")
        path = paths.pop()
        for backend in ("nccl", path):
            per_repeat = []
            for repeat in range(3):
                samples = groups[size, backend, repeat]
                if len(samples) != 10 or min(samples) <= 0:
                    raise ValueError(f"incomplete samples: {prefix} {run} {size}")
                per_repeat.append(statistics.median(samples))
            result[size, backend] = statistics.median(per_repeat)
        result[size, "path"] = path
    if len(groups) != len(SIZES) * 2 * 3:
        raise ValueError(f"unexpected samples: {prefix} run {run}")
    return result


def main():
    path = DATA / "summary.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("collective", "bytes", "placement", "uccl_path",
                         "uccl_run1_us", "nccl_run1_us", "uccl_run2_us",
                         "nccl_run2_us", "uccl_median_us", "nccl_median_us",
                         "speedup"))
        for prefix, collective, placement in OPERATIONS:
            first = load(prefix, collective, 1)
            second = load(prefix, collective, 2)
            for size in SIZES:
                backend = first[size, "path"]
                if backend != second[size, "path"]:
                    raise ValueError(f"project path changed: {collective} {size}")
                uccl = [first[size, backend], second[size, backend]]
                nccl = [first[size, "nccl"], second[size, "nccl"]]
                u = statistics.median(uccl)
                n = statistics.median(nccl)
                writer.writerow((collective, size, placement, backend,
                                 *(f"{v:.3f}" for v in
                                   (uccl[0], nccl[0], uccl[1], nccl[1], u, n)),
                                 f"{n / u:.4f}"))
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
