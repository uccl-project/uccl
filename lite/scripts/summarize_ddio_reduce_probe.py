#!/usr/bin/env python3
"""Summarize the 8xL4 GPU-write/CPU-reduction cache probe."""

import csv
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks/data/ddio_8l4_20261111"
SIZES = tuple(1 << power for power in (14, 16, 18, 19, 21, 23))
MODES = ("cpu_hot", "cpu_flushed", "gpu_sm_write",
         "gpu_sm_write_flushed", "gpu_copy_engine",
         "gpu_copy_engine_flushed")


def load(node, run):
    path = DATA / f"node{node}_run{run}.csv"
    raw = path.read_text()
    if "# correctness: PASS" not in raw or f"# node={node} " not in raw:
        raise ValueError(f"missing verification or node identity: {path}")
    rows = {}
    for row in csv.DictReader(line for line in raw.splitlines()
                              if not line.startswith("#")):
        key = int(row["bytes_per_source"]), row["condition"]
        value = float(row["median_us"])
        if key in rows or not math.isfinite(value) or value <= 0:
            raise ValueError(f"bad probe row: {path}, {key}")
        rows[key] = value
    if set(rows) != {(size, mode) for size in SIZES for mode in MODES}:
        raise ValueError(f"incomplete probe: {path}")
    return rows


def main():
    runs = {(node, run): load(node, run)
            for node in (0, 1) for run in (1, 2, 3)}
    output = DATA / "summary.csv"
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("node", "bytes_per_source", "cpu_hot_us",
                         "cpu_flushed_us", "gpu_sm_write_us",
                         "gpu_sm_write_flushed_us", "gpu_copy_engine_us",
                         "gpu_copy_engine_flushed_us",
                         "sm_flush_over_sm", "ce_flush_over_ce"))
        for node in (0, 1):
            for size in SIZES:
                values = {mode: statistics.median(
                    runs[node, run][size, mode] for run in (1, 2, 3))
                    for mode in MODES}
                writer.writerow((node, size,
                                 *(f"{values[mode]:.3f}" for mode in MODES),
                                 f'{values["gpu_sm_write_flushed"] / values["gpu_sm_write"]:.3f}',
                                 f'{values["gpu_copy_engine_flushed"] / values["gpu_copy_engine"]:.3f}'))
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
