#!/usr/bin/env python3
"""Validate paired eight-rank AG/RS logs and write their complete summary."""

import argparse
import csv
import math
import re
from pathlib import Path
from statistics import median


def uccl_rows(path: Path, operation: str, timing_mode: str) -> dict[int, float]:
    text = path.read_text()
    expected_timing = {"cuda-event": "CUDA event", "host": "host enqueue+completion"}[timing_mode]
    if f"# timing={expected_timing}; slowest of eight ranks" not in text:
        raise ValueError(f"wrong timing scope: {path}")
    expected = {
        "allgather": "# local ring, remote-rank IB exchange",
        "reducescatter": "# local direct ring reduction, remote-rank IB exchange",
        "allreduce": "# local ring 0->1->3->2->0; AG ring",
    }[operation]
    if expected not in text:
        raise ValueError(f"wrong or prepacked UCCL variant: {path}")
    ranks = {int(m) for m in re.findall(r"^# rank (\d+) node ", text, re.M)}
    if ranks != set(range(8)):
        raise ValueError(f"expected eight mapped ranks: {path}")
    rows = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 7 or not fields[0].isdigit():
            continue
        size = int(fields[0])
        latency, error = float(fields[3]), float(fields[6])
        if size in rows or not math.isfinite(latency) or latency <= 0:
            raise ValueError(f"invalid or duplicate UCCL timing: {path}")
        if not math.isfinite(error) or error > 1e-5:
            raise ValueError(f"UCCL correctness failure: {path}, {size}")
        rows[size] = latency
    if not rows:
        raise ValueError(f"no UCCL timings: {path}")
    return rows


def nccl_rows(path: Path, operation: str) -> dict[int, float]:
    text = path.read_text()
    if "nccl-library=22807" not in text or "# Out of bounds values : 0 OK" not in text:
        raise ValueError(f"wrong NCCL version or unchecked output: {path}")
    ranks = {int(m) for m in re.findall(r"^#  Rank\s+(\d+) Group ", text, re.M)}
    if ranks != set(range(8)):
        raise ValueError(f"nccl-tests lacks MPI rank mapping: {path}")
    rows = {}
    redop = "none" if operation == "allgather" else "sum"
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 13 or not fields[0].isdigit() or fields[2] != "float":
            continue
        size = int(fields[0])
        latency = float(fields[9] if operation == "allreduce" else fields[5])
        if fields[3] != redop or size in rows or not math.isfinite(latency) or latency <= 0:
            raise ValueError(f"invalid or duplicate NCCL timing: {path}")
        if fields[8] != "0" or fields[12] != "0":
            raise ValueError(f"NCCL correctness failure: {path}, {size}")
        rows[size] = latency
    if not rows:
        raise ValueError(f"no NCCL timings: {path}")
    return rows


def summarize(directory: Path, output: Path, repeats: int,
              operations: tuple[str, ...], timing_mode: str) -> None:
    fields = [
        "collective", "bytes", "uccl_min_us", "uccl_median_us", "uccl_max_us",
        "nccl_min_us", "nccl_median_us", "nccl_max_us", "speedup",
        "paired_min_speedup", "paired_median_speedup", "paired_max_speedup",
        "stable_3pct_win",
    ]
    summary = []
    for operation in operations:
        uccl = [uccl_rows(directory / f"{operation}_uccl_run{r}.txt", operation, timing_mode)
                for r in range(1, repeats + 1)]
        nccl = [nccl_rows(directory / f"{operation}_nccl_run{r}.txt", operation)
                for r in range(1, repeats + 1)]
        sizes = set(uccl[0])
        if any(set(run) != sizes for run in uccl + nccl):
            raise ValueError(f"size sweep mismatch: {operation}")
        for size in sorted(sizes):
            a = [run[size] for run in uccl]
            n = [run[size] for run in nccl]
            ratios = [baseline / candidate for baseline, candidate in zip(n, a)]
            speedup = median(n) / median(a)
            summary.append({
                "collective": operation,
                "bytes": size,
                "uccl_min_us": min(a),
                "uccl_median_us": median(a),
                "uccl_max_us": max(a),
                "nccl_min_us": min(n),
                "nccl_median_us": median(n),
                "nccl_max_us": max(n),
                "speedup": speedup,
                "paired_min_speedup": min(ratios),
                "paired_median_speedup": median(ratios),
                "paired_max_speedup": max(ratios),
                "stable_3pct_win": int(speedup >= 1.03 and max(a) < min(n)),
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--operations", default="allgather,reducescatter")
    parser.add_argument("--timing-mode", choices=("cuda-event", "host"),
                        default="cuda-event")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    operations = tuple(args.operations.split(","))
    if not operations or any(op not in ("allgather", "reducescatter", "allreduce")
                             for op in operations):
        parser.error("--operations must list allgather, reducescatter, and/or allreduce")
    summarize(args.directory, args.output, args.repeats, operations,
              args.timing_mode)
