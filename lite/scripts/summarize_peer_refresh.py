#!/usr/bin/env python3
"""Validate the four-L4 direct-peer AG/RS per-call comparison."""

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

SAMPLE = re.compile(r"^# sample,(all_gather|reduce_scatter),(\d+),(peer|nccl),(\d+),(\d+),([\d.]+)$", re.M)


def summarize(source: Path, samples_path: Path, summary_path: Path) -> None:
    text = source.read_text()
    if "NCCL version=22807" not in text or "every warmup and timed output checked" not in text:
        raise ValueError("wrong peer/NCCL comparison identity")
    if "mismatch" in text or "peer AG/RS:" in text:
        raise ValueError("peer comparison failed correctness")
    values = defaultdict(dict)
    samples = []
    for operation, size, backend, repeat, iteration, latency in SAMPLE.findall(text):
        size, repeat, iteration = int(size), int(repeat), int(iteration)
        latency = float(latency)
        if repeat not in (1, 2, 3) or not 1 <= iteration <= 10 or not math.isfinite(latency) or latency <= 0:
            raise ValueError("invalid peer sample")
        key = (operation, size, backend, repeat)
        if iteration in values[key]:
            raise ValueError("duplicate peer sample")
        values[key][iteration] = latency
        samples.append({"operation": operation, "bytes": size,
                        "backend": backend, "repeat": repeat,
                        "iteration": iteration, "latency_us": latency})
    expected_sizes = [1 << power for power in range(10, 31)]
    summary = []
    for operation, name in (("all_gather", "allgather"),
                            ("reduce_scatter", "reducescatter")):
        for size in expected_sizes:
            run_means = {}
            for backend in ("peer", "nccl"):
                per_run = []
                for repeat in (1, 2, 3):
                    observed = values[(operation, size, backend, repeat)]
                    if sorted(observed) != list(range(1, 11)):
                        raise ValueError(f"missing peer calls: {operation}, {size}, {backend}, {repeat}")
                    per_run.append(mean(observed.values()))
                run_means[backend] = per_run
            own, nccl = run_means["peer"], run_means["nccl"]
            summary.append({
                "operation": name, "bytes": size, "uccl_path": "direct-peer",
                "uccl_min_us": min(own), "uccl_median_us": median(own),
                "uccl_max_us": max(own), "nccl_min_us": min(nccl),
                "nccl_median_us": median(nccl), "nccl_max_us": max(nccl),
                "speedup": median(nccl) / median(own),
            })
    if len(samples) != 2 * len(expected_sizes) * 2 * 3 * 10:
        raise ValueError("unexpected peer sample count")
    for destination, rows in ((samples_path, samples), (summary_path, summary)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("samples", type=Path)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    summarize(args.source, args.samples, args.summary)
