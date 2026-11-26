#!/usr/bin/env python3
"""Validate the pipelined host-RDMA AllGather paired sweep."""

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


SAMPLE = re.compile(
    r"^# sample,allgather,(\d+),(host-rdma-multicast|nccl),"
    r"(\d+),(\d+),([\d.]+),(\d+)$", re.M)
RESULT = re.compile(
    r"^allgather,(\d+),(host-rdma-multicast|nccl),"
    r"([\d.]+),([\d.]+),([\d.]+)$", re.M)


def write_rows(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    content = args.log.read_text()
    if ("# timing=host enqueue+completion" not in content or
            "warmups=20 iterations=10 repeats=3" not in content or
            "# paired NCCL version=22807" not in content or
            "download=sm" not in content or "slots=3" not in content or
            "mismatch" in content):
        raise ValueError("unexpected benchmark configuration or output failure")
    samples = defaultdict(dict)
    for size, backend, repeat, call, latency, wrong in SAMPLE.findall(content):
        size, repeat, call = int(size), int(repeat), int(call)
        value = float(latency)
        key = (size, backend, repeat)
        if (not 1 <= repeat <= 3 or not 1 <= call <= 10 or
                not math.isfinite(value) or value <= 0 or int(wrong) != 0 or
                call in samples[key]):
            raise ValueError("invalid or duplicate timed call")
        samples[key][call] = value
    reported = {}
    for size, backend, med, low, high in RESULT.findall(content):
        key = (int(size), backend)
        if key in reported:
            raise ValueError("duplicate reported result")
        reported[key] = (float(med), float(low), float(high))
    sizes = [1 << power for power in range(10, 31)]
    summary = []
    for size in sizes:
        runs = {}
        for backend in ("host-rdma-multicast", "nccl"):
            values = []
            for repeat in (1, 2, 3):
                calls = samples[size, backend, repeat]
                if sorted(calls) != list(range(1, 11)):
                    raise ValueError(f"missing timed calls: {size} {backend} {repeat}")
                values.append(mean(calls.values()))
            actual = reported[size, backend]
            if (abs(median(values) - actual[0]) > 0.002 or
                    abs(min(values) - actual[1]) > 0.002 or
                    abs(max(values) - actual[2]) > 0.002):
                raise ValueError(f"mismatched run summary: {size} {backend}")
            runs[backend] = values
        own, nccl = runs["host-rdma-multicast"], runs["nccl"]
        summary.append({"operation": "allgather", "bytes": size,
                        "router_min_us": min(own),
                        "router_median_us": median(own),
                        "router_max_us": max(own),
                        "nccl_min_us": min(nccl),
                        "nccl_median_us": median(nccl),
                        "nccl_max_us": max(nccl),
                        "nccl_over_router": median(nccl) / median(own)})
    records = []
    for (size, backend, repeat), calls in sorted(samples.items()):
        for call, latency in sorted(calls.items()):
            records.append({"bytes": size, "backend": backend,
                            "repeat": repeat, "call": call,
                            "host_latency_us": latency, "wrong": 0})
    if len(records) != 21 * 2 * 3 * 10:
        raise ValueError("incorrect total timed call count")
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output / "samples.csv", list(records[0]), records)
    write_rows(args.output / "summary.csv", list(summary[0]), summary)
    print(f"validated {len(summary)} sizes and {len(records)} timed calls")


if __name__ == "__main__":
    main()
