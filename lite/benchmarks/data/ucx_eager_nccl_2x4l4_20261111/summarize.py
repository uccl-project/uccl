#!/usr/bin/env python3
"""Validate and summarize the matched-binary eager/auto versus NCCL logs."""

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parent
SAMPLE = re.compile(
    r"^# sample,allgather,(\d+),(host-rdma-multicast|nccl),"
    r"(\d+),(\d+),([\d.]+),(\d+)$", re.M)
RESULT = re.compile(
    r"^allgather,(\d+),(host-rdma-multicast|nccl),"
    r"([\d.]+),([\d.]+),([\d.]+)$", re.M)


def write_csv(name, rows):
    with (ROOT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_log(name, sizes):
    content = (ROOT / f"{name}.log").read_text()
    for required in ("warmups=20 iterations=10 repeats=3",
                     "# paired NCCL version=22807", "download=sm", "slots=3"):
        if required not in content:
            raise ValueError(f"{name}: missing {required}")
    samples = defaultdict(dict)
    for size, backend, repeat, call, latency, wrong in SAMPLE.findall(content):
        size, repeat, call = int(size), int(repeat), int(call)
        latency = float(latency)
        key = (size, backend, repeat)
        if size not in sizes or not 1 <= repeat <= 3 or not 1 <= call <= 10:
            raise ValueError(f"{name}: unexpected sample {key} {call}")
        if call in samples[key] or not math.isfinite(latency) or latency <= 0 or int(wrong):
            raise ValueError(f"{name}: invalid sample {key} {call}")
        samples[key][call] = latency
    reported = {}
    for size, backend, mid, low, high in RESULT.findall(content):
        key = (int(size), backend)
        if key in reported:
            raise ValueError(f"{name}: duplicate result {key}")
        reported[key] = tuple(map(float, (mid, low, high)))
    rows = []
    calls = []
    for size in sizes:
        for backend in ("host-rdma-multicast", "nccl"):
            run_means = []
            for repeat in range(1, 4):
                per_call = samples[size, backend, repeat]
                if sorted(per_call) != list(range(1, 11)):
                    raise ValueError(f"{name}: missing calls {(size, backend, repeat)}")
                run_means.append(mean(per_call.values()))
                for call, latency in sorted(per_call.items()):
                    calls.append(dict(case=name, bytes=size, backend=backend,
                                      repeat=repeat, call=call,
                                      host_latency_us=latency, wrong=0))
            actual = (median(run_means), min(run_means), max(run_means))
            if any(abs(a - b) > 0.002 for a, b in zip(actual, reported[size, backend])):
                raise ValueError(f"{name}: reported summary mismatch {(size, backend)}")
            rows.append(dict(case=name, bytes=size, backend=backend,
                             min_run_mean_us=actual[1],
                             median_run_mean_us=actual[0],
                             max_run_mean_us=actual[2]))
    if len(samples) != len(sizes) * 2 * 3 or len(reported) != len(sizes) * 2:
        raise ValueError(f"{name}: extra or missing records")
    return rows, calls


def main():
    cases = (
        ("auto_1k_64k", tuple(1 << p for p in range(10, 17))),
        ("eager_128k_256k", (131072, 262144)),
        ("eager_1k_64k", tuple(1 << p for p in range(10, 17))),
        ("auto_128k_256k", (131072, 262144)),
    )
    rows, calls = [], []
    for name, sizes in cases:
        a, b = parse_log(name, sizes)
        rows.extend(a)
        calls.extend(b)
    write_csv("summary.csv", rows)
    write_csv("samples.csv", calls)
    print(f"validated {len(rows)} results, {len(calls)} timed calls")


if __name__ == "__main__":
    main()
