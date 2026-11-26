#!/usr/bin/env python3
"""Summarize fresh checked rounds; retain every size and both fixed variants."""
import csv
import re
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
sample = re.compile(r"^# sample,allreduce,(\d+),(\d+),([\d.]+),([\deE.+-]+)$", re.M)
rounds = {}
for mode in ("baseline", "oneshot"):
    rounds[mode] = {}
    for r in range(1, 4):
        text = (root / "raw" / f"{mode}_{r}.log").read_text()
        sizes = {}
        for size, index, us, error in sample.findall(text):
            assert float(error) <= 1e-5, (mode, r, size, error)
            sizes.setdefault(int(size), []).append(float(us))
        assert sorted(sizes) == [1024 << i for i in range(9)]
        for size, values in sizes.items():
            assert len(values) == 30, (mode, r, size, len(values))
            rounds[mode].setdefault(size, []).append(statistics.mean(values))

with (root / "summary.csv").open("w") as f:
    writer = csv.writer(f)
    writer.writerow(["bytes", "baseline_us", "baseline_min_us", "baseline_max_us",
                     "oneshot_us", "oneshot_min_us", "oneshot_max_us",
                     "oneshot_over_baseline"])
    print("| Input/GPU | Baseline µs | One-shot µs | One-shot / baseline |")
    print("|---|---:|---:|---:|")
    for size in sorted(rounds["baseline"]):
        a = rounds["baseline"][size]
        b = rounds["oneshot"][size]
        ma, mb = statistics.median(a), statistics.median(b)
        writer.writerow([size, ma, min(a), max(a), mb, min(b), max(b), mb / ma])
        print(f"| {size // 1024} KiB | {ma:.2f} | {mb:.2f} | {mb / ma:.2f}× |")
