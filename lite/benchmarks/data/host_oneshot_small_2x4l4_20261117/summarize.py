#!/usr/bin/env python3
"""Summarize all three fresh variants without selecting winners by size."""
import csv
import re
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
sample = re.compile(r"^# sample,allreduce,(\d+),(\d+),([\d.]+),([\deE.+-]+)$", re.M)
rounds = {}
modes = ("baseline", "p2p", "host")
for mode in modes:
    rounds[mode] = {}
    for r in range(1, 4):
        text = (root / "raw" / f"{mode}_{r}.log").read_text()
        sizes = {}
        for size, index, us, error in sample.findall(text):
            assert float(error) <= 1e-5, (mode, r, size, error)
            sizes.setdefault(int(size), []).append(float(us))
        assert sorted(sizes) == [1024 << i for i in range(9)]
        for size, values in sizes.items():
            assert len(values) == 30
            rounds[mode].setdefault(size, []).append(statistics.mean(values))

with (root / "summary.csv").open("w") as f:
    writer = csv.writer(f)
    columns = ["bytes"]
    for mode in modes:
        columns += [f"{mode}_us", f"{mode}_min_us", f"{mode}_max_us"]
    columns += ["baseline_over_host", "p2p_over_host"]
    writer.writerow(columns)
    print("| Input/GPU | Current µs | P2P one-shot µs | Host one-shot µs | Current / host |")
    print("|---|---:|---:|---:|---:|")
    for size in sorted(rounds["baseline"]):
        row = [size]
        medians = {}
        for mode in modes:
            values = rounds[mode][size]
            medians[mode] = statistics.median(values)
            row += [medians[mode], min(values), max(values)]
        b, p, h = [medians[m] for m in modes]
        row += [b / h, p / h]
        writer.writerow(row)
        print(f"| {size // 1024} KiB | {b:.2f} | {p:.2f} | {h:.2f} | {b / h:.2f}× |")
