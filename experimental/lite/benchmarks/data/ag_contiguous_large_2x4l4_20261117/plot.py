#!/usr/bin/env python3
"""Plot the recorded fixed-configuration large AllGather comparison."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader((ROOT / "summary.csv").open()))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
for mode, label in [("packet_sm", "Existing SM packet pipeline"),
                    ("contiguous_dma", "Contiguous CE, full buffer"),
                    ("contiguous_sm", "Contiguous SM, full buffer")]:
    points = sorted((r for r in rows if r["mode"] == mode), key=lambda r: int(r["bytes"]))
    sizes = [int(r["bytes"]) for r in points]
    x = [n / (1 << 20) for n in sizes]
    median = [float(r["median_us"]) for r in points]
    low = [float(r["min_us"]) for r in points]
    high = [float(r["max_us"]) for r in points]
    line, = axes[0].plot(x, [v / 1000 for v in median], "o-", label=label, markersize=4)
    axes[0].fill_between(x, [v / 1000 for v in low], [v / 1000 for v in high],
                         color=line.get_color(), alpha=0.15)
    axes[1].plot(x, [7 * n / (8 * us * 1000) for n, us in zip(sizes, median)],
                 "o-", color=line.get_color(), label=label, markersize=4)
for ax in axes:
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Complete output per GPU (MiB)")
    ax.set_xticks([0.5, 2, 8, 32, 128, 512, 1024], ["0.5", "2", "8", "32", "128", "512", "1024"])
    ax.grid(True, alpha=0.25)
axes[0].set_yscale("log")
axes[0].set_ylabel("End-to-end latency (ms)")
axes[0].set_title("Median of 3 run means; shading = min–max")
axes[1].set_ylabel("AllGather bus bandwidth: 7N/(8t) (GB/s)")
axes[1].set_title("Fixed configurations; no best-per-size selection")
axes[1].legend(fontsize=8, loc="lower right")
fig.suptitle("2 hosts × 4 NVIDIA L4 • FP32 • slowest-process host submission through completion")
fig.savefig(ROOT / "results.png", dpi=180)
fig.savefig(ROOT / "results.svg")
