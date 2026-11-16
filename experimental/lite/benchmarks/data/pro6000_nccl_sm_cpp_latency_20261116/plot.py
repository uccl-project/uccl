#!/usr/bin/env python3
"""Plot the matched FP32/C++ latency data in the shared collective style."""

import argparse
import csv
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent
ROOT = DATA.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from figure_style import (clean_axis, collective_legend, configure,
                          metric_direction, save)
from summarize import main as verify_measurements


def draw(preview):
    verify_measurements()
    with (DATA / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    sizes = [int(row["bytes"]) for row in rows]
    if sizes != [1 << power for power in range(10, 21)]:
        raise ValueError("Expected the complete measured 1-KiB–1-MiB grid")
    configure(black_axes=True)
    # One latency panel: the matched experiment measures through 1 MiB.
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    fig.subplots_adjust(left=0.20, right=0.97, top=0.72, bottom=0.17)
    ax.tick_params(axis="both", labelsize=12)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, min_n_ticks=3))
    clean_axis(ax)
    metric_direction(ax)
    ax.set_xscale("log", base=2)
    ax.set_xlim(sizes[0] / 1.35, sizes[-1] * 1.35)
    ticks = sizes[::2]
    ax.set_xticks(ticks, [f"{n >> 10}K" if n < 1 << 20 else "1M" for n in ticks])
    ax.set_xlabel("Complete tensor bytes per rank")
    ax.set_ylabel("Latency (µs)")
    upper = 0.0
    for prefix, label, color, marker, linestyle in (
        ("nccl_default", "NCCL (default)", "#D8A17A", "s", (0, (5, 2.5))),
        ("nccl_sm_tuned", "NCCL SM (tuned)", "#82B4A3", "D", (0, (2, 1.5))),
        ("pcieccl_original", "UCCL Lite", "#0068B5", "o", "-"),
    ):
        values = [float(row[f"{prefix}_median_us"]) for row in rows]
        low = [float(row[f"{prefix}_min_us"]) for row in rows]
        high = [float(row[f"{prefix}_max_us"]) for row in rows]
        if not all(math.isfinite(a) and 0 < lo <= a <= hi
                   for lo, a, hi in zip(low, values, high)):
            raise ValueError("Invalid measurement or observed range")
        upper = max(upper, *high)
        is_uccl = prefix == "pcieccl_original"
        ax.fill_between(sizes, low, high, color=color,
                        alpha=0.12 if is_uccl else 0.08, linewidth=0)
        ax.plot(sizes, values, color=color, linewidth=3.6 if is_uccl else 2.2,
                marker=marker, linestyle=linestyle, markersize=6.5 if is_uccl else 5,
                markeredgecolor="white", markeredgewidth=0.7, label=label,
                zorder=4 if is_uccl else 2)
    ax.set_ylim(0, upper * 1.13)
    collective_legend(fig, ax)
    save(fig, "collectives-pro6000x8-allreduce-fp32-cpp-latency", preview)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    draw(parser.parse_args().preview)
