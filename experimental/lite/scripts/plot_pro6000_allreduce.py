#!/usr/bin/env python3
"""Publish the complete frozen-policy comparison using the repository style."""
import argparse
import csv
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
from figure_style import (
    clean_axis,
    collective_legend,
    collective_panels,
    configure,
    save,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--preview", action="store_true")
    p.add_argument(
        "--name", default="collectives-pro6000x8-allreduce-fp32-ce-optimized"
    )
    a = p.parse_args()
    rows = list(csv.DictReader(a.summary.open()))
    assert [int(r["bytes"]) for r in rows] == [1 << p for p in range(10, 31)]
    configure(black_axes=True)
    fig, axes = collective_panels()
    for ax in axes:
        clean_axis(ax)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Complete tensor bytes per rank")
    for j in (0, 1):
        subset = [
            r
            for r in rows
            if (int(r["bytes"]) <= 1 << 20 if j == 0 else int(r["bytes"]) >= 1 << 20)
        ]
        x = [int(r["bytes"]) for r in subset]
        axes[j].set_xlim(x[0] / 1.35, x[-1] * 1.35)
        upper = 0.0
        for key, label, color, marker, style in (
            ("nccl", "NCCL CE (tuned)", "#B899C8", "^", (0, (6, 2, 1, 2))),
            ("ours", "UCCL Lite", "#0068B5", "o", "-"),
        ):
            value = [float(r[key + "_median_us"]) for r in subset]
            low = [float(r[key + "_min_us"]) for r in subset]
            high = [float(r[key + "_max_us"]) for r in subset]
            if j:
                value = [n * 1.75 / (v * 1000) for n, v in zip(x, value)]
                low, high = (
                    [n * 1.75 / (v * 1000) for n, v in zip(x, high)],
                    [n * 1.75 / (v * 1000) for n, v in zip(x, low)],
                )
            upper = max(upper, *high)
            axes[j].fill_between(x, low, high, color=color, alpha=0.12, linewidth=0)
            axes[j].plot(
                x,
                value,
                label=label,
                color=color,
                marker=marker,
                linestyle=style,
                linewidth=3.6 if key == "ours" else 2.2,
                markersize=6.5 if key == "ours" else 5,
                markeredgecolor="white",
                markeredgewidth=0.7,
                zorder=4 if key == "ours" else 2,
            )
        axes[j].set_ylim(0, upper * 1.13)
    axes[0].set_ylabel("Latency (µs)")
    axes[0].set_xticks(
        [1 << p for p in (10, 12, 14, 16, 18, 20)],
        ["1K", "4K", "16K", "64K", "256K", "1M"],
    )
    axes[1].set_ylabel("Bus throughput (GB/s)")
    axes[1].set_xticks(
        [1 << p for p in (20, 22, 24, 26, 28, 30)],
        ["1M", "4M", "16M", "64M", "256M", "1G"],
    )
    collective_legend(fig, axes[0])
    save(fig, a.name, a.preview)


if __name__ == "__main__":
    main()
