#!/usr/bin/env python3
"""Publish the complete frozen-policy comparison using the repository style."""
import argparse
import csv
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figure_style import clean_axis, configure, metric_direction, save


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
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7))
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.18, top=0.72, wspace=0.31)
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
        for key, label, color, marker, style in (
            ("nccl", "NCCL CE (tuned)", "#8064A6", "D", (0, (2, 1.5))),
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
            axes[j].fill_between(x, low, high, color=color, alpha=0.12, linewidth=0)
            axes[j].plot(
                x,
                value,
                label=label,
                color=color,
                marker=marker,
                linestyle=style,
                linewidth=3.3 if key == "ours" else 2.2,
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=0.7,
            )
        metric_direction(axes[j], higher=bool(j))
    axes[0].set_ylabel("Completed-call latency (µs)")
    axes[0].set_xticks(
        [1 << p for p in (10, 12, 14, 16, 18, 20)],
        ["1K", "4K", "16K", "64K", "256K", "1M"],
    )
    axes[1].set_ylabel("AllReduce bus bandwidth (GB/s)")
    axes[1].set_xticks(
        [1 << p for p in (20, 22, 24, 26, 28, 30)],
        ["1M", "4M", "16M", "64M", "256M", "1G"],
    )
    x = [int(r["bytes"]) for r in rows]
    ratio = [float(r["median_speedup"]) for r in rows]
    axes[2].plot(x, ratio, color="#0068B5", marker="o", linewidth=2.7, markersize=5)
    axes[2].fill_between(
        x,
        [float(r["min_paired_speedup"]) for r in rows],
        [float(r["max_paired_speedup"]) for r in rows],
        color="#0068B5",
        alpha=0.12,
    )
    axes[2].axhline(1, color="#BF6C36", linestyle="--", linewidth=1.5)
    axes[2].set_yscale("log", base=2)
    axes[2].set_yticks([1, 2, 4, 8, 16, 32], ["1×", "2×", "4×", "8×", "16×", "32×"])
    axes[2].set_ylabel("NCCL CE time / UCCL Lite time")
    axes[2].set_xticks(
        [1 << p for p in (10, 14, 18, 22, 26, 30)],
        ["1K", "16K", "256K", "4M", "64M", "1G"],
    )
    metric_direction(axes[2], higher=True)
    wins = sum(int(r["paired_wins"]) == 5 for r in rows)
    fig.suptitle(
        f"Eight RTX PRO 6000 · FP32 in-place AllReduce · {wins}/21 sizes win in all five pairs",
        y=0.97,
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.885,
        "10 warmups + 20 continuous calls · slowest-rank host timing · frozen per-size policies · five independent sweeps",
        ha="center",
        fontsize=10,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[::-1],
        labels[::-1],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.87),
        ncol=2,
        frameon=False,
    )
    save(fig, a.name, a.preview)


if __name__ == "__main__":
    main()
