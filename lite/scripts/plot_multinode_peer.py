#!/usr/bin/env python3
"""Draw latency from paired two-node collective data."""

import argparse
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import BASELINE, BLUE, GRID, MUTED, ROOT
from figure_style import clean_axis, configure, heading, save


DATA = ROOT / "benchmarks/data/multinode_peer_20261110"


def read(name: str) -> dict[tuple[str, int], dict[str, str]]:
    with (DATA / name).open() as stream:
        return {(row["collective"], int(row["bytes"])): row
                for row in csv.DictReader(stream)}


def value(row: dict[str, str], backend: str) -> float:
    number = float(row[f"{backend}_median_us"])
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"invalid {backend} timing")
    return number


def large_message_figure(preview: bool, small, large, allreduce) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    heading(fig, "Two-node large-message latency",
            "2 × 4 L4  ·  FP32  ·  three paired runs  ·  NCCL 2.28.7", left=0.075)
    fig.subplots_adjust(left=0.16, right=0.985, top=0.71, bottom=0.20,
                        wspace=0.38)
    operations = (("allgather", "AllGather"),
                  ("reducescatter", "ReduceScatter"),
                  ("allreduce", "AllReduce"))
    for ax, size, title, limit, ticks in (
        (axes[0], 1 << 26, "64 MiB per rank", 9, [0, 2, 4, 6, 8]),
        (axes[1], 1 << 30, "1 GiB per rank", 130, [0, 25, 50, 75, 100, 125]),
    ):
        clean_axis(ax)
        ax.set_title(title, loc="left", pad=12)
        ax.set_xlim(0, limit)
        ax.set_ylim(len(operations) - 0.55, -0.55)
        ax.set_xticks(ticks)
        ax.set_yticks(range(len(operations)), [name for _, name in operations])
        ax.set_xlabel("Slowest-rank latency (ms) ↓")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        for y, (operation, _) in enumerate(operations):
            records = allreduce if operation == "allreduce" else (
                small if size <= 1 << 26 else large)
            row = records[operation, size]
            candidate = value(row, "uccl") / 1000
            baseline = value(row, "nccl") / 1000
            ax.barh(y + 0.18, baseline, height=0.32, color=BASELINE,
                    label="NCCL 2.28.7" if y == 0 else None)
            ax.barh(y - 0.18, candidate, height=0.32, color=BLUE,
                    label="UCCL Lite" if y == 0 else None)
            offset = 0.08 if size == 1 << 26 else 1.1
            digits = 2 if size == 1 << 26 else 1
            ax.text(baseline + offset, y + 0.18, f"{baseline:.{digits}f}",
                    va="center", fontsize=9)
            ax.text(candidate + offset, y - 0.18, f"{candidate:.{digits}f}",
                    va="center", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper right",
               bbox_to_anchor=(0.985, 0.85), ncol=2,
               frameon=False, fontsize=9.5)
    fig.text(0.16, 0.04,
             "AG/RS: out of place; AR: in place. Each panel has its own linear millisecond scale.",
             fontsize=9, color=MUTED)
    save(fig, "multinode-peer", preview)


def latency_figure(preview: bool, small) -> None:
    sizes = [1 << power for power in (10, 12, 14, 16, 18, 20)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7))
    heading(fig, "Two-node small-message latency",
            "2 × 4 L4  ·  FP32  ·  out of place  ·  NCCL 2.28.7", left=0.075)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.71, bottom=0.17, wspace=0.29)
    for ax, operation, title in ((axes[0], "allgather", "AllGather"),
                                 (axes[1], "reducescatter", "ReduceScatter")):
        clean_axis(ax)
        ax.set_title(title, loc="left", pad=12)
        maximum = 0
        for backend, label, color in (("uccl", "UCCL Lite", BLUE),
                                      ("nccl", "NCCL 2.28.7", BASELINE)):
            latencies = [value(small[operation, size], backend) for size in sizes]
            maximum = max(maximum, *latencies)
            ax.plot(sizes, latencies, color=color, linewidth=2.2, marker="o",
                    markersize=5, markeredgecolor="white", markeredgewidth=0.7,
                    label=label)
        ax.set_xscale("log", base=2)
        ax.set_xlim(sizes[0] / 1.4, sizes[-1] * 1.3)
        ax.set_xticks([1 << power for power in (10, 14, 20)],
                      ["1 KiB", "16 KiB", "1 MiB"])
        ax.set_ylim(0, maximum * 1.16)
        ax.set_xlabel("Complete tensor bytes per rank")
        ax.set_ylabel("Latency (µs) ↓")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right",
               bbox_to_anchor=(0.985, 0.85), ncol=2, frameon=False,
               fontsize=9.5)
    save(fig, "multinode-latency", preview)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true",
                        help="Also export PNG/PDF copies to build/figures/.")
    args = parser.parse_args()
    configure()
    small = read("summary.csv")
    large = read("summary_large.csv")
    allreduce = read("summary_ar.csv")
    large_message_figure(args.preview, small, large, allreduce)
    latency_figure(args.preview, small)


if __name__ == "__main__":
    main()
