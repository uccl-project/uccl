#!/usr/bin/env python3
"""Plot refreshed four-L4 and two-node collectives in the 8xL4 style."""

import argparse
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import ROOT, clean_axis, configure, save

DATA = ROOT / "benchmarks/data/refresh_20261110"
UCCL_COLOR = "#0068B5"
NCCL_COLOR = "#D65A00"
SMALL_SIZES = [1 << power for power in range(10, 21)]
LARGE_SIZES = [1 << power for power in range(20, 31)]
OPERATIONS = (
    ("single_native_summary.csv", "operation", "allreduce",
     "collectives-4l4-allreduce-refresh", 6 / 4),
    ("single_peer_summary.csv", "operation", "allgather",
     "collectives-4l4-allgather-refresh", 3 / 4),
    ("single_peer_summary.csv", "operation", "reducescatter",
     "collectives-4l4-reducescatter-refresh", 3 / 4),
    ("multinode_paired_summary.csv", "collective", "allreduce",
     "collectives-2x4l4-allreduce-refresh", 14 / 8),
    ("multinode_paired_summary.csv", "collective", "allgather",
     "collectives-2x4l4-allgather-refresh", 7 / 8),
    ("multinode_paired_summary.csv", "collective", "reducescatter",
     "collectives-2x4l4-reducescatter-refresh", 7 / 8),
)


def load(filename: str, operation_field: str, operation: str):
    with (DATA / filename).open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row[operation_field] == operation]
    result = {}
    for row in rows:
        size = int(row["bytes"])
        if size in result:
            raise ValueError(f"duplicate {operation} size: {size}")
        for field in ("uccl_median_us", "nccl_median_us"):
            value = float(row[field])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid {operation} latency: {size}, {field}")
        result[size] = row
    if sorted(result) != [1 << power for power in range(10, 31)]:
        raise ValueError(f"incomplete {operation} sweep")
    return result


def throughput(size: int, latency_us: float, factor: float) -> float:
    return factor * size / latency_us / 1000  # decimal GB/s


def plot_operation(data, filename: str, factor: float, preview: bool,
                   log_small: bool = False,
                   nccl_label: str = "NCCL 2.28.7",
                   x_label: str = "Complete tensor bytes per rank",
                   uccl_label: str = "UCCL-Lite") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.78, bottom=0.20,
                        wspace=0.28)
    for ax, sizes, measure, ylabel, ticks in (
        (axes[0], SMALL_SIZES, lambda size, us: us,
         "Latency (µs)", [1 << p for p in (10, 12, 14, 16, 18, 20)]),
        (axes[1], LARGE_SIZES,
         lambda size, us: throughput(size, us, factor),
         "Bus throughput (GB/s)",
         [1 << p for p in (20, 22, 24, 26, 28, 30)]),
    ):
        clean_axis(ax)
        ax.set_xscale("log", base=2)
        ax.set_xlim(sizes[0] / 1.35, sizes[-1] * 1.35)
        ax.set_xticks(ticks, [
            f"{size >> 10}K" if size < 1 << 20 else
            f"{size >> 20}M" if size < 1 << 30 else "1G"
            for size in ticks
        ])
        ax.set_xlabel(x_label)
        use_log = log_small and ax is axes[0]
        ax.set_ylabel("Latency (µs, log scale)" if use_log else ylabel)
        upper = 0.0
        lower = math.inf
        for field, label, color, marker, linestyle in (
            ("nccl_median_us", nccl_label, NCCL_COLOR, "s", (0, (5, 2.5))),
            ("uccl_median_us", uccl_label, UCCL_COLOR, "o", "-"),
        ):
            values = [measure(size, float(data[size][field]))
                      for size in sizes]
            upper = max(upper, *values)
            lower = min(lower, *values)
            ax.plot(sizes, values, color=color, linewidth=2.8,
                    marker=marker, linestyle=linestyle, markersize=6.0,
                    markeredgecolor="white", markeredgewidth=0.7,
                    label=label)
        if use_log:
            ax.set_yscale("log")
            ax.set_ylim(lower * 0.8, upper * 1.3)
            ticks_y = [20, 30, 50, 100, 200, 500, 1000, 2000, 5000]
            ax.set_yticks([tick for tick in ticks_y
                           if lower * 0.8 < tick < upper * 1.3])
            ax.set_yticklabels([str(tick) for tick in ticks_y
                                if lower * 0.8 < tick < upper * 1.3])
        else:
            ax.set_ylim(0, upper * 1.13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper center",
               bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=False,
               fontsize=12.5, handlelength=2.8, columnspacing=3.0,
               markerscale=1.2)
    save(fig, filename, preview)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    configure()
    for source, field, operation, filename, factor in OPERATIONS:
        data = load(source, field, operation)
        plot_operation(data, filename, factor, args.preview)


if __name__ == "__main__":
    main()
