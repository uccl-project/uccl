"""Plot the integrated PRO6000 x2 joint/P2P results against default NCCL."""
import argparse
import csv
import math
from pathlib import Path
from statistics import median
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA.parents[2] / "scripts"))
from figure_style import clean_axis, configure, save

OPERATIONS = (("allreduce", "AllReduce", 1.0),
              ("allgather", "AllGather", 0.5),
              ("reducescatter", "ReduceScatter", 0.5))


def load():
    with (DATA / "cases.csv").open() as stream:
        rows = [row for row in csv.DictReader(stream) if row["backend"] == "joint"]
    result = {(row["operation"], int(row["bytes"])): row for row in rows}
    expected = {(operation, 1 << power) for operation, *_ in OPERATIONS
                for power in range(10, 31)}
    if len(result) != len(rows) or set(result) != expected:
        raise ValueError("incomplete or duplicate PRO6000 joint results")
    for row in rows:
        for order in ("forward", "reverse"):
            if row[f"{order}_route"] != "peer":
                raise ValueError("figure expects every calibrated route to be peer")
            for prefix in ("", "nccl_"):
                value = float(row[f"{prefix}{order}_us"])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("invalid latency")
    return result


def make_figure(data, operation, title, factor):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.74, bottom=0.23, wspace=0.28)
    fig.suptitle(f"RTX PRO 6000 ×2 · {title}", y=0.98,
                 fontsize=15, fontweight="bold")
    for ax, powers, ylabel, throughput in (
        (axes[0], range(10, 21), "Latency (µs)", False),
        (axes[1], range(20, 31), "Bus throughput (GB/s)", True),
    ):
        sizes = [1 << power for power in powers]
        ticks = sizes[::2]
        clean_axis(ax)
        ax.set_xscale("log", base=2)
        ax.set_xlim(sizes[0] / 1.35, sizes[-1] * 1.35)
        ax.set_xticks(ticks, [f"{size >> 10}K" if size < 1 << 20 else
                             f"{size >> 20}M" if size < 1 << 30 else "1G"
                             for size in ticks])
        ax.set_xlabel("Complete tensor bytes per rank")
        ax.set_ylabel(ylabel)
        upper = 0.0
        for prefix, label, color, marker, linestyle in (
            ("nccl_", "NCCL 2.29.7 (default P2P)", "#D65A00", "s", (0, (5, 2.5))),
            ("", "UCCL Lite (joint/P2P)", "#0068B5", "o", "-"),
        ):
            runs = [[float(data[operation, size][f"{prefix}{order}_us"])
                     for order in ("forward", "reverse")] for size in sizes]
            values = [factor * size / median(us) / 1000 if throughput else median(us)
                      for size, us in zip(sizes, runs)]
            bounds = [[factor * size / us / 1000 if throughput else us
                       for us in pair] for size, pair in zip(sizes, runs)]
            low, high = [min(pair) for pair in bounds], [max(pair) for pair in bounds]
            upper = max(upper, *high)
            ax.fill_between(sizes, low, high, color=color, alpha=0.12, linewidth=0)
            ax.plot(sizes, values, color=color, linewidth=2.8, marker=marker,
                    linestyle=linestyle, markersize=6.0,
                    markeredgecolor="white", markeredgewidth=0.7, label=label)
        ax.set_ylim(0, upper * 1.13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper center",
               bbox_to_anchor=(0.5, 0.91), ncol=2, frameon=False,
               fontsize=12, handlelength=2.8, columnspacing=2.5)
    fig.text(0.5, 0.075,
             "BF16 · out of place · prepared eager · slowest-rank CUDA events including submission gaps",
             ha="center", fontsize=8.5, color="#607486")
    fig.text(0.5, 0.025,
             "Median of two run medians; shading: run range · 100 warmups + 5 × 20 calls/run · commit b0ac7b4a",
             ha="center", fontsize=8.5, color="#607486")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    configure()
    data = load()
    for operation, title, factor in OPERATIONS:
        fig = make_figure(data, operation, title, factor)
        save(fig, f"collectives-pro6000x2-{operation}-overview", args.preview)


if __name__ == "__main__":
    main()
