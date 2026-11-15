"""Plot the direct-medium ReduceScatter configuration against its fresh NCCL control."""

import csv
from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from figure_style import clean_axis, collective_panels, configure, save

DATA = Path(__file__).resolve().parent
NAME = "collectives-2x4l4-reducescatter-direct-medium"
STYLES = (
    ("candidate", "UCCL Lite", "#0068B5", "o", "-"),
    ("nccl", "NCCL", "#D8A17A", "s", (0, (5, 2.5))),
)


def main():
    with (DATA / "summary.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    configure(black_axes=True)
    fig, axes = collective_panels()
    for ax, powers, metric, ylabel in (
        (axes[0], range(10, 21), "median_us", "Latency (µs)"),
        (axes[1], range(20, 31), "bus_GBps", "Bus throughput (GB/s)"),
    ):
        sizes = [1 << power for power in powers]
        clean_axis(ax)
        ax.set_xscale("log", base=2)
        ax.set_xlim(sizes[0] / 1.35, sizes[-1] * 1.35)
        ticks = [1 << p for p in ((10, 12, 14, 16, 18, 20)
                                   if metric == "median_us" else
                                   (20, 22, 24, 26, 28, 30))]
        ax.set_xticks(ticks, [f"{size >> 10}K" if size < 1 << 20 else
                             f"{size >> 20}M" if size < 1 << 30 else "1G"
                             for size in ticks])
        ax.set_xlabel("Complete tensor bytes per rank")
        ax.set_ylabel(ylabel)
        upper = 0
        for mode, label, color, marker, linestyle in STYLES:
            selected = {int(row["input_bytes_per_gpu"]): row for row in rows
                        if row["mode"] == mode}
            values = [float(selected[size][metric]) for size in sizes]
            style = dict(color=color, marker=marker, linestyle=linestyle,
                         linewidth=3.6 if mode == "candidate" else 2.2,
                         markersize=6.5 if mode == "candidate" else 5,
                         markeredgecolor="white", markeredgewidth=0.7, label=label)
            if metric == "median_us" and mode == "candidate":
                low = [float(selected[size]["min_us"]) for size in sizes]
                high = [float(selected[size]["max_us"]) for size in sizes]
                error = [[value - bound for value, bound in zip(values, low)],
                         [bound - value for value, bound in zip(values, high)]]
                ax.errorbar(sizes, values, yerr=error, capsize=2, **style)
                upper = max(upper, *high)
            else:
                ax.plot(sizes, values, **style)
                upper = max(upper, *values)
        ax.set_ylim(0, upper * 1.13)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=2, frameon=False, fontsize=11.5, handlelength=1.6,
               handletextpad=0.5, labelspacing=0.3)
    save(fig, NAME, preview=True)
    shutil.copyfile(ROOT / "docs/assets" / f"{NAME}.svg", DATA / "results.svg")
    shutil.copyfile(ROOT / "build/figures" / f"{NAME}.png", DATA / "results.png")


if __name__ == "__main__":
    main()
