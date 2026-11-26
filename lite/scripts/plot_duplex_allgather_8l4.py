#!/usr/bin/env python3
"""Plot the eight-L4 duplex AllGather sweep against packet routing and NCCL."""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import ROOT, clean_axis, configure, save


DATA = ROOT / "benchmarks/data/duplex_allgather_8l4_20261111/summary.csv"
METHODS = (
    ("nccl_median_us", "NCCL", "#D65A00", "s"),
    ("packet_median_us", "UCCL-Lite packet router", "#0068B5", "o"),
    ("duplex_median_us", "UCCL-Lite duplex", "#148D91", "^"),
)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    with DATA.open() as stream:
        rows = list(csv.DictReader(stream))
    sizes = [int(row["bytes"]) for row in rows]
    if sizes != [1 << power for power in range(20, 31)]:
        raise ValueError("incomplete eight-L4 AllGather sweep")
    configure()
    fig, ax = plt.subplots(figsize=(8.3, 4.2))
    fig.subplots_adjust(left=0.11, right=0.985, top=0.76, bottom=0.20)
    clean_axis(ax)
    for field, label, color, marker in METHODS:
        values = [0.875 * size / float(row[field]) / 1000
                  for size, row in zip(sizes, rows)]
        ax.plot(sizes, values, color=color, marker=marker, linewidth=2.7,
                markersize=6.5, markeredgecolor="white",
                markeredgewidth=0.7, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlim(sizes[0] / 1.35, sizes[-1] * 1.35)
    ticks = [1 << power for power in (20, 22, 24, 26, 28, 30)]
    ax.set_xticks(ticks, ["1M", "4M", "16M", "64M", "256M", "1G"])
    ax.set_xlabel("Complete output bytes per GPU")
    ax.set_ylabel("Bus throughput (GB/s)")
    ax.set_ylim(0, 6.1)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=3, frameon=False, fontsize=11, columnspacing=2.0,
               handlelength=2.3)
    save(fig, "duplex-allgather-8l4", args.preview)


if __name__ == "__main__":
    main()
