#!/usr/bin/env python3
"""Plot the prepared eight-L4 resident contract against matched NCCL graphs."""
import argparse
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import ROOT, clean_axis, configure, save

DATA = ROOT / "benchmarks/data/transfer_4to8_20261114/summary.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    with DATA.open() as stream:
        rows = list(csv.DictReader(stream))
    data = {(r["experiment"], int(r["bytes"]), r["variant"]): r for r in rows}
    if len(data) != len(rows):
        raise ValueError("duplicate resident measurement")
    configure()
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.3))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.68, bottom=0.23, wspace=0.28)
    fig.text(0.5, 0.97, "Prepared resident collectives · 8×L4", ha="center",
             va="top", fontsize=16, fontweight="bold")
    sizes = [1 << power for power in range(10, 17)]
    for ax, op, title in zip(axes, ("ar", "ag", "rs"),
                             ("AllReduce", "AllGather", "ReduceScatter")):
        clean_axis(ax)
        ax.set_title(title)
        ax.set_xscale("log", base=2)
        ax.set_xlim(sizes[0] / 1.3, sizes[-1] * 1.3)
        ax.set_xticks(sizes, [f"{size >> 10}K" for size in sizes])
        ax.set_xlabel("Complete tensor bytes per rank")
        upper = 0
        for variant, label, color, marker, style in (
            ("resident", "UCCL-Lite resident", "#0068B5", "o", "-"),
            ("nccl-graph", "NCCL 2.28.9 graph", "#D65A00", "s", (0, (5, 2.5))),
        ):
            points = [data[f"confirm_tiny_{op}", size, variant] for size in sizes]
            for point in points:
                if point["timing"] != "host-global-completion" or int(point["sweeps"]) != 2:
                    raise ValueError("resident figure requires matched host confirmation")
                lo, median, hi = (float(point[k]) for k in ("min_us", "median_us", "max_us"))
                if not (math.isfinite(hi) and 0 < lo <= median <= hi):
                    raise ValueError("invalid resident measurement")
            values = [float(p["median_us"]) for p in points]
            upper = max(upper, *values)
            ax.plot(sizes, values, label=label, color=color, marker=marker,
                    linestyle=style, linewidth=2.8, markersize=6,
                    markeredgecolor="white", markeredgewidth=0.7)
        ax.set_ylim(0, upper * 1.14)
        ax.set_ylabel("Host completion latency (µs)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.89),
               ncol=2, frameon=False, fontsize=12, handlelength=2.8)
    fig.text(0.5, 0.075,
             "BF16 · input-ready, host-blocking · one resident CTA/GPU · preparation excluded · two reversed paired sweeps",
             ha="center", fontsize=9, color="#607486")
    fig.text(0.5, 0.025,
             "AR in place; AG/RS out of place. Workers stop during NCCL controls. Ordinary-call timings are shown separately.",
             ha="center", fontsize=9, color="#607486")
    save(fig, "collectives-8l4-resident-small", args.preview)


if __name__ == "__main__":
    main()
