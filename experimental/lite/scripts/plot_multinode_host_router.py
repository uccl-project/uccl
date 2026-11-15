#!/usr/bin/env python3
"""Plot the checked 1 MiB–1 GiB host-router AllGather for the README."""

import argparse
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import ROOT, clean_axis, collective_panels, configure, save

DATA = ROOT / "benchmarks/data/ag_host_router_2x4l4_20261115/validated/summary.csv"
NAME = "collectives-2x4l4-allgather-host-router"
MODES = (
    ("shared_ready", "UCCL Lite (CPU router)", "#0068B5", "o", "-"),
    ("nccl_shared_ready", "NCCL 2.28.7 Ring+Simple", "#D65A00", "s", (0, (5, 2.5))),
)


def load():
    result = {}
    with DATA.open(newline="") as source:
        for row in csv.DictReader(source):
            if row["mode"] not in {mode[0] for mode in MODES}:
                continue
            key = (int(row["bytes"]), row["mode"])
            if key in result:
                raise ValueError(f"duplicate point: {key}")
            low, us, high = (float(row[field]) for field in
                             ("min_us", "median_us", "max_us"))
            if not all(math.isfinite(value) and value > 0 for value in (low, us, high)) or not low <= us <= high:
                raise ValueError(f"invalid latency: {key}")
            bus = key[0] * 7 / 8 / us / 1000
            if abs(bus - float(row["bus_GBps"])) > 0.001:
                raise ValueError(f"inconsistent bandwidth: {key}")
            result[key] = (us, bus)
    expected = {(1 << power, mode[0]) for power in range(20, 31) for mode in MODES}
    if set(result) != expected:
        raise ValueError("incomplete or unexpected AllGather sweep")
    return result


def draw(preview=False):
    configure(black_axes=True)
    data = load()
    fig, axes = collective_panels()
    for ax, powers, metric, ylabel, title in (
        (axes[0], range(20, 24), 0, "Latency (µs)", "1–8 MiB latency"),
        (axes[1], range(20, 31), 1, "Bus throughput (GB/s)", "1 MiB–1 GiB throughput"),
    ):
        sizes = [1 << power for power in powers]
        clean_axis(ax)
        ax.set_xscale("log", base=2)
        ax.set_xlim(sizes[0] / 1.35, sizes[-1] * 1.35)
        ticks = sizes if metric == 0 else [1 << p for p in (20, 22, 24, 26, 28, 30)]
        ax.set_xticks(ticks, [str(size >> 20) for size in ticks])
        ax.set_xlabel("Complete output per GPU (MiB)")
        ax.set_ylabel(ylabel)
        upper = 0
        for mode, label, color, marker, linestyle in MODES:
            values = [data[size, mode][metric] for size in sizes]
            upper = max(upper, *values)
            ax.plot(sizes, values, color=color, marker=marker, linestyle=linestyle,
                    linewidth=3.6 if mode == "shared_ready" else 2.2,
                    markersize=6.5 if mode == "shared_ready" else 5, markeredgecolor="white",
                    markeredgewidth=0.7, label=label)
        ax.set_ylim(0, upper * 1.13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=1, frameon=False, fontsize=11.5, handlelength=1.6,
               handletextpad=0.5, labelspacing=0.3)
    save(fig, NAME, preview)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    draw(args.preview)


if __name__ == "__main__":
    main()
