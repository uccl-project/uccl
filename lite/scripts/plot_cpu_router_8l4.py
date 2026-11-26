#!/usr/bin/env python3
"""Plot the eight-L4 local/router calibration with a separate NCCL control."""

import argparse
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figure_style import BASELINE, BLUE, MUTED, ROOT, TEAL, clean_axis, configure, heading, save


DATA = ROOT / "benchmarks/data/local_fallback_8l4_20261110_v3"
CONTROL = ROOT / "benchmarks/data/cpu_router_8l4_20261110"
OPERATIONS = (
    ("all_reduce", "AllReduce", 14 / 8),
    ("all_gather", "AllGather", 7 / 8),
    ("reduce_scatter", "ReduceScatter", 7 / 8),
)
PLACEMENTS = (("in_place", "In place", "1"), ("out_of_place", "Out of place", "0"))
SIZES = [1 << power for power in range(20, 31)]


def load_rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open() as stream:
        return list(csv.DictReader(line for line in stream if not line.startswith("#")))


def load_placement(placement: str, in_place: str):
    calibration = load_rows(f"{placement}_calibration.csv")
    selection = load_rows(f"{placement}_auto.csv")
    records = {}
    decisions = {}
    for row in calibration:
        key = (row["collective"], int(row["bytes"]), row["backend"])
        if key in records or row["in_place"] != in_place:
            raise ValueError(f"duplicate or misplaced calibration row: {key}")
        times = [float(row[field]) for field in ("min_us", "median_us", "max_us")]
        if not all(math.isfinite(value) and value > 0 for value in times) or times != sorted(times):
            raise ValueError(f"invalid timing range: {key}")
        records[key] = times
    for row in selection:
        key = (row["collective"], int(row["bytes"]))
        if key in decisions or row["in_place"] != in_place:
            raise ValueError(f"duplicate or misplaced selection row: {key}")
        decisions[key] = row["backend"]
    expected = {(operation, size) for operation, _, _ in OPERATIONS for size in SIZES}
    if set(decisions) != expected or set(records) != {
        (operation, size, backend)
        for operation, size in expected
        for backend in ("fallback", "cpu-router")
    }:
        raise ValueError(f"incomplete eight-L4 {placement} sweep")
    if any(backend not in ("fallback", "cpu-router") for backend in decisions.values()):
        raise ValueError(f"unknown automatic backend in {placement}")
    return records, decisions


def load_nccl_control(placement: str, in_place: str):
    with (CONTROL / f"{placement}_calibration.csv").open() as stream:
        rows = list(csv.DictReader(line for line in stream if not line.startswith("#")))
    control = {}
    for row in rows:
        if row["backend"] != "fallback":
            continue
        key = (row["collective"], int(row["bytes"]))
        if key in control or row["in_place"] != in_place:
            raise ValueError(f"duplicate or misplaced NCCL control row: {key}")
        value = float(row["median_us"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid NCCL control timing: {key}")
        control[key] = value
    expected = {(operation, size) for operation, _, _ in OPERATIONS for size in SIZES}
    if set(control) != expected:
        raise ValueError(f"incomplete NCCL control for {placement}")
    return control


def bandwidth(size: int, latency_us: float, bus_factor: float) -> float:
    return bus_factor * size / latency_us / 1000  # decimal GB/s


def plot(preview: bool) -> None:
    configure()
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.8), sharey=True)
    heading(
        fig,
        "Eight-L4 UCCL Lite · adaptive collectives",
        "BF16  ·  local fallback / CPU router paired  ·  NCCL 2.28.9 separate control",
        left=0.075,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.73, bottom=0.15,
                        hspace=0.43, wspace=0.20)
    upper = 0.0
    for row_index, (placement, placement_title, in_place) in enumerate(PLACEMENTS):
        records, decisions = load_placement(placement, in_place)
        control = load_nccl_control(placement, in_place)
        for column, (operation, title, factor) in enumerate(OPERATIONS):
            ax = axes[row_index, column]
            clean_axis(ax)
            ax.set_title(f"{title} · {placement_title}", loc="left", pad=10)
            ax.set_xscale("log", base=2)
            ax.set_xlim(SIZES[0] / 1.35, SIZES[-1] * 1.35)
            ax.set_xticks([1 << power for power in (20, 24, 28, 30)],
                          ["1 MiB", "16 MiB", "256 MiB", "1 GiB"])
            if row_index == 1:
                ax.set_xlabel("Complete tensor bytes per rank")
            if column == 0:
                ax.set_ylabel("Bus bandwidth (GB/s) ↑")
            nccl = [bandwidth(size, control[operation, size], factor)
                    for size in SIZES]
            upper = max(upper, *nccl)
            ax.plot(SIZES, nccl, color=BASELINE, linewidth=1.7,
                    linestyle=(0, (4, 3)), zorder=1)
            for backend, color in (("fallback", TEAL), ("cpu-router", BLUE)):
                times = [records[operation, size, backend] for size in SIZES]
                medians = [bandwidth(size, timing[1], factor)
                           for size, timing in zip(SIZES, times)]
                lows = [bandwidth(size, timing[2], factor)
                        for size, timing in zip(SIZES, times)]
                highs = [bandwidth(size, timing[0], factor)
                         for size, timing in zip(SIZES, times)]
                upper = max(upper, *highs)
                ax.fill_between(SIZES, lows, highs, color=color, alpha=0.13,
                                linewidth=0)
                ax.plot(SIZES, medians, color=color, linewidth=2.1, zorder=2)
                if backend == "fallback":
                    ax.scatter(SIZES, medians, s=31, color=color, edgecolors="white",
                               linewidths=0.6, zorder=3)
                else:
                    selected = [decisions[operation, size] == "cpu-router"
                                for size in SIZES]
                    for chosen in (True, False):
                        points = [(size, value) for size, value, flag
                                  in zip(SIZES, medians, selected) if flag == chosen]
                        if points:
                            ax.scatter(*zip(*points), s=37,
                                       facecolors=BLUE if chosen else "white",
                                       edgecolors=BLUE, linewidths=1.3, zorder=4)
    for ax in axes.flat:
        ax.set_ylim(0, upper * 1.10)
    fig.legend(
        handles=[
            Line2D([], [], color=BASELINE,
                   linestyle=(0, (4, 3)), label="NCCL 2.28.9 · separate control"),
            Line2D([], [], color=TEAL, marker="o", markersize=6,
                   label="UCCL Lite local fallback"),
            Line2D([], [], color=BLUE, marker="o", markersize=6,
                   label="CPU router · auto selects"),
            Line2D([], [], color=BLUE, marker="o", markerfacecolor="white",
                   markersize=6, label="CPU router · auto keeps local"),
        ],
        loc="upper left", bbox_to_anchor=(0.075, 0.84), ncol=4,
        frameon=False, fontsize=8.7,
    )
    fig.text(0.075, 0.035,
            "Shaded bands: min–max of paired run medians. NCCL is a separate control in this local/router replay.",
             color=MUTED, fontsize=9)
    save(fig, "cpu-router-8l4", preview)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="also write PNG and PDF previews")
    plot(parser.parse_args().preview)
