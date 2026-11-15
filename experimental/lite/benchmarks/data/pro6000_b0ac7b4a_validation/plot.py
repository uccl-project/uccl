"""Plot PRO6000 joint/P2P and NCCL references with explicit size-order scopes."""
import argparse
import csv
import math
from pathlib import Path
from statistics import median
import sys

import matplotlib
matplotlib.use("Agg")

DATA = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA.parents[2] / "scripts"))
from figure_style import (clean_axis, collective_legend, collective_panels,
                          configure, save)

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
    tuned_path = DATA.parent / "nccl_tuned_pro6000_20261115" / "cases.csv"
    with tuned_path.open() as stream:
        tuned_rows = list(csv.DictReader(stream))
    tuned = {(row["operation"], int(row["bytes"])): row for row in tuned_rows}
    if len(tuned) != len(tuned_rows) or set(tuned) != expected:
        raise ValueError("incomplete or duplicate tuned NCCL results")
    for key, row in tuned.items():
        for order in ("forward", "reverse"):
            value = float(row[f"{order}_us"])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("invalid tuned NCCL latency")
            result[key][f"tuned_{order}_us"] = value
    ascending_path = DATA.parent / "nccl_ascending_pro6000_20261115" / "cases.csv"
    with ascending_path.open() as stream:
        ascending_rows = list(csv.DictReader(stream))
    ascending = {(row["operation"], int(row["bytes"])): row for row in ascending_rows}
    if len(ascending) != len(ascending_rows) or set(ascending) != expected:
        raise ValueError("incomplete or duplicate ascending NCCL results")
    for key, row in ascending.items():
        for run in ("run1", "run2"):
            value = float(row[f"{run}_us"])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("invalid ascending NCCL latency")
            result[key][f"ascending_{run}_us"] = value
    return result


def make_figure(data, operation, title, factor):
    fig, axes = collective_panels(f"RTX PRO 6000 ×2 · {title}")
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
            ("nccl_", "NCCL (default P2P)", "#D8A17A", "s", (0, (5, 2.5))),
            ("tuned_", "NCCL SM (tuned)", "#82B4A3", "D", (0, (2, 1.5))),
            ("ascending_", "NCCL CE (tuned)", "#B899C8", "^", (0, (6, 2, 1, 2))),
            ("", "UCCL Lite", "#0068B5", "o", "-"),
        ):
            orders = ("run1", "run2") if prefix == "ascending_" else ("forward", "reverse")
            runs = [[float(data[operation, size][f"{prefix}{order}_us"])
                     for order in orders] for size in sizes]
            values = [factor * size / median(us) / 1000 if throughput else median(us)
                      for size, us in zip(sizes, runs)]
            bounds = [[factor * size / us / 1000 if throughput else us
                       for us in pair] for size, pair in zip(sizes, runs)]
            low, high = [min(pair) for pair in bounds], [max(pair) for pair in bounds]
            upper = max(upper, *high)
            is_uccl = prefix == ""
            ax.fill_between(sizes, low, high, color=color,
                            alpha=0.12 if is_uccl else 0.08, linewidth=0)
            ax.plot(sizes, values, color=color, linewidth=3.6 if is_uccl else 2.2,
                    marker=marker, linestyle=linestyle, markersize=6.5 if is_uccl else 5,
                    markeredgecolor="white", markeredgewidth=0.7, label=label,
                    zorder=4 if is_uccl else 2)
        ax.set_ylim(0, upper * 1.13)
    collective_legend(fig, axes[0])
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
