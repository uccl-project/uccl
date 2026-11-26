#!/usr/bin/env python3
"""Plot measured eight-L4 CPU-router fusion latency bars."""

import csv
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import ROOT, clean_axis, configure, save


DATA = ROOT / "benchmarks/data/fusion_8l4_20261110"
METHODS = (
    ("cublas+nccl", "cuBLAS + NCCL", "#D65A00"),
    ("cublas+router", "cuBLAS + UCCL-Lite", "#0068B5"),
    ("fused+router", "Fused UCCL-Lite", "#148D91"),
)
SHAPES = {
    "gemm-ar": ((64, 1024, 1024), (128, 1024, 1024),
                (128, 2048, 1024), (256, 2048, 2048),
                (512, 4096, 4096)),
    "gemm-rs": ((64, 1024, 1024), (128, 1024, 1024),
                (128, 2048, 1024), (256, 2048, 2048),
                (512, 4096, 4096)),
    "ag-gemm": ((128, 1024, 1024), (512, 4096, 4096),
                (1024, 1024, 4096)),
}


def load():
    samples = defaultdict(list)
    for op, prefix in (("gemm-ar", "samples_ar128"),
                       ("gemm-rs", "samples"),
                       ("ag-gemm", "samples_ag_tuned")):
        for run in (1, 2):
            path = DATA / f"{prefix}_run{run}.csv"
            with path.open() as stream:
                for row in csv.DictReader(stream):
                    if row["op"] != op:
                        continue
                    key = (op, int(row["m"]), int(row["n"]),
                           int(row["k"]), row["method"])
                    samples[(run, key)].append(float(row["latency_ms"]))
    expected = {(op, *shape, method) for op, shapes in SHAPES.items()
                for shape in shapes for method, *_ in METHODS}
    if {key for _, key in samples} != expected:
        raise ValueError("incomplete fusion sweep")
    result = {}
    for key in expected:
        per_run = []
        for run in (1, 2):
            values = samples[(run, key)]
            if len(values) != 50 or min(values) <= 0:
                raise ValueError(f"bad fusion samples for run {run}: {key}")
            per_run.append(statistics.median(values))
        result[key] = statistics.median(per_run)
    return result


def plot(data, op, preview):
    shapes = SHAPES[op]
    groups = (shapes[:3], shapes[3:]) if op != "ag-gemm" else \
             (shapes[:1], shapes[1:])
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1),
                             gridspec_kw={"width_ratios": [len(groups[0]),
                                                           len(groups[1])]})
    fig.subplots_adjust(left=0.07, right=0.99, top=0.77, bottom=0.23,
                        wspace=0.28)
    for ax, group in zip(axes, groups):
        clean_axis(ax)
        x = list(range(len(group)))
        width = 0.24 if len(group) > 1 else 0.18
        for idx, (method, label, color) in enumerate(METHODS):
            values = [data[(op, *shape, method)] for shape in group]
            bars = ax.bar([i + (idx - 1) * width for i in x], values,
                          width=width, label=label, color=color)
            for bar, value in zip(bars, values):
                ax.annotate(f"{value:.2f}",
                            (bar.get_x() + bar.get_width() / 2, value),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x, [f"({m},{n},{k})" for m, n, k in group])
        ax.tick_params(axis="x", labelsize=8.5)
        ax.set_ylabel("Latency (ms)")
        ax.set_xlabel("Shape (M, N, K)")
        upper = max(data[(op, *shape, method)] for shape in group
                    for method, *_ in METHODS)
        ax.set_ylim(0, upper * 1.25)
        ax.margins(x=0.07)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98),
               ncol=3, frameon=False, fontsize=11.5, columnspacing=2.2)
    save(fig, f"fusion-8l4-{op}", preview)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    configure()
    data = load()
    for op in SHAPES:
        plot(data, op, args.preview)
        for shape in SHAPES[op]:
            print(op, shape, *(f"{data[(op, *shape, method)]:.4f}"
                                 for method, *_ in METHODS))


if __name__ == "__main__":
    main()
