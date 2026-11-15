#!/usr/bin/env python3
"""Summarize and plot eight-L4 model-shape AG+GEMM and GEMM+RS samples."""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import ROOT, clean_axis, configure, figure_metric_direction, save


DATA = ROOT / "benchmarks/data/fusion_8l4_models_ag_rs_20261111"
MODELS = (
    ("Qwen3.5-9B", 4096, 1536),
    ("Qwen3.5-27B", 5120, 2176),
    ("Gemma 4 31B IT", 5376, 2688),
    ("Qwen3-32B", 5120, 3200),
)
PHASES = {
    "gemm-rs": (("decode", (1, 32, 64)),
                ("prefill", (2048, 4096, 8192))),
    "ag-gemm": (("decode", (8, 32, 64)),
                ("prefill", (2048, 4096, 8192))),
}
METHODS = (
    ("cublas+nccl", "cuBLAS + NCCL", "#D8A17A"),
    ("cublas+router", "cuBLAS + UCCL Lite", "#82B4A3"),
    ("fused+router", "Fused UCCL Lite", "#0068B5"),
)


def load(op, data_dir):
    samples = defaultdict(list)
    name = op.replace("-", "_")
    for run in (1, 2):
        with (data_dir / f"samples_{name}_run{run}.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["op"] != op:
                    raise ValueError(f"unexpected operation: {row['op']}")
                key = (int(row["m"]), int(row["n"]), int(row["k"]),
                       row["method"])
                samples[(run, key)].append(float(row["latency_ms"]))
    expected = {(m, n, k, method)
                for _, n, k in MODELS
                for _, sizes in PHASES[op] for m in sizes
                for method, _, _ in METHODS}
    if {key for _, key in samples} != expected:
        raise ValueError(f"incomplete or unexpected {op} sweep")
    result = {}
    for key in expected:
        medians = []
        for run in (1, 2):
            values = samples[(run, key)]
            if len(values) != 50 or min(values) <= 0:
                raise ValueError(f"bad samples for {op} run {run}: {key}")
            medians.append(statistics.median(values))
        result[key] = (*medians, statistics.median(medians))
    return result


def write_summary(op, data, data_dir):
    name = op.replace("-", "_")
    with (data_dir / f"summary_{name}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("model", "phase", "m", "n", "k", "method",
                         "run1_median_ms", "run2_median_ms", "median_ms"))
        for model, n, k in MODELS:
            for phase, sizes in PHASES[op]:
                for m in sizes:
                    for method, _, _ in METHODS:
                        writer.writerow((model, phase, m, n, k, method,
                                         *(f"{v:.6f}" for v in
                                           data[(m, n, k, method)])))


def plot(op, data, phase, sizes, preview, tag):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8),
                             sharey=phase == "decode")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.13,
                        hspace=0.4, wspace=0.25)
    shared_upper = max(data[(m, n, k, method)][2]
                       for _, n, k in MODELS for m in sizes
                       for method, _, _ in METHODS) * 1.28
    width = 0.24
    for ax, (model, n, k) in zip(axes.flat, MODELS):
        clean_axis(ax)
        for index, (method, label, color) in enumerate(METHODS):
            values = [data[(m, n, k, method)][2] for m in sizes]
            bars = ax.bar([i + (index - 1) * width for i in range(len(sizes))],
                          values, width=width, label=label, color=color)
            ax.bar_label(bars, fmt="%.3f" if phase == "decode" else "%.2f",
                         padding=2, fontsize=7, color="black")
        ax.set_title(f"{model} · N={n}, K={k}", color="black")
        ax.set_xticks(range(len(sizes)), [str(m) for m in sizes])
        ax.set_ylabel("Latency (ms)")
        local_upper = max(data[(m, n, k, method)][2]
                          for m in sizes for method, _, _ in METHODS) * 1.28
        ax.set_ylim(0, shared_upper if phase == "decode" else local_upper)
        ax.set_xlabel("M (tokens)")
        ax.tick_params(axis="both", colors="black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
    figure_metric_direction(fig)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 0.96),
               ncol=3, frameon=False, fontsize=10, labelcolor="black")
    title = "GEMM + ReduceScatter" if op == "gemm-rs" else "AllGather + GEMM"
    fig.suptitle(f"8×L4 {title} · {phase}", y=0.995,
                 fontsize=16, fontweight="bold", color="black")
    suffix = f"-{tag}" if tag else ""
    save(fig, f"fusion-8l4-model-{op}-{phase}{suffix}-labeled-bars", preview)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=tuple(PHASES))
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    configure(black_axes=True)
    for op in (args.op,) if args.op else PHASES:
        data = load(op, args.data_dir)
        write_summary(op, data, args.data_dir)
        for phase, sizes in PHASES[op]:
            plot(op, data, phase, sizes, args.preview, args.tag)


if __name__ == "__main__":
    main()
