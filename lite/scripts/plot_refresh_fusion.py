#!/usr/bin/env python3
"""Draw paired complete-operation latency bars from the fresh fusion CSVs."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figure_style import (GRID, INK, ROOT, clean_axis, configure,
                          figure_metric_direction, heading, save)

DATA = ROOT / "benchmarks/data/refresh_20261110"
UCCL_COLOR = "#0068B5"
NCCL_COLOR = "#D8A17A"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def bars(ax, labels, baseline, own, title, xlabel, baseline_label,
         own_label="UCCL Lite selected"):
    y = np.arange(len(labels))
    clean_axis(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.barh(y - 0.19, baseline, height=0.36, color=NCCL_COLOR,
            label=baseline_label)
    ax.barh(y + 0.19, own, height=0.36, color=UCCL_COLOR,
            label=own_label)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=12, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, max(baseline + own) * 1.24)
    for index, (base, candidate) in enumerate(zip(baseline, own)):
        ax.text(base, index - 0.19, f" {base:.3g}", va="center", fontsize=8,
                color=INK)
        ax.text(candidate, index + 0.19, f" {candidate:.3g}", va="center",
                fontsize=8, color=INK)


def dense(preview: bool) -> None:
    data = rows(DATA / "fusion_dense_summary.csv")
    expected = {(shape, op) for shape in (
        "llama8-m128", "llama8-m256", "llama8-m512", "llama8-m1024",
        "llama8-m2048", "qwen32-m2048", "llama70-m2048",
        "reference-m4096") for op in ("AG", "RS", "AR")}
    if {(row["shape"], row["operation"]) for row in data} != expected:
        raise ValueError("incomplete dense fusion summary")
    figure, axes = plt.subplots(3, 2, figsize=(15, 13))
    heading(figure, "Dense fusion · all refreshed shapes",
            "4 × L4 · FP16 · complete-operation latency · lower is better",
            left=0.12)
    for row_index, (op, title) in enumerate((
        ("AG", "AllGather + GEMM"),
        ("RS", "GEMM + ReduceScatter"),
        ("AR", "GEMM + AllReduce"),
    )):
        selected = [row for row in data if row["operation"] == op]
        for column, group in enumerate((selected[:4], selected[4:])):
            def label(shape):
                model, m = shape.rsplit("-m", 1)
                model = {"llama8": "Llama 8B", "qwen32": "Qwen 32B",
                         "llama70": "Llama 70B", "reference": "Reference"}[model]
                return f"{model} · M={m}"
            labels = [label(row["shape"]) for row in group]
            baseline = [float(row["baseline_median_ms"]) for row in group]
            own = [float(row["uccl_median_ms"]) for row in group]
            bars(axes[row_index, column], labels, baseline, own,
                 title + (" · smaller shapes" if column == 0 else " · larger shapes"),
                 "Latency (ms) ↓", "NCCL collective + cuBLAS")
    figure_metric_direction(figure, y=0.94)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles[::-1], labels[::-1], loc="upper right",
                  bbox_to_anchor=(0.98, 0.87), ncol=2, frameon=False,
                  fontsize=11)
    figure.tight_layout(rect=(0.02, 0.04, 0.99, 0.78), h_pad=2.2, w_pad=3)
    save(figure, "dense-fusion-4l4-refresh", preview)


def moe(preview: bool) -> None:
    data = rows(DATA / "fusion_moe_summary.csv")
    if [int(row["tokens_per_rank"]) for row in data] != [1 << p for p in range(3, 16)]:
        raise ValueError("incomplete MoE token sweep")
    figure, axes = plt.subplots(1, 2, figsize=(15, 9))
    heading(figure, "Expert-parallel MoE · complete layer",
            "4 × L4 · BF16 · complete-layer latency · lower is better",
            left=0.12)
    for ax, group, title in ((axes[0], data[:7], "Small batches"),
                             (axes[1], data[7:], "Large batches")):
        labels = [row["tokens_per_rank"] + " tokens/rank" for row in group]
        baseline = [float(row["baseline_us"]) / 1000 for row in group]
        own = [float(row["uccl_us"]) / 1000 for row in group]
        bars(ax, labels, baseline, own, title, "Full-layer latency (ms) ↓",
             "NCCL + grouped cuBLAS", "UCCL Lite fused")
    figure_metric_direction(figure, y=0.94)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles[::-1], labels[::-1], loc="upper right",
                  bbox_to_anchor=(0.98, 0.87), ncol=2, frameon=False,
                  fontsize=11)
    figure.tight_layout(rect=(0.02, 0.04, 0.99, 0.78), w_pad=3)
    save(figure, "moe-fusion-4l4-refresh", preview)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    configure(black_axes=True)
    dense(args.preview)
    moe(args.preview)
