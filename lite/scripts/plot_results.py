#!/usr/bin/env python3
"""Render absolute latency and bandwidth from recorded UCCL Lite results."""

import argparse
import csv
import math
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figure_style import BASELINE, BLUE, GRID, MUTED, ROOT
from figure_style import clean_axis, configure, heading, save


DATA = ROOT / "benchmarks/data"


def rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open() as stream:
        return list(csv.DictReader(stream))


def positive(values, label: str) -> None:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"missing or invalid values in {label}")


def size_label(size: int) -> str:
    for divisor, suffix in ((1 << 30, "GiB"), (1 << 20, "MiB"), (1 << 10, "KiB")):
        if size >= divisor:
            return f"{size / divisor:g} {suffix}"
    return f"{size} B"


def size_axis(ax, sizes, ticks, title: str, ylabel: str, top: float) -> None:
    clean_axis(ax)
    ax.set_title(title, loc="left", pad=12)
    ax.set_xscale("log", base=2)
    ax.set_xlim(min(sizes) / 1.4, max(sizes) * 1.35)
    ax.set_xticks(ticks, [size_label(size) for size in ticks])
    ax.set_ylim(0, top)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Complete tensor bytes per rank")


def measured_line(ax, sizes, values, label: str, color: str):
    positive(values, label)
    ax.plot(sizes, values, color=color, linewidth=2.2,
            marker="o", markersize=4.8, markeredgecolor="white",
            markeredgewidth=0.7, label=label)


def primitive_collectives(preview: bool) -> None:
    data = {(row["collective"], row["mode"], int(row["bytes"]), row["backend"]): row
            for row in rows("primitives_8l4.csv")}
    if len(data) != 288:
        raise ValueError("eight-L4 primitive archive is incomplete")
    for operation, name, title in (
        ("all_gather", "allgather", "AllGather"),
        ("reduce_scatter", "reducescatter", "ReduceScatter"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.3))
        heading(fig, f"Historical eight-GPU {title}",
                "8 × L4  ·  BF16  ·  no GPU peer access  ·  three-run medians",
                left=0.075)
        fig.subplots_adjust(left=0.09, right=0.985, top=0.74, bottom=0.105,
                            hspace=0.48, wspace=0.28)
        all_small_values = []
        all_large_values = []
        for row_index, mode in enumerate(("in_place", "out_of_place")):
            sizes = sorted(size for op, placement, size, backend in data
                           if op == operation and placement == mode and backend == "kernel")
            if len(sizes) != 24:
                raise ValueError(f"incomplete {title} {mode} sweep")
            small = [size for size in sizes if size <= 1 << 19]
            large = [size for size in sizes if size >= 1 << 20]
            placement = "In place" if mode == "in_place" else "Out of place"
            for column, selected, field, metric, ylabel, ticks, collected in (
                (0, small, "median_us", "latency ↓", "Latency (µs)",
                 [128, 4096, 131072], all_small_values),
                (1, large, "bus_bw_gbs", "bus bandwidth ↑",
                 "Bus bandwidth (GB/s)",
                 [1 << power for power in (20, 24, 28, 30)], all_large_values),
            ):
                ax = axes[row_index, column]
                for label, backend, color in (
                    ("CPU-router SM", "kernel", BLUE),
                    ("NCCL 2.27.5", "nccl", BASELINE),
                ):
                    values = [float(data[operation, mode, size, backend][field])
                              for size in selected]
                    measured_line(ax, selected, values, label, color)
                    collected.extend(values)
                size_axis(ax, selected, ticks, f"{placement} · {metric}", ylabel, 1)
        for row in range(2):
            axes[row, 0].set_ylim(0, max(all_small_values) * 1.15)
            axes[row, 1].set_ylim(0, max(all_large_values) * 1.15)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right",
                   bbox_to_anchor=(0.985, 0.86), ncol=2, frameon=False,
                   fontsize=9.5)
        save(fig, name, preview)


def dense_records() -> dict[str, list[tuple[str, int, float, float]]]:
    text = (DATA / "dense_fusion.txt").read_text()
    records: dict[str, list[tuple[str, int, float, float]]] = {
        name: [] for name in ("AG", "RS", "AR")
    }
    pattern = (r"^(\S+)\s+(AG|RS)\s+\((\d+),(\d+),(\d+)\)\s+"
               r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$")
    for match in re.finditer(pattern, text, re.MULTILINE):
        model, operation, m, _, _, baseline, _, fused, _ = match.groups()
        records[operation].append((model, int(m), float(baseline), float(fused)))
    pattern_ar = (r"^(\S+)\s+\((\d+),\d+,\d+\)\s+\d+\s+"
                  r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+x\s+[\d.]+x$")
    for match in re.finditer(pattern_ar, text, re.MULTILINE):
        model, m, baseline, _, fused = match.groups()
        records["AR"].append((model, int(m), float(baseline), float(fused)))
    if any(len(points) != 8 for points in records.values()):
        raise ValueError("dense comparison must have eight shapes per operation")
    for operation, points in records.items():
        positive([value for _, _, baseline, fused in points
                  for value in (baseline, fused)], operation)
    return records


def dense_label(model: str, m: int) -> str:
    name = {
        "Llama-3.1-8B": "Llama 8B",
        "Llama-3.1-70B": "Llama 70B",
        "Qwen2.5-32B": "Qwen 32B",
        "Flux-reference": "Reference",
    }[model]
    return f"{name} · {m // 1024}k" if m >= 1024 else f"{name} · {m}"


def dense(preview: bool) -> None:
    records = dense_records()
    fig, axes = plt.subplots(3, 2, figsize=(14.8, 12.5))
    heading(fig, "Dense fusion · all recorded shapes",
            "4 × L4  ·  FP16  ·  complete-operation latency  ·  lower is better",
            left=0.12)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.76, bottom=0.08,
                        hspace=0.66, wspace=0.52)
    for row, (operation, title) in enumerate(zip(
        ("AG", "RS", "AR"),
        ("AllGather + GEMM", "GEMM + ReduceScatter", "GEMM + AllReduce"),
    )):
        for column, group in enumerate((records[operation][:4],
                                        records[operation][4:])):
            ax = axes[row, column]
            y = np.arange(len(group))
            baseline = [point[2] for point in group]
            fused = [point[3] for point in group]
            ax.barh(y + 0.18, baseline, height=0.32, color=BASELINE,
                    label="cuBLAS + NCCL")
            ax.barh(y - 0.18, fused, height=0.32, color=BLUE,
                    label="UCCL Lite fused")
            ax.set_title(f"{title} · {'smaller' if column == 0 else 'larger'} shapes",
                         loc="left", pad=12)
            ax.set_xlim(0, 1.55 if column == 0 else 30)
            ax.set_xticks([0, 0.5, 1.0, 1.5] if column == 0
                          else [0, 10, 20, 30])
            ax.set_ylim(len(group) - 0.55, -0.55)
            ax.set_yticks(y, [dense_label(model, m) for model, m, _, _ in group])
            ax.tick_params(axis="both", which="both", length=0, pad=6,
                           labelsize=9.2)
            ax.set_axisbelow(True)
            ax.grid(axis="x", color=GRID, linewidth=0.8)
            for index, (base, own) in enumerate(zip(baseline, fused)):
                offset = 0.025 if column == 0 else 0.25
                ax.text(base + offset, index + 0.18, f"{base:.2f}",
                        va="center", fontsize=8.5)
                ax.text(own + offset, index - 0.18, f"{own:.2f}",
                        va="center", fontsize=8.5)
            ax.set_xlabel("Complete-operation latency (ms) ↓")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper right",
               bbox_to_anchor=(0.97, 0.84), ncol=2, frameon=False,
               fontsize=9.5)
    fig.text(0.12, 0.035,
             "Smaller and larger shape columns have different linear latency scales; values are labeled on each bar.",
             fontsize=9, color=MUTED)
    save(fig, "dense-fusion-shapes", preview)


def moe_rows(name: str) -> dict[tuple[int, str], float]:
    points = {}
    for line in (DATA / name).read_text().splitlines():
        fields = line.split()
        if fields and fields[0].isdigit():
            points[int(fields[0]), fields[3]] = float(fields[4])
    return points


def moe(preview: bool) -> None:
    points = moe_rows("moe_layer.txt")
    groups = (("8–1k tokens", [8, 32, 128, 512, 1024]),
              ("2k–8k tokens", [2048, 4096, 8192]),
              ("16k–32k tokens", [16384, 32768]))
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.0),
                             gridspec_kw={"width_ratios": [1.35, 1, 0.78]})
    heading(fig, "Expert-parallel MoE · full layer",
            "4 × L4  ·  DeepSeek-shaped BF16  ·  plan construction excluded",
            left=0.065)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.70, bottom=0.20,
                        wspace=0.30)
    for ax, (title, sizes) in zip(axes, groups):
        baseline = [points[size, "nccl+cublas"] / 1000 for size in sizes]
        fused = [points[size, "fused"] / 1000 for size in sizes]
        positive(baseline + fused, title)
        x = np.arange(len(sizes))
        ax.bar(x - 0.19, baseline, width=0.36, color=BASELINE,
               label="NCCL + grouped cuBLAS")
        ax.bar(x + 0.19, fused, width=0.36, color=BLUE,
               label="UCCL Lite fused")
        ax.set_title(title, loc="left", pad=12)
        ax.set_xticks(x, [f"{size // 1024}k" if size >= 1024 else str(size)
                          for size in sizes])
        ax.set_ylim(0, max(baseline + fused) * 1.20)
        ax.set_ylabel("Latency (ms) ↓")
        ax.set_xlabel("Tokens per rank")
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(axis="both", which="both", length=0, pad=7)
        for values, offset in ((baseline, -0.19), (fused, 0.19)):
            for index, value in enumerate(values):
                label = f"{value:.2f}" if value < 10 else f"{value:.1f}"
                ax.text(index + offset, value + max(baseline + fused) * 0.02,
                        label, ha="center", va="bottom", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper right",
               bbox_to_anchor=(0.985, 0.85), ncol=2, frameon=False,
               fontsize=9.5)
    fig.text(0.065, 0.045,
             "Each panel shows full-layer latency on its own linear millisecond scale.",
             fontsize=9, color=MUTED)
    save(fig, "moe-fusion", preview)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true",
                        help="Also export PNG/PDF copies to build/figures/.")
    args = parser.parse_args()
    configure()
    primitive_collectives(args.preview)
    dense(args.preview)
    moe(args.preview)


if __name__ == "__main__":
    main()
