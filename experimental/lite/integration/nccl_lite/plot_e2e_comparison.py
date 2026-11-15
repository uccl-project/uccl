#!/usr/bin/env python3
"""Plot serving and training results from the validated large-message comparison."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from figure_style import clean_axis, configure, metric_direction, save


def label_reductions(ax, data, cases, scale=1):
    maximum = max(data[c][b + "_s"] * scale for c in cases for b in ("nccl", "packet"))
    for x, case in enumerate(cases):
        top = max(data[case][b + "_s"] * scale for b in ("nccl", "packet"))
        y = top + maximum * 0.18
        ax.plot([x - 0.17, x - 0.17, x + 0.17, x + 0.17],
                [y - maximum * 0.02, y, y, y - maximum * 0.02],
                color="#0068B5", linewidth=1.2)
        ax.text(x, y + maximum * 0.02,
                f"↓ {data[case]['latency_reduction_pct']:.1f}%",
                ha="center", va="bottom", color="#0068B5", fontsize=14,
                fontweight="bold")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    data = summary["serving"]
    cases = ["8:1024:1", "32:1024:1", "32:2048:1", "64:256:32", "32:2048:32"]
    if set(data) != set(cases):
        raise ValueError("expected all five serving cases, including the short-output control")
    labels = []
    for case in cases:
        requests, inputs, outputs = map(int, case.split(":"))
        labels.append(f"{requests} × {inputs:,}\n{outputs} output")
    configure()
    plt.rcParams.update({"axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12})
    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.76, bottom=0.23)
    fig.suptitle("GLM-4.5-Air-FP8 serving · 8×NVIDIA L4", y=0.97,
                 fontsize=16, fontweight="bold")
    fig.text(0.5, 0.88, "Whole client batch · includes queueing",
             ha="center", fontsize=12)
    positions = list(range(len(cases)))
    clean_axis(ax)
    ax.set_xticks(positions, labels, fontsize=12)
    ax.set_xlabel("Requests × input tokens; output tokens below")
    maximum = max(data[c][b + "_s"] for c in cases for b in ("nccl", "packet"))
    for backend, label, color, shift in (
        ("nccl", "NCCL (default)", "#F6C99F", -0.17),
        ("packet", "UCCL Lite", "#0068B5", 0.17),
    ):
        values = [data[c][backend + "_s"] for c in cases]
        bars = ax.bar([x + shift for x in positions], values, width=0.22,
                           color=color, label=label)
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=12)
    ax.set_ylabel("Whole client batch latency (s)")
    ax.set_ylim(0, maximum * 1.38)
    ax.legend(loc="upper left", frameon=False, fontsize=12)
    label_reductions(ax, data, cases)
    metric_direction(ax)
    save(fig, "sglang-large-prefill-e2e-8l4", args.preview)

    data = summary["training"]
    cases = ["ddp", "fsdp"]
    if set(data) != set(cases):
        raise ValueError("expected both DDP and adapted-FSDP training results")
    labels = ["DDP", "FSDP\nexperimental adapter"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.76, bottom=0.23)
    fig.suptitle("638M-parameter transformer training · 8×NVIDIA L4", y=0.97,
                 fontsize=16, fontweight="bold")
    fig.text(0.5, 0.87, "Full step: forward + backward + communication + SGD",
             ha="center", fontsize=12)
    positions = list(range(len(cases)))
    clean_axis(ax)
    ax.set_xticks(positions, labels, fontsize=12)
    for backend, label, color, shift in (
        ("nccl", "NCCL (default)", "#F6C99F", -0.17),
        ("packet", "UCCL Lite", "#0068B5", 0.17),
    ):
        values = [data[c][backend + "_s"] * 1000 for c in cases]
        bars = ax.bar([x + shift for x in positions], values, width=0.22,
                           color=color, label=label)
        ax.bar_label(bars, fmt="%.0f", padding=4, fontsize=12)
    ax.set_ylabel("Training step latency (ms)")
    ax.set_ylim(0, 1050)
    ax.set_xlim(-0.65, 1.65)
    ax.legend(loc="upper left", frameon=False, fontsize=12)
    label_reductions(ax, data, cases, scale=1000)
    metric_direction(ax)
    save(fig, "training-large-message-e2e-8l4", args.preview)


if __name__ == "__main__":
    main()
