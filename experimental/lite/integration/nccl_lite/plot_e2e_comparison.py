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
from figure_style import clean_axis, configure, save


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
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.31, wspace=0.23)
    fig.suptitle("GLM-4.5-Air-FP8 serving · 8×NVIDIA L4", y=0.97,
                 fontsize=16, fontweight="bold")
    fig.text(0.5, 0.88, "Long prompts expose a larger end-to-end benefit; the short-prompt control retains a modest gain",
             ha="center", fontsize=10)
    positions = list(range(len(cases)))
    for ax in axes:
        clean_axis(ax)
        ax.set_xticks(positions, labels, fontsize=8.5)
        ax.set_xlabel("Requests × input tokens; output tokens below")
    maximum = max(data[c][b + "_s"] for c in cases for b in ("nccl", "packet"))
    for backend, label, color, shift in (
        ("nccl", "NCCL (default)", "#D8A17A", -0.19),
        ("packet", "UCCL Lite", "#0068B5", 0.19),
    ):
        values = [data[c][backend + "_s"] for c in cases]
        bars = axes[0].bar([x + shift for x in positions], values, width=0.36,
                           color=color, label=label)
        axes[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    axes[0].set_ylabel("Whole client batch latency (s)")
    axes[0].set_ylim(0, maximum * 1.27)
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    gains = [data[c]["latency_reduction_pct"] for c in cases]
    bars = axes[1].bar(positions, gains, width=0.6, color="#0068B5")
    axes[1].bar_label(bars, fmt="%.1f%%", padding=4, fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Latency reduction versus NCCL (%)")
    axes[1].set_ylim(min(0, min(gains) * 1.2), max(gains) * 1.3)
    fig.text(0.07, 0.13,
             "NCCL 2.27.5 · BF16 collectives · eager TP8/DP8/EP8 · includes queueing and HTTP completion; excludes startup/warmups.\n"
             "First four cases: 2 sessions × 3 calls/backend, KV fraction 0.87. Last case: 1 session × 3 calls/backend, KV fraction 0.90.\n"
             "The same fixed server profile is used for both backends within each study; no per-size best-variant selection.",
             fontsize=8.3, va="top", linespacing=1.5)
    save(fig, "sglang-large-prefill-e2e-8l4", args.preview)

    data = summary["training"]
    cases = ["ddp", "fsdp"]
    if set(data) != set(cases):
        raise ValueError("expected both DDP and adapted-FSDP training results")
    labels = ["DDP", "FSDP · experimental adapter"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.77, bottom=0.29, wspace=0.23)
    fig.suptitle("638M-parameter transformer training · 8×NVIDIA L4", y=0.97,
                 fontsize=16, fontweight="bold")
    fig.text(0.5, 0.87, "Full step: forward + backward + communication + SGD",
             ha="center", fontsize=10)
    positions = list(range(len(cases)))
    for ax in axes:
        clean_axis(ax)
        ax.set_xticks(positions, labels, fontsize=9)
    for backend, label, color, shift in (
        ("nccl", "NCCL (default)", "#D8A17A", -0.19),
        ("packet", "UCCL Lite", "#0068B5", 0.19),
    ):
        values = [data[c][backend + "_s"] * 1000 for c in cases]
        bars = axes[0].bar([x + shift for x in positions], values, width=0.36,
                           color=color, label=label)
        axes[0].bar_label(bars, fmt="%.0f", padding=4, fontsize=10)
    axes[0].set_ylabel("Training step latency (ms)")
    axes[0].set_ylim(0, 1000)
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    gains = [data[c]["latency_reduction_pct"] for c in cases]
    bars = axes[1].bar(positions, gains, width=0.6, color="#0068B5")
    axes[1].bar_label(bars, fmt="%.1f%%", padding=4, fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Latency reduction versus NCCL (%)")
    axes[1].set_ylim(0, 30)
    fig.text(0.07, 0.14,
             "NCCL 2.28.9 · BF16 · random weights / generated tokens · 1 × 128 tokens per GPU · excludes data loading and warmups.\n"
             "Median of 2 job medians/backend/mode; 3 warmup + 10 timed steps/job. Fixed variants; no per-size best selection.\n"
             "FSDP uses a separate ReduceScatter communicator with PyTorch 2.9.1; the ordinary shared-communicator path fails the stream guard.",
             fontsize=8.3, va="top", linespacing=1.5)
    save(fig, "training-large-message-e2e-8l4", args.preview)


if __name__ == "__main__":
    main()
