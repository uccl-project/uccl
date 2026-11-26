#!/usr/bin/env python3
"""Match the existing serving/training figures using validated PRO 6000 data."""
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


def draw(data, kind, preview):
    serving = kind == "serving"
    cases = list(data)
    labels = []
    for case in cases:
        dims = list(map(int, case.split(":")))
        labels.append(f"{dims[0]} × {dims[1]}" + (f"\n{dims[2]} output" if serving else ""))
    configure(black_axes=True)
    plt.rcParams.update({"axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12})
    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.76, bottom=0.23)
    title = "serving" if serving else "block fine-tuning"
    fig.suptitle(f"GPT-2-large {title} · 2×RTX PRO 6000", y=0.97, fontsize=16, fontweight="bold")
    subtitle = ("Weight-sharded adapter · whole HTTP batch" if serving else
                "Full step: forward + backward + AG/RS + SGD")
    fig.text(0.5, 0.88, subtitle, ha="center", fontsize=12)
    clean_axis(ax)
    positions = list(range(len(cases)))
    maximum = max(data[c][b + "_s"] * 1000 for c in cases for b in ("nccl", "peer"))
    for backend, label, color, shift in (
        ("nccl", "NCCL (default)", "#F6C99F", -0.17),
        ("peer", "UCCL Lite", "#0068B5", 0.17),
    ):
        values = [data[c][backend + "_s"] * 1000 for c in cases]
        bars = ax.bar([x + shift for x in positions], values, width=0.22, color=color, label=label)
        ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=12)
    for x, case in enumerate(cases):
        top = max(data[case][b + "_s"] * 1000 for b in ("nccl", "peer"))
        y = top + maximum * 0.18
        ax.plot([x - .17, x - .17, x + .17, x + .17],
                [y - maximum * .02, y, y, y - maximum * .02], color="#0068B5", linewidth=1.2)
        ax.text(x, y + maximum * .02, f"↓ {data[case]['latency_reduction_pct']:.1f}%",
                ha="center", va="bottom", color="#0068B5", fontsize=14, fontweight="bold")
    ax.set_xticks(positions, labels)
    ax.set_xlabel("Requests × input tokens; output tokens below" if serving else
                  "Global batch × sequence tokens")
    ax.set_ylabel("Whole client batch latency (ms)" if serving else "Training step latency (ms)")
    ax.set_ylim(0, maximum * 1.4)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=2,
              frameon=False, fontsize=12, borderaxespad=0)
    metric_direction(ax)
    save(fig, f"{kind}-large-message-e2e-pro6000", preview)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    for kind in ("serving", "training"):
        draw(summary[kind], kind, args.preview)


if __name__ == "__main__":
    main()
