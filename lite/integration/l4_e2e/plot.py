#!/usr/bin/env python3
"""Plot the L4 results using the shared serving/training figure style."""
import argparse
from io import StringIO
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from figure_style import clean_axis, configure, metric_direction


def draw(summary, kind, output, preview):
    serving = kind == "serving"
    rows = [row for row in summary["cases"] if row["kind"] == kind]
    configure(black_axes=True)
    plt.rcParams.update({"axes.labelsize": 13, "xtick.labelsize": 12,
                         "ytick.labelsize": 12})
    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.80, bottom=0.19)
    hardware = "4×NVIDIA L4" if summary["nodes"] == 1 else "2×4 NVIDIA L4"
    fig.suptitle(f"{summary['model_label']} {kind} · {hardware}",
                 y=0.97, fontsize=16, fontweight="bold")
    clean_axis(ax)
    positions = list(range(len(rows)))
    maximum = max(row[key] for row in rows for key in ("nccl_ms", "peer_ms"))
    for key, label, color, shift, phases in (
        ("nccl_ms", "NCCL (default)", "#F6C99F", -0.17, (0, 3)),
        ("peer_ms", "UCCL Lite", "#0068B5", 0.17, (1, 2)),
    ):
        errors = [abs(row["phase_medians_ms"][phases[1]] -
                      row["phase_medians_ms"][phases[0]]) / 2 for row in rows]
        bars = ax.bar([x + shift for x in positions], [row[key] for row in rows],
                      width=0.22, color=color, label=label, yerr=errors,
                      capsize=3, error_kw={"elinewidth": 1})
        ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=12)
    for x, row in enumerate(rows):
        y = max(row["nccl_ms"], row["peer_ms"]) + maximum * 0.18
        ax.plot([x - .17, x - .17, x + .17, x + .17],
                [y - maximum * .02, y, y, y - maximum * .02],
                color="#0068B5", linewidth=1.2)
        ax.text(x, y + maximum * .02, f"↓ {row['latency_reduction_pct']:.1f}%",
                ha="center", va="bottom", color="#0068B5",
                fontsize=14, fontweight="bold")
    ax.set_xticks(positions, [row["case"].replace(":", " × ") for row in rows])
    ax.set_xlabel("Requests × input × output tokens" if serving else
                  "Global batch × sequence tokens")
    ax.set_ylabel("Whole client batch latency (ms)" if serving else
                  "Training step latency (ms)")
    ax.set_ylim(0, maximum * 1.4)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=2,
              frameon=False, fontsize=12, borderaxespad=0)
    metric_direction(ax)
    name = f"{kind}-e2e-{summary['nodes']}x4l4"
    svg = StringIO()
    fig.savefig(svg, format="svg", metadata={"Date": None}, facecolor="white")
    (output / f"{name}.svg").write_text(
        "\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + "\n")
    fig.savefig(preview / f"{name}.png", dpi=140, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path, default=ROOT / "build/l4_e2e/figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.preview_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.summary.read_text())
    for kind in ("serving", "training"):
        draw(summary, kind, args.out, args.preview_dir)


if __name__ == "__main__":
    main()
