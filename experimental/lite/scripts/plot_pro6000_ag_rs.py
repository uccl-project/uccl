#!/usr/bin/env python3
"""Plot audited, frozen-policy FP32 AG/RS validation using the collective style."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from figure_style import ROOT, clean_axis, collective_legend, collective_panels, configure, save
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--draft", action="store_true", help="Write build-only PNG previews")
    args = parser.parse_args()
    metadata = json.loads((args.summary / "summary.json").read_text())
    if not metadata.get("policy_sha256") or not metadata.get("validation"):
        raise ValueError("An audited validation summary is required")
    configure(black_axes=True)
    for op, label in (("allgather", "Full output bytes per rank"),
                      ("reducescatter", "Full input bytes per rank")):
        with (args.summary / (op + ".csv")).open() as source:
            rows = list(csv.DictReader(source))
        if [int(row["bytes"]) for row in rows] != [1 << p for p in range(10, 31)]:
            raise ValueError("Expected the complete 1 KiB–1 GiB grid")
        fig, axes = collective_panels()
        for ax in axes:
            clean_axis(ax)
            ax.set_xscale("log", base=2)
            ax.set_xlabel(label)
        for panel in (0, 1):
            subset = [row for row in rows if (int(row["bytes"]) <= 1 << 20
                      if panel == 0 else int(row["bytes"]) >= 1 << 20)]
            sizes = [int(row["bytes"]) for row in subset]
            axes[panel].set_xlim(sizes[0] / 1.35, sizes[-1] * 1.35)
            upper = 0.0
            for key, name, color, marker, style in (
                ("nccl", "NCCL (tuned SM/CE)", "#B899C8", "^", (0, (6, 2, 1, 2))),
                ("ours", "UCCL Lite", "#0068B5", "o", "-"),
            ):
                value = [float(row[key + "_median_us"]) for row in subset]
                low = [float(row[key + "_min_us"]) for row in subset]
                high = [float(row[key + "_max_us"]) for row in subset]
                if panel:
                    value = [n * 0.875 / (v * 1000) for n, v in zip(sizes, value)]
                    low, high = (
                        [n * 0.875 / (v * 1000) for n, v in zip(sizes, high)],
                        [n * 0.875 / (v * 1000) for n, v in zip(sizes, low)],
                    )
                upper = max(upper, *high)
                axes[panel].fill_between(sizes, low, high, color=color, alpha=0.12, linewidth=0)
                axes[panel].plot(sizes, value, label=name, color=color, marker=marker,
                                 linestyle=style, linewidth=3.6 if key == "ours" else 2.2,
                                 markersize=6.5 if key == "ours" else 5,
                                 markeredgecolor="white", markeredgewidth=0.7,
                                 zorder=4 if key == "ours" else 2)
            axes[panel].set_ylim(0, upper * 1.13)
        axes[0].set_ylabel("Latency (µs)")
        axes[0].set_xticks([1 << p for p in (10, 12, 14, 16, 18, 20)],
                          ["1K", "4K", "16K", "64K", "256K", "1M"])
        axes[1].set_ylabel("Bus throughput (GB/s)")
        axes[1].set_xticks([1 << p for p in (20, 22, 24, 26, 28, 30)],
                          ["1M", "4M", "16M", "64M", "256M", "1G"])
        collective_legend(fig, axes[0])
        if args.draft:
            folder = ROOT / "build/figures"
            folder.mkdir(parents=True, exist_ok=True)
            fig.savefig(folder / f"collectives-pro6000x8-{op}-fp32-draft.png", dpi=160, facecolor="white")
            plt.close(fig)
        else:
            save(fig, f"collectives-pro6000x8-{op}-fp32", args.preview)


if __name__ == "__main__":
    main()
