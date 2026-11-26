"""Plot the fixed UCCL Lite configuration and its paired NCCL control."""

import argparse
import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA.parents[2] / "scripts"))
from figure_style import clean_axis, configure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", type=Path, help="optional PNG output path")
    args = parser.parse_args()
    with (DATA / "final_summary.csv").open() as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["variant"] == "hybrid_parallel"
                and int(row["bytes"]) <= 1 << 20]
    sizes = [int(row["bytes"]) for row in rows]

    configure()
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    fig.subplots_adjust(left=0.14, right=0.985, top=0.78, bottom=0.20)
    clean_axis(ax)
    ax.set_xscale("log", base=2)
    ax.set_xlim(sizes[0] / 1.35, sizes[-1] * 1.35)
    ax.set_xticks([1 << power for power in (10, 12, 14, 16, 18, 20)],
                  ["1K", "4K", "16K", "64K", "256K", "1M"])
    ax.set_xlabel("Complete tensor bytes per rank")
    ax.set_ylabel("Latency (µs)")
    upper = 0.0
    for field, label, color, marker, linestyle in (
        ("paired_nccl_median_us", "NCCL 2.28.9", "#D65A00", "s", (0, (5, 2.5))),
        ("median_us", "UCCL-Lite", "#0068B5", "o", "-"),
    ):
        values = [float(row[field]) for row in rows]
        upper = max(upper, *values)
        ax.plot(sizes, values, color=color, linewidth=2.8, marker=marker,
                linestyle=linestyle, markersize=6.0,
                markeredgecolor="white", markeredgewidth=0.7, label=label)
    ax.set_ylim(0, upper * 1.13)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper center",
               bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=False,
               fontsize=12.5, handlelength=2.8, columnspacing=3.0,
               markerscale=1.2)
    fig.savefig(DATA / "summary.svg", metadata={"Date": None}, facecolor="white")
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.preview, dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
