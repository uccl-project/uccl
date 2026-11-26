"""Plot fixed AR/AG configurations and their paired NCCL controls."""

import argparse
import csv
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA.parents[2] / "scripts"))
from figure_style import clean_axis, configure

OPERATIONS = (("allreduce", "compact_combined"), ("allgather", "combined"))


def make_figure(rows, operation, variant):
    selected = sorted(
        (row for row in rows if row["collective"] == operation
         and row["variant"] == variant and int(row["bytes"]) <= 1 << 20),
        key=lambda row: int(row["bytes"]),
    )
    sizes = [int(row["bytes"]) for row in selected]
    if sizes != [1 << power for power in range(10, 21)]:
        raise ValueError(f"incomplete confirmation series: {operation}")

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
        values = [float(row[field]) for row in selected]
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError(f"invalid timing: {operation}, {field}")
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
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path, help="optional PNG output directory")
    args = parser.parse_args()
    with (DATA / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    configure()
    for operation, variant in OPERATIONS:
        fig = make_figure(rows, operation, variant)
        fig.savefig(DATA / f"{operation}.svg", metadata={"Date": None},
                    facecolor="white")
        if args.preview_dir:
            args.preview_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.preview_dir / f"{operation}.png", dpi=160,
                        facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
