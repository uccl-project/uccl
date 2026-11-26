#!/usr/bin/env python3
"""Plot audited diagnostic GPU-copy time; not a final performance comparison."""
import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figure_style import configure, clean_axis, save, BLUE, TEAL, AMBER


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--staged", type=Path, required=True)
    p.add_argument("--ring", type=Path, required=True)
    p.add_argument("--preview", action="store_true")
    a = p.parse_args()
    results = [json.loads(path.read_text()) for path in (a.staged, a.ring)]
    for data in results:
        assert data["size"] == 1 << 30 and len(data["ranks"]) == 8
        assert not data[
            "diagnostics"
        ], "Use complete traces from the compatible profiler"
        assert all(len(r["calls"]) == 20 for r in data["ranks"])
    configure(black_axes=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.30, top=0.95, wspace=0.3)
    labels = ["UCCL Lite\nstaged CE", "UCCL Lite\nring CE"]
    for ax in axes:
        clean_axis(ax)
    busy = [mean(r["means"]["copy_busy_ms"] for r in d["ranks"]) for d in results]
    span = [mean(r["means"]["kernel_ms"] for r in d["ranks"]) for d in results]
    axes[0].bar([0, 1], busy, color=BLUE, label="Payload CE copies")
    axes[0].bar(
        [0, 1],
        [s - b for s, b in zip(span, busy)],
        bottom=busy,
        color=AMBER,
        label="Remaining kernel time",
    )
    axes[0].set_ylim(0, 51)
    axes[0].set_xticks([0, 1], labels)
    axes[0].set_ylabel("Mean per-rank GPU kernel time (ms)")
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.27),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    for i, d in enumerate(results):
        groups = [[], [], []]
        for r in d["ranks"]:
            for e in r["edges"]:
                level = (e["src"] ^ e["dst"]).bit_length() - 1
                groups[level].append(e["GBps"])
        x = [j + (i - 0.5) * 0.34 for j in range(3)]
        axes[1].bar(
            x,
            [mean(g) for g in groups],
            width=0.32,
            color=BLUE if i == 0 else TEAL,
            label=labels[i].replace("\n", " · "),
        )
        for position, group in zip(x, groups):
            axes[1].plot(
                [position] * len(group), group, ".", color="#172A3A", markersize=3
            )
    axes[1].set_xticks([0, 1, 2], ["PIX", "NODE", "SYS"])
    axes[1].set_ylim(0, 60)
    axes[1].set_ylabel("Outgoing payload-copy rate (GB/s)")
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.27),
        frameon=False,
        fontsize=9,
        ncol=2,
    )
    save(fig, "collectives-pro6000x8-allreduce-ce-bottleneck", a.preview)


if __name__ == "__main__":
    main()
