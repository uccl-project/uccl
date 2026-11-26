#!/usr/bin/env python3
"""Plot the complete local reduction-placement ablation, independently of README figures."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import plot_results as style
from paired_summary import summarize

DATA = style.ROOT / "benchmarks/data/reduction-placement"
VARIANTS = (
    ("default", "CPU · 512 KiB (default)", "cpu-router", style.PALE),
    ("cpu", "CPU · 8 MiB", "cpu-router", style.LIGHT),
    ("flat", "GPU flat · 8 MiB", "cpu-router-gpu-reduce", style.MEDIUM),
    ("striped", "GPU striped · 8 MiB", "cpu-router-gpu-striped", style.NAVY),
)


def load(variant, backend, operation, data=DATA):
    with (data / f"{variant}_{operation}_samples.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    sizes = [1 << n for n in range(10, 31)]
    summary = summarize(rows, operation, sizes, 5, 20, 4, backend)
    points = {
        r["full_array_bytes"]: r["speedup_vs_local_nccl"]
        for r in summary
        if r["backend"] == backend
    }
    samples = {
        (size, name, repeat): []
        for size in sizes
        for name in ("nccl", backend)
        for repeat in range(5)
    }
    for row in rows:
        samples[int(row["bytes"]), row["backend"], int(row["repeat"])].append(
            float(row["slowest_rank_us"])
        )
    ranges = {}
    for size in sizes:
        paired = [
            statistics.mean(samples[size, "nccl", r])
            / statistics.mean(samples[size, backend, r])
            for r in range(5)
        ]
        ranges[size] = (min(paired), max(paired))
    return points, ranges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--gpu-progress", action="store_true")
    args = parser.parse_args()
    data = style.ROOT / "benchmarks/data/gpu-progress" if args.gpu_progress else DATA
    variants = (
        (
            ("control", "CPU packet handoff", "cpu-router-gpu-striped", style.LIGHT),
            ("candidate", "GPU packet handoff", "gpu-progress-striped", style.NAVY),
        )
        if args.gpu_progress
        else VARIANTS
    )
    provenance = json.loads((data / "ablation_provenance.json").read_text())
    for name, digest in provenance["data_sha256"].items():
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Archived data checksum mismatch: {name}")
    style.plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": style.GRID,
            "axes.labelcolor": style.MUTED,
            "text.color": style.INK,
            "xtick.color": style.MUTED,
            "ytick.color": style.MUTED,
            "svg.fonttype": "none",
            "svg.hashsalt": "uccl-lite-reduction-ablation",
        }
    )
    fig, axes = style.figure(
        rows=2,
        title=(
            "No-P2P collectives · GPU progress ablation"
            if args.gpu_progress
            else "No-P2P collectives · reduction-placement ablation"
        ),
        subtitle=(
            "4 × NVIDIA L4  /  BF16  /  eager calls  /  fixed 8 MiB packets"
            if args.gpu_progress
            else "4 × NVIDIA L4  /  BF16  /  eager calls  /  default vs fixed 8 MiB variants"
        ),
    )
    for row, operation in enumerate(("allgather", "reducescatter")):
        records = {
            v: load(v, backend, operation, data) for v, _, backend, _ in variants
        }
        for col, exponents in enumerate((range(10, 21), range(21, 31))):
            sizes = [1 << n for n in exponents]
            series = []
            for variant, label, _, color in variants:
                values, ranges = records[variant]
                series.append(
                    (
                        label,
                        color,
                        [values[s] for s in sizes],
                        ([ranges[s][0] for s in sizes], [ranges[s][1] for s in sizes]),
                    )
                )
            upper = max(1.25, max(max(s[3][1]) for s in series) * 1.12)
            labels = [
                style.size_label(s)
                .replace(" KiB", "K")
                .replace(" MiB", "M")
                .replace(" GiB", "G")
                for s in sizes
            ]
            title = ("AllGather" if row == 0 else "ReduceScatter") + (
                " · 1 KiB–1 MiB" if col == 0 else " · 2 MiB–1 GiB"
            )
            style.bars(
                axes[row, col],
                labels,
                series,
                title,
                "Paired NCCL / candidate (×)",
                ylim=upper,
                speedup=True,
            )
            axes[row, col].set_xlabel("Full-array size")
    (style.ROOT / "docs/assets/experiments").mkdir(parents=True, exist_ok=True)
    (style.ROOT / "build/figures/experiments").mkdir(parents=True, exist_ok=True)
    style.save(
        fig,
        (
            "experiments/gpu-progress"
            if args.gpu_progress
            else "experiments/reduction-placement"
        ),
        [(label, color) for _, label, _, color in variants],
        "Dashed line: each configuration's paired NCCL 2.30.7 = 1×. Five-run means; whiskers: min–max paired run ratios, not confidence intervals.\n"
        + (
            "All 21 sizes retained. Same executable and GPU arithmetic; only the progress engine changes. GPU progress creates no packet-routing CPU workers.\n"
            if args.gpu_progress
            else "All 21 sizes retained. One executable; separate paired sweeps, no per-size best-variant selection. AllGather changes only with packet geometry.\n"
        )
        + "Includes eager host-submission gaps; no model-level performance claim.",
        args.preview,
    )


if __name__ == "__main__":
    main()
