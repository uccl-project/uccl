#!/usr/bin/env python3
"""Plot the qualified eight-L4 drop-in against matched native/original controls."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import AMBER, BASELINE, BLUE, MUTED, PURPLE, ROOT, TEAL
from figure_style import clean_axis, configure, save


DATA = ROOT / "benchmarks/data/nccl_local8_strict_20261119"
OPERATIONS = [
    ("ar", "all_reduce", "AllReduce", "allreduce"),
    ("ag", "all_gather", "AllGather", "allgather"),
    ("rs", "reduce_scatter", "ReduceScatter", "reducescatter"),
]
LAYOUTS = {"out": "out-of-place", "in": "in-place"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    sources = [DATA / "native_comparison.json", DATA / "comparison.csv"]
    checksums = json.loads((DATA / "checksums.json").read_text())
    for path in sources:
        if sha(path) != checksums[path.name]:
            raise ValueError(f"qualified data checksum mismatch: {path.name}")
    native_records = json.loads(sources[0].read_text())
    with sources[1].open(newline="") as stream:
        original_records = list(csv.DictReader(stream))
    names = {name: op for op, name, *_ in OPERATIONS}
    layouts = {name: key for key, name in LAYOUTS.items()}
    native = {(names[r["operation"]], int(r["bytes"]), layouts[r["layout"]]): r
              for r in native_records}
    original = {(r["operation"], int(r["bytes"]), r["layout"]): r
                for r in original_records}
    expected = {(op, 1 << power, layout) for op, *_ in OPERATIONS
                for power in range(10, 31) for layout in LAYOUTS}
    for records, rows, baseline in [(native_records, native, "native_us"),
                                    (original_records, original, "reference_us")]:
        if len(records) != len(expected) or set(rows) != expected:
            raise ValueError("expected 126 unique size/collective/layout points")
        for key, row in rows.items():
            candidate, reference = float(row["candidate_us"]), float(row[baseline])
            if not all(math.isfinite(us) and us > 0 for us in (candidate, reference)):
                raise ValueError(f"invalid timing: {key}")
            if not math.isclose(float(row["ratio"]), candidate / reference,
                                rel_tol=1e-12):
                raise ValueError(f"inconsistent latency ratio: {key}")
    for op, size, _ in expected:
        if original[op, size, "out"]["reference_us"] != original[op, size, "in"]["reference_us"]:
            raise ValueError("original baseline must use its recorded layout")
    return native, original, sources


def size_label(power):
    if power == 30:
        return "1G"
    return f"{1 << (power - 20)}M" if power >= 20 else f"{1 << (power - 10)}K"


def plot(rows, op, title, filename, comparison, preview, exported):
    series = [
        ("candidate_us", "out", "Drop-in · out-of-place", BLUE, "o", "-"),
        ("candidate_us", "in", "Drop-in · in-place", TEAL, "s", "--"),
    ]
    if comparison == "native":
        series += [
            ("native_us", "out", "Native NCCL · out-of-place", AMBER, "^", "-"),
            ("native_us", "in", "Native NCCL · in-place", PURPLE, "D", "--"),
        ]
        subtitle = "1 × 8 L4 · BF16 · nccl-tests batch timing"
        footnotes = ["NCCL 2.28.9 · 5 warmups / 20 timed iterations",
                     "Two runs/backend · fixed drop-in configuration"]
    else:
        layout = "in" if op == "ar" else "out"
        series.append(("reference_us", layout, f"Original UCCL · {LAYOUTS[layout]}",
                       BASELINE, "^", ":"))
        subtitle = "1 × 8 L4 · BF16 · matched per-call timing"
        footnotes = ["2 jobs × 3 reps × 10 timed calls · 5 warmups/rep",
                     "Original: 1 process · drop-in: 8 ranks"]

    fig, axes = plt.subplots(2, 1, figsize=(4.8, 6.9))
    fig.subplots_adjust(left=.19, right=.97, top=.71, bottom=.16, hspace=.70)
    fig.text(.5, .985, title, ha="center", va="top", fontsize=16, fontweight="bold")
    fig.text(.5, .941, subtitle, ha="center", fontsize=10)
    for ax, panel, powers, ticks in [
        (axes[0], "latency", range(10, 21), [10, 12, 14, 16, 18, 20]),
        (axes[1], "throughput", range(20, 31), [20, 22, 24, 26, 28, 30]),
    ]:
        clean_axis(ax)
        sizes = [1 << power for power in powers]
        ax.set_xscale("log", base=2)
        ax.set_xlim(sizes[0] / 1.15, sizes[-1] * 1.15)
        ax.set_xticks([1 << power for power in ticks], [size_label(p) for p in ticks])
        ax.tick_params(axis="both", labelsize=10)
        ax.set_xlabel("Complete buffer size (KiB / MiB / GiB)", fontsize=10)
        ax.set_ylabel("Latency (µs)" if panel == "latency"
                      else "Bus throughput (GB/s)", fontsize=11)
        ax.set_title("↓ Lower is better" if panel == "latency"
                     else "↑ Higher is better", loc="right", fontsize=10)
        upper = 0
        for field, layout, label, color, marker, line in series:
            latency = [float(rows[op, size, layout][field]) for size in sizes]
            # Match the multi-node figures and eight-rank nccl-tests bus
            # normalization using complete AG output or AR/RS input bytes.
            factor = 1.75 if op == "ar" else .875
            values = latency if panel == "latency" else [
                factor * size / (us * 1000) for size, us in zip(sizes, latency)]
            ax.plot(sizes, values, label=label, color=color, marker=marker,
                    linestyle=line, linewidth=2.2, markersize=4.5,
                    markeredgecolor="white", markeredgewidth=.5)
            upper = max(upper, *values)
            for size, us, value in zip(sizes, latency, values):
                exported.append({"comparison": comparison, "operation": op,
                                 "panel": panel, "series": field, "layout": layout,
                                 "bytes": size, "latency_us": us, "plotted_value": value})
        ax.set_ylim(0, upper * 1.12)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .92),
               ncol=1, frameon=False, fontsize=10, handlelength=2.4, labelspacing=.35)
    for y, line in zip([.065, .035], footnotes):
        fig.text(.5, y, line, ha="center", fontsize=8.5, color=MUTED)
    name = f"collectives-8l4-dropin-{comparison}-{filename}"
    save(fig, name, preview)
    return ROOT / "docs/assets" / (name + ".svg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true",
                        help="Also export PNG/PDF previews under build/figures/.")
    args = parser.parse_args()
    native, original, sources = load()
    configure(black_axes=True)
    exported, figures = [], []
    for comparison, rows in [("native", native), ("original", original)]:
        for op, _, title, filename in OPERATIONS:
            figures.append(plot(rows, op, title, filename, comparison,
                                args.preview, exported))
    if len(exported) != 462:
        raise ValueError("incomplete figure export")
    inputs = DATA / "readme_figure_inputs.csv"
    with inputs.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(exported)
    provenance = {
        "python_version": platform.python_version(),
        "matplotlib_version": matplotlib.__version__,
        "sources": {str(p.relative_to(ROOT)): sha(p) for p in sources},
        "plot_script_sha256": sha(Path(__file__)),
        "style_script_sha256": sha(ROOT / "scripts/figure_style.py"),
        "figure_sha256": {str(p.relative_to(ROOT)): sha(p) for p in figures},
        "figure_inputs_sha256": sha(inputs),
        "hardware": "wxz-l4, one host with eight NVIDIA L4s, four per NUMA node",
        "dtype": "BF16; SUM for AllReduce and ReduceScatter",
        "native_comparison": {
            "baseline": "NCCL 2.28.9, default algorithms/protocols/channels",
            "timing": "unchanged nccl-tests maximum-rank batch mean",
            "warmups": 5, "timed_iterations": 20, "runs_per_backend": 2,
            "order": ["native", "drop-in", "drop-in", "native"],
            "aggregation": "median of the two runs per backend/layout/size",
            "placement": "eight MPI ranks, one GPU per rank, both backends",
            "layouts": list(LAYOUTS.values()),
        },
        "original_comparison": {
            "baseline": "freshly rerun original README UCCL Lite binaries",
            "timing": "input-ready maximum-rank per-call CUDA events; includes host submission and ABI staging/copies",
            "excluded": "preparation, input generation and verification",
            "warmups_per_repetition": 5, "timed_calls_per_repetition": 10,
            "repetitions_per_job": 3, "jobs_per_backend_layout": 2,
            "aggregation": "median of repetition medians, then median of two job medians",
            "original_placement": "one process, eight GPUs; AR in-place, AG/RS out-of-place",
            "dropin_placement": "eight processes, one GPU each; both layouts",
        },
        "throughput": "bus bandwidth: AR 1.75*N/t, AG/RS 0.875*N/t; decimal GB/s",
        "size_definition": "AG complete output; AR/RS complete input per rank",
        "best_per_size_envelope": False,
        "native_nccl_fallback": False,
        "plotted_points": len(exported),
    }
    (DATA / "readme_figure_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
