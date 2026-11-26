#!/usr/bin/env python3
"""Use qualified .so timings in the existing four-series eight-L4 figure style."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")

from figure_style import ROOT, configure, save


DATA = ROOT / "benchmarks/data/nccl_local8_strict_20261119"
OVERVIEW = ROOT / "benchmarks/data/overview_8l4_20261114"
TUNED = ROOT / "benchmarks/data/nccl_tuned_testbeds12_20261115"
OPERATIONS = [("ar", "all_reduce", "allreduce", "in", 1.75),
              ("ag", "all_gather", "allgather", "out", .875),
              ("rs", "reduce_scatter", "reducescatter", "out", .875)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true",
                        help="Also export PNG/PDF previews under build/figures/.")
    args = parser.parse_args()
    sources = [DATA / "comparison.csv", OVERVIEW / "summary.csv", TUNED / "summary.csv"]
    for path in sources:
        checksums = json.loads((path.parent / "checksums.json").read_text())
        if sha(path) != checksums[path.name]:
            raise ValueError(f"measurement checksum mismatch: {path}")
    with sources[0].open(newline="") as stream:
        records = list(csv.DictReader(stream))
    candidates = {(r["operation"], int(r["bytes"]), r["layout"]): r for r in records}
    expected = {(op, 1 << power, layout) for op, *_ in OPERATIONS
                for power in range(10, 31) for layout in ("out", "in")}
    if len(records) != len(expected) or set(candidates) != expected:
        raise ValueError("expected 126 unique qualified size/collective/layout points")
    for row in records:
        if not math.isfinite(float(row["candidate_us"])) or float(row["candidate_us"]) <= 0:
            raise ValueError("invalid qualified latency")

    # Reuse the original renderer, including colors, legends, bounds and axes.
    # Only the UCCL Lite timing field changes; all three NCCL controls are kept.
    spec = importlib.util.spec_from_file_location("uccl_lite_overview", OVERVIEW / "plot.py")
    overview = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overview)
    data = overview.load()
    configure(black_axes=True)
    exported, figures = [], []
    # The qualified UCCL path uses host packets in both logical testbeds.
    # Reuse its measurement explicitly; select NCCL controls by testbed.
    for mode in ("gpu", "shm"):
        for op, operation, filename, layout, factor in OPERATIONS:
            for power in range(10, 31):
                size = 1 << power
                data[mode, operation, size]["uccl_median_us"] = candidates[op, size, layout]["candidate_us"]
            figure = overview.make_figure(data, mode, operation, factor)
            prefix = "collectives-8l4-" + ("shm-" if mode == "shm" else "")
            name = f"{prefix}dropin-{filename}"
            save(figure, name, args.preview)
            figures.append(ROOT / "docs/assets" / (name + ".svg"))

            for panel, powers in [("latency", range(10, 21)), ("throughput", range(20, 31))]:
                for series, field, source in [
                    ("uccl", "uccl_median_us", sources[0]),
                    ("prior", "nccl_median_us", sources[1]),
                    ("sm", "nccl_sm_median_us", sources[2]),
                    ("ce", "nccl_ce_median_us", sources[2]),
                ]:
                    for power in powers:
                        size = 1 << power
                        row = data[mode, operation, size]
                        latency = float(row[field])
                        low, high = latency, latency
                        if series in ("sm", "ce"):
                            low = float(row[field.replace("median_us", "min_run_us")])
                            high = float(row[field.replace("median_us", "max_run_us")])
                        value = latency
                        if panel == "throughput":
                            value = factor * size / latency / 1000
                            low, high = factor * size / high / 1000, factor * size / low / 1000
                        exported.append(dict(testbed=mode, operation=op, panel=panel, series=series,
                                             layout=layout, bytes=size, latency_us=latency,
                                             plotted_value=value, lower_bound=low, upper_bound=high,
                                             source=str(source.relative_to(ROOT)),
                                             source_testbed="gpu" if series == "uccl" else mode))
    if len(exported) != 528:
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
        "overview_renderer_sha256": sha(OVERVIEW / "plot.py"),
        "style_script_sha256": sha(ROOT / "scripts/figure_style.py"),
        "figure_sha256": {str(p.relative_to(ROOT)): sha(p) for p in figures},
        "figure_inputs_sha256": sha(inputs),
        "hardware": "wxz-l4, eight NVIDIA L4s, four per NUMA node",
        "dtype": "BF16; SUM for AllReduce and ReduceScatter",
        "layouts": "AR in-place; AG/RS out-of-place, all series",
        "uccl_implementation": "qualified strict NCCL .so, labeled UCCL Lite",
        "uccl_curve_reuse": "Testbed 2 reuses the measured Testbed 1 curve on the same host/configuration: the qualified transport uses mapped host packets, no GPU P2P; this is not an independent SHM run",
        "nccl_testbeds": "gpu: default peer policy; shm: separately measured with P2P disabled",
        "uccl_timing": "input-ready maximum-rank per-call CUDA events; includes host submission and ABI staging/copies",
        "uccl_excluded": "preparation, input generation and verification",
        "uccl_warmups_per_repetition": 5,
        "uccl_timed_calls_per_repetition": 10,
        "uccl_repetitions_per_job": 3,
        "uccl_jobs_per_layout": 2,
        "uccl_aggregation": "median of repetition medians, then median of two job medians",
        "uccl_placement": "eight MPI processes, one GPU each",
        "nccl_controls": "unchanged historical default/CE/SM curves and CE/SM run bounds; NCCL 2.28.9",
        "nccl_placement": "one process, eight GPUs; separate measurement windows",
        "nccl_timing": "input-ready maximum-rank per-call CUDA events including host submission; two sweeps/runs, each with three repetitions of five warmups and ten timed calls",
        "nccl_selection": "default unchanged; CE/SM per-size calibrated settings frozen before independent validation",
        "throughput": "bus bandwidth: AR 1.75*N/t, AG/RS 0.875*N/t; decimal GB/s",
        "size_definition": "AG complete output; AR/RS complete input per rank",
        "uccl_best_per_size_envelope": False,
        "native_nccl_fallback": False,
        "plotted_points": len(exported),
    }
    (DATA / "readme_figure_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
