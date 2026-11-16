#!/usr/bin/env python3
"""Replot the archived eight-PRO-6000 serving experiment without selecting runs."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data/pro6000_8gpu_20261115"
sys.path.insert(0, str(ROOT / "scripts"))
from figure_style import clean_axis, configure, metric_direction, save


def load_records():
    provenance = json.loads((DATA / "provenance.json").read_text())
    for name, record in provenance["files"].items():
        content = (DATA / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError(f"Archive hash mismatch: {name}")
        if "source_sha256" in record:
            if hashlib.sha256(gzip.decompress(content)).hexdigest() != record["source_sha256"]:
                raise ValueError(f"Decompressed source hash mismatch: {name}")
    with (DATA / "results/final/comparison.csv").open() as stream:
        final = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    with (DATA / "forced-p2p/comparison.csv").open() as stream:
        followup = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    sources = (
        ("results/final-validation/nccl", final, "nccl"),
        ("results/final-original/pcieccl", final, "original"),
        ("results/final-validation/pcieccl", final, "best"),
        ("forced-p2p/default", followup, "nccl_default_fresh"),
        ("forced-p2p/p2p", followup, "nccl_p2p"),
    )
    for folder, rows, prefix in sources:
        assert [int(r["concurrency"]) for r in rows] == [1, 2, 4, 8, 16, 32, 64]
        for row in rows:
            case = f"isl8192-osl1024-bs{int(row['concurrency'])}"
            raw = json.loads((DATA / folder / "cases" / f"{case}.json").read_text())
            metrics = raw["metrics"]
            assert raw["status"] == "complete" and metrics["request_errors"] == 0
            assert raw["case"]["warmup_seconds"] == 60
            assert metrics["measurement_seconds"] == 180
            expected = {
                "tok_s": metrics["measurement_output_tokens"] / 180,
                "tpot_ms": metrics["tpot_ms"]["p50"],
                "ttft_ms": metrics["ttft_ms"]["p50"],
            }
            for metric, value in expected.items():
                assert abs(row[f"{prefix}_{metric}"] - value) < 1e-8
    for prior, later in zip(final, followup):
        for before, after in (("nccl", "nccl_default"),
                              ("original", "pcieccl_original"),
                              ("best", "pcieccl_c03")):
            for metric in ("tok_s", "tpot_ms", "ttft_ms"):
                assert prior[f"{before}_{metric}"] == later[f"{after}_prior_{metric}"]
    print(f"Verified {len(provenance['files'])} archive hashes and 35 raw serving cases")
    return final, followup


def draw(rows, series, metric, name, subtitle, preview):
    configure(black_axes=True)
    plt.rcParams.update({"axes.labelsize": 13, "xtick.labelsize": 12,
                         "ytick.labelsize": 12})
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.68, bottom=0.24)
    fig.suptitle("MiniMax-M2.7 FP8 · 8×RTX PRO 6000", y=0.97,
                 fontsize=16, fontweight="bold")
    fig.text(0.5, 0.90, subtitle, ha="center", fontsize=12)
    fig.text(0.5, 0.84, "Experimental PCIeCCL · quality acceptance not met",
             ha="center", fontsize=11, color="#9B4A28")
    clean_axis(ax)
    positions = list(range(len(rows)))
    width = 0.20 if len(series) == 3 else 0.17
    for i, (prefix, label, color) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * (width + 0.025)
        values = [r[f"{prefix}_{metric}"] for r in rows]
        bars = ax.bar([x + offset for x in positions], values, width=width,
                      color=color, label=label)
        ax.bar_label(bars, fmt="%.0f" if metric == "tok_s" else "%.1f",
                     padding=3, fontsize=8, rotation=90)
    ax.set_xticks(positions, [str(int(r["concurrency"])) for r in rows])
    ax.set_xlabel("Concurrent requests · 8,192 input / 1,024 output tokens")
    ax.set_ylabel({"tok_s": "Output throughput (tokens/s)",
                   "tpot_ms": "Median time per output token (ms)",
                   "ttft_ms": "Median time to first token (ms)"}[metric])
    maximum = max(r[f"{s[0]}_{metric}"] for r in rows for s in series)
    ax.set_ylim(0, maximum * 1.28)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.03, 1.14), ncol=2,
              frameon=False, fontsize=10, borderaxespad=0)
    metric_direction(ax, higher=metric == "tok_s")
    fig.text(0.12, 0.07, "60 s warmup + 180 s window per case · one matrix per backend · no best-run selection",
             fontsize=10, color="#607486")
    save(fig, name, preview)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    final, followup = load_records()
    primary = (("nccl", "NCCL (default)", "#F6C99F"),
               ("original", "PCIeCCL (original)", "#8999A8"),
               ("best", "PCIeCCL (EAT c03)", "#0068B5"))
    for metric, suffix in (("tok_s", "throughput"), ("tpot_ms", "tpot"),
                           ("ttft_ms", "ttft")):
        draw(final, primary, metric, f"vllm-pcie-pro6000x8-{suffix}",
             "vLLM · TP8 + expert parallelism · final comparison", args.preview)
    transport = (("nccl_default_fresh", "NCCL (default, follow-up)", "#F6C99F"),
                 ("nccl_p2p", "NCCL (P2P forced, follow-up)", "#8064A6"),
                 ("pcieccl_original_prior", "PCIeCCL (original, prior)", "#8999A8"),
                 ("pcieccl_c03_prior", "PCIeCCL (EAT c03, prior)", "#0068B5"))
    draw(followup, transport, "tok_s", "vllm-pcie-pro6000x8-transport",
         "Transport follow-up · PCIeCCL measurements reused from final comparison",
         args.preview)


if __name__ == "__main__":
    main()
