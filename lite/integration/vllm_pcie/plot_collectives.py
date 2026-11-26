#!/usr/bin/env python3
"""Verify and plot the independently measured eight-PRO-6000 c03 AllReduce."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import sys

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data/pro6000_8gpu_allreduce_20261116"
sys.path.insert(0, str(ROOT / "scripts"))
from figure_style import (clean_axis, collective_legend, collective_panels,
                          configure, save)

SIZES = [1 << p for p in range(10, 31)]


def load_run(name, repeats, iterations):
    path = DATA / "raw" / f"{name}.jsonl"
    metadata = json.loads(Path(str(path) + ".metadata.json").read_text())
    if metadata["status"] != "complete":
        raise ValueError(f"Incomplete run: {name}")
    samples = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = (row["bytes"], row["repeat"], row["iteration"], row["rank"])
        if key in samples or not row["checked"] or not math.isfinite(row["us"]) or row["us"] <= 0:
            raise ValueError(f"Invalid or duplicate sample: {name}, {key}")
        samples[key] = row["us"]
    expected = {(size, r, i, rank) for size in SIZES for r in range(repeats)
                for i in range(iterations) for rank in range(8)}
    if set(samples) != expected:
        raise ValueError(f"Missing or extra samples: {name}")
    return {size: median(median(max(samples[size, r, i, rank] for rank in range(8))
                               for i in range(iterations)) for r in range(repeats))
            for size in SIZES}


def summarize():
    provenance = json.loads((DATA / "provenance.json").read_text())
    for path, digest in provenance["files"].items():
        if hashlib.sha256((DATA / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Archive hash mismatch: {path}")
    frozen_path = DATA / "raw/frozen-selection.json"
    selection = json.loads(frozen_path.read_text())
    plan = json.loads((DATA / "raw/validation-plan.json").read_text())
    if hashlib.sha256(frozen_path.read_bytes()).hexdigest() != plan["selection_sha256"]:
        raise ValueError("Validation does not match frozen calibration")
    calibration = {name: load_run(f"calibration-{name}-forward", 2, 3)
                   for name in selection["configs"] if name != "c03"}
    for name in calibration:
        raw = DATA / "raw" / f"calibration-{name}-forward.jsonl"
        if hashlib.sha256(raw.read_bytes()).hexdigest() != selection["calibration_hashes"][name]:
            raise ValueError("Calibration hash mismatch")
    for size in SIZES:
        for family, names in (("sm", ("default", "sm-p2p", "sm-128")),
                              ("ce", ("ce-2", "ce-8"))):
            if selection["schedule"][str(size)][family] != min(names, key=lambda n: calibration[n][size]):
                raise ValueError("Frozen policy is inconsistent with calibration")
    validation = {(name, order): load_run(f"validation-{name}-{order}", 3, 5)
                  for name in selection["configs"] for order in ("forward", "reverse")}
    for name in selection["configs"]:
        for order in ("forward", "reverse"):
            prefix = f"validation-{name}-{order}"
            metadata = json.loads((DATA / "raw" / f"{prefix}.jsonl.metadata.json").read_text())
            if metadata["harness_sha256"] != plan["harness_sha256"]:
                raise ValueError("Validation harness changed")
            if metadata["arguments"]["reverse"] != (order == "reverse"):
                raise ValueError("Incorrect size order")
            if name == "c03":
                continue
            logs = list((DATA / "raw").glob(f"{prefix}.nccl.*.log"))
            if len(logs) != 8:
                raise ValueError("Missing rank transport logs")
            for log in logs:
                channels = [line for line in log.read_text().splitlines() if "via " in line]
                if not channels or (name.startswith("ce-") and any("P2P/CUMEM/CE" not in line for line in channels)):
                    raise ValueError(f"Unexpected transport: {log}")
                if not name.startswith("ce-") and any("/CE" in line for line in channels):
                    raise ValueError(f"Unexpected CE transport: {log}")
    rows = []
    for size in SIZES:
        sm, ce = (selection["schedule"][str(size)][key] for key in ("sm", "ce"))
        row = {"bytes": size, "sm_config": sm, "ce_config": ce}
        for prefix, name in (("nccl", "default"), ("sm", sm), ("ce", ce), ("c03", "c03")):
            for order in ("forward", "reverse"):
                row[f"{prefix}_{order}_us"] = validation[name, order][size]
        rows.append(row)
    with (DATA / "cases.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("Verified archive, frozen selection, transport logs and 30,240 checked validation rank-invocations")
    return {row["bytes"]: row for row in rows}


def draw(data, preview):
    configure(black_axes=True)
    fig, axes = collective_panels()
    for ax, powers, ylabel, throughput in (
        (axes[0], range(10, 21), "Latency (µs)", False),
        (axes[1], range(20, 31), "Bus throughput (GB/s)", True),
    ):
        sizes = [1 << p for p in powers]
        clean_axis(ax)
        ax.set_xscale("log", base=2)
        ax.set_xlim(sizes[0] / 1.35, sizes[-1] * 1.35)
        ticks = sizes[::2]
        ax.set_xticks(ticks, [f"{n >> 10}K" if n < 1 << 20 else
                             f"{n >> 20}M" if n < 1 << 30 else "1G" for n in ticks])
        ax.set_xlabel("Complete tensor bytes per rank")
        ax.set_ylabel(ylabel)
        upper = 0.0
        for prefix, label, color, marker, linestyle in (
            ("nccl", "NCCL (default)", "#D8A17A", "s", (0, (5, 2.5))),
            ("sm", "NCCL SM (tuned)", "#82B4A3", "D", (0, (2, 1.5))),
            ("ce", "NCCL CE (tuned)", "#B899C8", "^", (0, (6, 2, 1, 2))),
            ("c03", "PCIeCCL (EAT)", "#0068B5", "o", "-"),
        ):
            runs = [[data[size][f"{prefix}_{order}_us"] for order in ("forward", "reverse")]
                    for size in sizes]
            values = [1.75 * size / median(us) / 1000 if throughput else median(us)
                      for size, us in zip(sizes, runs)]
            bounds = [[1.75 * size / us / 1000 if throughput else us for us in pair]
                      for size, pair in zip(sizes, runs)]
            low, high = [min(pair) for pair in bounds], [max(pair) for pair in bounds]
            upper = max(upper, *high)
            is_c03 = prefix == "c03"
            ax.fill_between(sizes, low, high, color=color,
                            alpha=0.12 if is_c03 else 0.08, linewidth=0)
            ax.plot(sizes, values, color=color, linewidth=3.6 if is_c03 else 2.2,
                    marker=marker, linestyle=linestyle, markersize=6.5 if is_c03 else 5,
                    markeredgecolor="white", markeredgewidth=0.7, label=label,
                    zorder=4 if is_c03 else 2)
        ax.set_ylim(0, upper * 1.13)
    collective_legend(fig, axes[0])
    save(fig, "collectives-pro6000x8-allreduce-overview", preview)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    draw(summarize(), args.preview)


if __name__ == "__main__":
    main()
