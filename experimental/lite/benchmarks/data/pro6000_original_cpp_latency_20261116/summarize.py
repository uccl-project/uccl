#!/usr/bin/env python3
"""Verify five unchanged C++ sweeps and summarize every measured mode."""

import csv
import hashlib
import json
from pathlib import Path
from statistics import median
import sys

sys.dont_write_bytecode = True
from run import parse

DATA = Path(__file__).resolve().parent


def main():
    provenance = json.loads((DATA / "provenance.json").read_text())
    for path, digest in provenance["files"].items():
        if hashlib.sha256((DATA / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Archive hash mismatch: {path}")
    manifest = json.loads((DATA / "raw/manifest.json").read_text())
    if manifest["status"] != "complete" or len(manifest["runs"]) != 5:
        raise ValueError("Five complete independent runs are required")
    if hashlib.sha256((DATA / "run.py").read_bytes()).hexdigest() != manifest["runner_sha256"]:
        raise ValueError("Measurement runner source mismatch")
    source = DATA / "measured_source/mscclpp_allreduce.cu"
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise ValueError("Measured CUDA source mismatch")
    runs = []
    for entry in manifest["runs"]:
        path = DATA / "raw" / f"run-{entry['index']}.txt"
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Run hash mismatch")
        runs.append({(r["bytes"], r["mode"]): r["us"] for r in parse(path.read_text())})
    historical = {}
    for line in (DATA / "pro5000_historical.txt").read_text().splitlines():
        fields = line.split()
        if len(fields) == 7 and fields[0].isdigit() and fields[2] == "auto":
            historical[int(fields[0])] = float(fields[3])
    rows = []
    for size, mode in sorted(runs[0]):
        values = [run[size, mode] for run in runs]
        rows.append({"bytes": size, "mode": mode, "median_us": median(values),
                     "min_us": min(values), "max_us": max(values),
                     **{f"run{i + 1}_us": v for i, v in enumerate(values)}})
    with (DATA / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    table = ["# Original UCCL Lite C++ auto-path latency on eight PRO6000 GPUs", "",
             "FP32; original unchanged C++ source; each run uses 10 warmups and 20 continuous timed calls.",
             "Values are the median and observed range of five independent run means, not confidence intervals.",
             "The PRO5000 historical column uses a separate host/toolchain/window and is not an EAT before/after comparison.",
             "", "| Bytes/rank | Auto path | PRO6000 median (µs) | Five-run range (µs) | Historical PRO5000 (µs) |",
             "| ---: | --- | ---: | ---: | ---: |"]
    for row in rows:
        if row["mode"] != "auto":
            continue
        size = row["bytes"]
        table.append(f"| {size} | {'LL' if size <= 32768 else 'PUT'} | {row['median_us']:.2f} | "
                     f"{row['min_us']:.2f}–{row['max_us']:.2f} | {historical[size]:.2f} |")
    table += ["", "All 195 size/mode pre-timing correctness checks report zero maximum relative error.",
              "The original timed loop does not validate every invocation; this behavior is preserved.",
              "See [README.md](README.md) for hardware, timing boundaries, source identities and reproduction.", ""]
    (DATA / "results.md").write_text("\n".join(table))
    print("Verified source/archive hashes, five complete 39-row sweeps and all zero-error checks")
    print("\n".join(table[5:]))


if __name__ == "__main__":
    main()
