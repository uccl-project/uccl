#!/usr/bin/env python3
"""Verify packaged raw GPU rows, profiling identities, and derived diagnostics."""
import argparse
import json
from pathlib import Path

from analyze_pro6000_timeline import analyze, file_sha
from run_pro6000_allreduce import parse, placement_audit, transport


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input", type=Path, required=True, help="Extracted measurement archive"
    )
    a = p.parse_args()
    frozen = json.loads((a.input / "validation/frozen.json").read_text())
    folders = sorted((a.input / "profiles").glob("profile-final-*"))
    if {folder.name for folder in folders} != {
        "profile-final-staged",
        "profile-final-ring",
        "profile-final-host-ring",
        "profile-final-nccl",
    }:
        raise ValueError("Missing or unexpected diagnostic trace")
    streams = a.input / "profiles/stream-comparison"
    if streams.exists():
        cases = sorted(streams.iterdir())
        if {path.name for path in cases} != {"shared-64m", "separate-64m"}:
            raise ValueError("Missing stream-comparison diagnostic")
        folders += cases
    for folder in folders:
        projection = json.loads((folder / "gpu-rows.projection.json").read_text())
        rows = folder / "gpu-rows.sqlite"
        if file_sha(rows) != projection["output_sha256"]:
            raise ValueError("Raw GPU rows changed")
        record_path = next(folder.glob("timeline-*.json"))
        record = json.loads(record_path.read_text())
        log = folder / (record["label"] + ".txt")
        if record["status"] != "pass" or record["identities"] != frozen["identities"]:
            raise ValueError("Profiling source/binary identity mismatch")
        if (
            record["platform_sha256"] != frozen["platform_sha256"]
            or file_sha(log) != record["log_sha256"]
        ):
            raise ValueError("Profiling platform/log mismatch")
        if record["begin"] != record["end"]:
            raise ValueError("Expected a single-size diagnostic")
        parse(log.read_text(), [record["begin"]])
        if placement_audit(log.read_text(), 4) != record["placement_audit"]:
            raise ValueError("Profiling placement mismatch")
        status = json.loads((folder / "profile-status.json").read_text())
        if (
            not status["complete"]
            or projection["source_sha256"]
            != status["reports"][projection["source_name"]]
        ):
            raise ValueError("Missing or changed full trace export")
        for name, digest in status["reports"].items():
            path = folder / name
            if path.exists() and file_sha(path) != digest:
                raise ValueError("Changed full profiler report")
        if record["backend"] == "nccl":
            actual = transport(folder / "debug", record["label"], [record["begin"]])
            if json.loads(json.dumps(actual)) != record["transport"]:
                raise ValueError("Profiling NCCL transport mismatch")
        saved = json.loads((folder / "compact-analysis.json").read_text())
        actual = analyze(
            rows,
            saved["kernel"],
            saved["size"],
            saved["kernels_per_call"],
            saved["aggregate_window"],
        )
        if actual != saved or actual["diagnostics"]:
            raise ValueError("Incomplete trace or changed analysis")
        print("PASS", folder.name, "all ranks, calls and peer payload")


if __name__ == "__main__":
    main()
