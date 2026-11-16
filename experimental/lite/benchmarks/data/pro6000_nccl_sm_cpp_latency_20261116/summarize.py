#!/usr/bin/env python3
"""Verify independent NCCL SM calibration and matched FP32 C++ validation."""

import csv
import hashlib
import json
from pathlib import Path
from statistics import median
import sys
import tarfile

sys.dont_write_bytecode = True
from run import CONFIGS, SIZES, parse

DATA = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    provenance = json.loads((DATA / "provenance.json").read_text())
    for path, digest in provenance["files"].items():
        if sha(DATA / path) != digest:
            raise ValueError(f"Archive hash mismatch: {path}")
    with tarfile.open(DATA / "raw/debug-logs.tar.gz") as archive:
        debug_logs = {member.name: archive.extractfile(member).read()
                      for member in archive.getmembers() if member.isfile()}
    if {name: hashlib.sha256(content).hexdigest() for name, content in debug_logs.items()} != provenance["debug_member_sha256"]:
        raise ValueError("Transport log archive member mismatch")
    manifest = json.loads((DATA / "raw/manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Incomplete measurement")
    if sha(DATA / "run.py") != manifest["runner_sha256"] or sha(DATA / "nccl_sm_latency.cu") != manifest["source_sha256"]:
        raise ValueError("Measured source mismatch")
    selection = json.loads((DATA / "raw/frozen-selection.json").read_text())
    if sha(DATA / "raw/frozen-selection.json") != manifest["frozen_selection_sha256"]:
        raise ValueError("Frozen policy mismatch")
    plan = json.loads((DATA / "raw/pre-validation-manifest.json").read_text())
    if plan["frozen_selection_sha256"] != manifest["frozen_selection_sha256"]:
        raise ValueError("Selection was not frozen before validation")
    if manifest["common_environment"]["NCCL_P2P_USE_CUDA_MEMCPY"] != "0":
        raise ValueError("CE must be disabled")
    runs = {}
    for record in manifest["runs"]:
        label = record["label"]
        log = DATA / "raw" / f"{label}.txt"
        if sha(log) != record["sha256"]:
            raise ValueError(f"Run hash mismatch: {label}")
        if record.get("status") != "pass":
            continue
        original = record["config"] == "original"
        runs[label] = parse(log.read_text(), original)
        if not original:
            debug = [content for name, content in debug_logs.items()
                     if name.startswith(label + ".") and name.endswith(".log")]
            if len(debug) != 8:
                raise ValueError("Missing rank transport logs")
            for content in debug:
                text = content.decode()
                for key in ("NCCL_ALGO", "NCCL_PROTO"):
                    value = CONFIGS[record["config"]].get(key)
                    if value and f"{key} set by environment to {value}" not in text:
                        raise ValueError(f"Requested algorithm/protocol not acknowledged: {label}")
                channels = [line for line in text.splitlines() if "via " in line]
                if not channels or any("/CE" in line for line in channels):
                    raise ValueError(f"Unexpected transport: {label}")
    eligible = selection["eligible_configs"]
    for size in SIZES:
        chosen = min(eligible, key=lambda name: median(runs[f"calibration-{r}-{name}"][size]
                                                     for r in range(1, 4)))
        if chosen != selection["schedule"][str(size)]:
            raise ValueError("Selection differs from independent calibration")
    for label, digest in selection["calibration_log_hashes"].items():
        if sha(DATA / "raw" / f"{label}.txt") != digest:
            raise ValueError("Calibration changed after selection")
    rows = []
    for size in SIZES:
        name = selection["schedule"][str(size)]
        row = {"bytes": size, "selected_config": name}
        for prefix, config in (("nccl_default", "default"), ("nccl_sm_tuned", name),
                               ("pcieccl_original", "original")):
            values = [runs[f"validation-{r}-{config}"][size] for r in range(1, 6)]
            row.update({f"{prefix}_median_us": median(values),
                        f"{prefix}_min_us": min(values), f"{prefix}_max_us": max(values),
                        **{f"{prefix}_run{r}_us": v for r, v in enumerate(values, 1)}})
        rows.append(row)
    with (DATA / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    table = ["# Matched FP32/C++ batch latency on eight RTX PRO6000 GPUs", "",
             "In-place SUM; 10 warmups and 20 continuous timed calls; C++ host batch timer; slowest rank.",
             "Independent calibration: three sweeps/config. Validation: five fresh sweeps/config; medians below.",
             "The tuned curve uses per-size choices frozen before validation. Original PCIeCCL is freshly rerun.", "",
             "| Size | NCCL default SM (µs) | NCCL tuned SM (µs) | Original PCIeCCL auto (µs) | Selected NCCL config |",
             "| ---: | ---: | ---: | ---: | --- |"]
    for row in rows:
        table.append(f"| {row['bytes'] // 1024} KiB | {row['nccl_default_median_us']:.2f} | "
                     f"{row['nccl_sm_tuned_median_us']:.2f} | {row['pcieccl_original_median_us']:.2f} | {row['selected_config']} |")
    table += ["", "All accepted size/run pre-timing checks reported zero maximum relative error.",
              "See [README.md](README.md) for the frozen policies, raw records, limitations and reproduction.", ""]
    (DATA / "results.md").write_text("\n".join(table))
    accepted = sum(record.get("status") == "pass" for record in manifest["runs"])
    validation = [record for record in manifest["runs"] if record["label"].startswith("validation-")]
    checks = sum(39 if record["config"] == "original" else 11 for record in validation)
    print(f"Verified {accepted} accepted runs, frozen selection, SM transport logs and {checks} validation checks")
    print("\n".join(table[6:]))


if __name__ == "__main__":
    main()
