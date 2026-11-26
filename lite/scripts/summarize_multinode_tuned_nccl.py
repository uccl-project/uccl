#!/usr/bin/env python3
"""Merge newly paired RING+SIMPLE spike rechecks with the full default sweep."""

import argparse
import csv
import math
import re
from pathlib import Path
from statistics import mean, median


CASES = {
    "allreduce": (262144, 524288, 1048576),
    "allgather": (262144, 524288),
    "reducescatter": (131072, 262144, 524288),
}
NAMES = {"allreduce": "AllReduce", "allgather": "AllGather",
         "reducescatter": "ReduceScatter"}
SAMPLE = re.compile(
    r"^# sample,(allreduce|allgather|reducescatter),(\d+),(\d+),"
    r"([\d.]+),([\d.eE+-]+)$", re.M)
SELECTED = re.compile(
    r"^# selected,(allreduce|allgather|reducescatter),(\d+),([^,\n]+)$",
    re.M)
TUNING = re.compile(
    r"NCCL INFO (AllReduce|AllGather|ReduceScatter): (\d+) Bytes -> "
    r"Algo (\w+) proto (\w+) channel\{Lo\.\.Hi\}=\{(\d+)\.\.(\d+)\}")


def write_rows(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_summary", type=Path)
    parser.add_argument("paired_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    with args.base_summary.open(newline="") as handle:
        base = list(csv.DictReader(handle))
    assert len(base) == 63
    by_key = {(row["collective"], int(row["bytes"])): row for row in base}
    assert len(by_key) == len(base)
    targeted = []
    samples = []
    for operation, sizes in CASES.items():
        for size in sizes:
            tuning = (args.output_directory / "tuning" /
                      f"{operation}_{size}.log").read_text()
            assert "NCCL version 2.28.7" in tuning
            selected_tuning = set(TUNING.findall(tuning))
            logged_size = size if operation == "allreduce" else size // 8
            assert selected_tuning == {
                (NAMES[operation], str(logged_size), "RING", "SIMPLE", "0", "1")}
            run_uccl, run_nccl, paths = [], [], set()
            directory = args.paired_directory / f"{operation}_{size}"
            for repeat in (1, 2, 3):
                own = (directory / f"{operation}_uccl_run{repeat}.txt").read_text()
                control = (directory / f"{operation}_nccl_run{repeat}.txt").read_text()
                assert "# timing=host enqueue+completion; slowest of eight ranks" in own
                assert "NCCL 22807" in control and "every invocation fully checked" in control
                assert "warmups=20 iterations=10" in control
                placement = "in-place" if operation == "allreduce" else "out-of-place"
                assert NAMES[operation] in control and "float32" in control
                assert placement in control and "mismatch" not in own
                selected = SELECTED.findall(own)
                assert len(selected) == 1 and selected[0][:2] == (operation, str(size))
                paths.add(selected[0][2])
                uccl_calls = {}
                for op, bytes_, call, latency, error in SAMPLE.findall(own):
                    assert op == operation and int(bytes_) == size
                    value, err = float(latency), float(error)
                    assert math.isfinite(value) and value > 0 and math.isfinite(err)
                    assert err <= 1e-5 and int(call) not in uccl_calls
                    uccl_calls[int(call)] = value
                    samples.append({"collective": operation, "bytes": size,
                                    "backend": "uccl", "repeat": repeat,
                                    "iteration": int(call), "latency_us": value,
                                    "error": err})
                assert sorted(uccl_calls) == list(range(1, 11))
                run_uccl.append(mean(uccl_calls.values()))
                nccl_calls = {}
                warmup_calls = set()
                reader = csv.DictReader(line for line in control.splitlines()
                                        if not line.startswith("#"))
                for row in reader:
                    assert row["backend"] == "nccl" and int(row["bytes"]) == size
                    assert int(row["wrong"]) == 0
                    if row["phase"] == "warmup":
                        warmup_calls.add(int(row["iteration"]))
                        continue
                    assert row["phase"] == "timed"
                    call, value = int(row["iteration"]), float(row["time_us"])
                    assert math.isfinite(value) and value > 0 and call not in nccl_calls
                    nccl_calls[call] = value
                    samples.append({"collective": operation, "bytes": size,
                                    "backend": "nccl", "repeat": repeat,
                                    "iteration": call, "latency_us": value,
                                    "error": 0})
                assert sorted(nccl_calls) == list(range(1, 11))
                assert warmup_calls == set(range(1, 21))
                run_nccl.append(mean(nccl_calls.values()))
            assert len(paths) == 1
            path = paths.pop()
            assert path == by_key[operation, size]["uccl_path"]
            row = {"collective": operation, "bytes": size, "uccl_path": path,
                   "uccl_min_us": min(run_uccl),
                   "uccl_median_us": median(run_uccl),
                   "uccl_max_us": max(run_uccl),
                   "nccl_min_us": min(run_nccl),
                   "nccl_median_us": median(run_nccl),
                   "nccl_max_us": max(run_nccl),
                   "speedup": median(run_nccl) / median(run_uccl),
                   "nccl_config": "RING+SIMPLE",
                   "source": f"paired_tuned/{operation}_{size}"}
            targeted.append(row)
    assert len(targeted) == 8 and len(samples) == 480
    args.output_directory.mkdir(parents=True, exist_ok=True)
    fields = list(base[0]) + ["nccl_config", "source"]
    write_rows(args.output_directory / "targeted_summary.csv", fields, targeted)
    write_rows(args.output_directory / "targeted_samples.csv", list(samples[0]), samples)
    for row in base:
        row["nccl_config"] = "default"
        row["source"] = "../multinode_serial_20261111/paired_w20"
    for row in targeted:
        by_key[row["collective"], int(row["bytes"])].update(row)
    write_rows(args.output_directory / "tuned_summary.csv", fields, base)
    print("validated eight 3-round paired rechecks and 480 timed samples")


if __name__ == "__main__":
    main()
