#!/usr/bin/env python3
"""Validate the paired NCCL LL/SIMPLE diagnosis and tuning selections."""

import argparse
import csv
import re
import statistics
from pathlib import Path


CASES = (
    ("allreduce", 524288),
    ("allgather", 524288),
    ("reducescatter", 524288),
    ("allreduce", 1048576),
)
TUNING = re.compile(
    r"NCCL INFO (AllReduce|AllGather|ReduceScatter): (\d+) Bytes -> "
    r"Algo (\w+) proto (\w+) channel\{Lo\.\.Hi\}=\{(\d+)\.\.(\d+)\}"
)
NAMES = {"AllReduce": "allreduce", "AllGather": "allgather",
         "ReduceScatter": "reducescatter"}


def write_csv(path, columns, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    samples = []
    summary = []
    for operation, size in CASES:
        for config in ("default", "simple"):
            run_means = []
            for repeat in (1, 2, 3):
                path = root / "paired" / f"{operation}_{size}_{config}_run{repeat}.log"
                content = path.read_text()
                assert "NCCL 22807" in content and "warmups=20 iterations=10" in content
                assert "every invocation fully checked" in content
                rows = list(csv.DictReader(line for line in content.splitlines()
                                           if not line.startswith("#")))
                assert len(rows) == 30
                timed = []
                for row in rows:
                    assert row["backend"] == "nccl" and int(row["bytes"]) == size
                    assert int(row["wrong"]) == 0
                    phase, call = row["phase"], int(row["iteration"])
                    assert (phase == "warmup" and 1 <= call <= 20 or
                            phase == "timed" and 1 <= call <= 10)
                    if phase == "timed":
                        latency = float(row["time_us"])
                        timed.append(latency)
                        samples.append((operation, size, config, repeat, call, latency))
                assert len(timed) == 10
                run_means.append(statistics.mean(timed))
            summary.append((operation, size, config,
                            round(min(run_means), 3),
                            round(statistics.median(run_means), 3),
                            round(max(run_means), 3)))
    write_csv(root / "paired_samples.csv",
              ("operation", "full_bytes_per_rank", "config", "repeat",
               "call", "host_latency_us"), samples)
    write_csv(root / "paired_summary.csv",
              ("operation", "full_bytes_per_rank", "config",
               "min_run_mean_us", "median_run_mean_us", "max_run_mean_us"),
              summary)

    selections = set()
    for path in sorted((root / "tuning").glob("*_tuning.log")):
        config = "simple" if path.name.endswith("_simple_tuning.log") else "default"
        content = path.read_text()
        assert "NCCL version 2.28.7" in content
        for name, logged_bytes, algorithm, protocol, lo, hi in TUNING.findall(content):
            operation = NAMES[name]
            full_bytes = int(logged_bytes) * (1 if operation == "allreduce" else 8)
            selections.add((operation, full_bytes, config, int(logged_bytes),
                            algorithm, protocol, int(lo), int(hi)))
    assert len(selections) == 21  # six default sizes per op; one SIMPLE size per op.
    write_csv(root / "tuning_selection.csv",
              ("operation", "full_bytes_per_rank", "config",
               "nccl_logged_bytes", "algorithm", "protocol",
               "channel_lo", "channel_hi"), sorted(selections))
    print(f"validated {len(samples)} timed samples and {len(selections)} tuning rows")


if __name__ == "__main__":
    main()
