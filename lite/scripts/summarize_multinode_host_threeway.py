#!/usr/bin/env python3
"""Validate selected host-multicast/direct/NCCL three-way rechecks."""

import argparse
import csv
import math
import re
from pathlib import Path
from statistics import mean, median


CASES = (
    ("threeway_auto_1024", 1024, "host_auto"),
    ("threeway_auto_65536", 65536, "host_auto"),
    ("threeway_2m", 2097152, "host_put"),
    ("threeway_put_128k_2097152", 2097152, "host_put_128k"),
    ("threeway_put_128k_4194304", 4194304, "host_put_128k"),
)
DIRECT = re.compile(r"^# sample,allgather,(\d+),(\d+),([\d.]+),([\d.eE+-]+)$", re.M)
HOST = re.compile(r"^# sample,allgather,(\d+),host-rdma-multicast,1,(\d+),([\d.]+),(\d+)$", re.M)


def write_rows(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    summaries = []
    samples = []
    for folder, size, host_method in CASES:
        directory = args.root / folder
        for backend in ("direct", host_method, "nccl"):
            run_means = []
            file_backend = "host_auto" if host_method == "host_auto" else "host_put"
            for repeat in (1, 2, 3):
                path = directory / f"{file_backend if backend == host_method else backend}_run{repeat}.log"
                content = path.read_text()
                if "mismatch" in content:
                    raise ValueError(f"incorrect output: {path}")
                if backend == "direct":
                    assert "# timing=host enqueue+completion; slowest of eight ranks" in content
                    matches = DIRECT.findall(content)
                elif backend == "nccl":
                    assert "NCCL 22807" in content and "every invocation fully checked" in content
                    matches = []
                    reader = csv.DictReader(line for line in content.splitlines()
                                            if not line.startswith("#"))
                    for row in reader:
                        assert row["backend"] == "nccl" and int(row["wrong"]) == 0
                        if row["phase"] == "timed":
                            matches.append((row["bytes"], row["iteration"],
                                            row["time_us"], row["wrong"]))
                else:
                    assert "# timing=host enqueue+completion" in content
                    assert "warmups=20 iterations=10 repeats=1" in content
                    matches = HOST.findall(content)
                calls = {}
                for bytes_, call, latency, error in matches:
                    value, err = float(latency), float(error)
                    call = int(call)
                    assert (int(bytes_) == size and 1 <= call <= 10 and
                            math.isfinite(value) and value > 0 and
                            math.isfinite(err) and err <= 1e-5 and
                            call not in calls)
                    calls[call] = value
                    samples.append({"case": folder, "bytes": size,
                                    "backend": backend, "repeat": repeat,
                                    "call": call, "host_latency_us": value,
                                    "error": err})
                assert sorted(calls) == list(range(1, 11)), path
                run_means.append(mean(calls.values()))
            summaries.append({"case": folder, "bytes": size,
                              "backend": backend,
                              "min_run_mean_us": round(min(run_means), 3),
                              "median_run_mean_us": round(median(run_means), 3),
                              "max_run_mean_us": round(max(run_means), 3)})
    assert len(summaries) == 15 and len(samples) == 450
    write_rows(args.root / "threeway_summary.csv", list(summaries[0]), summaries)
    write_rows(args.root / "threeway_samples.csv", list(samples[0]), samples)
    print("validated five three-way cases and 450 timed calls")


if __name__ == "__main__":
    main()
