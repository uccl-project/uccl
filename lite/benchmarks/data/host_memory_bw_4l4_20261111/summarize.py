#!/usr/bin/env python3
"""Validate the NUMA-local STREAM triad logs and regenerate CSV summaries."""

import csv
import json
import math
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent
ELEMENTS = 1 << 26
BYTES_PER_PASS = 3 * ELEMENTS * 8


def write_rows(name, rows):
    with (ROOT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_log(name, threads, concurrent):
    lines = (ROOT / f"{name}.log").read_text().splitlines()
    header = f"# elements={ELEMENTS} bytes_per_array={ELEMENTS * 8} threads={threads}"
    if header not in lines or ("READY" in lines) != concurrent:
        raise ValueError(f"{name}: wrong benchmark setup")
    samples = {}
    for line in lines:
        if not line.startswith("sample,"):
            continue
        _, call, seconds, gbps = line.split(",")
        call, seconds, gbps = int(call), float(seconds), float(gbps)
        if (call in samples or not 0 <= call < 10 or
                not math.isfinite(seconds) or seconds <= 0 or
                abs(gbps - BYTES_PER_PASS / seconds / 1e9) > 0.005):
            raise ValueError(f"{name}: invalid sample {call}")
        samples[call] = gbps
    if sorted(samples) != list(range(10)):
        raise ValueError(f"{name}: missing samples")
    reported = [line for line in lines if line.startswith("summary,")]
    if len(reported) != 1:
        raise ValueError(f"{name}: missing summary")
    _, low, mid, high, label, checksum = reported[0].split(",")
    if label != "checksum" or float(checksum) != (ELEMENTS // 4096) * 7:
        raise ValueError(f"{name}: wrong computation")
    values = sorted(samples.values())
    actual = (values[0], median(values), values[-1])
    if any(abs(a - float(b)) > 0.001 for a, b in zip(actual, (low, mid, high))):
        raise ValueError(f"{name}: summary mismatch")
    return actual[1]


def main():
    records = []
    for mode in ("isolated", "concurrent"):
        commands = json.loads((ROOT / f"commands_{mode}.json").read_text())
        expected = 12
        if len(commands) != expected or any(c["exit_code"] for c in commands):
            raise ValueError(f"{mode}: incomplete commands")
        for node in (0, 1):
            for threads in (12, 24):
                values = []
                for run in (1, 2, 3):
                    name = f"{mode}_node{node}_threads{threads}_run{run}"
                    if not any(c["case"] == name for c in commands):
                        raise ValueError(f"{mode}: missing command {name}")
                    values.append(read_log(name, threads, mode == "concurrent"))
                records.append(dict(mode=mode, numa_node=node, threads_per_node=threads,
                                    working_set_bytes=3 * ELEMENTS * 8,
                                    min_run_median_gbs=min(values),
                                    median_run_median_gbs=median(values),
                                    max_run_median_gbs=max(values)))
    write_rows("summary.csv", records)
    aggregate = []
    for threads in (12, 24):
        for run in (1, 2, 3):
            node0 = read_log(f"concurrent_node0_threads{threads}_run{run}", threads, True)
            node1 = read_log(f"concurrent_node1_threads{threads}_run{run}", threads, True)
            aggregate.append(dict(threads_per_node=threads, run=run,
                                  node0_gbs=node0, node1_gbs=node1,
                                  sum_gbs=node0 + node1))
    write_rows("concurrent_aggregate.csv", aggregate)
    print("validated 24 runs, 240 timed passes, and both NUMA checksum results")


if __name__ == "__main__":
    main()
