#!/usr/bin/env python3
"""Validate and summarize the paired four-L4 CPU-router AllGather log."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    samples = defaultdict(dict)
    run_means = {}
    reported = {}
    for line in args.log.read_text().splitlines():
        fields = line.split(",")
        if fields[0] in ("# sample", "# host_sample"):
            kind, op, size, backend, repeat, call, value = fields
            assert op == "all_gather"
            key = (int(size), backend, int(repeat), int(call))
            assert kind not in samples[key]
            samples[key][kind] = float(value)
        elif fields[0] == "# host_run_mean":
            _, op, size, backend, repeat, value = fields
            assert op == "all_gather"
            run_means[int(size), backend, int(repeat)] = float(value)
        elif fields[0] == "all_gather":
            _, size, backend, median, _min, _max = fields
            reported[int(size), backend] = float(median)
    sizes = [1 << power for power in range(10, 31)]
    for size in sizes:
        backends = ("peer", "pull", "nccl", "cpu-router") if size <= 32768 else (
            "peer", "nccl", "cpu-router"
        )
        for backend in backends:
            assert (size, backend) in reported
            for repeat in range(1, 4):
                values = [samples[size, backend, repeat, call] for call in range(1, 11)]
                assert all(set(value) == {"# sample", "# host_sample"} for value in values)
                host_mean = statistics.mean(value["# host_sample"] for value in values)
                assert abs(host_mean - run_means[size, backend, repeat]) < 0.002
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "samples.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("bytes", "backend", "repeat", "call", "cuda_event_us", "host_clock_us"))
        for key in sorted(samples):
            row = samples[key]
            writer.writerow((*key, row["# sample"], row["# host_sample"]))
    with (args.output / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("bytes", "backend", "cuda_event_median_us", "host_clock_median_us",
                         "host_clock_min_us", "host_clock_max_us"))
        for size, backend in sorted(reported):
            host = [run_means[size, backend, repeat] for repeat in range(1, 4)]
            writer.writerow((size, backend, reported[size, backend], statistics.median(host),
                             min(host), max(host)))
    print(f"validated {len(sizes)} sizes, {len(samples)} timed calls")


if __name__ == "__main__":
    main()
