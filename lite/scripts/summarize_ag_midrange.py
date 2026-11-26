#!/usr/bin/env python3
"""Validate paired AllGather trials and select the fastest non-router UCCL path."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def write_rows(path, columns, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("prefix", type=Path)
    args = parser.parse_args()
    timed = defaultdict(dict)
    host_means = {}
    reported = {}
    chosen = {}
    for line in args.log.read_text().splitlines():
        fields = line.split(",")
        if fields[0] in ("# sample", "# host_sample"):
            kind, op, size, backend, repeat, call, value = fields
            assert op == "all_gather"
            key = (int(size), backend, int(repeat), int(call))
            assert kind not in timed[key]
            timed[key][kind] = float(value)
        elif fields[0] == "# host_run_mean":
            _, op, size, backend, repeat, value = fields
            assert op == "all_gather"
            host_means[int(size), backend, int(repeat)] = float(value)
        elif fields[0] == "all_gather":
            _, size, backend, med, low, high = fields
            key = (int(size), backend)
            assert key not in reported
            reported[key] = (float(med), float(low), float(high))
        elif fields[0] == "# selected" and fields[1] == "all_gather":
            _, _, size, backend = fields
            assert int(size) not in chosen
            chosen[int(size)] = backend
    sizes = [1 << power for power in range(10, 31)]
    for size in sizes:
        backends = ["peer", "nccl", "cpu-router"]
        if size <= 32768:
            backends.append("pull")
        if size <= 2097152:
            backends.append("push-serial")
        for backend in backends:
            assert (size, backend) in reported
            means = []
            for repeat in range(1, 4):
                values = [timed[size, backend, repeat, call] for call in range(1, 11)]
                assert all(set(value) == {"# sample", "# host_sample"} for value in values)
                means.append(statistics.mean(value["# sample"] for value in values))
                host = statistics.mean(value["# host_sample"] for value in values)
                assert abs(host - host_means[size, backend, repeat]) < 0.002
            assert abs(statistics.median(means) - reported[size, backend][0]) < 0.002
        candidates = [backend for backend in backends if backend not in ("nccl", "cpu-router")]
        best = min(candidates, key=lambda backend: reported[size, backend][0])
        selected = chosen[size]
        assert selected == best or (selected.startswith("peer-") and best == "peer")
    assert len(chosen) == len(sizes)
    assert len(timed) == sum((3 + (size <= 32768) + (size <= 2097152)) * 30 for size in sizes)
    args.prefix.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.prefix.with_name(args.prefix.name + "_samples.csv"),
               ("bytes", "backend", "repeat", "call", "cuda_event_us", "host_clock_us"),
               ((*key, value["# sample"], value["# host_sample"])
                for key, value in sorted(timed.items())))
    write_rows(args.prefix.with_name(args.prefix.name + "_candidates.csv"),
               ("bytes", "backend", "cuda_event_median_us", "cuda_event_min_us",
                "cuda_event_max_us", "host_clock_median_us"),
               ((size, backend, *values,
                 statistics.median(host_means[size, backend, repeat] for repeat in range(1, 4)))
                for (size, backend), values in sorted(reported.items())))
    write_rows(args.prefix.with_name(args.prefix.name + "_selected.csv"),
               ("operation", "bytes", "uccl_path", "uccl_median_us", "nccl_median_us",
                "uccl_min_us", "uccl_max_us", "nccl_min_us", "nccl_max_us"),
               (("allgather", size, chosen[size], reported[size, chosen[size]][0],
                 reported[size, "nccl"][0], *reported[size, chosen[size]][1:],
                 *reported[size, "nccl"][1:]) for size in sizes))
    print(f"validated {len(sizes)} sizes and {len(timed)} timed calls")


if __name__ == "__main__":
    main()
