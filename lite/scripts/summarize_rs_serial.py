#!/usr/bin/env python3
"""Validate the four-L4 ReduceScatter sweep and selected UCCL path."""

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
    samples = defaultdict(dict)
    reported = {}
    chosen = {}
    for line in args.log.read_text().splitlines():
        fields = line.split(",")
        if fields[0] == "# sample" and fields[1] == "reduce_scatter":
            _, _, size, backend, repeat, call, value = fields
            key = int(size), backend, int(repeat), int(call)
            assert key not in samples
            samples[key] = float(value)
        elif fields[0] == "reduce_scatter":
            _, size, backend, med, low, high = fields
            key = int(size), backend
            assert key not in reported
            reported[key] = float(med), float(low), float(high)
        elif fields[0] == "# selected" and fields[1] == "reduce_scatter":
            _, _, size, path = fields
            assert int(size) not in chosen
            chosen[int(size)] = path
    sizes = [1 << power for power in range(10, 31)]
    for size in sizes:
        backends = ["peer", "nccl"]
        if size <= 32768:
            backends.append("pull")
        if size <= 8388608:
            backends.append("push-serial")
        if size <= 8192:
            backends.append("scatter-ll")
        if size <= 1048576:
            backends.append("scatter-fence")
        for backend in backends:
            assert (size, backend) in reported
            means = []
            for repeat in range(1, 4):
                values = [samples[size, backend, repeat, call]
                          for call in range(1, 11)]
                means.append(statistics.mean(values))
            assert abs(statistics.median(means) - reported[size, backend][0]) < 0.002
        best = min((backend for backend in backends if backend != "nccl"),
                   key=lambda backend: reported[size, backend][0])
        selected = chosen[size]
        assert selected == best or (selected.startswith("peer-") and best == "peer")
    assert len(chosen) == len(sizes)
    assert len(samples) == sum(
        (2 + (size <= 32768) + (size <= 8388608) +
         (size <= 8192) + (size <= 1048576)) * 30 for size in sizes)
    args.prefix.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.prefix.with_name(args.prefix.name + "_samples.csv"),
               ("bytes", "backend", "repeat", "call", "cuda_event_us"),
               ((*key, value) for key, value in sorted(samples.items())))
    write_rows(args.prefix.with_name(args.prefix.name + "_candidates.csv"),
               ("bytes", "backend", "cuda_event_median_us",
                "cuda_event_min_us", "cuda_event_max_us"),
               ((size, backend, *values)
                for (size, backend), values in sorted(reported.items())))
    write_rows(args.prefix.with_name(args.prefix.name + "_selected.csv"),
               ("operation", "bytes", "uccl_path", "uccl_median_us",
                "nccl_median_us", "uccl_min_us", "uccl_max_us",
                "nccl_min_us", "nccl_max_us"),
               (("reducescatter", size, chosen[size],
                 reported[size, chosen[size] if (size, chosen[size]) in reported else "peer"][0],
                 reported[size, "nccl"][0],
                 *reported[size, chosen[size] if (size, chosen[size]) in reported else "peer"][1:],
                 *reported[size, "nccl"][1:]) for size in sizes))
    print(f"validated {len(sizes)} sizes and {len(samples)} timed calls")


if __name__ == "__main__":
    main()
