#!/usr/bin/env python3
"""Validate the three-way small/medium AllGather comparison."""

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


DIRECT_SAMPLE = re.compile(
    r"^# sample,allgather,(\d+),(\d+),([\d.]+),([\d.eE+-]+)$", re.M)
HOST_SAMPLE = re.compile(
    r"^# sample,allgather,(\d+),host-rdma-multicast,1,(\d+),"
    r"([\d.]+),(\d+)$", re.M)
SELECTED = re.compile(r"^# selected,allgather,(\d+),([^,\n]+)$", re.M)


def write_rows(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    commands = (root / "commands.txt").read_text()
    assert "UCX_RNDV_SCHEME=put_zcopy" in commands
    assert "UCX_RNDV_THRESH=0" in commands
    sizes = [1 << power for power in range(10, 22)]
    samples = []
    runs = defaultdict(list)
    paths = defaultdict(set)
    for repeat in (1, 2, 3):
        for backend in ("direct", "host_auto", "host_put"):
            content = (root / f"{backend}_run{repeat}.log").read_text()
            assert "timing=host enqueue+completion" in content
            assert "mismatch" not in content
            if backend == "direct":
                assert "# local adaptive ring/multicast" in content
                for size, path in SELECTED.findall(content):
                    paths[int(size)].add(path)
                records = DIRECT_SAMPLE.findall(content)
            else:
                assert "warmups=20 iterations=10 repeats=1" in content
                assert "download=sm" in content
                records = HOST_SAMPLE.findall(content)
            by_size = defaultdict(dict)
            for size, call, latency, error in records:
                size, call, value = int(size), int(call), float(latency)
                err = float(error)
                assert (size in sizes and 1 <= call <= 10 and
                        math.isfinite(value) and value > 0 and
                        math.isfinite(err) and err <= 1e-5 and
                        call not in by_size[size])
                by_size[size][call] = value
                samples.append({"bytes": size, "backend": backend,
                                "repeat": repeat, "call": call,
                                "host_latency_us": value, "error": err})
            for size in sizes:
                assert sorted(by_size[size]) == list(range(1, 11))
                runs[size, backend].append(mean(by_size[size].values()))
    assert len(samples) == 1080
    assert all(paths[size] == {"serial-local"} for size in sizes)
    summary = []
    for size in sizes:
        row = {"operation": "allgather", "bytes": size,
               "direct_uccl_path": paths[size].pop()}
        for backend in ("direct", "host_auto", "host_put"):
            values = runs[size, backend]
            assert len(values) == 3
            row[backend + "_min_us"] = min(values)
            row[backend + "_median_us"] = median(values)
            row[backend + "_max_us"] = max(values)
        summary.append(row)
    write_rows(root / "samples.csv", list(samples[0]), samples)
    write_rows(root / "summary.csv", list(summary[0]), summary)
    print(f"validated {len(sizes)} sizes and {len(samples)} timed calls")


if __name__ == "__main__":
    main()
