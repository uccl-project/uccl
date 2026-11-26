#!/usr/bin/env python3
"""Validate paired eight-rank host-clock samples with full per-call checks."""

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

OPS = ("allreduce", "allgather", "reducescatter")
NAMES = {"allreduce": "AllReduce", "allgather": "AllGather",
         "reducescatter": "ReduceScatter"}
PATHS = {
    "allreduce": "local ring RS + four-rank IB exchange + local ring AG",
    "allgather": "local ring AG + remote-rank IB exchange",
    "reducescatter": "local direct ring RS + remote-rank IB exchange",
}
MARKERS = {
    "allreduce": "# local ring 0->1->3->2->0; AG ring",
    "allgather": "# local ring, remote-rank IB exchange",
    "reducescatter": "# local direct ring reduction, remote-rank IB exchange",
}
AUTO_MARKERS = {
    "allreduce": "AG adaptive flat/ring",
    "allgather": "# local adaptive ring/multicast",
    "reducescatter": "# local adaptive direct/packed/flat reduction",
}
SELECTED = re.compile(r"^# selected,(allreduce|allgather|reducescatter),(\d+),([^,\n]+)$", re.M)
OWN_SAMPLE = re.compile(r"^# sample,(allreduce|allgather|reducescatter),(\d+),(\d+),([\d.]+),([\d.eE+-]+)$", re.M)


def summarize(directory: Path, samples_path: Path, summary_path: Path,
              auto: bool = False, min_bytes: int = 1024,
              max_bytes: int = 1 << 30) -> None:
    if (min_bytes < 1024 or min_bytes > max_bytes or
            min_bytes & (min_bytes - 1) or max_bytes & (max_bytes - 1)):
        raise ValueError("size bounds must be powers of two from 1 KiB")
    own = defaultdict(dict)
    baseline = defaultdict(dict)
    selected_paths = {}
    samples = []
    for operation in OPS:
        for repeat in (1, 2, 3):
            own_log = (directory / f"{operation}_uccl_run{repeat}.txt").read_text()
            nccl_log = (directory / f"{operation}_nccl_run{repeat}.txt").read_text()
            if "# timing=host enqueue+completion; slowest of eight ranks" not in own_log:
                raise ValueError("wrong UCCL timing scope")
            if (AUTO_MARKERS if auto else MARKERS)[operation] not in own_log:
                raise ValueError("wrong UCCL path")
            if auto:
                for op, size, path in SELECTED.findall(own_log):
                    key = (op, int(size), repeat)
                    if key in selected_paths or op != operation:
                        raise ValueError("invalid or duplicate selected path")
                    selected_paths[key] = path
            if "NCCL 22807" not in nccl_log or "every invocation fully checked" not in nccl_log:
                raise ValueError("wrong NCCL control identity")
            placement = "in-place" if operation == "allreduce" else "out-of-place"
            if NAMES[operation] not in nccl_log or "float32" not in nccl_log or placement not in nccl_log:
                raise ValueError("wrong NCCL operation, dtype, or placement")
            if "mismatch" in own_log or "output mismatch" in nccl_log:
                raise ValueError("correctness failure")
            for op, size, iteration, latency, error in OWN_SAMPLE.findall(own_log):
                size, iteration = int(size), int(iteration)
                latency, error = float(latency), float(error)
                if op != operation or not 1 <= iteration <= 10 or not math.isfinite(latency) or latency <= 0 or not math.isfinite(error) or error > 1e-5:
                    raise ValueError("invalid UCCL call sample")
                key = (operation, size, repeat)
                if iteration in own[key]:
                    raise ValueError("duplicate UCCL sample")
                own[key][iteration] = latency
                samples.append({"collective": operation, "bytes": size,
                                "backend": "uccl", "repeat": repeat,
                                "iteration": iteration, "latency_us": latency,
                                "error": error})
            reader = csv.DictReader(line for line in nccl_log.splitlines()
                                    if not line.startswith("#"))
            for row in reader:
                if row["backend"] != "nccl" or int(row["wrong"]) != 0:
                    raise ValueError("NCCL output mismatch")
                if row["phase"] != "timed":
                    continue
                size, iteration = int(row["bytes"]), int(row["iteration"])
                latency, wrong = float(row["time_us"]), int(row["wrong"])
                if not 1 <= iteration <= 10 or wrong or not math.isfinite(latency) or latency <= 0:
                    raise ValueError("invalid NCCL call sample")
                key = (operation, size, repeat)
                if iteration in baseline[key]:
                    raise ValueError("duplicate NCCL sample")
                baseline[key][iteration] = latency
                samples.append({"collective": operation, "bytes": size,
                                "backend": "nccl", "repeat": repeat,
                                "iteration": iteration, "latency_us": latency,
                                "error": wrong})
    summary = []
    for operation in OPS:
        for size in (1 << power for power in range(min_bytes.bit_length() - 1,
                                                 max_bytes.bit_length())):
            own_runs, nccl_runs = [], []
            for repeat in (1, 2, 3):
                own_calls = own[(operation, size, repeat)]
                nccl_calls = baseline[(operation, size, repeat)]
                if sorted(own_calls) != list(range(1, 11)) or sorted(nccl_calls) != list(range(1, 11)):
                    raise ValueError(f"missing paired calls: {operation}, {size}, {repeat}")
                own_runs.append(mean(own_calls.values()))
                nccl_runs.append(mean(nccl_calls.values()))
            if auto:
                paths = {selected_paths.get((operation, size, repeat))
                         for repeat in (1, 2, 3)}
                if None in paths or len(paths) != 1:
                    raise ValueError(f"missing or inconsistent adaptive path: {operation}, {size}")
                path = paths.pop()
            else:
                path = PATHS[operation]
            summary.append({
                "collective": operation, "bytes": size,
                "uccl_path": path,
                "uccl_min_us": min(own_runs),
                "uccl_median_us": median(own_runs),
                "uccl_max_us": max(own_runs),
                "nccl_min_us": min(nccl_runs),
                "nccl_median_us": median(nccl_runs),
                "nccl_max_us": max(nccl_runs),
                "speedup": median(nccl_runs) / median(own_runs),
            })
    for destination, rows in ((samples_path, samples), (summary_path, summary)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("samples", type=Path)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--min-bytes", type=int, default=1024)
    parser.add_argument("--max-bytes", type=int, default=1 << 30)
    args = parser.parse_args()
    summarize(args.directory, args.samples, args.summary, args.auto,
              args.min_bytes, args.max_bytes)
