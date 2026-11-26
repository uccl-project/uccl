#!/usr/bin/env python3
"""Validate paired four-L4 AllReduce samples and select the fastest path."""

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

SAMPLE = re.compile(r"^# sample,(\d+),([^,]+),(\d+),([\d.]+),([\d.eE+-]+)$", re.M)


def summarize(directory: Path, samples_path: Path, candidates_path: Path,
              summary_path: Path) -> None:
    own = defaultdict(dict)
    status = {}
    baseline = defaultdict(dict)
    samples = []
    for repeat in (1, 2, 3):
        own_log = (directory / f"allreduce_uccl_run{repeat}.log").read_text()
        nccl_log = (directory / f"allreduce_nccl_run{repeat}.log").read_text()
        if "maxrelerr" not in own_log:
            raise ValueError("missing UCCL correctness column")
        if any(token not in nccl_log for token in
               ("NCCL 22807", "AllReduce", "float32", "in-place",
                "every invocation fully checked")):
            raise ValueError("wrong NCCL control identity")
        for size, variant, iteration, latency, error in SAMPLE.findall(own_log):
            size, iteration = int(size), int(iteration)
            latency, error = float(latency), float(error)
            if not 1 <= iteration <= 10 or not math.isfinite(latency) or latency <= 0:
                raise ValueError("invalid UCCL sample")
            key = (size, variant, repeat)
            if iteration in own[key]:
                raise ValueError("duplicate UCCL sample")
            own[key][iteration] = (latency, error)
            samples.append({"operation": "allreduce", "bytes": size,
                            "backend": "uccl", "variant": variant,
                            "repeat": repeat, "iteration": iteration,
                            "latency_us": latency, "error": error})
        for line in own_log.splitlines():
            fields = line.split()
            if len(fields) != 7 or not fields[0].isdigit():
                continue
            key = (int(fields[0]), fields[2], repeat)
            if key in status:
                raise ValueError("duplicate UCCL summary")
            status[key] = float(fields[-1])
        reader = csv.DictReader(line for line in nccl_log.splitlines()
                                if not line.startswith("#"))
        for row in reader:
            if row["backend"] != "nccl" or int(row["wrong"]) != 0:
                raise ValueError("NCCL output mismatch")
            if row["phase"] != "timed":
                continue
            size, iteration = int(row["bytes"]), int(row["iteration"])
            latency = float(row["time_us"])
            if not 1 <= iteration <= 10 or not math.isfinite(latency) or latency <= 0:
                raise ValueError("invalid NCCL sample")
            key = (size, repeat)
            if iteration in baseline[key]:
                raise ValueError("duplicate NCCL sample")
            baseline[key][iteration] = latency
            samples.append({"operation": "allreduce", "bytes": size,
                            "backend": "nccl", "variant": "NCCL 2.28.7",
                            "repeat": repeat, "iteration": iteration,
                            "latency_us": latency, "error": 0})
    candidates = []
    summary = []
    for size in (1 << power for power in range(10, 31)):
        nccl_runs = []
        for repeat in (1, 2, 3):
            calls = baseline[(size, repeat)]
            if sorted(calls) != list(range(1, 11)):
                raise ValueError(f"missing NCCL calls: {size}, {repeat}")
            nccl_runs.append(mean(calls.values()))
        variants = sorted({key[1] for key in own if key[0] == size
                           and key[1] != "A0-null"})
        eligible = []
        for variant in variants:
            run_means = []
            correct = True
            for repeat in (1, 2, 3):
                calls = own[(size, variant, repeat)]
                if sorted(calls) != list(range(1, 11)):
                    raise ValueError(f"missing UCCL calls: {size}, {variant}, {repeat}")
                run_means.append(mean(value for value, _ in calls.values()))
                if any(not math.isfinite(error) or error > 1e-5
                       for _, error in calls.values()):
                    correct = False
                check = status.get((size, variant, repeat))
                if check is None or not math.isfinite(check) or check > 1e-5:
                    correct = False
            candidates.append({"operation": "allreduce", "bytes": size,
                               "variant": variant, "correct": int(correct),
                               "min_us": min(run_means),
                               "median_us": median(run_means),
                               "max_us": max(run_means)})
            if correct:
                eligible.append((median(run_means), variant, run_means))
        if not eligible:
            raise ValueError(f"no correct UCCL path: {size}")
        _, winner, own_runs = min(eligible)
        summary.append({"operation": "allreduce", "bytes": size,
                        "uccl_path": winner,
                        "uccl_min_us": min(own_runs),
                        "uccl_median_us": median(own_runs),
                        "uccl_max_us": max(own_runs),
                        "nccl_min_us": min(nccl_runs),
                        "nccl_median_us": median(nccl_runs),
                        "nccl_max_us": max(nccl_runs),
                        "speedup": median(nccl_runs) / median(own_runs)})
    for destination, rows in ((samples_path, samples),
                              (candidates_path, candidates),
                              (summary_path, summary)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys(),
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("samples", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    summarize(args.directory, args.samples, args.candidates, args.summary)
