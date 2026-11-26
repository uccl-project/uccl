#!/usr/bin/env python3
"""Validate paired four-L4 peer-copy, peer-read, and NCCL call samples."""

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

OPS = ("all_gather", "reduce_scatter")
BACKENDS = ("peer", "pull", "nccl")
SAMPLE = re.compile(
    r"^# sample,(all_gather|reduce_scatter),(\d+),(peer|pull|nccl),"
    r"(\d+),(\d+),([\d.]+)$", re.M)
SELECTED = re.compile(
    r"^# selected,(all_gather|reduce_scatter),(\d+),(peer|peer-compact|peer-noself|pull)$", re.M)


def direct_self_reduce(size: int) -> bool:
    return (size == (256 << 10) or
            (1 << 20) <= size <= (8 << 20) or size >= (32 << 20))


def summarize(source: Path, samples_path: Path, candidates_path: Path,
              summary_path: Path, begin: int, end: int) -> None:
    text = source.read_text()
    if "NCCL version=22807" not in text or "every warmup and timed output checked" not in text:
        raise ValueError("wrong peer/NCCL comparison identity")
    if "mismatch" in text or "peer AG/RS:" in text:
        raise ValueError("peer comparison failed correctness")
    match = re.search(r"^# direct peer-read path eligible through (\d+) bytes$",
                      text, re.M)
    if not match:
        raise ValueError("missing peer-read capacity")
    pull_max = int(match.group(1))
    match = re.search(
        r"^# compact peer completion through AG (\d+) B, RS (\d+) B$",
        text, re.M)
    if not match:
        raise ValueError("missing compact completion cutoffs")
    if "# RS direct self contribution at 256 KiB, 1-8 MiB, and >=32 MiB" not in text:
        raise ValueError("missing RS direct-self policy")
    compact_max = {"all_gather": int(match.group(1)),
                   "reduce_scatter": int(match.group(2))}
    sizes = []
    size = begin
    while size <= end:
        sizes.append(size)
        if size > end // 2:
            break
        size *= 2
    observed = defaultdict(dict)
    sample_rows = []
    for op, size, backend, repeat, iteration, latency in SAMPLE.findall(text):
        size, repeat, iteration = int(size), int(repeat), int(iteration)
        latency = float(latency)
        if size not in sizes or repeat not in (1, 2, 3) or not 1 <= iteration <= 10 or not math.isfinite(latency) or latency <= 0:
            raise ValueError("invalid per-call sample")
        key = (op, size, backend, repeat)
        if iteration in observed[key]:
            raise ValueError("duplicate per-call sample")
        observed[key][iteration] = latency
        sample_rows.append({"collective": op, "bytes": size,
                            "backend": backend, "repeat": repeat,
                            "iteration": iteration, "latency_us": latency})
    selected = {}
    for op, size, path in SELECTED.findall(text):
        key = op, int(size)
        if key in selected:
            raise ValueError("duplicate selected path")
        selected[key] = path
    reader = csv.DictReader(line for line in text.splitlines()
                            if not line.startswith("#"))
    reported = {}
    for row in reader:
        key = row["collective"], int(row["bytes"]), row["backend"]
        if key in reported or key[0] not in OPS or key[2] not in BACKENDS:
            raise ValueError("invalid candidate summary")
        reported[key] = float(row["median_us"])
    candidates = []
    summary = []
    for op in OPS:
        name = {"all_gather": "allgather",
                "reduce_scatter": "reducescatter"}[op]
        for size in sizes:
            means = {}
            expected = ("peer", "pull", "nccl") if size <= pull_max else ("peer", "nccl")
            for backend in expected:
                runs = []
                for repeat in (1, 2, 3):
                    calls = observed[(op, size, backend, repeat)]
                    if sorted(calls) != list(range(1, 11)):
                        raise ValueError(f"missing call samples: {op}, {size}, {backend}, {repeat}")
                    runs.append(mean(calls.values()))
                means[backend] = runs
                if abs(median(runs) - reported[(op, size, backend)]) > 0.02:
                    raise ValueError("candidate summary disagrees with call samples")
                candidate_path = (
                    "peer-noself" if backend == "peer" and
                    op == "reduce_scatter" and direct_self_reduce(size)
                    else "peer-compact" if backend == "peer" and
                    size <= compact_max[op] else backend)
                candidates.append({"collective": name, "bytes": size,
                                   "backend": backend, "path": candidate_path,
                                   "min_us": min(runs),
                                   "median_us": median(runs),
                                   "max_us": max(runs)})
            own = min((median(means[backend]), backend)
                      for backend in expected if backend != "nccl")
            path = ("peer-noself" if own[1] == "peer" and
                    op == "reduce_scatter" and direct_self_reduce(size)
                    else "peer-compact" if own[1] == "peer" and
                    size <= compact_max[op] else own[1])
            if selected.get((op, size)) != path:
                raise ValueError("selected path disagrees with measured medians")
            own_runs = means[own[1]]
            nccl_runs = means["nccl"]
            summary.append({
                "operation": name, "bytes": size, "uccl_path": path,
                "uccl_min_us": min(own_runs),
                "uccl_median_us": median(own_runs),
                "uccl_max_us": max(own_runs),
                "nccl_min_us": min(nccl_runs),
                "nccl_median_us": median(nccl_runs),
                "nccl_max_us": max(nccl_runs),
                "speedup": median(nccl_runs) / median(own_runs),
            })
    for destination, rows in ((samples_path, sample_rows),
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
    parser.add_argument("source", type=Path)
    parser.add_argument("samples", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--begin", type=int, default=1024)
    parser.add_argument("--end", type=int, default=1 << 30)
    args = parser.parse_args()
    summarize(args.source, args.samples, args.candidates, args.summary,
              args.begin, args.end)
