#!/usr/bin/env python3
"""Validate fresh paired and router-only runs; plot every UCCL router size."""

import csv
import json
import math
from collections import defaultdict
from statistics import median

from figure_style import ROOT


DATA = ROOT / "benchmarks/data/cpu_router_4l4_figures_20261111"
OUTPUT = DATA / "figure_summary.csv"
CASES = (
    ("allreduce", "all_reduce", 1, "paired_allreduce_in", "allreduce_in", 5),
    ("allgather", "all_gather", 0, "paired_allgather_out", "allgather_out", 20),
    ("reducescatter", "reduce_scatter", 0, "paired_reducescatter_out",
     "reducescatter_out", 5),
)
SIZES = tuple(1 << power for power in range(10, 31))


def read_samples(name):
    with (DATA / f"{name}_samples.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def check_commands():
    paired = json.loads((DATA / "commands_paired.json").read_text())
    router_only = json.loads((DATA / "commands.json").read_text())
    if len(paired) != 3 or len(router_only) != 3:
        raise ValueError("missing benchmark commands")
    for group, flag in ((paired, "--calibrate"), (router_only, "--router-only")):
        for case in group:
            if case["exit_code"] or flag not in case["cmd"]:
                raise ValueError(f"invalid command {case['case']}")
    return {case["case"]: case for case in paired + router_only}


def verify_router_only(name, operation, placement):
    content = (DATA / f"{name}.log").read_text()
    if ("# correctness: PASS (every invocation, all ranks)" not in content or
            "uccl-lite-router-only:no-fallback-v1" not in content or
            "NCCL INFO" in content):
        raise ValueError(f"router-only replay failed: {name}")
    rows = read_samples(name)
    if len(rows) != len(SIZES) * 3 * 10:
        raise ValueError(f"incomplete router-only samples: {name}")
    grouped = defaultdict(dict)
    for row in rows:
        size = int(row["bytes"])
        repeat, call = int(row["repeat"]), int(row["iteration"])
        latency = float(row["slowest_rank_us"])
        key = (size, repeat)
        if (row["collective"] != operation or size not in SIZES or
                int(row["in_place"]) != placement or
                row["backend"] not in ("cpu-router", "gpu-progress-small-rs") or
                not 0 <= repeat < 3 or not 0 <= call < 10 or
                not math.isfinite(latency) or latency <= 0 or
                call in grouped[key]):
            raise ValueError(f"invalid router-only call: {name}, {key}, {call}")
        grouped[key][call] = latency
    for size in SIZES:
        for repeat in range(3):
            if sorted(grouped[size, repeat]) != list(range(10)):
                raise ValueError(f"missing router-only calls: {name}, {size}")


def paired_results(name, operation, placement):
    if "# correctness: PASS (every invocation, all ranks)" not in (DATA / f"{name}.log").read_text():
        raise ValueError(f"paired output check failed: {name}")
    rows = read_samples(name)
    if len(rows) != len(SIZES) * 2 * 3 * 10:
        raise ValueError(f"incomplete paired samples: {name}")
    grouped = defaultdict(dict)
    for row in rows:
        size, backend = int(row["bytes"]), row["backend"]
        repeat, call = int(row["repeat"]), int(row["iteration"])
        latency = float(row["slowest_rank_us"])
        key = (size, backend, repeat)
        if (row["collective"] != operation or size not in SIZES or
                int(row["in_place"]) != placement or
                not 0 <= repeat < 3 or not 0 <= call < 10 or
                not math.isfinite(latency) or latency <= 0 or
                call in grouped[key]):
            raise ValueError(f"invalid paired call: {name}, {key}, {call}")
        grouped[key][call] = latency
    result = []
    for size in SIZES:
        backends = {backend for b, backend, _ in grouped if b == size}
        if len(backends) != 2 or "nccl" not in backends:
            raise ValueError(f"missing paired NCCL control: {name}, {size}")
        router = next(backend for backend in backends if backend != "nccl")
        if router not in ("cpu-router", "gpu-progress-small-rs"):
            raise ValueError(f"non-router UCCL path: {router}")
        values = {}
        for label, backend in (("uccl", router), ("nccl", "nccl")):
            run_medians = []
            for repeat in range(3):
                calls = grouped[size, backend, repeat]
                if sorted(calls) != list(range(10)):
                    raise ValueError(f"missing paired calls: {name}, {size}")
                run_medians.append(median(calls.values()))
            for stat, value in zip(("min", "median", "max"),
                                   (min(run_medians), median(run_medians), max(run_medians))):
                values[f"{label}_{stat}_us"] = value
        result.append((size, router, values))
    if len(grouped) != len(SIZES) * 2 * 3:
        raise ValueError(f"extra paired sample groups: {name}")
    return result


def build():
    commands = check_commands()
    rows = []
    for name, operation, placement, paired, router_only, warmups in CASES:
        if paired not in commands or router_only not in commands:
            raise ValueError(f"missing {name} command")
        verify_router_only(router_only, operation, placement)
        for size, router, values in paired_results(paired, operation, placement):
            rows.append(dict(operation=name, bytes=size, dtype="bf16", gpus=4,
                             placement="in-place" if placement else "out-of-place",
                             uccl_path=router, **values,
                             nccl_version="2.28.7", nccl_config="default",
                             timing="slowest-gpu-cuda-event", warmups=warmups,
                             source=f"{paired}_samples.csv", source_case=paired,
                             router_only_check=f"{router_only}_samples.csv"))
    return rows


def main():
    rows = build()
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("validated 3,780 paired calls and 1,890 router-only calls; "
          "wrote 63 router-only UCCL figure points")


if __name__ == "__main__":
    main()
