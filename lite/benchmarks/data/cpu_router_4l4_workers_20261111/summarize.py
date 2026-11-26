#!/usr/bin/env python3
"""Validate the four-L4 worker-count sweep and reproduce its summaries."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent
OPS = ("all_reduce", "all_gather", "reduce_scatter")


def write_rows(name, rows):
    with (ROOT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def argument(cmd, flag, default=None):
    return cmd[cmd.index(flag) + 1] if flag in cmd else default


def read_samples(name):
    with (ROOT / f"{name}_samples.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def expected(command):
    cmd = command["cmd"]
    begin, end = int(argument(cmd, "--begin")), int(argument(cmd, "--end"))
    sizes = tuple(1 << power for power in range(begin.bit_length() - 1,
                                                 end.bit_length()))
    if sizes[0] != begin or sizes[-1] != end:
        raise ValueError(f"non-power-of-two range {command['case']}")
    op = argument(cmd, "--op")
    ops = OPS if op == "all" else (dict(allgather="all_gather",
                                          allreduce="all_reduce",
                                          reducescatter="reduce_scatter")[op],)
    return cmd, ops, sizes


def paired_case(command):
    name = command["case"]
    cmd, ops, sizes = expected(command)
    if "# correctness: PASS (every invocation, all ranks)" not in (ROOT / f"{name}.log").read_text():
        raise ValueError(f"{name}: output check failed")
    workers = int(argument(cmd, "--workers-per-node"))
    placement = int("--in-place" in cmd)
    if int(argument(cmd, "--iterations")) != 10 or int(argument(cmd, "--repeats")) != 3:
        raise ValueError(f"{name}: not a complete paired run")
    if int(argument(cmd, "--warmup")) < 5:
        raise ValueError(f"{name}: insufficient warmups")
    samples = read_samples(name)
    if len(samples) != len(ops) * len(sizes) * 2 * 3 * 10:
        raise ValueError(f"{name}: wrong sample count {len(samples)}")
    grouped = defaultdict(dict)
    for row in samples:
        op, size, backend = row["collective"], int(row["bytes"]), row["backend"]
        repeat, call = int(row["repeat"]), int(row["iteration"])
        latency = float(row["slowest_rank_us"])
        key = (op, size, backend, repeat)
        if (op not in ops or size not in sizes or
                int(row["in_place"]) != placement or
                not 0 <= repeat < 3 or not 0 <= call < 10 or
                not math.isfinite(latency) or latency <= 0 or
                call in grouped[key]):
            raise ValueError(f"{name}: bad timed call {key}, {call}")
        grouped[key][call] = latency
    result = []
    for op in ops:
        for size in sizes:
            backends = {backend for o, b, backend, _ in grouped if o == op and b == size}
            if len(backends) != 2 or "nccl" not in backends:
                raise ValueError(f"{name}: missing paired control {op}, {size}")
            router = next(backend for backend in backends if backend != "nccl")
            stats = {}
            for backend in ("nccl", router):
                values = []
                for repeat in range(3):
                    calls = grouped[op, size, backend, repeat]
                    if sorted(calls) != list(range(10)):
                        raise ValueError(f"{name}: missing calls {op}, {size}, {backend}")
                    values.append(median(calls.values()))
                stats[backend] = (min(values), median(values), max(values))
            control, candidate = stats["nccl"], stats[router]
            gain = control[1] / candidate[1]
            stable = gain >= 1.03 and candidate[2] < control[0]
            for backend in ("nccl", router):
                low, middle, high = stats[backend]
                result.append(dict(case=name, workers_per_node=workers,
                                   collective=op, bytes=size, in_place=placement,
                                   backend=backend, min_us=low, median_us=middle,
                                   max_us=high, nccl_over_backend=control[1] / middle,
                                   stable_router_gain=int(stable) if backend == router else 0,
                                   source=f"{name}_samples.csv"))
    if len(grouped) != len(ops) * len(sizes) * 2 * 3:
        raise ValueError(f"{name}: extra sample groups")
    return result, len(samples)


def auto_case(command):
    name = command["case"]
    cmd, ops, sizes = expected(command)
    if "# correctness: PASS (every invocation, all ranks)" not in (ROOT / f"{name}.log").read_text():
        raise ValueError(f"{name}: replay output check failed")
    placement = int("--in-place" in cmd)
    samples = read_samples(name)
    if len(samples) != len(ops) * len(sizes) * 3:
        raise ValueError(f"{name}: wrong replay sample count")
    grouped = defaultdict(dict)
    for row in samples:
        op, size = row["collective"], int(row["bytes"])
        repeat = int(row["repeat"])
        latency = float(row["slowest_rank_us"])
        if (op not in ops or size not in sizes or
                int(row["in_place"]) != placement or not 0 <= repeat < 3 or
                not math.isfinite(latency) or latency <= 0 or
                repeat in grouped[op, size]):
            raise ValueError(f"{name}: invalid replay sample")
        grouped[op, size][repeat] = row["backend"]
    result = []
    for op in ops:
        for size in sizes:
            calls = grouped[op, size]
            if sorted(calls) != [0, 1, 2] or len(set(calls.values())) != 1:
                raise ValueError(f"{name}: incomplete replay {op}, {size}")
            backend = calls[0]
            result.append(dict(case=name, collective=op, bytes=size,
                               in_place=placement, workers_per_node=8,
                               backend=backend,
                               cpu_router_selected=int(backend != "nccl")))
    return result


def main():
    commands = json.loads((ROOT / "commands.json").read_text())
    if len({c["case"] for c in commands}) != len(commands):
        raise ValueError("duplicate case")
    summary, auto, timed_calls = [], [], 0
    for command in commands:
        if "--profile" in command["cmd"]:
            auto.extend(auto_case(command))
        else:
            rows, count = paired_case(command)
            summary.extend(rows)
            timed_calls += count
    chosen = {(r["collective"], r["bytes"], r["in_place"]):
              r["stable_router_gain"] for r in summary
              if r["case"] in ("ag_only_w8_out", "ag_only_w8_in") and
              r["backend"] != "nccl"}
    for row in auto:
        expect = chosen.get((row["collective"], row["bytes"], row["in_place"]), 0)
        if int(row["cpu_router_selected"]) != int(expect):
            raise ValueError(f"unexpected auto decision {row}")
    write_rows("summary.csv", summary)
    write_rows("auto_selection.csv", auto)
    print(f"validated {timed_calls} paired calls, {len(summary)} summary rows, "
          f"{len(auto)} automatic decisions")


if __name__ == "__main__":
    main()
