#!/usr/bin/env python3
"""Validate complete BF16 four-L4 router/NCCL samples and rebuild summaries."""

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent
OPS = ("all_reduce", "all_gather", "reduce_scatter")
FULL = tuple(1 << power for power in range(10, 31))
CASES = {
    "out_of_place": (OPS, FULL),
    "in_place": (OPS, FULL),
    "ar_compact": (("all_reduce",), (131072, 262144)),
    "rs_gpu_progress": (("reduce_scatter",), tuple(1 << p for p in range(16, 23))),
    "rs_striped": (("reduce_scatter",), tuple(1 << p for p in range(18, 23))),
    "ar_gpu_progress": (("all_reduce",), tuple(1 << p for p in range(17, 23))),
    "pilot_1k": (OPS, (1024,)),
}


def write_rows(name, rows):
    with (ROOT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_samples(name):
    with (ROOT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_case(name, operations, sizes):
    if "# correctness: PASS (every invocation, all ranks)" not in (ROOT / f"{name}.log").read_text():
        raise ValueError(f"{name}: output check did not pass")
    raw = read_samples(f"{name}_samples.csv")
    grouped = defaultdict(dict)
    for row in raw:
        op, size, backend = row["collective"], int(row["bytes"]), row["backend"]
        placement, repeat, iteration = (int(row[key]) for key in
                                        ("in_place", "repeat", "iteration"))
        latency = float(row["slowest_rank_us"])
        key = (op, size, backend, repeat)
        if (op not in operations or size not in sizes or
                placement != int(name == "in_place") or
                not 0 <= repeat < 3 or not 0 <= iteration < 10 or
                not math.isfinite(latency) or latency <= 0 or
                iteration in grouped[key]):
            raise ValueError(f"{name}: invalid sample {key}, {iteration}")
        grouped[key][iteration] = latency
    if len(raw) != len(operations) * len(sizes) * 2 * 3 * 10:
        raise ValueError(f"{name}: wrong timed call count {len(raw)}")
    result = []
    for op in operations:
        for size in sizes:
            backends = {backend for o, b, backend, _ in grouped if o == op and b == size}
            if len(backends) != 2 or "nccl" not in backends:
                raise ValueError(f"{name}: missing paired backend {op}, {size}")
            router = next(backend for backend in backends if backend != "nccl")
            stats = {}
            for backend in ("nccl", router):
                medians = []
                for repeat in range(3):
                    calls = grouped[op, size, backend, repeat]
                    if sorted(calls) != list(range(10)):
                        raise ValueError(f"{name}: missing timed calls {op}, {size}, {backend}")
                    medians.append(median(calls.values()))
                stats[backend] = (min(medians), median(medians), max(medians))
            control = stats["nccl"]
            candidate = stats[router]
            stable = control[1] / candidate[1] >= 1.03 and candidate[2] < control[0]
            for backend in ("nccl", router):
                low, middle, high = stats[backend]
                result.append(dict(case=name, collective=op, bytes=size,
                                   in_place=int(name == "in_place"), backend=backend,
                                   min_us=low, median_us=middle, max_us=high,
                                   nccl_over_backend=control[1] / middle,
                                   stable_router_gain=int(stable) if backend == router else 0))
    if len(grouped) != len(operations) * len(sizes) * 2 * 3:
        raise ValueError(f"{name}: unexpected sample groups")
    return result, len(raw)


def validate_auto(name, placement):
    if "# correctness: PASS (every invocation, all ranks)" not in (ROOT / f"{name}.log").read_text():
        raise ValueError(f"{name}: output check did not pass")
    raw = read_samples(f"{name}_samples.csv")
    if len(raw) != len(OPS) * len(FULL) * 3:
        raise ValueError(f"{name}: wrong replay sample count")
    grouped = defaultdict(list)
    for row in raw:
        op, size = row["collective"], int(row["bytes"])
        if (op not in OPS or size not in FULL or
                int(row["in_place"]) != placement or
                not math.isfinite(float(row["slowest_rank_us"])) or
                float(row["slowest_rank_us"]) <= 0):
            raise ValueError(f"{name}: invalid replay sample")
        grouped[op, size].append(row)
    result = []
    for op in OPS:
        for size in FULL:
            rows = grouped[op, size]
            if len(rows) != 3 or sorted(int(r["repeat"]) for r in rows) != [0, 1, 2]:
                raise ValueError(f"{name}: incomplete replay {op}, {size}")
            backends = {r["backend"] for r in rows}
            if len(backends) != 1:
                raise ValueError(f"{name}: replay changed backend {op}, {size}")
            backend = backends.pop()
            result.append(dict(case=name, collective=op, bytes=size,
                               in_place=placement, backend=backend,
                               cpu_router_selected=int(backend != "nccl")))
    return result


def main():
    summary, timed_calls = [], 0
    for name, (operations, sizes) in CASES.items():
        rows, calls = summarize_case(name, operations, sizes)
        summary.extend(rows)
        timed_calls += calls
    auto = validate_auto("out_auto", 0) + validate_auto("in_auto", 1)
    write_rows("summary.csv", summary)
    write_rows("auto_selection.csv", auto)
    print(f"validated {timed_calls} paired timed calls, "
          f"{len(summary)} summary rows, {len(auto)} auto-replay keys")


if __name__ == "__main__":
    main()
