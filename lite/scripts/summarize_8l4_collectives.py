#!/usr/bin/env python3
"""Select the measured eight-L4 UCCL Lite path for each BF16 collective."""

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks/data"
SMALL_SIZES = [1 << power for power in range(10, 20)]
LARGE_SIZES = [1 << power for power in range(20, 31)]


def rows(path: Path) -> dict[tuple[str, int, str], dict[str, str]]:
    text = path.read_text()
    if "# correctness: PASS" not in text:
        raise ValueError(f"missing output verification: {path}")
    result = {}
    for row in csv.DictReader(line for line in text.splitlines()
                              if not line.startswith("#")):
        operation = row["collective"]
        key = operation, int(row["bytes"]), row["backend"]
        if key in result:
            raise ValueError(f"duplicate measurement: {path}, {key}")
        timing = float(row["median_us"])
        if not math.isfinite(timing) or timing <= 0:
            raise ValueError(f"invalid timing: {path}, {key}")
        result[key] = row
    return result


def main() -> None:
    results = []
    small_dir = DATA / "collectives_small_8l4_20261110"
    for operation, placement, prefix in (("all_reduce", "in_place", "ar"),
                                         ("all_gather", "out_of_place", "out")):
        local = rows(small_dir / f"{prefix}_local.csv")
        nccl = rows(small_dir / f"{prefix}_nccl.csv")
        for size in SMALL_SIZES:
            variants = (
                ("local", float(local[operation, size, "fallback"]
                                ["median_us"])),
                ("cpu-router", float(local[operation, size, "cpu-router"]
                                     ["median_us"])),
            )
            path, candidate = min(variants, key=lambda item: item[1])
            if path == "cpu-router":
                candidate = float(nccl[operation, size, "cpu-router"]
                                  ["median_us"])
            control = float(nccl[operation, size, "fallback"]["median_us"])
            results.append((operation, size, placement, path, candidate,
                            control, control / candidate))
    local_dir = DATA / "local_fallback_8l4_20261110_v3"
    paired_dir = DATA / "router_vs_nccl_8l4_20261110"
    for operation, placement in (("all_reduce", "in_place"),
                                 ("all_gather", "out_of_place")):
        local = rows(local_dir / f"{placement}_calibration.csv")
        paired = rows(paired_dir / f"{placement}_calibration.csv")
        for size in LARGE_SIZES:
            key = operation, size
            router = float(local[operation, size, "cpu-router"]["median_us"])
            fallback = float(local[operation, size, "fallback"]["median_us"])
            if router >= fallback:
                raise ValueError(f"CPU router is not the fastest local path: {key}")
            candidate = float(paired[operation, size, "cpu-router"]["median_us"])
            control = float(paired[operation, size, "fallback"]["median_us"])
            results.append((operation, size, placement, "cpu-router", candidate,
                            control, control / candidate))
    rs_dir = DATA / "reducescatter_8l4_20261110_v2"
    rs_local = rows(rs_dir / "local.csv")
    rs_nccl = rows(rs_dir / "nccl.csv")
    for size in SMALL_SIZES + LARGE_SIZES:
        operation = "reduce_scatter"
        router_backend = ("gpu-progress-small-rs" if size == (128 << 10)
                          else "cpu-router")
        local_time = float(rs_local[operation, size, "fallback"]["median_us"])
        router_time = float(rs_local[operation, size, router_backend]["median_us"])
        if router_time < local_time:
            path = router_backend
            candidate = float(rs_nccl[operation, size, router_backend]["median_us"])
        else:
            path, candidate = "local", local_time
        control = float(rs_nccl[operation, size, "fallback"]["median_us"])
        results.append((operation, size, "out_of_place", path, candidate,
                        control, control / candidate))
    # Keep the original local/router choice evidence above, but use the fresh
    # paired NCCL control and tuned packet-router timing for AllGather.
    ag_path = DATA / "ag_packet_8l4_20261111/summary.csv"
    with ag_path.open() as stream:
        ag_rows = {int(row["bytes"]): row for row in csv.DictReader(stream)}
    if set(ag_rows) != set(SMALL_SIZES + LARGE_SIZES):
        raise ValueError("incomplete tuned eight-L4 AllGather summary")
    ar_path = DATA / "ar_compact_8l4_20261111/summary.csv"
    with ar_path.open() as stream:
        ar_rows = {int(row["bytes"]): row for row in csv.DictReader(stream)}
    if set(ar_rows) != {128 << 10, 256 << 10}:
        raise ValueError("incomplete compact eight-L4 AllReduce summary")
    updated = []
    for operation, size, placement, path, candidate, control, speedup in results:
        if operation == "all_gather":
            if path != "cpu-router":
                raise ValueError(f"unexpected selected AllGather path: {size}")
            row = ag_rows[size]
            candidate = float(row["uccl_median_us"])
            control = float(row["nccl_median_us"])
            speedup = control / candidate
        elif operation == "all_reduce" and size in ar_rows:
            row = ar_rows[size]
            candidate = float(row["uccl_median_us"])
            control = float(row["nccl_median_us"])
            if candidate >= float(row["local_median_us"]):
                raise ValueError(f"compact router did not beat local: {size}")
            path = "cpu-router"
            speedup = control / candidate
        updated.append((operation, size, placement, path, candidate,
                        control, speedup))
    results = updated
    order = {operation: index for index, operation in enumerate(
        ("all_reduce", "all_gather", "reduce_scatter"))}
    results.sort(key=lambda result: (order[result[0]], result[1]))
    output = DATA / "collectives_8l4_20261111.csv"
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("collective", "bytes", "placement", "uccl_path",
                         "uccl_median_us", "nccl_median_us", "speedup"))
        writer.writerows(results)
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
