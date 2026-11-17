#!/usr/bin/env python3
"""Validate complete fresh sweeps and produce all 126 GPU/SHM figure points."""
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks/data/full_refresh_8l4_20261114"
REPORT = ROOT / "build/reports" / DATA.name
OPS = {"ar": "all_reduce", "ag": "all_gather", "rs": "reduce_scatter"}
SIZES = [1 << power for power in range(10, 31)]
VARIANTS = ("nccl", "router-base", "current")


def main():
    medians, implementations = defaultdict(list), {}
    timed = warmup = 0
    for mode in ("gpu", "shm"):
        for op in OPS:
            for suffix in ("a", "b"):
                stem = f"full_{mode}_{op}_{suffix}"
                record = json.loads((DATA / "raw" / (stem + ".json")).read_text())
                assert record["returncode"] == 0 and not record["timed_out"]
                assert not record["foreign_gpu_users"] and not record["gpu_users_after"]
                assert record["testbed"] == mode
                command = record["command"]
                assert int(command[command.index("--begin")+1]) == SIZES[0]
                assert int(command[command.index("--end")+1]) == SIZES[-1]
                assert int(command[command.index("--repeats")+1]) == 3
                assert int(command[command.index("--iterations")+1]) == 10
                assert int(command[command.index("--warmup")+1]) == 5
                assert ("--shm-only" in command) == (mode == "shm")
                if mode == "shm":
                    assert record["environment"]["NCCL_P2P_DISABLE"] == "1"
                    assert record["environment"]["NCCL_SHM_DISABLE"] == "0"
                samples, reported = defaultdict(list), {}
                for line in (DATA / "raw" / (stem + ".log")).read_text().splitlines():
                    if line.startswith("# implementation,"):
                        _, size, label, implementation = line.split(",")
                        key = mode, op, int(size), label
                        assert implementations.setdefault(key, implementation) == implementation
                    elif line.startswith(("sample,", "median,")):
                        kind, size, variant, repeat, iteration, us = line.split(",")
                        key = int(size), variant, int(repeat)
                        value = float(us)
                        assert math.isfinite(value) and value > 0
                        if kind == "sample":
                            samples[key].append((int(iteration), value))
                        else:
                            assert key not in reported
                            reported[key] = value
                expected = {(size, variant, repeat) for size in SIZES for variant in VARIANTS for repeat in range(3)}
                assert set(samples) == set(reported) == expected, stem
                for size in SIZES:
                    for variant in VARIANTS:
                        repetitions = []
                        for repeat in range(3):
                            records = samples[size, variant, repeat]
                            assert sorted(i for i, _ in records) == list(range(10))
                            value = statistics.median(us for _, us in records)
                            assert abs(value - reported[size, variant, repeat]) <= 2e-6
                            repetitions.append(value)
                            timed += 10
                            warmup += 5
                        medians[mode, op, size, variant].append(statistics.median(repetitions))
    summary, candidates = [], []
    for mode in ("gpu", "shm"):
        for op, collective in OPS.items():
            for size in SIZES:
                values = {}
                for variant in VARIANTS:
                    a, b = medians[mode, op, size, variant]
                    values[variant] = statistics.median((a, b))
                    candidates.append(dict(testbed=mode, collective=collective, bytes=size,
                                           variant=variant, implementation=implementations[mode, op, size, variant],
                                           sweep_a_us=a, sweep_b_us=b, median_us=values[variant],
                                           min_us=min(a, b), max_us=max(a, b)))
                uccl, nccl, base = (values[v] for v in ("current", "nccl", "router-base"))
                summary.append(dict(testbed=mode, collective=collective, bytes=size,
                                    placement="in_place" if op == "ar" else "out_of_place",
                                    uccl_median_us=uccl, nccl_median_us=nccl, base_median_us=base,
                                    speedup=nccl/uccl, reduction_vs_base_pct=100*(1-uccl/base),
                                    variant=implementations[mode, op, size, "current"], status="fresh-full-sweep"))
    for name, rows in (("summary.csv", summary), ("candidates.csv", candidates)):
        with (DATA / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    assert len(summary) == 126 and timed == 22680 and warmup == 11340
    (DATA / "sample_count.json").write_text(json.dumps(dict(timed_calls=timed, warmup_calls=warmup,
                                                           figure_points=len(summary)), indent=2) + "\n")
    text = ["# Fresh full-range results", "",
            "BF16; AR in place, AG/RS out of place. Complete bytes per GPU. Every value uses two fresh reversed sweeps; each sweep has three repetitions, each with five warmups and ten timed calls per backend/size. Slowest-GPU CUDA events include enqueue gaps; all inputs are ready. No historical points or per-size best-variant selection.", ""]
    for mode in ("gpu", "shm"):
        text += ["## 8×L4 " + mode.upper(), "",
                 "Each cell is **UCCL / paired NCCL µs**. † marks a slower UCCL median.", "",
                 "| Complete bytes/GPU | AllReduce | AllGather | ReduceScatter |",
                 "| --- | ---: | ---: | ---: |"]
        for size in SIZES:
            label = f"{size >> 10} KiB" if size < 1 << 20 else f"{size >> 20} MiB" if size < 1 << 30 else "1 GiB"
            cells = []
            for collective in OPS.values():
                row = next(r for r in summary if r["testbed"] == mode and r["collective"] == collective and r["bytes"] == size)
                u, n = row["uccl_median_us"], row["nccl_median_us"]
                cells.append(f"{u:,.2f} / {n:,.2f}" + (" †" if u > n else ""))
            text.append("| " + " | ".join([label, *cells]) + " |")
        text.append("")
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "results.md").write_text("\n".join(text) + "\n")
    print(f"Validated {timed} timed calls and {warmup} warmups; {len(summary)} fresh points.")


if __name__ == "__main__":
    main()
