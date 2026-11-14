#!/usr/bin/env python3
"""Validate paired full RS sweeps; report every point and every regression."""
from collections import defaultdict
import argparse
import csv
import json
import math
from pathlib import Path
import statistics as stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks/data/rs_recovery_8l4_20261114"
SIZES = [1 << p for p in range(10, 31)]
VARIANTS = ("nccl", "previous", "current")


def summarize_controls():
    rows = []
    total = 0
    for mode in ("gpu", "shm"):
        for op in ("ar", "ag"):
            stem = f"control_{mode}_{op}"
            record = json.loads((DATA / "raw" / (stem + ".json")).read_text())
            assert record["returncode"] == 0 and not record["timed_out"]
            assert not record["foreign_gpu_users"] and not record["gpu_users_before"]
            assert not record["gpu_users_after"]
            assert record["binary_sha256"] == json.loads((DATA / "validation.json").read_text())["binary_sha256"]
            samples, medians, impls = defaultdict(list), {}, {}
            for line in (DATA / "raw" / (stem + ".log")).read_text().splitlines():
                if line.startswith("# implementation,"):
                    _, n, v, impl = line.split(",")
                    key = int(n), v
                    assert impls.setdefault(key, impl) == impl
                elif line.startswith(("sample,", "median,")):
                    kind, n, v, r, i, us = line.split(",")
                    key = int(n), v, int(r)
                    value = float(us)
                    assert math.isfinite(value) and value > 0
                    if kind == "sample":
                        samples[key].append((int(i), value))
                    else:
                        assert key not in medians
                        medians[key] = value
            expected = {(n, v, r) for n in SIZES for v in VARIANTS for r in range(3)}
            assert set(samples) == set(medians) == expected
            for key, calls in samples.items():
                assert sorted(i for i, _ in calls) == list(range(10))
                assert abs(stats.median(us for _, us in calls) - medians[key]) <= 2e-6
                total += len(calls)
            for n in SIZES:
                assert impls[n, "current"] == impls[n, "previous"]
                times = {v: stats.median(medians[n, v, r] for r in range(3)) for v in VARIANTS}
                rows.append(dict(testbed=mode, collective=op, bytes=n,
                                 implementation=impls[n, "current"],
                                 previous_us=times["previous"], current_us=times["current"],
                                 nccl_us=times["nccl"],
                                 change_pct=100 * (times["current"] / times["previous"] - 1)))
    assert total == 7560
    with (DATA / "controls.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    result = dict(checked_timed_calls=total, checked_warmups=3780,
                  identical_implementations_at_all_84_points=True,
                  single_sweep=True,
                  slowdowns_over_5pct=[r for r in rows if r["change_pct"] > 5])
    (DATA / "control_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    values, impls, hashes = {}, {}, set()
    total = 0
    for mode in ("gpu", "shm"):
        for sweep in ("a", "b"):
            stem = f"confirm_{mode}_rs_{sweep}"
            record = json.loads((DATA / "raw" / (stem + ".json")).read_text())
            assert record["returncode"] == 0 and not record["timed_out"]
            assert not record["foreign_gpu_users"] and not record["gpu_users_before"]
            assert not record["gpu_users_after"] and record["testbed"] == mode
            hashes.add(record["binary_sha256"])
            command = record["command"]
            for flag, expected in (("--begin", "1024"), ("--end", str(1 << 30)),
                                   ("--repeats", "3"), ("--iterations", "10"),
                                   ("--warmup", "5"), ("--blocks", "16"),
                                   ("--variants", ",".join(VARIANTS)), ("--op", "rs")):
                assert command[command.index(flag) + 1] == expected
            assert ("--reverse" in command) == (sweep == "b")
            assert ("--shm-only" in command) == (mode == "shm")
            if mode == "shm":
                assert record["environment"]["NCCL_P2P_DISABLE"] == "1"
                assert record["environment"]["NCCL_SHM_DISABLE"] == "0"
            samples, reported = defaultdict(list), {}
            for line in (DATA / "raw" / (stem + ".log")).read_text().splitlines():
                if line.startswith("# implementation,"):
                    _, size, label, impl = line.split(",")
                    key = mode, int(size), label
                    assert impls.setdefault(key, impl) == impl
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
            expected = {(n, v, r) for n in SIZES for v in VARIANTS for r in range(3)}
            assert set(samples) == set(reported) == expected
            for n in SIZES:
                for v in VARIANTS:
                    repeats = []
                    for r in range(3):
                        calls = samples[n, v, r]
                        assert sorted(i for i, _ in calls) == list(range(10))
                        median = stats.median(us for _, us in calls)
                        assert abs(median - reported[n, v, r]) <= 2e-6
                        repeats.append(median)
                        total += len(calls)
                    values[mode, n, v, sweep] = stats.median(repeats)
    assert len(hashes) == 1 and total == 7560
    rows, candidates, targets, collateral = [], [], [], []
    for mode in ("gpu", "shm"):
        for n in SIZES:
            medians = {}
            for v in VARIANTS:
                a, b = (values[mode, n, v, sweep] for sweep in ("a", "b"))
                medians[v] = stats.median((a, b))
                candidates.append(dict(testbed=mode, collective="reduce_scatter", bytes=n,
                                       variant=v, implementation=impls[mode, n, v],
                                       sweep_a_us=a, sweep_b_us=b, median_us=medians[v]))
            u, p, nccl = (medians[v] for v in ("current", "previous", "nccl"))
            rows.append(dict(testbed=mode, collective="reduce_scatter", bytes=n,
                             placement="out_of_place", uccl_median_us=u, nccl_median_us=nccl,
                             previous_median_us=p, speedup=nccl/u, reduction_vs_previous_pct=100*(1-u/p),
                             variant=impls[mode, n, "current"], status="fresh-rs-recovery"))
            if 1 << 20 <= n <= 8 << 20:
                assert impls[mode, n, "current"] == "router-ring"
                for sweep in ("a", "b"):
                    good = values[mode, n, "current", sweep] < values[mode, n, "nccl", sweep]
                    recovered = values[mode, n, "current", sweep] < values[mode, n, "previous", sweep]
                    targets.append(dict(testbed=mode, bytes=n, sweep=sweep,
                                        beats_nccl=good, beats_previous=recovered))
            else:
                assert impls[mode, n, "current"] == impls[mode, n, "previous"]
                if u > p * 1.05:
                    collateral.append(dict(testbed=mode, bytes=n, change_pct=100*(u/p-1)))
    for name, records in (("summary.csv", rows), ("candidates.csv", candidates)):
        with (DATA / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=records[0].keys(), lineterminator="\n")
            w.writeheader()
            w.writerows(records)
    validation = dict(timed_calls=total, warmup_calls=3780, figure_points=42,
                      binary_sha256=next(iter(hashes)), targets=targets,
                      all_target_sweeps_beat_nccl=all(r["beats_nccl"] for r in targets),
                      all_target_sweeps_beat_previous=all(r["beats_previous"] for r in targets),
                      unchanged_size_regressions_over_5pct=collateral)
    (DATA / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    lines = ["# Full-range ReduceScatter confirmation", "",
             "Hardware: wxz-l4, eight NVIDIA L4 GPUs, two NUMA nodes (ranks 0–3 and 4–7). CUDA 12.9, driver 580.178.04, NCCL 2.28.9. Previous means the fixed pre-recovery selection from 5b9a7f86, measured again in the same executable. GPU mode uses default NCCL P2P policy; SHM explicitly disables P2P. Both use mapped-host UCCL transport.", "",
             "BF16, out of place. Complete input bytes per GPU. Two opposite-order sweeps, each with three repetitions of five warmups and ten timed calls per size/backend. Median of the two sweep medians; each sweep median is the median of its three repetition medians. Slowest-GPU input-ready CUDA events include enqueue gaps. Every output is checked. No per-size winner selection: current fixes the ring at 1–8 MiB and retains previous algorithms elsewhere.", "",
             "| Mode | Complete bytes/GPU | Previous µs | Current µs | Paired NCCL µs | NCCL/current |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f'| {r["testbed"]} | {r["bytes"]} | {r["previous_median_us"]:.2f} | {r["uccl_median_us"]:.2f} | {r["nccl_median_us"]:.2f} | {r["speedup"]:.3f}× |')
    (DATA / "results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(validation, indent=2))
    assert validation["all_target_sweeps_beat_previous"], "recovery not confirmed against previous"
    assert validation["all_target_sweeps_beat_nccl"], "target NCCL regressions remain"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", action="store_true")
    args = parser.parse_args()
    main()
    if args.controls:
        summarize_controls()
