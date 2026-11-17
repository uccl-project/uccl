#!/usr/bin/env python3
"""Validate per-call records and summarize fixed candidates (no size oracle)."""
import csv
from collections import defaultdict
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks/data/transfer_4to8_20261114"
REPORT = ROOT / "build/reports" / DATA.name


def summarize():
    points = defaultdict(list)
    count = 0
    for path in sorted(DATA.glob("*.log")):
        manifest = path.with_suffix(".json")
        if not manifest.exists():
            continue
        record = json.loads(manifest.read_text())
        if record["returncode"] or record["foreign_gpu_users"] or record["gpu_users_after"]:
            raise ValueError("invalid run: " + path.name)
        samples, reported = defaultdict(list), {}
        for line in path.read_text().splitlines():
            if not line.startswith(("sample,", "median,")):
                continue
            kind, size, variant, repeat, iteration, us = line.split(",")
            key = (int(size), variant, int(repeat))
            if kind == "sample":
                samples[key].append((int(iteration), float(us)))
                count += 1
            else:
                if key in reported:
                    raise ValueError("duplicate median")
                reported[key] = float(us)
        if not samples:
            continue
        cmd = record["command"]
        iterations, repeats = (int(cmd[cmd.index(flag) + 1]) for flag in ("--iterations", "--repeats"))
        begin, end = (int(cmd[cmd.index(flag) + 1]) for flag in ("--begin", "--end"))
        variants = cmd[cmd.index("--variants") + 1].split(",")
        sizes = []
        while begin <= end:
            sizes.append(begin)
            begin *= 2
        expected = {(size, v, r) for size in sizes for v in variants for r in range(repeats)}
        if set(samples) != expected or set(reported) != expected:
            raise ValueError("incomplete benchmark: " + path.name)
        grouped = defaultdict(list)
        for key, rows in samples.items():
            if sorted(i for i, _ in rows) != list(range(iterations)):
                raise ValueError("incomplete/duplicate samples")
            median = statistics.median(t for _, t in rows)
            if abs(median - reported[key]) > 2e-6:
                raise ValueError("median mismatch")
            grouped[key[:2]].append(median)
        experiment = path.stem
        if experiment.endswith(("_a", "_b")):
            experiment = experiment[:-2]
        scope = "host-global-completion" if "--host-timing" in cmd else "slowest-GPU-event"
        for (size, variant), medians in grouped.items():
            points[experiment, scope, size, variant].append(statistics.median(medians))
    rows = []
    for (experiment, scope, size, variant), values in sorted(points.items()):
        rows.append(dict(experiment=experiment, timing=scope, bytes=size, variant=variant,
                         sweeps=len(values), min_us=min(values), median_us=statistics.median(values), max_us=max(values)))
    with (DATA / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (DATA / "sample_count.json").write_text(json.dumps(dict(checked_timed_calls=count, summary_points=len(rows)), indent=2) + "\n")
    print(f"validated {count} checked timed calls; {len(rows)} points")
    write_tables(rows)
    return rows


def write_tables(rows):
    values = {(r["experiment"], r["bytes"], r["variant"]): r for r in rows}

    def get(experiment, size, variant):
        return values[experiment, size, variant]["median_us"]

    def complete(experiment, size, variants, sweeps=2):
        return all(values.get((experiment, size, v), {}).get("sweeps", 0) == sweeps for v in variants)

    def size_label(size):
        return f"{size // 1048576} MiB" if size >= 1048576 else f"{size // 1024} KiB"

    text = ["# Confirmed transfer results", "",
            "All times are microseconds. Each confirmation point is the median of two sweep medians; each sweep has five repetitions of twenty checked calls, with ten warmups per backend/repetition. Variants rotate and the second sweep reverses their order. No best-per-size curve is selected. See [method and limitations](../../../benchmarks/data/transfer_4to8_20261114/README.md).", "",
            "## Specialized AllGather: ordinary event timing", "",
            "Fixed 16-block mini-host geometry; out of place; complete output bytes/GPU. Positive reduction means faster than the current tuned router.", "",
            "| Size | Tuned router | Mini-host | NCCL | Reduction vs tuned router |",
            "| --- | ---: | ---: | ---: | ---: |"]
    for size in (131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216):
        if complete("confirm_ag", size, ["router-tuned", "mini", "nccl"]):
            old, new, nccl = (get("confirm_ag", size, v) for v in ("router-tuned", "mini", "nccl"))
            text.append(f"| {size_label(size)} | {old:.2f} | {new:.2f} | {nccl:.2f} | {100*(1-new/old):.1f}% |")
    text += ["", "## AllGather attribution: existing GPU progress at larger sizes", "",
             "A separate reversed paired experiment keeps GPU per-source progress enabled at every size, with either the regular 64-block cap or a 16-block cap. Mini-host remains fixed at 16 blocks. This isolates the new kernel from the existing progress option.", "",
             "| Size | Limited-progress tuned router | Full GPU progress | Full GPU progress, 16-block cap | Mini-host | NCCL |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for size in (131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216):
        variants = ["router-tuned", "router-full-ag", "router-full-ag-compact", "mini", "nccl"]
        if complete("ag_attribution", size, variants):
            times = " | ".join(f"{get('ag_attribution', size, v):.2f}" for v in variants)
            text.append(f"| {size_label(size)} | {times} |")
    text += ["", "## Prepared resident mode: matched host completion timing", "",
             "AR is in place; AG/RS are out of place. All inputs are ready before the timer. Persistent workers are stopped in every control window. Graph NCCL is a single eight-device fork/join graph. These values are separate from event-timed ordinary calls.", "",
             "| Operation | Size | Tuned router | Resident | NCCL graph | Eager NCCL | Tuned router / resident |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for op in ("ag", "rs", "ar"):
        for size in (1024, 8192, 16384, 32768, 65536):
            ex = "confirm_tiny_" + op
            if complete(ex, size, ["router-tuned", "resident", "nccl-graph", "nccl"]):
                tuned, resident, graph, nccl = (get(ex, size, v) for v in ("router-tuned", "resident", "nccl-graph", "nccl"))
                text.append(f"| {op.upper()} | {size_label(size)} | {tuned:.2f} | {resident:.2f} | {graph:.2f} | {nccl:.2f} | {tuned/resident:.2f}× |")
    text += ["", "## Existing options: ordinary event timing", "",
             "Base and every tuned variant use the same current source and common packet rules. The variants are defined in README.md. The one-CTA option applies only through 64 KiB; it is not a new path above that cutoff.", "",
             "| Operation | Size | Base | Progress/compact | + Parallel launch | + One CTA | NCCL | Reduction: base to parallel |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for op in ("ag", "rs", "ar"):
        for size in (1024, 65536, 1048576, 4194304):
            ex = "tuning_" + op
            variants = ["router-base", "router-progress", "router-tuned", "router-single", "nccl"]
            if complete(ex, size, variants):
                base, progress, tuned, single, nccl = (get(ex, size, v) for v in variants)
                text.append(f"| {op.upper()} | {size_label(size)} | {base:.2f} | {progress:.2f} | {tuned:.2f} | {single:.2f} | {nccl:.2f} | {100*(1-tuned/base):.1f}% |")
    text += ["", "## Sparse-call diagnostic: 5 ms host idle", "",
             "One diagnostic sweep, three repetitions of ten calls; not pooled with confirmation. The idle interval precedes input generation and is excluded from host completion timing.", "",
             "| Operation | Size | Progress | + Parallel launch | Resident | NCCL graph | Eager NCCL |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for op in ("ag", "rs", "ar"):
        for size in (1024, 65536):
            ex = "sparse_" + op
            variants = ["router-progress", "router-tuned", "resident", "nccl-graph", "nccl"]
            if complete(ex, size, variants, 1):
                times = " | ".join(f"{get(ex, size, v):.2f}" for v in variants)
                text.append(f"| {op.upper()} | {size_label(size)} | {times} |")
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "results.md").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    summarize()
