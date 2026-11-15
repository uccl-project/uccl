#!/usr/bin/env python3
"""Validate complete comparison runs and summarize every measured workload."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def compare(groups):
    if set(groups) != {"nccl", "packet"}:
        raise ValueError(f"missing backend: {list(groups)}")
    medians = {backend: statistics.median(values) for backend, values in groups.items()}
    return {
        "run_medians_s": dict(groups),
        "nccl_s": medians["nccl"], "packet_s": medians["packet"],
        "latency_reduction_pct": 100 * (1 - medians["packet"] / medians["nccl"]),
        "speedup": medians["nccl"] / medians["packet"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving", type=Path, nargs="+", default=[])
    parser.add_argument("--training", type=Path, nargs="+", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    serving = defaultdict(lambda: defaultdict(list))
    serving_rows = defaultdict(list)
    for directory in args.serving:
        manifest = json.loads((directory / "manifest.json").read_text())
        argv = manifest["argv"]
        planned = ["nccl", "packet", "packet", "nccl"]
        if "--backends" in argv:
            planned = []
            for value in argv[argv.index("--backends") + 1:]:
                if value not in ("nccl", "packet"):
                    break
                planned.append(value)
        if len(manifest["phases"]) != len(planned):
            raise ValueError(f"missing planned phases: {directory}")
        samples = rows(directory / "samples.jsonl")
        expected = set()
        for phase in manifest["phases"]:
            if not phase["complete"]:
                raise ValueError(f"unfinished phase: {directory}/{phase['label']}")
            command = phase["client"]
            backend = command[command.index("--backend") + 1]
            log = (directory / f"{phase['label']}_server.log").read_text()
            for operation in ("AG", "RS", "AR"):
                intercepted = f"UCCL Lite NCCL proxy: rank=0 op={operation} BF16" in log
                if intercepted != (backend == "packet"):
                    raise ValueError(f"unexpected interception: {directory}/{phase['label']}/{operation}")
            begin = command.index("--cases") + 1
            end = command.index("--repeats")
            repeats = int(command[end + 1])
            for case in command[begin:end]:
                selected = [r for r in samples if r["run_id"] == phase["label"] and r["case"] == case]
                if len(selected) != repeats or {r["repeat"] for r in selected} != set(range(repeats)):
                    raise ValueError(f"missing/duplicate samples: {directory}/{phase['label']}/{case}")
                expected.update((phase["label"], case, i) for i in range(repeats))
                if len({r["backend"] for r in selected}) != 1:
                    raise ValueError("backend changed within a phase")
                serving[case][selected[0]["backend"]].append(
                    statistics.median(r["client_batch_s"] for r in selected)
                )
                serving_rows[case].extend(selected)
        actual = {(r["run_id"], r["case"], r["repeat"]) for r in samples}
        if actual != expected or len(samples) != len(expected):
            raise ValueError(f"unexpected serving samples: {directory}")
    result = {"serving": {}, "training": {}}
    for case, groups in serving.items():
        selected = serving_rows[case]
        if len({r["prompt_sha256"] for r in selected}) != 1:
            raise ValueError(f"prompt mismatch: {case}")
        summary = compare(groups)
        summary["samples_per_backend"] = {
            b: sum(r["backend"] == b for r in selected) for b in groups
        }
        summary["exact_text_match_all_runs"] = len({r["output_sha256"] for r in selected}) == 1
        requests, inputs, _ = map(int, case.split(":"))
        summary["input_tokens_per_s"] = {
            b: requests * inputs / summary[f"{b}_s"] for b in groups
        }
        result["serving"][case] = summary
    training = defaultdict(list)
    for directory in args.training:
        manifest = json.loads((directory / "manifest.json").read_text())
        samples = rows(directory / "samples.jsonl")
        expected = {trial["label"] for trial in manifest["trials"]}
        if (len(expected) != manifest["planned_trials"]
                or not all(trial["complete"] for trial in manifest["trials"])
                or {r["run_id"] for r in samples} != expected or len(samples) != len(expected)):
            raise ValueError(f"incomplete training run: {directory}")
        for row in samples:
            if row["backend"] == "packet":
                log = (directory / f"{row['run_id']}.log").read_text()
                operations = ("AR",) if row["sharding"] == "ddp" else ("AG", "RS")
                if any(f"UCCL Lite NCCL proxy: rank=0 op={op} BF16" not in log for op in operations):
                    raise ValueError(f"missing training interception: {row['run_id']}")
            training[row["sharding"]].append(row)
    for sharding, selected in training.items():
        for key in ("width", "layers", "vocab", "sequence", "batch", "bucket_mib", "fsdp_rs_group",
                    "seed", "warmup", "steps", "parameters", "runtime_nccl_version"):
            if len({r[key] for r in selected}) != 1:
                raise ValueError(f"training configuration mismatch: {sharding}/{key}")
        groups = defaultdict(list)
        for row in selected:
            if not row["rank_parameter_samples_identical"]:
                raise ValueError("training parameter samples differ across ranks")
            groups[row["backend"]].append(row["median_step_s"])
        summary = compare(groups)
        reference = selected[0]
        if any(len(r["parameter_samples"]) != len(reference["parameter_samples"]) for r in selected):
            raise ValueError("parameter sample length changed")
        summary["parameter_sample_max_abs_difference"] = max(
            abs(a - b) for r in selected
            for a, b in zip(r["parameter_samples"], reference["parameter_samples"])
        )
        summary["loss_max_abs_difference"] = max(
            abs(a - b) for r in selected
            for actual, expected in zip(r["rank_losses"], reference["rank_losses"])
            for a, b in zip(actual, expected)
        )
        summary["parameters"] = reference["parameters"]
        summary["gradient_bytes"] = reference["gradient_bytes"]
        summary["timed_steps_per_run"] = reference["steps"]
        result["training"][sharding] = summary
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for category, cases in result.items():
        for case, summary in cases.items():
            print(f"{category} {case}: NCCL {summary['nccl_s']:.4f} s, "
                  f"packet {summary['packet_s']:.4f} s, "
                  f"{summary['latency_reduction_pct']:.2f}% lower, {summary['speedup']:.3f}x")


if __name__ == "__main__":
    main()
