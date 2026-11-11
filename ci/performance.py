"""Extract benchmark samples and compare explicitly reviewed hardware baselines."""

import argparse
import json
import math
from pathlib import Path
import re
import statistics

from catalog import read_catalog


def threshold(baseline):
    reference = baseline["value"]
    tolerance = baseline.get("tolerance", 0.1)
    if not math.isfinite(reference) or reference <= 0 or not 0 <= tolerance < 1:
        raise ValueError(
            "Baseline values must be finite and positive; tolerance must be in [0, 1)"
        )
    if baseline["direction"] == "higher":
        return reference * (1 - tolerance)
    if baseline["direction"] == "lower":
        # Performance is inverse latency: a 90% floor allows baseline / 0.9.
        return reference / (1 - tolerance)
    raise ValueError("Metric direction must be higher or lower")


def compare(value, baseline):
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Measurements must be finite and positive")
    limit = threshold(baseline)
    return value >= limit if baseline["direction"] == "higher" else value <= limit


def gate_policy(baseline):
    gate = baseline.get("gate", "hard")
    if gate not in {"hard", "soft"}:
        raise ValueError("Baseline gate must be hard or soft")
    reason = baseline.get("reason")
    if gate == "soft" and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError("Soft performance gates require a reason")
    return gate


def evaluate(results_dir, baselines):
    catalog, _ = read_catalog()
    definitions = {t["id"]: t for t in catalog["tests"]}
    for baseline in baselines.values():
        gate_policy(baseline)
        threshold(baseline)
    measurements, failures, warnings = [], [], []
    for path in sorted(results_dir.glob("*/results.json")):
        target = path.parent.name
        payload = json.loads(path.read_text())
        if payload.get("infrastructure_error"):
            failures.append(f"{target}: infrastructure failed")
        for name in {r["id"] for r in payload["results"] if r["tier"] == "performance"}:
            records = [
                r
                for r in payload["results"]
                if r["id"] == name and not r.get("not_participating")
            ]
            if any(r["exit_code"] for r in records):
                failures.append(f"{target}/{name}: benchmark failed")
                continue
            if any(r.get("skip_reason") for r in records):
                continue  # Busy-node skips provide no measurement or candidate.
            definition = definitions[name]
            for metric in definition.get("metrics", []):
                samples = []
                for log in path.parent.rglob(name + ".log"):
                    samples.extend(
                        float(m.group(1))
                        for m in re.finditer(
                            metric["pattern"], log.read_text(errors="replace")
                        )
                    )
                key = f"{target}/{name}/{metric['name']}"
                if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
                    failures.append(f"{key}: missing/nonfinite/nonpositive measurement")
                    continue
                value = statistics.median(samples)
                baseline = baselines.get(key)
                gate = gate_policy(baseline) if baseline else "unbaselined"
                status = (
                    "unbaselined"
                    if baseline is None
                    else ("pass" if compare(value, baseline) else "regression")
                )
                if status == "regression" and gate == "soft":
                    status = "warning"
                measurements.append(
                    {
                        "key": key,
                        "value": value,
                        "samples": samples,
                        "unit": metric["unit"],
                        "direction": metric["direction"],
                        "status": status,
                        "gate": gate,
                        "threshold": threshold(baseline) if baseline else None,
                    }
                )
                if status in {"regression", "warning"}:
                    message = (
                        f"{key}: {value} {metric['unit']} violates {gate} threshold "
                        f"{threshold(baseline):.6g} {metric['unit']}"
                    )
                    if status == "warning":
                        warnings.append(f"{message}; {baseline['reason']}")
                    else:
                        failures.append(message)
    return {"measurements": measurements, "failures": failures, "warnings": warnings}


def candidate_baselines(measurements, baselines):
    candidate = {}
    for metric in measurements:
        reviewed = baselines.get(metric["key"], {})
        entry = {
            "value": metric["value"],
            "direction": metric["direction"],
            "unit": metric["unit"],
            "tolerance": reviewed.get("tolerance", 0.1),
            "gate": gate_policy(reviewed) if reviewed else "soft",
        }
        if entry["gate"] == "soft":
            entry["reason"] = reviewed.get(
                "reason", "New metric; pending repeatable calibration and gate review."
            )
        candidate[metric["key"]] = entry
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path(__file__).parent / "config" / "baseline.json",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        help="Write a reviewable baseline candidate; never overwrite the active baseline implicitly",
    )
    args = parser.parse_args()
    baseline = json.loads(args.baselines.read_text())["baselines"]
    report = evaluate(args.results, baseline)
    (args.results / "performance.json").write_text(json.dumps(report, indent=2))
    if args.candidate:
        candidate = candidate_baselines(report["measurements"], baseline)
        args.candidate.write_text(
            json.dumps({"schema": 1, "baselines": candidate}, indent=2) + "\n"
        )
    print(json.dumps(report, indent=2))
    return int(bool(report["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
