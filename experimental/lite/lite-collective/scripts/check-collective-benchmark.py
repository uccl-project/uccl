#!/usr/bin/env python3

import argparse
import csv
import math
import re
import sys
from pathlib import Path


FALLBACK_PATTERNS = (
    re.compile(r"\bfallback to nccl/rccl\b", re.IGNORECASE),
    re.compile(r"\bNo FallBack implementation\b", re.IGNORECASE),
    re.compile(r"\brequiring NCCL fallback\b", re.IGNORECASE),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse lite-collective benchmark logs and compare mscclpp against NCCL."
    )
    parser.add_argument(
        "result_dir",
        type=Path,
        help="Benchmark output directory containing runs.csv.",
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=1.0,
        help="Required nccl_time / mscclpp_time ratio. Default: 1.0.",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Do not fail mscclpp rows that appear to use NCCL fallback.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        help="Output path for parsed per-size metrics. Default: <result_dir>/metrics.csv.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Output path for comparison summary. Default: <result_dir>/summary.csv.",
    )
    return parser.parse_args()


def parse_float(token):
    try:
        return float(token)
    except ValueError:
        return None


def parse_int_like(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_nccl_tests_metrics(log_path):
    rows = []
    if not log_path.exists():
        return rows

    for line in log_path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 9 or not parts[0].isdigit():
            continue

        size = int(parts[0])
        count = parse_int_like(parts[1])

        time_us = parse_float(parts[5])
        algbw = parse_float(parts[6])
        busbw = parse_float(parts[7])
        wrong = parse_int_like(parts[8])
        if time_us is None or algbw is None or busbw is None:
            continue
        rows.append(
            {
                "size": size,
                "count": count if count is not None else "",
                "time_us": time_us,
                "algbw": algbw,
                "busbw": busbw,
                "wrong": wrong,
            }
        )
    return rows


def log_has_fallback(log_path):
    if not log_path.exists():
        return False
    text = log_path.read_text(errors="replace")
    return any(pattern.search(text) for pattern in FALLBACK_PATTERNS)


def load_run_rows(runs_csv):
    with runs_csv.open(newline="") as f:
        return list(csv.DictReader(f))


def write_metrics(metrics_csv, run_rows):
    fieldnames = [
        "collective",
        "backend",
        "topology",
        "size",
        "count",
        "time_us",
        "algbw",
        "busbw",
        "wrong",
        "run_status",
        "exit_code",
        "fallback_detected",
        "log_path",
    ]

    metrics = []
    for run in run_rows:
        log_path = Path(run["log_path"])
        fallback_detected = log_has_fallback(log_path)
        parsed_rows = parse_nccl_tests_metrics(log_path)
        if not parsed_rows:
            metrics.append(
                {
                    "collective": run["collective"],
                    "backend": run["backend"],
                    "topology": run["topology"],
                    "size": "",
                    "count": "",
                    "time_us": "",
                    "algbw": "",
                    "busbw": "",
                    "wrong": "",
                    "run_status": run["status"],
                    "exit_code": run["exit_code"],
                    "fallback_detected": str(fallback_detected).lower(),
                    "log_path": str(log_path),
                }
            )
            continue

        for parsed in parsed_rows:
            metrics.append(
                {
                    "collective": run["collective"],
                    "backend": run["backend"],
                    "topology": run["topology"],
                    "size": parsed["size"],
                    "count": parsed["count"],
                    "time_us": parsed["time_us"],
                    "algbw": parsed["algbw"],
                    "busbw": parsed["busbw"],
                    "wrong": parsed["wrong"],
                    "run_status": run["status"],
                    "exit_code": run["exit_code"],
                    "fallback_detected": str(fallback_detected).lower(),
                    "log_path": str(log_path),
                }
            )

    with metrics_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)
    return metrics


def is_bad_metric(row, require_native):
    if row["run_status"] != "PASS":
        return "run failed"
    if parse_int_like(row["exit_code"]) != 0:
        return "non-zero exit"
    if row["time_us"] == "":
        return "no metrics parsed"
    wrong = parse_int_like(row["wrong"])
    if wrong is not None and wrong != 0:
        return "correctness failure"
    if require_native and row["backend"] == "mscclpp" and row["fallback_detected"] == "true":
        return "fallback detected"
    return ""


def geometric_mean(values):
    positives = [value for value in values if value > 0]
    if not positives:
        return 0.0
    return math.exp(sum(math.log(value) for value in positives) / len(positives))


def build_summary(metrics, min_speedup, require_native):
    by_key = {}
    for row in metrics:
        if row["size"] == "":
            key = (row["collective"], row["topology"], "")
        else:
            key = (row["collective"], row["topology"], str(row["size"]))
        by_key.setdefault(key, {})[row["backend"]] = row

    summary_rows = []
    speedups_by_group = {}
    for (collective, topology, size), backends in sorted(by_key.items()):
        nccl = backends.get("nccl")
        mscclpp = backends.get("mscclpp")
        status = "PASS"
        reason = ""
        speedup = ""

        if nccl is None:
            status, reason = "FAIL", "missing nccl row"
        elif mscclpp is None:
            status, reason = "FAIL", "missing mscclpp row"
        else:
            reason = is_bad_metric(nccl, False) or is_bad_metric(mscclpp, require_native)
            if reason:
                status = "FAIL"
            else:
                nccl_time = float(nccl["time_us"])
                mscclpp_time = float(mscclpp["time_us"])
                if mscclpp_time <= 0:
                    status, reason = "FAIL", "invalid mscclpp time"
                else:
                    speedup_value = nccl_time / mscclpp_time
                    speedup = f"{speedup_value:.6f}"
                    if speedup_value < min_speedup:
                        status = "FAIL"
                        reason = f"speedup {speedup_value:.3f} < {min_speedup:.3f}"
                    else:
                        speedups_by_group.setdefault((collective, topology), []).append(
                            speedup_value
                        )

        summary_rows.append(
            {
                "collective": collective,
                "topology": topology,
                "size": size,
                "status": status,
                "speedup": speedup,
                "reason": reason,
            }
        )

    for (collective, topology), speedups in sorted(speedups_by_group.items()):
        summary_rows.append(
            {
                "collective": collective,
                "topology": topology,
                "size": "geomean",
                "status": "PASS",
                "speedup": f"{geometric_mean(speedups):.6f}",
                "reason": "",
            }
        )
    return summary_rows


def write_summary(summary_csv, summary_rows):
    fieldnames = ["collective", "topology", "size", "status", "speedup", "reason"]
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main():
    args = parse_args()
    result_dir = args.result_dir
    runs_csv = result_dir / "runs.csv"
    metrics_csv = args.metrics_csv or result_dir / "metrics.csv"
    summary_csv = args.summary_csv or result_dir / "summary.csv"

    if not runs_csv.exists():
        print(f"error: missing {runs_csv}", file=sys.stderr)
        return 2

    run_rows = load_run_rows(runs_csv)
    metrics = write_metrics(metrics_csv, run_rows)
    summary_rows = build_summary(
        metrics, args.min_speedup, require_native=not args.allow_fallback
    )
    write_summary(summary_csv, summary_rows)

    failures = [row for row in summary_rows if row["status"] != "PASS"]
    print(f"metrics: {metrics_csv}")
    print(f"summary: {summary_csv}")
    if failures:
        for row in failures[:20]:
            print(
                "FAIL "
                f"{row['collective']} {row['topology']} size={row['size']}: "
                f"{row['reason']}",
                file=sys.stderr,
            )
        if len(failures) > 20:
            print(f"... {len(failures) - 20} more failures", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
