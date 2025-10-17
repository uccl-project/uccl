#!/usr/bin/env python3
"""
Find the most recent server/client logs and print Avg ms / BW in a table.
"""

import re
import sys
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

UCCL_HOME = os.environ.get("UCCL_HOME")
if UCCL_HOME:
    LOG_DIR_DEFAULT = Path(UCCL_HOME) / "p2p" / "tcpx" / "logs"
else:
    LOG_DIR_DEFAULT = Path(__file__).resolve().parents[1] / "logs"
AVG_RE = re.compile(
    r"\[PERF\]\s*Avg(?:\s*\([^)]*\))?:\s*([0-9.]+)\s*ms,\s*(?:BW|Bandwidth):\s*([0-9.]+)\s*GB/s",
    re.IGNORECASE,
)
SUMMARY_RE = re.compile(
    r"Total time:\s*([0-9.]+)\s*ms\s*[\r\n]+Bandwidth:\s*([0-9.]+)\s*GB/s",
    re.IGNORECASE,
)
GPU_RE = re.compile(r"\[PERF\]\s*GPU:\s*(\d+)")


def find_latest_log(log_dir: Path, role: str, gpu: int) -> Optional[Path]:
    pattern = f"fullmesh_{role}_gpu{gpu}_*.log"
    candidates = list(log_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_log(path: Path) -> Tuple[Optional[str], Optional[str]]:
    avg_ms = bw = None
    text = ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        print(f"[WARN] Failed to read {path}: {exc}", file=sys.stderr)
        return avg_ms, bw

    matches = list(AVG_RE.finditer(text))
    if matches:
        last = matches[-1]
        return last.group(1), last.group(2)

    summary_matches = list(SUMMARY_RE.finditer(text))
    if summary_matches:
        last = summary_matches[-1]
        return last.group(1), last.group(2)

    return avg_ms, bw


def discover_gpu_ids(log_dir: Path) -> List[int]:
    ids = set()
    for role in ("server", "client"):
        for path in log_dir.glob(f"fullmesh_{role}_gpu*_*.log"):
            m = re.search(r"_gpu(\d+)_", path.name)
            if m:
                ids.add(int(m.group(1)))
    if not ids:
        return list(range(8))
    return sorted(ids)


def main() -> None:
    args = sys.argv[1:]

    log_dir = Path(args[0]) if args else LOG_DIR_DEFAULT
    if not log_dir.exists():
        print(f"[ERROR] Log directory {log_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    if len(args) > 1:
        gpu_ids: Iterable[int] = [int(x) for x in args[1].split(",") if x.strip()]
    else:
        gpu_ids = discover_gpu_ids(log_dir)

    rows = []
    for gpu in gpu_ids:
        server_log = find_latest_log(log_dir, "server", gpu)
        client_log = find_latest_log(log_dir, "client", gpu)
        server_avg, server_bw = parse_log(server_log) if server_log else (None, None)
        client_avg, client_bw = parse_log(client_log) if client_log else (None, None)
        rows.append(
            (
                str(gpu),
                server_avg or "?",
                server_bw or "?",
                server_log.name if server_log else "-",
                client_avg or "?",
                client_bw or "?",
                client_log.name if client_log else "-",
            )
        )

    header = (
        "GPU",
        "Server Avg ms",
        "Server BW",
        "Server Log",
        "Client Avg ms",
        "Client BW",
        "Client Log",
    )
    widths = [
        max(len(str(row[i])) for row in ([header] + rows)) for i in range(len(header))
    ]
    fmt = "  ".join(f"{{:{w}}}" for w in widths)
    print(fmt.format(*header))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*row))
    if rows:

        def safe_float(value: str) -> Optional[float]:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        server_avg_vals = [
            safe_float(row[1]) for row in rows if safe_float(row[1]) is not None
        ]
        server_bw_vals = [
            safe_float(row[2]) for row in rows if safe_float(row[2]) is not None
        ]
        client_avg_vals = [
            safe_float(row[4]) for row in rows if safe_float(row[4]) is not None
        ]
        client_bw_vals = [
            safe_float(row[5]) for row in rows if safe_float(row[5]) is not None
        ]

        def mean_fmt(values: List[float]) -> str:
            return f"{sum(values) / len(values):.3f}" if values else "-"

        avg_row = (
            "AVG",
            mean_fmt(server_avg_vals),
            mean_fmt(server_bw_vals),
            "-",
            mean_fmt(client_avg_vals),
            mean_fmt(client_bw_vals),
            "-",
        )
        print(fmt.format(*avg_row))


if __name__ == "__main__":
    main()
