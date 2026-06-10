#!/usr/bin/env python3
"""Generate AllReduce performance plots and markdown summaries.

The data sources are nccl-tests logs kept under .tmp/collective-benchmarks.
Rows use out-of-place values for the primary plotted comparison, while
in-place values are retained in plot_data.csv for cases where they diverge.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "plots" / "allreduce"
DOC_PATH = ROOT / "doc" / "perf-allreduce.md"

SETUP_ORDER = ["1nx4g", "2nx1g", "2nx2g", "2nx4g"]

SIZE_LABELS = {
    128: "128B",
    256: "256B",
    512: "512B",
    1024: "1KiB",
    2048: "2KiB",
    4096: "4KiB",
    8192: "8KiB",
    16384: "16KiB",
    32768: "32KiB",
    65536: "64KiB",
    131072: "128KiB",
    262144: "256KiB",
    524288: "512KiB",
    1048576: "1MiB",
    2097152: "2MiB",
    4194304: "4MiB",
    8388608: "8MiB",
    16777216: "16MiB",
    33554432: "32MiB",
    67108864: "64MiB",
    134217728: "128MiB",
    268435456: "256MiB",
    536870912: "512MiB",
    1073741824: "1GiB",
}

LOG_ROW_RE = re.compile(
    r"^\s*(\d+)\s+\d+\s+\w+\s+\w+\s+\S+\s+"
    r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\S+)\s+"
    r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\S+)"
)


@dataclass
class Metrics:
    out_time_us: float
    out_busbw: float
    in_time_us: float
    in_busbw: float
    log: str


@dataclass
class Row:
    setup: str
    size_bytes: int
    size: str
    lite_time_us: float
    nccl_time_us: float
    lite_busbw: float
    nccl_busbw: float
    lite_in_time_us: float
    nccl_in_time_us: float
    lite_in_busbw: float
    nccl_in_busbw: float
    lite_log: str
    nccl_log: str


def size_label(size: int) -> str:
    return SIZE_LABELS.get(size, f"{size}B")


def repo_relative(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_nccl_tests_log(path: Path) -> tuple[dict[int, Metrics], int]:
    rows: dict[int, Metrics] = {}
    wrong = 0
    for line in path.read_text().splitlines():
        match = LOG_ROW_RE.match(line)
        if not match:
            continue
        size = int(match.group(1))
        wrong += int(float(match.group(5))) + int(float(match.group(9)))
        rows[size] = Metrics(
            out_time_us=float(match.group(2)),
            out_busbw=float(match.group(4)),
            in_time_us=float(match.group(6)),
            in_busbw=float(match.group(8)),
            log=repo_relative(path),
        )
    return rows, wrong


def merge_logs(paths: Iterable[Path]) -> dict[int, Metrics]:
    merged: dict[int, Metrics] = {}
    total_wrong = 0
    for path in paths:
        rows, wrong = parse_nccl_tests_log(path)
        total_wrong += wrong
        merged.update(rows)
    if total_wrong:
        joined = ", ".join(repo_relative(p) for p in paths)
        raise RuntimeError(f"#wrong is non-zero in {joined}: {total_wrong}")
    return merged


def log(*parts: str) -> Path:
    return ROOT / ".tmp" / "collective-benchmarks" / Path(*parts)


SOURCES = {
    "1nx4g": {
        "display": "1nx4g",
        "lite": [
            log("ar-topologies-20260610-064813", "allreduce_1nx4g_mscclpp_128B_1G.log"),
        ],
        "nccl": [
            log("ar-topologies-20260610-064813", "allreduce_1nx4g_nccl_128B_1G.log"),
        ],
    },
    "2nx1g": {
        "display": "2nx1g",
        "lite": [
            log("ar-topologies-20260610-064813", "allreduce_2nx1g_mscclpp_128B_1G.log"),
            log("ar-2nx1g-lite-current-20260610-143936", "lite_current_2nx1g_1M_64M.log"),
            log("ar-2nx1g-small-final-20260610-160846", "lite_2nx1g_128B_1M.log"),
            log("ar-2nx1g-large-seq-20260610-145943", "lite_2nx1g_64M_512M.log"),
            log("ar-2nx1g-large-repeat-20260610-150034", "lite_2nx1g_128M_512M_repeat.log"),
            log("ar-2nx1g-1g-final-20260610-155839", "lite_2nx1g_1G.log"),
        ],
        "nccl": [
            log("ar-topologies-20260610-064813", "allreduce_2nx1g_nccl_128B_1G.log"),
            log("ar-2nx1g-nccl-fresh-20260610-140134", "nccl_2nx1g_1M_64M.log"),
            log("ar-2nx1g-small-final-20260610-160846", "nccl_2nx1g_128B_1M.log"),
            log("ar-2nx1g-large-seq-20260610-145943", "nccl_2nx1g_64M_512M.log"),
            log("ar-2nx1g-1g-final-20260610-155839", "nccl_2nx1g_1G.log"),
        ],
    },
    "2nx2g": {
        "display": "2nx2g",
        "lite": [
            log("ar-topologies-20260610-064813", "allreduce_2nx2g_mscclpp_128B_1G.log"),
            log("ar-topologies-20260610-064813", "allreduce_2nx2g_mscclpp_small_final_guardfix.log"),
        ],
        "nccl": [
            log("ar-topologies-20260610-064813", "allreduce_2nx2g_nccl_128B_1G.log"),
        ],
    },
    "2nx4g": {
        "display": "2nx4g",
        "lite": [
            log("ar-small-opt-20260610-020910", "allreduce_lite_small_final_reviewfix_128B_1M.log"),
            log("ar-final-1M-1G-20260610-055550", "allreduce_mscclpp_1M_1G.log"),
        ],
        "nccl": [
            log("ar-small-opt-20260610-020910", "allreduce_nccl_fresh_128B_1M.log"),
            log("ar-final-1M-1G-20260610-055550", "allreduce_nccl_1M_1G.log"),
        ],
    },
}


def load_rows() -> list[Row]:
    rows: list[Row] = []
    for setup in SETUP_ORDER:
        source = SOURCES[setup]
        lite = merge_logs(source["lite"])
        nccl = merge_logs(source["nccl"])
        for size in sorted(set(lite) & set(nccl)):
            rows.append(
                Row(
                    setup=setup,
                    size_bytes=size,
                    size=size_label(size),
                    lite_time_us=lite[size].out_time_us,
                    nccl_time_us=nccl[size].out_time_us,
                    lite_busbw=lite[size].out_busbw,
                    nccl_busbw=nccl[size].out_busbw,
                    lite_in_time_us=lite[size].in_time_us,
                    nccl_in_time_us=nccl[size].in_time_us,
                    lite_in_busbw=lite[size].in_busbw,
                    nccl_in_busbw=nccl[size].in_busbw,
                    lite_log=lite[size].log,
                    nccl_log=nccl[size].log,
                )
            )
    return rows


def geomean(values: Iterable[float]) -> float:
    vals = [v for v in values if v > 0 and math.isfinite(v)]
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def rows_for_setup(rows: list[Row], setup: str) -> list[Row]:
    return sorted([r for r in rows if r.setup == setup], key=lambda r: r.size_bytes)


def write_plot_data(rows: list[Row]) -> None:
    path = OUT_DIR / "plot_data.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(Row.__dataclass_fields__.keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def plot_setup(rows: list[Row], setup: str) -> list[str]:
    setup_rows = rows_for_setup(rows, setup)
    files: list[str] = []
    if not setup_rows:
        return files

    def common_x(selected: list[Row]) -> tuple[list[int], list[str]]:
        return [r.size_bytes for r in selected], [r.size for r in selected]

    latency_rows = [r for r in setup_rows if r.size_bytes <= 1024 * 1024]
    if latency_rows:
        x, labels = common_x(latency_rows)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(x, [r.lite_time_us for r in latency_rows], marker="o", label="Lite")
        ax.plot(x, [r.nccl_time_us for r in latency_rows], marker="o", label="NCCL")
        ax.set_xscale("log", base=2)
        ax.set_ylabel("Latency (us)")
        ax.set_title(f"{setup}: AllReduce latency, 128B-1MiB")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        name = f"{setup}_latency_128B_1MiB.png"
        fig.savefig(OUT_DIR / name, dpi=160)
        plt.close(fig)
        files.append(name)

    bus_rows = [r for r in setup_rows if r.size_bytes >= 1024 * 1024]
    if bus_rows:
        x, labels = common_x(bus_rows)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(x, [r.lite_busbw for r in bus_rows], marker="o", label="Lite")
        ax.plot(x, [r.nccl_busbw for r in bus_rows], marker="o", label="NCCL")
        ax.set_xscale("log", base=2)
        ax.set_ylabel("Bus bandwidth (GB/s)")
        ax.set_title(f"{setup}: AllReduce bus bandwidth, 1MiB-1GiB")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        name = f"{setup}_busbw_1MiB_1GiB.png"
        fig.savefig(OUT_DIR / name, dpi=160)
        plt.close(fig)
        files.append(name)
    return files


def markdown_table(rows: list[Row], metric: str) -> str:
    if metric == "latency":
        header = "| Size | Lite us | NCCL us | Lite speedup |\n| ---: | ---: | ---: | ---: |"
        body = [
            f"| {r.size} | {r.lite_time_us:.2f} | {r.nccl_time_us:.2f} | {r.nccl_time_us / r.lite_time_us:.2f}x |"
            for r in rows
        ]
    else:
        header = "| Size | Lite GB/s | NCCL GB/s | Lite / NCCL |\n| ---: | ---: | ---: | ---: |"
        body = [
            f"| {r.size} | {r.lite_busbw:.2f} | {r.nccl_busbw:.2f} | {r.lite_busbw / r.nccl_busbw:.2f}x |"
            for r in rows
        ]
    return "\n".join([header, *body])


def markdown_content(rows: list[Row], plot_files: dict[str, list[str]], plot_prefix: str) -> str:
    lines: list[str] = [
        "# Lite AllReduce vs NCCL Performance",
        "",
        "Date: 2026-06-10",
        "",
        "Primary tables and plots use out-of-place nccl-tests results. Ratios greater than `1.00x` favor Lite: latency speedup is `NCCL us / Lite us`, and bus bandwidth ratio is `Lite GB/s / NCCL GB/s`. In-place values are retained in `plot_data.csv` because AllReduce can diverge materially for some paths.",
        "",
        "## Benchmark settings and sources",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        "| Iterations | `50` |",
        "| Warmup | `20` |",
        "| Size convention | latency: `128B-1MiB`; bus bandwidth: `1MiB-1GiB` |",
        "| Multi-node hosts | `10.10.55.1,10.10.55.2` |",
        "| NCCL multi-node baseline | `NCCL_NET_GDR_LEVEL=0` no-GDR where explicitly re-run |",
        "| Generated plots | `plots/allreduce/` |",
        "",
        "All plotted source rows reported `#wrong=0`.",
        "",
        "## Summary",
        "",
        "| Setup | Latency geomean Lite speedup | BusBW geomean Lite/NCCL | 1GiB Lite GB/s | 1GiB NCCL GB/s | 1GiB ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for setup in SETUP_ORDER:
        setup_rows = rows_for_setup(rows, setup)
        latency_rows = [r for r in setup_rows if r.size_bytes <= 1024 * 1024]
        bus_rows = [r for r in setup_rows if r.size_bytes >= 1024 * 1024]
        one_gib = next((r for r in setup_rows if r.size_bytes == 1024 * 1024 * 1024), None)
        lat_gm = geomean(r.nccl_time_us / r.lite_time_us for r in latency_rows) if latency_rows else float("nan")
        bw_gm = geomean(r.lite_busbw / r.nccl_busbw for r in bus_rows) if bus_rows else float("nan")
        if one_gib:
            lines.append(
                f"| `{setup}` | {lat_gm:.2f}x | {bw_gm:.2f}x | {one_gib.lite_busbw:.2f} | {one_gib.nccl_busbw:.2f} | {one_gib.lite_busbw / one_gib.nccl_busbw:.2f}x |"
            )
    lines.append("")

    for setup in SETUP_ORDER:
        setup_rows = rows_for_setup(rows, setup)
        if not setup_rows:
            continue
        lines.extend([f"## {SOURCES[setup]['display']}", ""])
        latency_rows = [r for r in setup_rows if r.size_bytes <= 1024 * 1024]
        if latency_rows:
            lines.extend(["### Latency, 128B-1MiB (us)", "", markdown_table(latency_rows, "latency"), ""])
        bus_rows = [r for r in setup_rows if r.size_bytes >= 1024 * 1024]
        if bus_rows:
            lines.extend(["### Bus bandwidth, 1MiB-1GiB (GB/s)", "", markdown_table(bus_rows, "busbw"), ""])
        if setup in plot_files:
            lines.append("Plots: " + ", ".join(f"[`{p}`]({plot_prefix}{p})" for p in plot_files[setup]))
            lines.append("")

    lines.extend(["## Source logs", ""])
    for setup in SETUP_ORDER:
        lines.append(f"### {setup}")
        for kind in ("lite", "nccl"):
            for path in SOURCES[setup][kind]:
                lines.append(f"- {kind}: `{repo_relative(path)}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_markdown(rows: list[Row], plot_files: dict[str, list[str]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.md").write_text(markdown_content(rows, plot_files, ""))
    DOC_PATH.write_text(markdown_content(rows, plot_files, "../plots/allreduce/"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    rows = sorted(rows, key=lambda r: (SETUP_ORDER.index(r.setup), r.size_bytes))
    write_plot_data(rows)
    plot_files = {setup: plot_setup(rows, setup) for setup in SETUP_ORDER}
    write_markdown(rows, plot_files)


if __name__ == "__main__":
    main()
