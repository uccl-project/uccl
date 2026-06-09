#!/usr/bin/env python3
"""Generate AllGather performance plots and markdown summaries.

The data sources are nccl-tests logs kept under .tmp/collective-benchmarks.
For the two-node topologies, the NCCL baseline is the per-size tuned envelope
from the earlier NCCL tuning sweep.  For 1nx4g, the script uses the final
single-node host/CudaIpc logs.
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
OUT_DIR = ROOT / "plots" / "allgather"
DOC_PATH = ROOT / "doc" / "perf-allgather.md"

TUNED_PLOT_DATA = (
    ROOT
    / ".tmp"
    / "collective-benchmarks"
    / "ag-nccl-tune-20260609-022425"
    / "plots"
    / "plot_data.csv"
)

LOG_SOURCES = {
    "1nx4g_host": {
        "display": "1nx4g host-memory vs NCCL SHM-only",
        "lite_full": ROOT
        / ".tmp"
        / "collective-benchmarks"
        / "ag-1nx4g-host-beat-check-20260609-085713"
        / "lite_host_allgather_1nx4g.log",
        "nccl_full": ROOT
        / ".tmp"
        / "collective-benchmarks"
        / "ag-1nx4g-host-beat-check-20260609-085713"
        / "nccl_shm_allgather_1nx4g.log",
        "lite_large": ROOT
        / ".tmp"
        / "collective-benchmarks"
        / "ag-1nx4g-host-refactor-final-20260609-101051"
        / "lite_host_allgather_1nx4g_large.log",
        "nccl_large": ROOT
        / ".tmp"
        / "collective-benchmarks"
        / "ag-1nx4g-host-refactor-final-20260609-101051"
        / "nccl_shm_allgather_1nx4g_large.log",
        "nccl_label": "NCCL SHM-only",
    },
    "1nx4g_cudaipc": {
        "display": "1nx4g CudaIpc vs NCCL",
        "lite_full": ROOT
        / ".tmp"
        / "collective-benchmarks"
        / "ag-1nx4g-postfix-20260609-060049"
        / "lite_allgather_1nx4g.log",
        "nccl_full": ROOT
        / ".tmp"
        / "collective-benchmarks"
        / "ag-1nx4g-postfix-20260609-060049"
        / "nccl_allgather_1nx4g.log",
        "nccl_label": "NCCL",
    },
}

SETUP_ORDER = ["1nx4g_host", "1nx4g_cudaipc", "2nx1g", "2nx2g", "2nx4g"]

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
class Row:
    setup: str
    size_bytes: int
    size: str
    lite_time_us: float
    nccl_time_us: float
    lite_busbw: float
    nccl_busbw: float
    nccl_cfg: str
    nccl_log: str
    nccl_label: str


def size_label(size: int) -> str:
    return SIZE_LABELS.get(size, f"{size}B")


def parse_nccl_tests_log(path: Path) -> tuple[dict[int, tuple[float, float]], int]:
    rows: dict[int, tuple[float, float]] = {}
    wrong = 0
    for line in path.read_text().splitlines():
        match = LOG_ROW_RE.match(line)
        if not match:
            continue
        size = int(match.group(1))
        out_time = float(match.group(2))
        out_busbw = float(match.group(4))
        wrong += int(float(match.group(5))) + int(float(match.group(9)))
        rows[size] = (out_time, out_busbw)
    return rows, wrong


def load_1nx_rows() -> list[Row]:
    rows: list[Row] = []
    for setup, source in LOG_SOURCES.items():
        lite, lite_wrong = parse_nccl_tests_log(source["lite_full"])
        nccl, nccl_wrong = parse_nccl_tests_log(source["nccl_full"])
        if "lite_large" in source:
            large_lite, large_lite_wrong = parse_nccl_tests_log(source["lite_large"])
            large_nccl, large_nccl_wrong = parse_nccl_tests_log(source["nccl_large"])
            lite.update(large_lite)
            nccl.update(large_nccl)
            lite_wrong += large_lite_wrong
            nccl_wrong += large_nccl_wrong
        if lite_wrong or nccl_wrong:
            raise RuntimeError(f"{setup} has #wrong: lite={lite_wrong}, nccl={nccl_wrong}")
        for size in sorted(set(lite) & set(nccl)):
            rows.append(
                Row(
                    setup=setup,
                    size_bytes=size,
                    size=size_label(size),
                    lite_time_us=lite[size][0],
                    nccl_time_us=nccl[size][0],
                    lite_busbw=lite[size][1],
                    nccl_busbw=nccl[size][1],
                    nccl_cfg="shm_only" if setup == "1nx4g_host" else "default",
                    nccl_log=repo_relative(
                        source.get(
                            "nccl_large"
                            if size >= 4 * 1024 * 1024
                            and "nccl_large" in source
                            else "nccl_full"
                        )
                    ),
                    nccl_label=source["nccl_label"],
                )
            )
    return rows


def load_tuned_2nx_rows() -> list[Row]:
    by_key: dict[tuple[str, int], dict[str, str]] = {}
    with TUNED_PLOT_DATA.open(newline="") as f:
        for row in csv.DictReader(f):
            setup = row["setup"]
            size = int(row["size_bytes"])
            if setup not in {"2nx1g", "2nx2g", "2nx4g"}:
                continue
            entry = by_key.setdefault((setup, size), {})
            metric = row["metric"]
            if metric == "latency":
                entry["lite_time_us"] = row["lite"]
                entry["nccl_time_us"] = row["nccl_tuned"]
                entry["lite_latency_busbw"] = row["lite_busbw"]
                entry["nccl_latency_busbw"] = row["nccl_busbw"]
                entry["latency_cfg"] = row["nccl_cfg"]
                entry["latency_log"] = row["nccl_log"]
            elif metric == "busbw":
                entry["lite_busbw"] = row["lite"]
                entry["nccl_busbw"] = row["nccl_tuned"]
                entry["lite_bus_time_us"] = row["lite_time_us"]
                entry["nccl_bus_time_us"] = row["nccl_time_us"]
                entry["busbw_cfg"] = row["nccl_cfg"]
                entry["busbw_log"] = row["nccl_log"]

    rows: list[Row] = []
    for (setup, size), entry in sorted(by_key.items()):
        rows.append(
            Row(
                setup=setup,
                size_bytes=size,
                size=size_label(size),
                lite_time_us=float(entry.get("lite_time_us") or entry.get("lite_bus_time_us")),
                nccl_time_us=float(entry.get("nccl_time_us") or entry.get("nccl_bus_time_us")),
                lite_busbw=float(entry.get("lite_busbw") or entry.get("lite_latency_busbw")),
                nccl_busbw=float(entry.get("nccl_busbw") or entry.get("nccl_latency_busbw")),
                nccl_cfg=entry.get("busbw_cfg") or entry.get("latency_cfg") or "",
                nccl_log=entry.get("busbw_log") or entry.get("latency_log") or "",
                nccl_label="NCCL tuned",
            )
        )
    return rows


def geomean(values: Iterable[float]) -> float:
    vals = [v for v in values if v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def repo_relative(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def rows_for_setup(rows: list[Row], setup: str) -> list[Row]:
    return sorted([r for r in rows if r.setup == setup], key=lambda r: r.size_bytes)


def write_plot_data(rows: list[Row]) -> None:
    path = OUT_DIR / "plot_data.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "setup",
                "size",
                "size_bytes",
                "lite_time_us",
                "nccl_time_us",
                "lite_busbw",
                "nccl_busbw",
                "nccl_cfg",
                "nccl_log",
                "nccl_label",
            ],
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
        ax.plot(x, [r.nccl_time_us for r in latency_rows], marker="o", label=latency_rows[0].nccl_label)
        ax.set_xscale("log", base=2)
        ax.set_ylabel("Latency (us)")
        ax.set_title(f"{setup}: AllGather latency, 128B-1MiB")
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
        ax.plot(x, [r.nccl_busbw for r in bus_rows], marker="o", label=bus_rows[0].nccl_label)
        ax.set_xscale("log", base=2)
        ax.set_ylabel("Bus bandwidth (GB/s)")
        ax.set_title(f"{setup}: AllGather bus bandwidth, 1MiB-1GiB")
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


def setup_display(setup: str) -> str:
    displays = {
        "1nx4g_host": "1nx4g host-memory (no CudaIpc payload) vs NCCL SHM-only",
        "1nx4g_cudaipc": "1nx4g CudaIpc vs NCCL",
        "2nx1g": "2nx1g",
        "2nx2g": "2nx2g",
        "2nx4g": "2nx4g",
    }
    return displays[setup]


def write_markdown(rows: list[Row], plot_files: dict[str, list[str]]) -> None:
    lines: list[str] = [
        "# Lite AllGather vs NCCL Performance",
        "",
        "Date: 2026-06-09",
        "",
        "All rows use out-of-place nccl-tests results. Ratios greater than `1.00x` favor Lite: latency speedup is `NCCL us / Lite us`, and bus bandwidth ratio is `Lite GB/s / NCCL GB/s`.",
        "",
        "## Benchmark settings and sources",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        "| Iterations | `50` |",
        "| Warmup | `20` |",
        "| Size convention | latency: `128B-1MiB`; bus bandwidth: `1MiB-1GiB` |",
        "| Multi-node hosts | `10.10.55.1,10.10.55.2` |",
        "| 1nx4g host baseline | NCCL SHM-only: `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=0` |",
        "| 2nx NCCL baseline | Per-size tuned NCCL envelope from `.tmp/collective-benchmarks/ag-nccl-tune-20260609-022425/` |",
        "| Refactor validation | `.tmp/collective-benchmarks/ag-refactor-validate-20260609-103556/` |",
        "| Generated plots | `plots/allgather/` |",
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
        lines.extend([f"## {setup_display(setup)}", ""])
        latency_rows = [r for r in setup_rows if r.size_bytes <= 1024 * 1024]
        if latency_rows:
            lines.extend(["### Latency, 128B-1MiB (us)", "", markdown_table(latency_rows, "latency"), ""])
        bus_rows = [r for r in setup_rows if r.size_bytes >= 1024 * 1024]
        if bus_rows:
            lines.extend(["### Bus bandwidth, 1MiB-1GiB (GB/s)", "", markdown_table(bus_rows, "busbw"), ""])
        if setup in plot_files:
            lines.append("Plots: " + ", ".join(f"[`{p}`](../plots/allgather/{p})" for p in plot_files[setup]))
            lines.append("")

    lines.extend(["## Plot files", ""])
    for setup in SETUP_ORDER:
        for p in plot_files.get(setup, []):
            lines.append(f"- `{p}`")
    lines.append("")

    content = "\n".join(lines)
    DOC_PATH.write_text(content)
    (OUT_DIR / "summary.md").write_text(content)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_1nx_rows() + load_tuned_2nx_rows()
    rows = sorted(rows, key=lambda r: (SETUP_ORDER.index(r.setup), r.size_bytes))
    write_plot_data(rows)
    plot_files = {setup: plot_setup(rows, setup) for setup in SETUP_ORDER}
    write_markdown(rows, plot_files)


if __name__ == "__main__":
    main()
