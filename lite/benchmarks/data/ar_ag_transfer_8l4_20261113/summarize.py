"""Summarize two fixed-configuration confirmation sweeps, without selection."""

import csv
from pathlib import Path
from statistics import median

DATA = Path(__file__).resolve().parent
BASE_VARIANTS = ("previous", "batch", "progress", "combined", "compact_combined")


def load(name):
    with (DATA / "raw" / f"{name}.csv").open() as stream:
        rows = list(csv.DictReader(line for line in stream if not line.startswith("#")))
    assert len(rows) == 26, name
    return {(int(row["bytes"]), row["backend"] == "fallback"): row for row in rows}


def main():
    result = []
    for op in ("allreduce", "allgather"):
        variants = BASE_VARIANTS + (("cpu_parallel", "cpu_parallel_compact")
                                    if op == "allreduce" else ())
        sweeps = {(variant, sweep): load(f"final_{op}_{variant}_{sweep}")
                  for variant in variants for sweep in ("a", "b")}
        for variant in variants:
            for size in (1 << power for power in range(10, 23)):
                router = [float(sweeps[variant, sweep][size, False]["median_us"])
                          for sweep in ("a", "b")]
                nccl = [float(sweeps[variant, sweep][size, True]["median_us"])
                        for sweep in ("a", "b")]
                previous = median(float(sweeps["previous", sweep][size, False]["median_us"])
                                  for sweep in ("a", "b"))
                point = median(router)
                result.append({
                    "collective": op,
                    "in_place": int(op == "allreduce"),
                    "variant": variant,
                    "bytes": size,
                    "median_us": point,
                    "sweep_min_us": min(router),
                    "sweep_max_us": max(router),
                    "paired_nccl_median_us": median(nccl),
                    "paired_nccl_sweep_min_us": min(nccl),
                    "paired_nccl_sweep_max_us": max(nccl),
                    "paired_nccl_over_router": median(nccl) / point,
                    "reduction_vs_previous_pct": (1 - point / previous) * 100,
                })
    with (DATA / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=result[0].keys())
        writer.writeheader()
        for row in result:
            writer.writerow({key: f"{value:.6f}" if isinstance(value, float) else value
                             for key, value in row.items()})
    print(f"Wrote {len(result)} points; no best-per-size selection.")


if __name__ == "__main__":
    main()
