"""Validate complete paired collective samples and summarize every run."""

import math
import statistics


def summarize(
    samples, operation, sizes, repeats, iterations, ranks, router_backend="cpu-router"
):
    """Require every checked iteration; summarize means, not best observations."""
    expected = {
        (size, backend, repeat, iteration)
        for size in sizes
        for backend in ("nccl", router_backend)
        for repeat in range(repeats)
        for iteration in range(iterations)
    }
    values = {}
    canonical = {"allgather": "all_gather", "reducescatter": "reduce_scatter"}[
        operation
    ]
    for row in samples:
        if row["collective"] != canonical or row["in_place"] != "0":
            raise ValueError("unexpected operation or placement")
        key = (
            int(row["bytes"]),
            row["backend"],
            int(row["repeat"]),
            int(row["iteration"]),
        )
        value = float(row["slowest_rank_us"])
        if (
            key not in expected
            or key in values
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("unexpected, duplicate, or invalid timing record")
        values[key] = value
    if set(values) != expected:
        raise ValueError("incomplete run: missing required timing records")
    result = []
    for size in sizes:
        backend_stats = {}
        for backend in ("nccl", router_backend):
            run_means = [
                statistics.mean(values[size, backend, r, i] for i in range(iterations))
                for r in range(repeats)
            ]
            backend_stats[backend] = (
                statistics.mean(run_means),
                min(run_means),
                max(run_means),
            )
        for backend, (mean, low, high) in backend_stats.items():
            result.append(
                {
                    "collective": operation,
                    "full_array_bytes": size,
                    "ranks": ranks,
                    "dtype": "bf16",
                    "backend": backend,
                    "runs": repeats,
                    "iterations_per_run": iterations,
                    "mean_us": mean,
                    "min_run_mean_us": low,
                    "max_run_mean_us": high,
                    "algorithm_GBs": size / mean / 1000,
                    "nccl_bus_GBs": size / mean / 1000 * (ranks - 1) / ranks,
                    "speedup_vs_local_nccl": backend_stats["nccl"][0] / mean,
                }
            )
    return result
