"""Host-only checks for complete, non-cherry-picked paired measurement summaries."""

import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "comparison", Path(__file__).resolve().parents[1] / "scripts/paired_summary.py"
)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class SummaryTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            dict(
                collective="all_gather",
                bytes="1024",
                in_place="0",
                backend=b,
                repeat=str(r),
                iteration=str(i),
                slowest_rank_us=str((r + 1) * (2 if b == "nccl" else 1)),
            )
            for b in ("nccl", "cpu-router")
            for r in range(5)
            for i in range(2)
        ]

    def summary(self, rows):
        return comparison.summarize(rows, "allgather", [1024], 5, 2, 4)

    def test_mean_and_bandwidth(self):
        nccl, router = self.summary(self.rows)
        self.assertEqual(nccl["mean_us"], 6)
        self.assertEqual(router["mean_us"], 3)
        self.assertEqual(router["speedup_vs_local_nccl"], 2)
        self.assertEqual(router["min_run_mean_us"], 1)
        self.assertEqual(router["max_run_mean_us"], 5)
        self.assertAlmostEqual(router["nccl_bus_GBs"], 1024 / 3 / 1000 * 0.75)

    def test_missing_and_duplicate(self):
        for rows in (self.rows[:-1], self.rows + [self.rows[0]], []):
            with self.assertRaises(ValueError):
                self.summary(rows)

    def test_progress_backend_requires_explicit_identity(self):
        for backend in ("gpu-progress-flat", "gpu-progress-striped"):
            rows = [
                {**r, "backend": backend} if r["backend"] == "cpu-router" else r
                for r in self.rows
            ]
            result = comparison.summarize(rows, "allgather", [1024], 5, 2, 4, backend)
            self.assertEqual(result[1]["speedup_vs_local_nccl"], 2)
            with self.assertRaises(ValueError):
                self.summary(rows)

    def test_reduce_scatter_name(self):
        rows = [{**row, "collective": "reduce_scatter"} for row in self.rows]
        result = comparison.summarize(rows, "reducescatter", [1024], 5, 2, 4)
        self.assertEqual(result[1]["speedup_vs_local_nccl"], 2)

    def test_gpu_reduction_is_separate_backend(self):
        rows = [
            (
                {**row, "backend": "cpu-router-gpu-reduce"}
                if row["backend"] == "cpu-router"
                else row
            )
            for row in self.rows
        ]
        result = comparison.summarize(
            rows, "allgather", [1024], 5, 2, 4, "cpu-router-gpu-reduce"
        )
        self.assertEqual(result[1]["backend"], "cpu-router-gpu-reduce")
        with self.assertRaises(ValueError):
            self.summary(rows)

    def test_striped_is_not_contiguous_reduction(self):
        rows = [
            (
                {**row, "backend": "cpu-router-gpu-striped"}
                if row["backend"] == "cpu-router"
                else row
            )
            for row in self.rows
        ]
        result = comparison.summarize(
            rows, "allgather", [1024], 5, 2, 4, "cpu-router-gpu-striped"
        )
        self.assertEqual(result[1]["speedup_vs_local_nccl"], 2)
        with self.assertRaises(ValueError):
            comparison.summarize(
                rows, "allgather", [1024], 5, 2, 4, "cpu-router-gpu-reduce"
            )

    def test_invalid_fields(self):
        for field, value in (
            ("collective", "reducescatter"),
            ("in_place", "1"),
            ("backend", "unknown-backend"),
            ("repeat", "5"),
            ("bytes", "2048"),
            ("iteration", "2"),
            ("slowest_rank_us", "nan"),
            ("slowest_rank_us", "inf"),
            ("slowest_rank_us", "0"),
            ("slowest_rank_us", "-1"),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.summary([{**self.rows[0], field: value}] + self.rows[1:])


if __name__ == "__main__":
    unittest.main()
