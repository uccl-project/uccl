#!/usr/bin/env python3
"""Unit tests for MCCL results, counters and paired comparisons."""

import tempfile
import unittest
from pathlib import Path

from mccl_bench import (
    counter_delta,
    parse_results,
    read_counters,
    summarize,
    size_sweep,
    validate_coverage,
)

ROW = "67108864 16777216 float sum -1 2196.6 30.55 30.55 0 2694.3 24.91 24.91 0"


class BenchTest(unittest.TestCase):
    def test_table(self):
        rows = parse_results("NET/Plugin UCCL\n" + ROW)
        self.assertEqual(rows[0]["bytes"], 67108864)
        self.assertEqual(rows[0]["mode"], "out_of_place")
        self.assertEqual(rows[1]["time_us"], 2694.3)

    def test_no_false_pass(self):
        for text in (
            "UCCL-MCCL-RDMA correctness PASS",
            ROW.replace("30.55 0", "30.55 1"),
            ROW.replace("2196.6", "nan"),
            "67108864 1 float sum -1 100 1 1 0",
            ROW.replace("67108864", "-1"),
            ROW.replace("16777216", "0"),
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_results(text)

    def test_na_is_not_validation(self):
        rows = parse_results(ROW.rsplit(" ", 1)[0] + " N/A")
        self.assertEqual(rows[1]["validation"], "unsupported")
        with self.assertRaises(ValueError):
            parse_results("0 0 float none -1 1 0 0 0 1 0 0 N/A")

    def test_counters(self):
        self.assertEqual(
            counter_delta({"tx": 2, "rx": 5}, {"tx": 4, "rx": 8}, "4octets"),
            {"tx_bytes": 8, "rx_bytes": 12},
        )
        for unit, after in (
            ("unknown", {"tx": 4, "rx": 8}),
            ("bytes", {"tx": 1, "rx": 8}),
        ):
            with self.assertRaises(ValueError):
                counter_delta({"tx": 2, "rx": 5}, after, unit)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            counters = root / "mlx5_0/ports/1/counters"
            counters.mkdir(parents=True)
            (counters / "port_xmit_data").write_text("2\n")
            (counters / "port_rcv_data").write_text("5\n")
            snapshot = read_counters(root, "mlx5_0:1")
            self.assertEqual(snapshot["values"], {"tx": 2, "rx": 5})
            self.assertEqual(snapshot["unit"], "4octets")

    def test_paired_summary(self):
        records = []
        for index in range(5):
            for backend, time in (("native", 100), ("uccl", 90)):
                records.append(
                    {
                        "backend": backend,
                        "run": index,
                        "collective": "allreduce",
                        "rows": [{**parse_results(ROW)[0], "time_us": time}],
                    }
                )
        result = summarize(records)
        self.assertEqual(result[0]["paired_latency_change_pct"], -10)
        self.assertEqual(result[0]["pairs"], 5)
        with self.assertRaises(ValueError):
            summarize(records[:-1])
        records[-1]["rows"][0]["count"] = 1
        with self.assertRaises(ValueError):
            summarize(records)
        records[-1]["rows"][0]["count"] = 16777216
        records[-1]["fixed_environment"] = {"UCCL_NUM_ENGINES": "99"}
        with self.assertRaises(ValueError):
            summarize(records)

    def test_size_coverage(self):
        self.assertEqual(size_sweep("8", "32", 2), [8, 16, 32])
        with self.assertRaises(ValueError):
            validate_coverage(parse_results(ROW), size_sweep("8", "64M", 2))
        validate_coverage(parse_results(ROW), [67108864])
        for factor in (0, 1):
            with self.assertRaises(ValueError):
                size_sweep("8", "64M", factor)


if __name__ == "__main__":
    unittest.main()
