"""Regression tests for test selection and the controller's failure contract."""

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import queue
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import catalog
from performance import compare, evaluate
from process import run
from report import report_results
from run import Worker, collect_results, write_summary


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.targets = {
            "cpu": dict(enabled=True, hosts=[], gpus=0, capabilities=["cpu"]),
            "gpu": dict(enabled=True, hosts=["a", "b"], gpus=1, capabilities=["cuda"]),
            "offline": dict(
                enabled=False,
                hosts=["c"],
                gpus=1,
                capabilities=["cuda"],
                reason="offline",
            ),
        }
        self.catalog = {
            "tests": [
                dict(
                    id="host",
                    enabled=True,
                    tier="host",
                    sources=["p2p/tests/test_host.py"],
                    paths=["p2p/**"],
                    targets=["cpu"],
                    requires=["cpu"],
                    timeout=5,
                    command="true",
                ),
                dict(
                    id="gpu",
                    enabled=True,
                    tier="integration",
                    sources=["ep/test_gpu.cu"],
                    paths=["ep/**"],
                    targets=["gpu", "offline"],
                    requires=["cuda"],
                    timeout=5,
                    command="true",
                ),
                dict(
                    id="off",
                    enabled=False,
                    tier="integration",
                    sources=["off/test_missing.cc"],
                    reason="no hardware",
                ),
            ]
        }

    def selected(self, **kw):
        return catalog.plan(self.catalog, self.targets, **kw)

    def test_changed_component_and_offline_target(self):
        p = self.selected(changed=["ep/kernel.cu"])
        self.assertEqual(p["include"], [{"target": "gpu", "tests": ["gpu"]}])
        self.assertEqual(p["decisions"][0]["reason"], "unaffected")

    def test_documentation_only(self):
        self.assertEqual(self.selected(changed=["docs/readme.md"])["include"], [])

    def test_unknown_code_and_shared_headers_fail_open(self):
        for changed in [
            None,
            ["new_component/code.cc"],
            ["include/shared.h"],
            ["ci/worker.py"],
        ]:
            with self.subTest(changed=changed):
                self.assertEqual(len(self.selected(changed=changed)["include"]), 2)

    def test_explicit_selection_overrides_paths_and_exclusion_wins(self):
        self.assertEqual(
            self.selected(changed=[], only=["gpu"])["include"][0]["target"], "gpu"
        )
        with self.assertRaises(ValueError):
            self.selected(only=["gpu"], exclude=["gpu"])
        with self.assertRaises(ValueError):
            self.selected(only=["typo"])
        with self.assertRaises(ValueError):
            self.selected(only=["off"])
        with self.assertRaises(ValueError):
            self.selected(only=["gpu"], target_names=["offline"])

    def test_missing_base_and_deleted_paths(self):
        self.assertIsNone(catalog.changed_files(None))
        self.assertIsNone(catalog.changed_files("0" * 40))
        self.assertIsNone(catalog.changed_files("nonexistent-ci-base"))
        self.assertEqual(
            self.selected(changed=["ep/deleted.cu"])["include"][0]["target"], "gpu"
        )

    def test_registry_audits_new_tests_and_capabilities(self):
        files = [p for t in self.catalog["tests"] for p in t["sources"]]
        catalog.validate(self.catalog, self.targets, files)
        with self.assertRaisesRegex(ValueError, "Unregistered"):
            catalog.validate(self.catalog, self.targets, files + ["new/test_x.py"])
        self.catalog["tests"][1]["nodes"] = 3
        with self.assertRaisesRegex(ValueError, "incompatible"):
            catalog.validate(self.catalog, self.targets)
        self.catalog["tests"][1]["nodes"] = 0
        with self.assertRaisesRegex(ValueError, "nodes must be a positive"):
            catalog.validate(self.catalog, self.targets)
        self.catalog["tests"][1]["nodes"] = 1
        self.catalog["tests"][1]["gpus"] = -1
        with self.assertRaisesRegex(ValueError, "gpus must be a nonnegative"):
            catalog.validate(self.catalog, self.targets)

    def test_host_interfaces_follow_alias_when_ranks_are_reordered(self):
        target = self.targets["gpu"]
        target["interfaces"] = {"a": "eth-a", "b": "eth-b"}
        catalog.validate(self.catalog, self.targets)
        self.assertEqual(
            [catalog.target_interface(target, host) for host in ["b", "a"]],
            ["eth-b", "eth-a"],
        )
        with self.assertRaisesRegex(ValueError, "No network interface"):
            catalog.target_interface(target, "unknown")
        self.assertEqual(catalog.target_interface({"interface": "ib0"}, "a"), "ib0")

    def test_interface_inventory_rejects_missing_and_unknown_hosts(self):
        for interfaces in [
            {"a": "eth-a"},
            {"a": "eth-a", "typo": "eth-b"},
            {"a": "eth-a", "b": ""},
        ]:
            with self.subTest(interfaces=interfaces):
                self.targets["gpu"]["interfaces"] = interfaces
                with self.assertRaisesRegex(ValueError, "interface"):
                    catalog.validate(self.catalog, self.targets)

    def test_gh200_two_node_coverage_and_directory_exclusions(self):
        registry, targets = catalog.read_catalog()
        definitions = {test["id"]: test for test in registry["tests"]}
        default = catalog.plan(registry, targets, target_names=["gh200"])
        performance = catalog.plan(
            registry, targets, tiers=["performance"], target_names=["gh200"]
        )
        normal_ids = set(default["include"][0]["tests"])
        perf_ids = set(performance["include"][0]["tests"])
        self.assertTrue(
            {
                "ep.low_latency",
                "p2p.rdma_native.read",
                "p2p.rdma_native.write",
                "nixl.internode.rdma",
            }
            <= normal_ids
        )
        self.assertTrue(
            {"ep.bench.low_latency", "p2p.bandwidth", "p2p.bandwidth_gpu"} <= perf_ids
        )
        self.assertFalse(normal_ids & perf_ids)
        self.assertEqual(targets["gh200"]["hosts"], ["gh200_1", "gh200_0"])
        for name in normal_ids | perf_ids:
            self.assertFalse(
                any(
                    source.startswith(("examples/", "experimental/"))
                    for source in definitions[name]["sources"]
                ),
                name,
            )


class ProcessTests(unittest.TestCase):
    def test_exit_status_timeout_and_owned_descendant_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log"
            result = run([sys.executable, "-c", "raise SystemExit(17)"], log, 5)
            self.assertEqual(result["exit_code"], 17)
            result = run(
                [sys.executable, "-c", "import time; time.sleep(60)"], log, 0.2
            )
            self.assertEqual(result["exit_code"], 124)
            self.assertLess(result["seconds"], 6)
            # A successful parent must not leave its background child running.
            marker = Path(tmp) / "orphan"
            code = (
                'import subprocess,sys; subprocess.Popen([sys.executable,"-c",'
                '"import time,pathlib,sys; time.sleep(1); pathlib.Path(sys.argv[1]).touch()",'
                "sys.argv[1]])"
            )
            result = run([sys.executable, "-c", code, str(marker)], log, 5)
            self.assertEqual(result["exit_code"], 0)
            time.sleep(1.1)
            self.assertFalse(marker.exists())

    def test_disconnect_terminates_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker = Worker(
                [
                    sys.executable,
                    "-c",
                    "import sys,json; sys.path.insert(0,sys.argv[1]); from process import run; "
                    'print(json.dumps({"event":"ready"}),flush=True); '
                    'r=run([sys.executable,"-c","import time; time.sleep(60)"],sys.argv[2],60,watch_stdin=True); '
                    'print(json.dumps({"event":"result",**r}),flush=True)',
                    str(catalog.ROOT / "ci"),
                    str(Path(tmp) / "child.log"),
                ],
                Path(tmp) / "worker.log",
            )
            worker.receive("ready", timeout=5)
            worker.child.stdin.close()
            result = worker.receive("result", timeout=7)
            self.assertEqual(result["exit_code"], 125)
            worker.close()

    def test_failed_rank_cancels_peer(self):
        class Fake:
            def __init__(self, initial=None):
                self.events = queue.Queue()
                self.sent = []
                if initial:
                    self.events.put(initial)

            def send(self, message):
                self.sent.append(message)
                self.events.put(dict(event="result", id="t", exit_code=125))

        one = Fake(dict(event="result", id="t", exit_code=1))
        two = Fake()
        results = collect_results([one, two], "t", 2)
        self.assertEqual([r["exit_code"] for r in results], [1, 125])
        self.assertEqual(two.sent, [{"cancel": "t"}])

    def test_infrastructure_failure_is_junit_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            write_summary([], output, "SSH lost")
            self.assertIn("<error", (output / "junit.xml").read_text())
            self.assertEqual(
                json.loads((output / "results.json").read_text())[
                    "infrastructure_error"
                ],
                "SSH lost",
            )


class PerformanceTests(unittest.TestCase):
    def test_additional_benchmark_metrics(self):
        fixtures = {
            "ep.bench.pcie_bench": "CPU wait time: 100 us, bw: 12.5 GBps\n",
            "ep.bench.gpu_to_cpu_bench": "Block 0: Average latency = 2.0 us over 1000 iterations\nBlock 1: Average latency = 4.0 us over 1000 iterations\n",
            "ep.bench.batched_gpu_to_cpu_bench": "Overall avg GPU-measured latency  : 5.0 µs\nThroughput : 6.0 Mops/s\n",
            "p2p.bench.nccl": "[Server] 1 MiB : 999 Gbps\n[Client] 1 MiB : 7.0 Gbps | 0.875 GB/s\n",
            "p2p.bench.ray_p2p": "[Client/WRITE] 1 MiB : 8.0 Gbps | 1.0 GB/s\n[Client/READ] 1 MiB : 9.0 Gbps | 1.125 GB/s\n",
            "p2p.bench.uccl_compression": "[Client] 1 MiB : 10.0 Gbps | 1.25 GB/s\n",
            "ep.bench.pack_unpack": "Testing: Decode batch (128 tokens, 7168 hidden)\nTotal Time (Pack + Unpack):\n-----\n  Triton : 0.11 ms\nTesting: Prefill batch (4096 tokens, 7168 hidden)\nTotal Time (Pack + Unpack):\n-----\n  Triton : 0.22 ms\n",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "gh200"
            target.mkdir()
            (target / "results.json").write_text(
                json.dumps(
                    {
                        "results": [
                            dict(id=name, tier="performance", exit_code=0)
                            for name in fixtures
                        ]
                    }
                )
            )
            for name, content in fixtures.items():
                (target / (name + ".log")).write_text(content)
            result = evaluate(root, {})
            self.assertFalse(result["failures"])
            self.assertEqual(
                sorted(m["value"] for m in result["measurements"]),
                [0.11, 0.22, 3, 5, 6, 7, 8, 9, 10, 12.5],
            )
            # A missing decode result must not be replaced by the prefill result.
            path = target / "ep.bench.pack_unpack.log"
            path.write_text(
                fixtures["ep.bench.pack_unpack"].replace("  Triton : 0.11 ms\n", "")
            )
            self.assertTrue(
                any("decode_triton_ms" in f for f in evaluate(root, {})["failures"])
            )

    def test_report_uses_slowest_participating_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "gpu"
            target.mkdir()
            (target / "results.json").write_text(
                json.dumps(
                    {
                        "results": [
                            dict(id="pair", node=0, exit_code=0, seconds=2),
                            dict(id="pair", node=1, exit_code=1, seconds=3),
                            dict(id="single", node=0, exit_code=0, seconds=1),
                            dict(
                                id="single",
                                node=1,
                                exit_code=0,
                                seconds=99,
                                not_participating=True,
                            ),
                        ]
                    }
                )
            )
            output = io.StringIO()
            with redirect_stdout(output):
                report_results(root)
            self.assertIn("| gpu | `pair` | FAIL | 3.000 |", output.getvalue())
            self.assertIn("| gpu | `single` | PASS | 1.000 |", output.getvalue())

    def test_ep_latency_and_gpu_bandwidth_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "gh200"
            target.mkdir()
            (target / "results.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {"id": name, "tier": "performance", "exit_code": 0}
                            for name in ["ep.bench.low_latency", "p2p.bandwidth_gpu"]
                        ]
                    }
                )
            )
            (target / "ep.bench.low_latency.log").write_text(
                "[rank 0] Dispatch + combine bandwidth: 40.00 GB/s, avg_t=22.00 us, min_t=20.00 us, max_t=25.00 us\n"
                "[rank 1] Dispatch + combine bandwidth: 42.00 GB/s, avg_t=20.00 us, min_t=19.00 us, max_t=24.00 us\n"
            )
            (target / "p2p.bandwidth_gpu.log").write_text(
                "[Client] 1.0 MB : 300.00 Gbps | 37.50 GB/s | 27.96 us\n"
            )
            result = evaluate(root, {})
            self.assertEqual(result["failures"], [])
            measurements = {m["key"]: m for m in result["measurements"]}
            self.assertEqual(
                measurements["gh200/ep.bench.low_latency/dispatch_combine_us"]["value"],
                21,
            )
            self.assertEqual(
                measurements["gh200/p2p.bandwidth_gpu/bandwidth_gbps"]["value"], 300
            )
            self.assertTrue(
                all(m["status"] == "unbaselined" for m in measurements.values())
            )

    def test_thresholds_and_nonfinite(self):
        self.assertTrue(compare(90, dict(value=100, direction="higher", tolerance=0.1)))
        self.assertFalse(
            compare(89, dict(value=100, direction="higher", tolerance=0.1))
        )
        self.assertTrue(compare(109, dict(value=100, direction="lower", tolerance=0.1)))
        self.assertTrue(
            compare(100 / 0.9, dict(value=100, direction="lower", tolerance=0.1))
        )
        self.assertFalse(
            compare(111.12, dict(value=100, direction="lower", tolerance=0.1))
        )
        for value in [float("nan"), float("inf"), 0, -1]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                compare(value, dict(value=100, direction="lower"))
        for reference in [float("nan"), float("inf"), 0, -1]:
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                compare(100, dict(value=reference, direction="higher"))

    def test_missing_measurement_and_reviewed_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "gpu"
            target.mkdir()
            (target / "results.json").write_text(
                json.dumps(
                    dict(results=[dict(id="bench", tier="performance", exit_code=0)])
                )
            )
            definition = dict(
                tests=[
                    dict(
                        id="bench",
                        metrics=[
                            dict(
                                name="bw",
                                pattern=r"([0-9.]+) Gbps",
                                unit="Gbps",
                                direction="higher",
                            )
                        ],
                    )
                ]
            )
            with patch("performance.read_catalog", return_value=(definition, {})):
                self.assertTrue(evaluate(root, {})["failures"])
                (target / "bench.log").write_text("80 Gbps\n100 Gbps\n")
                result = evaluate(root, {})
                self.assertEqual(result["measurements"][0]["value"], 90)
                self.assertEqual(result["measurements"][0]["status"], "unbaselined")
                self.assertTrue(
                    evaluate(
                        root, {"gpu/bench/bw": dict(value=200, direction="higher")}
                    )["failures"]
                )


if __name__ == "__main__":
    unittest.main()
