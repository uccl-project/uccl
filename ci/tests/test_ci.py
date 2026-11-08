"""Regression tests for test selection and the controller's failure contract."""

import copy
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
    def test_thresholds_and_nonfinite(self):
        self.assertTrue(compare(90, dict(value=100, direction="higher", tolerance=0.1)))
        self.assertFalse(
            compare(89, dict(value=100, direction="higher", tolerance=0.1))
        )
        self.assertTrue(compare(109, dict(value=100, direction="lower", tolerance=0.1)))
        self.assertFalse(
            compare(111, dict(value=100, direction="lower", tolerance=0.1))
        )
        with self.assertRaises(ValueError):
            compare(float("nan"), dict(value=100, direction="lower"))

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
