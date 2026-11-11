"""Regression tests for test selection and the controller's failure contract."""

import copy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import catalog
import idle
import run as controller
from performance import candidate_baselines, compare, evaluate, gate_policy
from performance import main as performance_main
from process import run
from report import report_logs, report_results
from run import Worker, collect_results, write_summary


class AdmissionTests(unittest.TestCase):
    def process(self, pid, uid=1000, ppid=1, name="python3", **kwargs):
        return dict(
            pid=pid,
            uid=uid,
            ppid=ppid,
            pgid=pid,
            name=name,
            state="S",
            kernel=False,
            start="100",
            **kwargs,
        )

    def test_own_process_tree_is_excluded(self):
        snapshot = {
            1: self.process(1, uid=0, name="systemd"),
            10: self.process(10, name="bash"),
            20: self.process(20, ppid=10),
            21: self.process(21, ppid=20),
            30: dict(self.process(30), state="R"),
        }
        own = idle.owned_processes(snapshot, pid=20)
        self.assertEqual(own, {1, 10, 20, 21})
        self.assertNotIn(30, own)

    def test_cpu_activity_does_not_reserve_an_idle_gpu(self):
        snapshot = {
            10: dict(self.process(10, name="python"), state="R", ticks=10000),
            20: dict(self.process(20, name="jupyter"), ticks=10000),
            30: dict(self.process(30, name="mpirun"), state="D", ticks=10000),
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("idle.Path.read_text", return_value=""),
            patch("idle.processes", return_value=snapshot),
            patch("idle.owned_processes", return_value=set()),
            patch("idle.query", return_value=self.xml()),
        ):
            result = idle.probe(dict(gpus=1, capabilities=["cuda"]), tmp, lease=False)
        self.assertEqual(result["state"], "idle")

    def test_own_gpu_activity_does_not_hide_a_foreign_context(self):
        xml = self.xml(50, 100).replace(
            "</processes>", "<process_info><pid>60</pid></process_info></processes>"
        )
        with patch("idle.query", return_value=xml):
            self.assertEqual(
                idle.gpu_jobs(dict(gpus=1, capabilities=["cuda"]), {50}),
                ["GPU process PID 60"],
            )

    def xml(self, pid=None, utilization=0):
        processes = (
            "" if pid is None else f"<process_info><pid>{pid}</pid></process_info>"
        )
        return f"<nvidia_smi_log><gpu><processes>{processes}</processes><utilization><gpu_util>{utilization} %</gpu_util></utilization></gpu></nvidia_smi_log>"

    def test_nvidia_idle_foreign_and_own_contexts(self):
        target = dict(gpus=1, capabilities=["cuda"])
        with patch("idle.query", return_value=self.xml()):
            self.assertEqual(idle.gpu_jobs(target, set()), [])
        with patch("idle.query", return_value=self.xml(50, 0)):
            self.assertIn("PID 50", idle.gpu_jobs(target, set())[0])
        with patch("idle.query", return_value=self.xml(50, 100)):
            self.assertEqual(idle.gpu_jobs(target, {50}), [])

    def test_idle_desktop_context_does_not_reserve_gpu(self):
        xml = self.xml(50).replace("<pid>50</pid>", "<pid>50</pid><type>G</type>")
        with patch("idle.query", return_value=xml):
            self.assertEqual(
                idle.gpu_jobs(dict(gpus=1, capabilities=["cuda"]), set()), []
            )

    def test_gpu_activity_without_visible_context_is_busy(self):
        with patch("idle.query", return_value=self.xml(utilization=20)):
            self.assertTrue(
                idle.gpu_jobs(dict(gpus=1, capabilities=["cuda"]), set(), settle=False)
            )

    def test_recent_ci_utilization_settles_before_next_suite(self):
        with (
            patch("idle.query", side_effect=[self.xml(utilization=20), self.xml()]),
            patch("idle.time.sleep"),
        ):
            self.assertEqual(
                idle.gpu_jobs(dict(gpus=1, capabilities=["cuda"]), set()), []
            )

    def test_amd_kfd_and_utilization_checks(self):
        target = dict(gpus=1, capabilities=["rocm"])
        usage = json.dumps({"card0": {"GPU use (%)": "0"}})
        with patch("idle.query", side_effect=["No KFD PIDs currently running", usage]):
            self.assertEqual(idle.gpu_jobs(target, set()), [])
        with patch(
            "idle.query", side_effect=["PID 55 is using 1 DRM device(s)", usage]
        ):
            self.assertEqual(idle.gpu_jobs(target, set()), ["GPU process PID 55"])
        with patch("idle.query", return_value="unrecognized ROCm output"):
            with self.assertRaises(ValueError):
                idle.gpu_jobs(target, set())

    def test_unknown_activity_never_admits_a_node(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("idle.processes", side_effect=PermissionError),
        ):
            result = idle.probe(dict(gpus=0), tmp, lease=False)
            self.assertEqual(result["state"], "unavailable")
            self.assertTrue(idle.skip_reason(result))
        for error in (ValueError, AttributeError, IndexError):
            with (
                self.subTest(error=error),
                tempfile.TemporaryDirectory() as tmp,
                patch("idle.cleanup_stale", side_effect=error),
            ):
                result = idle.probe(dict(gpus=0), tmp, lease=False, cleanup=True)
                self.assertEqual(result["state"], "unavailable")

    def test_active_lease_prevents_cleanup(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("idle.fcntl.flock", side_effect=BlockingIOError),
            patch("idle.cleanup_stale") as cleanup,
        ):
            self.assertEqual(idle.probe(dict(gpus=0), tmp)["state"], "busy")
            cleanup.assert_not_called()

    def test_stale_cleanup_checks_pid_identity_and_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "owned-processes"
            records.mkdir()
            data = self.process(50, uid=os.getuid())
            data["run"] = str(root / "runs" / "old" / "src")
            record = records / "50.json"
            record.write_text(json.dumps(data))
            with (
                patch("idle.processes", return_value={50: dict(data, start="reused")}),
                patch("idle.os.killpg") as kill,
            ):
                self.assertEqual(idle.cleanup_stale(root), [])
                kill.assert_not_called()
            record.write_text(json.dumps(data))
            with (
                patch(
                    "idle.processes", return_value={50: dict(data, uid=os.getuid() + 1)}
                ),
                patch("idle.os.killpg") as kill,
            ):
                with self.assertRaises(ValueError):
                    idle.cleanup_stale(root)
                kill.assert_not_called()
            record.write_text(json.dumps(data))
            with (
                patch("idle.processes", return_value={50: data}),
                patch("idle.os.killpg") as kill,
                patch("idle.time.monotonic", side_effect=[0, 3]),
            ):
                self.assertEqual(idle.cleanup_stale(root), [50])
                self.assertEqual(
                    [c.args for c in kill.call_args_list],
                    [(50, signal.SIGTERM), (50, signal.SIGKILL)],
                )
            record.write_text(json.dumps(data))
            child = dict(self.process(51, uid=os.getuid(), ppid=50), pgid=50)
            with (
                patch(
                    "idle.processes", side_effect=[{50: data, 51: child}, {51: child}]
                ),
                patch("idle.os.killpg") as kill,
                patch("idle.time.monotonic", side_effect=[0, 3]),
            ):
                self.assertEqual(idle.cleanup_stale(root), [50])
                self.assertEqual(
                    [c.args for c in kill.call_args_list],
                    [(50, signal.SIGTERM), (50, signal.SIGKILL)],
                )

    def test_cpu_suite_on_gpu_node_still_checks_gpu_activity(self):
        with (
            patch("idle.shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("idle.query", return_value=self.xml(50)),
        ):
            self.assertEqual(
                idle.gpu_jobs(dict(gpus=0, capabilities=["cpu"]), set()),
                ["GPU process PID 50"],
            )

    def test_busy_peer_skips_target_before_upload_or_workers(self):
        definition = dict(id="t", tier="performance", nodes=2)
        target = dict(enabled=True, hosts=["a", "b"], gpus=1, capabilities=["cuda"])
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(sys, "argv", ["run.py", "--target", "gpu", "--output", tmp]),
            patch(
                "run.read_catalog",
                return_value=({"tests": [definition]}, {"gpu": target}),
            ),
            patch("run.validate"),
            patch("run.tracked_files", return_value=[]),
            patch(
                "run.plan",
                return_value={"include": [{"target": "gpu", "tests": ["t"]}]},
            ),
            patch("run.subprocess.check_output", side_effect=["sha\n", b""]),
            patch(
                "run.remote_preflight",
                return_value={
                    "a": {"state": "idle", "reason": ""},
                    "b": {"state": "busy", "reason": "real job"},
                },
            ),
            patch("run.archive_source") as archive,
            patch("run.Worker") as worker,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(controller.main(), 0)
            archive.assert_not_called()
            worker.assert_not_called()
            payload = json.loads((Path(tmp) / "gpu/results.json").read_text())
            self.assertIsNone(payload["infrastructure_error"])
            self.assertEqual(len(payload["results"]), 2)
            self.assertTrue(
                all(r["skip_reason"] == "b: real job" for r in payload["results"])
            )

    def test_unreachable_peer_skips_before_upload(self):
        definition = dict(id="t", tier="performance", nodes=2)
        target = dict(enabled=True, hosts=["a", "b"], gpus=1, capabilities=["cuda"])
        failures = [
            subprocess.CompletedProcess(
                [], 255, stdout="", stderr="Connection refused"
            ),
            subprocess.TimeoutExpired("ssh", 60),
        ]
        for failure in failures:
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as tmp,
                patch.object(
                    sys, "argv", ["run.py", "--target", "gpu", "--output", tmp]
                ),
                patch(
                    "run.read_catalog",
                    return_value=({"tests": [definition]}, {"gpu": target}),
                ),
                patch("run.validate"),
                patch("run.tracked_files", return_value=[]),
                patch(
                    "run.plan",
                    return_value={"include": [{"target": "gpu", "tests": ["t"]}]},
                ),
                patch("run.subprocess.check_output", side_effect=["sha\n", b""]),
                patch(
                    "run.subprocess.run",
                    side_effect=[
                        subprocess.CompletedProcess(
                            [], 0, stdout='{"state":"idle","reason":""}'
                        ),
                        failure,
                    ],
                ),
                patch("run.archive_source") as archive,
                patch("run.Worker") as worker,
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(controller.main(), 0)
                archive.assert_not_called()
                worker.assert_not_called()
                payload = json.loads((Path(tmp) / "gpu/results.json").read_text())
                self.assertIsNone(payload["infrastructure_error"])
                self.assertEqual(len(payload["results"]), 2)
                self.assertTrue(
                    all("b: SSH" in r["skip_reason"] for r in payload["results"])
                )
                admission = json.loads((Path(tmp) / "gpu/admission.json").read_text())
                self.assertEqual(admission["b"]["state"], "unavailable")
                self.assertFalse(evaluate(Path(tmp), {})["measurements"])
                self.assertFalse(evaluate(Path(tmp), {})["failures"])

    def test_remote_probe_errors_are_not_connection_skips(self):
        target = dict(gpus=1, capabilities=["cuda"])
        with patch(
            "run.subprocess.run",
            return_value=subprocess.CompletedProcess(
                [], 1, stdout="", stderr="Python setup failed"
            ),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                controller.remote_preflight(["ssh"], ["a"], target, "/ci")
            self.assertEqual(controller.unreachable_hosts(["ssh"], ["a"]), {})
        with (
            patch(
                "run.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    [], 0, stdout="not JSON", stderr=""
                ),
            ),
            self.assertRaises(ValueError),
        ):
            controller.remote_preflight(["ssh"], ["a"], target, "/ci")

    def test_connection_loss_preserves_observed_test_failure(self):
        class Fake:
            def __init__(self, message):
                self.events = queue.Queue()
                self.events.put(message)

            def send(self, message):
                pass

        for code in [0, 1, 125]:
            with self.subTest(code=code), tempfile.TemporaryDirectory() as tmp:
                partial = dict(
                    event="result",
                    id="t",
                    tier="performance",
                    target="gpu",
                    node=0,
                    exit_code=code,
                    seconds=1,
                )
                with self.assertRaises(controller.WorkerError) as caught:
                    collect_results(
                        [Fake(partial), Fake(dict(event="fatal", error="SSH lost"))],
                        "t",
                        1,
                    )
                results = caught.exception.results
                controller.mark_skipped(
                    results,
                    ["t"],
                    {"t": dict(tier="performance", nodes=2)},
                    "gpu",
                    "b: SSH unavailable",
                    unavailable=True,
                )
                self.assertEqual(len(results), 2)
                self.assertEqual(bool(results[0]["exit_code"]), code == 1)
                self.assertEqual(results[1]["skip_reason"], "b: SSH unavailable")
                write_summary(results, Path(tmp) / "gpu")
                report = evaluate(Path(tmp), {})
                self.assertEqual(bool(report["failures"]), code == 1)
                self.assertEqual(report["measurements"], [])

    def test_artifact_collection_ssh_loss_skips_target(self):
        definition = dict(id="t", tier="performance", nodes=1)
        target = dict(enabled=True, hosts=["a"], gpus=1, capabilities=["cuda"])
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(sys, "argv", ["run.py", "--target", "gpu", "--output", tmp]),
            patch(
                "run.read_catalog",
                return_value=({"tests": [definition]}, {"gpu": target}),
            ),
            patch("run.validate"),
            patch("run.tracked_files", return_value=[]),
            patch(
                "run.plan",
                return_value={"include": [{"target": "gpu", "tests": ["t"]}]},
            ),
            patch("run.subprocess.check_output", side_effect=["sha\n", b""]),
            patch(
                "run.remote_preflight",
                return_value={"a": dict(state="idle", reason="")},
            ),
            patch(
                "run.archive_source",
                side_effect=lambda path: path.write_bytes(b"archive"),
            ),
            patch("run.check"),
            patch("run.Worker"),
            patch(
                "run.receive_all",
                side_effect=[
                    [dict(address="a")],
                    [],
                    [dict(id="t")],
                    [dict(id="t", ok=True)],
                ],
            ),
            patch(
                "run.collect_results",
                return_value=[
                    dict(
                        id="t",
                        tier="performance",
                        target="gpu",
                        node=0,
                        exit_code=0,
                        seconds=1,
                    )
                ],
            ),
            patch(
                "run.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    [], 255, stderr="Connection closed"
                ),
            ),
            patch("run.unreachable_hosts", return_value={"a": "Connection refused"}),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(controller.main(), 0)
            payload = json.loads((Path(tmp) / "gpu/results.json").read_text())
            self.assertIsNone(payload["infrastructure_error"])
            self.assertIn(
                "SSH unavailable during artifact collection",
                payload["results"][0]["skip_reason"],
            )
            self.assertEqual(
                evaluate(Path(tmp), {}), dict(measurements=[], failures=[], warnings=[])
            )

    @unittest.skipUnless(
        sys.platform.startswith("linux"), "Linux process identity requires /proc"
    )
    def test_recorded_stale_group_cleanup_leaves_unrecorded_process_alive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "runs" / "old" / "src"
            work.mkdir(parents=True)
            command = [sys.executable, "-c", "import time; time.sleep(60)"]
            stale = subprocess.Popen(command, cwd=work, start_new_session=True)
            other = subprocess.Popen(command, start_new_session=True)
            try:
                record = idle.record_process(
                    stale.pid, {"CI_ROOT": str(root), "CI_REPO": str(work)}
                )
                with open(root / idle.GPU_LOCK_FILE, "a") as lease:
                    idle.fcntl.flock(lease, idle.fcntl.LOCK_EX | idle.fcntl.LOCK_NB)
                    self.assertEqual(idle.cleanup_stale(root), [stale.pid])
                stale.wait(timeout=5)
                self.assertFalse(record.exists())
                self.assertIsNone(other.poll())
            finally:
                for child in (stale, other):
                    if child.poll() is None:
                        child.terminate()
                    child.wait(timeout=5)

    def test_busy_peer_interrupts_admission_without_waiting_for_setup(self):
        class Fake:
            def __init__(self):
                self.events = queue.Queue()

        a, b = Fake(), Fake()
        b.events.put(dict(event="initialized", skip_reason="job started"))
        with self.assertRaises(controller.TargetSkipped):
            controller.receive_all([a, b], "initialized", timeout=0.5)

    def test_running_ci_yields_without_killing_external_process(self):
        external = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"]
        )
        try:
            with tempfile.TemporaryDirectory() as tmp:
                result = run(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    Path(tmp) / "log",
                    10,
                    guard=lambda: "external job started",
                    guard_interval=0.05,
                )
                self.assertEqual(result["exit_code"], 0)
                self.assertEqual(result["skip_reason"], "external job started")
                self.assertIsNone(external.poll())
        finally:
            external.terminate()
            external.wait()

    def test_skip_cancels_peer_and_never_becomes_a_performance_pass(self):
        class Fake:
            def __init__(self, initial=None):
                self.events = queue.Queue()
                if initial:
                    self.events.put(initial)

            def send(self, message):
                self.events.put(
                    dict(
                        event="result",
                        id="t",
                        exit_code=125,
                        skip_reason=message["skip_reason"],
                    )
                )

        values = collect_results(
            [
                Fake(dict(event="result", id="t", exit_code=0, skip_reason="busy")),
                Fake(),
            ],
            "t",
            2,
        )
        self.assertTrue(all(v["skip_reason"] == "busy" for v in values))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = controller.skip_results(
                ["t"], {"t": dict(tier="performance", nodes=2)}, "gpu", "busy"
            )
            write_summary(rows, root / "gpu")
            xml = (root / "gpu/junit.xml").read_text()
            self.assertIn('skipped="2"', xml)
            self.assertNotIn("<failure", xml)
            with patch(
                "performance.read_catalog", return_value=({"tests": [{"id": "t"}]}, {})
            ):
                self.assertEqual(
                    evaluate(root, {}),
                    {"measurements": [], "failures": [], "warnings": []},
                )
            output = io.StringIO()
            with redirect_stdout(output):
                report_results(root)
            self.assertIn("SKIP", output.getvalue())
            self.assertIn("| gpu | 0 | 0 | 2 | OK |", output.getvalue())


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
                    id="cpu",
                    enabled=True,
                    tier="cpu",
                    sources=["p2p/tests/test_cpu.py"],
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
                    reason="no GPU",
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

    def test_cpu_suites_cannot_be_assigned_to_lab_targets(self):
        self.catalog["tests"][0]["targets"] = ["gpu"]
        self.catalog["tests"][0]["requires"] = []
        with self.assertRaisesRegex(ValueError, "CPU suites must use only the cpu"):
            catalog.validate(self.catalog, self.targets)
        registered, _ = catalog.read_catalog()
        cpu_suites = [
            t for t in registered["tests"] if t["enabled"] and t["tier"] == "cpu"
        ]
        self.assertTrue(cpu_suites)
        self.assertTrue(all(t["targets"] == ["cpu"] for t in cpu_suites))

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
                    source.startswith("experimental/")
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
    def test_job_log_groups_show_tail_as_literal_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "cpu/node-0/example.log"
            path.parent.mkdir(parents=True)
            path.write_text(
                "old line\nsecond line\n::error::literal test output\nlast line\n"
            )
            output = io.StringIO()
            with redirect_stdout(output):
                report_logs(root, tail=2)
            text = output.getvalue()
            self.assertIn("::group::cpu/node-0/example.log", text)
            self.assertIn("Showing last 2 of 4 lines", text)
            self.assertNotIn("old line", text)
            self.assertIn("last line", text)
            self.assertLess(text.index("::stop-commands::"), text.index("::error::"))
            self.assertTrue(text.endswith("::endgroup::\n"))

    def test_hard_and_soft_gate_exit_status_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "gh200"
            target.mkdir()
            name = "p2p.bench.nccl"
            key = f"gh200/{name}/bandwidth_gbps"
            (target / "results.json").write_text(
                json.dumps(
                    dict(
                        results=[
                            dict(id=name, tier="performance", exit_code=0, seconds=1)
                        ]
                    )
                )
            )
            (target / f"{name}.log").write_text("[Client] 1 MiB : 67.37 Gbps\n")
            baseline_file = root / "baseline.json"
            candidate_file = root / "candidate.json"
            for gate, code, status in [
                ("hard", 1, "regression"),
                ("soft", 0, "warning"),
            ]:
                with self.subTest(gate=gate):
                    baseline = dict(
                        value=79.71,
                        direction="higher",
                        tolerance=0.1,
                        gate=gate,
                        reason="Short, noisy calibration.",
                    )
                    baseline_file.write_text(
                        json.dumps(dict(baselines={key: baseline}))
                    )
                    args = [
                        "performance.py",
                        str(root),
                        "--baselines",
                        str(baseline_file),
                        "--candidate",
                        str(candidate_file),
                    ]
                    with (
                        patch.object(sys, "argv", args),
                        redirect_stdout(io.StringIO()),
                    ):
                        self.assertEqual(performance_main(), code)
                    result = json.loads((root / "performance.json").read_text())
                    metric = result["measurements"][0]
                    self.assertEqual(metric["status"], status)
                    self.assertEqual(metric["gate"], gate)
                    self.assertAlmostEqual(metric["threshold"], 71.739)
                    self.assertEqual(bool(result["failures"]), gate == "hard")
                    self.assertEqual(bool(result["warnings"]), gate == "soft")
                    candidate = json.loads(candidate_file.read_text())["baselines"][key]
                    self.assertEqual(candidate["gate"], gate)
                    self.assertEqual(
                        json.loads(baseline_file.read_text())["baselines"][key],
                        baseline,
                    )
                    output = io.StringIO()
                    with redirect_stdout(output):
                        report_results(root)
                    self.assertIn(
                        "advisory" if gate == "soft" else "required", output.getvalue()
                    )
                    self.assertIn(
                        "WARNING:" if gate == "soft" else "FAILED:", output.getvalue()
                    )

    def test_soft_latency_warns_only_above_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "gb10"
            target.mkdir()
            name = "ep.bench.gpu_to_cpu_bench"
            (target / "results.json").write_text(
                json.dumps(
                    dict(results=[dict(id=name, tier="performance", exit_code=0)])
                )
            )
            baselines = {
                f"gb10/{name}/latency_us": dict(
                    value=1,
                    direction="lower",
                    gate="soft",
                    reason="Short latency probe.",
                )
            }
            for value, status in [(1, "pass"), (2, "warning")]:
                (target / f"{name}.log").write_text(
                    f"Block 0: Average latency = {value} us over 1000 iterations\n"
                )
                result = evaluate(root, baselines)
                self.assertEqual(result["measurements"][0]["status"], status)
                self.assertFalse(result["failures"])
                self.assertEqual(bool(result["warnings"]), status == "warning")

    def test_soft_gate_does_not_hide_benchmark_or_measurement_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "gh200"
            target.mkdir()
            name = "p2p.bench.nccl"
            baselines = {
                f"gh200/{name}/bandwidth_gbps": dict(
                    value=100,
                    direction="higher",
                    gate="soft",
                    reason="Short benchmark.",
                )
            }
            cases = [
                (0, None, ""),
                (0, None, "0"),
                (0, None, "nan"),
                (0, None, "inf"),
                (1, None, "80"),
                (0, "SSH disconnected", "80"),
            ]
            for code, error, value in cases:
                with self.subTest(code=code, error=error, value=value):
                    (target / "results.json").write_text(
                        json.dumps(
                            dict(
                                results=[
                                    dict(id=name, tier="performance", exit_code=code)
                                ],
                                infrastructure_error=error,
                            )
                        )
                    )
                    (target / f"{name}.log").write_text(
                        f"[Client] 1 MiB : {value} Gbps\n"
                    )
                    self.assertTrue(evaluate(root, baselines)["failures"])

    def test_gate_validation_and_new_candidate_requires_review(self):
        self.assertEqual(gate_policy({}), "hard")
        for baseline in [
            dict(gate="sof"),
            dict(gate="soft"),
            dict(gate="soft", reason=" "),
            dict(gate="soft", reason=123),
        ]:
            with self.subTest(baseline=baseline), self.assertRaises(ValueError):
                gate_policy(baseline)
        metric = dict(key="gpu/new/bw", value=100, direction="higher", unit="Gbps")
        candidate = candidate_baselines([metric], {})[metric["key"]]
        self.assertEqual(candidate["gate"], "soft")
        self.assertIn("review", candidate["reason"])

    def test_ring_benchmarks_compare_the_same_configuration(self):
        fixtures = {
            "ep.rb.cas": "Warps: 64, Rings: 4, Proxies: 4, Throughput: 99 Mops/sec\nWarps: 128, Rings: 4, Proxies: 4, Throughput: 1.2 Mops/sec\nWarps: 128, Rings: 8, Proxies: 8, Throughput: 99 Mops/sec\n",
            "ep.rb.fc": "   64 | 99\n  128 | 2.3\n  256 | 99\n",
            "ep.rb.lock": "===== Test Results =====\nConfiguration:\n  Warps: 128\n  Proxies: 4\nGPU Side:\n  Throughput: 99 Mops/sec\nCPU Side:\n  Total processed: 123\n  Throughput: 3.4 Mops/sec\n===== Test Results =====\nConfiguration:\n  Warps: 256\n  Proxies: 4\nCPU Side:\n  Total processed: 456\n  Throughput: 99 Mops/sec\n",
            "ep.rb.multi_ring": "Warps: 64, Proxies: 4, Rings/proxy: 16, Throughput: 99 Mops/sec\nWarps: 128, Proxies: 4, Rings/proxy: 32, Throughput: 4.5 Mops/sec\n",
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
                sorted(m["value"] for m in result["measurements"]), [1.2, 2.3, 3.4, 4.5]
            )
            # A missing selected case must not borrow a different warp count.
            for name, content in fixtures.items():
                (target / (name + ".log")).write_text(content.replace("128", "512"))
            self.assertEqual(len(evaluate(root, {})["failures"]), 4)

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
