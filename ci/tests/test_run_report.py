"""Tests for the cross-target CI report rendered from untrusted artifacts."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import catalog
import run_report

CONFIG = Path(__file__).resolve().parents[1] / "config"

RUN_URL = "https://github.com/uccl-project/uccl/actions/runs/9"
TARGETS = {
    "cpu": {"enabled": True, "hosts": []},
    "l4": {"enabled": True, "hosts": ["labhost0", "labhost1"], "interface": "ethlab0"},
    "gh200": {"enabled": True, "hosts": ["labhost2", "labhost3"]},
    "gb10": {"enabled": True, "hosts": ["labhost4"]},
}
SUITES = [
    "ep.a",
    "ep.b",
    "ep.c",
    "ep.d",
    "p2p.e",
    "p2p.f",
    "collective.g",
    "ci.selftest",
]


def result(name, code=0, node=0, seconds=12.0, **extra):
    return {"id": name, "exit_code": code, "node": node, "seconds": seconds, **extra}


class Fixture:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = self.root / "config"
        self.config.mkdir()
        (self.config / "targets.json").write_text(json.dumps(TARGETS))
        tests = [{"id": name} for name in SUITES]
        (self.config / "tests.json").write_text(json.dumps({"tests": tests}))

    def run(self, name, plan=None, results=None, infra=None, logs=None, perf=None):
        root = self.root / name
        root.mkdir(parents=True)
        if plan is not None:
            (root / "ci-plan").mkdir()
            include = [{"target": t, "tests": tests} for t, tests in plan.items()]
            (root / "ci-plan" / "ci-plan.json").write_text(
                json.dumps({"schema": 1, "include": include})
            )
        for target, records in (results or {}).items():
            art = root / ("ci-cpu" if target == "cpu" else f"ci-gpu-{target}")
            (art / target).mkdir(parents=True)
            data = (
                records
                if isinstance(records, str)
                else json.dumps(
                    {
                        "results": records,
                        "infrastructure_error": (infra or {}).get(target),
                    }
                )
            )
            (art / target / "results.json").write_text(data)
        for (target, node, filename), text in (logs or {}).items():
            art = root / ("ci-cpu" if target == "cpu" else f"ci-gpu-{target}")
            path = art / target / "labhost0" / f"node-{node}" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        for target, data in (perf or {}).items():
            (root / f"ci-gpu-{target}" / "performance.json").write_text(
                json.dumps(data)
            )
        return root

    def render(self, artifacts, baseline=None, jobs=None):
        argv = [str(artifacts), "--config", str(self.config), "--run-url", RUN_URL]
        if baseline is not None:
            argv += ["--baseline", str(baseline)]
        if jobs is not None:
            argv += ["--jobs-env", "TEST_JOBS"]
        out = self.root / "out.md"
        with patch.dict(os.environ, {"TEST_JOBS": json.dumps(jobs or {})}):
            run_report.main(argv + ["--out", str(out)])
        return out.read_text()


class RunReportTests(unittest.TestCase):
    def setUp(self):
        self.f = Fixture()
        self.addCleanup(self.f.tmp.cleanup)

    def test_classifies_new_old_not_compared_timeout_and_signal(self):
        plan = {"l4": ["ep.a", "ep.b", "ep.c", "ep.d", "p2p.e"]}
        current = self.f.run(
            "current",
            plan,
            {
                "l4": [
                    result("ep.a", 1),
                    result("ep.b", 1),
                    result("ep.c", 124, seconds=900),
                    result("ep.d"),
                    result("p2p.e", -11, node=1),
                    result("p2p.e", 0, node=0, not_participating=True),
                ]
            },
        )
        base = self.f.root / "base"
        self.f.run(
            "base/100",
            plan,
            {
                "l4": [
                    result("ep.a"),
                    result("ep.b", 1),
                    result("p2p.e", skip_reason="GPU process PID 1"),
                ]
            },
        )
        self.f.run("base/90", plan, {"l4": [result("ep.c", 1)]})
        body = self.f.render(current, base)
        self.assertTrue(body.startswith("**[FAILED]**"))
        self.assertIn("1/5 executed suites passed", body)
        self.assertIn("**New:** <code>ep.a</code> (exit 1 after 12 s)", body)
        self.assertIn("**Old:** <code>ep.b</code>", body)
        self.assertIn("**Old:** <code>ep.c</code> (timed out after 900 s)", body)
        self.assertIn(
            "**Not compared:** <code>p2p.e</code> (killed by SIGSEGV after 12 s)", body
        )
        self.assertIn("[#100]", body)
        self.assertIn("[#90]", body)

    def test_skips_infra_and_missing_results_use_fixed_labels(self):
        plan = {"l4": ["ep.a"], "gh200": ["ep.a", "ep.b"], "gb10": ["p2p.e"]}
        current = self.f.run(
            "current",
            plan,
            {
                "gh200": [
                    result(
                        "ep.a",
                        skip_reason="labhost2: SSH unavailable: connect to host 10.1.2.3 port 22",
                    ),
                    result(
                        "ep.b",
                        skip_reason="labhost2: SSH unavailable: connect to host 10.1.2.3 port 22",
                    ),
                ],
                "gb10": [],
            },
            infra={
                "gb10": "Expected initialized, got {'log': '/ssd1/uccl-ci/runs/x/results/node-1/environment.log'}"
            },
            logs={
                (
                    "gb10",
                    1,
                    "environment.log",
                ): "Collecting x\nERROR: [Errno 28] No space left on device: '/ssd1/cache/labhost4'\n"
            },
        )
        body = self.f.render(current)
        self.assertTrue(body.startswith("**[INFRA ERROR]**"))
        self.assertIn("**Node unreachable:** 2 suites not run", body)
        self.assertIn("**No results uploaded:** 1 suite not run", body)
        self.assertIn("**Infrastructure error:** 1 suite not run", body)
        self.assertIn("environment setup failed on node 1", body)
        self.assertIn("No space left on device", body)
        for secret in ("10.1.2.3", "labhost2", "labhost4", "/ssd1", "SSH unavailable"):
            self.assertNotIn(secret, body)

    def test_busy_only_run_is_incomplete_not_success(self):
        current = self.f.run(
            "current",
            {"cpu": ["ci.selftest"], "l4": ["ep.a"]},
            {
                "cpu": [result("ci.selftest")],
                "l4": [
                    result(
                        "ep.a",
                        skip_reason="GPU process PID 4; GPU utilization is nonzero",
                    )
                ],
            },
        )
        body = self.f.render(current)
        self.assertTrue(body.startswith("**[INCOMPLETE]**"))
        self.assertIn("**GPU busy:** 1 suite not run", body)
        self.assertNotIn("PID", body)

    def test_success(self):
        current = self.f.run(
            "current", {"cpu": ["ci.selftest"]}, {"cpu": [result("ci.selftest")]}
        )
        body = self.f.render(current)
        self.assertTrue(body.startswith("**[SUCCESS]**"))
        self.assertNotIn("no main baseline", body)
        self.assertNotIn("<details>", body)

    def test_untrusted_log_text_is_redacted_and_cannot_escape_its_fence(self):
        log = (
            "setup\n/var/lib/uccl-ci/runs/gha-1-l4/src/p2p/engine.cc:10: error: boom ``` @maintainer see #123 "
            "<!-- uccl-ci-status --> from 192.168.0.1 on labhost0 via ethlab0, mail user@example.com\n"
        )
        current = self.f.run(
            "current",
            {"l4": ["p2p.e"]},
            {"l4": [result("p2p.e", 2)]},
            logs={("l4", 0, "p2p.e.log"): log},
        )
        body = self.f.render(current)
        self.assertIn("p2p/engine.cc:10: error: boom", body)
        self.assertIn("````text", body)
        for raw in (
            "@maintainer",
            "192.168.0.1",
            "labhost0",
            "ethlab0",
            "/var/lib",
            "user@example.com",
            "<!-- uccl-ci-status -->",
        ):
            self.assertNotIn(raw, body)

    def test_malformed_ids_and_unreadable_results_are_not_rendered(self):
        current = self.f.run(
            "current",
            {"l4": ["ep.a"], "gh200": ["ep.a"], "cpu": ["ep.a", "bad|<b>id"]},
            {
                "l4": [result("ep.a"), result("ep.a|<script>", 1)],
                "gh200": "not json",
                "cpu": [result("ep.a")],
            },
        )
        body = self.f.render(current)
        self.assertIn("1 result record(s) with malformed suite IDs were ignored", body)
        self.assertNotIn("<script>", body)
        self.assertNotIn("bad|", body)
        self.assertIn("**Infrastructure error:** 1 suite not run", body)

    def test_new_suite_is_kept_and_labelled(self):
        current = self.f.run(
            "current", {"l4": ["ep.brand_new"]}, {"l4": [result("ep.brand_new", 1)]}
        )
        body = self.f.render(current)
        self.assertIn("<code>ep.brand_new</code> (new suite)", body)
        self.assertIn("no main baseline available", body)

    def test_large_reports_stay_under_the_size_limit(self):
        names = [f"p2p.case_{i:04d}_{'x' * 60}" for i in range(3000)]
        current = self.f.run(
            "current",
            {"l4": names},
            {"l4": [result(n, 1) for n in names]},
            logs={
                ("l4", 0, f"{n}.log"): "E   AssertionError: " + "y" * 400
                for n in names[:40]
            },
        )
        body = self.f.render(current)
        self.assertLessEqual(len(body), run_report.MAX_REPORT)
        self.assertIn("… and 2997 more", body)
        self.assertIn("… and 2975 more; see the run artifacts.", body)

    def test_missing_plan_falls_back_to_job_results(self):
        current = self.f.run("current")
        body = self.f.render(
            current, jobs={"plan": {"result": "failure"}, "cpu": {"result": "skipped"}}
        )
        self.assertTrue(body.startswith("**[FAILED]**"))
        self.assertIn("planning failed", body)
        self.assertIn("| plan | failure |", body)

    def test_performance_gates_show_only_non_passing_metrics(self):
        current = self.f.run(
            "current",
            {"l4": ["p2p.e"]},
            {"l4": [result("p2p.e")]},
            perf={
                "l4": {
                    "measurements": [
                        {
                            "key": "l4/p2p.e/write",
                            "value": 21.5,
                            "unit": "GB/s",
                            "status": "warning",
                            "gate": "soft",
                        },
                        {
                            "key": "l4/p2p.e/read",
                            "value": 40.1,
                            "unit": "GB/s",
                            "status": "pass",
                            "gate": "hard",
                        },
                        {
                            "key": "l4/p2p.e/new",
                            "value": 3.0,
                            "unit": "us",
                            "status": "unbaselined",
                            "gate": "unbaselined",
                        },
                    ],
                    "failures": [],
                    "warnings": ["l4/p2p.e/write: 21.5 GB/s violates soft threshold"],
                }
            },
        )
        body = self.f.render(current)
        self.assertTrue(body.startswith("**[SUCCESS]**"))
        self.assertIn("### Performance gates not passing", body)
        self.assertIn(
            "<code>l4/p2p.e/write</code> | 21.5 <code>GB/s</code> | advisory | 🟡 warning",
            body,
        )
        self.assertIn("1 performance metric(s) have no reviewed baseline yet", body)
        self.assertNotIn("l4/p2p.e/read", body)

    def test_required_performance_failure_fails_the_run(self):
        current = self.f.run(
            "current",
            {"l4": ["p2p.e"]},
            {"l4": [result("p2p.e")]},
            perf={
                "l4": {
                    "measurements": [
                        {
                            "key": "l4/p2p.e/read",
                            "value": 9.0,
                            "unit": "GB/s",
                            "status": "regression",
                            "gate": "hard",
                        },
                    ],
                    "failures": [
                        "l4/p2p.e/read: 9.0 GB/s violates hard threshold 36 GB/s"
                    ],
                    "warnings": [],
                }
            },
        )
        body = self.f.render(current)
        self.assertTrue(body.startswith("**[FAILED]**"))
        self.assertIn("| required | ❌ regression |", body)
        self.assertNotIn("missing or invalid", body)
        bad = self.f.run(
            "bad",
            {"l4": ["p2p.e"]},
            {"l4": [result("p2p.e")]},
            perf={
                "l4": {
                    "measurements": [],
                    "failures": ["l4/p2p.e/read: missing measurement"],
                }
            },
        )
        body = self.f.render(bad)
        self.assertTrue(body.startswith("**[FAILED]**"))
        self.assertIn("L4: 1 benchmark measurement(s) missing or invalid", body)

    def test_rejects_untrusted_urls(self):
        current = self.f.run(
            "current", {"cpu": ["ci.selftest"]}, {"cpu": [result("ci.selftest")]}
        )
        for extra in (
            ["--repo-url", "https://evil.example/x/y"],
            ["--run-url", "https://evil.example/run"],
        ):
            argv = [
                str(current),
                "--config",
                str(self.f.config),
                "--run-url",
                RUN_URL,
            ] + extra
            with self.assertRaises(SystemExit), patch("sys.stderr"):
                run_report.main(argv)

    def test_pull_request_run_without_gpu_dispatch_is_not_an_infra_error(self):
        plan = {"cpu": ["ci.selftest"], "l4": ["ep.a"], "gh200": ["ep.b"]}
        current = self.f.run("current", plan, {"cpu": [result("ci.selftest")]})
        jobs = {"plan": {"result": "success"}, "gpu": {"result": "skipped"}}
        body = self.f.render(current, jobs=jobs)
        self.assertTrue(body.startswith("**[SUCCESS]**"))
        self.assertIn("1/1 executed suites passed", body)
        self.assertIn("GPU job skipped, so L4, GH200 did not run", body)
        self.assertNotIn("No results uploaded", body)
        jobs["gpu"]["result"] = "failure"
        self.assertTrue(
            self.f.render(current, jobs=jobs).startswith("**[INFRA ERROR]**")
        )

    def test_failed_job_with_passing_suites_is_not_success(self):
        # For example ci/performance.py raised before writing performance.json.
        current = self.f.run("current", {"l4": ["p2p.e"]}, {"l4": [result("p2p.e")]})
        jobs = {
            "plan": {"result": "success"},
            "cpu": {"result": "skipped"},
            "gpu": {"result": "failure"},
        }
        body = self.f.render(current, jobs=jobs)
        self.assertTrue(body.startswith("**[FAILED]**"))
        self.assertIn("1/1 executed suites passed; GPU job failed", body)
        self.assertNotIn("no main baseline", body)

    def test_cancelled_job_is_not_success(self):
        current = self.f.run(
            "current", {"cpu": ["ci.selftest"]}, {"cpu": [result("ci.selftest")]}
        )
        body = self.f.render(current, jobs={"cpu": {"result": "cancelled"}})
        self.assertTrue(body.startswith("**[FAILED]**"))
        self.assertIn("1/1 executed suites passed; CPU job cancelled", body)

    def test_failed_job_without_artifact_is_named(self):
        current = self.f.run("current", {"cpu": ["ci.selftest"]})
        body = self.f.render(current, jobs={"cpu": {"result": "failure"}})
        self.assertTrue(body.startswith("**[INFRA ERROR]**"))
        self.assertIn("1 of 1 planned suites did not run; CPU job failed", body)
        self.assertIn("**No results uploaded:** 1 suite not run", body)

    def test_successful_jobs_leave_the_report_unchanged(self):
        current = self.f.run(
            "current",
            {"cpu": ["ci.selftest"], "l4": ["p2p.e"]},
            {"cpu": [result("ci.selftest")], "l4": [result("p2p.e")]},
            perf={
                "l4": {
                    "measurements": [
                        {"key": "l4/p2p.e/read", "value": 40.1, "status": "pass"}
                    ],
                    "failures": [],
                }
            },
        )
        jobs = {name: {"result": "success"} for name in ("plan", "cpu", "gpu")}
        body = self.f.render(current, jobs=jobs)
        self.assertTrue(body.startswith("**[SUCCESS]**"))
        self.assertEqual(body, self.f.render(current))

    def test_run_without_selected_suites(self):
        current = self.f.run("current", {})
        body = self.f.render(current)
        self.assertTrue(body.startswith("**[SUCCESS]**"))
        self.assertIn("no suites were selected", body)
        self.assertNotIn("### Suites", body)

    def test_configured_hosts_and_interfaces_are_redacted(self):
        # Resolve names the way the runner does, so a renamed config key fails here.
        targets = json.loads((CONFIG / "targets.json").read_text())
        names = set()
        for target in targets.values():
            for host in target.get("hosts", []):
                names.add(host)
                try:
                    names.add(catalog.target_interface(target, host))
                except (KeyError, ValueError):
                    pass
        self.assertTrue(names)
        redact = run_report.Redactor(targets)
        for name in names:
            self.assertNotIn(name, redact(f"ssh uccl@{name}: {name}/mlx failed"))

    def test_host_names_are_replaced_only_as_whole_words(self):
        redact = run_report.Redactor({"x": {"hosts": ["node1"], "interface": "eth9"}})
        self.assertEqual(
            redact("node1x on node1 via eth9"), "node1x on <host> via <host>"
        )
        self.assertEqual(run_report.Redactor({})("node1"), "node1")

    def test_target_labels_follow_the_config(self):
        self.assertEqual(run_report.label("cpu"), "CPU (GitHub)")
        self.assertEqual(run_report.label("amd"), "AMD")


if __name__ == "__main__":
    unittest.main()
