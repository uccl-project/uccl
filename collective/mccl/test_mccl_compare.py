#!/usr/bin/env python3
"""End-to-end collection fixtures and a real escaped-stdout timeout regression."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from mccl_process import run_command
from bench_mccl_compare import capture_environment

ROOT = Path(__file__).resolve().parent


class CompareTest(unittest.TestCase):
    def test_collect_and_analyze(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plugin = root / "libmccl-net-uccl.so"
            plugin.touch()
            binary = root / "all_reduce_perf"
            binary.write_text(
                f"#!{sys.executable}\nimport os\n"
                "print('Selected UCCL-MCCL-RDMA' if "
                "os.environ['MCCL_NET'] == 'UCCL-MCCL-RDMA' else 'Using network IB')\n"
                "print('8 2 float sum -1 10 1 1 0 11 1 1 0')\n"
            )
            binary.chmod(0o755)
            output = root / "evidence"
            command = [
                sys.executable,
                str(ROOT / "bench_mccl_compare.py"),
                "--tests-home",
                str(root),
                "--plugin",
                str(plugin),
                "--native-net",
                "IB",
                "--native-proof",
                "Using network IB",
                "--collective",
                "allreduce",
                "--no-launcher",
                "--min-bytes",
                "8",
                "--max-bytes",
                "8",
                "--repeats",
                "5",
                "--output",
                str(output),
            ]
            env = {
                **os.environ,
                "UCCL_NUM_ENGINES": "2",
                "MCCL_ALGO": "Ring",
                "UCCL_RCMODE": "1",
                "UCCL_PIN_TO_NUMA": "0",
                "UCCL_ROCE_TRAFFIC_CLASS": "8",
            }
            subprocess.run(command, env=env, check=True, timeout=30)
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["state"], "unverified")
            self.assertEqual(len(manifest["records"]), 10)
            self.assertEqual(manifest["fixed_environment"]["UCCL_NUM_ENGINES"], "2")
            for key in ("UCCL_RCMODE", "UCCL_PIN_TO_NUMA", "UCCL_ROCE_TRAFFIC_CLASS"):
                self.assertEqual(manifest["fixed_environment"][key], env[key])
            self.assertIn("LD_LIBRARY_PATH", manifest["records"][0]["environment"])
            self.assertEqual(capture_environment({"MCCL_SECRET": "private"}), {})
            analyze = [
                sys.executable,
                str(ROOT / "analyze_mccl_bench.py"),
                str(manifest_path),
            ]
            result = subprocess.run(analyze, text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["comparison"][0]["pairs"], 5)
            manifest["records"].pop()
            manifest_path.write_text(json.dumps(manifest))
            result = subprocess.run(analyze, capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            # Existing output directories cannot silently mix evidence.
            self.assertNotEqual(
                subprocess.run(command, capture_output=True, timeout=10).returncode, 0
            )

    def test_escaped_stdout_has_bounded_timeout(self):
        command = [
            sys.executable,
            "-c",
            "import subprocess,sys; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],"
            "start_new_session=True); print(p.pid,flush=True)",
        ]
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "child.pid"
            command[-1] += f"; open({str(pid_file)!r},'w').write(str(p.pid))"
            start = time.monotonic()
            try:
                status, output, timed_out = run_command(command, os.environ.copy(), 0.2)
                self.assertTrue(timed_out)
                self.assertLess(time.monotonic() - start, 18)
                self.assertEqual(status, 0)
            finally:
                if pid_file.exists():
                    try:
                        os.killpg(int(pid_file.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_same_group_child_without_inherited_stdout_is_killed(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "child.pid"
            heartbeat = Path(directory) / "heartbeat"
            child = (
                "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                f"f=open({str(heartbeat)!r},'w'); "
                "print('ready',flush=True); "
                "exec('while True:\\n f.write(\"x\"); f.flush(); time.sleep(.02)')"
            )
            command = [
                sys.executable,
                "-c",
                "import subprocess,sys,time; "
                f"p=subprocess.Popen([sys.executable,'-c',{child!r}],"
                "stdout=subprocess.PIPE,stderr=subprocess.DEVNULL); "
                f"open({str(pid_file)!r},'w').write(str(p.pid)); "
                "p.stdout.readline(); p.stdout.close(); time.sleep(60)",
            ]
            try:
                _, _, timed_out = run_command(command, os.environ.copy(), 0.5)
                self.assertTrue(timed_out)
                time.sleep(0.1)
                length = heartbeat.stat().st_size
                time.sleep(0.2)
                self.assertEqual(heartbeat.stat().st_size, length)
            finally:
                if pid_file.exists():
                    try:
                        os.kill(int(pid_file.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main()
