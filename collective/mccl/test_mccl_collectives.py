#!/usr/bin/env python3
"""Host-only tests for the MCCL collective correctness runner."""

import os
import pathlib
import stat
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNNER = ROOT / "collective" / "mccl" / "run_mccl_correctness.py"
BINARIES = (
    "all_reduce_perf",
    "all_gather_perf",
    "reduce_scatter_perf",
    "alltoall_perf",
)


def make_fake_test(path, output):
    path.write_text(f"#!/bin/sh\nprintf '%s\\n' '{output}'\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def run_runner(tests_home, plugin, *extra):
    return subprocess.run(
        (
            sys.executable,
            str(RUNNER),
            "--tests-home",
            str(tests_home),
            "--plugin",
            str(plugin),
            "--no-launcher",
            "--iterations",
            "2",
            *extra,
        ),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )


def main():
    with tempfile.TemporaryDirectory(prefix="uccl-mccl-tests-") as tmp:
        root = pathlib.Path(tmp)
        plugin = root / "libmccl-net-uccl.so"
        plugin.touch()
        for binary in BINARIES:
            make_fake_test(root / binary, "UCCL-MCCL-RDMA correctness")

        passed = run_runner(root, plugin)
        assert passed.returncode == 0, passed.stderr
        assert passed.stdout.count("PASS") == 4

        make_fake_test(root / "alltoall_perf", "native MCCL fallback")
        rejected = run_runner(root, plugin, "--collective", "alltoall")
        assert rejected.returncode == 1
        assert "did not confirm" in rejected.stderr

        missing = root / "reduce_scatter_perf"
        missing.unlink()
        rejected = run_runner(root, plugin, "--collective", "reducescatter")
        assert rejected.returncode == 2
        assert "no test binary" in rejected.stderr


if __name__ == "__main__":
    main()
