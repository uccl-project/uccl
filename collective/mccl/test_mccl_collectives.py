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
ROW = "67108864 16777216 float sum -1 2196.6 30.55 30.55 0 2694.3 24.91 24.91 0"
ROWS = "\n".join(
    f"{2 ** power} {2 ** power // 4} " + ROW.split(" ", 2)[2] for power in range(3, 27)
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
            make_fake_test(root / binary, "UCCL-MCCL-RDMA correctness\n" + ROWS)

        passed = run_runner(root, plugin)
        assert passed.returncode == 0, passed.stderr
        assert passed.stdout.count("PASS") == 4

        make_fake_test(root / "alltoall_perf", "native MCCL fallback")
        rejected = run_runner(root, plugin, "--collective", "alltoall")
        assert rejected.returncode == 1
        assert "did not confirm" in rejected.stderr

        for invalid in (
            "UCCL-MCCL-RDMA PASS",
            "UCCL-MCCL-RDMA\n" + ROW.replace("30.55 0", "30.55 1"),
            "UCCL-MCCL-RDMA\n" + ROW,
        ):
            make_fake_test(root / "alltoall_perf", invalid)
            rejected = run_runner(root, plugin, "--collective", "alltoall")
            assert rejected.returncode == 1

        missing = root / "reduce_scatter_perf"
        missing.unlink()
        rejected = run_runner(root, plugin, "--collective", "reducescatter")
        assert rejected.returncode == 2
        assert "no test binary" in rejected.stderr


if __name__ == "__main__":
    main()
