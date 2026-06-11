"""Per-primitive correctness tests via the C++ microbench.

Set UCCL_GIN_RUN_PRIMITIVES=1 to run, plus:
  UCCL_GIN_ROOT          path to experimental/uccl_gin
  UCCL_GIN_MPI_HOSTS     e.g. 172.31.70.225:8,172.31.71.140:8
  LOCAL_WORLD_SIZE       ranks per node (default 8)
  UCCL_GIN_MPIRUN        path to mpirun (default /opt/amazon/openmpi/bin/mpirun)
  UCCL_GIN_TEST_SIZES    comma-separated bytes (default 1024,4096,65536,262144,1048576)
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Sequence


_ENV_EXPORTS = (
    "LD_LIBRARY_PATH", "NCCL_NET_PLUGIN", "FI_PROVIDER",
    "FI_EFA_USE_DEVICE_RDMA", "OFI_NCCL_FORCE_NUM_RAILS",
    "NCCL_SOCKET_IFNAME", "LOCAL_WORLD_SIZE",
)


def _mpirun_cmd(root: Path, binary: str, extra_args: Sequence[str] = ()) -> list[str]:
    hosts = os.environ["UCCL_GIN_MPI_HOSTS"]
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
    mpirun = os.environ.get("UCCL_GIN_MPIRUN", "/opt/amazon/openmpi/bin/mpirun")
    cmd = [
        mpirun, "--host", hosts,
        "-np", str(local_world * 2), "-npernode", str(local_world),
    ]
    for key in _ENV_EXPORTS:
        if key in os.environ:
            cmd.extend(["-x", key])
    cmd.append(str(root / "build" / binary))
    if extra_args:
        cmd.extend(str(a) for a in extra_args)
    return cmd


def _run(root: Path, binary: str, extra_args: Sequence[str] = ()) -> str:
    cmd = _mpirun_cmd(root, binary, extra_args)
    proc = subprocess.run(
        cmd, cwd=root, env=dict(os.environ),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=int(os.environ.get("UCCL_GIN_TEST_TIMEOUT", "120")),
    )
    if proc.returncode != 0:
        raise AssertionError(f"mpirun failed (rc={proc.returncode})\n{proc.stdout}")
    return proc.stdout


def _default_sizes() -> str:
    return os.environ.get("UCCL_GIN_TEST_SIZES", "1024,4096,65536,262144,1048576")


def test_put_red_add_correctness() -> None:
    if os.environ.get("UCCL_GIN_RUN_PRIMITIVES") != "1":
        return
    root = Path(os.environ["UCCL_GIN_ROOT"]).resolve()
    size = os.environ.get("UCCL_GIN_TEST_PUT_SIZE", "65536")
    out = _run(root, "uccl_gin_microbench", [
        "--sizes", size, "--iters", "4", "--warmup", "1", "--no-nccl",
        "--only", "put-add", "--correctness-only",
    ])
    assert f"UCCL-put/add bytes={size}: PASS" in out, out
    print(out)


def test_tail_add_correctness() -> None:
    if os.environ.get("UCCL_GIN_RUN_PRIMITIVES") != "1":
        return
    root = Path(os.environ["UCCL_GIN_ROOT"]).resolve()
    size = os.environ.get("UCCL_GIN_TEST_TAIL_SIZE", "65536")
    out = _run(root, "uccl_gin_microbench", [
        "--sizes", size, "--iters", "4", "--warmup", "1", "--no-nccl",
        "--only", "tail-add", "--correctness-only",
    ])
    assert f"UCCL-tail/q bytes={size}: PASS" in out, out
    print(out)


def test_put_quiet_correctness() -> None:
    if os.environ.get("UCCL_GIN_RUN_PRIMITIVES") != "1":
        return
    root = Path(os.environ["UCCL_GIN_ROOT"]).resolve()
    size = os.environ.get("UCCL_GIN_TEST_QUIET_SIZE", "65536")
    out = _run(root, "uccl_gin_microbench", [
        "--sizes", size, "--iters", "4", "--warmup", "1", "--no-nccl",
        "--only", "quiet", "--correctness-only",
    ])
    assert f"UCCL-put+q source-reuse bytes={size}: PASS" in out, out
    print(out)


def test_red_add_counter() -> None:
    if os.environ.get("UCCL_GIN_RUN_PRIMITIVES") != "1":
        return
    root = Path(os.environ["UCCL_GIN_ROOT"]).resolve()
    out = _run(root, "uccl_gin_microbench", [
        "--sizes", "1024", "--iters", "2", "--warmup", "1", "--no-nccl",
        "--only", "red-add", "--correctness-only",
    ])
    assert "UCCL-red_add counter: PASS" in out, out
    print(out)


def test_size_sweep() -> None:
    if os.environ.get("UCCL_GIN_RUN_PRIMITIVES") != "1":
        return
    root = Path(os.environ["UCCL_GIN_ROOT"]).resolve()
    out = _run(root, "uccl_gin_microbench", [
        "--sizes", _default_sizes(), "--iters", "10", "--warmup", "2", "--no-nccl",
        "--correctness-only",
    ])
    assert "all correctness PASS" in out, out
    print(out)


def test_all_primitives_comprehensive() -> None:
    """Single-shot all-primitive correctness run with multiple sizes."""
    if os.environ.get("UCCL_GIN_RUN_PRIMITIVES") != "1":
        return
    root = Path(os.environ["UCCL_GIN_ROOT"]).resolve()
    sizes = os.environ.get("UCCL_GIN_TEST_SIZES", "1024,4096,65536,262144,1048576,16777216")
    out = _run(root, "uccl_gin_microbench", [
        "--sizes", sizes, "--iters", "4", "--warmup", "1", "--no-nccl",
        "--correctness-only",
    ])
    required = (
        "UCCL-red_add counter: PASS",
        "UCCL-put/add",
        "UCCL-tail/q",
        "UCCL-put+q source-reuse",
        "all correctness PASS",
    )
    for r in required:
        assert r in out, f"missing '{r}' in output:\n{out}"
    print(out)
