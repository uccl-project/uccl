from __future__ import annotations

import os
from pathlib import Path
import subprocess


def test_context_smoke() -> None:
    if os.environ.get("UCCL_GIN_RUN_CONTEXT_SMOKE") != "1":
        return

    root = Path(os.environ["UCCL_GIN_ROOT"]).resolve()
    hosts = os.environ["UCCL_GIN_MPI_HOSTS"]
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
    mpirun = os.environ.get("UCCL_GIN_MPIRUN", "/opt/amazon/openmpi/bin/mpirun")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "python") + os.pathsep + env.get("PYTHONPATH", "")
    exports = [
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "NCCL_SOCKET_IFNAME",
        "LOCAL_WORLD_SIZE",
        "UCCL_GIN_CONTEXT_BYTES",
    ]
    cmd = [
        mpirun,
        "--host",
        hosts,
        "-np",
        str(local_world_size * 2),
        "-npernode",
        str(local_world_size),
    ]
    for key in exports:
        if key in env:
            cmd.extend(["-x", key])
    cmd.extend(["python", "-m", "uccl_gin.context_smoke"])

    proc = subprocess.run(
        cmd,
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=int(os.environ.get("UCCL_GIN_TEST_TIMEOUT", "120")),
    )
    print(proc.stdout)
    assert proc.returncode == 0, proc.stdout
    assert "uccl_gin Context smoke PASS" in proc.stdout
