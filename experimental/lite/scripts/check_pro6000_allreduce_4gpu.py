#!/usr/bin/env python3
"""Exercise the shared experimental kernels on the documented four-L4 host."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--lock", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
binary = root / "build/uccl_lite_pro6000_ar"
a.output.mkdir(parents=True, exist_ok=False)
env = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    "UCCL_LITE_PLACEMENT": "single",
    "UCCL_LITE_CPU_BASES": "0,2,1,3",
}
cases = {
    "ll-1": [
        "--mode",
        "ll",
        "--blocks",
        "1",
        "--threads",
        "1024",
        "--end",
        str(4 << 20),
    ],
    "ll-32": ["--mode", "ll", "--blocks", "32", "--end", str(4 << 20)],
    "put": ["--mode", "put", "--blocks", "64"],
    "chunk-shared": [
        "--mode",
        "chunk-ce",
        "--blocks",
        "4",
        "--workers",
        "8",
        "--tile",
        str(16 << 20),
    ],
    "chunk-streams": [
        "--mode",
        "chunk-ce",
        "--blocks",
        "16",
        "--workers",
        "4",
        "--streams",
        "3",
        "--tile",
        str(4 << 20),
    ],
    "host": [
        "--mode",
        "host-ce",
        "--blocks",
        "4",
        "--workers",
        "8",
        "--tile",
        str(64 << 20),
    ],
}
manifest = {
    "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    "environment": {
        k: env[k]
        for k in ("CUDA_VISIBLE_DEVICES", "UCCL_LITE_PLACEMENT", "UCCL_LITE_CPU_BASES")
    },
    "runs": [],
}
with a.lock.open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name, cmd in {
        "topology": ["nvidia-smi", "topo", "-m"],
        "gpu": ["nvidia-smi", "-q"],
        "cpu": ["lscpu"],
        "linked": ["ldd", str(binary)],
    }.items():
        (a.output / (name + ".txt")).write_bytes(subprocess.check_output(cmd))
    for name, args in cases.items():
        active = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        )
        if active.strip():
            raise RuntimeError("GPU busy")
        cmd = [str(binary), "--seed", "3001", *args]
        print("START " + name, flush=True)
        with (a.output / (name + ".txt")).open("x") as out:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                status = proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                status = -9
        text = (a.output / (name + ".txt")).read_text()
        rows = [l.split() for l in text.splitlines() if l.startswith("RESULT ")]
        expected = [1 << i for i in range(10, 23 if name.startswith("ll-") else 31)]
        passed = (
            status == 0
            and [int(r[1]) for r in rows] == expected
            and all(int(r[-1]) == 0 for r in rows)
        )
        manifest["runs"].append(
            dict(
                name=name,
                command=cmd,
                returncode=status,
                passed=passed,
                log_sha256=hashlib.sha256(text.encode()).hexdigest(),
            )
        )
        (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(("PASS " if passed else "FAIL ") + name, flush=True)
        if not passed:
            raise RuntimeError("Four-GPU check failed")
