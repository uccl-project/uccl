#!/usr/bin/env python3
"""Own the GPU lock; retain source/library identity and every fresh-process run."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backends", nargs="+", default=["nccl", "peer", "peer", "nccl"],
                        choices=["nccl", "peer"])
    parser.add_argument("--kinds", nargs="+", default=["serving", "training"],
                        choices=["serving", "training", "check"])
    parser.add_argument("--lock", type=Path, default=Path.home() / "dev/uccl-eat-pro6000/eval.lock")
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    env = os.environ.copy()
    if any(k.startswith("NCCL_") for k in env):
        raise RuntimeError("unset inherited NCCL settings for the default baseline")
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    # Load exactly the environment's NCCL into both Torch and the C ABI bridge.
    import nvidia.nccl
    nccl = Path(nvidia.nccl.__path__[0]) / "lib/libnccl.so.2"
    env["LD_PRELOAD"] = str(nccl.resolve())
    sources = [*HERE.glob("*.py"), HERE / "collectives.cc", HERE / "Makefile",
               ROOT / "src/collectives/peer_collectives.cu",
               ROOT / "src/cpu_router/collective_dispatch.cc",
               ROOT / "src/topology/discovery.cc",
               ROOT / "src/topology/hardware.cc",
               ROOT / "include/uccl_lite/topology.hpp",
               ROOT / "include/uccl_lite/peer_plan.hpp",
               ROOT / "include/uccl_lite/planner.hpp",
               ROOT / "src/planner/planner.cc",
               ROOT / "src/planner/peer_plan.cc",
               ROOT / "include/uccl_lite/collectives/device_primitives.cuh",
               ROOT / "build/peer_e2e/libuccl_lite_peer_e2e.so", nccl,
               args.checkpoint / "model.safetensors", args.checkpoint / "config.json"]
    manifest = {"argv": sys.argv, "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "sha256": {str(p): digest(p) for p in sources}, "trials": [],
                "environment": {k: v for k, v in env.items() if k.startswith(
                    ("NCCL_", "CUDA_", "OMP_", "MKL_", "LD_", "UCCL_"))}}
    with args.lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, command in {
            "gpu.txt": ["nvidia-smi", "-q"],
            "topology.txt": ["nvidia-smi", "topo", "-m"],
            "cpu.txt": ["lscpu"],
            "packages.txt": [sys.executable, "-m", "pip", "list", "--format=freeze"],
        }.items():
            result = subprocess.run(command, text=True, capture_output=True)
            (args.out / name).write_text(result.stdout + result.stderr)
        for phase, backend in enumerate(args.backends):
            kinds = args.kinds if phase % 2 == 0 else list(reversed(args.kinds))
            for kind in kinds:
                label = f"{phase:02d}_{backend}_{kind}"
                command = [sys.executable, str(HERE / "benchmark.py"), "--checkpoint", str(args.checkpoint),
                           "--backend", backend, "--kind", kind,
                           "--out", str(args.out / f"{label}.json"), *extra]
                record = {"label": label, "command": command, "complete": False,
                          "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                manifest["trials"].append(record)
                (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                print(f"Starting {label}", flush=True)
                with (args.out / f"{label}.log").open("w") as log:
                    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=1800)
                record.update(returncode=result.returncode, complete=result.returncode == 0)
                (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                if result.returncode:
                    raise RuntimeError(f"{label} failed; see retained log")


if __name__ == "__main__":
    main()
