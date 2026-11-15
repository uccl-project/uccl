#!/usr/bin/env python3
"""Run fresh eight-rank training trials in NCCL/packet/packet/NCCL order."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--real-nccl", type=Path, required=True)
    parser.add_argument("--sharding", nargs="+", choices=["ddp", "fsdp"],
                        default=["ddp", "fsdp"])
    parser.add_argument("--backends", nargs="+", choices=["nccl", "packet"],
                        default=["nccl", "packet", "packet", "nccl"])
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.real_nccl = args.real_nccl.resolve(strict=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.exists():
        parser.error("out-dir already contains a run; choose a fresh directory")
    training_args = args.training_args
    if training_args[:1] == ["--"]:
        training_args = training_args[1:]
    env = os.environ.copy()
    env.update(UCCL_LITE_REAL_NCCL=str(args.real_nccl),
               UCCL_LITE_NCCL_MAX_BYTES="268435456", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    # Ensure PyTorch and the proxy share the requested NCCL binary.
    env["LD_PRELOAD"] = str(args.real_nccl) + (
        ":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else ""
    )
    if any(key in env for key in ("UCCL_LITE_NCCL_TRACE", "UCCL_LITE_MP_PACKET_PROFILE",
                                  "UCCL_LITE_NCCL_TRACE_SHAPES")):
        parser.error("unset tracing/profiling variables for timing")
    files = [Path(__file__).resolve(), HERE / "bench_training_step.py",
             HERE / "run_packet_8l4.sh", HERE / "build/libuccl_lite_nccl_compat.so",
             args.real_nccl]
    manifest = {
        "argv": sys.argv, "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
        "environment": {k: v for k, v in env.items() if k.startswith(
            ("UCCL_", "NCCL_", "CUDA_", "OMP_", "MKL_", "LD_", "PYTHON"))},
        "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "planned_trials": len(args.backends) * len(args.sharding),
        "trials": [],
    }
    for phase, backend in enumerate(args.backends):
        shardings = args.sharding if phase % 2 == 0 else list(reversed(args.sharding))
        for sharding in shardings:
            label = f"{phase:02d}_{backend}_{sharding}"
            command = [str(HERE / "run_packet_8l4.sh"), "env",
                       f"UCCL_LITE_NCCL_ENABLE={int(backend == 'packet')}",
                       sys.executable, "-m", "torch.distributed.run", "--standalone",
                       "--nproc-per-node=8", str(HERE / "bench_training_step.py"),
                       "--backend", backend, "--sharding", sharding, "--run-id", label,
                       "--out", str(args.out_dir / "samples.jsonl"), *training_args]
            record = {"label": label, "command": command, "complete": False}
            manifest["trials"].append(record)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Starting {label}", flush=True)
            with (args.out_dir / f"{label}.log").open("w") as log:
                child = subprocess.Popen(command, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    returncode = child.wait(timeout=600)
                finally:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
            record["returncode"] = returncode
            record["complete"] = returncode == 0
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            if returncode:
                raise RuntimeError(f"{label} failed with {returncode}; see trial log")
            time.sleep(2)


if __name__ == "__main__":
    main()
