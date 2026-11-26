"""Unmodified DDP and stock FSDP, using a calibrated NCCL ABI replacement."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys

from dropin_profile import HERE, identity, launch_environment, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-nccl", type=Path, required=True)
    p.add_argument("--profile", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--backends", nargs="+", choices=["nccl", "dropin", "passthrough"],
                   default=["nccl", "dropin", "dropin", "nccl"])
    p.add_argument("--sharding", nargs="+", choices=["ddp", "fsdp"], default=["ddp", "fsdp"])
    p.add_argument("--trace", action="store_true")
    p.add_argument("training_args", nargs=argparse.REMAINDER)
    args = p.parse_args()
    args.real_nccl = args.real_nccl.resolve(strict=True)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    training_args = args.training_args[1:] if args.training_args[:1] == ["--"] else args.training_args
    if "--fsdp-rs-group" in training_args:
        p.error("this runner requires stock FSDP")
    ident, details = identity(args.real_nccl)
    manifest = {"identity": ident, "details": details, "argv": sys.argv, "trials": [],
                "bench_sha256": sha256(HERE / "bench_training_step.py"),
                "profile_sha256": sha256(args.profile) if args.profile else None,
                "trace": args.trace, "stock_fsdp": True}
    with open("/tmp/uccl-lite-integration-wxz-l4.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for phase, backend in enumerate(args.backends):
            shardings = args.sharding if phase % 2 == 0 else list(reversed(args.sharding))
            for sharding in shardings:
                label = f"{phase:02d}_{backend}_{sharding}"
                env = launch_environment(args.real_nccl, args.profile if backend == "dropin" else None)
                env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                if backend == "nccl":
                    env["LD_PRELOAD"] = str(args.real_nccl)
                if args.trace:
                    env["UCCL_LITE_DROPIN_TRACE"] = "1"
                command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=8",
                           str(HERE / "bench_training_step.py"), "--sharding", sharding,
                           "--fsdp-rs-group", "stock", "--backend", backend, "--run-id", label,
                           "--out", str(args.out_dir / "samples.jsonl"), *training_args]
                print(f"Starting {label}", flush=True)
                with (args.out_dir / f"{label}.log").open("w") as log:
                    child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        code = child.wait(timeout=900)
                    finally:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        child.wait()
                manifest["trials"].append({"backend": backend, "sharding": sharding, "command": command, "returncode": code})
                (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                if code:
                    raise RuntimeError(f"{label} failed; see its log")
    rows = [json.loads(x) for x in (args.out_dir / "samples.jsonl").read_text().splitlines()]
    summary = {}
    for sharding in args.sharding:
        jobs = {backend: [r for r in rows if r["sharding"] == sharding and r["backend"] == backend]
                for backend in set(args.backends)}
        result = {backend: {"job_ms": [r["median_step_s"] * 1000 for r in group],
                            "median_ms": statistics.median(r["median_step_s"] * 1000 for r in group)}
                  for backend, group in jobs.items()}
        if "nccl" in jobs and "dropin" in jobs:
            result["latency_change_pct"] = (result["dropin"]["median_ms"] / result["nccl"]["median_ms"] - 1) * 100
            result["no_regression_5pct"] = result["latency_change_pct"] <= 5
            base = jobs["nccl"][0]
            differences = [abs(x - y) for r in jobs["dropin"] for a, b in zip(r["rank_losses"], base["rank_losses"]) for x, y in zip(a, b)]
            result["max_loss_difference"] = max(differences)
            result["max_parameter_sample_difference"] = max(abs(x - y) for r in jobs["dropin"] for x, y in zip(r["parameter_samples"], base["parameter_samples"]))
        summary[sharding] = result
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if any(r.get("no_regression_5pct") is False for r in summary.values()):
        raise RuntimeError("end-to-end regression exceeded 5%; profile must not be promoted")


if __name__ == "__main__":
    main()
