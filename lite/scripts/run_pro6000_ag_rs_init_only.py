#!/usr/bin/env python3
"""Repeat frozen validation with NCCL initialization-only logging.

The original trace validation attests per-size algorithm/protocol dispatch.
This control removes per-operation TUNING messages from the measured batches,
while retaining actual per-rank transport edges logged during initialization.
It uses exactly the frozen candidates, shapes, repetitions, seeds and binaries.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from run_pro6000_ag_rs import ROOT, parse, sha
from summarize_pro6000_ag_rs import audit, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path("/home/wxzheng/uccl-eval.lock"))
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    parent = args.epoch / "validation"
    manifest, records, _ = audit(parent)
    policy_hash = sha(args.epoch / "policy.json")
    require(json.loads((args.epoch / "report/summary.json").read_text())["policy_sha256"] == policy_hash,
            "Original validation is not audited against this policy")
    for record in records:
        require(record["job"].get("policy_sha256") == policy_hash, "Wrong parent policy")
    identities = manifest["identities"]
    for path, digest in identities.items():
        require(sha(ROOT / path) == digest, "Measured source/binary changed: " + path)
    for path, digest in manifest["libraries"].items():
        require(sha(Path(path)) == digest, "Runtime library changed: " + path)
    lock = args.lock.open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for attempt in range(20):
        hardware = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu,uuid,pci.bus_id,driver_version",
            "--format=csv,noheader"], text=True)
        busy = [row for row in hardware.splitlines() if any(int(row.split(",")[i].split()[0]) for i in (2, 3))]
        if not busy:
            break
        if attempt == 19:
            raise RuntimeError("GPUs did not become idle: " + str(busy))
        time.sleep(1)
    require(hardware == manifest["hardware"], "Hardware identity differs")
    require(subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True) == manifest["topology"], "Topology differs")
    require(subprocess.check_output(["lscpu", "-p=CPU,SOCKET,NODE"], text=True) == manifest["cpu_topology"], "CPU topology differs")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(parent / "source", args.output / "source")
    control_source = Path(__file__).resolve()
    shutil.copyfile(control_source, args.output / control_source.name)
    manifest = {**manifest, "started": time.time(), "logging": "initialization-only",
                "diagnostic_validation_sha256": sha(parent / "results.json"),
                "control_tool": control_source.name, "control_tool_sha256": sha(control_source)}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NCCL_", "UCCL_", "CUDA_", "LD_LIBRARY"))}
    env.update(manifest["base_env"])
    results = []
    for prior in records:
        job = prior["job"]
        name = job["name"]
        for path, digest in identities.items():
            require(sha(ROOT / path) == digest, "Measured epoch changed: " + path)
        require(sha(control_source) == manifest["control_tool_sha256"], "Control tool changed")
        job_env = {**env, **job.get("env", {})}
        if job["backend"] == "nccl":
            job_env.update(NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,ENV,GRAPH",
                           NCCL_DEBUG_FILE=str((args.output / (name + ".nccl.%h.%p.log")).resolve()))
        record = {"job": job, "command": prior["command"], "env": {k: v for k, v in job_env.items()
                  if k.startswith(("NCCL_", "UCCL_", "CUDA_", "LD_LIBRARY"))}, "started": time.time()}
        print("START", name, flush=True)
        with (args.output / (name + ".log")).open("w") as log:
            process = subprocess.Popen(record["command"], cwd=ROOT, env=job_env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                record["returncode"] = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                record["returncode"] = -9
        record["elapsed"] = time.time() - record["started"]
        try:
            require(record["returncode"] == 0, "Process exit " + str(record["returncode"]))
            record["rows"] = parse((args.output / (name + ".log")).read_text(),
                                    [int(s) for s in prior["rows"]], manifest["ranks"])
        except Exception as error:
            record["failure"] = str(error)
        record["log_sha256"] = sha(args.output / (name + ".log"))
        (args.output / (name + ".json")).write_text(json.dumps(record, indent=2) + "\n")
        results.append(record)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print("FAIL" if "failure" in record else "PASS", name, f'{record["elapsed"]:.1f}s', flush=True)
        if "failure" in record:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            raise RuntimeError(record["failure"])
    require({path: sha(ROOT / path) for path in identities} == identities, "Final identity mismatch")
    require({path: sha(Path(path)) for path in manifest["libraries"]} == manifest["libraries"], "Library changed")
    (args.output / "completed.json").write_text(json.dumps({"finished": time.time(),
        "identities": identities, "jobs": len(results)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
