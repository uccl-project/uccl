#!/usr/bin/env python3
"""Run isolated checked AG/RS experiments; preserve failed jobs and exact inputs.

Jobs are JSON objects with name, backend (ours/nccl), op, args and env. Sizes
are TOTAL AllGather output / ReduceScatter input bytes. This runner does not
select winners; calibration and frozen validation must use separate job files.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
import re
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    "src/collectives/pro6000_ag_rs.cu",
    "include/uccl_lite/pro6000_ag_rs.cuh",
    "include/uccl_lite/pro6000_allreduce.cuh",
    "include/uccl_lite/allreduce_validation.cuh",
    "scripts/run_pro6000_ag_rs.py",
    "Makefile",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(log, sizes, ranks):
    rows, rank_rows = {}, {}
    for line in log.splitlines():
        fields = line.split()
        if fields and fields[0] == "RESULT":
            size, us, bw, bad = int(fields[1]), float(fields[2]), float(fields[3]), int(fields[4])
            if size in rows or bad or not math.isfinite(us) or us <= 0:
                raise ValueError("Invalid/duplicate result: " + line)
            if not math.isfinite(bw) or abs(bw - size * ((ranks - 1) / ranks) / (us * 1000)) > 0.0001:
                raise ValueError("Incorrect bus bandwidth: " + line)
            rows[size] = us
        elif fields and fields[0] == "RANK":
            size, rank, us, bad = int(fields[1]), int(fields[2]), float(fields[3]), int(fields[4])
            if (size, rank) in rank_rows or bad or not math.isfinite(us) or us <= 0:
                raise ValueError("Invalid/duplicate rank: " + line)
            rank_rows[size, rank] = us
    if set(rows) != set(sizes) or set(rank_rows) != {(s, r) for s in sizes for r in range(ranks)}:
        raise ValueError("Incomplete size/rank coverage")
    for size, us in rows.items():
        if us != max(rank_rows[size, r] for r in range(ranks)):
            raise ValueError("Result is not slowest rank")
    if "CHECK_FAILURE" in log or "AFFINITY_FAILURE" in log:
        raise ValueError("Correctness or placement failure")
    affinity = re.findall(r"AFFINITY rank=(\d+) tid=(\d+) before_cpus=(\d+) matched_cpus=(\d+)", log)
    if {int(row[0]) for row in affinity} != set(range(ranks)):
        raise ValueError("Missing thread placement audit")
    if len({(row[0], row[1]) for row in affinity}) != len(affinity):
        raise ValueError("Duplicate thread placement audit")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, choices=(4, 8), default=8)
    parser.add_argument("--lock", type=Path, default=Path("/home/wxzheng/uccl-eval.lock"))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()
    jobs = json.loads(args.jobs.read_text())
    if len({job["name"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate job names")
    lock = args.lock.open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Utilization is a sampled counter and can retain the preceding completed
    # job for a sampling period after all CUDA contexts have been released.
    for attempt in range(20):
        devices = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu,uuid,pci.bus_id,driver_version", "--format=csv,noheader"
        ], text=True)
        rows = devices.splitlines()[:args.ranks]
        if len(rows) != args.ranks:
            raise RuntimeError("Insufficient visible GPUs")
        busy = [row for row in rows if any(int(row.split(",")[i].split()[0]) for i in (2, 3))]
        if not busy:
            break
        if attempt == 19:
            raise RuntimeError("GPUs did not become idle: " + str(busy))
        time.sleep(1)
    args.output.mkdir(parents=True, exist_ok=False)
    snapshots = args.output / "source"
    snapshots.mkdir()
    identities = {}
    for path in SOURCES:
        source = ROOT / path
        identities[path] = sha(source)
        target = snapshots / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    binaries = {"ours": ROOT / "build/uccl_lite_pro6000_ag_rs", "nccl": ROOT / "build/uccl_lite_pro6000_ag_rs_nccl"}
    for backend in {job["backend"] for job in jobs}:
        path = binaries[backend]
        identities[str(path.relative_to(ROOT))] = sha(path)
    env = {k: v for k, v in os.environ.items() if not k.startswith("NCCL_")}
    env.update(NCCL_IB_DISABLE="1", NCCL_NET="Socket", CUDA_VISIBLE_DEVICES=",".join(map(str, range(args.ranks))))
    libraries = {}
    for backend in {job["backend"] for job in jobs}:
        linked = subprocess.check_output(["ldd", str(binaries[backend])], env=env, text=True)
        if "not found" in linked:
            raise RuntimeError("Unresolved runtime library: " + linked)
        for path in re.findall(r"=> (/\S+)", linked):
            if any(name in path for name in ("mscclpp", "nccl", "cudart")):
                libraries[path] = sha(Path(path))
    manifest = {
        "libraries": libraries,
        "jobs": jobs, "ranks": args.ranks, "identities": identities,
        "hardware": devices, "topology": subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True),
        "cpu_topology": subprocess.check_output(["lscpu", "-p=CPU,SOCKET,NODE"], text=True),
        "base_env": {k: v for k, v in env.items() if k.startswith(("NCCL_", "UCCL_", "CUDA_", "LD_LIBRARY"))},
        "timing": "steady_clock enqueue through stream completion, slowest rank, total bytes",
        "started": time.time(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    for job in jobs:
        name = job["name"]
        if not name or Path(name).name != name:
            raise ValueError("Unsafe job name")
        for path, digest in identities.items():
            if sha(ROOT / path) != digest:
                raise RuntimeError("Source/binary changed during experiment: " + path)
        options = {"begin": 1024, "end": 1 << 30, **job.get("args", {})}
        if "bytes" in options:
            sizes = [options["bytes"]]
        else:
            sizes, size = [], options["begin"]
            while size <= options["end"]:
                sizes.append(size)
                size *= 2
        command = [str(binaries[job["backend"]]), "--op", job["op"]]
        for key, value in options.items():
            command.extend(["--" + key, str(value)])
        job_env = {**env, **job.get("env", {})}
        if job["backend"] == "nccl":
            job_env.update(NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,ENV,GRAPH,TUNING", NCCL_DEBUG_FILE=str((args.output / (name + ".nccl.%h.%p.log")).resolve()))
        record = {"job": job, "command": command, "env": {k: v for k, v in job_env.items() if k.startswith(("NCCL_", "UCCL_", "CUDA_", "LD_LIBRARY"))}, "started": time.time()}
        print("START", name, flush=True)
        with (args.output / (name + ".log")).open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=job_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                record["returncode"] = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                record["returncode"] = -9
                record["failure"] = "timeout"
        record["elapsed"] = time.time() - record["started"]
        try:
            if record["returncode"]:
                raise ValueError("Process exit " + str(record["returncode"]))
            record["rows"] = parse((args.output / (name + ".log")).read_text(), sizes, args.ranks)
        except Exception as error:
            record["failure"] = str(error)
        record["log_sha256"] = sha(args.output / (name + ".log"))
        (args.output / (name + ".json")).write_text(json.dumps(record, indent=2) + "\n")
        results.append(record)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print("FAIL" if "failure" in record else "PASS", name, f'{record["elapsed"]:.1f}s', flush=True)
        if "failure" in record:
            # Never continue after a failed process: forked rank children may
            # still own CUDA contexts even if the group leader exited first.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if not args.keep_going:
                raise RuntimeError(record["failure"])
            time.sleep(2)
    final = {path: sha(ROOT / path) for path in identities}
    if final != identities:
        raise RuntimeError("Final source/binary identity mismatch")
    if {path: sha(Path(path)) for path in libraries} != libraries:
        raise RuntimeError("Runtime library changed")
    (args.output / "completed.json").write_text(json.dumps({"finished": time.time(), "identities": final, "jobs": len(results)}, indent=2) + "\n")
    if any("failure" in result for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
