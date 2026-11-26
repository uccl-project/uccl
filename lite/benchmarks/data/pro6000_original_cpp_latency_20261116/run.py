#!/usr/bin/env python3
"""Run the unchanged PRO5000 C++ benchmark on the eight-PRO6000 host."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

SOURCE_SHA = "80d7cb3d121121468744d872e7cc149ffcf83b7856a19b8059f3370bda0db3bb"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse(text):
    rows = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 7 or not fields[0].isdigit():
            continue
        size, count, mode, us, algbw, busbw, error = fields
        key = (int(size), mode)
        if key in rows or float(error) != 0 or float(us) <= 0 or int(count) * 4 != int(size):
            raise RuntimeError(f"Invalid or failed result: {line}")
        rows[key] = {"bytes": int(size), "mode": mode, "us": float(us),
                     "maxrelerr": float(error)}
    expected = {(1 << power, mode) for power in range(10, 21)
                for mode in ("ll", "put", "ce", "auto")
                if mode != "ll" or power <= 15}
    if set(rows) != expected:
        raise RuntimeError("Incomplete size/mode sweep")
    return list(rows.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    source = Path(__file__).parent / "measured_source/mscclpp_allreduce.cu"
    binary = root / "build/uccl_lite_pcieccl_original_latency"
    output = root / "build/original-cpp-latency"
    output.mkdir(parents=True, exist_ok=False)
    if sha(source) != SOURCE_SHA:
        raise RuntimeError("Historical source identity mismatch")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NCCL_", "VLLM_"))}
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    # The isolated CUDA toolkit is not installed in the system loader cache.
    toolkit = args.experiment / "deps/cuda133-packages/nvidia/cu13"
    env["LD_LIBRARY_PATH"] = str(toolkit / "lib") + ":" + str(toolkit / "lib64")
    command = [str(binary), "-b", "1024", "-e", "1048576", "-n", "20",
               "-w", "10", "-m", "all", "-c", "1"]
    record = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": sha(source), "binary_sha256": sha(binary),
              "runner_sha256": sha(__file__), "command": command,
              "environment_overrides": {key: env[key] for key in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")},
              "runs": [], "historical_stdout_label": "PRO5000 label is hardcoded; use actual GPU identity"}
    with open("/home/wxzheng/uccl-eval.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        commands = {
            "topology.txt": ["nvidia-smi", "topo", "-m"],
            "gpu-identity.txt": ["nvidia-smi", "-q"],
            "cpu-identity.txt": ["lscpu"],
            "linked-libraries.txt": ["ldd", str(binary)],
            "compiler.txt": [str(args.experiment / "deps/cuda133-packages/nvidia/cu13/bin/nvcc"), "--version"],
        }
        for name, argv in commands.items():
            (output / name).write_bytes(subprocess.check_output(argv, env=env))
        dependency = args.experiment / "deps/mscclpp"
        record["mscclpp_library_sha256"] = sha(dependency / "lib/libmscclpp.so")
        record["mscclpp_version_header"] = (dependency / "include/mscclpp/version.hpp").read_text()
        headers = dependency / "include"
        record["mscclpp_header_hashes"] = {str(p.relative_to(headers)): sha(p)
                                            for p in sorted(headers.rglob('*')) if p.is_file()}
        for index in range(1, args.runs + 1):
            processes = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
            if processes.strip():
                raise RuntimeError("GPU processes present; refusing concurrent run")
            print(f"START original C++ run {index}/{args.runs}", flush=True)
            started = time.time()
            log = output / f"run-{index}.txt"
            with log.open("w") as stream:
                subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                               check=True, timeout=120)
            rows = parse(log.read_text())
            if sha(source) != SOURCE_SHA or sha(binary) != record["binary_sha256"]:
                raise RuntimeError("Source/binary changed during measurement")
            record["runs"].append({"index": index, "rows": len(rows),
                                   "sha256": sha(log), "elapsed_s": time.time() - started})
            print(f"PASS run {index}: {len(rows)} size/mode checks", flush=True)
    record.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
