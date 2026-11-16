#!/usr/bin/env python3
"""Calibrate NCCL SM, then validate frozen choices with matched C++ timing."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
from statistics import median
import subprocess
import time

P2P = {"NCCL_P2P_LEVEL": "SYS", "NCCL_SHM_DISABLE": "1"}
CONFIGS = {
    "default": {},
    "p2p": P2P,
    "p2p-128": {**P2P, "NCCL_NTHREADS": "128", "NCCL_BUFFSIZE": "1048576"},
    "ring-ll": {**P2P, "NCCL_ALGO": "Ring", "NCCL_PROTO": "LL"},
    "tree-ll": {**P2P, "NCCL_ALGO": "Tree", "NCCL_PROTO": "LL"},
    "ring-ll128": {**P2P, "NCCL_ALGO": "Ring", "NCCL_PROTO": "LL128"},
    "tree-ll128": {**P2P, "NCCL_ALGO": "Tree", "NCCL_PROTO": "LL128"},
    "ring-simple": {**P2P, "NCCL_ALGO": "Ring", "NCCL_PROTO": "Simple"},
    "tree-simple": {**P2P, "NCCL_ALGO": "Tree", "NCCL_PROTO": "Simple"},
    "ring-ll-ch2": {**P2P, "NCCL_ALGO": "Ring", "NCCL_PROTO": "LL",
                    "NCCL_MIN_NCHANNELS": "2", "NCCL_MAX_NCHANNELS": "2"},
    "ring-simple-ch2": {**P2P, "NCCL_ALGO": "Ring", "NCCL_PROTO": "Simple",
                        "NCCL_MIN_NCHANNELS": "2", "NCCL_MAX_NCHANNELS": "2"},
    "ring-simple-ch8": {**P2P, "NCCL_ALGO": "Ring", "NCCL_PROTO": "Simple",
                        "NCCL_MIN_NCHANNELS": "8", "NCCL_MAX_NCHANNELS": "8",
                        "NCCL_BUFFSIZE": "1048576"},
}
SIZES = [1 << p for p in range(10, 21)]
ORIGINAL_SOURCE = "80d7cb3d121121468744d872e7cc149ffcf83b7856a19b8059f3370bda0db3bb"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse(text, original=False):
    rows = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 7 or not fields[0].isdigit():
            continue
        size, count, mode, us, _, _, error = fields
        if original and mode != "auto":
            continue
        size, us, error = int(size), float(us), float(error)
        if size in rows or error != 0 or not math.isfinite(us) or us <= 0 or int(count) * 4 != size:
            raise ValueError(f"Invalid or failed row: {line}")
        rows[size] = us
    if set(rows) != set(SIZES):
        raise ValueError("Incomplete size grid")
    if original:
        numeric = [line.split() for line in text.splitlines()
                   if len(line.split()) == 7 and line.split()[0].isdigit()]
        if len(numeric) != 39 or any(float(row[-1]) != 0 for row in numeric):
            raise ValueError("Original PCIeCCL full-mode check failed")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    binary = root / "build/uccl_lite_nccl_sm_cpp_latency"
    original = args.original_root / "build/uccl_lite_pcieccl_original_latency"
    original_source = args.original_root / "benchmarks/data/pro6000_original_cpp_latency_20261116/measured_source/mscclpp_allreduce.cu"
    if sha(original_source) != ORIGINAL_SOURCE:
        raise ValueError("Original PCIeCCL source mismatch")
    output = root / "build/nccl-sm-cpp-latency"
    output.mkdir(parents=True, exist_ok=False)
    (output / "debug").mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NCCL_", "VLLM_"))}
    toolkit = args.experiment / "deps/cuda133-packages/nvidia/cu13"
    env.update(CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", NCCL_IB_DISABLE="1",
               NCCL_NET="Socket", NCCL_P2P_USE_CUDA_MEMCPY="0",
               LD_LIBRARY_PATH=f"{toolkit}/lib:{toolkit}/lib64")
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(),
                "arguments": {k: str(v) for k, v in vars(args).items()},
                "configs": CONFIGS, "common_environment": {k: v for k, v in env.items()
                    if k.startswith("NCCL_") or k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")},
                "source_sha256": sha(Path(__file__).parent / "nccl_sm_latency.cu"),
                "runner_sha256": sha(__file__), "binary_sha256": sha(binary),
                "original_binary_sha256": sha(original), "original_source_sha256": sha(original_source),
                "nccl_library_sha256": sha(args.experiment / "env/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2"),
                "runs": []}

    def run(label, name, reference=False):
        processes = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        if processes.strip():
            raise RuntimeError("GPU processes present; refusing concurrent run")
        local_env = {**env, **({} if reference else CONFIGS[name])}
        if not reference:
            local_env.update(NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,ENV,GRAPH,TUNING",
                             NCCL_DEBUG_FILE=str(output / "debug" / f"{label}.%h.%p.log"))
        command = [str(original if reference else binary), "-b", "1024", "-e", "1048576",
                   "-w", "10", "-n", "20"]
        if reference:
            command += ["-m", "all", "-c", "1"]
        entry = {"label": label, "config": name, "command": command,
                 "started_utc": datetime.now(timezone.utc).isoformat()}
        print("START " + label, flush=True)
        log = output / f"{label}.txt"
        with log.open("w") as stream:
            process = subprocess.Popen(command, env=local_env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                status = process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                status = -9
        entry.update(returncode=status, sha256=sha(log))
        if status != 0:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            manifest["runs"].append(entry)
            print(f"FAIL {label}: returncode={status}", flush=True)
            return None
        values = parse(log.read_text(), reference)
        entry["status"] = "pass"
        manifest["runs"].append(entry)
        if sha(binary) != manifest["binary_sha256"] or sha(original) != manifest["original_binary_sha256"]:
            raise RuntimeError("Measured binary changed")
        print("PASS " + label, flush=True)
        return values

    with open("/home/wxzheng/uccl-eval.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, command in {
            "topology.txt": ["nvidia-smi", "topo", "-m"],
            "gpu-identity.txt": ["nvidia-smi", "-q"],
            "cpu-identity.txt": ["lscpu"],
            "linked-libraries.txt": ["ldd", str(binary)],
            "compiler.txt": [str(toolkit / "bin/nvcc"), "--version"],
        }.items():
            (output / name).write_bytes(subprocess.check_output(command, env=env))
        calibration = {name: [] for name in CONFIGS}
        rejected = set()
        rng = random.Random(20261116)
        for repeat in range(1, 4):
            names = list(CONFIGS)
            rng.shuffle(names)
            for name in names:
                if name in rejected:
                    continue
                values = run(f"calibration-{repeat}-{name}", name)
                if values is None:
                    rejected.add(name)
                else:
                    calibration[name].append(values)
        valid = [name for name in CONFIGS if name not in rejected and len(calibration[name]) == 3]
        if "default" not in valid:
            raise RuntimeError("Default NCCL calibration failed")
        schedule = {str(size): min(valid, key=lambda n: median(r[size] for r in calibration[n]))
                    for size in SIZES}
        frozen = {"created_utc": datetime.now(timezone.utc).isoformat(), "schedule": schedule,
                  "eligible_configs": valid, "rejected_configs": sorted(rejected),
                  "calibration_log_hashes": {r["label"]: r["sha256"] for r in manifest["runs"]}}
        (output / "frozen-selection.json").write_text(json.dumps(frozen, indent=2) + "\n")
        manifest["frozen_selection_sha256"] = sha(output / "frozen-selection.json")
        manifest["validation_started_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "pre-validation-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        names = [name for name in CONFIGS if name in set(schedule.values()) or name == "default"]
        for repeat in range(1, 6):
            order = ["original"] + names if repeat % 2 else list(reversed(names)) + ["original"]
            for name in order:
                if run(f"validation-{repeat}-{name}", name, reference=name == "original") is None:
                    raise RuntimeError("Independent validation failed; no replacement policy allowed")
        manifest.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
