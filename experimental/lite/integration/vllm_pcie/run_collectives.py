#!/usr/bin/env python3
"""Run isolated eight-GPU AllReduce calibration and independent validation."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

P2P = {"NCCL_P2P_LEVEL": "SYS", "NCCL_SHM_DISABLE": "1"}
CONFIGS = {
    "default": {},
    "sm-p2p": P2P,
    "sm-128": {**P2P, "NCCL_NTHREADS": "128", "NCCL_BUFFSIZE": "1048576"},
    "ce-2": {**P2P, "NCCL_P2P_USE_CUDA_MEMCPY": "1",
             "NCCL_MIN_NCHANNELS": "2", "NCCL_MAX_NCHANNELS": "2"},
    "ce-8": {**P2P, "NCCL_P2P_USE_CUDA_MEMCPY": "1",
             "NCCL_MIN_NCHANNELS": "8", "NCCL_MAX_NCHANNELS": "8",
             "NCCL_BUFFSIZE": "16777216"},
    "c03": {},
}


def medians(path):
    groups = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if not row["checked"] or row["us"] <= 0:
            raise RuntimeError("Invalid measurement")
        key = (row["bytes"], row["repeat"], row["iteration"])
        groups.setdefault(key, []).append(row["us"])
    repeats = {}
    for (size, repeat, _), values in groups.items():
        if len(values) != 8:
            raise RuntimeError("Missing rank sample")
        repeats.setdefault((size, repeat), []).append(max(values))
    cases = {}
    for (size, _), values in repeats.items():
        cases.setdefault(size, []).append(statistics.median(values))
    return {size: statistics.median(values) for size, values in cases.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("probe", "calibration", "validation"))
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = root / "build/measurements"
    output.mkdir(parents=True, exist_ok=True)
    experiment = args.experiment.resolve()
    extension, = (experiment / "env/.venv/lib/python3.12/site-packages/vllm").glob("_pcieccl_C*.so")
    source = experiment / "source/vllm-0.26.0/csrc/pcieccl/pcieccl.cu"
    clean_env = {k: v for k, v in os.environ.items()
                 if not k.startswith(("NCCL_", "VLLM_"))}
    clean_env.update(CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", OMP_NUM_THREADS="1",
                     NCCL_IB_DISABLE="1", NCCL_NET="Socket")
    if args.stage == "validation" and not (output / "frozen-selection.json").exists():
        raise RuntimeError("Calibrate and freeze selection before validation")
    with open("/home/wxzheng/uccl-eval.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        processes = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        if processes.strip():
            raise RuntimeError("GPU processes present; refusing concurrent benchmark")
        if args.stage == "validation":
            plan = {"started_unix": time.time(), "selection_sha256": hashlib.sha256(
                (output / "frozen-selection.json").read_bytes()).hexdigest(),
                "harness_sha256": hashlib.sha256((root / "integration/vllm_pcie/collective_bench.py").read_bytes()).hexdigest(),
                "orders": ["forward", "reverse"], "configs": CONFIGS}
            plan_path = output / "validation-plan.json"
            if not plan_path.exists():
                plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        for name, command in {
            "topology": ["nvidia-smi", "topo", "-m"],
            "gpu-identity": ["nvidia-smi", "-q"],
            "cpu-identity": ["lscpu"],
        }.items():
            if not (output / f"{name}.txt").exists():
                (output / f"{name}.txt").write_bytes(subprocess.check_output(command))
        if args.stage == "probe":
            jobs = [("ce-2", False), ("c03", False)]
        else:
            names = list(CONFIGS) if args.stage == "validation" else list(CONFIGS)[:-1]
            jobs = [(name, reverse) for reverse in ((False, True) if args.stage == "validation" else (False,))
                    for name in (names if not reverse else list(reversed(names)))]
        for name, reverse in jobs:
            label = f"{args.stage}-{name}-{'reverse' if reverse else 'forward'}"
            path = output / f"{label}.jsonl"
            complete = Path(str(path) + ".metadata.json")
            if complete.exists() and json.loads(complete.read_text()).get("status") == "complete":
                continue
            env = {**clean_env, **CONFIGS[name], "NCCL_DEBUG": "INFO",
                   "NCCL_DEBUG_SUBSYS": "INIT,ENV,GRAPH,P2P,SHM",
                   "NCCL_DEBUG_FILE": str(output / f"{label}.nccl.%h.%p.log")}
            command = [sys.executable, str(root / "integration/vllm_pcie/collective_bench.py"),
                       "--backend", "c03" if name == "c03" else "nccl",
                       "--label", label, "--output", str(path),
                       "--reference", str(root / "build"), "--extension", str(extension),
                       "--source", str(source)]
            if args.stage == "probe":
                command += ["--min-power", "20", "--max-power", "20", "--warmups", "2",
                            "--repeats", "1", "--iterations", "2"]
            elif args.stage == "calibration":
                command += ["--warmups", "4", "--repeats", "2", "--iterations", "3"]
            else:
                command += ["--warmups", "10", "--repeats", "3", "--iterations", "5",
                            "--seed", "13" if reverse else "7"]
            if reverse:
                command.append("--reverse")
            (output / f"{label}.command.json").write_text(json.dumps(command, indent=2) + "\n")
            print(f"START {label}", flush=True)
            started = time.time()
            with (output / f"{label}.log").open("w") as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=900)
            if result.returncode:
                raise RuntimeError(f"{label} failed; see {label}.log")
            print(f"COMPLETE {label} elapsed={time.time() - started:.1f}s", flush=True)
        if args.stage == "calibration":
            values = {name: medians(output / f"calibration-{name}-forward.jsonl")
                      for name in list(CONFIGS)[:-1]}
            schedule = {}
            for size in (1 << p for p in range(10, 31)):
                schedule[str(size)] = {
                    "sm": min(("default", "sm-p2p", "sm-128"), key=lambda n: values[n][size]),
                    "ce": min(("ce-2", "ce-8"), key=lambda n: values[n][size]),
                }
            frozen = {"configs": CONFIGS, "schedule": schedule,
                      "calibration_hashes": {name: hashlib.sha256(
                          (output / f"calibration-{name}-forward.jsonl").read_bytes()).hexdigest()
                          for name in list(CONFIGS)[:-1]}}
            (output / "frozen-selection.json").write_text(json.dumps(frozen, indent=2) + "\n")
        if args.stage == "validation":
            if not (output / "frozen-selection.json").exists():
                raise RuntimeError("Missing independent frozen calibration")
            (output / "COMPLETE.json").write_text(json.dumps({"status": "complete", "dtype": "bf16",
                "sizes": 21, "orders": 2, "warmups": 10, "repeats": 3, "iterations": 5,
                "configs": CONFIGS}, indent=2) + "\n")


if __name__ == "__main__":
    main()
