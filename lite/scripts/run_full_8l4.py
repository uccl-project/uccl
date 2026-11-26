#!/usr/bin/env python3
"""Fresh matched 1-KiB--1-GiB GPU/SHM sweeps; refuse to overwrite evidence."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/full_refresh_8l4_20261114"
DATA = ROOT / "benchmarks/data/full_refresh_8l4_20261114"
ENV = {k: os.environ[k] for k in ("HOME", "PATH") if k in os.environ}
ENV.update(LD_LIBRARY_PATH=str(ROOT / "build/nccl-2.28.9/lib") + ":/usr/local/cuda-12.9/lib64",
           NCCL_NET="Socket", NCCL_NET_PLUGIN="none", NCCL_IB_DISABLE="1")


def users():
    return sorted({int(x) for x in subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).split()})


def run(name, args, mode="gpu", binary="uccl_lite_compact_collectives", probe=False, timeout=2400):
    raw = DATA / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    log, manifest = raw / (name + ".log"), raw / (name + ".json")
    if log.exists() or manifest.exists():
        raise RuntimeError("refusing to overwrite: " + name)
    env = dict(ENV)
    if mode == "shm":
        env.update(NCCL_P2P_DISABLE="1", NCCL_SHM_DISABLE="0")
        if binary == "uccl_lite_compact_collectives":
            args = [*args, "--shm-only"]
    if probe:
        env.update(NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,GRAPH,ENV")
    command = [str(BUILD / binary), *args]
    before = users()
    if before:
        raise RuntimeError("GPU occupied: " + str(before))
    record = dict(command=command, environment=env, testbed=mode,
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  binary_sha256=hashlib.sha256(Path(command[0]).read_bytes()).hexdigest(),
                  gpu_users_before=before)
    start = time.monotonic()
    foreign, timed_out = [], False
    with log.open("x") as out:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                foreign = [p for p in users() if p != child.pid]
                timed_out = time.monotonic() - start > timeout
                if foreign or timed_out:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                    break
                time.sleep(.5)
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
            raise
        status = child.wait()
    record.update(returncode=status, seconds=time.monotonic()-start, timed_out=timed_out,
                  foreign_gpu_users=foreign, gpu_users_after=users())
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    print(name, "exit", status, "seconds", round(record["seconds"], 1), flush=True)
    if status or foreign or timed_out or record["gpu_users_after"]:
        raise RuntimeError(log.read_text()[-3000:])
    if probe:
        edges = re.findall(r"Channel\s+(\d+)\s+:\s+(\d+)\[\d+\]\s+->\s+(\d+)\[\d+\]\s+via\s+(\S+)", log.read_text())
        if not edges or {int(e[1]) for e in edges} != set(range(8)):
            raise RuntimeError("incomplete NCCL transport evidence")
        if mode == "shm" and any(not e[3].startswith("SHM/") for e in edges):
            raise RuntimeError("non-SHM NCCL edge in SHM testbed")
        (raw / (name + "_edges.json")).write_text(json.dumps(edges, indent=2) + "\n")


def bench(name, mode, op, begin=1024, end=1 << 30, reverse=False,
          warmup=5, iterations=10, repeats=3, probe=False, variants="nccl,router-base,current"):
    args = ["--gpus", "8", "--op", op, "--begin", str(begin), "--end", str(end),
            "--warmup", str(warmup), "--iterations", str(iterations), "--repeats", str(repeats),
            "--variants", variants]
    if reverse:
        args.append("--reverse")
    run(name, args, mode, probe=probe)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["checks", "probe", "tail", "full"])
    a = parser.parse_args()
    if a.phase == "checks":
        for n in (4, 8):
            run("compact_check_" + str(n), ["--check", "--gpus", str(n)])
        run("compact_check_shm", ["--check", "--gpus", "8"], "shm")
        for binary in ("router_policy_check", "dispatcher_check", "router_core_check", "cpu_bf16_check", "dispatcher_cuda_check"):
            run(binary, [], binary="uccl_lite_" + binary)
        for mode in ("gpu", "shm"):
            for binary in ("router_check", "router_graph_check"):
                run(mode + "_" + binary, ["--gpus", "8", "--ag-per-source"], mode,
                    binary="uccl_lite_" + binary)
    elif a.phase == "probe":
        for mode in ("gpu", "shm"):
            for op in ("ar", "ag", "rs"):
                bench("probe_" + mode + "_" + op, mode, op, end=1024,
                      warmup=1, iterations=2, repeats=1, probe=True)
    elif a.phase == "tail":
        for mode in ("gpu", "shm"):
            for op in ("ar", "ag", "rs"):
                bench("tail_" + mode + "_" + op, mode, op, begin=1 << 30,
                      warmup=1, iterations=2, repeats=1)
    elif a.phase == "full":
        for reverse, suffix in ((False, "a"), (True, "b")):
            for mode in (("gpu", "shm") if not reverse else ("shm", "gpu")):
                for op in (("ar", "ag", "rs") if not reverse else ("rs", "ag", "ar")):
                    bench("full_" + mode + "_" + op + "_" + suffix,
                          mode, op, reverse=reverse)
