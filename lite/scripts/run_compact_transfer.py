#!/usr/bin/env python3
"""Reproducible, sequential 4->8 L4 experiments. Never overwrite raw runs."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/transfer_4to8_20261114"
DATA = ROOT / "benchmarks/data/transfer_4to8_20261114"
ENV = {k: os.environ[k] for k in ("HOME", "PATH") if k in os.environ}
ENV.update(LD_LIBRARY_PATH=str(ROOT / "build/nccl-2.28.9/lib") + ":/usr/local/cuda-12.9/lib64",
           NCCL_NET="Socket", NCCL_NET_PLUGIN="none", NCCL_IB_DISABLE="1")


def gpu_users():
    text = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    return sorted({int(x) for x in text.split()})


def run(name, args, binary="uccl_lite_compact_collectives", limit=300):
    DATA.mkdir(parents=True, exist_ok=True)
    log, manifest = DATA / (name + ".log"), DATA / (name + ".json")
    if log.exists() or manifest.exists():
        raise RuntimeError("refusing to overwrite " + name)
    before = gpu_users()
    if before:
        raise RuntimeError("GPUs occupied: " + str(before))
    command = [str(BUILD / binary), *args]
    record = dict(command=command, environment=ENV, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), gpu_users_before=before,
                  binary_sha256=hashlib.sha256(Path(command[0]).read_bytes()).hexdigest())
    start = time.monotonic()
    with log.open("w") as f:
        child = subprocess.Popen(command, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT)
        foreign = []
        while child.poll() is None:
            foreign = [p for p in gpu_users() if p != child.pid]
            if foreign or time.monotonic() - start > limit:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                break
            time.sleep(.5)
        status = child.wait()
    record.update(returncode=status, seconds=time.monotonic()-start, foreign_gpu_users=foreign, gpu_users_after=gpu_users())
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    print(name, "exit", status, "seconds", round(record["seconds"], 1), flush=True)
    if status or foreign or record["gpu_users_after"]:
        raise RuntimeError(log.read_text()[-3000:])


def bench(name, op, variants, begin=1024, end=1048576, host=False, blocks=16,
          warmup=5, iters=10, repeats=3, reverse=False, idle=0):
    args = ["--gpus", "8", "--op", op, "--variants", variants,
            "--begin", str(begin), "--end", str(end), "--blocks", str(blocks),
            "--warmup", str(warmup), "--iterations", str(iters), "--repeats", str(repeats), "--idle-us", str(idle)]
    if host:
        args += ["--host-timing"]
    if reverse:
        args += ["--reverse"]
    run(name, args, limit=600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["checks", "screen", "confirm", "tuning", "sparse", "regression", "attribution"])
    a = parser.parse_args()
    if a.phase == "checks":
        for n in (8, 4):
            run("check_compact_" + str(n), ["--check", "--gpus", str(n)])
        for name in ("router_policy_check", "dispatcher_check", "cpu_bf16_check", "router_core_check"):
            run("check_" + name, [], binary="uccl_lite_" + name)
    elif a.phase == "screen":
        for blocks in (8, 16, 32):
            bench("screen_ag_blocks" + str(blocks), "ag", "nccl,router-tuned,mini", begin=131072, end=16777216, blocks=blocks)
        for op in ("ag", "rs", "ar"):
            bench("screen_tiny_" + op, op, "nccl,nccl-graph,router-base,router-progress,router-tuned,router-single,resident", end=65536, host=True)
    elif a.phase == "confirm":
        for reverse, suffix in ((False, "a"), (True, "b")):
            bench("confirm_ag_" + suffix, "ag", "nccl,router-tuned,mini", begin=131072, end=16777216,
                  blocks=16, warmup=10, iters=20, repeats=5, reverse=reverse)
            for op in ("ag", "rs", "ar"):
                bench("confirm_tiny_" + op + "_" + suffix, op,
                      "nccl,nccl-graph,router-base,router-progress,router-tuned,router-single,resident",
                      end=65536, host=True, warmup=10, iters=20, repeats=5, reverse=reverse)
    elif a.phase == "sparse":
        for op in ("ag", "rs", "ar"):
            bench("sparse_" + op, op, "nccl,nccl-graph,router-progress,router-tuned,resident",
                  end=65536, host=True, idle=5000)
    elif a.phase == "tuning":
        for reverse, suffix in ((False, "a"), (True, "b")):
            for op in ("ag", "rs", "ar"):
                bench("tuning_" + op + "_" + suffix, op,
                      "nccl,router-base,router-progress,router-tuned,router-single",
                      end=4194304, warmup=10, iters=20, repeats=5, reverse=reverse)
    elif a.phase == "attribution":
        for reverse, suffix in ((False, "a"), (True, "b")):
            bench("ag_attribution_" + suffix, "ag",
                  "nccl,router-tuned,router-full-ag,router-full-ag-compact,mini",
                  begin=131072, end=16777216, warmup=10, iters=20,
                  repeats=5, reverse=reverse)
    elif a.phase == "regression":
        for n in (8, 4):
            run("final_compact_check_" + str(n), ["--check", "--gpus", str(n)])
            for name in ("router_check", "router_graph_check"):
                run("final_" + name + "_" + str(n), ["--gpus", str(n), "--ag-per-source"], binary="uccl_lite_" + name)
        run("final_dispatcher_cuda", [], binary="uccl_lite_dispatcher_cuda_check")
