#!/usr/bin/env python3
"""Calibrate AllReduce SM/CE, then compare frozen policies with NCCL defaults.

Run on the recorded eight-PRO6000 host. No collective implementation changes
or validation-time tuning. All timings use initialization-only NCCL logging;
separate diagnostic runs attest actual per-size algorithm/protocol dispatch.
"""
import argparse
import csv
import fcntl
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import random
import re
import shutil
import signal
from statistics import median
import subprocess
import time

from prepare_pro6000_ag_rs import catalog
from run_pro6000_ag_rs import parse as parse_ag_rs
from run_pro6000_allreduce import parse as parse_ar, placement_audit

ROOT = Path(__file__).resolve().parents[1]
OPS = ("allreduce", "allgather", "reducescatter")
SIZES = [1 << p for p in range(10, 31)]
SOURCES = ["Makefile", "src/collectives/pro6000_allreduce.cu",
           "src/collectives/pro6000_ag_rs.cu", "include/uccl_lite/pro6000_allreduce.cuh",
           "include/uccl_lite/pro6000_ag_rs.cuh", "include/uccl_lite/allreduce_validation.cuh",
           "scripts/run_pro6000_nccl_baselines.py", "scripts/prepare_pro6000_ag_rs.py",
           "scripts/run_pro6000_ag_rs.py", "scripts/run_pro6000_allreduce.py"]
BINARIES = {"ar-ours": "build/uccl_lite_pro6000_ar",
            "ar-nccl": "build/uccl_lite_pro6000_ar_nccl",
            "ag-rs-ours": "build/uccl_lite_pro6000_ag_rs",
            "ag-rs-nccl": "build/uccl_lite_pro6000_ag_rs_nccl"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as out:
        json.dump(value, out, indent=2)
        out.write("\n")


def make_catalog(ar, ag_rs):
    configs, selected = {}, {}
    for op in OPS:
        ident = op + "-nccl-default"
        configs[ident] = dict(op=op, backend="nccl", args={}, env={})
        selected[op] = {"default": {str(n): ident for n in SIZES}}
    selected["allreduce"]["ours"] = {}
    for size, name in ar["policy"]["ours"].items():
        ident = "allreduce-ours-" + name
        configs[ident] = dict(op="allreduce", backend="ours", args=ar["ours_candidates"][name], env={})
        selected["allreduce"]["ours"][size] = ident
    for name, env in ar["nccl_candidates"].items():
        configs["allreduce-nccl-" + name] = dict(op="allreduce", backend="nccl", args={}, env=env)
    for ident, config in catalog().items():
        if ident.startswith("allgather-nccl-sm-"):
            configs[ident.replace("allgather", "allreduce")] = {**config, "op": "allreduce"}
    peer = {"NCCL_P2P_LEVEL": "SYS", "NCCL_SHM_DISABLE": "1",
            "NCCL_P2P_USE_CUDA_MEMCPY": "0", "NCCL_ALGO": "Tree"}
    for proto in ("auto", "LL", "LL128", "Simple"):
        env = peer if proto == "auto" else {**peer, "NCCL_PROTO": proto}
        configs["allreduce-nccl-sm-tree-" + proto.lower()] = dict(
            op="allreduce", backend="nccl", env=env,
            args={"end": 16 << 20} if proto in ("LL", "LL128") else {})
    for channels in (2, 8, 32):
        configs[f"allreduce-nccl-sm-tree-simple-ch{channels}"] = dict(
            op="allreduce", backend="nccl", args={}, env={**peer, "NCCL_PROTO": "Simple",
            "NCCL_MIN_NCHANNELS": str(channels), "NCCL_MAX_NCHANNELS": str(channels)})
    for op in OPS[1:]:
        for backend in ("ours", "nccl"):
            selected[op][backend] = ag_rs["selected"][op][backend]
            for ident in set(selected[op][backend].values()):
                configs[ident] = ag_rs["candidates"][ident]
    return configs, selected


def choose_allreduce(records, configs):
    expected = {c for c, v in configs.items() if v["op"] == "allreduce" and v["backend"] == "nccl"}
    measured = {(r["job"]["candidate"], r["job"]["round"]): r["rows"] for r in records}
    require(len(measured) == len(records), "Duplicate calibration round")
    require(set(measured) == {(c, r) for c in expected for r in range(3)}, "Incomplete calibration")
    chosen = {}
    for size in map(str, SIZES):
        eligible = [c for c in sorted(expected) if configs[c]["env"].get("NCCL_P2P_USE_CUDA_MEMCPY") in ("0", "1")
                    and all(size in measured[c, r] for r in range(3))]
        require(eligible, "Missing tuned candidate at " + size)
        chosen[size] = min(eligible, key=lambda c: (median(measured[c, r][size] for r in range(3)), c))
    return chosen


def jobs_for_policy(policy, stage):
    jobs = []
    for repeat in range(5 if stage == "validation" else 1):
        for op in (OPS if repeat % 2 == 0 else OPS[::-1]):
            groups = []
            for backend in (("ours", "nccl", "default") if repeat % 2 == 0 else ("default", "nccl", "ours")):
                if stage == "diagnostic" and backend == "ours":
                    continue
                ids = sorted(set(policy["selected"][op][backend].values()))
                random.Random(901 + repeat).shuffle(ids)
                group = []
                for ident in ids:
                    sizes = [int(n) for n, c in policy["selected"][op][backend].items() if c == ident]
                    group.append(dict(name=f"{stage}-{repeat}-{ident}", candidate=ident,
                                      round=repeat, stage=stage, begin=min(sizes), end=max(sizes)))
                groups.append(group)
            jobs.extend(job for row in zip_longest(*groups) for job in row if job)
    return jobs


def command_for(config, job, root):
    key = ("ar-" if config["op"] == "allreduce" else "ag-rs-") + config["backend"]
    args = {"begin": 1024, "end": 1 << 30, **config["args"], "seed": 7000 + job["round"] * 37}
    for key_arg in ("begin", "end"):
        if key_arg in job:
            args[key_arg] = job[key_arg]
    command = [str(root / BINARIES[key])]
    if config["op"] != "allreduce":
        command.extend(["--op", config["op"]])
    for key_arg, value in args.items():
        command.extend(["--" + key_arg, str(value)])
    return command, [n for n in SIZES if args["begin"] <= n <= args["end"]]


def check_log(text, config, sizes):
    if config["op"] == "allreduce":
        require("ranks=8 FP32 in-place SUM warmups=10 continuous=20 capacity=1073741824" in text,
                "Wrong AllReduce timing or layout")
        values = parse_ar(text, sizes)
    else:
        require(f'op={config["op"]} ranks=8 dtype=FP32 out-of-place total-bytes' in text,
                "Wrong AG/RS operation or layout")
        require("warmups=10 continuous=20 capacity=1073741824" in text, "Wrong AG/RS timing")
        values = parse_ag_rs(text, sizes, 8)
    require("CHECK_FAILURE" not in text and "AFFINITY_FAILURE" not in text, "Failed correctness/placement")
    placement_audit(text, 4)
    return {str(n): v for n, v in values.items()}


def check_transport(folder, job, config, sizes):
    logs = sorted(folder.glob(job["name"] + ".nccl.*.log"))
    require(len(logs) == 8, "Missing NCCL rank logs")
    dispatch, edge_types = [], set()
    op = {"allreduce": "AllReduce", "allgather": "AllGather", "reducescatter": "ReduceScatter"}[config["op"]]
    for path in logs:
        text = path.read_text()
        edges = [l.split(" via ", 1)[1] for l in text.splitlines() if "Channel " in l and " via " in l]
        require(edges, "Missing transport edges: " + path.name)
        selections = re.findall(op + r": (\d+) Bytes -> Algo (\w+) proto (\w+)", text)
        dispatch.extend(selections)
        env = config["env"]
        if env.get("NCCL_P2P_USE_CUDA_MEMCPY") == "1":
            require(all("P2P/" in e and "/CE" in e for e in edges), "CE candidate used different transport")
            require(all(p == "SIMPLE" for _, _, p in selections), "CE candidate used non-Simple")
        elif env.get("NCCL_P2P_USE_CUDA_MEMCPY") == "0":
            require(all("/CE" not in e for e in edges), "SM candidate used CE")
        if env.get("NCCL_SHM_DISABLE") == "1":
            require(all("P2P/" in e for e in edges), "Forced peer candidate fell back")
        edge_types.update(e.split()[0] for e in edges)
    if job["stage"] == "diagnostic":
        divisor = 1 if config["op"] == "allreduce" else 8
        require({int(n) * divisor for n, _, _ in dispatch} == set(sizes), "Missing per-size NCCL dispatch")
    else:
        require(not dispatch, "Timing run contains per-operation tracing")
    return {"hashes": {p.name: sha(p) for p in logs}, "transports": sorted(edge_types),
            "dispatch": [list(row) for row in sorted(set(dispatch))]}


class Runner:
    def __init__(self, output, ar_path, ag_rs_path):
        output.mkdir(parents=True, exist_ok=False)
        self.out = output
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("NCCL_", "UCCL_", "CUDA_"))}
        self.env.update(CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", UCCL_LITE_PLACEMENT="local-four")
        for path in (Path("/etc/nccl.conf"), Path.home() / ".nccl.conf"):
            require(not path.exists(), "NCCL default requires absent configuration file: " + str(path))
        configs = {}
        self.identities = {p: sha(ROOT / p) for p in SOURCES + list(BINARIES.values())}
        for name, path in (("allreduce", ar_path), ("ag-rs", ag_rs_path)):
            configs[name] = json.loads(path.read_text())
            for source, digest in configs[name]["identities"].items():
                if source.endswith((".cu", ".cuh")):
                    require(sha(ROOT / source) == digest, "Historical kernel changed: " + source)
            shutil.copyfile(path, output / (name + "-prior-policy.json"))
        self.configs, self.selected = make_catalog(configs["allreduce"], configs["ag-rs"])
        libraries = {}
        links = {}
        for key, binary in BINARIES.items():
            links[key] = subprocess.check_output(["ldd", str(ROOT / binary)], env=self.env, text=True)
            require("not found" not in links[key], "Missing runtime library")
            if key.endswith("ours"):
                require("libnccl" not in links[key], "UCCL binary links NCCL")
            for path in re.findall(r"=> (/\S+)", links[key]):
                if any(lib in path for lib in ("mscclpp", "nccl", "cudart")):
                    libraries[path] = sha(path)
        for name in SOURCES:
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        self.manifest = dict(identities=self.identities, libraries=libraries, links=links,
            root=str(ROOT), started=time.time(), configs=self.configs,
            base_env={k: v for k, v in self.env.items() if k.startswith(("NCCL_", "UCCL_", "CUDA_", "LD_LIBRARY"))},
            gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id,driver_version", "--format=csv,noheader"], text=True),
            topology=subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True),
            cpu=subprocess.check_output(["lscpu", "-p=CPU,SOCKET,NODE"], text=True),
            default_config_files="/etc/nccl.conf and ~/.nccl.conf absent; inherited NCCL_* variables removed",
            prior_policies={name: sha(output / name) for name in ("allreduce-prior-policy.json", "ag-rs-prior-policy.json")})
        write(output / "manifest.json", self.manifest)

    def check_identity(self):
        require({p: sha(ROOT / p) for p in self.identities} == self.identities, "Source/binary changed during run")
        require({p: sha(p) for p in self.manifest["libraries"]} == self.manifest["libraries"], "Runtime library changed")

    def run(self, jobs, stage, policy_hash=None):
        folder = self.out / stage
        folder.mkdir()
        write(folder / "jobs.json", jobs)
        records = []
        for job in jobs:
            self.check_identity()
            config = self.configs[job["candidate"]]
            command, sizes = command_for(config, job, ROOT)
            env = {**self.env, **config["env"]}
            if config["backend"] == "nccl":
                env.update(NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,ENV,GRAPH" + (",TUNING" if stage == "diagnostic" else ""),
                           NCCL_DEBUG_FILE=str(folder / (job["name"] + ".nccl.%h.%p.log")))
            record = dict(job=job, command=command, env={k: v for k, v in env.items()
                          if k.startswith(("NCCL_", "UCCL_", "CUDA_", "LD_LIBRARY"))},
                          started=time.time(), policy_sha256=policy_hash)
            print("START", job["name"], flush=True)
            log = folder / (job["name"] + ".log")
            with log.open("x") as stream:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    record["returncode"] = process.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    record["returncode"] = -9
            record.update(elapsed=time.time() - record["started"], log_sha256=sha(log))
            try:
                require(record["returncode"] == 0, "Process exit " + str(record["returncode"]))
                record["rows"] = check_log(log.read_text(), config, sizes)
                if config["backend"] == "nccl":
                    record["transport"] = check_transport(folder, job, config, sizes)
            except Exception as error:
                record["failure"] = str(error)
            write(folder / (job["name"] + ".json"), record)
            records.append(record)
            print("FAIL" if "failure" in record else "PASS", job["name"], f'{record["elapsed"]:.1f}s', flush=True)
            if "failure" in record:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                raise RuntimeError(record["failure"])
        self.check_identity()
        write(folder / "results.json", records)
        return records


def audit(root):
    manifest = json.loads((root / "manifest.json").read_text())
    require(json.loads((root / "completed.json").read_text())["identities"] == manifest["identities"], "Incomplete epoch")
    for name, digest in manifest["identities"].items():
        if not name.startswith("build/"):
            require(sha(root / "source" / name) == digest, "Source snapshot changed")
    for name, digest in manifest["prior_policies"].items():
        require(sha(root / name) == digest, "Prior policy changed")
    configs, inherited = make_catalog(json.loads((root / "allreduce-prior-policy.json").read_text()),
                                     json.loads((root / "ag-rs-prior-policy.json").read_text()))
    require(configs == manifest["configs"], "Candidate catalog changed")
    policy_path = root / "policy.json"
    policy = json.loads(policy_path.read_text())
    require(policy["manifest_sha256"] == sha(root / "manifest.json"), "Manifest changed after freeze")
    records, checks, rank_logs = {}, 0, 0
    for stage in ("calibration", "diagnostic", "validation"):
        folder = root / stage
        records[stage] = json.loads((folder / "results.json").read_text())
        jobs = json.loads((folder / "jobs.json").read_text())
        require([r["job"] for r in records[stage]] == jobs, "Job plan mismatch")
        if stage != "calibration":
            require(jobs == jobs_for_policy(policy, stage), "Validation differs from frozen plan")
        for record in records[stage]:
            job = record["job"]
            config = configs[job["candidate"]]
            require(not record.get("failure") and record["returncode"] == 0, "Failed measurement")
            require(json.loads((folder / (job["name"] + ".json")).read_text()) == record, "Record changed")
            command, sizes = command_for(config, job, Path(manifest["root"]))
            require(record["command"] == command, "Command differs from candidate")
            expected_env = {**manifest["base_env"], **config["env"]}
            if config["backend"] == "nccl":
                expected_env.update(NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,ENV,GRAPH" + (",TUNING" if stage == "diagnostic" else ""),
                    NCCL_DEBUG_FILE=record["env"]["NCCL_DEBUG_FILE"])
            require(record["env"] == expected_env, "Environment differs from candidate")
            if stage != "calibration":
                require(record["policy_sha256"] == sha(policy_path), "Policy changed after freeze")
            log = folder / (job["name"] + ".log")
            require(sha(log) == record["log_sha256"], "Log changed")
            require(check_log(log.read_text(), config, sizes) == record["rows"], "Timing differs from raw log")
            if config["backend"] == "nccl":
                transport = check_transport(folder, job, config, sizes)
                require(transport == record["transport"], "NCCL diagnostics changed")
                rank_logs += len(transport["hashes"])
            checks += len(sizes) * 8 * 10
    require(policy["calibration_sha256"] == sha(root / "calibration/results.json"), "Calibration changed after freeze")
    inherited["allreduce"]["nccl"] = choose_allreduce(records["calibration"], configs)
    require(inherited == policy["selected"], "Policy selection changed")
    return policy, records["validation"], {"full_per_rank_checks": checks, "nccl_rank_logs": rank_logs,
                                          "jobs": {s: len(r) for s, r in records.items()}}


def summarize(root, output):
    policy, records, evidence = audit(root)
    output.mkdir(parents=True, exist_ok=False)
    by_round = {(r["job"]["candidate"], r["job"]["round"]): r["rows"] for r in records}
    report = {"policy_sha256": sha(root / "policy.json"), "evidence": evidence, "operations": {}}
    for op in OPS:
        rows = []
        for size in map(str, SIZES):
            row = {"bytes": int(size)}
            for backend in ("ours", "nccl", "default"):
                ident = policy["selected"][op][backend][size]
                values = [by_round[ident, r][size] for r in range(5)]
                row.update({backend + "_candidate": ident, backend + "_median_us": median(values),
                            backend + "_min_us": min(values), backend + "_max_us": max(values)})
                row.update({f"{backend}_r{r}_us": v for r, v in enumerate(values)})
            rows.append(row)
        with (output / (op + ".csv")).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        report["operations"][op] = {"ours_median_wins_vs_tuned": sum(r["ours_median_us"] < r["nccl_median_us"] for r in rows),
            "ours_median_wins_vs_default": sum(r["ours_median_us"] < r["default_median_us"] for r in rows),
            "ours_losing_sizes_vs_tuned": [r["bytes"] for r in rows if r["ours_median_us"] >= r["nccl_median_us"]]}
    write(output / "summary.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "summarize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--ar-policy", type=Path)
    parser.add_argument("--ag-rs-policy", type=Path)
    args = parser.parse_args()
    if args.stage == "summarize":
        summarize(args.input, args.output)
        return
    with Path("/home/wxzheng/uccl-eval.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        devices = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"], text=True)
        require(len(devices.splitlines()) == 8 and all(int(v.split()[0]) == 0 for l in devices.splitlines() for v in l.split(",")), "GPUs are not idle")
        runner = Runner(args.output.resolve(), args.ar_policy, args.ag_rs_policy)
        candidates = sorted(c for c, v in runner.configs.items() if v["op"] == "allreduce" and v["backend"] == "nccl")
        jobs = []
        for repeat in range(3):
            order = candidates.copy()
            random.Random(20261116 + repeat).shuffle(order)
            jobs.extend(dict(name=f"calibration-{repeat}-{c}", candidate=c, round=repeat, stage="calibration") for c in order)
        records = runner.run(jobs, "calibration")
        runner.selected["allreduce"]["nccl"] = choose_allreduce(records, runner.configs)
        policy = dict(selected=runner.selected, manifest_sha256=sha(runner.out / "manifest.json"),
                      calibration_sha256=sha(runner.out / "calibration/results.json"), frozen=time.time())
        write(runner.out / "policy.json", policy)
        policy_hash = sha(runner.out / "policy.json")
        for stage in ("diagnostic", "validation"):
            runner.run(jobs_for_policy(policy, stage), stage, policy_hash)
        runner.check_identity()
        write(runner.out / "completed.json", dict(identities=runner.identities, finished=time.time()))
        summarize(runner.out, runner.out / "report")


if __name__ == "__main__":
    main()
