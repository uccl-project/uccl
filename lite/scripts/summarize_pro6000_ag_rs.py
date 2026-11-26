#!/usr/bin/env python3
"""Audit checked AG/RS logs, freeze calibration, and summarize held-out runs."""
import argparse
import csv
from itertools import zip_longest
import json
from pathlib import Path
import random
import re
from statistics import median
import time

from run_pro6000_ag_rs import parse, sha

SIZES = [1 << p for p in range(10, 31)]
OPS = ("allgather", "reducescatter")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        json.dump(value, output, indent=2)
        output.write("\n")


def configuration(job):
    args = dict(job.get("args", {}))
    args.pop("seed", None)
    return {"backend": job["backend"], "op": job["op"], "args": args,
            "env": job.get("env", {})}


def audit(root, dispatch_reference=None):
    manifest = json.loads((root / "manifest.json").read_text())
    completed = json.loads((root / "completed.json").read_text())
    records = json.loads((root / "results.json").read_text())
    require(completed["identities"] == manifest["identities"], "Epoch changed before completion")
    require(completed["jobs"] == len(records), "Incomplete job count")
    require([r["job"] for r in records] == manifest["jobs"], "Job plan differs from executed records")
    require(len({r["job"]["name"] for r in records}) == len(records), "Duplicate job name")
    for path, digest in manifest["identities"].items():
        if path.startswith("build/"):
            continue  # Binary identity was attested by runner; binaries are not vendored.
        require(sha(root / "source" / path) == digest, "Source snapshot hash mismatch: " + path)
    init_only = manifest.get("logging") == "initialization-only"
    reference_records = None
    if init_only:
        require(dispatch_reference is not None, "Initialization-only runs require dispatch diagnostics")
        dm, dr, _ = audit(dispatch_reference)
        require(not dm.get("logging"), "Dispatch reference must contain per-operation traces")
        require(sha(dispatch_reference / "results.json") == manifest["diagnostic_validation_sha256"],
                "Dispatch reference changed")
        for key in ("identities", "libraries", "hardware", "topology", "base_env", "cpu_topology", "jobs"):
            require(dm[key] == manifest[key], "Diagnostic/control mismatch: " + key)
        require(sha(root / manifest["control_tool"]) == manifest["control_tool_sha256"], "Control tool snapshot changed")
        reference_records = {r["job"]["name"]: r for r in dr}
    checks = 0
    transport_logs = 0
    for record in records:
        job = record["job"]
        name = job["name"]
        if reference_records is not None:
            prior = reference_records[name]
            require(record["command"] == prior["command"], "Control command differs")
            omitted = {"NCCL_DEBUG_SUBSYS", "NCCL_DEBUG_FILE"}
            require({k: v for k, v in record["env"].items() if k not in omitted} ==
                    {k: v for k, v in prior["env"].items() if k not in omitted}, "Control environment differs")
            if job["backend"] == "nccl":
                require(record["env"].get("NCCL_DEBUG_SUBSYS") == "INIT,ENV,GRAPH", "Per-operation logging not disabled")
        require(not record.get("failure") and record["returncode"] == 0, "Failed job: " + name)
        require(json.loads((root / (name + ".json")).read_text()) == record, "Record mismatch: " + name)
        log_path = root / (name + ".log")
        require(sha(log_path) == record["log_sha256"], "Log changed: " + name)
        log = log_path.read_text()
        require(f'op={job["op"]} ranks={manifest["ranks"]} dtype=FP32 out-of-place total-bytes' in log,
                "Wrong operation, rank count, dtype or placement")
        options = job.get("args", {})
        if "bytes" in options:
            sizes = [options["bytes"]]
        else:
            sizes, size = [], options.get("begin", 1024)
            while size <= options.get("end", 1 << 30):
                sizes.append(size)
                size *= 2
        rows = parse(log, sizes, manifest["ranks"])
        require(rows == {int(k): v for k, v in record["rows"].items()}, "Parsed times differ from record")
        width = 1 if record["env"].get("UCCL_LITE_PLACEMENT") in ("single", "legacy") else 4
        require(all(int(w) == width for w in re.findall(r"matched_cpus=(\d+)", log)), "Wrong enforced CPU mask")
        stress = int(options.get("stress", 0))
        if stress:
            stress_rows = re.findall(r"STRESS rank=(\d+) bytes=(\d+) calls=(\d+) errors=(\d+)", log)
            require(len(stress_rows) == len(sizes) * manifest["ranks"], "Missing stress checks")
            require({(int(r), int(s)) for r, s, _, _ in stress_rows} ==
                    {(r, s) for r in range(manifest["ranks"]) for s in sizes}, "Stress rank/size mismatch")
            require(all(int(c) == stress and int(e) == 0 for _, _, c, e in stress_rows), "Stress failure")
        checks += len(sizes) * manifest["ranks"] * (10 + (stress + 1 if stress else 0))
        if job["backend"] == "nccl":
            debug = sorted(root.glob(name + ".nccl.*.log"))
            require(len(debug) == manifest["ranks"], "Missing NCCL rank logs")
            op = "AllGather" if job["op"] == "allgather" else "ReduceScatter"
            dispatch = []
            for path in debug:
                text = path.read_text()
                edges = [l.split(" via ", 1)[1] for l in text.splitlines() if "Channel " in l and " via " in l]
                require(edges, "Missing NCCL transport edges")
                selections = re.findall(op + r": (\d+) Bytes -> Algo (\w+) proto (\w+)", text)
                dispatch.extend(selections)
                if job.get("env", {}).get("NCCL_P2P_USE_CUDA_MEMCPY") == "1":
                    require(all("P2P/" in e and "/CE" in e for e in edges), "CE control used a different transport")
                    require(all(p == "SIMPLE" for _, _, p in selections), "CE control used non-Simple protocol")
                elif job.get("env", {}).get("NCCL_P2P_USE_CUDA_MEMCPY") == "0":
                    require(all("/CE" not in e for e in edges), "SM control used CE transport")
                if job.get("env", {}).get("NCCL_SHM_DISABLE") == "1":
                    require(all("P2P/" in e for e in edges), "Forced peer control fell back")
            if init_only:
                require(not dispatch, "Unexpected per-operation logging in control")
            else:
                require({int(s) * manifest["ranks"] for s, _, _ in dispatch} == set(sizes),
                        "Missing actual NCCL dispatch sizes: " + name)
            transport_logs += len(debug)
    return manifest, records, {"jobs": len(records), "full_checks": checks, "nccl_rank_logs": transport_logs}


def select_policy(manifest, records, rounds):
    require(manifest["ranks"] == 8, "PRO policy requires eight ranks")
    candidates, measured = {}, {}
    for record in records:
        job = record["job"]
        require(job.get("stage") == "calibration", "Not a calibration job")
        ident = job["candidate"]
        config = configuration(job)
        require(config["args"].get("capacity", 1 << 30) == 1 << 30, "Calibration capacity must be 1 GiB")
        require(not config["args"].get("graph", 0), "Eager policy cannot mix graph timing")
        require(config["args"].get("warmups", 10) == 10 and config["args"].get("iterations", 20) == 20,
                "Calibration must use matched 10-warmup/20-call timing")
        require(ident not in candidates or candidates[ident] == config, "Candidate geometry changed")
        candidates[ident] = config
        require((ident, job["round"]) not in measured, "Duplicate candidate/round")
        measured[ident, job["round"]] = record["rows"]
    require(set(measured) == {(c, r) for c in candidates for r in range(rounds)}, "Incomplete calibration rounds")
    selected = {}
    for op in OPS:
        selected[op] = {}
        for backend in ("ours", "nccl"):
            chosen = {}
            for size in SIZES:
                eligible = [c for c, config in candidates.items() if config["op"] == op and config["backend"] == backend
                            and all(str(size) in measured[c, r] for r in range(rounds))]
                require(eligible, f"No candidate for {op}/{backend}/{size}")
                chosen[str(size)] = min(eligible, key=lambda c: (median(measured[c, r][str(size)] for r in range(rounds)), c))
            selected[op][backend] = chosen
    return candidates, selected


def freeze(calibration, output, rounds, validation_rounds):
    manifest, records, evidence = audit(calibration)
    candidates, selected = select_policy(manifest, records, rounds)
    policy = {"schema": 1, "created": time.time(), "calibration_rounds": rounds,
              "validation_rounds": validation_rounds, "candidates": candidates, "selected": selected,
              "identities": manifest["identities"], "libraries": manifest["libraries"],
              "hardware": manifest["hardware"], "topology": manifest["topology"],
              "base_env": manifest["base_env"], "cpu_topology": manifest["cpu_topology"],
              "calibration_manifest_sha256": sha(calibration / "manifest.json"),
              "calibration_results_sha256": sha(calibration / "results.json"), "calibration_evidence": evidence}
    write_json(output, policy)
    print("Frozen policy", sha(output))


def validation_jobs(path, output):
    policy = json.loads(path.read_text())
    jobs = []
    for repeat in range(policy["validation_rounds"]):
        for op in (OPS if repeat % 2 == 0 else OPS[::-1]):
            groups = []
            for backend in (("ours", "nccl") if repeat % 2 == 0 else ("nccl", "ours")):
                ids = sorted(set(policy["selected"][op][backend].values()))
                random.Random(901 + repeat).shuffle(ids)
                groups.append(ids)
            for pair in zip_longest(*groups):
                for ident in pair:
                    if ident is None:
                        continue
                    config = policy["candidates"][ident]
                    jobs.append({**config, "args": {**config["args"], "seed": 20000 + repeat * 97},
                                 "name": f"v{repeat}-{ident}", "candidate": ident, "round": repeat,
                                 "stage": "validation", "policy_sha256": sha(path)})
    write_json(output, jobs)
    print(len(jobs), "validation jobs")


def summarize(calibration, validation, policy_path, output, diagnostic=None):
    policy = json.loads(policy_path.read_text())
    cm, cr, ce = audit(calibration)
    vm, vr, ve = audit(validation, diagnostic)
    require(sha(calibration / "manifest.json") == policy["calibration_manifest_sha256"] and
            sha(calibration / "results.json") == policy["calibration_results_sha256"], "Calibration changed after freeze")
    candidates, selected = select_policy(cm, cr, policy["calibration_rounds"])
    require(candidates == policy["candidates"] and selected == policy["selected"], "Policy was not selected from calibration")
    for key in ("identities", "libraries", "hardware", "topology", "base_env", "cpu_topology"):
        require(cm[key] == vm[key] == policy[key], "Calibration/validation epoch mismatch: " + key)
    by_round = {}
    ids = {c for op in OPS for backend in ("ours", "nccl") for c in selected[op][backend].values()}
    rounds = policy["validation_rounds"]
    for record in vr:
        job = record["job"]
        require(job.get("stage") == "validation" and job.get("policy_sha256") == sha(policy_path), "Validation did not use frozen policy")
        ident, repeat = job["candidate"], job["round"]
        require(configuration(job) == candidates[ident], "Validation configuration differs from calibration")
        require((ident, repeat) not in by_round, "Duplicate validation round")
        by_round[ident, repeat] = record["rows"]
    require(set(by_round) == {(c, r) for c in ids for r in range(rounds)}, "Validation coverage differs from frozen plan")
    output.mkdir(parents=True, exist_ok=False)
    report = {"policy_sha256": sha(policy_path), "calibration": ce, "validation": ve, "logging": vm.get("logging", "per-operation traces"), "operations": {}}
    for op in OPS:
        rows = []
        for size in SIZES:
            row = {"bytes": size}
            samples = {}
            for backend in ("ours", "nccl"):
                ident = selected[op][backend][str(size)]
                values = [by_round[ident, r][str(size)] for r in range(rounds)]
                samples[backend] = values
                row.update({backend + "_candidate": ident, backend + "_median_us": median(values),
                            backend + "_min_us": min(values), backend + "_max_us": max(values)})
                for r, value in enumerate(values):
                    row[f"{backend}_r{r}_us"] = value
            row["speedup"] = row["nccl_median_us"] / row["ours_median_us"]
            row["paired_wins"] = sum(a < b for a, b in zip(samples["ours"], samples["nccl"]))
            row["separated_ranges"] = max(samples["ours"]) < min(samples["nccl"])
            rows.append(row)
        with (output / (op + ".csv")).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        report["operations"][op] = {
            "median_wins": sum(r["speedup"] > 1 for r in rows),
            "all_round_wins": sum(r["paired_wins"] == rounds for r in rows),
            "separated_range_wins": sum(r["separated_ranges"] for r in rows),
            "losing_sizes": [r["bytes"] for r in rows if r["paired_wins"] != rounds],
        }
    write_json(output / "summary.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("audit")
    check.add_argument("input", type=Path)
    check.add_argument("--diagnostic", type=Path)
    lock = sub.add_parser("freeze")
    lock.add_argument("--calibration", type=Path, required=True)
    lock.add_argument("--output", type=Path, required=True)
    lock.add_argument("--rounds", type=int, default=3)
    lock.add_argument("--validation-rounds", type=int, default=5)
    jobs = sub.add_parser("validation-jobs")
    jobs.add_argument("--policy", type=Path, required=True)
    jobs.add_argument("--output", type=Path, required=True)
    report = sub.add_parser("summarize")
    for option in ("calibration", "validation", "policy", "output"):
        report.add_argument("--" + option, type=Path, required=True)
    report.add_argument("--diagnostic", type=Path)
    args = parser.parse_args()
    if args.command == "audit":
        print(json.dumps(audit(args.input, args.diagnostic)[2], indent=2))
    elif args.command == "freeze":
        freeze(args.calibration, args.output, args.rounds, args.validation_rounds)
    elif args.command == "validation-jobs":
        validation_jobs(args.policy, args.output)
    else:
        summarize(args.calibration, args.validation, args.policy, args.output, args.diagnostic)


if __name__ == "__main__":
    main()
