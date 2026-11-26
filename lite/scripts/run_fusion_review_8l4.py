#!/usr/bin/env python3
"""Check and compare historical/current standalone controls for 8-L4 fusion."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks/data/fusion_review_8l4_20261115"
BUILD = ROOT / "build/fusion_review_8l4_20261115"


def runner():
    spec = importlib.util.spec_from_file_location("full8", ROOT / "scripts/run_full_8l4.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DATA, module.BUILD = DATA, BUILD
    return module


def summarize():
    groups = {}
    for path in sorted(DATA.glob("samples_*.csv")):
        # File names carry the independent experiment configuration and run.
        _, profile, batch, run = path.stem.split("_")
        for row in csv.DictReader(path.open()):
            key = (profile, batch, row["op"], int(row["m"]), int(row["n"]),
                   int(row["k"]), row["method"])
            groups.setdefault(key, {}).setdefault(run, []).append(float(row["latency_ms"]))
    with (DATA / "summary.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(("profile", "batch", "op", "m", "n", "k", "method",
                         "run1_ms", "run2_ms", "median_ms", "samples"))
        for key, runs in sorted(groups.items()):
            assert set(runs) == {"1", "2"}, (key, runs.keys())
            assert all(len(v) == 30 for v in runs.values()), key
            medians = [statistics.median(runs[r]) for r in ("1", "2")]
            writer.writerow((*key, *medians, statistics.median(medians),
                             sum(map(len, runs.values()))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("checks", "sweep", "summary"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.phase == "summary":
        summarize()
        return
    r = runner()
    DATA.mkdir(parents=True, exist_ok=True)
    if args.phase == "checks":
        for name in ("router_policy_check", "dispatcher_check", "dispatcher_cuda_check",
                     "cpu_bf16_check", "router_core_check"):
            r.run(name + "_full_affinity", [], binary="uccl_lite_" + name)
        for devices in ("0,1,4,5", "0,1,2,3,4,5,6,7"):
            for name in ("router_check", "router_graph_check"):
                for mode, flags in (("cpu", ["--batch-cpu-jobs"]),
                                    ("gpu", ["--ag-per-source"])):
                    r.run(f"{name}_{devices.replace(',', '-')}_{mode}",
                          ["--devices", devices, *flags], binary="uccl_lite_" + name)
        return
    os.sched_setaffinity(0, {47})
    sources = [Path(__file__).resolve(), ROOT / "benchmarks/fusion/router_fusion_8l4.cu",
               ROOT / "benchmarks/cpu_router/compact_transport.cuh",
               ROOT / "src/cpu_router/packet_pipeline.cu",
               ROOT / "include/uccl_lite/cpu_router/packet_pipeline.hpp"]
    manifest = DATA / "source_hashes.json"
    if manifest.exists() and not args.resume:
        raise RuntimeError("use a fresh destination for a new sweep")
    if args.resume:
        version = 2
        manifest = DATA / f"source_hashes_v{version}.json"
        while manifest.exists():
            version += 1
            manifest = DATA / f"source_hashes_v{version}.json"
    manifest.write_text(json.dumps({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sources}, indent=2) + "\n")
    for name, command in (("hardware.csv", ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,pci.bus_id", "--format=csv"]),
                          ("topology.txt", ["nvidia-smi", "topo", "-m"]),
                          ("cpu.json", ["lscpu", "-J"])):
        (DATA / name).write_text(subprocess.check_output(command, text=True))
    # Two endpoint model shapes cover narrow/wide GEMMs and decode/prefill.
    # AG uses eight rows for its smallest integral eight-rank input.
    shapes = [(op, m, n, k) for op in ("gemm-ar", "gemm-rs", "ag-gemm")
              for n, k in ((4096, 1536), (5120, 3200))
              for m in ((8, 64, 2048, 8192) if op == "ag-gemm" else (1, 64, 2048, 8192))]
    for run in (1, 2):
        for index, (op, m, n, k) in enumerate(shapes if run == 1 else reversed(shapes)):
            # Reverse configuration order as well as shape order in run two.
            configs = [("historical", 0), ("current", 0), ("current", 1)]
            if run == 2:
                configs.reverse()
            for profile, batch in configs:
                sample = DATA / f"samples_{profile}_{batch}_{run}.csv"
                name = f"sweep_{run}_{index:02d}_{profile}_{batch}"
                base, attempt, complete = name, 0, False
                if args.resume:
                    while (DATA / "raw" / (name + ".json")).exists():
                        record = json.loads((DATA / "raw" / (name + ".json")).read_text())
                        if record["returncode"] == 0 and not record["foreign_gpu_users"] and not record["gpu_users_after"]:
                            complete = True
                            break
                        attempt += 1
                        name = base + f"_retry{attempt}"
                    if complete:
                        print("retain completed", name, flush=True)
                        continue
                if attempt:
                    # Preserve any incomplete call samples before retrying.
                    if sample.exists():
                        with sample.open() as stream:
                            reader = csv.DictReader(stream)
                            fields, rows = reader.fieldnames, list(reader)
                        def match(row):
                            return (row["op"], int(row["m"]), int(row["n"]), int(row["k"])) == (op, m, n, k)
                        if any(map(match, rows)):
                            (DATA / "raw" / (name + "_partial.csv")).write_bytes(sample.read_bytes())
                            with sample.open("w") as stream:
                                writer = csv.DictWriter(stream, fieldnames=fields)
                                writer.writeheader()
                                writer.writerows(row for row in rows if not match(row))
                command = ["--op", op, "--m", str(m), "--n", str(n), "--k", str(k),
                           "--warmup", "2", "--iterations", "10", "--repeats", "3",
                           "--router-profile", profile, "--batch-cpu-jobs", str(batch),
                           "--samples", str(sample)]
                # The third config isolates the fused batching option; controls
                # already have paired measurements in the first two configs.
                if batch:
                    command.extend(("--method", "fused"))
                r.run(name, command,
                      binary="uccl_lite_router_fusion_8l4")
    summarize()


if __name__ == "__main__":
    main()
