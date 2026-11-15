#!/usr/bin/env python3
"""Explicit 128/512-KiB resident AR packet comparison, plus control boundaries."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics

DATA = Path(__file__).resolve().parent
ROOT = DATA.parents[2]
spec = importlib.util.spec_from_file_location("full8", ROOT / "scripts/run_full_8l4.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
r.DATA, r.BUILD = DATA, ROOT / "build/fusion_review_8l4_20261115"
os.sched_setaffinity(0, {47})


def run_case(name, args, **kwargs):
    base, attempt = name, 0
    while (DATA / "raw" / (name + ".json")).exists():
        record = json.loads((DATA / "raw" / (name + ".json")).read_text())
        if record["returncode"] == 0 and not record["foreign_gpu_users"] and not record["gpu_users_after"]:
            print("retain completed", name, flush=True)
            return
        attempt += 1
        name = base + f"_retry{attempt}"
    if attempt and "--samples" in args:
        sample = Path(args[args.index("--samples") + 1])
        if sample.exists():
            with sample.open() as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
            shape = tuple(args[args.index(option) + 1] for option in ("--op", "--m", "--n", "--k"))
            def match(row):
                return tuple(row[k] for k in ("op", "m", "n", "k")) == shape
            if any(map(match, rows)):
                (DATA / "raw" / (name + "_excluded.csv")).write_bytes(sample.read_bytes())
                with sample.open("w") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(row for row in rows if not match(row))
    r.run(name, args, **kwargs)


for ranks in (4, 8):
    run_case(f"host_ring_check_{ranks}", [str(ranks)],
          binary="uccl_lite_host_ring_check", timeout=120)

# These boundary correctness controls exercise the new benchmark wiring at RS
# ring endpoints and mini-host AG endpoints, outside the 24-shape comparison.
for m in (128, 1024):
    run_case(f"rs_ring_boundary_m{m}",
          ["--op", "gemm-rs", "--m", str(m), "--n", "4096", "--k", "1536",
           "--router-profile", "current", "--warmup", "2", "--repeats", "1",
           "--iterations", "3"], binary="uccl_lite_router_fusion_8l4", timeout=120)
for k in (2048, 8192):
    run_case(f"mini_boundary_k{k}",
          ["--op", "ag-gemm", "--m", "64", "--n", "4096", "--k", str(k),
           "--router-profile", "current", "--warmup", "2", "--repeats", "1",
           "--iterations", "3"], binary="uccl_lite_router_fusion_8l4", timeout=120)

shapes = [(m, n, k) for n, k in ((4096, 1536), (5120, 3200)) for m in (2048, 8192)]
for run in (1, 2):
    for m, n, k in (shapes if run == 1 else reversed(shapes)):
        for packet in ((128, 512) if run == 1 else (512, 128)):
            run_case(f"packet_ar_{run}_{m}_{n}_{packet}",
                  ["--op", "gemm-ar", "--m", str(m), "--n", str(n), "--k", str(k),
                   "--router-profile", "current", "--packet-kib", str(packet),
                   "--warmup", "2", "--repeats", "3", "--iterations", "10",
                   "--samples", str(DATA / f"packet_ar_{packet}_{run}.csv")],
                  binary="uccl_lite_router_fusion_8l4", timeout=120)

groups = {}
for path in DATA.glob("packet_ar_*.csv"):
    _, _, packet, run = path.stem.split("_")
    for row in csv.DictReader(path.open()):
        key = (int(packet), int(row["m"]), int(row["n"]), int(row["k"]), row["method"])
        groups.setdefault(key, {}).setdefault(run, []).append(float(row["latency_ms"]))
with (DATA / "packet_summary.csv").open("w") as stream:
    writer = csv.writer(stream)
    writer.writerow(("packet_kib", "m", "n", "k", "method", "run1_ms", "run2_ms", "median_ms"))
    for key, runs in sorted(groups.items()):
        assert set(runs) == {"1", "2"} and all(len(v) == 30 for v in runs.values()), key
        medians = [statistics.median(runs[run]) for run in ("1", "2")]
        writer.writerow((*key, *medians, statistics.median(medians)))
(DATA / "packet_candidate_script_hash.json").write_text(json.dumps({
    "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
}, indent=2) + "\n")
