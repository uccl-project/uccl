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

for ranks in (4, 8):
    r.run(f"host_ring_check_{ranks}", [str(ranks)],
          binary="uccl_lite_host_ring_check", timeout=120)

# These boundary correctness controls exercise the new benchmark wiring at RS
# ring endpoints and mini-host AG endpoints, outside the 24-shape comparison.
for m in (128, 1024):
    r.run(f"rs_ring_boundary_m{m}",
          ["--op", "gemm-rs", "--m", str(m), "--n", "4096", "--k", "1536",
           "--router-profile", "current", "--warmup", "2", "--repeats", "1",
           "--iterations", "3"], binary="uccl_lite_router_fusion_8l4", timeout=120)
for k in (2048, 8192):
    r.run(f"mini_boundary_k{k}",
          ["--op", "ag-gemm", "--m", "64", "--n", "4096", "--k", str(k),
           "--router-profile", "current", "--warmup", "2", "--repeats", "1",
           "--iterations", "3"], binary="uccl_lite_router_fusion_8l4", timeout=120)

shapes = [(m, n, k) for n, k in ((4096, 1536), (5120, 3200)) for m in (2048, 8192)]
for run in (1, 2):
    for m, n, k in (shapes if run == 1 else reversed(shapes)):
        for packet in ((128, 512) if run == 1 else (512, 128)):
            r.run(f"packet_ar_{run}_{m}_{n}_{packet}",
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
