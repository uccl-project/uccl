"""Calibrate exact NCCL shapes, including both layouts, with fresh ABBA jobs."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys

from dropin_profile import HERE, identity, launch_environment, sha256, write_profile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-nccl", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cases", type=Path, help="JSON list of [op, NCCL element count]; 1=AG, 2=RS, 3=AR")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iterations", type=int, default=12)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    args.real_nccl = args.real_nccl.resolve(strict=True)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    sizes = [65536, 262144, 1048576, 4194304, 16777216, 67108864]
    cases = json.loads(args.cases.read_text()) if args.cases else (
        [[op, size // (2 if op == 3 else 16)] for op in (1, 2, 3) for size in sizes] +
        [[1, 4227328], [2, 4227328], [1, 6291968], [2, 6291968],
         [3, 67112960], [3, 67115008], [3, 33820672]])
    cases = sorted(set(tuple(x) for x in cases))
    case_file = args.out_dir / "cases.json"
    case_file.write_text(json.dumps(cases))
    ident, details = identity(args.real_nccl)
    manifest = {"identity": ident, "details": details, "argv": sys.argv, "trials": [],
                "source_sha256": {str(p): sha256(p) for p in
                                  (HERE / "bench_dropin.py", Path(__file__), HERE / "dropin.cc")}}
    candidate = args.out_dir / "candidate.tsv"
    write_profile(candidate, ident, details, args.real_nccl,
                  [[op, count, 1, 1, args.repeats] for op, count in cases], candidate=True)
    trials = []
    # Match the integration testbed's existing lock; never overlap GPU trials.
    with open("/tmp/uccl-lite-integration-wxz-l4.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for index, backend in enumerate(("nccl", "packet", "packet", "nccl")):
            label = f"{index:02d}_{backend}"
            out = args.out_dir / f"{label}.jsonl"
            env = launch_environment(args.real_nccl, candidate if backend == "packet" else None,
                                     calibrating=backend == "packet")
            if backend == "nccl":
                env["LD_PRELOAD"] = str(args.real_nccl)
                env["SGLANG_NCCL_SO_PATH"] = str(args.real_nccl)
            env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=8",
                       str(HERE / "bench_dropin.py"), "--cases", str(case_file), "--out", str(out),
                       "--warmup", str(args.warmup), "--iterations", str(args.iterations), "--repeats", str(args.repeats)]
            print(f"Starting {label}", flush=True)
            with (args.out_dir / f"{label}.log").open("w") as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = child.wait(timeout=1800)
                finally:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
            manifest["trials"].append({"backend": backend, "command": command, "returncode": code})
            (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            if code:
                raise RuntimeError(f"{label} failed: see {args.out_dir / (label + '.log')}")
            trials.append([json.loads(line) for line in out.read_text().splitlines()])
    records, selected = [], []
    for op, count in cases:
        layouts = {}
        for layout, alignment in [(layout, a) for a in (16, 2, 4, 8) for layout in ("out", "inplace")]:
            jobs = []
            for trial in trials:
                row = next(r for r in trial if (r["op"], r["count"], r["layout"], r["alignment"]) == (op, count, layout, alignment))
                assert row["correct"] and row["repeats"] >= 3
                jobs.append(statistics.median(row["repeat_ms"]))
            layouts[f"{layout}_{alignment}"] = {"jobs_ms": jobs, "nccl_ms": min(jobs[0], jobs[3]),
                               "packet_ms": max(jobs[1], jobs[2])}
        # Both independent packet jobs must beat both controls for BOTH layouts.
        gain = min(r["nccl_ms"] / r["packet_ms"] for r in layouts.values())
        accepted = gain >= 1.05
        records.append({"op": op, "count": count, "layouts": layouts, "minimum_speedup": gain, "selected": accepted})
    # The TSV stores actual conservative timing for the limiting layout.
    selected = []
    for r in records:
        if r["selected"]:
            limiting = min(r["layouts"].values(), key=lambda x: x["nccl_ms"] / x["packet_ms"])
            selected.append([r["op"], r["count"], limiting["nccl_ms"], limiting["packet_ms"], args.repeats])
    write_profile(args.out_dir / "profile.tsv", ident, details, args.real_nccl, selected)
    (args.out_dir / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"Selected {len(selected)}/{len(cases)} exact shapes; profile: {args.out_dir / 'profile.tsv'}")


if __name__ == "__main__":
    main()
