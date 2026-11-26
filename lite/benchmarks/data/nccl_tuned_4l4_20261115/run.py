"""Calibrate NCCL SM/CE separately, freeze choices, then validate fresh runs."""
import argparse
import collections
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "build/nccl_tuned_4l4_20261115"
MPI = Path("/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun")
SIZES = [1 << p for p in range(10, 31)]
OPS = ("allreduce", "allgather", "reducescatter")
CONFIGS = {
    "sm_default": {},
    "sm_p2p": {},
    "sm_buffer1m": {"NCCL_BUFFSIZE": "1048576"},
    "sm_buffer8m": {"NCCL_BUFFSIZE": "8388608"},
    "sm_channels4": {"NCCL_MIN_NCHANNELS": "4", "NCCL_MAX_NCHANNELS": "4"},
    "sm_channels8": {"NCCL_MIN_NCHANNELS": "8", "NCCL_MAX_NCHANNELS": "8"},
    "sm_threads128": {"NCCL_NTHREADS": "128"},
    "sm_ring_simple": {"NCCL_ALGO": "Ring", "NCCL_PROTO": "Simple"},
    "sm_peer_read": {"NCCL_P2P_READ_ENABLE": "1"},
    "sm_native_buffer1m": {"NCCL_BUFFSIZE": "1048576"},
    "sm_native_buffer8m": {"NCCL_BUFFSIZE": "8388608"},
    "sm_native_channels4": {"NCCL_MIN_NCHANNELS": "4", "NCCL_MAX_NCHANNELS": "4"},
    "sm_native_channels8": {"NCCL_MIN_NCHANNELS": "8", "NCCL_MAX_NCHANNELS": "8"},
    "ce_simple": {},
    "ce_buffer1m": {"NCCL_BUFFSIZE": "1048576"},
    "ce_buffer8m": {"NCCL_BUFFSIZE": "8388608"},
    "ce_buffer16m": {"NCCL_BUFFSIZE": "16777216"},
    "ce_channels2": {"NCCL_MIN_NCHANNELS": "2", "NCCL_MAX_NCHANNELS": "2"},
    "ce_channels4": {"NCCL_MIN_NCHANNELS": "4", "NCCL_MAX_NCHANNELS": "4"},
    "ce_channels8": {"NCCL_MIN_NCHANNELS": "8", "NCCL_MAX_NCHANNELS": "8"},
    "ce_buffer8m_channels8": {"NCCL_BUFFSIZE": "8388608", "NCCL_MIN_NCHANNELS": "8", "NCCL_MAX_NCHANNELS": "8"},
}
for name, settings in CONFIGS.items():
    settings["NCCL_P2P_USE_CUDA_MEMCPY"] = "1" if name.startswith("ce_") else "0"
    if name != "sm_default" and not name.startswith("sm_native_"):
        settings["NCCL_P2P_LEVEL"] = "SYS"
    if name.startswith("ce_"):
        # NCCL 2.28.7's CE transport copies only Simple-protocol payloads.
        settings.update(NCCL_PROTO="Simple", NCCL_ALGO="Ring")


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def idle():
    active = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"
    ], text=True).strip()
    if active:
        raise RuntimeError("GPU occupied: " + active)


def samples(path, op):
    result = collections.defaultdict(list)
    for line in path.read_text().splitlines():
        fields = line.split(",")
        if op == "allreduce" and line.startswith("nccl,timed,"):
            assert int(fields[5]) == 0, "incorrect output"
            result[int(fields[3])].append(float(fields[4]))
        elif op != "allreduce" and line.startswith("# sample,"):
            assert fields[3] == "nccl"
            result[int(fields[2])].append(float(fields[6]))
    assert sorted(result) == SIZES, f"incomplete sweep: {path}"
    assert all(len(values) == 10 and min(values) > 0 for values in result.values())
    return {str(size): statistics.mean(values) for size, values in result.items()}


def run(stage, config, op, repeat):
    name = f"{stage}_{config}_{op}_{repeat}"
    record = OUT / (name + ".json")
    if record.exists():
        return json.loads(record.read_text())
    idle()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NCCL_", "UCCL_"))}
    env.update(CONFIGS[config], CUDA_VISIBLE_DEVICES="0,1,2,3", NCCL_DEBUG="INFO",
               NCCL_DEBUG_SUBSYS="INIT,ENV,P2P,SHM,TUNING" if stage == "calibrate" else "INIT,ENV,P2P,SHM",
               NCCL_DEBUG_FILE=str(OUT / (name + ".debug.%p.log")))
    if op == "allreduce":
        command = [str(MPI), "--host", "localhost:4", "-np", "4", "--map-by", "ppr:4:node",
                   "--bind-to", "none", "--mca", "pml", "ob1", "--mca", "btl", "self,vader,tcp",
                   str(ROOT / "build/uccl_lite_nccl_4l4_mpi_baseline"), "--op", op,
                   "-b", "1024", "-e", str(1 << 30), "-w", "5", "-n", "10"]
    else:
        command = [str(OUT / "peer_nccl"), "--ag-only" if op == "allgather" else "--rs-only",
                   "-b", "1024", "-e", str(1 << 30), "-w", "20", "-n", "10", "-r", "1"]
    started = datetime.now(timezone.utc).isoformat()
    print(name + ": starting", flush=True)
    with (OUT / (name + ".csv")).open("w") as stdout, (OUT / (name + ".stderr")).open("w") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        try:
            rc = process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            rc = 124
    result = dict(name=name, stage=stage, config=config, operation=op, repeat=repeat,
                  command=command, environment=CONFIGS[config], returncode=rc,
                  started_utc=started, ended_utc=datetime.now(timezone.utc).isoformat())
    if rc == 0:
        result["means_us"] = samples(OUT / (name + ".csv"), op)
    dump(name + ".json", result)
    print(name + (": PASS" if rc == 0 else f": FAIL {rc} (retained)"), flush=True)
    if rc == 124:
        time.sleep(2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("calibrate", "evaluate", "confirm"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dump("candidates.json", CONFIGS)
    paths = [ROOT / "build/uccl_lite_nccl_4l4_mpi_baseline", OUT / "peer_nccl",
             ROOT / "benchmarks/collectives/nccl_4l4_mpi_baseline.cu", OUT / "peer_nccl.cu",
             Path.home() / "third_party/nccl/build/lib/libnccl.so.2"]
    dump(args.stage + "_hashes.json", {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    if args.stage == "calibrate":
        results = []
        for repeat in range(2):
            configs = list(CONFIGS)
            random.Random(202611150 + repeat).shuffle(configs)
            for config in configs:
                for op in OPS if repeat == 0 else reversed(OPS):
                    results.append(run(args.stage, config, op, repeat))
        schedule = []
        for op in OPS:
            for size in SIZES:
                for family in ("sm", "ce"):
                    scores = {}
                    for config in CONFIGS:
                        if not config.startswith(family + "_"):
                            continue
                        matches = [r for r in results if r["operation"] == op and r["config"] == config]
                        if len(matches) == 2 and all(r["returncode"] == 0 for r in matches):
                            scores[config] = statistics.median(r["means_us"][str(size)] for r in matches)
                    if not scores:
                        raise RuntimeError(f"No valid {family} candidate for {op}")
                    eligible = dict(scores)
                    if family == "sm":
                        control = {r["repeat"]: r["means_us"][str(size)] for r in results
                                   if r["operation"] == op and r["config"] == "sm_default"}
                        # Retain default settings unless calibration shows a
                        # repeatable improvement beyond small timing noise.
                        for candidate in scores:
                            if candidate == "sm_default":
                                continue
                            trials = [r for r in results if r["operation"] == op and r["config"] == candidate]
                            if (scores[candidate] > scores["sm_default"] * 0.97 or
                                not all(r["means_us"][str(size)] < control[r["repeat"]] for r in trials)):
                                del eligible[candidate]
                    best = min(eligible, key=eligible.get)
                    schedule.append(dict(operation=op, bytes=size, family=family, config=best,
                                         calibration_us=scores[best]))
        dump("schedule.json", schedule)
    elif args.stage == "evaluate":
        schedule = json.loads((OUT / "schedule.json").read_text())
        # Rerun entire ascending sweeps for all selected configurations. This keeps
        # allocation, communicator lifetime and size order identical to calibration.
        jobs = {(r["operation"], r["config"]) for r in schedule}
        jobs.update((op, "sm_default") for op in OPS)
        for repeat in range(3):
            ordered = sorted(jobs)
            random.Random(202611160 + repeat).shuffle(ordered)
            for op, config in ordered:
                result = run(args.stage, config, op, repeat)
                if result["returncode"] != 0:
                    raise RuntimeError("Frozen candidate failed independent validation")
    else:
        schedule = json.loads((OUT / "schedule.json").read_text())
        final = []
        for selected in schedule:
            row = dict(selected, calibrated_configuration=selected["config"],
                       selection_reason="calibrated configuration")
            if row["family"] == "sm" and row["config"] != "sm_default":
                own, default = [], []
                for repeat in range(3):
                    for config, values in ((row["config"], own), ("sm_default", default)):
                        record = json.loads((OUT / f'evaluate_{config}_{row["operation"]}_{repeat}.json').read_text())
                        assert record["returncode"] == 0
                        values.append(record["means_us"][str(row["bytes"])])
                if (statistics.median(own) > statistics.median(default) * .97 or
                    not all(a < b for a, b in zip(own, default))):
                    row["config"] = "sm_default"
                    row["selection_reason"] = "default fallback: validation gain not repeatable"
            final.append(row)
        dump("final_schedule.json", final)
        # Only the default fallback changed. Confirm it in three new complete
        # sweeps per operation; do not select another winner from validation.
        for repeat in range(3):
            for op in OPS if repeat % 2 == 0 else reversed(OPS):
                result = run(args.stage, "sm_default", op, repeat)
                if result["returncode"] != 0:
                    raise RuntimeError("Default fallback confirmation failed")


if __name__ == "__main__":
    main()
