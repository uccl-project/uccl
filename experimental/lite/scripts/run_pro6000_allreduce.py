#!/usr/bin/env python3
"""Locked calibration and frozen independent validation of FP32/C++ AllReduce."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
from statistics import median
import subprocess
import time

SIZES = [1 << p for p in range(10, 31)]
ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path("/home/wxzheng/eat-pcieccl-vllm-pro6000-20261115-074910")
CE = {
    "NCCL_P2P_USE_CUDA_MEMCPY": "1",
    "NCCL_P2P_LEVEL": "SYS",
    "NCCL_SHM_DISABLE": "1",
    "NCCL_PROTO": "Simple",
}
NCCL = {
    "ce-auto": CE,
    "ce-ring": {**CE, "NCCL_ALGO": "Ring"},
    "ce-tree": {**CE, "NCCL_ALGO": "Tree"},
}
for channels in (1, 2, 4, 8, 16):
    NCCL[f"ce-ring-ch{channels}"] = {
        **CE,
        "NCCL_ALGO": "Ring",
        "NCCL_MIN_NCHANNELS": str(channels),
        "NCCL_MAX_NCHANNELS": str(channels),
    }
for channels in (2, 4, 8):
    for buffer in (1, 16):
        NCCL[f"ce-ring-ch{channels}-b{buffer}m"] = {
            **NCCL[f"ce-ring-ch{channels}"],
            "NCCL_BUFFSIZE": str(buffer << 20),
        }
for channels in (2, 4, 8):
    NCCL[f"ce-ring-ch{channels}-t128"] = {
        **NCCL[f"ce-ring-ch{channels}"],
        "NCCL_NTHREADS": "128",
    }
for channels in (4, 8):
    NCCL[f"ce-ring-ch{channels}-b64m"] = {
        **NCCL[f"ce-ring-ch{channels}"],
        "NCCL_BUFFSIZE": str(64 << 20),
    }

OURS = {}
for blocks in (1, 4, 16, 32):
    for threads in (256, 1024):
        OURS[f"ll-{blocks}-{threads}"] = dict(
            mode="ll", blocks=blocks, threads=threads, end=4 << 20
        )
for blocks in (4, 16, 64):
    OURS[f"put-{blocks}"] = dict(mode="put", blocks=blocks, threads=256)
for blocks in (16, 64):
    OURS[f"drained-ce-{blocks}"] = dict(
        mode="drained-ce", blocks=blocks, threads=512, begin=1 << 20
    )
for lanes in (1, 4, 16):
    for tile in (1, 4, 16, 64):
        OURS[f"chunk-ce-{lanes}-{tile}m"] = dict(
            mode="chunk-ce", blocks=lanes, threads=512, tile=tile << 20, begin=1 << 20
        )

for lanes in (1, 4, 8, 16):
    for tile in (4, 16, 64):
        for workers in (4, 8):
            OURS[f"chunk-ce-{lanes}-{tile}m-w{workers}"] = dict(
                mode="chunk-ce",
                blocks=lanes,
                threads=256,
                tile=tile << 20,
                workers=workers,
                begin=1 << 20,
            )

for lanes in (1, 4, 8, 16):
    for tile in (4, 16, 64):
        OURS[f"chunk-ce-{lanes}-{tile}m-s3"] = dict(
            mode="chunk-ce",
            blocks=lanes,
            threads=256,
            tile=tile << 20,
            workers=8,
            streams=3,
            begin=1 << 20,
        )

for lanes in (1, 4, 8, 16):
    for tile in (4, 16, 64, 256):
        OURS[f"host-ce-{lanes}-{tile}m"] = dict(
            mode="host-ce",
            blocks=lanes,
            threads=256,
            tile=tile << 20,
            workers=8,
            begin=1 << 20,
        )
for lanes in (8, 16, 32):
    for tile in (1, 4):
        OURS[f"chunk-ce-{lanes}-{tile}m-w4-s3"] = dict(
            mode="chunk-ce",
            blocks=lanes,
            threads=256,
            tile=tile << 20,
            workers=4,
            streams=3,
            begin=1 << 20,
        )

for blocks in (32, 64, 128):
    OURS[f"staged-ce-{blocks}"] = dict(
        mode="staged-ce", blocks=blocks, threads=512, begin=1 << 20
    )
OURS["drained-ce-128"] = dict(mode="drained-ce", blocks=128, threads=512, begin=1 << 20)

# The first validation epoch found a reuse failure at an unselected small-size
# control point. Calibrate all repaired CE controls over the complete grid too.
for config in OURS.values():
    if config["mode"] in ("drained-ce", "staged-ce"):
        config["begin"] = 1024


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def parse(log, sizes):
    rows = {}
    ranks = {}
    for line in log.splitlines():
        f = line.split()
        if f and f[0] == "RESULT":
            if len(f) != 5:
                raise ValueError("Malformed result")
            size, us, bw, bad = int(f[1]), float(f[2]), float(f[3]), int(f[4])
            if (
                size in rows
                or bad
                or not math.isfinite(us)
                or us <= 0
                or not math.isfinite(bw)
            ):
                raise ValueError("Invalid or failed result: " + line)
            if abs(bw - size * 1.75 / (us * 1000)) > 0.0001:
                raise ValueError("Inconsistent bandwidth")
            rows[size] = us
        elif f and f[0] == "RANK":
            size, rank, us, bad = int(f[1]), int(f[2]), float(f[3]), int(f[4])
            if (size, rank) in ranks or bad or not math.isfinite(us) or us <= 0:
                raise ValueError("Invalid rank: " + line)
            ranks[size, rank] = us
    if set(rows) != set(sizes) or set(ranks) != {
        (s, r) for s in sizes for r in range(8)
    }:
        raise ValueError("Incomplete size/rank coverage")
    for s, us in rows.items():
        if us != max(ranks[s, r] for r in range(8)):
            raise ValueError("Not slowest rank")
    return rows


def transport(debug, label, sizes):
    logs = sorted(debug.glob(label + ".*.log"))
    if len(logs) != 8:
        raise ValueError(f"Expected eight NCCL rank logs, got {len(logs)}")
    edges = []
    selections = {}
    for p in logs:
        lines = [
            l for l in p.read_text().splitlines() if " via " in l and "Channel " in l
        ]
        if not lines or any("/CE" not in l or "P2P/" not in l for l in lines):
            raise ValueError("Non-CE or absent transport: " + str(p))
        edges.extend(lines)
        for size, algorithm, protocol in re.findall(
            r"AllReduce: (\d+) Bytes -> Algo (\w+) proto (\w+)", p.read_text()
        ):
            if protocol != "SIMPLE":
                raise ValueError(
                    "Collective is not using the CE-compatible Simple protocol"
                )
            selections[int(size)] = algorithm + "/" + protocol
    if set(selections) != set(sizes):
        raise ValueError("Missing actual NCCL algorithm/protocol selection for a size")
    return {
        "rank_logs": len(logs),
        "ce_edges": len(edges),
        "selections": selections,
        "hashes": {p.name: sha(p) for p in logs},
    }


class Runner:
    def __init__(self, output, placement):
        self.out = output
        output.mkdir(parents=True, exist_ok=True)
        (output / "debug").mkdir(exist_ok=True)
        self.env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("NCCL_", "VLLM_", "UCCL_"))
        }
        cuda = EXPERIMENT / "deps/cuda133-packages/nvidia/cu13"
        self.env.update(
            CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
            LD_LIBRARY_PATH=f"{cuda}/lib:{cuda}/lib64",
            NCCL_IB_DISABLE="1",
            NCCL_NET="Socket",
            UCCL_LITE_PLACEMENT=placement,
        )
        self.binaries = {
            "ours": ROOT / "build/uccl_lite_pro6000_ar",
            "nccl": ROOT / "build/uccl_lite_pro6000_ar_nccl",
        }
        self.identities = {
            str(p.relative_to(ROOT)): sha(p)
            for p in [
                *self.binaries.values(),
                ROOT / "src/collectives/pro6000_allreduce.cu",
                ROOT / "include/uccl_lite/pro6000_allreduce.cuh",
                ROOT / "include/uccl_lite/allreduce_validation.cuh",
            ]
        }
        self.platform = {
            "gpu_identity": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,pci.bus_id,driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            ),
            "topology": subprocess.check_output(
                ["nvidia-smi", "topo", "-m"], text=True
            ),
            "cpu_topology": subprocess.check_output(
                ["lscpu", "-p=CPU,SOCKET,NODE"], text=True
            ),
            "mscclpp_sha256": sha(EXPERIMENT / "deps/mscclpp/lib/libmscclpp.so"),
            "nccl_sha256": sha(
                EXPERIMENT
                / "env/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2"
            ),
            "dtype": "FP32",
            "operation": "in-place SUM",
            "ranks": 8,
            "warmups": 10,
            "continuous_calls": 20,
            "capacity": 1 << 30,
        }
        self.platform_sha256 = hashlib.sha256(
            json.dumps(self.platform, sort_keys=True).encode()
        ).hexdigest()
        self.records = []

    def run(self, label, backend, name, begin=None, end=None, seed=1, profile=False):
        if (self.out / f"{label}.json").exists():
            raise ValueError("Refusing to overwrite " + label)
        active = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        )
        for _ in range(10):
            if not active.strip():
                break
            time.sleep(0.2)
            active = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                text=True,
            )
        if active.strip():
            raise RuntimeError("GPU busy: " + active)
        config = (NCCL if backend == "nccl" else OURS)[name]
        lo, hi = begin or config.get("begin", 1024), end or config.get("end", 1 << 30)
        sizes = [s for s in SIZES if lo <= s <= hi]
        command = [
            str(self.binaries[backend]),
            "--begin",
            str(lo),
            "--end",
            str(hi),
            "--seed",
            str(seed),
        ]
        env = self.env.copy()
        if backend == "nccl":
            env.update(config)
            env.update(
                NCCL_DEBUG="INFO",
                NCCL_DEBUG_SUBSYS="INIT,ENV,GRAPH,TUNING,P2P,SHM",
                NCCL_DEBUG_FILE=str(self.out / "debug" / f"{label}.%h.%p.log"),
            )
        else:
            for k, v in config.items():
                if k not in ("begin", "end"):
                    command += ["--" + k, str(v)]
            if profile:
                command += ["--profile", "1"]
        for p, identity in self.identities.items():
            if sha(ROOT / p) != identity:
                raise RuntimeError("Source/binary changed during run: " + p)
        rec = dict(
            label=label,
            backend=backend,
            config=name,
            begin=lo,
            end=hi,
            seed=seed,
            command=command,
            started=now(),
            identities=self.identities,
            platform_sha256=self.platform_sha256,
            environment={
                k: v
                for k, v in env.items()
                if k.startswith(("NCCL_", "UCCL_", "CUDA_")) or k == "LD_LIBRARY_PATH"
            },
        )
        dump(self.out / f"{label}.json", rec)
        print("START " + label, flush=True)
        with (self.out / f"{label}.txt").open("x") as out:
            p = subprocess.Popen(
                command,
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                status = p.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                status = -9
            if status:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        rec.update(
            finished=now(), returncode=status, log_sha256=sha(self.out / f"{label}.txt")
        )
        values = None
        try:
            if status:
                raise ValueError("process return code " + str(status))
            values = parse((self.out / f"{label}.txt").read_text(), sizes)
            if backend == "nccl":
                rec["transport"] = transport(self.out / "debug", label, sizes)
            rec["values"] = {str(k): v for k, v in values.items()}
            rec["status"] = "pass"
        except Exception as e:
            rec.update(status="fail", reason=str(e))
            values = None
        dump(self.out / f"{label}.json", rec)
        self.records.append(rec)
        print(
            rec["status"].upper() + " " + label + " " + rec.get("reason", ""),
            flush=True,
        )
        return values

    def hardware(self):
        dump(self.out / "platform.json", self.platform)
        for relative in self.identities:
            if relative.startswith("build/"):
                continue
            target = self.out / "measured_source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / relative).read_bytes())
        for name, cmd in {
            "topology": ["nvidia-smi", "topo", "-m"],
            "gpu": ["nvidia-smi", "-q"],
            "cpu": ["lscpu"],
            "links-ours": ["ldd", str(self.binaries["ours"])],
            "links-nccl": ["ldd", str(self.binaries["nccl"])],
        }.items():
            (self.out / (name + ".txt")).write_bytes(
                subprocess.check_output(cmd, env=self.env)
            )
        (self.out / "measured_source" / "run_pro6000_allreduce.py").write_bytes(
            Path(__file__).read_bytes()
        )
        headers = EXPERIMENT / "deps/mscclpp/include/mscclpp"
        dump(
            self.out / "dependency-headers.json",
            {str(p): sha(p) for p in headers.rglob("*.hpp")},
        )
        (self.out / "mscclpp-version.hpp").write_bytes(
            (headers / "version.hpp").read_bytes()
        )
        libs = [
            EXPERIMENT / "deps/mscclpp/lib/libmscclpp.so",
            EXPERIMENT
            / "env/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2",
        ]
        dump(
            self.out / "identities.json",
            {
                "sources_binaries": self.identities,
                "libraries": {str(p): sha(p) for p in libs},
                "runner_sha256": sha(__file__),
            },
        )


def main():
    global EXPERIMENT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=["probe", "calibrate", "freeze", "validate", "profile"]
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", choices=["ours", "nccl", "both"], default="both")
    p.add_argument("--configs", help="Comma-separated candidate names")
    p.add_argument(
        "--placement", choices=["local-four", "single", "legacy"], default="local-four"
    )
    p.add_argument("--begin", type=int)
    p.add_argument("--end", type=int)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--calibration", type=Path)
    p.add_argument("--experiment", type=Path, default=EXPERIMENT)
    a = p.parse_args()
    EXPERIMENT = a.experiment.resolve()
    with open("/home/wxzheng/uccl-eval.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r = Runner(a.output.resolve(), a.placement)
        if a.stage in ("probe", "profile", "calibrate"):
            candidates = [
                (b, n)
                for b, cs in [("nccl", NCCL), ("ours", OURS)]
                for n in cs
                if a.backend in ("both", b)
                and (not a.configs or n in a.configs.split(","))
            ]
            if not candidates:
                raise ValueError("No candidates")
            for repeat in range(a.repeat if a.stage == "calibrate" else 1):
                for backend, name in candidates:
                    label = f"{a.stage}-{repeat+1}-{backend}-{name}"
                    if (a.output / f"{label}.json").exists():
                        raise ValueError("Refusing to overwrite " + label)
            r.hardware()
            for repeat in range(a.repeat if a.stage == "calibrate" else 1):
                random.Random(20261116 + repeat).shuffle(candidates)
                for b, n in candidates:
                    r.run(
                        f"{a.stage}-{repeat+1}-{b}-{n}",
                        b,
                        n,
                        a.begin,
                        a.end,
                        seed=101 + repeat,
                        profile=a.stage == "profile",
                    )
        elif a.stage == "freeze":
            if not a.calibration:
                raise ValueError("--calibration required")
            records = [
                json.loads(f.read_text())
                for f in a.calibration.glob("calibrate-*.json")
            ]
            policy = {}
            for b in ("ours", "nccl"):
                selected = {}
                for size in SIZES:
                    eligible = {}
                    for n in OURS if b == "ours" else NCCL:
                        rs = [
                            v for v in records if v["backend"] == b and v["config"] == n
                        ]
                        if len(rs) != a.repeat or any(
                            v["status"] != "pass" for v in rs
                        ):
                            continue
                        if any(
                            v["identities"] != r.identities
                            or v["platform_sha256"] != r.platform_sha256
                            for v in rs
                        ):
                            raise ValueError(
                                "Calibration source/binary identity mismatch"
                            )
                        if any(
                            v["environment"]["UCCL_LITE_PLACEMENT"] != a.placement
                            for v in rs
                        ):
                            raise ValueError("Placement mismatch")
                        if all(str(size) in v["values"] for v in rs):
                            eligible[n] = median(v["values"][str(size)] for v in rs)
                    if not eligible:
                        raise ValueError(f"No eligible {b} {size}")
                    selected[str(size)] = min(eligible, key=eligible.get)
                policy[b] = selected
            frozen = dict(
                created=now(),
                policy=policy,
                identities=r.identities,
                placement=a.placement,
                platform=r.platform,
                platform_sha256=r.platform_sha256,
                nccl_candidates=NCCL,
                ours_candidates=OURS,
                calibration_hashes={
                    f.name: sha(f) for f in a.calibration.glob("calibrate-*")
                },
            )
            target = a.output / "frozen.json"
            if target.exists():
                raise ValueError("Refusing to replace frozen policy")
            dump(target, frozen)
        else:
            frozen = json.loads((a.output / "frozen.json").read_text())
            if (
                frozen["identities"] != r.identities
                or frozen["placement"] != a.placement
                or frozen["platform_sha256"] != r.platform_sha256
            ):
                raise ValueError("Frozen identities/placement changed")
            if (a.output / "validation-start.json").exists():
                raise ValueError("Refusing to replace an existing validation window")
            frozen_hash = sha(a.output / "frozen.json")
            r.hardware()
            dump(
                a.output / "validation-start.json",
                dict(started=now(), frozen_sha256=frozen_hash),
            )
            # Every chosen fixed configuration gets a fresh full sweep. The
            # composite reads only the pre-selected size/config pairs.
            for repeat in range(1, 6):
                order = ["ours", "nccl"] if repeat % 2 else ["nccl", "ours"]
                names = {b: sorted(set(frozen["policy"][b].values())) for b in order}
                for index in range(max(map(len, names.values()))):
                    for b in order:
                        if index >= len(names[b]):
                            continue
                        n = names[b][index]
                        # LL has a capacity limit; every selected size is covered.
                        end = OURS[n].get("end", 1 << 30) if b == "ours" else 1 << 30
                        if (
                            r.run(
                                f"validation-{repeat}-{b}-{n}",
                                b,
                                n,
                                1024,
                                end,
                                seed=1001 + repeat,
                            )
                            is None
                        ):
                            raise RuntimeError(
                                "Validation failed; no replacement or hidden retry"
                            )
            if sha(a.output / "frozen.json") != frozen_hash:
                raise RuntimeError("Policy changed")
            dump(
                a.output / "validation-complete.json",
                dict(finished=now(), frozen_sha256=frozen_hash),
            )


if __name__ == "__main__":
    main()
