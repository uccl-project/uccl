"""Local identity and launcher for the calibrated NCCL ABI drop-in."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

HERE = Path(__file__).resolve().parent
LIBRARY = HERE / "build/libuccl_lite_nccl.so"

# One fixed configuration; included in the profile identity. The launcher does
# not change NCCL algorithms, transports, channels, or graph/application flags.
PACKET = {
    "AG": "1", "AG_PIPELINE": "1", "RS": "1", "RS_CPU": "1",
    "RS_GPU_MAX_BYTES": "65536", "RS_RETAIN": "1", "AR": "1",
    "AR_GPU_MAX_BYTES": "16384", "SPLIT_CREDITS": "1", "SCRATCH": "1",
    "HIERARCHICAL": "1", "HIERARCHICAL_RS": "1", "PARALLEL_AR": "1",
    "SPIN": "0", "WORKER_CPUS": "5,11,17,23,29,35,41,47",
    "HELPER_CPUS": "4,10,16,22,28,34,40,46",
}


def packet_environment():
    result = {"UCCL_LITE_MP_PACKET_" + k: v for k, v in PACKET.items()}
    result.update(UCCL_LITE_MP_BIND_PACKET_NUMA="1",
                  UCCL_LITE_MP_BIND_REMOTE_PARTIAL="0",
                  UCCL_LITE_MP_ISOLATE_PACKET_WORKER="1")
    return result


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(real_nccl):
    def command(*args):
        return subprocess.check_output(args, text=True).strip()
    env = {k: v for k, v in os.environ.items() if k.startswith(
        ("NCCL_", "CUDA_")) and k not in ("NCCL_DEBUG", "NCCL_DEBUG_SUBSYS", "NCCL_DEBUG_FILE")}
    details = {
        "hostname": platform.node(), "kernel": platform.release(),
        "gpus": command("nvidia-smi", "--query-gpu=uuid,name,pci.bus_id,driver_version", "--format=csv,noheader"),
        "topology": command("nvidia-smi", "topo", "-m"),
        "cpus": command("lscpu", "-p=CPU,CORE,SOCKET,NODE"),
        "cpu_model": next(x for x in Path("/proc/cpuinfo").read_text().splitlines() if x.startswith("model name")),
        "launch_affinity": sorted(os.sched_getaffinity(0)),
        "environment": env, "packet": packet_environment(),
        "nccl_sha256": sha256(real_nccl), "library_sha256": sha256(LIBRARY),
        "cudart_sha256": sha256("/usr/local/cuda/lib64/libcudart.so"),
        "launcher_sha256": sha256(__file__),
    }
    return hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest(), details


def versions(real_nccl):
    runtime = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
    values = []
    for library, name in ((runtime, "cudaDriverGetVersion"),
                          (runtime, "cudaRuntimeGetVersion"),
                          (ctypes.CDLL(str(real_nccl)), "ncclGetVersion")):
        value = ctypes.c_int()
        if getattr(library, name)(ctypes.byref(value)):
            raise RuntimeError(f"{name} failed")
        values.append(value.value)
    return values


def write_profile(path, ident, details, real_nccl, rows, candidate=False):
    uuids = [line.split(",")[0].strip().removeprefix("GPU-").replace("-", "")
             for line in details["gpus"].splitlines()]
    if len(uuids) != 8 or any("NVIDIA L4" not in line for line in details["gpus"].splitlines()):
        raise ValueError("calibration requires the eight-L4 testbed")
    text = ("UCCL_LITE_CANDIDATE_V1" if candidate else "UCCL_LITE_DROPIN_V1") + " " + ident + "\n"
    text += " ".join(map(str, versions(real_nccl))) + "\n"
    text += "\n".join(uuids) + "\n"
    text += "".join(" ".join(map(str, row)) + "\n" for row in rows)
    Path(path).write_text(text)


def launch_environment(real_nccl, profile=None, calibrating=False):
    real_nccl = Path(real_nccl).resolve(strict=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("UCCL_LITE_MP_")
           and not k.startswith("UCCL_LITE_DROPIN_")}
    env.update(packet_environment())
    env["UCCL_LITE_REAL_NCCL"] = str(real_nccl)
    other_preloads = [v for v in os.environ.get("LD_PRELOAD", "").replace(":", " ").split()
                      if "libnccl" not in v and "libuccl_lite_nccl" not in v]
    env["LD_PRELOAD"] = ":".join([str(LIBRARY), str(real_nccl), *other_preloads])
    env["SGLANG_NCCL_SO_PATH"] = str(LIBRARY)
    if profile:
        ident, _ = identity(real_nccl)
        header = Path(profile).read_text().splitlines()[0].split()
        expected = "UCCL_LITE_CANDIDATE_V1" if calibrating else "UCCL_LITE_DROPIN_V1"
        if header != [expected, ident]:
            print("UCCL Lite: profile identity mismatch; using NCCL", file=sys.stderr)
        else:
            env["UCCL_LITE_DROPIN_PROFILE"] = str(Path(profile).resolve())
            env["UCCL_LITE_DROPIN_IDENTITY"] = ident
            if calibrating:
                env["UCCL_LITE_DROPIN_CALIBRATING"] = "1"
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-nccl", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required")
    env = launch_environment(args.real_nccl, args.profile)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
