#!/usr/bin/env python3
"""Run isolated SGLang controls and packet trials with identical server flags."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--real-nccl", type=Path, required=True)
    parser.add_argument("--backends", nargs="+", default=["nccl", "packet", "packet", "nccl"],
                        choices=["nccl", "packet"])
    parser.add_argument("--cases", nargs="+", default=["8:2048:1", "32:1024:1", "32:2048:1"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--case-warmups", type=int, default=1)
    parser.add_argument("--max-bytes", type=int, default=134217728)
    parser.add_argument("--port", type=int, default=30089)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("server_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError:
            parser.error(f"port {args.port} is already in use")
    args.real_nccl = args.real_nccl.resolve(strict=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.exists():
        parser.error("out-dir already contains a run; choose a fresh directory")
    if args.repeats < 1 or args.case_warmups < 0:
        parser.error("repeats must be positive and case-warmups nonnegative")
    if not 16 <= args.max_bytes <= 1 << 30:
        parser.error("max-bytes must be between 16 bytes and 1 GiB")
    server_args = args.server_args
    if server_args[:1] == ["--"]:
        server_args = server_args[1:]
    env = os.environ.copy()
    env.update(UCCL_LITE_REAL_NCCL=str(args.real_nccl),
               UCCL_LITE_NCCL_MAX_BYTES=str(args.max_bytes),
               PORT=str(args.port), SGLANG_PYTHON=sys.executable,
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if any(key in env for key in ("UCCL_LITE_NCCL_TRACE", "UCCL_LITE_MP_PACKET_PROFILE",
                                  "UCCL_LITE_NCCL_TRACE_SHAPES")):
        parser.error("unset tracing/profiling variables for timing")
    with (args.out_dir / "gpu.txt").open("w") as gpu_info:
        subprocess.run(["nvidia-smi", "-q"], stdout=gpu_info, check=True)
    manifest = {
        "argv": sys.argv, "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
        "environment": {k: v for k, v in env.items() if k.startswith(
            ("UCCL_", "NCCL_", "SGLANG_", "CUDA_", "OMP_", "MKL_", "LD_", "PYTHON"))},
        "sha256": {str(p): sha256(p) for p in [
            args.real_nccl, HERE / "build/libuccl_lite_nccl_compat.so",
            HERE / "bench_e2e_workloads.py", HERE / "launch_glm45_air.sh",
            HERE / "run_packet_8l4.sh"]},
        "phases": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for phase, backend in enumerate(args.backends):
        label = f"{phase:02d}_{backend}"
        command = [str(HERE / "run_packet_8l4.sh"), "env",
                   f"UCCL_LITE_NCCL_ENABLE={int(backend == 'packet')}",
                   str(HERE / "launch_glm45_air.sh"), *server_args]
        record = {"label": label, "command": command, "complete": False}
        manifest["phases"].append(record)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Starting {label}", flush=True)
        with (args.out_dir / f"{label}_server.log").open("w") as log:
            server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            try:
                deadline = time.monotonic() + args.startup_timeout
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f"{label} exited with {server.returncode}; see server log")
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2):
                            break
                    except (urllib.error.URLError, TimeoutError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"{label} startup timeout")
                        time.sleep(1)
                cases = args.cases if phase % 2 == 0 else list(reversed(args.cases))
                client = [sys.executable, str(HERE / "bench_e2e_workloads.py"),
                          "--url", f"http://127.0.0.1:{args.port}/generate",
                          "--backend", backend, "--run-id", label,
                          "--cases", *cases, "--repeats", str(args.repeats),
                          "--warmup-repeats", "1", "--case-warmups", str(args.case_warmups),
                          "--out", str(args.out_dir / "samples.jsonl")]
                if env.get("MODEL_PATH"):
                    client.extend(["--model", env["MODEL_PATH"]])
                record["client"] = client
                subprocess.run(client, check=True, timeout=7200)
                record["complete"] = True
            finally:
                # Only signal this server's newly created process group.
                try:
                    os.killpg(server.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    pass
                try:
                    os.killpg(server.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                server.wait()
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                time.sleep(3)


if __name__ == "__main__":
    main()
