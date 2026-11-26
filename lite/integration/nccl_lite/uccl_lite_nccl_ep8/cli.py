"""Run an application with NCCL or the validated L4 EP8 proxy profile."""

import argparse
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys


SCRIPTS = {
    "packet": "run_packet_8l4.sh",
    "sglang-ag-gate": "run_sglang_ag_gate_8l4.sh",
    "sglang-ag-ready-rs": "run_sglang_ag_ready_rs_8l4.sh",
    "sglang-ready-combine": "run_sglang_ready_combine_8l4.sh",
    "sglang-ready-rs": "run_sglang_ready_rs_8l4.sh",
    "sglang-rs-sum2": "run_sglang_rs_sum2_8l4.sh",
}


def _real_nccl(requested):
    if not requested:
        requested = os.environ.get("UCCL_LITE_REAL_NCCL")
    if not requested:
        try:
            distribution = importlib.metadata.distribution("nvidia-nccl-cu12")
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(
                "pass --real-nccl with the absolute path to PyTorch's libnccl.so.2"
            ) from exc
        requested = distribution.locate_file("nvidia/nccl/lib/libnccl.so.2")
    path = Path(requested).expanduser().resolve()
    if not path.is_file() or not path.name.startswith("libnccl.so"):
        raise ValueError(f"real NCCL library not found: {path}")
    return path


def _check_l4_ep8():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("could not inspect GPUs for the l4-ep8 profile") from exc
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 8 or any(row != "NVIDIA L4, 8.9" for row in rows):
        raise ValueError("the l4-ep8 profile requires eight visible NVIDIA L4 GPUs")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and len([device for device in visible.split(",") if device.strip()]) != 8:
        raise ValueError("the l4-ep8 profile requires eight CUDA-visible GPUs")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run any EP configuration with NCCL compatibility or a validated UCCL Lite profile"
    )
    parser.add_argument(
        "--mode", choices=("compatible", "nccl", *SCRIPTS), default="compatible"
    )
    parser.add_argument(
        "--profile",
        choices=("l4-ep8",),
        help="required for accelerated modes; asserts the tested single-node 8xL4 setup",
    )
    parser.add_argument(
        "--real-nccl",
        help="absolute path to the libnccl.so.2 used by PyTorch; "
        "defaults to UCCL_LITE_REAL_NCCL or the installed nvidia-nccl-cu12 wheel",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if args.mode == "nccl":
        if args.profile:
            parser.error("--profile is only used with an accelerated mode")
        os.execvpe(command[0], command, os.environ.copy())
    if args.mode == "compatible":
        if args.profile:
            parser.error("--profile is only used with an accelerated mode")
    else:
        if args.profile != "l4-ep8":
            parser.error("accelerated modes require the explicit --profile l4-ep8")
        try:
            _check_l4_ep8()
        except ValueError as exc:
            parser.error(str(exc))

    package = Path(__file__).resolve().parent
    proxy = package / "libuccl_lite_nccl_compat.so"
    script = package / SCRIPTS[args.mode] if args.mode in SCRIPTS else None
    hooks = package / "sglang_hooks" / "sitecustomize.py"
    if not proxy.is_file() or (script is not None and not script.is_file()):
        parser.error("the wheel is missing its native library or launch script")
    if args.mode.startswith("sglang-") and not hooks.is_file():
        parser.error("the wheel is missing its SGLang hooks")
    try:
        real_nccl = _real_nccl(args.real_nccl)
    except ValueError as exc:
        parser.error(str(exc))
    if real_nccl == proxy:
        parser.error("--real-nccl cannot point to the proxy library")

    environment = os.environ.copy()
    environment["UCCL_LITE_COMPAT_SO"] = str(proxy)
    environment["UCCL_LITE_REAL_NCCL"] = str(real_nccl)
    if args.mode == "compatible":
        preload = environment.get("LD_PRELOAD")
        environment["LD_PRELOAD"] = str(proxy) + (":" + preload if preload else "")
        environment["SGLANG_NCCL_SO_PATH"] = str(real_nccl)
        environment["UCCL_LITE_NCCL_ENABLE"] = "0"
        os.execvpe(command[0], command, environment)
    os.execvpe("/bin/bash", ["bash", str(script), *command], environment)


if __name__ == "__main__":
    sys.exit(main())
