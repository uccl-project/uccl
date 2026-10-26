#!/usr/bin/env python3
"""Compile the production MCCL callbacks against host-only transport doubles."""

import argparse
import os
import pathlib
import shlex
import shutil
import subprocess
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[2]
MCCL = ROOT / "collective/mccl"
RDMA = ROOT / "collective/rdma"


def copy_section(source, start, end, target):
    """Compile verbatim production definitions, failing if boundaries change."""
    text = source.read_text()
    begin = text.index(start)
    finish = text.index(end, begin)
    target.write_text(text[begin:finish])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitize", choices=("thread", "address"))
    parser.add_argument("--backend", choices=("musa", "cuda", "hip"), default="musa")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="uccl-mccl-behavior-") as directory:
        tmp = pathlib.Path(directory)
        # Copy unchanged so quoted transport includes resolve to the doubles,
        # not to the SDK-dependent headers next to the production source.
        shutil.copy2(RDMA / "mccl_plugin_musa.cc", tmp)
        shutil.copy2(RDMA / "util_buffpool.h", tmp)
        # The full transport TU requires a real verbs/GPU installation. Compile
        # its unchanged completion methods against the controlled CQ below.
        # No function-body rewriting or duplicate completion algorithm.
        copy_section(
            RDMA / "transport.cc",
            "void UcclFlow::poll_flow_cq(",
            "void UcclFlow::rc_recv(",
            tmp / "flow_completion.inc",
        )
        copy_section(
            RDMA / "transport.cc",
            "bool RDMAEndpoint::uccl_poll_ureq_once(",
            "int RDMAEndpoint::uccl_recv_async(",
            tmp / "endpoint_completion.inc",
        )
        copy_section(
            RDMA / "transport.h",
            "  inline int check_need_flush(",
            "  /**",
            tmp / "flush_check.inc",
        )
        copy_section(
            RDMA / "rdma_io.h",
            "enum ReqType",
            "/**\n * @brief Each RDMAContext",
            tmp / "request.inc",
        )
        for name in (
            "transport.h",
            "transport_config.h",
            "util/debug.h",
            "util/gpu_rt.h",
            "util_rdma.h",
            "infiniband/verbs.h",
        ):
            path = tmp / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('#include "host_transport.h"\n')
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        flags = ["-std=c++17", "-g", "-O1", "-pthread"]
        if args.backend == "musa":
            flags += ["-DUCCL_USE_MUSA"]
        elif args.backend == "hip":
            flags += ["-D__HIP_PLATFORM_AMD__"]
        if args.sanitize:
            flags += [f"-fsanitize={args.sanitize}", "-fno-omit-frame-pointer"]
        failures = []
        for test, cases in (
            ("connection_test", [None]),
            ("flush_test", ["fresh", "stale", "pending", "noop", "noop_stale", "last"]),
        ):
            executable = tmp / test
            subprocess.run(
                compiler
                + flags
                + [
                    "-I",
                    str(tmp),
                    "-I",
                    str(MCCL / "tests"),
                    "-I",
                    str(MCCL),
                    str(MCCL / f"tests/{test}.cc"),
                    "-o",
                    str(executable),
                ],
                check=True,
            )
            for case in cases:
                command = [str(executable)] + ([case] if case else [])
                result = subprocess.run(command, timeout=120)
                if result.returncode:
                    failures.append((test, case, result.returncode))
        if failures:
            raise SystemExit(f"Behavior tests failed: {failures}")


if __name__ == "__main__":
    main()
