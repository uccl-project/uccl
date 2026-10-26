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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitize", choices=("thread", "address"))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="uccl-mccl-behavior-") as directory:
        tmp = pathlib.Path(directory)
        # Copy unchanged so quoted transport includes resolve to the doubles,
        # not to the SDK-dependent headers next to the production source.
        shutil.copy2(RDMA / "mccl_plugin_musa.cc", tmp)
        shutil.copy2(RDMA / "util_buffpool.h", tmp)
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
        if args.sanitize:
            flags += [f"-fsanitize={args.sanitize}", "-fno-omit-frame-pointer"]
        executable = tmp / "connection_test"
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
                str(MCCL / "tests/connection_test.cc"),
                "-o",
                str(executable),
            ],
            check=True,
        )
        subprocess.run([str(executable)], check=True, timeout=120)


if __name__ == "__main__":
    main()
