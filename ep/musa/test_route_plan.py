#!/usr/bin/env python3
"""Compile the SDK-independent EP plan and oracle tests."""

import argparse
import os
import pathlib
import shlex
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitize", choices=("address", "undefined"))
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="uccl-ep-plan-") as directory:
        executable = pathlib.Path(directory) / "route_plan_test"
        flags = ["-std=c++17", "-O1", "-g", "-Wall", "-Wextra", "-Werror"]
        if args.sanitize:
            flags += [f"-fsanitize={args.sanitize}", "-fno-omit-frame-pointer"]
        subprocess.run(
            shlex.split(os.environ.get("CXX", "c++"))
            + flags
            + [
                "-I",
                str(root),
                str(root / "route_plan.cc"),
                str(root / "tests/route_plan_test.cc"),
                "-o",
                str(executable),
            ],
            check=True,
        )
        subprocess.run([str(executable)], check=True, timeout=60)


if __name__ == "__main__":
    main()
