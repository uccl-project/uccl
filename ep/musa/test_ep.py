#!/usr/bin/env python3
"""Host tests of production EP control code; not GPU or MPI validation."""

import argparse
import os
import pathlib
import shlex
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitize", choices=("address", "undefined", "thread"))
    parser.add_argument("--layout-only", action="store_true")
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parent
    flags = [
        "-std=c++17",
        "-O1",
        "-g",
        "-pthread",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-ffp-contract=off",
    ]
    if args.sanitize:
        flags += [f"-fsanitize={args.sanitize}", "-fno-omit-frame-pointer"]
    cases = [("layout_test", ["route_plan.cc", "exchange_plan.cc"])]
    if not args.layout_only:
        cases.append(
            ("session_test", ["route_plan.cc", "exchange_plan.cc", "session.cc"])
        )
    with tempfile.TemporaryDirectory(prefix="uccl-musa-ep-") as directory:
        for test, sources in cases:
            executable = pathlib.Path(directory) / test
            subprocess.run(
                shlex.split(os.environ.get("CXX", "c++"))
                + flags
                + ["-I", str(root / "tests/fakes"), "-I", str(root)]
                + [str(root / path) for path in sources]
                + [str(root / f"tests/{test}.cc"), "-o", str(executable)],
                check=True,
            )
            subprocess.run([str(executable)], check=True, timeout=120)


if __name__ == "__main__":
    main()
