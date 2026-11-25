#!/usr/bin/env python3
"""Run the MCCL net-plugin correctness matrix."""

import argparse
import os
import pathlib
import subprocess
import sys


COLLECTIVES = {
    "allreduce": ("all_reduce_perf", "allreduce_perf"),
    "allgather": ("all_gather_perf", "allgather_perf"),
    "reducescatter": ("reduce_scatter_perf", "reducescatter_perf"),
    "alltoall": ("alltoall_perf", "all_to_all_perf"),
}


def find_binary(tests_home, names):
    candidates = []
    for name in names:
        candidates.extend(
            (
                tests_home / name,
                tests_home / "build" / name,
                tests_home / "build" / "bin" / name,
            )
        )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def build_command(args, binary):
    test_args = [
        "-b",
        args.min_bytes,
        "-e",
        args.max_bytes,
        "-f",
        str(args.factor),
        "-n",
        str(args.iterations),
    ]
    test_args.extend(args.extra)
    if args.no_launcher:
        return [str(binary), *test_args]
    return [
        args.launcher,
        *args.launcher_arg,
        "-np",
        str(args.nranks),
        str(binary),
        *test_args,
    ]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--tests-home",
        type=pathlib.Path,
        default=os.environ.get("MCCL_TESTS_HOME"),
        help="root of the built MCCL tests tree",
    )
    result.add_argument(
        "--plugin",
        type=pathlib.Path,
        default=os.environ.get("UCCL_MCCL_PLUGIN"),
        help="path to libmccl-net-uccl.so",
    )
    result.add_argument("--nranks", type=int, default=2)
    result.add_argument("--iterations", type=int, default=20)
    result.add_argument("--min-bytes", default="8")
    result.add_argument("--max-bytes", default="64M")
    result.add_argument("--factor", type=int, default=2)
    result.add_argument("--timeout", type=int, default=180)
    result.add_argument("--launcher", default="mpirun")
    result.add_argument("--launcher-arg", action="append", default=[])
    result.add_argument("--no-launcher", action="store_true")
    result.add_argument(
        "--collective",
        action="append",
        choices=tuple(COLLECTIVES),
        dest="collectives",
        help="collective to run; repeat to select a subset",
    )
    result.add_argument(
        "--allow-plugin-log-missing",
        action="store_true",
        help="do not require the UCCL plugin name in MCCL output",
    )
    result.add_argument("extra", nargs=argparse.REMAINDER)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.tests_home is None:
        print("MCCL_TESTS_HOME or --tests-home is required", file=sys.stderr)
        return 2
    if args.plugin is None:
        print("UCCL_MCCL_PLUGIN or --plugin is required", file=sys.stderr)
        return 2
    args.tests_home = pathlib.Path(args.tests_home).resolve()
    args.plugin = pathlib.Path(args.plugin).resolve()
    if not args.plugin.is_file():
        print(f"MCCL plugin does not exist: {args.plugin}", file=sys.stderr)
        return 2
    if args.nranks < 1 or args.iterations < 1 or args.timeout < 1:
        print("nranks, iterations and timeout must be positive", file=sys.stderr)
        return 2

    collectives = args.collectives or list(COLLECTIVES)
    env = os.environ.copy()
    env["MCCL_NET_PLUGIN"] = "uccl"
    env["MCCL_NET"] = "UCCL-MCCL-RDMA"
    env["MCCL_DEBUG"] = "INFO"
    env["MCCL_DEBUG_SUBSYS"] = "NET"
    plugin_dir = str(args.plugin.parent)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        part for part in (plugin_dir, env.get("LD_LIBRARY_PATH", "")) if part
    )

    for collective in collectives:
        binary = find_binary(args.tests_home, COLLECTIVES[collective])
        if binary is None:
            print(
                f"{collective}: no test binary found under {args.tests_home}",
                file=sys.stderr,
            )
            return 2
        command = build_command(args, binary)
        print(f"[MCCL][{collective}] {' '.join(command)}", flush=True)
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                env=env,
                timeout=args.timeout,
            )
        except subprocess.TimeoutExpired:
            print(f"{collective}: timed out after {args.timeout}s", file=sys.stderr)
            return 1
        output = completed.stdout + completed.stderr
        print(output, end="")
        if completed.returncode != 0:
            print(
                f"{collective}: exited with {completed.returncode}",
                file=sys.stderr,
            )
            return 1
        if (
            not args.allow_plugin_log_missing
            and "UCCL-MCCL-RDMA" not in output
        ):
            print(
                f"{collective}: MCCL output did not confirm UCCL-MCCL-RDMA",
                file=sys.stderr,
            )
            return 1
        print(f"[MCCL][{collective}] PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
