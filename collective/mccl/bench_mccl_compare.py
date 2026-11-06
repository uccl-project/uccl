#!/usr/bin/env python3
"""Collect paired native/UCCL MCCL runs without claiming hardware acceptance."""

import hashlib
import json
import os
import pathlib
import platform
import subprocess
import time

from mccl_bench import (
    counter_delta,
    expected_sizes,
    parse_results,
    read_counters,
    validate_coverage,
)
from mccl_process import run_command
from run_mccl_correctness import COLLECTIVES, build_command, find_binary, parser

ROOT = pathlib.Path(__file__).resolve().parents[2]


def capture_environment(env):
    return {
        key: value
        for key, value in env.items()
        if (
            key.startswith(("UCCL_", "MCCL_", "OMP_"))
            or key in ("MUSA_VISIBLE_DEVICES", "LD_PRELOAD", "LD_LIBRARY_PATH")
        )
        and not any(
            word in key for word in ("TOKEN", "PASSWORD", "SECRET", "CREDENTIAL")
        )
        and not key.endswith("_KEY")
    }


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_info():
    def git(*arguments):
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *arguments], text=True
        ).strip()

    paths = ("collective", "include", "ep", "uccl")
    diff = git("diff", "--binary", "HEAD", "--", *paths)
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *paths)
    return {
        "sha": git("rev-parse", "HEAD"),
        "dirty": bool(diff or untracked),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "untracked": {path: sha256(ROOT / path) for path in untracked.splitlines()},
    }


def main(argv=None):
    arguments = parser()
    arguments.description = __doc__
    arguments.add_argument("--output", type=pathlib.Path, required=True)
    arguments.add_argument(
        "--native-net",
        required=True,
        help="verified built-in network name on this MCCL version",
    )
    arguments.add_argument(
        "--native-proof",
        required=True,
        help="exact selected-network log phrase, not just IB capability",
    )
    arguments.add_argument("--repeats", type=int, default=5)
    arguments.add_argument(
        "--counter",
        action="append",
        default=[],
        help="local sysfs device:port; remote counters need separate capture",
    )
    args = arguments.parse_args(argv)
    if (
        not args.tests_home
        or not args.plugin
        or args.repeats < 1
        or args.nranks < 1
        or args.timeout < 1
        or args.iterations < 1
        or args.native_net == "UCCL-MCCL-RDMA"
        or not args.native_proof
    ):
        arguments.error("provide valid test paths, native backend, repeats and timeout")
    args.plugin = pathlib.Path(args.plugin).resolve()
    args.tests_home = pathlib.Path(args.tests_home).resolve()
    if not args.plugin.is_file():
        arguments.error("plugin does not exist")
    # Refuse to mix new evidence with files from an earlier run.
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    configuration = {
        "nranks": args.nranks,
        "iterations": args.iterations,
        "min_bytes": args.min_bytes,
        "max_bytes": args.max_bytes,
        "factor": args.factor,
        "extra": args.extra,
        "launcher": args.launcher,
        "launcher_args": args.launcher_arg,
        "no_launcher": args.no_launcher,
    }
    fixed_environment = capture_environment(os.environ)
    collectives = args.collectives or list(COLLECTIVES)
    manifest = {
        "schema_version": 1,
        "source": source_info(),
        "plugin_sha256": sha256(args.plugin),
        "runner_sha256": sha256(__file__),
        "host": platform.node(),
        "state": "unverified",
        "note": "path/topology/runtime acceptance requires testbed review",
        "counter_scope": "launcher host only",
        "records": records,
        "repeats": args.repeats,
        "collectives": collectives,
        "case_configuration": configuration,
        "fixed_environment": fixed_environment,
        "expected_sizes": {},
    }
    failure = None
    try:
        for collective in collectives:
            coverage = expected_sizes(args, collective)
            manifest["expected_sizes"][collective] = coverage
            binary = find_binary(args.tests_home, COLLECTIVES[collective])
            if binary is None:
                raise ValueError(f"missing {collective} binary")
            for run in range(args.repeats):
                order = ("native", "uccl") if run % 2 == 0 else ("uccl", "native")
                for backend in order:
                    env = os.environ.copy()
                    # Remove inherited plugin path/selection in the native process.
                    env.pop("MCCL_NET_PLUGIN", None)
                    env["LD_LIBRARY_PATH"] = os.pathsep.join(
                        part
                        for part in env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
                        if part and pathlib.Path(part).resolve() != args.plugin.parent
                    )
                    env["MCCL_NET"] = args.native_net
                    if backend == "uccl":
                        env["MCCL_NET_PLUGIN"] = "uccl"
                        env["MCCL_NET"] = "UCCL-MCCL-RDMA"
                        env["LD_LIBRARY_PATH"] = (
                            str(args.plugin.parent)
                            + os.pathsep
                            + env["LD_LIBRARY_PATH"]
                        )
                    env["MCCL_DEBUG"], env["MCCL_DEBUG_SUBSYS"] = "INFO", "NET"
                    command = build_command(args, binary)
                    before = {
                        dev: read_counters("/sys/class/infiniband", dev)
                        for dev in args.counter
                    }
                    start = time.monotonic()
                    status, output, timed_out = run_command(command, env, args.timeout)
                    elapsed = time.monotonic() - start
                    raw = f"{collective}-{run}-{backend}.log"
                    (args.output / raw).write_text(output)
                    record = {
                        "collective": collective,
                        "run": run,
                        "backend": backend,
                        "command": command,
                        "binary_sha256": sha256(binary),
                        "elapsed_s": elapsed,
                        "raw": raw,
                        "exit_code": status,
                        "timed_out": timed_out,
                        "environment": capture_environment(env),
                        "counters_before": before,
                        "case_configuration": configuration,
                        "fixed_environment": fixed_environment,
                    }
                    records.append(record)
                    after = {
                        dev: read_counters("/sys/class/infiniband", dev)
                        for dev in args.counter
                    }
                    record["counters_after"] = after
                    record["counter_bytes"] = {
                        dev: counter_delta(
                            before[dev]["values"],
                            after[dev]["values"],
                            before[dev]["unit"],
                        )
                        for dev in before
                    }
                    if timed_out or status:
                        raise ValueError(f"{raw}: timeout or nonzero exit")
                    if backend == "uccl" and "UCCL-MCCL-RDMA" not in output:
                        raise ValueError(f"{raw}: UCCL not confirmed")
                    if backend == "native" and (
                        "UCCL-MCCL-RDMA" in output or args.native_proof not in output
                    ):
                        raise ValueError(f"{raw}: native selection not confirmed")
                    record["rows"] = parse_results(output)
                    validate_coverage(record["rows"], coverage)
    except (OSError, ValueError, KeyError) as error:
        failure = str(error)
        manifest["state"], manifest["error"] = "failed", failure
    finally:
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    if failure:
        raise SystemExit(failure)


if __name__ == "__main__":
    main()
