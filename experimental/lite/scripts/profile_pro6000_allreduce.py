#!/usr/bin/env python3
"""Capture diagnostic CUDA timelines; never use instrumented times for selection."""
import argparse
import fcntl
from pathlib import Path
import subprocess

from run_pro6000_allreduce import Runner, dump, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--nsys", default="/usr/local/cuda-12.9/bin/nsys")
    p.add_argument("--backend", choices=("ours", "nccl"), default="ours")
    p.add_argument("--config", default="staged-ce-128")
    p.add_argument("--size", type=int, default=1 << 30)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    with open("/home/wxzheng/uccl-eval.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r = Runner(a.output.resolve(), "local-four")
        temporary = a.output.resolve() / "tmp"
        temporary.mkdir()
        r.env["TMPDIR"] = str(temporary)
        r.hardware()
        (a.output / "nsys-version.txt").write_bytes(
            subprocess.check_output([a.nsys, "--version"])
        )
        (a.output / "measured_source" / Path(__file__).name).write_bytes(
            Path(__file__).read_bytes()
        )
        label = "timeline-" + a.backend + "-" + a.config
        values = r.run(
            label,
            a.backend,
            a.config,
            a.size,
            a.size,
            seed=4001,
            launcher=[
                a.nsys,
                "profile",
                "--trace=cuda",
                "--sample=none",
                "--cpuctxsw=none",
                "--trace-fork-before-exec=true",
                "--wait=primary",
                "--duration=30",
                "--kill=none",
                "--force-overwrite=false",
                "--export=sqlite",
                "--output=" + str(a.output.resolve() / label),
            ],
        )
        if values is None:
            raise RuntimeError("Profile failed; evidence retained")
        reports = [a.output / (label + suffix) for suffix in (".nsys-rep", ".sqlite")]
        complete = all(path.is_file() for path in reports)
        dump(
            a.output / "profile-status.json",
            dict(
                complete=complete,
                reports={path.name: sha(path) for path in reports if path.is_file()},
            ),
        )
        if not complete:
            raise RuntimeError("Collective passed but profiler export is missing")


if __name__ == "__main__":
    main()
