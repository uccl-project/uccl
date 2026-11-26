#!/usr/bin/env python3
"""Paired RS recovery experiments; all raw candidates, including losses, retained."""
import argparse
import csv
from collections import defaultdict
import statistics

import run_full_8l4 as runner

runner.BUILD = runner.ROOT / "build/rs_recovery_8l4_20261114"
runner.DATA = runner.ROOT / "benchmarks/data/rs_recovery_8l4_20261114"


def bench(name, extra=(), begin=1048576, end=8388608, mode="gpu", reverse=False,
          warmup=3, iterations=5, repeats=3, variants="nccl,current,rs-candidate"):
    args = ["--gpus", "8", "--op", "rs", "--begin", str(begin), "--end", str(end),
            "--warmup", str(warmup), "--iterations", str(iterations), "--repeats", str(repeats),
            "--variants", variants, *extra]
    if reverse:
        args.append("--reverse")
    runner.run(name, args, mode)
    groups = defaultdict(list)
    for line in (runner.DATA / "raw" / (name + ".log")).read_text().splitlines():
        if line.startswith("median,"):
            _, size, variant, _, _, us = line.split(",")
            groups[int(size), variant].append(float(us))
    print(name, [(size, {v: round(statistics.median(groups[size, v]), 2)
                        for v in variants.split(",")}) for size in sorted({n for n, v in groups})], flush=True)


def config(packet, minimum=131072, workers=4, blocks=64, gpu=False, stripe=False, direct=False):
    args = ["--rs-packet-bytes", str(packet), "--rs-min-packet-bytes", str(minimum),
            "--rs-workers", str(workers), "--rs-blocks", str(blocks)]
    if gpu:
        args.append("--rs-gpu")
    if stripe:
        args.append("--rs-interleave")
    if direct:
        args.append("--rs-direct")
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["screen", "checks", "core", "full", "controls"])
    args = parser.parse_args()
    if args.phase == "screen":
        variants = [
            ("cpu_w4_p128", config(131072)),
            ("cpu_w8_p128", config(131072, workers=8)),
            ("cpu_w16_p128", config(131072, workers=16)),
            ("cpu_w4_p512", config(524288)),
            ("cpu_w8_p512", config(524288, workers=8)),
            ("cpu_w16_p512", config(524288, workers=16)),
            ("cpu_w8_p1m_min1m", config(1048576, 1048576, workers=8)),
            ("cpu_w16_p1m_min1m", config(1048576, 1048576, workers=16)),
            ("cpu_w8_p1m_direct", config(1048576, 1048576, workers=8, direct=True)),
            ("gpu_flat_p128", config(131072, gpu=True)),
            ("gpu_flat_p512", config(524288, gpu=True)),
            ("gpu_stripe_p128", config(131072, gpu=True, stripe=True)),
            ("gpu_stripe_p512", config(524288, gpu=True, stripe=True)),
            ("gpu_stripe_p1m", config(1048576, gpu=True, stripe=True)),
            ("gpu_stripe_p2m", config(2097152, gpu=True, stripe=True)),
            ("cpu_stripe_p512", config(524288, workers=8, stripe=True)),
        ]
        for name, cfg in variants:
            bench("screen_" + name, cfg)

    elif args.phase == "checks":
        for n in (4, 8):
            runner.run("final_ring_check_" + str(n), [str(n)],
                       binary="uccl_lite_host_ring_check", timeout=300)
        runner.run("final_ring_check_shm", ["8"], "shm",
                   binary="uccl_lite_host_ring_check", timeout=300)
        runner.run("final_dispatcher_cuda", [], binary="uccl_lite_dispatcher_cuda_check")
        for mode in ("gpu", "shm"):
            for binary in ("router_check", "router_graph_check"):
                runner.run("final_" + mode + "_" + binary,
                           ["--gpus", "8", "--ag-per-source"], mode,
                           binary="uccl_lite_" + binary)
    elif args.phase == "core":
        for mode in ("gpu", "shm"):
            bench("core_" + mode, ["--blocks", "16"], mode=mode,
                  warmup=5, iterations=10, variants="nccl,previous,current")
    elif args.phase == "full":
        for suffix, modes in (("a", ("gpu", "shm")), ("b", ("shm", "gpu"))):
            for mode in modes:
                bench("confirm_" + mode + "_rs_" + suffix,
                      ["--blocks", "16"], begin=1024, end=1 << 30, mode=mode,
                      reverse=suffix == "b", warmup=5, iterations=10,
                      variants="nccl,previous,current")
    elif args.phase == "controls":
        for mode in ("gpu", "shm"):
            for op in ("ar", "ag"):
                runner.bench("control_" + mode + "_" + op, mode, op,
                             variants="nccl,previous,current")
