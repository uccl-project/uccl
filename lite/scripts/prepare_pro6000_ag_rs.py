#!/usr/bin/env python3
"""Write the fixed, independently repeated FP32 AG/RS calibration catalog."""
import argparse
import json
from pathlib import Path
import random


def catalog():
    ours = {}
    for blocks in (1, 4, 8, 32):
        ours[f"ll-b{blocks}"] = dict(mode="ll", blocks=blocks, threads=256, end=4 << 20)
    for blocks, threads in ((8, 256), (32, 128), (64, 128), (128, 128), (64, 64), (128, 64)):
        ours[f"ring-sm-b{blocks}-t{threads}"] = dict(mode="ring-put", blocks=blocks,
                                                    threads=threads, begin=1 << 18, end=32 << 20)
    ours["put-b32"] = dict(mode="put", blocks=32, threads=128, begin=1 << 18, end=32 << 20)
    for lanes, tile, workers in ((4, 1, 4), (8, 1, 4), (4, 4, 4), (4, 4, 8),
                                 (8, 4, 8), (4, 8, 8), (2, 16, 8)):
        ours[f"ring-ce-l{lanes}-t{tile}-w{workers}"] = dict(
            mode="ring-ce", blocks=lanes, threads=256, workers=workers, streams=2,
            tile=tile << 20, begin=1 << 20)
    peer = {"NCCL_P2P_LEVEL": "SYS", "NCCL_SHM_DISABLE": "1"}
    sm = {**peer, "NCCL_P2P_USE_CUDA_MEMCPY": "0", "NCCL_ALGO": "Ring"}
    ce = {**peer, "NCCL_P2P_USE_CUDA_MEMCPY": "1", "NCCL_ALGO": "Ring", "NCCL_PROTO": "Simple"}
    nccl = {"default": ({}, {}), "sm-auto": ({**peer, "NCCL_P2P_USE_CUDA_MEMCPY": "0"}, {})}
    for proto in ("LL", "LL128"):
        nccl["sm-" + proto.lower()] = ({**sm, "NCCL_PROTO": proto}, {"end": 16 << 20})
    for channels in (2, 4, 8, 16, 32):
        env = {**sm, "NCCL_PROTO": "Simple", "NCCL_MIN_NCHANNELS": str(channels),
               "NCCL_MAX_NCHANNELS": str(channels)}
        nccl[f"sm-simple-ch{channels}"] = (env, {})
        if channels in (2, 8, 32):
            for threads in (128, 256):
                nccl[f"sm-simple-ch{channels}-t{threads}"] = ({**env, "NCCL_NTHREADS": str(threads)}, {})
    nccl["ce-auto"] = ({**peer, "NCCL_P2P_USE_CUDA_MEMCPY": "1", "NCCL_PROTO": "Simple"}, {"begin": 1 << 20})
    for channels, buffer in ((2, 16), (4, 16), (8, 16), (2, 32), (4, 32), (8, 32),
                             (4, 64), (8, 64), (6, 16), (6, 32)):
        env = {**ce, "NCCL_MIN_NCHANNELS": str(channels), "NCCL_MAX_NCHANNELS": str(channels),
               "NCCL_BUFFSIZE": str(buffer << 20)}
        nccl[f"ce-ch{channels}-b{buffer}"] = (env, {"begin": 1 << 20})
    configs = {}
    for op in ("allgather", "reducescatter"):
        for name, args in ours.items():
            configs[f"{op}-ours-{name}"] = dict(backend="ours", op=op, args=args, env={})
        for name, (env, args) in nccl.items():
            configs[f"{op}-nccl-{name}"] = dict(backend="nccl", op=op, args=args, env=env)
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    configs = catalog()
    jobs = []
    for repeat in range(args.rounds):
        order = sorted(configs)
        random.Random(20261116 + repeat).shuffle(order)
        for ident in order:
            config = configs[ident]
            jobs.append({**config, "args": {**config["args"], "seed": 1000 + repeat * 37},
                         "name": f"c{repeat}-{ident}", "candidate": ident,
                         "stage": "calibration", "round": repeat})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(jobs, output, indent=2)
        output.write("\n")
    print(len(configs), "operation/configuration pairs;", len(jobs), "calibration jobs")


if __name__ == "__main__":
    main()
