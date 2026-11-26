#!/usr/bin/env python3
"""Diagnostic CUDA-event communication fractions; excluded from performance tables.

Each caller stream is timed across the complete operation and around each
collective. The latter includes peer readiness waits and host submission gaps.
It is an occupied stream interval, not a sum of NCCL kernel durations. Because
this adapter serializes compute and communication per rank, intervals on one
rank do not overlap. The remainder includes compute and Python/driver gaps.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from benchmark import Collectives, Model, synchronize


def events():
    return [torch.cuda.Event(enable_timing=True) for _ in range(2)]


def record(markers):
    for r in range(2):
        markers[r].record(torch.cuda.current_stream(r))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backend", choices=["nccl", "peer"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compile-blocks", action="store_true")
    parser.add_argument("--serving", default="2:32:16")
    parser.add_argument("--training", default="2:32")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("choose a fresh output")
    torch.set_num_threads(1)
    collective = Collectives(args.backend)
    model = Model(args.checkpoint, collective, args.compile_blocks)
    original_call = collective.call
    marks = []

    def timed_call(operation, inputs):
        start, end = events(), events()
        record(start)
        output = original_call(operation, inputs)
        record(end)
        marks.append((operation, inputs[0].numel() * (4 if operation == 1 else 2), start, end))
        return output

    result = {"backend": args.backend, "checkpoint": str(args.checkpoint),
              "complete_block_bytes": model.complete_block_bytes, "rows": []}
    for kind, specification in [("serving", args.serving), ("training", args.training)]:
        dims = list(map(int, specification.split(":")))
        batch, sequence = dims[:2]
        generator = torch.Generator().manual_seed(12345)
        tokens = torch.randint(model.config["vocab_size"], (batch, sequence + (kind == "training")),
                               generator=generator)
        tokens = [x.to(r) for r, x in enumerate(tokens.chunk(2))]
        optimizer = torch.optim.SGD(model.parameters, lr=1e-3, foreach=True)

        def invoke():
            if kind == "serving":
                model.generate(tokens, dims[2])
            else:
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model.forward([x[:, :-1] for x in tokens])
                losses = [F.cross_entropy(x.flatten(0, 1).float(), t[:, 1:].flatten())
                          for x, t in zip(logits, tokens)]
                torch.autograd.backward(losses, [torch.full((), 0.5, device=r) for r in range(2)])
                optimizer.step()

        for _ in range(3):
            invoke()
        synchronize()
        baseline = []
        for _ in range(3):
            start = time.perf_counter()
            invoke()
            synchronize()
            baseline.append((time.perf_counter() - start) * 1000)
        collective.call = timed_call
        for repeat in range(3):
            marks.clear()
            start, end = events(), events()
            record(start)
            wall = time.perf_counter()
            invoke()
            record(end)
            synchronize()
            wall_ms = (time.perf_counter() - wall) * 1000
            total = [start[r].elapsed_time(end[r]) for r in range(2)]
            communication = [sum(a[r].elapsed_time(b[r]) for _, _, a, b in marks) for r in range(2)]
            row = {"kind": kind, "case": specification, "repeat": repeat,
                   "wall_ms": wall_ms, "uninstrumented_wall_ms": baseline,
                   "total_stream_ms": total, "communication_stream_ms": communication,
                   "communication_fraction": [c / t for c, t in zip(communication, total)],
                   "operation_ms": {str(op): [sum(a[r].elapsed_time(b[r]) for o, _, a, b in marks if o == op)
                                               for r in range(2)] for op in (1, 2)},
                   "collective_counts": {str(op): sum(o == op for o, *_ in marks) for op in (1, 2)}}
            result["rows"].append(row)
            print(json.dumps(row), flush=True)
        collective.call = original_call
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    collective.close()


if __name__ == "__main__":
    main()
