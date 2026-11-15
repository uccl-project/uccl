"""Full-buffer signed/index-dependent oracle, with mixed placement per rank."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from bench_dropin import Nccl, tensors


def pattern(count, offset=0):
    return (((torch.arange(count, device="cuda", dtype=torch.int32) + offset) % 31).float() - 15) / 32


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    nccl = Nccl(rank)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    rows = []
    for index, (op, count) in enumerate(json.loads(args.cases.read_text())):
        if rank == 0:
            print(f"Checking op={op} count={count}", flush=True)
        st = streams[index % 2]
        layout = "inplace" if rank % 2 else "out"
        alignment = (2, 4, 8, 16)[rank % 4]
        source, target = tensors(op, count, rank, layout, alignment=alignment)
        with torch.cuda.stream(st):
            values = pattern(source.numel()) + (-1 if rank % 2 else 1) * (rank + 1) / 16
            if op == 2:
                for d, shard in enumerate(values.chunk(8)):
                    shard.add_(d / 16)
            source.copy_(values)
            nccl.call(op, source, target, count, st)
        st.synchronize()
        if op == 1:
            for peer, shard in enumerate(target.chunk(8)):
                oracle = (pattern(count) + (-1 if peer % 2 else 1) * (peer + 1) / 16).bfloat16()
                assert torch.equal(shard, oracle), (rank, op, count, peer)
        else:
            oracle = pattern(count, rank * count if op == 2 else 0) * 8 - 0.25
            if op == 2:
                oracle.add_(rank / 2)
            assert torch.equal(target, oracle.bfloat16()), (rank, op, count)
        rows.append({"op": op, "count": count, "full_buffer_oracle": "pass"})
        del source, target, values, oracle
    affinity = {}
    for task in Path("/proc/self/task").iterdir():
        try:
            name = (task / "comm").read_text().strip()
            if name in ("uccl_ar_lane", "uccl_rs_worker"):
                affinity[name] = sorted(os.sched_getaffinity(int(task.name)))
        except ProcessLookupError:
            pass
    assert affinity == {"uccl_ar_lane": [rank * 6 + 4], "uccl_rs_worker": [rank * 6 + 5]}, affinity
    rank_affinity = [None] * 8
    dist.all_gather_object(rank_affinity, affinity)
    nccl.check(nccl.lib.ncclCommDestroy(nccl.comm))
    dist.barrier()
    if rank == 0:
        args.out.write_text(json.dumps({"cases": rows, "rank_worker_affinity": rank_affinity}, indent=2) + "\n")
        print(f"PASS: {len(rows)} full-buffer signed/index-dependent cases; mixed rank layouts and alignments")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
