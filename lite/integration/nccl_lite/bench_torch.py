"""Eight-process BF16 collective timing, including SGLang message sizes."""

import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["LOCAL_RANK"])
    warmup = int(os.environ.get("UCCL_BENCH_WARMUP", "3"))
    iterations = int(os.environ.get("UCCL_BENCH_ITERS", "8"))
    repeats = int(os.environ.get("UCCL_BENCH_REPEATS", "1"))
    if warmup < 1 or iterations < 1 or repeats < 1:
        raise ValueError("benchmark iterations must be positive")
    isolate_worker = os.environ.get("UCCL_BENCH_ISOLATE_WORKER") == "1"
    isolate_helper = os.environ.get("UCCL_BENCH_ISOLATE_HELPER") == "1"
    if isolate_helper and not isolate_worker:
        raise ValueError("helper isolation requires worker isolation")
    if isolate_worker:
        base = rank * 6
        available_cores = 4 if isolate_helper else 5
        scheduler_cpus = set(range(base, base + available_cores)) | set(
            range(base + 48, base + 48 + available_cores)
        )
        for task in Path("/proc/self/task").iterdir():
            try:
                os.sched_setaffinity(int(task.name), scheduler_cpus)
            except ProcessLookupError:
                pass
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    dist.barrier()
    time.sleep(0.5)
    primer = torch.empty((2048,), dtype=torch.bfloat16, device="cuda")
    for _ in range(20):
        primer.fill_(rank + 1)
        dist.all_reduce(primer, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
    rs_primer = torch.full((1024,), rank + 1, dtype=torch.bfloat16,
                           device="cuda")
    rs_result = torch.empty((128,), dtype=torch.bfloat16, device="cuda")
    for _ in range(20):
        dist.reduce_scatter_tensor(rs_result, rs_primer, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
    dist.barrier()
    del primer, rs_primer, rs_result
    configured_sizes = os.environ.get("UCCL_BENCH_SIZES_BYTES")
    if configured_sizes:
        sizes = [int(value) for value in configured_sizes.split(",")]
        if not sizes or any(value < 16 or value % 16 for value in sizes):
            raise ValueError("sizes must be positive multiples of 16 bytes")
        ops = os.environ.get("UCCL_BENCH_OPS", "AR,AG,RS").split(",")
        if not ops or any(op not in ("AR", "AG", "RS") for op in ops):
            raise ValueError("ops must be AR, AG, or RS")
        cases = [(op, size_bytes) for op in ops for size_bytes in sizes]
    else:
        cases = [(op, mib * 1024 * 1024) for op, mib in [
            ("AG", 4), ("RS", 4), ("AG", 8), ("RS", 8),
            ("AG", 16), ("RS", 16), ("AR", 4), ("AR", 8), ("AR", 16),
        ]]
    for op, size_bytes in cases:
        elements = size_bytes // 2
        shard = elements // 8
        if op == "AG":
            source = torch.full((shard,), rank + 1, dtype=torch.bfloat16, device="cuda")
            target = torch.empty((elements,), dtype=torch.bfloat16, device="cuda")
            run = lambda: dist.all_gather_into_tensor(target, source)
        elif op == "RS":
            source = torch.full((elements,), rank + 1, dtype=torch.bfloat16, device="cuda")
            target = torch.empty((shard,), dtype=torch.bfloat16, device="cuda")
            run = lambda: dist.reduce_scatter_tensor(target, source, op=dist.ReduceOp.SUM)
        else:
            source = torch.full((elements,), rank + 1, dtype=torch.bfloat16, device="cuda")
            run = lambda: dist.all_reduce(source, op=dist.ReduceOp.SUM)

        repeat_medians = []
        for _ in range(repeats):
            for _ in range(warmup):
                source.fill_(rank + 1)
                run()
                torch.cuda.synchronize()
            times = []
            for _ in range(iterations):
                source.fill_(rank + 1)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                run()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
            repeat_medians.append(statistics.median(times))
        if op == "AG":
            for peer in range(8):
                assert torch.all(target[peer * shard : (peer + 1) * shard] == peer + 1)
        elif op == "RS":
            assert torch.all(target == 36)
        else:
            assert torch.all(source == 36)
        all_rank_repeat_medians = [None] * 8
        dist.all_gather_object(all_rank_repeat_medians, repeat_medians)
        if rank == 0:
            repeat_max = [max(rank_times[index]
                              for rank_times in all_rank_repeat_medians)
                          for index in range(repeats)]
            print(json.dumps({"op": op, "bytes": size_bytes,
                              "mib": size_bytes / (1024 * 1024),
                              "warmup": warmup,
                              "iterations": iterations,
                              "repeats": repeats,
                              "isolate_worker": isolate_worker,
                              "isolate_helper": isolate_helper,
                              "max_rank_median_ms": statistics.median(repeat_max),
                              "repeat_max_rank_median_ms": repeat_max,
                              "rank_repeat_medians_ms": all_rank_repeat_medians,
                              "rank0_median_ms": statistics.median(repeat_medians)}),
                  flush=True)
        del run, source
        if op != "AR":
            del target
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
