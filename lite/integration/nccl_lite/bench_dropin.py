"""Direct NCCL ABI correctness and timing on eight ranks (Gloo control plane)."""

import argparse
import ctypes as C
import json
import os
from pathlib import Path
import statistics

import torch
import torch.distributed as dist


class UniqueId(C.Structure):
    _fields_ = [("internal", C.c_char * 128)]


class Nccl:
    def __init__(self, rank):
        self.lib = C.CDLL(os.environ["SGLANG_NCCL_SO_PATH"])
        signatures = {
            "ncclGetUniqueId": [C.c_void_p],
            "ncclCommInitRank": [C.c_void_p, C.c_int, UniqueId, C.c_int],
            "ncclAllGather": [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int, C.c_void_p, C.c_void_p],
            "ncclReduceScatter": [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int, C.c_int, C.c_void_p, C.c_void_p],
            "ncclAllReduce": [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int, C.c_int, C.c_void_p, C.c_void_p],
            "ncclCommDestroy": [C.c_void_p], "ncclGroupStart": [], "ncclGroupEnd": [],
        }
        for name, sig in signatures.items():
            getattr(self.lib, name).argtypes = sig
            getattr(self.lib, name).restype = C.c_int
        unique = C.create_string_buffer(128)
        if rank == 0:
            self.check(self.lib.ncclGetUniqueId(unique))
        ids = [unique.raw]
        dist.broadcast_object_list(ids, src=0)
        unique = UniqueId.from_buffer_copy(ids[0])
        self.comm = C.c_void_p()
        self.check(self.lib.ncclCommInitRank(C.byref(self.comm), 8, unique, rank))

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"NCCL result {code}")

    def call(self, op, source, target, count, stream, dtype=9, reduction=0):
        args = [source.data_ptr(), target.data_ptr(), count, dtype]
        if op != 1:
            args.append(reduction)
        args += [self.comm, stream.cuda_stream]
        self.check(getattr(self.lib, {1: "ncclAllGather", 2: "ncclReduceScatter", 3: "ncclAllReduce"}[op])(*args))


def tensors(op, count, rank, layout, dtype=torch.bfloat16, alignment=2):
    ni, no = count * (8 if op == 2 else 1), count * (8 if op == 1 else 1)
    offset = alignment // 2
    if layout == "inplace":
        backing = torch.empty(max(ni, no) + offset, device="cuda", dtype=dtype)[offset:]
        source = backing[rank * count:(rank + 1) * count] if op == 1 else backing[:ni]
        target = backing[rank * count:(rank + 1) * count] if op == 2 else backing[:no]
    else:
        source = torch.empty(ni + offset, device="cuda", dtype=dtype)[offset:]
        target = torch.empty(no + offset, device="cuda", dtype=dtype)[offset:]
    return source, target


def expected(op, count, rank, step, device="cuda"):
    # Signed exactly representable inputs with destination-dependent RS values.
    sign = lambda r: -1 if r % 2 else 1
    if op == 1:
        return torch.cat([torch.full((count,), sign(r) * (r + 1) / 16 + step / 32,
                                     device=device, dtype=torch.bfloat16) for r in range(8)])
    value = sum(sign(r) * (r + 1) / 16 for r in range(8)) + step / 4
    if op == 2:
        value += rank / 2
    return torch.full((count,), value, device=device, dtype=torch.bfloat16)


def fill(source, op, rank, step):
    source.fill_((-1 if rank % 2 else 1) * (rank + 1) / 16 + step / 32)
    if op == 2:
        for d, shard in enumerate(source.chunk(8)):
            shard.add_(d / 16)


def edge_checks(nccl, rank, cases):
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    pending = []
    # Several collective kinds and streams are outstanding before any sync.
    for i in range(9):
        op = i % 3 + 1
        count = min(c for o, c in cases if o == op)
        source, target = tensors(op, count, rank, "inplace" if i % 2 else "out")
        st = streams[i % 2]
        with torch.cuda.stream(st):
            fill(source, op, rank, i)
            nccl.call(op, source, target, count, st)
        pending.append((source, target, op, count, i))
    torch.cuda.synchronize()
    for source, target, op, count, step in pending:
        assert torch.equal(target, expected(op, count, rank, step)), ("stream", rank, op, step)
    # Unsupported reduction and dtype use the real NCCL after packet traffic.
    for dtype, nccl_dtype, reduction, value in ((torch.float32, 7, 0, 36),
                                                (torch.bfloat16, 9, 2, 8)):
        source = torch.full((count,), rank + 1, device="cuda", dtype=dtype)
        target = torch.empty_like(source)
        nccl.call(3, source, target, count, torch.cuda.current_stream(), nccl_dtype, reduction)
        torch.cuda.synchronize()
        assert torch.all(target == value)
    # A group transitions to NCCL and must retain nested group semantics.
    source = torch.full((count,), rank + 1, device="cuda", dtype=torch.bfloat16)
    target = torch.empty_like(source)
    nccl.check(nccl.lib.ncclGroupStart())
    nccl.check(nccl.lib.ncclGroupStart())
    nccl.call(3, source, target, count, torch.cuda.current_stream())
    nccl.check(nccl.lib.ncclGroupEnd())
    nccl.check(nccl.lib.ncclGroupEnd())
    torch.cuda.synchronize()
    assert torch.all(target == 36)
    # Repeat replay uses fresh inputs. There are no packet sequence counters
    # captured in the graph. No warmup disables the packet path on this comm.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        nccl.call(3, source, target, count, torch.cuda.current_stream())
    for i in range(3):
        source.fill_(rank + i + 1)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.all(target == 36 + 8 * i)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iterations", type=int, default=12)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    base = rank * 6
    cpus = set(range(base, base + 4)) | set(range(base + 48, base + 52))
    for task in Path("/proc/self/task").iterdir():
        try:
            os.sched_setaffinity(int(task.name), cpus)
        except ProcessLookupError:
            pass
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    nccl = Nccl(rank)
    cases = json.loads(args.cases.read_text())
    for op, count in cases:
        for layout, alignment in [(layout, a) for a in (16, 2, 4, 8) for layout in ("out", "inplace")]:
            source, target = tensors(op, count, rank, layout, alignment=alignment)
            st = torch.cuda.current_stream()
            values = []
            for repeat in range(args.repeats):
                times = []
                for i in range(args.warmup + args.iterations):
                    step = i % 4
                    fill(source, op, rank, step)
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    nccl.call(op, source, target, count, st)
                    end.record()
                    end.synchronize()
                    if i >= args.warmup:
                        times.append(begin.elapsed_time(end))
                    if i == args.warmup + args.iterations - 1:
                        assert torch.equal(target, expected(op, count, rank, step)), (rank, op, count, layout)
                values.append(statistics.median(times))
            ranks = [None] * 8
            dist.all_gather_object(ranks, values)
            if rank == 0:
                row = {"op": op, "count": count, "layout": layout, "alignment": alignment, "correct": True,
                       "warmup": args.warmup, "iterations": args.iterations, "repeats": args.repeats,
                       "rank_medians_ms": ranks, "repeat_ms": [max(r[i] for r in ranks) for i in range(args.repeats)]}
                with args.out.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps({k: v for k, v in row.items() if k != "rank_medians_ms"}), flush=True)
            del source, target
    edge_checks(nccl, rank, cases)
    nccl.check(nccl.lib.ncclCommDestroy(nccl.comm))
    # A second communicator transitions directly from live packet traffic to
    # capture (the preceding graph test followed a group transition).
    nccl = Nccl(rank)
    count = min(c for o, c in cases if o == 3)
    source, target = tensors(3, count, rank, "out")
    source.fill_(rank + 1)
    nccl.call(3, source, target, count, torch.cuda.current_stream())
    torch.cuda.synchronize()
    assert torch.all(target == 36)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        nccl.call(3, source, target, count, torch.cuda.current_stream())
    for i in range(3):
        source.fill_(rank + i + 1)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.all(target == 36 + 8 * i)
    del graph
    torch.cuda.synchronize()
    nccl.check(nccl.lib.ncclCommDestroy(nccl.comm))
    dist.destroy_process_group()
    if rank == 0:
        print("UCCL Lite drop-in: layouts, mixed streams, fallback, nested groups and graph replay passed", flush=True)


if __name__ == "__main__":
    main()
