# ukernel collective optimization — framework, attempts, and plan

How the allreduce/alltoall optimization work is organized: the design
principle, the three planes we optimize in, the measured attempts with
their verdicts, and the forward plan. Supporting data:
[ce_contention.md](ce_contention.md), [fused_rs_reduce.md](fused_rs_reduce.md).

## Design principle

The GPU only computes; all synchronization and waiting is orchestrated by
the CPU (host polls, host signals, host-enqueued tasks). Device-side
spins on flags (NCCL's LL `readLL`, mscclpp mbarrier waits) are
deliberately out of scope — the original motivation of the project is
unloading that wait from the SMs. Everything below respects this.

## The three planes

### 1. Data plane — who moves bytes

| path | read | write | verdict |
|---|---|---|---|
| CE copy | local | peer | fast per copy at <=8-way concurrency (B300 ~630-700 GB/s); degrades 2-3x under synchronized 56-way peaks (alltoall). Fine for ring steps (8-way). |
| SM copy task | local | peer | the alltoall-proven fast direction; handles synchronized peaks ~50% better than CE. |
| fused remote-read | peer | local | latency-bound (needs >64 blocks, collapses at 128) — dead end for the primary data path. |

Rule: copies stay on CE unless they fuse into a task that already runs
on the worker; never read-from-peer as the primary path.

### 2. Compute plane — what the worker does

The persistent worker processes tasks serially, so task count and
per-task cost matter:

- `Reduce` (local RMW) — baseline RS.
- `Reduce+copy` (fused RS, `UK_CCL_FUSE_REDUCE_COPY=1`) — one task does
  reduce then write-to-peer; removes the reduce->put host transition and
  the separate put op from the RS critical path.
- `Copy` to peer (device put) — worker competition with reduces regresses
  allreduce when used for AG; keep AG copies on CE.

### 3. Sync plane — who coordinates

The per-hop host chain is the dominant measured latency (50-200us gaps
between 27us CE copies at 8 ranks):

```
task done → dev drain → enqueue signal → host ring write → peer poll → enqueue next task
```

Every transition costs host scheduling/polling time. Signal aggregation
(G) was flat; drain busy-polling was tried and reverted. The remaining
cuts: device-completion flags (device writes, host polls — no atomics
needed), and fewer hops via deeper pipelining.

## Attempt log (verdicts)

| attempt | result |
|---|---|
| CE contention microbench (sync 56-way vs staggered) | CE sync 2-3x penalty; SM copy ~half as bad. Real but ring RS is only 8-way. |
| Fused remote-read reduce (`UK_CCL_FUSE_RS_REDUCE`) | 1.4-1.5x worse at 2/4/8 ranks; latency-bound. Dead end, kept behind flag. |
| Fused reduce+copy (`UK_CCL_FUSE_REDUCE_COPY`) | correct; neutral at 4 ranks (within noise); 8-rank pending a quiet machine. |
| AG via device path (`UK_CCL_PUT_PATH=device`) | regresses (worker serializes copies with reduces): 1223us vs 1066us fused-RS-only at 4 ranks. AG stays on CE. |
| Signal aggregation (G=2/4 at LT=16) | flat (2078/2124/2085us vs 2078 baseline). Signal COUNT is not the lever. |
| drain busy-poll (conditional, has_pending) | tried 2026-08-06, reverted 3 min later (2-rank regression / CPU load). |
| LT sweep (pipeline depth) | LT=16 best (2 tiles/shard); finer tiles regress — per-tile host cost caps depth. |

## Forward plan (the system applied)

1. **RS**: fused reduce+copy (done). Awaiting 8-rank validation.
2. **AG**: keep CE copies; the next win is the sync plane, not moving the
   copy to the worker.
3. **Sync**: host-mapped device-flag slots — each signal op owns a
   single-writer flag slot (plain store + `__threadfence_system`, no
   atomics, so it works on B300 where `HostNativeAtomicSupported=0`).
   The fused task writes the flag on completion; the host poll waits on
   it. Removes the dev-drain->enqueue->ring-write transitions.
4. **Pipeline**: once per-hop latency drops, deepen the ring pipeline
   (staging buffers like NCCL SIMPLE's `NCCL_STEPS` allow more steps in
   flight without touching user buffers).
5. Re-measure 2/4/8 ranks; 8-rank requires a quiet machine (vLLM was
   using GPUs 0-3).

## What we borrow from NCCL/mscclpp

NCCL's `genericOp<recvDirect, sendDirect, reduce, copy, ...>` fuses
copy/reduce/through in one kernel; its ring uses
`directRecvReduceDirectSend` (RS) and `directRecvCopyDirectSend` (AG
land+forward). We adopt the fusion shapes but keep the sync on the host:
our fused reduce+copy is the RS shape with the receive done by CE-land
instead of a remote read. The AG land+forward would need a remote read,
which our data-plane measurements rule out — so AG stays CE + host
signals. We do NOT borrow the LL/LL128 device flag spinning or mscclpp
mbarrier waits.
