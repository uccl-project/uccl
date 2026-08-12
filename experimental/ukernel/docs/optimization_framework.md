# ukernel collective optimization — framework, attempts, and plan

How the AllReduce/AllToAll optimization work is organized: the design
principle, the three planes we optimize in, the measured attempts with
their verdicts, and the forward plan. Supporting data:
[ce_contention.md](ce_contention.md),
[fused_rs_reduce.md](fused_rs_reduce.md),
[b300_native_nccl_measurements.md](b300_native_nccl_measurements.md).

## Design principle

The GPU only computes; all synchronization and waiting is orchestrated by
the CPU (host polls, host signals, host-enqueued tasks). Device-side
spins on flags (NCCL's LL `readLL`, mscclpp mbarrier waits) are
deliberately out of scope — the project's motivation is unloading that
wait from the SMs. Everything below respects this.

## The three planes

### 1. Data plane — who moves bytes

| path | read | write | verdict |
|---|---|---|---|
| CE copy | local | peer | fast per copy at <=8-way concurrency (B300 ~630-700 GB/s); degrades 2-3x under synchronized 56-way peaks. Fine for ring steps (8-way). Batch submission does not relieve the peak (ce_contention.md). |
| SM copy task | local | peer | the alltoall-proven fast direction; handles synchronized peaks ~50% better than CE; the basis of the fused RS/AG device copies. |
| fused remote-read | peer | local | latency-bound (throughput scales with block count, collapses at 128) — dead end for the primary data path. |

Rule: copies stay on CE unless they fuse into a task that already runs
on the worker; never read-from-peer as the primary path.

### 2. Compute plane — what the worker does

The persistent worker processes tasks serially, so task count and
per-task cost matter:

- `Reduce` (local RMW) — baseline RS.
- `Reduce+copy` (fused RS, `UK_CCL_FUSE_REDUCE_COPY=1`) — one task does
  reduce then write-to-peer; removes the reduce->put host transition and
  the separate put op from the RS critical path. +18.8% at 8 ranks.
- `Copy` to peer (fused AG, `UK_CCL_FUSE_AG_COPY=1`) — device copy task
  with inline completion flag; neutral at 4 ranks, ~5% at 8 ranks.
  Standalone AG device puts (unfused) regress AllReduce because they
  serialize with reduces on the worker.

### 3. Sync plane — who coordinates

The per-hop host chain was the dominant measured latency (50-200us gaps
between 27us CE copies at 8 ranks):

```
task done → dev drain → enqueue signal → host ring write → peer poll → enqueue next task
```

Every transition costs host scheduling/polling time. Signal aggregation
(G) was flat; drain busy-polling was tried and reverted. The remaining
cuts: **device-completion flags** (device writes a slot, host polls it —
no atomics needed; removes dev-drain + enqueue + ring-write), and fewer
hops via deeper pipelining.

## Attempt log (verdicts)

| attempt | result |
|---|---|
| CE contention microbench (sync 56-way vs staggered) | CE sync 2-3x penalty; SM copy ~half as bad. Real, but ring RS is only 8-way. |
| `cudaMemcpyBatchAsync` batch submission | no effect on the sync peak (~2.1 TB/s both ways); IPC adapter stays per-peer. |
| Fused remote-read reduce (`UK_CCL_FUSE_RS_REDUCE`) | 1.4-1.5x worse at 2/4/8 ranks; latency-bound. Dead end, kept behind flag. |
| Fused reduce+copy (`UK_CCL_FUSE_REDUCE_COPY`) + device flags | +7.5% (4r), +18.8% (8r); stress-validated wrong=0. Shipped. |
| Fused AG copy (`UK_CCL_FUSE_AG_COPY`) | neutral at 4r, ~5% at 8r (1560 -> 1487us). Shipped. |
| AG via standalone device puts (`UK_CCL_PUT_PATH=device` on AG) | regresses (worker serializes copies with reduces). AG fuses instead. |
| Signal aggregation (G=2/4 at LT=16) | flat. Signal COUNT is not the lever. |
| Drain busy-poll (conditional, has_pending) | tried, reverted (2-rank regression / CPU load). |
| LT sweep (pipeline depth) | LT=16 best; finer tiles regress — per-tile host cost caps depth. |
| REDUCE_ILP=16 | +24% per block on reduce bench; shim 95% of native at BLK=64; cannot reach native at <=32 blocks alone. |
| TMA bulk reduce (TMA_REDUCE=1, 224KB smem) | 99% of native at BLK=32 (509.5 GB/s); tail-chunk bug fixed; opt-in pending broader validation. |
| Warp-specialized TMA pipeline | ~10% SLOWER per block than single-buffer at full depth; parked WIP. |
| IPC window size / stream count | not the bottleneck (462-498 GB/s medians regardless of BATCH). |
| Idle-exit spin fix | removed 25-50ms jitter (was `__nanosleep(100)` = 10us sleeps); stability fix, median unchanged. |

## Forward plan

1. **RS/AG fusion is done** (fused reduce+copy + fused AG, device flags).
2. **Per-hop latency**: once fusion is fully in, re-measure the
   per-hop host cost; the remaining ring critical path is the next
   target — deepen the ring pipeline (staging buffers like NCCL
   SIMPLE's `NCCL_STEPS` allow more steps in flight without touching
   user buffers).
3. **Fewer-SM reduce**: the goal is native-class bandwidth at <=32
   blocks. Current lever set: ILP (done, capped), TMA bulk (done,
   99% at BLK=32), larger shim tiles (done), and multicast/NVLS
   (untested — the per-SM ceiling appears to be a memory/TMA-system
   property, ~14-15 GB/s payload per SM).
4. Re-measure 2/4/8 ranks on a quiet machine after each step.

## What we borrow from NCCL/mscclpp

NCCL's `genericOp<recvDirect, sendDirect, reduce, copy, ...>` fuses
copy/reduce/through in one kernel; its ring uses
`directRecvReduceDirectSend` (RS) and `directRecvCopyDirectSend` (AG
land+forward). We adopt the fusion shapes but keep sync on the host:
fused reduce+copy is the RS shape with the receive done by CE-land
instead of a remote read; fused AG is the copy shape as a device task.
We deliberately do NOT borrow the LL/LL128 device flag spinning or
mscclpp mbarrier waits (host-side completion is the project's core
constraint).
