# ukernel collective optimization — framework, attempts, and plan

How the AllReduce/AllToAll optimization work is organized: the design
principle, the three planes we optimize in, the measured attempts with
their verdicts, and the forward plan. Supporting data:
[b300_native_nccl_measurements.md](b300_native_nccl_measurements.md).
The CE-contention mechanism, put-path selection, and RS/AG fusion
write-ups are appendices A/B/C at the bottom of this document.

## Design principle

The GPU only computes; all synchronization and waiting is orchestrated by
the CPU (host polls, host signals, host-enqueued tasks). Device-side
spins on flags (NCCL's LL `readLL`, mscclpp mbarrier waits) are
deliberately out of scope — the project's motivation is unloading that
wait from the SMs. Everything below respects this.

### Put + reduce pipeline overlap status (2026-08-15)

With the register-spill fix the reduce is no longer the bottleneck (pure
reduce 604 GB/s @ 16 SMs, 1106 @ 32), so the remaining AllReduce gap is
the put/pipeline side. Measured on B300, 256M, 2 ranks, current vector
build:

- AllReduce 256M LT=8 TM=8M IB=8 BLK=32: 455-502 GB/s across runs
  (median ~475; native 512) = 93-98% of native. The pipeline overlap is
  working — the reduce phase is hidden behind the puts.
- Put engine ceiling is fine: AllGather in-place 256M 2 ranks hits
  **1017 GB/s** (native 833). The IPC CE path is not bandwidth-limited.
- All overlap variants regress at 2 ranks: signal aggregation G=2/4/8
  (448/393/407), fused AG copy (411), device put path (353). The optimal
  config stays CE + LT=8 + IB=8 + G=1.
- The residual ~2-7% is the ring's serialized per-tile dependency chain
  (reduce -> signal -> put -> signal -> next tile) plus run variance,
  not put/reduce overlap. RS remains the weakest phase (shim RS 521.6
  algbw vs native 817.5, 64%) — the target for reduce+receive fusion.

### RS chunking experiment — negative (2026-08-15)

`UK_CCL_RS_CHUNKS=K` splits each RS tile's put/reduce into K chunk ops
with independent pair ids, so chunk c's reduce can overlap chunk c+1's
put. Measured on B300, 256M RS, BLK=32 LT=8 (best per-config):

| chunks | time | algbw |
|---:|---:|---:|
| 1 | 505.7us | **530.8 GB/s** |
| 2 | 526.8us | 509.5 |
| 8 | 530.3us | 506.2 |

Chunking regresses — finer ops add per-tile dependency latency without
overlap gain, confirming the user's intuition that chunking == more
tiles (the LT sweep already showed LT=32, i.e. 8MB ops, at 370). The RS
gap is the per-tile dependency chain itself (host profiling: only ~16%
of the RS time is host dispatch; ~90us/tile is GPU/CE-side put-signal-
reduce latency), which neither tile size nor chunking addresses. C=4
failed to produce a result (not investigated — the direction is
decisive). The knob has been removed; the verdict stands.

### RS CE+device hybrid — negative (2026-08-15)

`UK_CCL_RS_HYBRID=1` splits each RS tile's send into a CE half and a
device-copy half (per-op `put_path_hint`), overlapping the CE engine and
the peer's worker on the same shard. Measured on B300, 256M RS, BLK=32
LT=8:

| ranks | CE only | hybrid | delta |
|---:|---:|---:|---:|
| 4 | 372.1 GB/s | 345.7 | -7% |
| 8 | 201.8 | 192.9 | -4% |

The device half competes with the reduce on the same worker, so the
hybrid loses even at 4/8 ranks. The alltoall device-path win (4/8 ranks)
was for pure copies (no reduce on the worker); RS's copy+reduce share
the worker, so splitting the send across engines does not help. The
knob has been removed; `put_path_hint` stays for the AllToAll hybrid.

## The three planes

### 1. Data plane — who moves bytes

| path | read | write | verdict |
|---|---|---|---|
| CE copy | local | peer | fast per copy at <=8-way concurrency (B300 ~630-700 GB/s); degrades 2-3x under synchronized 56-way peaks. Fine for ring steps (8-way). Batch submission does not relieve the peak (Appendix A). |
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
(G) was flat; the drain loop busy-polls (pauses, no scheduler yield)
while waits are pending and only yields when idle. The remaining cuts:
**device-completion flags** (device writes a slot, host polls it — no
atomics needed; removes dev-drain + enqueue + ring-write), and fewer
hops via deeper pipelining.

## Attempt log (verdicts)

| attempt | result |
|---|---|
| CE contention microbench (sync 56-way vs staggered) | CE sync 2-3x penalty; SM copy ~half as bad. Real, but ring RS is only 8-way. |
| `cudaMemcpyBatchAsync` batch submission | no effect on the sync peak (~2.1 TB/s both ways); IPC adapter stays per-peer. |
| Fused remote-read reduce (was `UK_CCL_FUSE_RS_REDUCE`) | 1.4-1.5x worse at 2/4/8 ranks; latency-bound. Dead end, removed. |
| Fused reduce+copy (`UK_CCL_FUSE_REDUCE_COPY`) + device flags | +7.5% (4r), +18.8% (8r); stress-validated wrong=0. Shipped. |
| Fused AG copy (`UK_CCL_FUSE_AG_COPY`) | neutral at 4r, ~5% at 8r (1560 -> 1487us). Shipped. |
| AG via standalone device puts (`UK_CCL_PUT_PATH=device` on AG) | regresses (worker serializes copies with reduces). AG fuses instead. |
| Signal aggregation (G=2/4 at LT=16) | flat. Signal COUNT is not the lever. |
| Drain busy-poll (conditional on pending waits) | shipped: spins on pauses while waits are pending, yields when idle; cuts 8-rank signal-drain latency. |
| LT sweep (pipeline depth) | LT=16 best; finer tiles regress — per-tile host cost caps depth. |
| REDUCE_ILP=16 | +24% per block on reduce bench; shim 95% of native at BLK=64; cannot reach native at <=32 blocks alone. |
| TMA bulk reduce (TMA_REDUCE=1, 224KB smem) | 99% of native at BLK=32 (509.5 GB/s); tail-chunk bug fixed; opt-in pending broader validation. |
| Warp-specialized TMA pipeline | ~10% SLOWER per block than single-buffer at full depth; removed (details in git history). |
| IPC window size / stream count | not the bottleneck (462-498 GB/s medians regardless of BATCH). |
| Idle-exit spin fix | removed 25-50ms jitter (was `__nanosleep(100)` = 10us sleeps); stability fix, median unchanged. |
| Lazy device worker (bind on first use) | reverted: the lazily created multi-block worker stalls on B300 (fifo bound=1, tail never advances) and hangs alltoall hybrid at 4/8 ranks. The first-launch warm-up also hit a CUDA 13.2/13.3 create/destroy context-poisoning issue. Dead end for now — zero-SM for all-CE is achieved at the plan level instead (pct=100 emits no device ops). |

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

## Appendix A — Copy-engine contention (merged from ce_contention.md)

8-GPU AllReduce kept falling short of native NCCL, and the suspicion was
copy-engine (CE) contention: when a collective starts synchronously, all
ranks' CE copies hit the fabric at the same instant. This appendix
records the standalone microbenchmark verification
([`bench/ce_contention.cu`](../bench/ce_contention.cu); run instructions
in [benchmarks.md](benchmarks.md)) and what the contention actually is.

### What the CE is

Each GPU has only a handful of DMA copy engines with very low
parallelism, scheduled per queue: only a few transfers are truly
concurrent, and each transfer is moved in blocks over hundreds of
microseconds. When transfers queue up, the engine also pays descriptor
switching and address resolution, first come first served.

### Unsync vs sync

- **Unsync**: each rank runs its own loop with no global alignment, so
  copies naturally stagger and only a fraction of the transfers are on
  the fabric at any instant. This is the engine's **ideal ceiling** and
  represents uncoordinated point-to-point traffic, not collectives.
- **Sync**: collectives carry completion dependencies — AllToAll waits
  for every peer, AllReduce is a ring chain — so each round is gated by
  the slowest peer and, after a few rounds, every rank's puts burst
  together. This is the collective norm.

Our shim is synchronous by construction: each round/tile waits for all
peer signals before the next round, so copy peaks naturally clump.
Tile-level pipelining only staggers the puts inside a round; the
round-to-round completion dependency remains.

### Two-level contention

At the sync instant, 8 ranks x 7 copies = 56 transfers enqueue almost
simultaneously:

1. **Per-GPU CE queue**: on average 7 transfers sit in front of each CE;
   FIFO scheduling favors whoever enqueued first — the barrier leader
   (rank 0) is consistently the fastest, which is the queueing
   fingerprint.
2. **Fabric/interface arbitration**: NVLink egress, ingress, and the
   memory system are shared. 56 transfers hitting every link at once
   degrade per-transfer effective bandwidth through link arbitration
   and packet interleaving.

This is not a bandwidth shortfall: an 8-rank synchronized round moves
only ~1.9 TB/s aggregate, far below the fabric ceiling — it is a
scheduling problem of "too many transfers x too little engine/interface
parallelism".

### Evidence (8 ranks, 256MB/rank, 32MB per copy)

| mode | per-copy time (us) | per-round aggregate |
|---|---:|---:|
| CE unsync | 54-60 | ~4.5 TB/s |
| CE sync peak | 69-161 (most ranks 2-3x worse) | ~1.9 TB/s |
| CE serial-sync (1 transfer per rank) | 54-156 | ~2.0 TB/s |
| SM copy sync peak | 52-103 | ~2.9 TB/s |

Three observations:

1. **CE contention is real**: the synchronized peak costs 2-3x per copy,
   and the first enqueuer (rank 0) always wins — clear queueing.
2. **Serializing per rank does not help**: 8 concurrent transfers
   degrade almost as much as 56. The bottleneck is not "one CE queue is
   too long" but "the synchronized start overloads fabric arbitration
   at once". This rules out the simple "CE queue is the only cause"
   explanation.
3. **SM copies survive the peak far better**: the same synchronized
   pattern through vectorized LD/ST reaches 2.9 TB/s aggregate, only
   1-2x slower per copy. Hundreds of thousands of threads spread the
   queueing into high-parallelism throughput, bypassing the CE's narrow
   scheduling entry.

### Batch submission does not relieve it (2026-08-12)

NCCL 2.28+ zero-SM collectives submit one AllToAll's copies as a single
`cudaMemcpyBatchAsync` call (`srcAccessOrder=Stream` +
`PreferOverlapWithCompute`, one stream). We mirrored that call exactly
in the microbenchmark's `--batch` mode and compared it against per-peer
submission on B300 (8 ranks, 256MB/rank, 32MB/copy, 20 and 50 iters):

| submission | unsync aggregate | sync aggregate | per-copy sync/unsync |
|---|---:|---:|---|
| per-peer (7 streams x 7 `cudaMemcpyAsync`) | ~4.5 TB/s | ~2.1 TB/s | 1.3-2.6x |
| batch (1 stream x 1 `cudaMemcpyBatchAsync`) | ~4.5 TB/s | ~2.1 TB/s | 1.3-2.6x |

Two independent runs agree: **batch submission does not help the
synchronized peak**. Rank 0 stays fastest in both modes and the queueing
degradation is identical, so the bottleneck is not driver
submission/descriptor overhead — it is the CE queue and fabric
arbitration themselves.

Inference for NCCL: its CE path does not dodge this contention either —
batching saves driver overhead and multicast sync saves host round
trips, but the transfer scheduling cost remains. Hence:

1. **Do not switch the IPC adapter to batch submission** — it does not
   help contention and would add risk for nothing.
2. The only CE levers left are symmetric-memory/multicast (cross-rank
   signal broadcast, multicast writes), which require CUDA 13 symm_mem
   and cannot be reproduced with plain IPC handles.

### Conclusions

1. CE contention is real and two-level: per-GPU CE queueing plus
   fabric/interface arbitration overload.
2. The CE is the main but not the only bottleneck: SM copies under the
   same synchronized peak are clearly better, so the fabric's burst
   start/arbitration has its own cost that the CE amplifies by ~50%.
3. Design implication: **fusion is the right direction** — a reduce
   kernel that LD/STs peer memory directly (dropping the CE put) moves
   the copy from an engine that cannot handle the synchronized peak to
   one that can, recovering most of the gap. The residual fabric cost
   is not recoverable by path switching; the ~4.5 TB/s unsync ceiling
   is not reachable for collectives.
4. Native NCCL's LL protocol does kernel copies without the CE for the
   same reason.

## Appendix B — Put path selection (merged from put_path_selection.md)

Design notes on how the ukernel CCL routes same-host cross-GPU puts, why
the original latency-metric load balancer was disabled, and what a
correct adaptive selector would look like.

### Current state

Same-host cross-GPU puts are pinned to **IPC** (`pick_put_path` returns
`PutPath::Ipc` unconditionally for same-host peers). Remote peers always
use RDMA. `UK_CCL_PUT_PATH=device|ipc|rdma` forces any path for A/B
benchmarking.

Measured on the A40 pair (2 ranks, 256MB AllGather):

| path | throughput | note |
|---|---|---|
| IPC (sliding window) | 70 GB/s aggregate | the fast path |
| device (kernel copy, 8 blocks) | ~36 GB/s | single-block was ~18 GB/s |
| RDMA loopback | ~2 GB/s | data loops through host/NIC |

Pinning same-host puts to IPC took 256MB AllReduce from 12.5ms to
5.5ms (beating native NCCL's 6.35ms) and AllGather from 15.9ms to
3.8ms.

**On B300 the device path wins for AllToAll**: with the copy op reduced
to plain vectorized LD/ST (the same mechanism NCCL uses intra-node; TMA
bulk copies hang on peer-mapped addresses), the device path at
`UK_CCL_DEV_BLOCKS=64` reaches ~400 GB/s algbw for 256MB alltoall at 8
ranks — ~15% faster than the IPC/CE path (715 -> ~600us) and it
flattens the rank-scaling curve (4r ~= 8r). It still loses marginally
at 2 ranks (324 vs 310us). The selector should therefore be rank-count
and message-size aware, not a single same-host default.

### Why the latency-metric balancer failed

The original `pick_put_path` picked the path minimizing
`inflight × latency_ns` (a Little's-law style queueing estimate). It was
wrong for two structural reasons:

1. **The sliding window inflates the IPC latency metric.** IPC
   completions are published in bursts after a window sync, so the
   measured per-put completion latency looks high — even though IPC has
   the highest throughput. The metric and the physical truth disagreed.
2. **The device path has low latency but low capacity.** A single-block
   persistent-kernel copy task completes fast (small tasks, low latency)
   but only sustains ~18 GB/s. Latency is a poor proxy for capacity,
   and the device path's capacity additionally depends on
   `blocks_per_worker` (8 blocks ≈ 40 GB/s, 64 blocks ≈ 52 GB/s), which
   the metric never knew.

### When each path can genuinely win

| path | when it can win | verdict |
|---|---|---|
| IPC | almost always, same-host — it *is* the GPU copy-engine/DMA path | correct default at 2 ranks and for small messages |
| device | (a) copy engines contended or PCIe-capped; (b) vectorized LD/ST at high block counts — **beats IPC by ~15% for 4+ rank large-message AllToAll on B300**; (c) the fused RS/AG path wins by removing per-hop host transitions | per-put device path wins large-message multi-rank AllToAll; the fused path owns AllReduce RS/AG |
| RDMA (same-host) | essentially never — data loops through host/NIC | keep excluded from same-host selection |

Cross-node traffic is always RDMA (the ring seams between nodes); the
selector only ever decides same-host IPC-vs-device.

### Correct adaptive design: capacity probes, not latency

A proper selector measures what it needs instead of guessing:

1. **At path setup, probe each candidate**: one large transfer for
   capacity (GB/s), one small transfer for latency; cache the results.
2. **Select by message size**: large messages → highest-capacity path;
   small messages → lowest-latency path.
3. **Same-host excludes RDMA** unconditionally.
4. **Re-probe periodically / on path failure** (e.g., IPC completions
   stall → fall back to device).
5. The capacity probe naturally accounts for configuration-dependent
   device throughput (it measures the current `blocks_per_worker`).

Estimated size: 100-200 lines, behind an env switch so the current
"pinned IPC" behavior stays the default until a machine demonstrates a
need.

### Recommendation

- Keep "same-host → IPC" as the default for 2 ranks and small messages.
- Use the device path for 4+ rank large-message AllToAll on B300
  (`UK_CCL_PUT_PATH=device UK_CCL_DEV_BLOCKS=64`) — measure and pick per
  rank count / message size.
- Do not reintroduce latency-metric-based selection — it was measuring
  the wrong quantity.
- Cross-node (the multi-node goal) is RDMA-only regardless; the selector
  only ever decides same-host IPC-vs-device.

## Appendix C — AllReduce copy+reduce fusion, fused RS/AG (merged from fused_rs_reduce.md)

Status: **implemented on `uk-300`** — `UK_CCL_FUSE_REDUCE_COPY=1`
(fused reduce+copy in the reduce-scatter phase) and
`UK_CCL_FUSE_AG_COPY=1` (fused device copy in the all-gather phase).
Measured gain: +7.5% at 4 ranks, +18.8% at 8 ranks (256M AllReduce,
OOP). The earlier remote-read reduce variant was a dead end and has
been removed (details in git history).

### Motivation

The AllReduce ring is a serialized per-hop critical path: RS (7 hops) +
AG (7 hops) at 8 ranks, with a host signal chain on every hop
(task done -> dev drain -> enqueue signal -> host ring write -> peer
poll -> enqueue next task). CE contention (Appendix A) and per-tile
host-signal latency dominate the gap to native NCCL (which keeps sync
in-kernel with LL flags). The fusion attacks both: reduce and copy
happen in one device task, and completion is signaled with a
device-written flag that the host polls directly.

### Design space: two fused-RS shapes

**Remote-read reduce — dead end (was UK_CCL_FUSE_RS_REDUCE)**. The
receiver's reduce kernel reads the peer's send-source buffer directly
over NVLink (NCCL LL-style), and the sender's signal means "data ready"
instead of "put landed". Measured 1.4-1.5x SLOWER at 2/4/8 ranks: the
remote read is latency-bound — throughput scales almost linearly with
block count (43 GB/s at BLK=8 -> 179 GB/s at BLK=64) and collapses at
BLK=128, which defeats the few-SM goal. It also proved that CE
contention is NOT the dominant AllReduce cost: a ring step has only 8
concurrent copies (one per rank), not the 56-way alltoall peak.

**Fused reduce+copy — the winning shape (UK_CCL_FUSE_REDUCE_COPY)**.
Each RS RecvReduce task reduces its shard and forwards it to the next
rank's accumulation buffer in the same task — device LD/ST write to the
peer (the alltoall-proven direction). This removes the reduce->put host
transition and the separate put op from the ring's per-hop critical
path.

### Completion signaling: device-flag slots

B300 reports `gpuDevAttrHostNativeAtomicSupported=0`, so kernels cannot
claim the shared IPC signal ring. Instead each fused task owns a
single-writer slot in a host-mapped flag area:

- The device task writes the salted tag with a plain store +
  `__threadfence_system` (no atomics); the matching WaitSignal polls the
  slot from the host. This removes the dev-drain -> enqueue -> host
  ring-write transitions.
- Slots are collision-free (`pair*K + tile`, K = plan tile bound);
  plans that exceed the fixed flag area fall back to host-written
  signals.
- G>1 uses counted waits: poll `flag_count` consecutive slots and
  complete when all match `base_tag + i`.
- Two correctness fixes are folded in: the unconditional salted tag in
  `make_cmd` (a fused-signal conditional zeroed tags for ordinary ops
  and deadlocked AllReduce), and a dedicated `signal_tag` field so the
  flag tag cannot clobber `TaskArgs.redTypeRaw` (slot 0 tripped the
  reduction assert).

Toggles: `UK_CCL_FUSE_REDUCE_COPY` (default 0, forces G=1), and
`UK_CCL_DEVICE_FLAGS` (default on when fused).

### Fused AG copy (UK_CCL_FUSE_AG_COPY=1)

The AG forward becomes a device copy task (read my output, write next's
output) with an inline device-completion flag — no CE, no host signal
per hop. It reuses the RS flag machinery (per-tile slots, counted waits,
capacity fallback). The executor routes these puts to the device backend
only; `UK_CCL_DEV_FIFOS=4` over-subscribes the SMs (4x64 > 148 SMs) and
collapses, so keep the default 2 workers.

### Results (B300, 256M AllReduce, LT=16 TM=8M IB=16 BLK=64, n=20, OOP)

| config | 4r | 8r |
|---|---:|---:|
| fuse=0 | 1177us | 2007us |
| fused RS | 1122us | 1560us |
| fused RS+AG | 1129us | **1487us** |
| native | ~669us | ~719us |

All wrong=0; 8-rank stress (n=100) wrong=0 (validates flag write
ordering under sustained load). RS fusion: +7.5% at 4r, +18.8% at 8r.
AG fusion is neutral at 4 ranks (worker serialization offsets the
host-chain savings) and ~5% at 8 ranks (1560 -> 1487us). The remaining
~770us gap at 8 ranks is the ring's serialized critical path (14 hops x
per-hop latency) and the residual AG/put pipeline.

LT sweep: LT=16 BLK=64 remains the fused-path optimum; deeper tiles
regress (per-tile host cost caps depth); BLK=128 over-subscribes and
collapses.

### Build speed

`persistent_kernel_ops.cu` with `TMA_REDUCE=1 REDUCE_SMEM_KB=224` takes
15-25min (TMA bulk/warp-spec template instantiations). C++-only
iterations relink in ~1min. Validation builds: `make VALIDATE=1 -j8
nccl` disables the TMA paths (the fused work runs on the vector LD/ST
path); keep `TMA_REDUCE=1` only for final perf builds.
