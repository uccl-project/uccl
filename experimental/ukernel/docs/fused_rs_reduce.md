# Fused reduce-scatter (allreduce copy+reduce fusion)

Status: implemented (branch uk-300, `UK_CCL_FUSE_RS_REDUCE=1`) and
benchmarked on B300 — **negative result, kept behind the flag (default
0)** as a reference. See "Results" below.

## Motivation

The CE contention microbenchmark
([ce_contention.md](ce_contention.md)) showed that a synchronized
8-rank alltoall peak costs 2-3x per CE copy vs staggered issue, while an
SM copy kernel under the same synchronized peak suffers only ~half as
much (2.9 TB/s vs 1.9 TB/s aggregate). The allreduce ring's
reduce-scatter phase is exactly that synchronized pattern: every rank's
CE put delivers a shard into the receiver's accumulation buffer, and all
ranks' puts peak together. This change removes the RS-phase CE entirely:
the receiver's reduce kernel reads the peer's send-source buffer directly
over NVLink (NCCL's LL-protocol mechanism), and the sender signals
"data ready" instead of "put landed".

## Protocol

### Ring reduce-scatter (fused)

Per step s, rank R sends shard `wrap(R-s)` to next and receives shard
`wrap(R-s-1)` from prev:

- **Send** becomes a standalone Signal to next (pair = `shard*2`):
  - s=0: the rank's own shard, valid at collective start — no dep.
  - s>0: the shard produced by the previous step's RecvReduce — dep on
    that reduce op.
- **Receive** keeps its WaitSignal (same pair), but the sender's signal
  now fires after its *producing reduce* completes (host-side, via the
  executor's existing Signal dispatch), not after a CE copy.
- **RecvReduce** resolves its src to the peer's buffer:
  - In-place: dst = my Input[off], src = peer's Input[off], RMW
    (dst = dst op src). Safe because there are no puts into my buffer
    anymore — my Input[off] is only written by my own reduce, which runs
    after the ring's earlier reader consumed the original value, and the
    next reader is gated on my reduce.
  - Out-of-place: dst = my Output[off], src = peer's buffer (Input at
    step 0, Output afterwards), plus src2 = my Input[off], 3-way fresh
    write (dst = src op src2) — no seed copy needed.

The allgather phase is unchanged (its puts keep the configured CE/device
path); in-place AG skips the Tmp->Output publish because the RS-held
shard already lives in Output (Input == Output in-place).

## Implementation map

- `coll_algo.cc` — `emit_ring_reduce_scatter(..., fused)`: Signal chunks
  replace RS puts; RecvReduce carries the peer's buffer role +
  `fuse_remote_src` + `reduce_mode`; Recv chunks mark
  `wait_standalone_signal`.
- `lower.cc` — Signal chunks lower to one standalone Signal per G-tile
  group (mirroring the receiver's WaitSignal groups); fused RecvReduce
  carries `src_peer`/`src2`/`reduce_mode`; fused Recv chunks skip the
  put-fused wait-count metadata (plain one-arrival waits).
- `device_backend.cc` — existing remote-src resolution (already used by
  the device put path) now serves fused reduces; `kCmdFlagReduce3Way`
  resolves src2 (local Input at the shard offset).
- `persistent_kernel_ops.cu` / `ops.h` — `read_two_src_reduce_store`
  (fresh 3-way LD/ST); remote-src reduces force the LD/ST path (TMA
  `cp.async.bulk` hangs on peer-mapped addresses, measured earlier).
- `task.h` — TaskArgs gains `taskFlags` (kFlagReduce3Way).
- `executor.cc` / `nccl.cc` — Cmd gains `src2_buf`; plan key covers the
  flag; `UK_CCL_FUSE_RS_REDUCE` (default 0).

## Test plan

Correctness (wrong=0) + bandwidth, 2/4/8 ranks, 256MB, in-place and
out-of-place:

1. baseline `UK_CCL_FUSE_RS_REDUCE=0` (current best LT/IB)
2. fused, CE path for AG (`=1`, default put path)
3. fused + `UK_CCL_PUT_PATH=device` (AG copies also via SM — the whole
   allreduce becomes SM-only, no CE at all)

Expected: fused should beat baseline at 4/8 ranks (CE contention is the
measured bottleneck); the residual fabric cost documented in
ce_contention.md bounds how much of the native gap can close.

## Results (B300, 256M allreduce, `LT=8 TM=8M IB=16 BLK=64`, n=20)

OOP time us / algbw GB/s:

| ranks | fuse=0 | fuse=1 | native |
|---:|---:|---:|---:|
| 2 | 567 / 473 | 844 / 318 | 529 / 507 |
| 4 | 1078 / 249 | 1501 / 179 | 676 / 595 |
| 8 | 2158 / 124 | 3105 / 86 | 720 / 373 |

All wrong=0. The fused path is ~1.4-1.5x SLOWER at every rank count.
`UK_CCL_PUT_PATH=device` on top does not help (4r: 1400us; the device
copy competes with the reduce for the worker, as observed before).

### Why it loses: the remote read is latency-bound

4-rank fuse=1 BLK sweep (256M):

| BLK | time us | algbw GB/s |
|---:|---:|---:|
| 8 | 6223 | 43 |
| 16 | 3518 | 76 |
| 32 | 2175 | 123 |
| 64 | 1501 | 179 |
| 128 | 71405 | 4 (over-subscription collapse — also happens to fuse=0) |

Throughput scales almost linearly with blocks: the fused reduce's
NVLink read latency needs far more blocks to hide than the local RMW
reduce, which defeats the few-SM goal and never catches the baseline
even at 64 blocks.

### What this says about the allreduce bottleneck

1. **CE contention is NOT the dominant allreduce cost.** In the ring's
   reduce-scatter each step has only 8 concurrent CE copies (one per
   rank), not the 56 of a synchronized alltoall peak — the CE contention
   microbenchmark overstates the ring's CE load. Removing the RS CE
   entirely (fuse=1) buys nothing and adds a slower data path.
2. **The dominant cost is the ring's serialized critical path.**
   256M/8 ranks = 32MB shards; with LT=8 (one tile per shard) the ring
   is fully serialized: 14 hops (RS 7 + AG 7) x per-hop latency ~= the
   measured ~2.1ms. LT=16 (2 tiles/shard, 2-deep pipeline) is best at
   1954us; finer tiles (LT=32/64/128: 2557/2667/2426us) regress because
   the per-tile host signal chain dominates (host-prof: enq 0.01us/op,
   sig 4-14us/sig, dev 21-36us/dev).
3. The ring's RS and AG phases cannot overlap at shard granularity: every
   shard's full sum completes at its holder at RS step n-2, so the AG
   starts only after the whole RS. The real pipeline lever is deepening
   the ring's internal tile pipeline, which is currently capped by the
   per-tile host signal cost.

### Direction

The fused remote-read path is dropped. The next levers, in order:

1. **Cut the per-tile host signal cost** (the cap on pipeline depth):
   signal aggregation tuned to not regress 2-rank, or device-side
   signal matching — this unlocks finer ring pipelining.
2. **CE + SM-copy hybrid** for the collectives whose CE load is genuinely
   concurrent: alltoall (56-way, device path already wins at 4+ ranks),
   then allgather/reduce-scatter rings (8-way per step — test whether
   the device path helps there).
3. Revisit deep pipelining once the per-tile cost is down.

## Fused reduce+copy (task fusion, `UK_CCL_FUSE_REDUCE_COPY=1`)

Instead of the dead remote-read, each RS RecvReduce task now ALSO copies
its reduced shard to the next rank's accumulation buffer (device LD/ST
write to peer — the alltoall-proven direction) in the same task. The
data-ready signal is a separate host-written Signal op: B300 reports
`gpuDevAttrHostNativeAtomicSupported=0`, so the IPC signal ring is not
GPU-mapped and the kernel cannot write it (device-side fused signals
are disabled on this machine). Per-hop host transitions drop from
reduce->put-enqueue->put-done->signal to reduce+copy-done->signal.

Intermediate results (256M allreduce, LT=16 TM=8M IB=16 BLK=64, n=20,
OOP):

| ranks | fuse=0 | fuse_reduce_copy=1 | native |
|---:|---:|---:|---:|
| 2 | 624us / 430 GB/s | 617us / 435 GB/s | 529 / 507 |
| 4 | 1005-1143us / 235-267 | 1066us / 252 | 676 / 595 |
| 8 | (blocked — vLLM on GPUs 0-3) | — | 720 / 373 |

2-rank is neutral by construction (n=2 has no fused tasks — pure
plumbing check). 4-rank is neutral within run noise. The 8-rank test is
pending a quiet machine; that is where the per-hop savings accumulate.
