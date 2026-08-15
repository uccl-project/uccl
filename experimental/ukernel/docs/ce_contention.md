# Copy-engine contention: mechanism, evidence, conclusions

8-GPU AllReduce kept falling short of native NCCL, and the suspicion was
copy-engine (CE) contention: when a collective starts synchronously, all
ranks' CE copies hit the fabric at the same instant. This document
verifies that hypothesis with the standalone microbenchmark
[`bench/ce_contention.cu`](../bench/ce_contention.cu) and records what
the contention actually is. Reproduction commands are in
[benchmarks.md](benchmarks.md).

## What the CE is

Each GPU has only a handful of DMA copy engines with very low
parallelism, scheduled per queue: only a few transfers are truly
concurrent, and each transfer is moved in blocks over hundreds of
microseconds. When transfers queue up, the engine also pays descriptor
switching and address resolution, first come first served.

## Unsync vs sync issue

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

## Two-level contention

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

## Evidence (8 ranks, 256MB/rank, 32MB per copy)

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

## Batch submission does not relieve it (2026-08-12)

NCCL 2.28+ zero-SM collectives (CE collectives) submit one AllToAll's
copies as a single `cudaMemcpyBatchAsync` call
(`srcAccessOrder=Stream` + `PreferOverlapWithCompute`, one stream). The
hypothesis worth testing: batch submission lets the driver distribute
the 7 copies across DMA queues, easing the sync peak. We added a
`--batch` mode to the microbenchmark that mirrors NCCL's call exactly
and compared it against per-peer submission on the B300 (8 ranks,
256MB/rank, 32MB/copy, 20 and 50 iters):

| submission | unsync aggregate | sync aggregate | per-copy sync/unsync |
|---|---:|---:|---|
| per-peer (7 streams x 7 `cudaMemcpyAsync`) | ~4.5 TB/s | ~2.1 TB/s | 1.3-2.6x |
| batch (1 stream x 1 `cudaMemcpyBatchAsync`) | ~4.5 TB/s | ~2.1 TB/s | 1.3-2.6x |

Two independent runs agree: **batch submission does not help the
synchronized peak**. Rank 0 stays fastest in both modes and the queueing
degradation is identical, so the bottleneck is not driver
submission/descriptor overhead — it is the CE queue and fabric
arbitration themselves.

The inference for NCCL is that its CE path does not dodge this
contention either: batching saves driver overhead and multicast sync
saves host round trips, but the transfer scheduling cost remains (NCCL
documents its CE collectives as trading bandwidth for SM usage). Hence:

1. **Do not switch the IPC adapter to batch submission** — it does not
   help contention and would add risk for nothing.
2. The only CE levers left are symmetric-memory/multicast (cross-rank
   signal broadcast, multicast writes), which require CUDA 13 symm_mem
   and cannot be reproduced with plain IPC handles.

## Conclusions

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
