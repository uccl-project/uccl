# Reduce kernel: tuning, scaling, and design

How the device reduce kernel reaches bandwidth at the fewest SMs: ILP
vectorization, tile size, TMA bulk, and the (parked) warp-specialized
pipeline. Consolidated from the earlier `reduce_ilp_tuning.md`,
`tile_sm_scaling.md`, and `warp_spec_reduce_design.md`.

## Goal and native anchor

Lift per-block reduce throughput so the shim reaches native-class
bandwidth at **32 blocks or fewer**. Native anchor on B300: 32 coll
channels ≈ 16 GB/s per channel, 510 GB/s @ 256M (2 ranks) — see
[b300_native_nccl_measurements.md](b300_native_nccl_measurements.md).
Throwing more blocks at it is not the goal.

## Why ILP

The reduce kernel is latency-bound: each thread keeps `U` independent
16B loads (of `src` and `dst`) in flight. Saturating DRAM needs
`in-flight bytes ≈ latency x bandwidth`; B300's HBM3e
latency-bandwidth product is ~5x A40's, so it needs ~5x the in-flight
bytes per block — the same kernel that saturates A40 at 8 blocks needs
32-128 blocks on B300. `REDUCE_ILP` (=4|8|16, default 4) raises `U`
without adding blocks. Cost: registers (each in-flight vector = 4 regs);
U=16 spills at 256 threads on A40 but fits on B300.

ILP is a **build-time knob** (`make ... REDUCE_ILP=8`): a runtime
dispatch over 4/8/16 tripled cicc+ptxas time to ~20 min per device file
on B300; one compile-time value builds the file in ~15-20s.

## ILP results

Reduce bench (256M payload, 256 threads, persistent worker, B300,
2026-08-04):

| blocks | ILP=4 | ILP=8 | ILP=16 |
|---:|---:|---:|---:|
| 8 | 57.4 | 60.2 | **71.0** |
| 16 | 112.6 | 119.2 | **140.9** |
| 32 | 219.7 | 232.4 | **274.9** |
| 64 | 420.6 | 440.9 | **515.3** |

Shim AllReduce 256M (`LT=8 TM=8M IB=16`, OOP, 0 wrong):

| blocks | ILP=4 | ILP=8 | ILP=16 |
|---:|---:|---:|---:|
| 8 | 172.3 | 181.8 | **198.0** |
| 32 | 365.0 | 387.3 | 389.6 |
| 64 | 442.4 | 427.8 | **489.8** |

Conclusions:

- U=16 is the clear winner on B300 (+24% per block on the reduce bench).
  The kernel at 64 blocks x U=16 hits parity with native (515 GB/s);
  the shim reaches 489.8 GB/s (95%) at 64 blocks.
- Per-block throughput is still ~8-9 GB/s (515/64) vs native's ~16 GB/s
  per channel — **ILP alone cannot reach native at <=32 blocks**
  (32 blocks x U=16: 390 GB/s, 76%).
- BLK=32 AllReduce plateaus at ~390 GB/s regardless of ILP: at that
  block count the reduce is no longer the only limiter
  (put/executor pipeline), so put/reduce overlap is complementary.
- A40 reference: saturates DRAM at ILP=4 / 32 blocks; higher ILP only
  adds register pressure (ILP=16 spills).

## Tile size scaling

`adaptive_tile_bytes()` splits a message into `UK_CCL_LARGE_TILES`
rings tiles; each ring tile becomes one reduce task split across
`UK_CCL_DEV_BLOCKS` blocks. Small tiles mean tiny per-block slices and
per-task fixed costs (host scheduling, multi-block barrier, put window)
dominate.

256M AllReduce, 2 ranks, B300 (GB/s, oop / ip; vectorized ILP path):

| LARGE_TILES (tile) | BLK=8 | BLK=16 | BLK=32 |
|---|---:|---:|---:|
| 64 (4MB) | 169 / 131 | 200 / 178 | 172-210 / 168-188 |
| 32 (8MB) | 242 / 165 | 265 / 222 | 273-303 / 249-266 |
| 16 (16MB) | 293 / 192 | 347-363 / 288 | 397-457 / 333-390 |
| 8 (32MB) | 299-308 / 197 | 378-398 / 310-321 | 440-465 / 393-413 |
| 4 (64MB) | 277 / 191 | 365 / 296 | 431 / 398 |

Findings:

- **Tile size is the dominant lever at low SM counts**: LT=64->8 more
  than doubles BLK=16 throughput (200 -> 398 GB/s).
- **Per-SM efficiency drops as SMs are added** (BLK=8: ~19 GB/s/SM,
  BLK=16: ~12.5, BLK=32: ~7): bandwidth saturates well before the
  per-SM ceiling. 32 SMs reach ~400, 64 SMs ~450; native gets ~510
  with 32 channels.
- Valid BLKs are powers of two: non-power-of-two BLK misaligns the
  per-block slice for the TMA path (crash).

### In-place gap

In-place AllReduce is 15-25% slower than out-of-place, NOT because of
the reduce kernel (its dst/src are always distinct: dst=Tmp
accumulation, src=Input). The ring stages in-place RS partials in
Tmp(0), and the AG phase ends with a local **Tmp->Output copy of the
held shard (128MB per rank at 256MB/2 ranks)** — that extra copy is the
gap. A chunked TMA bulk copy for that copy path shrank the gap from
~25% to ~8% at BLK=16; the remainder is put/copy pipeline dependencies
in the AG, not the copy kernel.

### 2-rank allreduce is NVLink-bound

At BLK=32 (~460 GB/s oop) the 2-rank ring moves ~460 GB/s per direction
over NVLink — the B300 single-direction ceiling is ~450-500 GB/s
(native measures 510). The allreduce is **NVLink-bound, not
reduce-bound** (pure reduce at BLK=32 is 640 GB/s). Remaining headroom
to native is the put path keeping all NVLink lanes busy (native uses 32
channels; the shim uses a single IPC sliding window).

## TMA bulk reduce (TMA_REDUCE=1)

Build: `make SM=103 ENABLE_TMA=0 REDUCE_ILP=4 REDUCE_SMEM_KB=224
TMA_REDUCE=1 TMA_WARPSPEC=0`. Per chunk: `cp.async.bulk` load src+dst
into smem (mbarrier complete_tx), reduce in smem, bulk-group store back,
`fence.proxy.async.global` at task completion.

Shim AllReduce 256M (`LT=8 TM=8M IB=16`), vs ILP=4 baseline:

| blocks | ILP=4 | TMA | wrong |
|---:|---:|---:|---:|
| 8 | 172 | **292** | 0 after tail fix |
| 32 | 365 | **509.5** | 0 after tail fix |
| 64 | 412 | 486 | 0 after tail fix |

TMA at 32 blocks = **509.5 GB/s ≈ 99% of native 515** — the
"32 blocks to parity" goal is met on bandwidth. Correctness history:
the initial wrongness was the odd-sized **tail chunk** of each block
slice (TMA bulk on a small final chunk wrote stale-smem garbage); the
tail now falls back to the ILP vector path and every size 1M..256M is
wrong=0.

Bigger per-block smem (REDUCE_SMEM_KB) gives modest gains — the
bottleneck at low block counts is per-chunk mbarrier wait + store
serialization, not raw in-flight bytes:

| blocks | 128KB | 192KB | 224KB |
|---:|---:|---:|---:|
| 8 | 123.4 | 131.8 | 134.7 |
| 16 | 241.8 | 256.8 | 262.7 |
| 32 | 461.0 | 488.5 | 493.1 |

`cp.reduce.async.bulk` (global->shared reduce on TMA) is integer-only
in current PTX — **not usable for f32 sum**.

TMA stays opt-in (`TMA_REDUCE=1`) pending broader validation
(multi-node RDMA + the put/reduce pipeline work); validation builds use
`make VALIDATE=1` which forces the vector path for speed.

## Warp-specialized TMA pipeline — parked

A producer/consumer pipeline (producer warp drives TMA loads/stores,
consumer warps reduce) was implemented with the canonical protocol —
mbarrier init-once + phase toggle per use (DeepEP/ThunderKittens
pattern; re-init per chunk is unsafe without a full `__syncthreads()`
fence), named barriers for consumer completion, and
`fence.proxy.async.shared::cta` before TMA stores. Correctness-verified
(wrong=0 at 1M/256M, BLK 16/32), but:

- **Warp-spec is ~10% SLOWER per block than single-buffer at full
  pipeline depth** (device bench 256M @ 32 blocks: 446 GB/s vs 493
  GB/s). Total in-flight bytes per block are identical (224KB smem cap);
  the smaller chunks only add per-chunk mbarrier/barrier overhead.
- The shim's small tiles made it worse: at BLK=32 with 4MB tiles each
  block gets 128KB/task -> the pipeline never fills.
- **Per-SM ceiling ≈ 14-15 GB/s payload (42-45 GB/s memory traffic) is
  a memory/TMA-system property, not pipeline structure.** Fewer-SMs-at-
  full-rate needs more in-flight bytes per SM: bigger tiles (shim),
  multicast/NVLS, or cluster-shared smem (2-CTA DSMEM doubles the
  per-SM budget).
- An intermittent full-speed wedge at deep pipelines (nfull >= 9) was
  never pinned down; low priority given the negative result.

Design details (protocol, smem layout, producer/consumer loops) are in
git history; the canonical mbarrier pattern is the reference for any
future cluster/multicast work.

## IPC window size is not the bottleneck

`UK_CCL_IPC_BATCH` sweep (LT=8, BLK=32, 256M, 2 ranks): medians sit in
462-498 GB/s regardless of window (4/8/16/32/64) with ±10% run-to-run
jitter inside every value. The put path already hits ~630 GB/s aggregate
— near NVLink bidirectional saturation — and host-side launch-ahead is
sufficient even at window 4. "Bigger window" and "more streams" are dead
ends; `kIpcSendBatchDefault` was set to 4 (best median, lower host event
pressure).

## Idle-exit spin fix

`idle_sleep()` used `__nanosleep(100)` — the argument is a multiple of
100ns, so each poll slept **10us**, not the ~100ns the poll-count
derivation assumes. A 500us idle grace therefore took ~50ms to actually
exit (seen as ~25ms periodic gaps in nsys traces). Fixed to
`__nanosleep(1)`; actual exit latency is now ~1.5-2.5ms. Effect: oop
run-to-run range tightened from ±10% to ±2% (median unchanged) — a
stability fix, not a peak fix.

## Practical guidance

- Default `REDUCE_ILP=4` for fast builds; `REDUCE_ILP=16` for peak
  runs (64 blocks ≈ 95% of native, 32 blocks ≈ 76% on the ILP path).
- Peak builds: `make SM=103 REDUCE_ILP=4 REDUCE_SMEM_KB=224
  TMA_REDUCE=1 TMA_WARPSPEC=0 -j8 nccl` (TMA reaches 99% of native at
  BLK=32).
- Validation builds: `make SM=103 VALIDATE=1 -j8 nccl`.
- Run-to-run variance is ±3-10%; compare medians over 3+ runs.
