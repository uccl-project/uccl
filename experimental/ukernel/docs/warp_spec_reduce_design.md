# Warp-specialized TMA reduce — design

Status: **implemented + validated, but no per-SM win over the single-buffer
path — parked as WIP.** `UK_TMA_WARPSPEC=1` (build-time knob in
`src/ccl/Makefile` / `src/device/Makefile`) enables the producer/consumer
pipeline; the canonical protocol (init-once + derived parity, named
barrier, correct arrive counts) is in place and correctness-verified.
Performance conclusion: see "Experiment results" below — the pipeline
does not raise the per-SM ceiling, so the default stays the single-buffer
TMA reduce.

> **UPDATE (2026-08-04, B300 experiments):** the design is implemented and
> correct (wrong=0 at 1M/256M, BLK 32/16) but **does not beat the
> single-buffer TMA path per SM** — see "Experiment results" below. The
> pipelining hypothesis (per-chunk load→reduce→store serialization was the
> per-SM limiter) is largely DISPROVEN. Recommend keeping the single-buffer
> TMA (224KB, BLK=32 ≈ 443-490 GB/s) as default and pursuing bigger shim
> tiles / multicast for the fewer-SMs goal.

## Goal and measured baseline

Goal: lift **per-block** reduce throughput so fewer SMs saturate the NVLink.
Measured so far on B300 (256MB all-reduce, 2 ranks, 224KB smem):

| config | BW | notes |
|---|---|---|
| native NCCL 2.29.7 | ~510 GB/s | 32 coll channels, ~16 GB/s payload per channel/SM |
| single-buffer TMA, BLK=32 | ~490 GB/s (409 oop) | validated, 0 wrong |
| single-buffer TMA, BLK=16 | ~420 GB/s | per-block cap shows here |
| single-buffer TMA, BLK=8 | ~300 GB/s | clearly block-limited |

The single-buffer path serializes per chunk: `bulk-load src+dst` →
`mbarrier wait` → `reduce in smem` → `bulk-store dst` → `wait_group` →
next chunk. Per-block throughput is ~constant at every block count, which
means the per-chunk **serialization** (not raw in-flight bytes) is the
limiter. Warp specialization turns that chain into a pipeline:
load[N+1]/store[N-1] overlap reduce[N].

## Canonical warp-specialization patterns (studied)

### FlashAttention-3 / CUTLASS (Hopper+)

- 128-256-thread CTA split into a producer warpgroup (TMA loads) and
  consumer warpgroups (WGMMA compute).
- `PipelineTmaAsync`: per-stage "full" (data ready) and "empty" (stage
  reusable) mbarriers; producer `arrive.expect_tx` + `cp.async.bulk`,
  consumers `try_wait.parity`, and the **phase is toggled after every
  wait** (`phase ^= 1`). mbarriers are initialized ONCE per kernel, never
  re-initialized per use.
- Named barriers (`bar.sync id, count`) coordinate events across warp
  groups ("all consumers finished a stage").

### DeepEP intranode (closest to our workload)

`thirdparty/DeepEP/csrc/kernels/utils.cuh:313-346` and `intranode.cu`:

- `mbarrier_init(mbar, 1)` once; `fence.mbarrier_init.release.cluster`
  after init (`utils.cuh:317`); `mbarrier_wait(mbar, phase)` toggles
  `phase ^= 1` inside the wait (`utils.cuh:325-333`).
- `mbarrier_arrive_and_expect_tx(mbar, num_bytes)` then `tma_load_1d(...)`
  then `mbarrier_wait(mbar, tma_phase)` — one arrival (the expect_tx) per
  use, phase monotonically toggling across many chunks
  (`intranode.cu:406-408`).
- `bar.sync %0, %1` with **register barrier IDs and counts** for per-role
  warp-group sync (`intranode.cu:337, 393, 446`).
- `fence.proxy.async.shared::cta` before TMA stores
  (`utils.cuh:313`, `tma_store_fence`).

### ThunderKittens B200 warp-specialized GEMM (mxfp8)

`thirdparty/ThunderKittens/kernels/gemm/mxfp8_b200/mxfp8_b200_gemm.cu`:

- One warpgroup of producer warps, one warpgroup of consumers; **every
  warp inside the producer has a distinct role** (load A/B tiles, load
  scales, smem→tmem, MMA launch) — specialization is per-warp, not just
  per-group.
- mbarriers `init_semaphore(sem, thread_count=0, transaction_count=1)`
  ONCE, by a single thread (`threadIdx.x == 32`), then an
  `arrive_aligned()` cluster barrier before any producer/consumer uses
  them (`:100-117`).
- Two semaphore arrays per pipeline stage: `tiles_arrived[i]` (load done)
  and `inputs_finished[i]` (consumers done); producer waits `finished`
  before reloading a stage, consumers wait `arrived` before computing
  (`:138-225`).
- Phase bits: `get_phasebit<1>(phasebits, stage)` +
  `update_phasebit<1>(phasebits, stage)` toggled per use; `expect_bytes`
  is called by the waiting warp right before `wait` (`:180-186`).
- Consumer→producer handoff for the epilogue uses the same pattern:
  `warpgroup::sync(1)` then `warpgroup::tma::cluster::arrive(outputs_finished)`
  (`:236-246`), and the store epilogue is itself pipelined
  (`store_async_read_wait<N>`).

## Key lessons for our pure reduce

1. **mbarrier: init once per task, toggle the phase per use.** Re-init per
   chunk is only safe if every thread is `__syncthreads()`-fenced around it
   (the validated single-buffer path does exactly this). In a pipelined
   warp-spec kernel there is no such fence: a consumer may be parked in
   `try_wait.parity` on a stage while the producer re-inits that stage's
   barrier for the next chunk. The re-init resets the phase, so a wait can
   observe a completed phase it did not wait for (spurious wakeup / hang).
   Init-once + phase-toggle has no such window — DeepEP and ThunderKittens
   both run it across thousands of chunk iterations.
   (History: the earlier "hang at 512 chunks" was the `try_wait` PTX scope
   bug — `.cluster` before the `.cta` fix in `tma_ops.h` — not the toggle
   concept; the scope is now `acquire.cta.shared::cta`.)
2. **The producer is (at most) a few warps; the rest compute.** TMA issues
   are single-thread, so one producer lane can drive the whole pipeline.
3. **Named barriers** are the standard way to say "all consumer warps
   finished a stage" before one thread signals the producer's `done`.
4. A store must be fenced to the generic/async proxy before its stage is
   reused or the task completes (`fence.proxy.async.global` +
   `__threadfence()` before the task-complete ring write).
5. `cp.reduce.async.bulk` (global→shared reduce on TMA) is integer-only in
   current PTX — **not usable for f32 sum**. We must load src and dst into
   smem and reduce there, as now.

## Our design

256 threads = 8 warps (persistent kernel default):

| role | warps | work |
|---|---|---|
| producer | warp 0 | TMA bulk-load src+dst into free stages; bulk-store the reduced dst after consumers finish |
| consumers | warps 1-7 (224 threads) | wait stage data-ready, reduce in smem, signal stage done |

### smem layout (224KB budget, 4 stages)

```
stage s: [src: C][dst: C][full mbarrier][done mbarrier]
C = (smem - 4*2*32) / 8, aligned to 32   (≈28.6KB at 224KB)
```

All 8 mbarriers initialized once at task start by producer lane 0
(`fence.mbarrier_init.release`), visible to all threads via the existing
task-start block sync. Per-task re-init is safe because the multi-block
task barrier quiesces the previous task's use.

### mbarrier protocol (per task; init once, phase-toggle per use)

- `full[s]` (count=1): producer `arrive.expect_tx(bytes)` +
  `cp.async.bulk` src+dst into stage s → consumers
  `try_wait.parity(full[s], full_phase[s])`, then `full_phase[s] ^= 1`.
- `done[s]` (count=1): after ALL consumers reduce (named barrier
  `bar.sync 1, 224`), warp-1 lane-0 `mbarrier.arrive(done[s])` →
  producer `try_wait.parity(done[s], done_phase[s])`, then
  `done_phase[s] ^= 1`.

### Producer loop (warp 0, lane 0 only)

```
prefill stages 0..3 with chunks 0..3: arrive.expect_tx(full[s]) + bulk-load
for c in 0..nfull-1:
  s = c % kNSlots
  try_wait(done[s], done_phase[s]); done_phase[s] ^= 1   // consumers done
  bulk-store stage-s dst -> gmem chunk c; wait_group<0>; fence.proxy.async.global
  if c+kNSlots < nfull:
    arrive.expect_tx(full[s]) + bulk-load chunk c+kNSlots into stage s
```

### Consumer loop (warps 1..7)

```
for c in 0..nfull-1:
  s = c % kNSlots
  try_wait(full[s], full_phase[s]); full_phase[s] ^= 1    // data ready
  reduce chunk c: 224 threads split the stage's elements (16B vector,
  stride nconsumer)
  bar.sync 1, 224                                          // all consumers done
  if warp==1 && lane==0: mbarrier.arrive(done[s])
```

### Tail

Remainder (< C bytes) uses the validated ILP vector path on ALL threads
(producer + consumers) after the pipeline drains — the tail-chunk TMA bug
already taught us to keep odd sizes off the bulk path.

## Why this beats the previous attempts

- **The two-slot double-buffer attempt** re-initialized mbarriers per
  chunk/load — against the canonical init-once + phase-toggle protocol
  (and had a slot-layout mystery).
- **The first warp-spec attempt** re-inited per load with parity 0 always,
  plus a per-warp `done` over-arrive. This design inits once, toggles
  phases, and signals `done` from exactly one consumer thread behind a
  named barrier.
- load[N+1] and store[N-1] overlap reduce[N] by construction, so the
  per-chunk critical path tends to `max(load, reduce, store)` instead of
  their sum.

## Open questions / experiments

1. Is the per-SM cap the TMA engine's issue rate rather than the
   load/reduce/store serialization? If so, warp-spec plateaus below 2x —
   the experiment answers this. Fallback: keep the validated
   single-buffered TMA (224KB, BLK=32 ≈ 490 GB/s).
2. Consumer ratio: 7 consumer warps means only 28.6KB chunks at 224KB
   smem; try `kNSlots=2/3` for larger chunks, or 2 producer warps (loads
   and stores separated, ThunderKittens-style) so store latency never
   blocks the load prefetch.
3. Measure BLK=8/16/32: the win condition is BLK=16 ≈ 490+ GB/s (fewer SMs
   than today's BLK=32 single-buffer).

## Experiment results (B300, 2026-08-04)

Implementation status: canonical protocol shipped and validated
(init-once + derived parity `(c/kNSlots)&1`, `bar.sync 1, 224` before a
single warp1-lane0 `cdone` arrive, `ready` count=2 for the two bulk loads,
tail via ILP). All completed runs were `wrong=0`.

### 256M allreduce, 2 ranks, via shim (4MB tiles — adaptive_tile_bytes
splits 256MB into 64 tiles)

| config | BLK=32 | BLK=16 | notes |
|---|---|---|---|
| single-buffer TMA 224KB (baseline) | ~443 GB/s | ~419 GB/s | validated earlier |
| warp-spec 4 slots (28.6KB chunks) | 178 GB/s | — | per-block task = 4 chunks, pipeline barely fills |
| warp-spec 2 slots (56KB chunks) | 255 GB/s | 165-200 GB/s | per-block task = 2-4 chunks |
| native NCCL 2.29.7 | 510 GB/s | — | 32 coll channels |

### Device bench, full pipeline depth (per block 8MB → nfull = 146-292)

| config | 256M @ 32 blocks |
|---|---|
| single-buffer TMA 224KB | 544us → 493 GB/s (15.4 GB/s/block) |
| warp-spec 4 slots | 574us → 446 GB/s (14.0 GB/s/block) |
| warp-spec 2 slots (debug build, 128M) | 299us → 428 GB/s (13.4 GB/s/block) |

### Findings

1. **Warp-spec is ~10% SLOWER per block than single-buffer at full depth.**
   The pipeline overlap (load[N+1]/store[N-1] vs reduce[N]) did not pay:
   total in-flight bytes per block are identical (224KB smem cap) — the
   4x-smaller chunks only add per-chunk mbarrier/barrier overhead.
2. **The shim's 4MB tiles make it worse:** at BLK=32 each block gets
   128KB/task → nfull=2-4, so the pipeline never fills and per-task fixed
   costs dominate. Larger tiles (UK_CCL_LARGE_TILES=8 → 32MB) would fill
   the pipeline but the full-depth device-bench numbers above say it still
   won't beat single-buffer.
3. **Per-SM ceiling ≈ 14-15 GB/s payload (42-45 GB/s memory traffic) on
   this workload is a memory/TMA-system property, not pipeline structure.**
   Fewer-SMs-at-full-rate needs more in-flight bytes per SM: bigger tiles
   (shim), multicast/NVLS, or cluster-shared smem (2-CTA DSMEM doubles the
   per-SM budget). 224KB/block is the single-CTA smem cap.
4. Intermittent wedge: non-debug 2-slot builds hang/crash at deep pipelines
   (nfull >= 9) while the identical code with `UK_WARPSPEC_DEBUG=1` prints
   completes. Suspect a full-speed hardware-protocol race (mbarrier or
   named-barrier timing), NOT a logical deadlock — needs a controlled
   debugger session to pin down. Given finding #1, low priority unless we
   pursue warp-spec further.

### Recommendation

Keep single-buffer TMA (224KB, BLK=32) as the default reduce kernel. The
fewer-SMs goal is better served by (a) larger shim tiles, (b) REDUCE_ILP
register path at high ILP, (c) multicast. Warp-spec stays as a documented
WIP with the canonical protocol as a reference for any future
cluster/multicast work.
