# uk-300 vs upstream — change review pack

Branch `uk-300` is the ukernel NCCL-compatible communication-shim /
spray-executor development line. This pack summarizes everything it
changed relative to upstream `origin/main` so it can be reviewed
without switching branches.

## Scope

- Base (merge-base with `origin/main`): `c93bb14f`
  `[UK] CCL engine with fused PutSignal and Python binding overhaul (#1031)`
- uk-300 is **98 commits ahead** of that base.
- Diff: **124 files changed, +17,710 / −5,223** (full patch:
  [`uk300-vs-upstream.patch`](uk300-vs-upstream.patch), ~28k lines).

`origin/main` has **6 commits not in uk-300** (upstream-only, unrelated
areas: P2P/HIP/CI/EP):

```
f071f2e3 [UK] CCL engine with fused PutSignal and Python binding overhaul (#1031)
a7176a99 [CI] Set NCCL_SOCKET_IFNAME for GH200 EP latency test (#1034)
3e06114f [P2P] Add Cambricon MLU (CNRT) support for cross-node P2P RDMA (#1011)
61ee4240 [EP]: Fall back when verbs atomics are unsupported (#1019)
6d872152 [CI] bring up gb10 (#1029)
d56e3492 [P2P] Fix HIP pointer attributes for ROCm and DTK (#1035)
```

## Diff stat (by area)

```
src/ccl/       20 files   (shim, coll_algo engine, executor, backends, tests)
src/device/    10 files   (persistent worker, reduce/copy kernels, ops)
src/transport/  5 files   (IPC/RDMA adapters, communicator signal matching)
bench/          4 files   (alltoall_perf, shim sweep, CE contention, device reduce)
docs/          10 files   (perf/design/comparison docs)
include/        4 files   (nccl.h ABI header, gpu_rt, util headers)
build/          3 files   (arch_defaults.mk, Makefile knobs, python binding)
```

## What changed, by subsystem

### 1. NCCL-compatible drop-in shim (`src/ccl/nccl.cc`, `include/nccl.h`)

- `libnccl.so.2` ABI-compatible shim (LD_PRELOAD / LD_LIBRARY_PATH swap)
  implementing `ncclAllReduce`, `ncclAllGather`, `ncclReduceScatter`,
  `ncclAllToAll` (extension), comm init/destroy, stream semantics.
- `ncclBarrier` extension added then **removed** (non-standard API; kept
  out to stay native-compatible).
- Async NCCL semantics: enqueue is non-blocking, completion via
  stream-ordered deps + host polling.

### 2. CCL engine (`coll_algo.cc/h`, `executor.cc/h`, `lower.cc`)

- Ring + binary-tree plan builder with per-op tags, tiling
  (`UK_CCL_LARGE_TILES`), fused PutSignal, signal aggregation
  (`UK_CCL_SIG_GROUP_TILES`), stream-ordered dependencies.
- Spray executor: enqueue/drain threads per backend, reserve-slot
  submission, adaptive put-path selection (IPC / device / RDMA),
  watchdog + run state dump (SIGUSR2).
- AllToAll: out-of-place pure-IPC path (no staging), in-place staged
  path (data-race fix), self-slice via CE on the user stream.
- Fusions: reduce+copy (RS), AG copy with inline device completion flag,
  CE+device hybrid (`UK_CCL_A2A_HYBRID`, `UK_CCL_A2A_HYBRID_CE_PCT`;
  RS hybrid tried and reverted as negative).
- Hybrid hardening this series: send side now honors `CE_PCT` (was
  hardcoded 50/50), boundary pct falls back to plain path, splits are
  16B-aligned (copy-kernel vector width).

### 3. Device kernels (`src/device/`)

- Persistent worker pool (single/multi-block) with idle-exit grace and
  async relaunch; leader-free best-effort multi-block worker.
- Reduce: ILP-vectorized (`REDUCE_ILP`), TMA bulk/chunk reduce
  (opt-in, smem mbarrier), per-block chunk rounding to the 16B vector
  width (fixes misaligned-start hangs on awkward task sizes).
- Copy op: vectorized LD/ST only (TMA to peer-mapped memory removed —
  hangs on B300; local TMA also removed as it gained nothing).
- `VALIDATE=1` fast build; per-GPU arch/ILP/TMA/smem auto-selection.

### 4. Transport (`src/transport/`)

- IPC adapter: per-peer send pipeline + window, `cudaMemcpyAsync` over
  the IPC handle (not `cudaMemcpyPeerAsync`), event-based completion,
  eager poll bursts, fused-signal completion ordering.
- RDMA: DMA-BUF/GDR registration with IPC fallback, signaled
  last-chunk + message-granularity credit, lock-free backpressure,
  salted-tag fused put+signal.
- Communicator: signal matching now serialized by a single
  `sig_maps_mu_` (per-peer locks raced on shared `unordered_map`s —
  intermittent SIGSEGV at 8 ranks), batched matching, cheap
  pending-wait check, review-driven hardening (ring claims, spin
  hygiene, close completion).

### 5. Benchmarks (`bench/`)

- `alltoall_perf` (direct `ncclAllToAll`, N-rank, verify + fill
  handshake), `bench_shim_param_sweep.sh`, `bench_device_reduce_blocks.sh`,
  `ce_contention.cu` (standalone CE contention microbench).

### 6. Docs (`docs/`)

- `perf_test_procedure.md`, `reduce_kernel.md`, `put_path_selection.md`,
  `nccl_compatibility.md`, `alltoall_comparison.md`, `ce_contention.md`,
  `fused_rs_reduce.md`, `optimization_framework.md`,
  `b300_native_nccl_measurements.md`, `benchmarks.md` — all English,
  with B300 2/4/8-rank shim-vs-native sweeps and mechanism analyses.

## Key results recorded in the docs

- AllReduce 256M: shim ~93–98% of native at optimal config
  (`LT=16 TM=8M IB=16 BLK=64`); pure-reduce kernel reaches native
  bandwidth at 16 SMs (604 GB/s) after ILP/register fixes, TMA variant
  at 32 SMs (512.8 GB/s) validated at 128M.
- AllToAll 256M hybrid: 4 ranks best `pct=50 blk=32` → 512.7 GB/s
  (81% native 632); 8 ranks best `pct=30 blk=32` → 402.8 GB/s
  (65% native 620). CE copy-engine contention grows with ranks, so more
  device-worker share wins at 8 ranks.

## Commit list (98, oldest → newest)

```text
9383303b [UK] CCL engine with fused PutSignal and Python binding overhaul (#1031)
b32325ad build: arch/TMA/ILP make knobs, conda linker fix, nvcc/GDR auto-detect
4d642f75 transport: pipelined IPC put window, RDMA DMA-BUF/GDR registration
ed901f4f device: ILP-vectorized reduce, TMA bulk/chunk reduce, idle-exit spin fix
6cea98cc ccl: coll_algo engine (ring + binary-tree, fused PutSignal), executor rework
67f30eee nccl: NCCL-compatible drop-in shim (nccl.h ABI + nccl.cc async semantics)
80f36475 ccl/test: unit + integration coverage for coll_algo, spray executor, e2e
5bd63428 bench: alltoall_perf harness + shim param sweep + device reduce scripts
8a35b99b docs: ukernel README, B300 perf/design notes, NCCL compatibility
bd2fbff4 bench: alltoall_perf N-rank support (--nranks, verify, id-file cleanup)
e116c7c2 bench: portable fill handshake for N-rank alltoall verify
cba91fdc nccl: drop ncclBarrier extension (non-standard API)
81f0e04b ccl/lower: alltoall in-place staging correctness
dbcdfa4f docs: 2/4/8-rank shim vs native AllReduce + AllToAll sweep (B300)
53094c48 nccl: AllToAll out-of-place support — no staging, pure IPC puts
8f7188b3 nccl: AllToAll self-slice via CE on the user stream
b7ead377 docs: AllToAll out-of-place results (312/544/706us at 2/4/8r)
6d94ae15 transport/ipc: per-peer send pipeline
8b6abbff transport/ipc: cut fused-signal completion tail
783ebb1e transport/ipc: cudaMemcpyAsync over the IPC handle
132be5e5 executor/transport: busy-poll signal drain while waits pending
bffb9b98 docs: per-peer IPC pipeline + copy-engine ceiling analysis
665a7884 executor: condition-variable wakeup for enqueue_loop
6534318f docs: dispatch-trim — copy span is the wall
08711847 nccl: UK_CCL_SIG_GROUP_TILES env (signal aggregation)
8dece67b docs: signal aggregation + chunk interleaving sweep
03a8d0f8 transport/ipc: drop dead peer_gpu_idx lookup + launch debug print
43887ef6 device: peer-dst copies bypass TMA (plain vectorized LD/ST)
1ad7a17f docs: device-backend puts beat CE at 4/8 ranks
50fdff55 device: remove TMA from the copy op entirely
34f00368 docs: copy op vectorized LD/ST only; device-path alltoall holds
41c1b14b docs: full-size sweep (1M..256M) shim vs native at 2/4/8 ranks
83fe7ac0 docs: consolidate — rewrite alltoall_comparison as current-state doc
460f5905 executor: host-side orchestration profiler (UK_CCL_HOST_PROF=1)
ee1c4d28 bench: standalone CE contention microbenchmark + B300 results
e28b82ee docs: CE contention mechanism analysis (ce_contention.md)
3d23c840 ccl: fused reduce+copy for allreduce RS (UK_CCL_FUSE_REDUCE_COPY)
be0bf133 docs: fused reduce+copy device-flag results (4r +7.5%, 8r +18.8%)
1457eedc ccl: fused AG copy — device copy task + inline completion flag
43496449 build: VALIDATE=1 — fast validation build
0e1e1755 fix: lower Put ordering — push before emit_group_signal
3aa5db27 docs: fused AG copy results + build-speed note
f5327b80 bench: ce_contention --batch mode + verification docs
c86563e1 bench: merge benchmarks/ into bench/
3ea85c99 docs: full English cleanup
df09d605 docs: correct busy-poll drain status in optimization framework
5a784433 transport: review fixes — clean close_comp, async plain signals
bc72064c transport: harden all reviewed problem areas
6cd70460 transport: fix review findings — ring claim, put-cache kind check
32dd8792 transport: fix RDMA slot force-write, size signal ring above cap
9e97585c transport: lock-free-read put cache via fixed-size seqlock
8ebe8c3d transport: drop tag-value imm heuristic for RDMA signal waits
3a9fd058 transport: lock-free RDMA backpressure
ae4e92df transport: RDMA signaled last-chunk + message-granularity credit
e0fc6ca8 transport: resolve same-host RDMA MR best-effort
3dc3e6d8 transport: accept salted tags in RDMA fused put+signal
87ad98ab device: gate CollPut fused-signal on HostNativeAtomicSupported
45e55c60 device: leader-free best-effort multi-block worker
b222cce7 docs: reduce per-SM efficiency measurements
76a03a94 build: auto-adapt arch and reduce-kernel knobs to the GPU
7bed72a7 build: rebuild objects when kernel knobs change
4fe64534 build: perf TMA builds use ILP=4 (compile-time fix)
8423e972 docs: leader-free kernel TMA validation status
b30ac88b device: review fixes — burst dispatch, async relaunch
6d690594 docs: clean-B300 TMA results — 32-SM allreduce parity (512.8 GB/s)
9be4479b docs: minimum SMs to feed native bandwidth with pure reduce
d6b5e4f4 device: gate the fast reduce set to fix ILP=16 compile time
07d450c9 device: read fifo tail with device scope in the multi-block worker
1372e7c5 docs: new-kernel pure-reduce measurement (64 SMs = 471 GB/s)
b577690b docs: why NCCL's 667 GB/s reduce is NVLink-fed; register-spill finding
880d4892 device: isolate reduce dispatch in its own TU (register-spill fix)
a953c65b docs: register-spill fix — 16 SMs feed native pure reduce (604 GB/s)
f7628d27 build: TMA opt-in until the RDC+TMA interaction is fixed
72d8be1e device: cap TMA chunk size at 32KB (cp.async.bulk truncation)
4b77ab4a docs: put+reduce pipeline overlap — allreduce 93-98% native
035abe2f ccl: add UK_CCL_RS_CHUNKS — split RS tiles into K chunks
13064da4 docs: RS chunking experiment — negative
30823daa ccl: add UK_CCL_OP_TRACE per-op completion latency diagnostic
02875dbc ccl: fix trace_op placement
5e27a610 ccl: RS CE+device hybrid (UK_CCL_RS_HYBRID)
3d7deaa8 docs: RS CE+device hybrid — negative at 4/8 ranks
f599c4d1 ccl: AllToAll CE+device hybrid (UK_CCL_A2A_HYBRID)
960d36d1 docs: AllToAll CE+device hybrid — positive at 4 ranks (+19%)
2ddc357d ccl: AllToAll hybrid CE fraction knob (UK_CCL_A2A_HYBRID_CE_PCT)
b6f4ba9b docs: clean AllToAll hybrid A/B
57765607 ccl: lazy device worker with warm-up launch (reverted later)
da437009 ccl: warm up worker with single block (reverted later)
4c3c4d6f ccl: keep warm-up worker resident (reverted later)
db0b3362 ccl: AllToAll hybrid send side honors CE pct (fix 50/50 hardcode)
8509f9a6 ccl: AllToAll hybrid — plain path at pct boundaries
5517e9c0 ccl: dev-stall dump includes worker bound state (debug)
43749168 ccl: revert lazy worker lifecycle to pre-created workers
bfa617a3 docs: correct AllToAll hybrid findings
e5ee021f ccl: align AllToAll hybrid split to element size
f3d16ec2 ccl: AllToAll hybrid split aligned to 16B copy vector width
9bc671c4 device: align per-block copy chunks to the 16B vector width
8a4b0855 transport: serialize signal-accounting maps with one mutex
0fe68650 bench: drop temporary crash-backtrace handler from alltoall_perf
```

## How to review

- Patch-at-once: `git apply review/uk300-vs-upstream.patch` on the base
  `c93bb14f`.
- Commit-by-commit: `git log c93bb14f..uk-300 --oneline` (subjects are
  grouped by area; the `docs:` commits carry the measurements and
  verdicts that justify each code change).
- Area review: `git diff c93bb14f uk-300 -- experimental/ukernel/src/transport`
  (etc.) for a per-subsystem view.

Note: the lazy-worker commits (`57765607`–`4c3c4d6f`) were reverted by
`43749168`; the revert is intentional (the lazily created multi-block
worker stalled on B300) and the current code base is the pre-created
worker lifecycle.
