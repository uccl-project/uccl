# AllToAll comparison plan (vs user-space MoE implementations)

## Decision

Do **not** benchmark AllToAll against native NCCL: NCCL has no dedicated
AllToAll primitive (only `ncclSend`/`ncclRecv` groups, which is how
engines and nccl-tests build it themselves). Training/inference engines
implement AllToAll (e.g., MoE dispatch/combine) on their own, so our
AllToAll competes with **their user-space implementations**, not with a
NCCL primitive.

## Our current position

- Native ukernel AllToAll (`CollKind::AllToAllPairwise`), measured via
  `test_perf_spray_allreduce --kind=alltoall`: 256MB 2.88ms (93 GB/s,
  8 blocks) on the A40 pair.
- The shim's `ncclAllToAll` is the same algorithm but is not reachable
  through nccl-tests (`alltoall_perf` uses `ncclSend`/`ncclRecv`, which
  the shim does not implement).

## Reference: `thirdparty/DeepEP`

The repo carries DeepEP (MoE dispatch/combine all-to-all):

- normal kernels: NVLink intranode, RDMA internode, asymmetric
  NVLink→RDMA domain forwarding;
- low-latency kernels;
- **SM number control** (matches our "fewer SMs, comparable
  performance" goal);
- FP8/BF16 support; DeepSeek-V3/R1 production shapes (4096 tokens ×
  7168 hidden, top-4/top-8 experts, FP8 dispatch + BF16 combine).

## Comparison plan

1. **Define common shapes** from DeepSeek-V3 production settings:
   prefill 4096×7168, decode 128×7168, top-4/top-8 experts; also a
   general pairwise all-to-all shape (per-expert token groups).
2. **Map our AllToAll to dispatch/combine semantics**: each rank holds
   a set of experts; tokens are grouped by destination expert and
   exchanged pairwise (dispatch), then results returned (combine).
3. **Harness**: same GPUs, same shapes, same iteration counts, same
   stream-sync timing; report latency and throughput per shape. Our
   side runs at the executor level (spray) or a small harness — not
   nccl-tests, since the shim lacks Send/Recv.
4. **Also compare SM usage** at matched performance (DeepEP exposes SM
   count; we sweep `UK_CCL_DEV_BLOCKS`).
5. **Known gaps to close before a fair run**:
   - FP8 dispatch: our device reduce path does not yet support Fp8;
   - internode AllToAll: our AllToAll is validated same-node only
     (internode would go through the RDMA adapter).

Open question: whether to compare at the library API level (our
pairwise AllToAll vs DeepEP's dispatch/combine on the same token-expert
layout) or at the kernel level (our spray AllToAll vs DeepEP kernels on
equivalent buffers).

## 2026-08-04: B300 shim alltoall data-integrity bug (found while benchmarking)

Added `bench/alltoall_perf.cu` (2-rank ncclAllToAll harness; shim uses
the ncclAllToAll extension, native builds the exchange from
ncclSend/ncclRecv). Verification (fill each rank's buffer with
rank-specific values, exchange, check the peer partition) exposes a
**data-integrity bug in the shim's alltoall**:

- rank0 -> rank1 direction is correct (rank1's partition 0 holds rank0's
  values).
- rank1 -> rank0 direction is broken: rank0's partition 1 is only
  partially/never written (observed both `peer[0..3]=2,2,2,2` with a bad
  element later, and `bad[0]=1 want=2` across runs — partial write), even
  after `cudaDeviceSynchronize()`.

Debug trail:
- Both ranks enqueue all 8 x 1MB puts to the IPC path (UK_CCL_DEBUG=2
  shows `[enqueue r1] op... put_path=1` and `[ipc-send_one r1]
  dst=... src=... peer=0`, dst increments 1MB as expected). No REJECT
  bounds failures.
- nsys shows the P2P memcpy stream is active (50 x 128MiB at 256MB/LT=1),
  so puts are launched.
- Events: per-put gpuEvent tracking lives in the transport layer
  (`transport/adapter/ipc_adapter.cc:440-480`); the collective
  completion signal lives in the executor (`ccl/executor.cc:380`,
  event pool + WaitValue gate). The two are not strictly serialized in
  the shim's ncclAllToAll path — the executor may signal completion
  before the peer's put data is visible, or the dst resolution
  (`try_resolve_remote_ipc_pointer` in `transport/communicator.cc`) maps
  the offset wrong for one direction.

**Status: perf comparison vs native is BLOCKED until this is fixed** —
the earlier "shim 258us vs native 404us" numbers assumed correct data and
are not trustworthy.

## 2026-08-04 (2): false alarm — it was the harness's fill/put race

Root cause of the "data-integrity bug": **the benchmark's verify fill
races the peer's one-sided IPC puts**, not a shim bug. IPC puts are
one-sided writes into the peer's buffer; if rank0's put lands before
rank1's verify-fill (same buffer), rank1's fill overwrites the exchanged
data and the check fails. The failing direction was random across runs,
which is the signature of a fill/put ordering race.

Confirmed with a minimal harness: raw `cudaIpcOpenMemHandle` +
`cudaMemcpyPeerAsync` writes land fully (128B..8MB all verified) — the
P2P mechanism is fine (an earlier "only 128B lands" was my own
`1UL<<20/4` precedence bug in the mini program).

Fix in `bench/alltoall_perf.cu`: file-based both-fills-done handshake
before the exchange (shim's `ncclBarrier` currently crashes with a GPU
error — separate shim bug, recorded below). After the fix:

### Result: shim alltoall is CORRECT and FAST

256MB alltoall, 2 ranks, GPU 6,7:

| config | avg | note |
|---|---|---|
| **shim BLK=1 LT=1 (single 128MB IPC put per rank)** | **~296us** | verify OK x3, busbw ~900 GB/s |
| shim BLK=1 LT=8 | ~330us | |
| shim default (LT=64, 1MB tiles) | ~585us | per-tile overhead |
| native NCCL (nccl-tests alltoall_perf) | 404.7us | in-place, ncclSend/Recv ring |

Shim (BLK=1, LT=1) is ~27% faster than native. Key insight: alltoall is
pure DMA put — **BLK=1 is best** (no reduce kernel / persistent worker
interference; the executor still launches a single-block worker that
idles), and **LT=1 makes the whole exchange one 128MB P2P put**, beating
native's tiled multi-channel path at this size.

### Separate shim bug found: ncclBarrier crashes

`ncclBarrier` (4-byte AllReduceRing) triggers "GPU error
backend/device_backend.cc:326: driver shutting down" on B300. The
harness works around it with a file handshake; the barrier path needs
investigation (likely the 4-byte ring put/reduce path).

## 2026-08-05: ncclBarrier fixed — root cause was buffer resolve, not TMA

`ncclBarrier` (and all sub-4KB allreduces: 1K/4K/16K) failed in
`Communicator::resolve_remote_buffer`: same-host transfers only need IPC
(rkey=0), but the code waited for the peer's MR first and returned
false on timeout — the same-host shortcut below was unreachable. MR
registration fails for small allocations, so every small collective
timed out. Fix: same-host requires `wait_ipc` only
(`4a61ad3d`). 1K..1M allreduces and `ncclBarrier` now pass.

While debugging, two genuine device-path bugs were also fixed (both
would hang/crash small TMA transfers even after the resolve fix):
- `%16` + 16B-alignment gates on the <=4KB TMA small paths (4-byte
  cp.async.bulk is illegal) (`fc168d48`).
- The small paths passed a **stack** `TmaSemaphore` to mbarrier ops —
  not a valid shared-memory mbarrier address. The mbarrier is now carved
  out of smem after the payload, and the store waits its bulk group
  (`5a0a4691`). `ncclBarrier` payload also moved 4B -> 256B (`f3e92df1`,
  harmless hardening).

### Worker-laziness reverted

Creating the persistent worker only on first device-task enqueue
deadlocked the shim's allreduce path (no output, GPU idle) and was
reverted (`89ce7108`). Pure-put alltoall still launches the idle
single-block worker; deferring it safely needs an executor-level trigger
that is understood (the enqueue-time lazy create blocked in
ensure_runtime's waitWorker inside the executor's enqueue thread).

## 2026-08-05 (2): lazy worker landed — submit-time ensure, workerless alltoall

Revisited after the resolve fix. The earlier allreduce "deadlock" was
actually the same-host buffer-resolve hang (every small collective
stalled), not the lazy create itself — with `4a61ad3d` in place the
safe design works (`199d520b`):

- `DeviceBackend` no longer creates the worker pool in its ctor.
- `submit()` (user thread, under api_mu_) scans the plan for
  `ExecOpKind::Reduce` and calls the new `BatchBackend::ensure_worker()`
  hook before publishing the run. Never inside `enqueue_loop`, whose
  blocking was the original deadlock hazard.
- `do_enqueue_reserved_batch` keeps a fallback `ensure_runtime()` for
  device puts routed at runtime (`UK_CCL_PUT_PATH=device`).
- `do_drain` returns 0 early when no worker was ever created.

Verified on B300:
- Pure alltoall (skip-verify): nsys reports **no CUDA kernel data** —
  the persistent worker is not launched for IPC-only collectives.
- alltoall 256MB: 281-288us, busbw ~930-953 GB/s (verify OK).
- allreduce 256MB: 461 oop / 420 ip (regression clean).
- ncclBarrier verify OK; 1K..1M allreduces pass.

Benefit is mostly SM/GPU-resource savings (the idle single-block worker
no longer occupies an SM); performance is unchanged within noise.

## 2026-08-05 (late) — in-place AllToAll data race found and fixed;
2/4/8-rank shim vs native

Extending `bench/alltoall_perf.cu` to N ranks (`--nranks`, N-rank
verify with per-partition values, separate send/recv buffers on the
native path, portable fill handshake) exposed a **real correctness bug
in the shim's in-place AllToAll**: partition p is simultaneously the
source of my Put to peer p and the target of peer p's Put into my
buffer, so an unstaged exchange reads data the peer's incoming copy
already overwrote. The old "Equal-split offsets never overlap, so no
staging" shortcut was wrong — 2/4/8-rank verify caught receivers
getting their own data back (the earlier 2-rank "verify OK" runs passed
only by timing luck).

Three bugs were fixed in the staging path (all now covered by the N-rank
harness verify):

1. **No staging for equal splits** — every in-place AllToAll send now
   stages Input->Scratch before the Put (`coll_algo.cc`).
2. **copies-done rendezvous deadlock** — the staged Put waited for the
   peer's "copies done" using the local chunk's pair id; the peer
   signals from ITS send chunk whose pair is `dst_rank*nranks+rank`, so
   the tags never matched and both sides waited forever (watchdog
   `done=2/6`). The wait now keys on the peer's send pair (`lower.cc`).
3. **Staging regions overlapped** — every chunk staged at scratch
   offset 0 (later chunks clobbered earlier ones) and the Put source
   lacked the chunk base (all Puts read chunk 0's data). Each chunk now
   owns a disjoint scratch region and the Put reads its own region
   (`lower.cc`).

### 256MB AllToAll, 2/4/8 ranks (all wrong=0)

Shim: out-of-place (`sendbuff != recvbuff`), `UK_CCL_LARGE_TILES=1`.
The self-slice is a cudaMemcpyAsync on the user stream and every peer
exchange is an IPC put, so **BLK is irrelevant** (BLK=1 == BLK=64 within
noise) and the persistent worker never moves AllToAll data. Native:
nccl-tests `alltoall_perf` (also out-of-place).

| ranks | shim (us) | shim algbw | shim busbw | native (us) | native algbw | native busbw |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 312 | 861 | 431 | 415.6 | 645.8 | 322.9 |
| 4 | 544 | 493 | 370 | 424.7 | 632.1 | 474.0 |
| 8 | 706 | 380 | 333 | 432.6 | 620.5 | 543.0 |

The shim wins at 2 ranks (one big IPC put per direction on NVLink) and
trails at 4/8: more peers means more per-peer puts, and the IPC send
window/launch overhead per put does not pipeline as well as native's
multi-channel path. The gap is now purely the IPC put path scaling —
the next lever is put-window / launch pipelining, not staging or SM
blocks.

## 2026-08-05 (late, 2) — per-peer send pipeline + copy-engine ceiling

### Send path restructure (UK_CCL_IPC_BATCH semantics changed)

The IPC send worker previously shared ONE global in-flight window
(`UK_CCL_IPC_BATCH=4`) and one 4-stream pool across all peers, with a
global FIFO completion barrier. It now gives **each peer its own send
ring, in-flight window and stream pool** (`UK_CCL_IPC_STREAMS_PER_PEER`,
default 4), and completes each peer FIFO (receivers match per-peer
sequences) while different peers complete out of order. This is the
right shape for multi-node scaling: N peers run N × window copies
concurrently instead of one global window.

### Copy-engine ceiling (raw cudaMemcpyAsync, no shim)

A raw alltoall microbenchmark (7 × cudaMemcpyAsync per rank on per-peer
streams, 256MB total) measures the hardware ceiling for our exact access
pattern: **191 / 297 / 368us at 2/4/8 ranks** (701 / 678 / 610 GB/s
outbound per rank). Two shim-side findings from chasing this:

- `cudaMemcpyPeerAsync` was ~450 GB/s per P2P copy; plain
  `cudaMemcpyAsync` over the IPC handle reaches ~670 GB/s (now used).
- The fused-PutSignal completion published after the shm-ring write,
  which can back-pressure on the receiver's drain cadence; the put
  completion now publishes first (the receiver completes when IT drains
  the signal), and the signal drain loop busy-polls while waits are
  registered.

### Where the remaining 8-rank time goes (one-shot [chain] timestamps)

submit → enqueued ≈ 190us (host dispatch of 14 ops: 7 fused puts + 7
waits, ~10-15us/op through communicator locks/resolve), copies span
~460-660us (per-copy p50 83us under 8-rank fabric contention vs 52us
raw), finalize → gate ≈ 15us. So the copy path itself already beats
native (raw 368us < native 433us); the shim's gap is host dispatch cost
plus the copy span being ~1.6x raw per copy under full-load contention.

Next targets, in order: batch/cache the per-put dispatch path
(resolved remote pointers + fewer locks), and keep the copy engine fed
back-to-back (launch all puts in one tight pass before polling
completions).

### Follow-up (same day): dispatch trimmed, copy span is the wall

The enqueue_loop now sleeps on a condition variable that `submit()`
notifies (it previously yielded to the scheduler between back-to-back
collectives), and the per-cycle dispatch instrumentation was gated to
only fire on cycles with real work. Final 256MB numbers (BLK=1,
LT=1): **~310 / ~580 / ~715us at 2/4/8 ranks** — the dispatch changes
do not move the end-to-end, because the 8-rank time is dominated by the
GPU-side copy span (~600us vs the raw 368us ceiling), not the host.
Under 8-rank full-load the 7 x 32M copies serialize on the copy engine
at ~1.6x the raw per-copy duration; LT=2/4 and more per-peer streams do
not help. The next lever is chunk interleaving with signal aggregation
(one signal per G tiles, cutting the per-tile completion cost that made
LT=8 regress), which is the same aggregation machinery AllReduce needs.

### Signal aggregation + chunk interleaving (UK_CCL_SIG_GROUP_TILES)

The aggregation machinery (one Signal/WaitSignal per `G` tiles of a
chunk pair, counted arrivals on the receiver) already existed in the
lowerer with default G=1; it is now tunable via
`UK_CCL_SIG_GROUP_TILES`. Sweeps (all wrong=0):

AllReduce 256M (LT=8 TM=8M BLK=64), OOP:

| G | 2r (us) | 4r (us) | 8r (us) |
|---:|---:|---:|---:|
| 1 | 610 | 1094 | 2201 |
| 2 | **575** | 1266 | 2149 |
| 4 | 653 | 1263 | **2114** |
| 8 | 654 | 1197 | 2252 |
| 16 | 671 | 1231 | 2177 |

Best is G=2 at 2 ranks and G=4 at 8 ranks (~4-6%); G>=4 regresses
2-rank (coarser group boundaries starve the short 2-rank pipeline — the
same reason the earlier G=8 default was reverted). No single default is
best everywhere, so the env override stays opt-in.

AllToAll 256M at 8 ranks (BLK=1, out-of-place):

| LT | G | avg (us) |
|---:|---:|---:|
| 1 | 1 | 773-784 |
| 2 | 2 | 765 |
| 4 | 4 | **714-723** |
| 4 | 8 | 716 |
| 8 | 8 | 742-751 |

LT=4 (4MB tiles) + G=4 gives the best 8-rank alltoall so far
(~715us, ~7% over the LT=1 baseline): smaller chunks interleave on the
copy engine better than 32MB monoliths, and G=4 amortizes the per-tile
signal/completion cost that made plain LT=8 regress. Still 1.9x the raw
CE ceiling (368us) — the copy span under 8-rank full load remains the
wall.

### Why the shim cannot beat native at 8 ranks with copy-engine IPC

The raw 368us ceiling is measured with ranks running **unsynchronized**
loops (they drift apart, so the fabric never sees all 56 copies peak
simultaneously). The shim's collective is synchronized by construction
(every rank waits for all 7 signals every iteration), so all 8 ranks
hammer the fabric in lockstep: per-copy duration under that load is
~83us (vs 52us raw), and the copy span is 460-660us. Native's 433us is
also synchronized, but it moves data with SM kernels that schedule
better under contention. So with the IPC/copy-engine-only constraint,
8-rank alltoall is CE-contention-bound (~600-715us) and cannot reach
native's 433us — the copy engine wins at 2 ranks (no contention) and
loses at 8.

To actually beat native at 8 ranks the data movement must leave the
copy engine: TMA bulk puts (`cp.async.bulk` to a peer-mapped address)
use a dedicated engine — not SM compute, so the "no SM copy" preference
still holds — and would be the next experiment.



### Device-backend puts (vectorized LD/ST) beat the copy engine at 4/8 ranks

NCCL's own intra-node data movement is per-thread vectorized LD/ST to
peer memory (LL protocol: `ld.volatile.global.b64`/`st.volatile.global.b64`;
symmetric kernels: 16B-packed load + `stcs`) — `cp.async.bulk` is only a
TODO comment in their code. Our device copy op already has the same
plain vectorized loop; it now takes that path for peer destinations
(`dst_rank >= 0` skips the local-only TMA fast path, which hangs on
peer-mapped addresses). Forcing alltoall puts through it with
`UK_CCL_PUT_PATH=device`:

256MB alltoall, `UK_CCL_DEV_BLOCKS=64 UK_CCL_LARGE_TILES=4
UK_CCL_SIG_GROUP_TILES=4`, all wrong=0:

| ranks | CE path | device path | native |
|---:|---:|---:|---:|
| 2 | 310us | 324us | 416us |
| 4 | 550-580us | 612us | 425us |
| 8 | 715us | **590-620us** | 433us |

The device path is ~15% faster than the CE at 8 ranks and flattens the
scaling curve (4r ~= 8r ~= 610us), matching NCCL's contention behavior;
the CE still wins at 2 ranks (310 vs 324, both beat native). This is
the same SM-threaded mechanism native uses — a deliberate tradeoff vs
the earlier "no SM for alltoall" preference, but it is the only path so
far that improves on the CE at 8 ranks. Suggested policy: CE for 2
ranks, device path for 4+.

### Why native has no staging copy

Native/nccl-tests AllToAll is **out-of-place** (separate sendbuff and
recvbuff): the sender reads only its own sendbuff and writes only the
receiver's recvbuff, so no partition is both read and written
concurrently — no staging needed. The shim's ncclAllToAll used to be
in-place-only, which creates exactly that aliasing (partition p is both
the source of my Put to peer p and the target of peer p's Put into my
buffer) and forced the Input->Scratch staging copy. Supporting
out-of-place removes the copy entirely; in-place remains available
through the staged variant for callers that need it.
