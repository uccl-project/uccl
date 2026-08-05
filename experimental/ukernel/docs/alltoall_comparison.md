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
