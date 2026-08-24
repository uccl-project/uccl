# AllToAll — design, B300 results, open questions

How the ukernel shim's AllToAll works, how it compares to native
nccl-tests alltoall, and what the remaining gap is. Full 1M..256M sweep
data is in [b300_native_nccl_measurements.md](b300_native_nccl_measurements.md);
the copy-engine contention mechanism behind the gap is analyzed in
[optimization_framework.md](optimization_framework.md) (Appendix A).

## Design

- **Out-of-place by default** (`sendbuff != recvbuff`, the nccl-tests /
  native shape). No partition is both read and written, so there is no
  aliasing and **no staging copy** — sends go straight from Input to the
  peer's Output. This is why native NCCL needs no staging either.
- **In-place** (`sendbuff == recvbuff`) is still accepted; partition p
  is then both the send source to peer p and the target of peer p's Put,
  so every send is staged through scratch (a real data race was found
  and fixed here — receivers got their own data back when the peer's
  source was clobbered mid-copy).
- **Signals**: fused PutSignal (the tag rides the put), aggregated with
  `UK_CCL_SIG_GROUP_TILES` (one signal per G tiles).
- **Put path** (same-host): IPC copy engine (BLK-independent) or device
  backend vectorized LD/ST (`UK_CCL_PUT_PATH=device`).

## L40S results (2026-08-24, pure CE, 256MB same-node)

Same-node `alltoall_perf` (ncclAllToAll), `UK_CCL_PUT_PATH=ipc`
(pure CE, zero SM), algbw:

| ranks | shim (before) | shim (rotated order) | native | verdict |
|---:|---:|---:|---:|---|
| 2 | 48.2 | 48.1 | 47.8 | shim ~tied/+1% |
| 4 | 12.9 | 16.0-16.2 | 15.3-16.1 | shim ~tied (was losing) |
| 8 | 7.1 | 7.7 | 7.8 | ~tied |
| 12 cross-node | — | 2.8 | 4.2 | native +50% (open gap) |

Correction: an earlier "native" comparison used a binary whose legacy
DT_RPATH silently loaded the shim lib, so those numbers were
shim-vs-shim. With the standard `ncclAlltoAll` symbol (added to the
shim as an alias) and an MPI-broadcast unique id, the real native
comparison above shows parity same-node — the rotation still matters
(12.9 -> 16, turning a loss into a tie) but does not overtake native.

The 4-rank jump is the **incast signature**: the lowering
issued every rank's per-peer copies in ascending peer order, so at the
synchronized collective start all ranks' first copy targeted the same
peer — overloading that peer's CE/ingress arbitration (the B300
"serializing per rank does not help" observation is consistent: its
serial test still aimed every rank's first transfer at the same
destination). `build_alltoall_pairwise_algo` now rotates the send order
(rank r sends to r+1, r+2, ...), a Latin square that spreads the first
wave across distinct destinations. Verify OK at 2/4/8 ranks and 128M.
The cross-node alltoall gap (2.8 vs 4.2 GB/s) is the next open item —
the RDMA/proxy path rather than the CE path.

## B300 results (256MB, 2/4/8 ranks, all wrong=0)

| ranks | CE path (BLK=1 LT=4 G=4) | device path (BLK=64 LT=4 G=4) | native |
|---:|---:|---:|---:|
| 2 | 310us / 861 GB/s | 324us / 829 GB/s | 416us / 646 GB/s |
| 4 | 550-580us | 612us / 437 GB/s | 425us / 632 GB/s |
| 8 | 715us | **590-620us / ~400 GB/s** | 433us / 620 GB/s |

Algbw = full per-rank buffer / time (nccl-tests convention). The device
path wins at 4/8 ranks (~15% over CE) and flattens the rank-scaling
curve (4r ~= 8r ~= 600us); the CE wins marginally at 2 ranks. Both beat
native at 2 ranks; neither reaches native at 4/8 yet.

## Mechanism findings

1. **Copy-engine ceiling**: raw `cudaMemcpyAsync` alltoall (no shim)
   does 191 / 297 / 368us at 2/4/8 ranks — but only with ranks running
   *unsynchronized* loops. The shim's collective is synchronized (every
   rank waits for all signals every iteration), so all 56 copies peak on
   the fabric together and the CE cannot reach its raw ceiling. Isolated
   in the standalone CE contention microbenchmark (see
   optimization_framework.md, Appendix A).
2. **TMA to peer memory is blocked on B300**: `cp.async.bulk`
   store/load to a peer-mapped (IPC) address hangs the kernel's
   bulk-group/mbarrier wait. Local TMA works. Native NCCL does not use
   TMA for intra-node copies either — its `copy` primitive is per-thread
   vectorized LD/ST.
3. **The device copy op is that vectorized LD/ST**: for both local and
   peer destinations (the TMA branch was removed). Forcing alltoall puts
   through it (`UK_CCL_PUT_PATH=device`) is what beats the CE at 4/8
   ranks — the same mechanism native uses.

## Recommended config

- 2 ranks: CE (`UK_CCL_PUT_PATH=ipc UK_CCL_DEV_BLOCKS=1
  UK_CCL_LARGE_TILES=4 UK_CCL_SIG_GROUP_TILES=4`).
- 4+ ranks: device (`UK_CCL_PUT_PATH=device UK_CCL_DEV_BLOCKS=64
  UK_CCL_LARGE_TILES=4 UK_CCL_SIG_GROUP_TILES=4`).

### CE + device hybrid (2026-08-15)

`UK_CCL_A2A_HYBRID=1` splits each per-peer send into a CE half and a
device-copy half (per-op `put_path_hint`), overlapping the CE engine and
the worker. Unlike RS there is no reduce on the worker, so the device
half does not compete with compute — this is the first positive fusion
result. Measured with `bench/alltoall_perf.cu` (direct ncclAllToAll),
256M, verify OK:

**Correction (2026-08-15, later the same day):** the send-side split
was hardcoded 50/50 while only the recv side honored
`UK_CCL_A2A_HYBRID_CE_PCT`, so the "CE only (pct=100)" baseline below
was actually a 50/50 hybrid — it always created device puts and forced
a worker launch. After the fix (`coll_algo.cc`), pct=100 emits no
device ops at all (plain IPC puts, zero worker SM), and pct=0 falls
back to the plain auto path. Degenerate splits (pct at the 0/100
boundaries) previously produced plan shapes that intermittently
segfaulted at 8 ranks — the boundary fallback avoids them.

An attempt to make the device worker itself lazy (bind on first use,
created by the drain thread) stalled the lazily created multi-block
worker on B300 — fifo bound=1 but tail never advanced, hanging the
hybrid at 4/8 ranks. Reverted to pre-created workers; the all-CE
zero-SM goal is met at the plan level instead (pct=100 has no device
ops, and pre-created workers idle-exit after the grace period).

| ranks | CE only | hybrid | delta |
|---:|---:|---:|---:|
| 4 | 449.5 GB/s | **533.0** | +19% |
| 8 | 387.1 | 399.4 | +3% |

4-rank hybrid reaches 84% of native (632); 8-rank 64% (620). The
post-fix tuning sweep below covers the split ratio and device-half
block count.

### Hybrid tuning sweep (2026-08-15, post-fix)

After the send-side pct fix, the alignment fix (16B copy vector width,
`coll_algo.cc` + `persistent_kernel_ops.cu` per-block chunk rounding)
and the signal-map concurrency fix (`communicator.cc` single
`sig_maps_mu_`), the hybrid is stable at 4/8 ranks (no hangs/segfaults
across 3×10 8-rank + 3×5 4-rank runs). Median algbw, 256M, `alltoall_perf`,
LT=4:

8 ranks:

| pct \ blk | 16 | 32 | 64 |
|---|---:|---:|---:|
| 100 (CE only) | 366.0 | 376.8 | — |
| 30 | 361.4 | **402.8** | — |
| 50 | 378.4 | 374.2 | 388.8 |
| 70 | 369.2 | 380.3 | 364.5 |

Best 8-rank config: **pct=30, blk=32 → 402.8 GB/s** (65% of native
620). More device share helps at 8 ranks because CE copy-engine
contention grows with rank count; the worker's LD/ST path does not
contend the same way.

4 ranks:

| pct \ blk | 32 | 64 |
|---|---:|---:|
| 100 (CE only) | 458.1 | — |
| 30 | 457.4 | — |
| 50 | **512.7** | 479.4 |
| 70 | 481.5 | — |

Best 4-rank config: **pct=50, blk=32 → 512.7 GB/s** (81% of native
632). At 4 ranks the balanced split wins; the device worker is not
contended enough to justify a bigger share.

Recommended defaults: 4 ranks `pct=50 blk=32`, 8 ranks `pct=30 blk=32`
(LT=4). Both beat pure CE and are stable; the remaining gap to native
is the synchronized CE peak + worker copy efficiency (see
[optimization_framework.md](optimization_framework.md), Appendix A).

Clean 4-rank A/B (median of 3, 256M, LT=4):

| config | algbw | vs single-put CE |
|---:|---:|---:|
| single CE put (H=0) | 476 GB/s | — |
| split 2 CE puts (pct=100) | 507 | +6.5% |
| hybrid CE+device (pct=50, blk=32) | **534** | +12% |

The hybrid's gain is a combination of the op split (+6.5%, two smaller
CE copies schedule better) and the CE+device overlap (+5%). Why BLK
shows up even at pct=100: the worker kernels are pre-created at
communicator init (BLK idle blocks polling the FIFO) and occupy SMs even
when no device op runs — the alltoall self-slice is done on the user
stream via CE (`external_self_slice`), so it is not the cause. Workers
stay pre-created (lazy creation was reverted — see
optimization_framework.md) but idle-exit after the 500us grace, and
after the send-side pct fix pct=100 emits no device ops at all, so
all-CE collectives run with zero device-worker SM occupancy.

Small/medium messages are worker/launch-bound on the device path and far
from native below ~128M; the CE path is better there but was only swept
at 256M.

## Open questions

- Closing the ~200us to native at 8 ranks (587 vs 433): the device-path
  copy kernel's chunking/pipelining and the per-put signal chain are the
  levers.
- A MoE-style comparison vs DeepEP (dispatch/combine shapes) was planned
  but not executed; revisit only if AllToAll becomes the workload focus.
