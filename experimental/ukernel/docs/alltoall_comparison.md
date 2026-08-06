# AllToAll — design, B300 results, open questions

How the ukernel shim's AllToAll works, how it compares to native
nccl-tests alltoall, and what the remaining gap is. Performance data:
full 1M..256M sweep in
[b300_native_nccl_measurements.md](b300_native_nccl_measurements.md).

## Design

- **Out-of-place by default** (`sendbuff != recvbuff`, the nccl-tests /
  native shape). No partition is both read and written, so there is no
  aliasing and **no staging copy** — sends go straight from Input to the
  peer's Output. This is why native NCCL needs no staging either.
- **In-place** (`sendbuff == recvbuff`) is still accepted; partition p
  is then both the send source to peer p and the target of peer p's
  Put, so every send is staged through scratch (a real data race was
  found and fixed here — receivers got their own data back when the
  peer's source was clobbered mid-copy).
- **Signals**: fused PutSignal (the tag rides the put), aggregated with
  `UK_CCL_SIG_GROUP_TILES` (one signal per G tiles).
- **Put path** (same-host): IPC copy engine (BLK-independent) or device
  backend vectorized LD/ST (`UK_CCL_PUT_PATH=device`).

## B300 results (256MB, 2/4/8 ranks, all wrong=0)

| ranks | CE path (BLK=1 LT=4 G=4) | device path (BLK=64 LT=4 G=4) | native |
|---:|---:|---:|---:|
| 2 | 310us / 861 GB/s | 324us / 829 GB/s | 416us / 646 GB/s |
| 4 | 550-580us | 612us / 437 GB/s | 425us / 632 GB/s |
| 8 | 715us | **590-620us / ~400 GB/s** | 433us / 620 GB/s |

Algbw = full per-rank buffer / time (nccl-tests convention). The device
path wins at 4/8 ranks (~15% over CE) and flattens the scaling curve
(4r ~= 8r ~= 600us); the CE wins marginally at 2 ranks. Both beat
native at 2 ranks; neither reaches native at 4/8 yet.

## Mechanism findings

1. **Copy-engine ceiling**: raw `cudaMemcpyAsync` alltoall (no shim)
   does 191 / 297 / 368us at 2/4/8 ranks — but only with ranks running
   *unsynchronized* loops. The shim's collective is synchronized (every
   rank waits for all signals every iteration), so all 56 copies peak on
   the fabric together and the CE cannot reach its raw ceiling.
2. **TMA to peer memory is blocked on B300**: `cp.async.bulk`
   store/load to a peer-mapped (IPC) address hangs the kernel's
   bulk-group/mbarrier wait. Local TMA works. NCCL 2.29.7 does not use
   TMA for intra-node copies either — its `copy` primitive is per-thread
   vectorized LD/ST, and `cp.async.bulk` is only a TODO comment.
3. **The device copy op is now that vectorized LD/ST**: for both local
   and peer destinations (the TMA branch was removed). Forcing alltoall
   puts through it (`UK_CCL_PUT_PATH=device`) is what beats the CE at
   4/8 ranks — same mechanism native uses.

## Recommended config

- 2 ranks: CE (`UK_CCL_PUT_PATH=ipc UK_CCL_DEV_BLOCKS=1
  UK_CCL_LARGE_TILES=4 UK_CCL_SIG_GROUP_TILES=4`).
- 4+ ranks: device (`UK_CCL_PUT_PATH=device UK_CCL_DEV_BLOCKS=64
  UK_CCL_LARGE_TILES=4 UK_CCL_SIG_GROUP_TILES=4`).

Small/medium messages are worker/launch-bound on the device path and far
from native below ~128M; the CE path is better there but was only swept
at 256M.

## Open questions

- Closing the ~200us to native at 8 ranks (587 vs 433): the device-path
  copy kernel's chunking/pipelining and the per-put signal chain are the
  levers.
- A MoE-style comparison vs DeepEP (dispatch/combine shapes) was planned
  but not executed; revisit only if AllToAll becomes the workload focus.

## History notes

Earlier sections of this file (2026-08-04..05) chronicled a data-integrity
hunt (which turned out to be a harness fill/put race), an `ncclBarrier`
extension (removed — non-standard API), a lazy-worker experiment
(reverted), and the in-place race fix. The conclusions above supersede
them; the old text is recoverable from git history.
