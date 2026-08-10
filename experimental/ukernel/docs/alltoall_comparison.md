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
   the fabric together and the CE cannot reach its raw ceiling. Isolated
   in the standalone CE contention microbenchmark below.
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

## CE contention microbenchmark (2026-08-10)

Standalone 8-process alltoall (`bench/ce_contention.cu`, one process per
GPU, no shim). 256MB per rank, 32MB per copy, per-peer streams, mmap
sense-reversing barrier for the synchronized mode. `--serial 1` puts all
7 copies on one stream (8 concurrent transfers instead of 56);
`--sm 1` replaces `cudaMemcpyAsync` with a vectorized 512x256 copy
kernel (LD/ST over NVLink, same pattern).

Per-copy time (us), 8 ranks, iters=20 (rank 0 is the barrier leader, so
it always enqueues a few us early — which is itself a queueing signal):

| rank | CE unsync | CE sync | SM unsync | SM sync | CE serial-sync |
|---:|---:|---:|---:|---:|---:|
| 0 | 58.5 | 69.2 | 49.6 | 52.0 | 53.6 |
| 1 | 59.7 | 109.5 | 52.1 | 67.9 | 89.5 |
| 2 | 54.1 | 141.0 | 51.2 | 82.4 | 118.2 |
| 3 | 56.3 | 160.6 | 51.0 | 91.4 | 145.8 |
| 4 | 53.5 | 160.8 | 49.3 | 102.9 | 155.7 |
| 5 | 55.1 | 134.4 | 50.1 | 103.1 | 141.3 |
| 6 | 53.9 | 148.4 | 49.9 | 98.2 | 155.7 |
| 7 | 56.0 | 141.0 | 51.0 | 95.8 | 146.3 |

2/4 ranks (same binary, 128MB/64MB per copy): CE sync ≈ CE unsync at 2
ranks (189-191us); at 4 ranks sync degrades to 99-181us vs 98-111us
unsync (ranks 1-3, rank 0 unaffected).

Round-time aggregate (8 ranks, 1.75GB per round):

| mode | avg round | aggregate |
|---|---:|---:|
| CE unsync | 390us | ~4.5 TB/s |
| CE sync | 931us | ~1.9 TB/s |
| CE serial-sync (8 concurrent) | 880us | ~2.0 TB/s |
| SM sync | 607us | ~2.9 TB/s |

Readings:

1. **CE contention is real**: synchronized 8-rank peaks cost 2-3x per
   copy vs staggered issue (69-161us vs 54-60us), and the barrier leader
   always wins — first-enqueued transfers cut the queue.
2. **The CE is only part of it**: the SM copy kernel under the same
   synchronized peak drops far less (52-103us, 2.9 TB/s aggregate). The
   fabric/NVLink startup and arbitration also cost, but the CE makes it
   ~50% worse than an SM copy would.
3. **Serializing per rank does not help**: 8 concurrent CE copies
   degrade almost as much as 56. The penalty is not "too many
   outstanding transfers", it is the synchronized start itself.

Implication for AllReduce: moving the copy work from the CE to SM
(fused copy+reduce reading peer buffers directly) targets the engine
that survives the synchronized peak better, so it should recover most
of the gap — but the residual fabric cost means full recovery to the
staggered ceiling is not expected from path switching alone.

Run it on the B300:

```bash
export PATH=/usr/local/cuda/bin:/usr/bin:/bin
nvcc -O3 -arch=sm_103 -o /tmp/ce_contention bench/ce_contention.cu
rm -f /tmp/ce_bar_* /tmp/ce_h_* /tmp/ce_rdy_*
for r in 0 1 2 3 4 5 6 7; do
  /tmp/ce_contention --rank $r --nranks 8 --bytes $((1<<28)) --iters 20 \
    > /tmp/ce_r$r.log 2>&1 &
done
wait
for r in 0 1 2 3 4 5 6 7; do cat /tmp/ce_r$r.log; done
```

## History notes

Earlier sections of this file (2026-08-04..05) chronicled a data-integrity
hunt (which turned out to be a harness fill/put race), an `ncclBarrier`
extension (removed — non-standard API), a lazy-worker experiment
(reverted), and the in-place race fix. The conclusions above supersede
them; the old text is recoverable from git history.
