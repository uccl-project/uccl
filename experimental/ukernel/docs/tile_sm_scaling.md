# Tile size vs SM count scaling (B300, single-buffer TMA reduce)

Date: 2026-08-04. Machine: mi-sky-b300 (B300 SXM6 x2 ranks, GPUs 6,7).
Build: `SM=103 ENABLE_TMA=0 REDUCE_ILP=4 REDUCE_SMEM_KB=224 TMA_REDUCE=1
TMA_WARPSPEC=0` (single-buffer TMA, 112KB chunks, 224KB smem).
Workload: `all_reduce_perf -b 256M -e 256M -g 1 -c 1` (default 20 iters).

## Motivation

Goal: fewer SMs at full rate. The shim's `adaptive_tile_bytes()` splits a
256MB allreduce into `UK_CCL_LARGE_TILES` (default 64) tiles of 4MB. Each
ring tile becomes one reduce task; the task is split across `UK_CCL_DEV_BLOCKS`
worker blocks. With 4MB tiles and BLK=32 each block gets only 128KB/task
→ nfull=1 chunk + tail: per-task fixed costs (host scheduling, multi-block
barrier, put window) dominate. Bigger tiles amortize that overhead.

## Results (oop / in-place, GB/s)

### LT sweep, BLK fixed

| LARGE_TILES (tile) | BLK=8 | BLK=16 | BLK=32 |
|---|---|---|---|
| 64 (4MB) | 169 / 131 | 200 / 178 | 172-210 / 168-188 |
| 32 (8MB) | 242 / 165 | 265 / 222 | 273-303 / 249-266 |
| 16 (16MB) | 293 / 192 | 347-363 / 288 | 397-457 / 333-390 |
| 8 (32MB) | 299-308 / 197 | 378-398 / 310-321 | 440-465 / 393-413 |
| 4 (64MB) | 277 / 191 | 365 / 296 | 431 / 398 |

Best point: **LT=8 (32MB tiles), BLK=32 → ~451 oop / ~413 ip**.
Fewer-SM point: **LT=8, BLK=16 → ~385-398 oop / ~311 ip** (2x the LT=64
baseline at the same SM count).

### BLK scaling at LT=8 (valid BLKs are powers of two)

| BLK (per rank) | total SMs | oop | ip |
|---|---|---|---|
| 8 | 16 | ~299 | ~197 |
| 16 | 32 | ~385-398 | ~311 |
| 32 | 64 | ~440-465 | ~393-413 |

BLK=12/24/48 crash with `misaligned address`: the tile (32MB) must divide
evenly by BLK, otherwise the per-block slice is not 16B-aligned for TMA.

## Analysis

1. **Tile size is the dominant lever at low SM counts.** LT=64→8 more than
   doubles BLK=16 throughput (200→398). The 4MB default was leaving ~90% of
   the per-task budget to fixed overhead.
2. **Per-SM efficiency drops as SMs are added** (BLK=8: ~19 GB/s/SM,
   BLK=16: ~12.5, BLK=32: ~7): total bandwidth saturates well before the
   per-SM ceiling. 32 SMs reach ~400, 64 SMs ~450; native NCCL 2.29.7 gets
   ~510 with 32 channels/GPU.
3. **In-place lags out-of-place by 15-20%** at every point. Suspects: the
   TMA chunk path loads src AND dst (identical buffers in-place → duplicated
   traffic), and/or put-path dependencies on the same buffer. Kernel-side
   fix candidate: detect `dst == src` and load once, halving TMA reads.
4. Run-to-run variance ~±3% (378-398 at LT=8 BLK=16); LT=64 BLK=32 showed
   163-210 across runs (task-count sensitive to host scheduling jitter).

## Commands

```bash
# sweep tile x blocks
for lt in 64 32 16 8 4; do
  for b in 8 16 32; do
    CUDA_VISIBLE_DEVICES=6,7 UK_CCL_DEV_BLOCKS=$b UK_CCL_LARGE_TILES=$lt \
      mpirun --mca hwloc_base_binding_policy none -np 2 \
      -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES -x UK_CCL_DEV_BLOCKS -x UK_CCL_LARGE_TILES \
      ../../thirdparty/nccl-tests/build/all_reduce_perf -b 256M -e 256M -g 1 -c 1 \
      | awk '/^   268/ {print $6, $7, $11}'
  done
done
```

## Update 2026-08-04 (2): vectorized smem reduce

`tma_bulk_reduce_chunk`'s per-chunk smem reduce was scalar; vectorized to
16B `TypedVec` (commit `8de98798`). Measured on B300:

### Pure reduce (device bench, 256M single task, no put/NVLink)

| BLK | before | after |
|---|---|---|
| 8 | — | 181 GB/s |
| 16 | ~262 | ~350 GB/s (+34%) |
| 32 | ~493 | ~640 GB/s (+30%) |
| 64 | — | ~1140 GB/s |

### Allreduce 256M (shim), LT=8 (32MB tiles)

| BLK | before oop | after oop | ip |
|---|---|---|---|
| 16 | 378-398 | 415-423 | 315-326 |
| 32 | 440-465 | 457-465 | 408-425 |

Conclusion: **the reduce kernel is no longer the bottleneck.** Pure reduce
at BLK=32 (640 GB/s) far exceeds the ~500 GB/s the 2-rank allreduce needs.
The allreduce ceiling (~425-465 oop, ~320-425 ip) is set by the
put/NVLink/task pipeline, not the reduce kernel.

### In-place gap root cause

In-place allreduce is 15-25% slower than out-of-place (BLK=16: 415-423 oop
vs 315-326 ip) NOT because of the reduce kernel (its dst/src are always
distinct: dst=Tmp accumulation, src=Input). The ring algorithm stages
in-place RS partials in Tmp(0) and the all-gather phase ends with a local
**Tmp→Output copy of the held shard (128MB per rank at 256MB/2 ranks)** —
that extra copy is the gap (~190us at BLK=16, i.e. ~680 GB/s copy rate).
Optimizing that copy (TMA chunk path) or fusing it is the lever for ip;
out-of-place already tracks the put-pipeline ceiling.

## Update 2026-08-04 (3): chunked TMA bulk copy closes the in-place gap

`copy()` gained a chunked cp.async.bulk path for large aligned messages
(commit `4c8c4935`): one smem buffer sized to the full dynamic-smem budget
(224KB chunk at a 224KB build — twice the reduce chunk), mbarrier load +
bulk-group store per chunk, vectorized tail. The in-place all-gather's
Tmp->Output shard copy now takes this path instead of the vectorized loop.

Allreduce 256M, LT=8 (32MB tiles):

| BLK | oop | ip before | ip after |
|---|---|---|---|
| 16 | 412-418 | 315-326 | **380-381** (+17-20%) |
| 32 | 442-457 | 408-425 | 406-425 (pipeline-limited) |

The in-place vs out-of-place gap shrank from ~25% to ~8% at BLK=16; the
remaining gap is put/copy pipeline dependencies in the all-gather, not the
copy kernel itself (device-bench copy throughput moved from ~680 GB/s
vectorized to the TMA path). All runs wrong=0.

## Update 2026-08-04 (4): pipeline bottleneck diagnosis (nsys)

Profiled the 2-rank 256M allreduce (LT=8, BLK=16, 3 iters) with nsys:

- `multiPersistentKernel` dominates CUDA GPU kernel time (~538ms across 24
  instances) but each instance is a worker lifecycle (task processing +
  idle-grace + relaunch), not pure compute — the shim default
  `device_idle_exit_us=500` exits the persistent worker after 500us idle
  and relaunches it per task group (24 instances / 3 iters / 2 ranks).
- Peer-to-peer memcpy (the IPC put) shows ~320 transfers, ~53us each
  (~17ms total): the put is DMA-engine work running concurrently with the
  reduce kernel, split into ~1.6MB window chunks rather than one 32MB
  bulk per tile.
- `UK_CCL_DEV_IDLE_EXIT_US=1000000` (keep worker resident) **deadlocks**
  the shim (GPU spins, no progress) — the relaunch path is load-bearing;
  the resident-worker path is not safe to default to without fixing the
  relaunch/exit handshake.

### Bandwidth budget

2-rank ring allreduce moves 256MB per rank each direction over NVLink.
At BLK=32 (~460 GB/s oop) that is ~460 GB/s per direction — the B300
NVLink single-direction ceiling is ~450-500 GB/s, and native NCCL 2.29.7
measures 510 GB/s. **The allreduce is now NVLink-bound, not reduce-bound**
(pure reduce at BLK=32 is 640 GB/s). Remaining headroom to native is the
put path's ability to keep all NVLink lanes busy (native uses 32 channels;
the shim uses a single IPC sliding window).

### Run-to-run variance

±10% between consecutive runs at LT=8/BLK=16 (410-455 oop, 372-417 ip);
BLK=32 is steadier (459-469 oop, 431-438 ip). Compare medians over 3+
runs, not single samples.
