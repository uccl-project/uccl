# B300 native NCCL measurements

Living log of native-NCCL baseline measurements on the B300 machine
(anchor for the ukernel shim: match bandwidth at the same or fewer
blocks/SMs). Append new runs with a date section; keep the commands so
measurements are reproducible.

## Environment

- Host: `mi-sky-b300`, NVIDIA B300 SXM6 AC
- NCCL: 2.29.7 + cuda13.2 (git `b81d6a5a3`), system install
  (`/lib/x86_64-linux-gnu/libnccl.so.2`)
- Interconnect: NV18 (18 bonded NVLinks) full mesh — every GPU pair is
  NV18; no PCIe hop between any two GPUs
- nccl-tests: MPI build (OpenMPI 4.1.x)

## 2026-08-04 — native AllReduce, two-process (`mpirun -np 2`, GPUs 6/7)

Command (CPU binding flag is irrelevant for native but kept for parity
with shim runs):

```bash
CUDA_VISIBLE_DEVICES=6,7 mpirun --mca hwloc_base_binding_policy none -np 2 \
    ../../thirdparty/nccl-tests/build/all_reduce_perf -b 1M -e 256M -f 2 -g 1 -c 1
```

`float sum`, 2 ranks, validation on, 0 wrong everywhere.

| size | OOP time (us) | OOP algbw (GB/s) | IP time (us) | IP algbw (GB/s) |
|---:|---:|---:|---:|---:|
| 1M | 16.69 | 62.84 | 15.72 | 66.72 |
| 2M | 22.00 | 95.31 | 22.08 | 94.98 |
| 4M | 40.19 | 104.37 | 39.65 | 105.78 |
| 8M | 46.62 | 179.93 | 44.93 | 186.72 |
| 16M | 55.42 | 302.72 | 55.13 | 304.32 |
| 32M | 84.76 | 395.86 | 82.69 | 405.81 |
| 64M | 149.3 | 449.62 | 149.5 | 448.89 |
| 128M | 277.5 | 483.63 | 276.6 | 485.31 |
| 256M | 521.8 | 514.41 | 520.9 | 515.38 |

## 2026-08-04 — kernel configuration (single-process `-g 2` profiling)

Commands:

```bash
# channel count / protocol
NCCL_DEBUG=INFO ./all_reduce_perf -b 256M -e 256M -g 2 -n 5 2>&1 \
    | grep -iE "channel|protocol|version"
# channel scaling
for c in 1 2 4 8 16 32; do
  NCCL_MAX_NCHANNELS=$c ./all_reduce_perf -b 256M -e 256M -g 2 -n 20 \
      2>/dev/null | awk '$1 ~ /^[0-9]+$/ && NF>=13 {print $6, $7}'
done
```

Findings:

- Kernel: `ncclDevKernel_AllReduce_Sum_f32_RING_LL` — **RING_LL protocol**
- 32 coll channels, 32 p2p channels per peer; NVLS multicast available
  (24 channels) but not used for the 2-rank AllReduce
- Transport: `P2P/direct pointer` (NVLink)
- Per-channel scaling is linear — no "few channels saturate" headroom:

| NCCL_MAX_NCHANNELS | 256M time (us) | algbw (GB/s) |
|---:|---:|---:|
| 1 | 11618 | 23.11 |
| 2 | 5826 | 46.07 |
| 4 | 2932 | 91.56 |
| 8 | 1493 | 179.82 |
| 16 | 793 | 338.59 |
| 32 | 526 | 510.14 |

- Per channel ≈ 16 GB/s at 32 channels (two-process: 515 GB/s total)
- Threads/block: not measured directly (ncu unavailable/hanging on this
  box); probe indirectly with `NCCL_NTHREADS=256` (see TODO). Expect
  512 for LL (default `NCCL_NTHREADS`).
- Single-process `-g 2` peak: 507-508 GB/s @256M (vs two-process
  514-515 GB/s)

## Reference — ukernel shim on B300 (same day, for comparison)

Best config found by sweep: `UK_CCL_LARGE_TILES=8 UK_CCL_TILE_MIN_BYTES=8M
UK_CCL_IPC_BATCH=16`, AllReduce 256M out-of-place:

| shim blocks (`UK_CCL_DEV_BLOCKS`) | 256M time (us) | algbw (GB/s) | vs native |
|---:|---:|---:|---:|
| 8 (default) | 1591 | 168.7 | 33% |
| 16 | 1017 | 264.0 | 51% |
| 32 | 655 | 409.6 | 80% |
| 64 | 553 | 485.2 | 94% |

Threads/block fixed at 256 (512 exceeds the ILP-reduce register budget).
Goal: reach native's 510-515 GB/s at **32 blocks or fewer** — i.e. raise
per-block throughput from ~12.8 to ~16 GB/s, and/or overlap puts with
reduce instead of adding blocks.

## TODO / open items

- `NCCL_NTHREADS=256` at 32 channels (thread-count sensitivity)
- Record native AllGather / ReduceScatter baselines the same way
# 8-card sweep (2026-08-05) — shim vs native at 4/8 ranks

Environment: all 8 B300 GPUs idle, same node. `mpirun -np N` with each
process selecting device = MPI local rank (no CUDA_VISIBLE_DEVICES
filtering; nccl-tests uses `localRank` as the device ordinal).

## 256MB AllReduce (algbw / busbw)

| ranks | shim | native |
|---|---|---|
| 2 | ~580us / 455 GB/s oop | ~529us / 507 GB/s |
| 4 | ~23-28ms / 9-12 GB/s | ~675us / 597 GB/s |
| 8 | ~45-60ms / 5-6 GB/s | ~719us / 653 GB/s |

All runs wrong=0. Native scales normally (2→8 ranks stays sub-ms); the
shim collapses at 4+ ranks (~40-70x slower than 2-rank).

## Bottleneck analysis (4-rank, -n 1, nsys)

- 336 tiled ops per iteration (2-rank: 224) — op count is fine (1.5x).
- P2P memcpys: 6144 x 4MB, ~9.9us each (~405 GB/s per copy) — the DMA
  puts are fast; total memcpy time only ~10ms/iter.
- multiPersistentKernel dominates kernel time but mostly idle-spin +
  relaunch at the default 500us grace; with
  `UK_CCL_DEV_IDLE_EXIT_US=50000` the real reduce work is ~3.8ms/iter.
- Remaining ~20ms/iter is per-op scheduling/signaling: 4M allreduce
  takes ~11ms (i.e. ~3.7ms per ring step for 1MB of data) — pure
  per-op signal/sync overhead, not bandwidth.
- 2-rank per-op cost is ~2.6us; 4-rank is ~83us (32x) — signaling
  coordination does not scale with peer count.

Conclusion: the shim's multi-rank allreduce is gated by per-tile
signal/sync overhead (one Signal/WaitSignal per tile, host-side ring
polling). Fix directions: signal aggregation (signal_group_tiles > 1),
batching waits, or device-side signal matching. 2-rank remains fast
(put/reduce overlap fine); 4+ ranks need this before the 8-GPU
capability is usable.

## 2026-08-05 (late) — post-revert re-measurement: the 23-60ms collapse
is NOT reproducible; NUMA is not the cause

Correction to the 8-card sweep above: the 23-60ms 4/8-rank collapse was
**not** caused by the code state at the time. A/B test on the same box:
commit `199d520b` (lazy worker, the only code delta active during the
sweep) was re-applied on top of the current tree and rebuilt — 4-rank
256M allreduce still measured ~1054us, identical to the reverted tree
(1054us), and 2-rank stayed ~562us. The collapse is therefore not
reproducible by any current or swept code state; it most plausibly came
from a transient condition during that measurement (host CPU load —
shim per-tile signaling is host-polling based, unlike native NCCL's
device-driven LL flags, which were measured fine at the same time — or
a different build configuration). The follow-up experiments (signal
aggregation `28f92027`, immediate Signal local-completion `44595e71`,
busy-wait `4d50febd`) were reverted because they regressed **2-rank**
in the 05:00-07:00 debug window, not because they caused the 4/8-rank
collapse.

Current clean numbers (tuned config `UK_CCL_LARGE_TILES=8
UK_CCL_TILE_MIN_BYTES=8388608 UK_CCL_IPC_BATCH=16 UK_CCL_DEV_BLOCKS=64`,
256M allreduce, `-g 1 -c 1 -n 20`, OOP):

| ranks | shim time (us) | shim algbw (GB/s) | shim busbw (GB/s) | native time (us) | native busbw (GB/s) |
|---:|---:|---:|---:|---:|---:|
| 2 | 518-585 | 462-518 | 462-518 | 529 | 507 |
| 4 | 1069-1075 | 250-253 | 374-378 | 676 | 595 |
| 8 | 1994-2009 | 133-135 | 234-236 | 720 | 652 |

All wrong=0. Native busbw rises with rank count (507 -> 595 -> 652);
the shim's busbw falls (463 -> 377 -> 234) — per-tile host signaling
does not pipeline across ring hops. Per-op cost from end-to-end time:
2.6us (2r, 224 ops) -> 3.2us (4r, 336 ops) -> 4.5us (8r, ~448 ops).
This per-tile signaling gap is real and is the thing to attack (signal
aggregation + batched waits, then put/reduce overlap).

Small messages carry a fixed latency floor: 4-rank 1M/4M/16M all take
~440-500us regardless of size (vs native ~25-40us) because a handful of
tiles cannot fill the pipeline; only 64M+ starts to move
(688us @64M, 1054us @256M). This floor is host signal chain latency
(signal drain -> enqueue cycle -> device dispatch), and it is exactly
what batching waits + aggregating signals attacks.

### NUMA experiments (B300 is 4 NUMA nodes: GPUs 0-1 / 2-3 / 4-5 / 6-7)

- 2-rank same NUMA (6,7 and 0,1): 580-585us / 459-463 GB/s.
- 2-rank cross NUMA (0,4): 518-554us / 484-518 GB/s — **no NUMA
  penalty**; if anything slightly faster, inside run-to-run noise.
- 4-rank on 2 NUMA nodes (0,1,2,3) vs all 4 NUMA nodes (0,2,4,6):
  identical (378 vs 371 GB/s busbw).
- 8-rank with `--map-by numa` CPU binding: no change (232 vs 234 GB/s).

Conclusion: the multi-rank collapse and the remaining scaling gap are
**not** NUMA effects. Signal rings are O(1)-per-peer polls
(write_idx/read_idx), so polling cost per rank is small; the latency is
in the per-tile host signal/wait chain (one Signal/WaitSignal per tile,
host-side ring polling, no batching at G=1).

### [tss] debug snapshot (4-rank, `UK_CCL_DEBUG=1`, `-w 0 -n 1`)

Debug printing inflates absolute times, but shows the shape:

- enqueue_loop cycle: median ~2-4us (ready -> enq_ring_done).
- put accepted -> fused Signal local-completion: median ~0.1-0.2ms per
  group (must wait for the next enqueue cycle; G=1 => one cycle's worth
  of host signaling per tile).
- arrival drains (sig_recv / tpt_done) come in bursts with median gaps
  ~0.14-0.17ms under debug.

The host signal chain, not DMA puts or reduce kernels, is the per-op
latency driver at 4+ ranks.

## 8-rank alltoall (harness is 2-rank only)

Not yet measured — `bench/alltoall_perf.cu` is hardcoded to 2 ranks.
Extending it (nranks param, unique-id broadcast, per-peer send/recv
loop) is the next step once the allreduce signaling issue is addressed.

## 2026-08-05 (late) — 2/4/8-rank sweep: shim vs native (allreduce +
alltoall), after the history cleanup

Both sides run on the same box with the same MPI invocation; only
`LD_LIBRARY_PATH` differs (shim = `build/nccl/lib`, native = system
NCCL 2.29.7). All results wrong=0.

### AllReduce (nccl-tests `all_reduce_perf -b 1M -e 256M -f 2 -g 1 -c 1 -n 20`)

Shim config: `UK_CCL_LARGE_TILES=8 UK_CCL_TILE_MIN_BYTES=8388608
UK_CCL_IPC_BATCH=16 UK_CCL_DEV_BLOCKS=64`. OOP algbw (GB/s):

| size | shim 2r | shim 4r | shim 8r | native 2r | native 4r | native 8r |
|---:|---:|---:|---:|---:|---:|---:|
| 1M | 5.9 | 2.4 | 0.9 | 55.8 | 44.9 | 32.5 |
| 2M | 11.5 | 4.6 | 1.8 | 94.3 | 72.8 | 57.6 |
| 4M | 21.6 | 9.2 | 3.4 | 104.2 | 92.9 | 71.0 |
| 8M | 42.6 | 17.8 | 6.8 | 184.5 | 170.8 | 106.8 |
| 16M | 79.7 | 34.3 | 13.4 | 296.3 | 165.8 | 155.0 |
| 32M | 128.0 | 64.5 | 25.4 | 391.0 | 271.0 | 195.4 |
| 64M | 192.6 | 103.6 | 51.4 | 440.6 | 361.1 | 239.9 |
| 128M | 316.2 | 176.2 | 87.9 | 482.2 | 382.8 | 336.2 |
| 256M | 464.5 | 261.7 | 138.2 | 514.9 | 399.0 | 373.3 |

256M ratio shim/native: 1.11x (2r), 1.53x (4r), 2.70x (8r). The shim
carries a fixed per-tile host signal floor (small sizes are
disproportionately slow) and busbw falls with rank count while native's
rises — the per-tile signal/wait chain remains the target for the next
optimization pass.

### AllToAll (256MB, in-place for the shim; `-g 1 -c 1 -n 20`)

Shim config: out-of-place `sendbuff != recvbuff`,
`UK_CCL_LARGE_TILES=1`; the self-slice is a copy-engine
cudaMemcpyAsync on the user stream and all peer exchanges are IPC puts,
so BLK is irrelevant (BLK=1 == BLK=64). See alltoall_comparison.md for
the in-place race fix and why native needs no staging (out-of-place).
Native = nccl-tests `alltoall_perf` (ncclSend/ncclRecv, out-of-place).

| ranks | shim time (us) | shim algbw (GB/s) | shim busbw (GB/s) | native time (us) | native algbw (GB/s) | native busbw (GB/s) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 312 | 861 | 431 | 415.6 | 645.8 | 322.9 |
| 4 | 544 | 493 | 370 | 424.7 | 632.1 | 474.0 |
| 8 | 706 | 380 | 333 | 432.6 | 620.5 | 543.0 |

The shim wins at 2 ranks and trails at 4/8 — the gap is per-peer IPC
put overhead (send window / launch pipelining), not staging or SM
blocks; the persistent worker is completely out of the AllToAll data
path.

## 2026-08-06 — full-size sweep: shim vs native, 2/4/8 ranks

### AllReduce (OOP algbw GB/s; shim LT=8 TM=8M IB=16 BLK=64)

| size | shim 2r | shim 4r | shim 8r | native 2r | native 4r | native 8r |
|---:|---:|---:|---:|---:|---:|---:|
| 1M | 5.4 | 1.9 | 0.9 | 66.0 | 41.6 | 32.2 |
| 2M | 10.5 | 3.9 | 1.7 | 98.0 | 74.0 | 58.3 |
| 4M | 21.1 | 7.4 | 3.3 | 106.5 | 94.3 | 71.6 |
| 8M | 39.1 | 14.6 | 6.9 | 180.9 | 175.4 | 107.7 |
| 16M | 76.2 | 27.8 | 12.7 | 297.8 | 166.0 | 154.7 |
| 32M | 134.1 | 54.2 | 24.7 | 394.4 | 272.1 | 196.0 |
| 64M | 202.2 | 90.0 | 48.4 | 442.5 | 362.2 | 240.2 |
| 128M | 324.6 | 148.6 | 82.0 | 481.7 | 385.4 | 337.2 |
| 256M | 495.0 | 246.3 | 128.4 | 515.8 | 399.9 | 373.5 |

### AllToAll (OOP algbw GB/s; shim = device path BLK=64 LT=4 G=4)

| size | shim 2r | shim 4r | shim 8r | native 2r | native 4r | native 8r |
|---:|---:|---:|---:|---:|---:|---:|
| 1M | 12.2 | 7.7 | 7.3 | 58.2 | 46.7 | 47.1 |
| 2M | 20.8 | 15.6 | 15.0 | 107.6 | 87.2 | 89.3 |
| 4M | 44.2 | 33.6 | 27.7 | 179.4 | 149.3 | 143.7 |
| 8M | 70.6 | 43.7 | 43.4 | 253.9 | 236.9 | 224.9 |
| 16M | 72.6 | 61.5 | 39.6 | 385.8 | 331.4 | 306.7 |
| 32M | 136.8 | 41.9 | 46.8 | 497.7 | 442.6 | 392.0 |
| 64M | 131.0 | 63.4 | 45.5 | 533.7 | 528.5 | 488.6 |
| 128M | 639.8 | 395.0 | 337.6 | 605.1 | 592.1 | 564.9 |
| 256M | 849.1 | 431.3 | 400.4 | 652.5 | 635.3 | 620.3 |

Notes: AllToAll shim small/medium sizes are worker/launch-bound (device
path, BLK=64) and far from native below 128M; at 128M+ it closes to
1.5-1.9x at 4/8 ranks and beats native at 2 ranks. The CE path is
better at small sizes but was only swept at 256M. AllReduce small sizes
carry the per-tile host-signal floor (fixed ~200-600us latency at
1-16M).
