# B300 native NCCL measurements

Native-NCCL baseline measurements on the B300 machine — the anchor for
the ukernel shim (match bandwidth at the same or fewer blocks/SMs).
Keep the commands so measurements are reproducible; append new runs with
a date section.

## Environment

- Host: `mi-sky-b300`, NVIDIA B300 SXM6 AC
- NCCL: 2.29.7 + cuda13.2 (git `b81d6a5a3`), system install
  (`/lib/x86_64-linux-gnu/libnccl.so.2`)
- Interconnect: NV18 (18 bonded NVLinks) full mesh — every GPU pair is
  NV18; no PCIe hop between any two GPUs
- nccl-tests: MPI build (OpenMPI 4.1.x)

## Native AllReduce, two-process baseline (2026-08-04)

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

## Kernel configuration (single-process `-g 2`, 2026-08-04)

```bash
NCCL_DEBUG=INFO ./all_reduce_perf -b 256M -e 256M -g 2 -n 5 2>&1 \
    | grep -iE "channel|protocol|version"
for c in 1 2 4 8 16 32; do
  NCCL_MAX_NCHANNELS=$c ./all_reduce_perf -b 256M -e 256M -g 2 -n 20 \
      2>/dev/null | awk '$1 ~ /^[0-9]+$/ && NF>=13 {print $6, $7}'
done
```

Findings:

- Kernel: `ncclDevKernel_AllReduce_Sum_f32_RING_LL` — **RING_LL protocol**;
  transport `P2P/direct pointer` (NVLink).
- 32 coll channels, 32 p2p channels per peer; NVLS multicast available
  (24 channels) but unused for the 2-rank AllReduce.
- Per-channel scaling is linear — no "few channels saturate" headroom:

| NCCL_MAX_NCHANNELS | 256M time (us) | algbw (GB/s) |
|---:|---:|---:|
| 1 | 11618 | 23.11 |
| 2 | 5826 | 46.07 |
| 4 | 2932 | 91.56 |
| 8 | 1493 | 179.82 |
| 16 | 793 | 338.59 |
| 32 | 526 | 510.14 |

- Per channel ≈ 16 GB/s at 32 channels (two-process: 515 GB/s total).
- Single-process `-g 2` peak: 507-508 GB/s @256M (vs two-process
  514-515 GB/s).

## Shim reference on the same day (for orientation)

Best config found by the 08-04 sweep: `UK_CCL_LARGE_TILES=8
UK_CCL_TILE_MIN_BYTES=8M UK_CCL_IPC_BATCH=16`, AllReduce 256M OOP. This
predates the fused reduce+copy work — see
[optimization_framework.md](optimization_framework.md), Appendix C for
the current numbers.

| shim blocks (`UK_CCL_DEV_BLOCKS`) | 256M time (us) | algbw (GB/s) | vs native |
|---:|---:|---:|---:|
| 8 (default) | 1591 | 168.7 | 33% |
| 16 | 1017 | 264.0 | 51% |
| 32 | 655 | 409.6 | 80% |
| 64 | 553 | 485.2 | 94% |

Threads/block fixed at 256 (512 exceeds the ILP-reduce register budget).
Goal: reach native's 510-515 GB/s at **32 blocks or fewer**.

## Latest full-size sweep (2026-08-06, 2/4/8 ranks)

Both sides run with the same MPI invocation; only `LD_LIBRARY_PATH`
differs (shim = `build/nccl/lib`, native = system NCCL 2.29.7). All
results wrong=0.

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

The shim carries a fixed per-tile host-signal floor (small sizes are
disproportionately slow) and busbw falls with rank count while native's
rises. The fused RS/AG work (optimization_framework.md, Appendix C)
improved the 256M 8-rank point from ~2000us to ~1490us; the small-size
floor and the ring critical path remain the targets.

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
path, BLK=64) and far from native below ~128M; at 128M+ it closes to
1.5-1.9x at 4/8 ranks and beats native at 2 ranks. See
[alltoall_comparison.md](alltoall_comparison.md) for the recommended
per-rank config.

## Native AllGather / ReduceScatter baselines (2026-08-15)

256MB, nccl-tests, n=20 w=5, all wrong=0. OOP = out-of-place,
ip = in-place; algbw / busbw in GB/s.

AllGather:

| ranks | oop time (us) | oop algbw | oop busbw | ip time (us) | ip algbw | ip busbw |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 323.1 | 830.8 | 415.4 | 301.8 | 889.5 | 444.8 |
| 4 | 377.1 | 711.8 | 533.8 | 375.1 | 715.5 | 536.7 |
| 8 | 402.8 | 666.4 | 583.1 | 399.6 | 671.8 | 587.8 |

ReduceScatter:

| ranks | oop time (us) | oop algbw | oop busbw | ip time (us) | ip algbw | ip busbw |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 330.5 | 812.3 | 406.2 | 325.5 | 824.8 | 412.4 |
| 4 | 380.2 | 706.1 | 529.6 | 378.1 | 710.0 | 532.5 |
| 8 | 397.7 | 675.0 | 590.6 | 392.7 | 683.6 | 598.1 |

## NCCL_NTHREADS sensitivity (2026-08-15)

256MB AllReduce, 2 ranks, native, n=20 w=5, all wrong=0. Default is
512; both 512 and 1024 sit at the sweet spot, 256 loses ~15% and 128
loses ~40% — thread count is a real lever on the B300 ring kernel, so
shim comparisons should keep native at its default 512.

| NCCL_NTHREADS | time (us) | algbw (GB/s) |
|---:|---:|---:|
| 128 | 887.5 | 302.5 |
| 256 | 610.7 | 439.6 |
| 512 (default) | 521.1 | 515.1 |
| 1024 | 519.2 | 517.0 |
