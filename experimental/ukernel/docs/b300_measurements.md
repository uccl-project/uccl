# B300 benchmark report (current uk-300)

Shim (ukernel) vs native NCCL on one 8-GPU B300 node, built from `uk-300` HEAD (three fixes since the last report: fused-RS thread sync, per-fifo TaskArgs pools, single-coordinator idle exit). All runs validated 0 wrong / AllToAll verify OK; medians of 3.

## Environment

- Host `mi-sky-b300`, 8× NVIDIA B300 SXM6 AC (sm_103), NVSwitch NVLink mesh, driver 610.57.04.
- Shim: `experimental/ukernel` (`build/nccl/lib`, ILP=16, TMA_REDUCE=0). Native: system NCCL 2.29.7+cuda13.2, 32 channels.
- nccl-tests MPI build; AllToAll via `bench/alltoall_perf` (ncclAllToAll). Sizes 1M..256M, ranks 2/4/8, n=5 w=1, 3 reps, median OOP busbw.

## Configs

| config | env |
|---|---|
| shim unfused | `UK_CCL_DEV_BLOCKS=32` |
| shim fused | + `FUSE_REDUCE_COPY=1 FUSE_AG_COPY=1 LARGE_TILES=16 TILE_MIN_BYTES=8M IPC_BATCH=16` |
| native | system NCCL |

## AllReduce — busbw GB/s (median of 3)

### S2

| size | shim unfused | shim fused | native | fused/unfused | native/fused |
|---:|---:|---:|---:|---:|---:|
| 1M | 5.2 | 6.5 | 48.6 | 1.26 | 7.46 |
| 4M | 19.8 | 24.7 | 94.9 | 1.24 | 3.84 |
| 16M | 41.8 | 84.7 | 282.7 | 2.03 | 3.34 |
| 64M | 55.9 | 202.2 | 435.3 | 3.62 | 2.15 |
| 256M | 211.9 | 329.6 | 509.8 | 1.55 | 1.55 |

### S4

| size | shim unfused | shim fused | native | fused/unfused | native/fused |
|---:|---:|---:|---:|---:|---:|
| 1M | 3.8 | 4.9 | 61.4 | 1.31 | 12.48 |
| 4M | 12.5 | 19.2 | 139.3 | 1.54 | 7.24 |
| 16M | 37.4 | 63.8 | 242.2 | 1.71 | 3.80 |
| 64M | 68.3 | 178.3 | 537.8 | 2.61 | 3.02 |
| 256M | 251.5 | 308.0 | 596.4 | 1.22 | 1.94 |

### S8

| size | shim unfused | shim fused | native | fused/unfused | native/fused |
|---:|---:|---:|---:|---:|---:|
| 1M | 1.3 | 2.7 | 49.9 | 2.08 | 18.28 |
| 4M | 5.5 | 10.8 | 122.1 | 1.99 | 11.28 |
| 16M | 18.7 | 37.8 | 273.0 | 2.03 | 7.22 |
| 64M | 46.7 | 116.7 | 417.1 | 2.50 | 3.57 |
| 256M | 171.8 | 270.0 | 654.9 | 1.57 | 2.43 |

## AllToAll — busbw GB/s (rank-0 median)

### S2

| size | shim | native | shim/native |
|---:|---:|---:|---:|
| 1M | 5.4 | 14.4 | 0.38 |
| 4M | 21.8 | 51.1 | 0.43 |
| 16M | 39.0 | 121.5 | 0.32 |
| 64M | 50.1 | 237.3 | 0.21 |
| 256M | 237.2 | 306.4 | 0.77 |

### S4

| size | shim | native | shim/native |
|---:|---:|---:|---:|
| 1M | 4.7 | 19.7 | 0.24 |
| 4M | 25.3 | 73.3 | 0.35 |
| 16M | 35.3 | 183.2 | 0.19 |
| 64M | 49.1 | 350.8 | 0.14 |
| 256M | 217.4 | 456.2 | 0.48 |

### S8

| size | shim | native | shim/native |
|---:|---:|---:|---:|
| 1M | 3.6 | 21.2 | 0.17 |
| 4M | 22.8 | 79.5 | 0.29 |
| 16M | 38.1 | 206.9 | 0.18 |
| 64M | 44.4 | 373.9 | 0.12 |
| 256M | 182.0 | 510.2 | 0.36 |

## Factor analysis

### Effect of message size

Both shim configs and native grow with size; the shim's growth is steeper below 64M (dispatch/latency-bound at 1-4M), then flattens. The fused path removes per-hop host transitions, which shows at 16M+ (fused/unfused climbs from ~1.3 at 1M to ~2-3.6 at 64M).

### Effect of rank count

Native AllReduce *gains* with ranks (256M: 510/596/655 GB/s at np2/4/8), scaling with the NVLink fan-out. The shim's busbw *drops* with ranks (fused 256M: 330/308/270), consistent with a ring whose per-hop host/CE cost grows with hop count. Unfused is noisier (212/252/172 — np4 measured above np2) but np8 still falls ~32% below np2. AllToAll shows the same rank penalty on the shim side (256M: 237/217/182) while native scales up (306/456/510).

### Effect of shim blocks (256M unfused AllReduce)

| blocks | np2 | np4 | np8 |
|---:|---:|---:|---:|
| 8 | 175 | 145 | 135 |
| 16 | 211 | 175 | 153 |
| 32 | 212 | 252 | 172 |
| 48 | 233 | 205 | 169 |
| 64 | 226 | 184 | 172 |

Blocks matter only at large sizes: b8 (~131-184) to b16-64 (~150-252) is
a ~1.2-1.7x lift; the biggest jump is b8→b16, and beyond ~32 blocks the
gain is flat or reverses (worker coordination overhead).

### Fused vs unfused

Fusion gives the largest single win on B300: 64M AllReduce is 2.5-3.6x better fused than unfused, 256M ~1.2-1.6x. The fused path keeps 256M shim/native at 0.41-0.65 (np8 worst, np2 best). The remaining gap is native's NVLink P2P/CUMEM transport: the shim's same-node data path is CE/IPC + worker copies by design.

### AllToAll: size and ranks

AllToAll shim rises to a ~44-50 GB/s plateau at 64M, then jumps to
182-237 GB/s at 256M (per-peer copies amortize the CE/worker overhead
only at the largest size), while native grows through the whole range
(306-510 at 256M). The shim/native ratio is therefore worst at 64M
(0.12-0.21) and best at 256M (0.36-0.77). Small sizes (1-4M) are
latency-bound on both sides (shim/native 0.17-0.43).

### Reproducible commands

```bash
cd ~/jinyao/uccl/thirdparty/nccl-tests/build
export LD_LIBRARY_PATH=~/jinyao/uccl/experimental/ukernel/build/nccl/lib:/usr/local/lib:/home/uccl/cuda132/lib64
mpirun --bind-to none -np 8 -x LD_LIBRARY_PATH -x UK_CCL_UNBIND=1 \
  -x UK_CCL_FUSE_REDUCE_COPY=1 -x UK_CCL_FUSE_AG_COPY=1 -x UK_CCL_DEV_BLOCKS=32 \
  -x UK_CCL_LARGE_TILES=16 -x UK_CCL_TILE_MIN_BYTES=8388608 -x UK_CCL_IPC_BATCH=16 \
  ./all_reduce_perf -b 1M -e 256M -f 4 -g 1 -c 1 -n 5 -w 1
```

Raw logs: `/tmp/uk_full_matrix/` on the testbed; runner: `bench/run_full_matrix.sh`.
