# B300 benchmark report — SM-budget revision (2026-09-04)

Shim (ukernel) vs native NCCL on one 8-GPU B300 node, built from
`uk-300` (HEAD `14391f89`; fused-RS thread sync, per-fifo TaskArgs pools,
single-coordinator idle exit). Medians of 3, nccl-tests `-n 10 -w 2`,
validation on, all cells 0 wrong.

## Environment

- Host `mi-sky-b300`, 8× NVIDIA B300 SXM6 AC (sm_103), NVSwitch NVLink
  mesh, driver 610.57.04.
- Shim: `experimental/ukernel` (`build/nccl/lib`, ILP=16, TMA_REDUCE=0).
  Native: system NCCL 2.29.7+cuda13.2.
- nccl-tests MPI build; AllToAll via `bench/alltoall_perf`
  (`ncclAllToAll`). Sizes 1M..256M, ranks 2/4/8, OOP busbw.
- Raw logs: `/tmp/b300_final`, `/tmp/b300_rotation`,
  `/tmp/b300_hostprof`, `/tmp/b300_sweep`, `/tmp/b300_confirm`.

## SM-budget design rule

Shim blocks `b` must be `<=` native coll channels `B` at the same
placement; every cell reports the best measured `b` in that budget
(selected at 256 MiB, ties go to fewer SMs). AllToAll is a pure CE/IPC
path and is reported at **0 worker SMs** (no device-path AllToAll is
measured; a device-path variant exists in the code base but is not part
of the reported system).

Measured budgets: `B = 32` coll channels at S2/S4/S8
(`NCCL_DEBUG=INFO`). Best block counts `b*`:

| collective | np2 | np4 | np8 |
|---|---:|---:|---:|
| AllReduce (fused) | 32 | 32 | 28 |
| ReduceScatter | 32 | 28 | 32 |
| AllGather | CE (0 SM) | CE (0 SM) | CE (0 SM) |
| AllToAll | CE (0 SM) | CE (0 SM) | CE (0 SM) |

AllReduce fused env: `UK_CCL_FUSE_REDUCE_COPY=1 UK_CCL_FUSE_AG_COPY=1
UK_CCL_DEV_BLOCKS=b UK_CCL_LARGE_TILES=16 UK_CCL_TILE_MIN_BYTES=8M
UK_CCL_IPC_BATCH=16`. ReduceScatter runs with `DEV_BLOCKS=b`; standalone
AllGather is data-only and the shim publishes its own shard with a
user-stream copy-engine memcpy (AllToAll-style, 2026-09-04 change), so
the AllGather plan has no worker ops and is measured at **0 worker SMs**
(`DEV_BLOCKS=1` keeps the pre-created worker minimal).

## AllReduce (fused) — busbw GB/s (median of 3), shim vs native 32ch

| size | sh2 | nat2 | sh4 | nat4 | sh8 | nat8 |
|---:|---:|---:|---:|---:|---:|---:|
| 1M | 6.61 | 53.97 | 5.02 | 65.72 | 2.83 | 53.78 |
| 4M | 27.15 | 96.81 | 17.54 | 141.38 | 11.19 | 124.62 |
| 16M | 81.95 | 288.24 | 62.42 | 246.96 | 38.84 | 270.05 |
| 64M | 219.46 | 434.51 | 180.53 | 539.79 | 109.87 | 419.74 |
| 256M | 341.14 | 508.64 | 317.23 | 596.40 | 247.10 | 651.84 |

256M ratio to native: 0.67 / 0.53 / 0.38 (np8 uses 28 blocks, **fewer
SMs than native's 32**).

## ReduceScatter — busbw GB/s (median of 3), shim vs native 32ch

| size | sh2 | nat2 | sh4 | nat4 | sh8 | nat8 |
|---:|---:|---:|---:|---:|---:|---:|
| 1M | 3.36 | 30.75 | 1.97 | 42.12 | 1.18 | 37.33 |
| 4M | 10.77 | 68.19 | 7.42 | 94.29 | 3.63 | 97.21 |
| 16M | 25.04 | 199.37 | 21.20 | 212.30 | 13.77 | 242.56 |
| 64M | 48.56 | 331.61 | 48.09 | 444.99 | 43.34 | 413.19 |
| 256M | 133.54 | 406.92 | 143.82 | 529.19 | 146.46 | 593.09 |

256M ratio to native: 0.33 / 0.27 / 0.25.

## AllGather — busbw GB/s (median of 3), shim CE path (0 SM) vs native 32ch

| size | sh2 | nat2 | sh4 | nat4 | sh8 | nat8 |
|---:|---:|---:|---:|---:|---:|---:|
| 1M | 5.68 | 33.27 | 4.10 | 43.64 | 2.03 | 39.04 |
| 4M | 21.64 | 72.63 | 17.27 | 108.89 | 7.04 | 112.51 |
| 16M | 42.20 | 201.31 | 40.94 | 202.11 | 26.67 | 287.46 |
| 64M | 60.01 | 335.81 | 52.27 | 449.87 | 48.75 | 413.62 |
| 256M | 226.40 | 413.91 | 205.15 | 533.45 | 215.69 | 584.94 |

256M ratio to native: 0.55 / 0.38 / 0.37. Ring hops ride CE/IPC and the
own-shard publish copy is externalized to the user-stream copy engine,
so standalone AllGather uses no worker SMs — and is 10-26% faster at
256M than the earlier worker-local-copy variant (193/172/171 GB/s).

## AllToAll — busbw GB/s (median of 3), shim pure CE path (0 SM)

| size | sh2 | nat2 | sh4 | nat4 | sh8 | nat8 |
|---:|---:|---:|---:|---:|---:|---:|
| 1M | 4.7 | 14.3 | 5.6 | 19.7 | 4.3 | 20.5 |
| 4M | 22.7 | 53.4 | 25.0 | 71.7 | 22.3 | 82.8 |
| 16M | 47.4 | 127.7 | 41.6 | 189.9 | 38.2 | 204.3 |
| 64M | 48.6 | 231.8 | 52.6 | 343.7 | 47.9 | 368.2 |
| 256M | 229.6 | 312.4 | 212.8 | 456.2 | 188.7 | 516.6 |

256M ratio to native: 0.73 / 0.47 / 0.37. The CE path's per-transfer
overhead is worst at 64M (0.13-0.21).

### Rotation A/B at 256M (CE path, 0 SM), medians

| np | rot0 (ascending) | rot1 (default rotated) |
|---:|---:|---:|
| 2 | 264.5 | 211.7 |
| 4 | 215.6 | 192.7 |
| 8 | 187.8 | 206.6 |

NVSwitch is mixed: rotation (default on) hurts np2/np4 (-20%/-11%) and
helps np8 (+10%); no default change is warranted.

## Host orchestration decomposition (1M AllReduce fused@b*, n=30/w=10)

HostProf per-collective medians (µs/collective, stage totals / 40 iters):

| ranks | enq | signal drain | transport drain | device drain |
|---:|---:|---:|---:|---:|
| 2 | 0.2 | 6.8 | 2.1 | 5.7 |
| 4 | 0.4 | 19.5 | 7.7 | 45.0 |
| 8 | 0.6 | 53.5 | 23.3 | 109.4 |

Enqueue/dispatch is negligible; the small-message floor grows with rank
count in the host drain paths (signal + device), not in dispatch.

## Factor analysis

- Message size: both sides grow with size; the shim is dispatch/latency
  bound at 1-4M (native 8-19× ahead on AllReduce at 1M) and flattens
  above 64M.
- Ranks: native AllReduce *gains* with ranks (NVLink fan-out), the shim
  loses busbw as hop count grows; at 256M fused shim drops
  341→317→247 while native rises 509→596→652.
- Blocks: fused AllReduce at 256M keeps rising to the 32-block budget
  (best = b*), so the NVLink large-message regime has no "fewer SMs"
  margin except the np8 cell (28 < 32); AllGather and AllToAll are both
  pure CE paths at 0 worker SMs, which is where the SM savings are
  largest.
- Fusion ablation (fixed b32, prior medians): fused/unfused 256M ratio
  to native 0.65/0.52/0.41 vs 0.42/0.42/0.26; fusion win is largest at
  16-64M (removes per-hop host transitions).

## Reproducible commands

```bash
cd ~/jinyao/uccl/thirdparty/nccl-tests/build
export LD_LIBRARY_PATH=~/jinyao/uccl/experimental/ukernel/build/nccl/lib:/usr/local/lib:/home/uccl/cuda132/lib64
mpirun --bind-to none -np 8 -x LD_LIBRARY_PATH -x UK_CCL_UNBIND=1 \
  -x UK_CCL_DEV_BLOCKS=28 -x UK_CCL_FUSE_REDUCE_COPY=1 \
  -x UK_CCL_FUSE_AG_COPY=1 -x UK_CCL_LARGE_TILES=16 \
  -x UK_CCL_TILE_MIN_BYTES=8388608 -x UK_CCL_IPC_BATCH=16 \
  ./all_reduce_perf -b 1M -e 256M -f 4 -g 1 -c 1 -n 10 -w 2
```

AllToAll (CE, 0 SM): `/tmp/alltoall_perf --bytes=268435456 --iters=10
--warmup=2` under the shim `LD_LIBRARY_PATH` with `UK_CCL_DEV_BLOCKS=1`.
