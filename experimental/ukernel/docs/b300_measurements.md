# B300 benchmark report (single node)

Shim (ukernel) vs native NCCL measurement on one 8-GPU B300 node, using
the same portable plan as the L40S report (`docs/l40s_measurements.md`),
single-node only. All runs validated: AllReduce 0 wrong, AllToAll
verify OK.

## Environment

- Host `mi-sky-b300` (ssh `b300`), 1 node × 8× NVIDIA B300 SXM6 AC
  (275040 MiB each, compute capability 10.3 = sm_103), NVSwitch
  (NVLink P2P between all pairs), driver 610.57.04.
- No system CUDA toolkit (`/usr/local/cuda` absent). Toolchain: conda
  CUDA 13.2 prefix merged into `~/cuda132`
  (`bin`/`include`/`lib64` symlink farm over
  `~/shuangma/mkernel_project/b300_tools/cuda13.2`); nvcc 12.8 cannot
  target sm_103 (fails with "Unsupported gpu architecture"), so 13.2 is
  required.
- gdrcopy: userspace `libgdrapi.so.2` in `/usr/local/lib`, kernel module
  `gdrdrv` loaded (`/dev/gdrdrv`); the shim's worker fifos fail fast
  without it.
- Shim: `experimental/ukernel` build (`build/nccl/lib`, compiled for
  sm_103, `REDUCE_ILP=16`, `TMA_REDUCE=0`); native: system NCCL
  **2.29.7+cuda13.2** (`/usr/lib/x86_64-linux-gnu/libnccl.so.2.29.7`).
- nccl-tests under `thirdparty/nccl-tests/build` (HPC OpenMPI
  `/usr/mpi/gcc/openmpi-4.1.9a1`); AllToAll via
  `experimental/ukernel/bench/alltoall_perf` (`ncclAllToAll`).
- Runs used `mpirun --bind-to none`, shim env
  `LD_LIBRARY_PATH=.../build/nccl/lib:/usr/local/lib:~/cuda132/lib64`
  (cudart 13 lives only in the conda tree), native env
  `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/lib:~/cuda132/lib64`.
  GPUs were idle (0 MiB) for every run; the runner waited out
  co-tenant jobs between configurations.

## Standard benchmark plan (single-node slice)

| axis | values |
|---|---|
| collectives | AllReduce (ring, fp32 sum, out-of-place), AllToAll (out-of-place) |
| sizes per rank | 1M, 4M, 16M, 64M, 256M |
| ranks | 2, 4, 8 (1 GPU/rank) |
| shim blocks | ladder 1/8/32 + **blocks = native coll channels (32)** |
| native | default, channel count recorded per config |

Notes:

- **Native coll channels = 32 for every rank count.** NCCL 2.29 prints
  per-channel topology lines (`Channel N/0 : ...`), not an
  `nchannels` summary; the count was taken as max channel id + 1
  (verified 2/4/8 ranks all use 32). The shim's channels-matched column
  is therefore blocks=32, same value as the ladder top.
- AllReduce: `-n 10 -w 2`, out-of-place busbw. AllToAll:
  `--iters=5 --warmup=2`, rank-0 busbw, same-node verify on.
- All runs 0 wrong; AllToAll verify OK on every config.

Reproducible command (8 ranks, shim):

```bash
cd ~/jinyao/uccl/thirdparty/nccl-tests/build
export LD_LIBRARY_PATH=~/jinyao/uccl/experimental/ukernel/build/nccl/lib:/usr/local/lib:/home/uccl/cuda132/lib64
mpirun --bind-to none -np 8 -x LD_LIBRARY_PATH -x UK_CCL_UNBIND=1 \
  -x UK_CCL_DEV_BLOCKS=32 ./all_reduce_perf -b 1M -e 256M -f 4 -g 1 -c 1 -n 10 -w 2
# AllToAll
mpirun --bind-to none -np 8 -x LD_LIBRARY_PATH -x UK_CCL_UNBIND=1 \
  -x UK_CCL_DEV_BLOCKS=8 /tmp/alltoall_perf --bytes=268435456 --iters=5 --warmup=2
```

The full matrix is automated by
`experimental/ukernel/bench/measure_single_node.sh` (logs in
`/tmp/uk_single_node/logs`), which waits for idle GPUs and aborts on
any wrong data.

## AllReduce — busbw GB/s

### S2 — 2 ranks

| size | shim b1 | shim b8 | shim b32 | shim b=ch(32) | native (32ch) |
|---:|---:|---:|---:|---:|---:|
| 1M | 4.14 | 5.11 | 5.44 | 5.44 | 54.59 |
| 4M | 13.38 | 20.20 | 18.06 | 18.06 | 98.40 |
| 16M | 26.35 | 42.31 | 40.66 | 40.66 | 286.89 |
| 64M | 34.41 | 60.14 | 52.24 | 52.24 | 431.41 |
| 256M | 29.77 | 175.39 | 211.56 | 211.56 | 508.99 |

### S4 — 4 ranks

| size | shim b1 | shim b8 | shim b32 | shim b=ch(32) | native (32ch) |
|---:|---:|---:|---:|---:|---:|
| 1M | 2.90 | 3.19 | 3.02 | 3.02 | 63.37 |
| 4M | 8.02 | 12.26 | 12.90 | 12.90 | 141.11 |
| 16M | 23.38 | 30.59 | 31.13 | 31.13 | 248.49 |
| 64M | 31.96 | 48.62 | 52.03 | 52.03 | 537.26 |
| 256M | 30.68 | 144.95 | 197.20 | 197.20 | 599.01 |

### S8 — 8 ranks

| size | shim b1 | shim b8 | shim b32 | shim b=ch(32) | native (32ch) |
|---:|---:|---:|---:|---:|---:|
| 1M | 1.20 | 1.17 | 1.16 | 1.16 | 53.97 |
| 4M | 4.24 | 4.04 | 4.57 | 4.57 | 124.31 |
| 16M | 13.55 | 16.61 | 18.65 | 18.65 | 268.39 |
| 64M | 28.23 | 39.26 | 40.41 | 40.41 | 422.20 |
| 256M | 34.23 | 124.38 | 147.46 | 147.46 | 653.27 |

## AllToAll — busbw GB/s (rank-0)

### S2 — 2 ranks

| size | shim (b8) | native (32ch) |
|---:|---:|---:|
| 1M | 4.1 | 13.6 |
| 4M | 17.9 | 54.8 |
| 16M | 36.0 | 140.3 |
| 64M | 77.4 | 243.8 |
| 256M | 181.5 | 315.7 |

### S4 — 4 ranks

| size | shim (b8) | native (32ch) |
|---:|---:|---:|
| 1M | 5.9 | 20.4 |
| 4M | 23.7 | 71.1 |
| 16M | 32.7 | 190.8 |
| 64M | 49.9 | 343.6 |
| 256M | 186.6 | 443.6 |

### S8 — 8 ranks

| size | shim (b8) | native (32ch) |
|---:|---:|---:|
| 1M | 3.1 | 21.8 |
| 4M | 22.4 | 81.3 |
| 16M | 37.1 | 203.3 |
| 64M | 49.9 | 373.3 |
| 256M | 183.4 | 517.7 |

## Analysis

- **Native dominates on B300 — unlike L40S.** The B300 node has full
  NVSwitch/NVLink P2P (native logs show `via P2P/CUMEM` on every
  channel), so native AllReduce reaches 431-653 GB/s at 64M+ and native
  AllToAll 244-518 GB/s. The shim's same-node data path is CE/IPC by
  design (keeps worker SMs out of the data path), which caps out around
  34-60 GB/s for the reduce path at 16-64M and ~150-210 GB/s for
  AllToAll at 256M. The 2-4× gap is the NVLink P2P vs copy-engine
  difference, not latency or scheduling.
- **blocks matter at 256M for shim AllReduce.** b1: 30-34 GB/s vs b32:
  147-211 GB/s. On L40S blocks were neutral at 16M+; on B300 the
  single-block reduce saturates at ~30-35 GB/s, and only the multi-block
  worker (8-32 blocks) can approach the CE ceiling. 1-4M sizes are still
  host-dispatch-latency bound (b1 ≈ b8 ≈ b32, native 10-20× ahead).
- **Shim AllToAll scales with size, not ranks**: 256M stays ~180-187
  GB/s from 2 to 8 ranks; small sizes (1M) are latency-bound (~3-6
  GB/s). Native AllToAll scales with both (13.6 → 517.7 GB/s at 256M
  across 2→8 ranks), tracking the fan-out-limited P2P graph.
- **b=channels = b32 by construction**: native uses 32 channels for all
  rank counts, so the channels-matched column coincides with the ladder
  top. The knob spans the same parallelism space as native's channels.

## Blocks-ladder saturation sweep (AllReduce)

`bench/sweep_ar_blocks.sh` reruns the same AllReduce matrix over a fine
blocks ladder (2,4,...,40,44,48,52,56,60,64; the worker caps blocks at
64 — the 64-bit exit-rendezvous mask — and at the SM count) for ranks
2/4/8, plus a native baseline. Single run per config; every row checked
0 wrong (one known exception below). Full CSV:
`/tmp/uk_single_node/logs/sweep_ar_blocks.csv` on the testbed.

### Saturation summary (first blocks reaching ≥95% of the observed max)

| ranks | size | shim max GB/s | 95%-sat blocks | native GB/s | max/native |
|---:|---:|---:|---:|---:|---:|
| 2 | 16M | 61.8 | 28 | 284.3 | 0.22 |
| 2 | 64M | 93.2 | 24 | 432.2 | 0.22 |
| 2 | 256M | 309.4 | 28 | 508.8 | 0.61 |
| 4 | 16M | 34.6 | 8 | 243.8 | 0.14 |
| 4 | 64M | 66.0 | 60 | 539.7 | 0.12 |
| 4 | 256M | 236.2 | 60 | 598.9 | 0.39 |
| 8 | 16M | 20.8 | 18 | 267.1 | 0.08 |
| 8 | 64M | 48.6 | 14 | 421.1 | 0.12 |
| 8 | 256M | 183.0 | 48 | 654.4 | 0.28 |

### 256M curve (representative points, GB/s)

| blocks | np2 | np4 | np8 |
|---:|---:|---:|---:|
| 2 | 57 | 57 | 60 |
| 8 | 183 | 147 | 131 |
| 16 | 210 | 178 | 153 |
| 20 | 207 | 202 | 158 |
| 24 | 289 | 194 | 151 |
| 28 | 306 | 199 | 162 |
| 32 | 267 | 200 | 172 |
| 40 | 256 | 186 | 170 |
| 48 | 183 | 197 | 177 |
| 56 | 231 | 181 | 175 |
| 64 | 211 | 199 | 183 |

Notes:

- **No block count reaches native on B300.** The best shim/native ratios
  at 256M are 0.61 (2 ranks) / 0.39 (4) / 0.28 (8); native's NVLink P2P
  bandwidth is not reachable through the CE/IPC data path. The L40S
  "shim wins at 16M+" story does not carry to an NVSwitch node.
- **blocks matter most at 256M** (b1 30-34 → b28+ 180-310 GB/s); at
  64M and below the ladder is mostly neutral, and 1-4M stays
  latency-bound. So the shim's sweet spot is 24-32 blocks on B300 (the
  auto default is 32), not 64 — 64 buys nothing but noise.
- **Noise caveat:** single-run values at 256M fluctuate ±20-30%
  run-to-run (e.g. np2 b32 measured 211 GB/s in the main suite vs 267
  here), so the saturation block is indicative, not exact; use medians
  of ≥3 runs for a paper figure.
- **Known correctness bug found by the sweep:** np=8 + 1M + blocks≥33
  produces flaky wrong sums (bad elements vary per run, up to thousands;
  `bench/ar_check.cu` shows missing peer contributions). It is a worker
  multi-block barrier/completion race that only manifests at the
  smallest size — 2M and up are clean at every block count. The sweep
  records these rows as wrong and continues; the race needs a separate
  debugging pass on the worker barrier.

## AllToAll: rotation A/B and CE contention on B300

The L40S incast fix rotates each rank's per-peer send order (Latin
square) so the synchronized collective start spreads across
destinations. A `UK_CCL_A2A_ROTATE` knob (default 1) was added to
`src/ccl` to A/B it on B300 (medians of 3, rank-0 busbw GB/s):

| ranks | size | rotate=1 | rotate=0 (ascending) |
|---:|---:|---:|---:|
| 2 | 16M | 38.4 | 35.4 |
| 2 | 64M | 61.0 | 54.8 |
| 2 | 256M | 195.6 | 218.0 |
| 4 | 16M | 35.8 | 37.1 |
| 4 | 64M | 47.6 | 46.7 |
| 4 | 256M | **186.7** | **228.9** |
| 8 | 16M | 38.4 | 35.2 |
| 8 | 64M | 44.7 | 46.7 |
| 8 | 256M | 185.6 | 188.3 |

**Rotation does not help on B300; at np4/256M ascending is ~23% faster**
(228.9 vs 186.7, consistent across all three reps). Other configs are
within noise. The L40S incast fix targets a PCIe CE/ingress arbitration
problem that NVSwitch handles differently (the framework doc already
flagged "NVSwitch measured the opposite CE-concurrency behavior"); the
knob should become link-kind-aware or default off on NVSwitch nodes.

`bench/ce_contention` microbench on B300: synchronized vs staggered
start at 64MB/copy shows only ~1% penalty, but at 256MB/copy the sync
penalty is 2-3x for some ranks (r3/r7 per-copy 82 → 30 GB/s). So CE
synchronized-start contention exists on B300 at large copies, yet it
does not translate into a benefit for the rotation ordering in
`ncclAllToAll`.

## Fused RS/AG AllReduce at b32 (fused@b32 matrix, 2026-09-01)

The fused reduce+copy path (`UK_CCL_FUSE_REDUCE_COPY=1` +
`UK_CCL_FUSE_AG_COPY=1`) at `UK_CCL_DEV_BLOCKS=32`, with the fused-path
tile config (LT=16 TM=8M IB=16, the Appendix C optimum), across ranks
2/4/8 and sizes 1M..256M (f4), n=10 w=2, OOP/In-place busbw. Native =
system NCCL 2.29.7, 32 channels = 32 SMs (native's exact SM count is its
channel count), so fused@b32 is the SM-matched comparison. All runs 0
wrong (this matrix previously could not be completed — see the bug
below).

### AllReduce — busbw GB/s (fused@b32 tuned vs native)

| size | fused np2 | native np2 | fused np4 | native np4 | fused np8 | native np8 |
|---:|---:|---:|---:|---:|---:|---:|
| 1M | 6.8 | 58.3 | 5.1 | 38.3 | 2.9 | 54.1 |
| 4M | 26.0 | 94.6 | 19.1 | 97.5 | 11.4 | 123.7 |
| 16M | 85.1 | 288.0 | 65.3 | 205.8 | 38.8 | 268.4 |
| 64M | 212.2 | 432.4 | 186.5 | 479.8 | 114.7 | 419.6 |
| 256M | 336.1 | 508.7 | 311.8 | 597.0 | 286.5 | 651.7 |

256M medians of 3 (busbw GB/s): fused 345.0 / 313.9 / 269.5 vs native
509.7 / 598.8 / 651.7 for np2/4/8. Fusion keeps the 256M shim/native
ratio near 0.66-0.52 from 2 to 8 ranks (vs the unfused b32 0.29-0.42 in
the main matrix above), and 64M+ now scales with rank count instead of
flatlining — the fused path removes the per-hop host transitions that
capped the unfused ring.

Note: the np4 native 256M cell above is a re-run confirmation (~597,
3×); the in-matrix run measured 516.6 while a co-tenant job had grabbed
the GPUs mid-run. Other cells ran on idle GPUs.

### Correctness bug found and fixed (blocks the pre-fix matrix)

Pre-fix, fused reduce+copy produced **wrong sums at np≥4** in a
size/block-dependent window: np4 + default 1M tiles wrong at ~17M up,
np4 + tuned tiles clean, np8 + tuned wrong at 16M/32M (deterministic),
np8 + b64 wrong even at 64M/256M. Fused RS alone reproduced; fused AG
alone was clean; `UK_CCL_DEVICE_FLAGS=0` did not change it (not the
flag protocol). `ar_check` showed missing peer contributions (np4 dev =
-1/4 × expected = one rank's share absent; all ranks wrong on the same
indices; segments 512B-aligned; totals varied run to run → race, not
determinism). Same signature on L40S but ~19% of elements wrong vs ~1%
on B300 (PCIe timing window is wider).

Root cause: in `run_reduce`, the reduce writes the accumulator with
16B `TypedVec` stores strided by `nthread`, then the forward copy reads
it back with 32B `Vec` loads that span 8 floats written by *different*
threads — with no `__syncthreads()` between reduce and copy, a fast
thread forwarded the pre-reduce accumulator, so the next ring rank
missed this peer's contribution. Fix: block barrier between the reduce
and the forward copy in both the fast and generic dtype paths
(`reduce_dispatch.h`). Post-fix: np4/64M, np4/256M, np8/16M, np8/32M,
np8/64M, np8/256M all 0 wrong on B300; L40S np4/64M default and tuned
configs 0 wrong (fused RS-only, AG-only, and RS+AG).

## Testbed notes (B300 bring-up)

1. **CUDA 13.2 is required for sm_103.** The conda nvcc 12.8 on this box
   rejects `compute_103` (supports up to sm_101); the build works with
   the existing conda 13.2 prefix (headers under
   `targets/x86_64-linux/include`, libs under `lib`). A merged
   `~/cuda132` tree with `bin`/`include`/`lib64` satisfies the
   Makefile's standard layout.
2. **cudart 13 lives only in the conda tree.** Neither the shim nor the
   nccl-tests binaries find `libcudart.so.13` by default; the LD paths
   in the report include `~/cuda132/lib64`.
3. **gdrcopy kernel module is `gdrdrv`**, not `gdrcopy` (easy to miss
   with `lsmod | grep gdrcopy`). The shim worker fifos fail fast without
   it.
4. **NCCL 2.29 prints channels as topology lines**, e.g.
   `Channel 28/0 : 1[1] -> 0[0] via P2P/CUMEM`; there is no
   `nchannels` summary (that appeared in 2.31). Channel count = max
   `Channel N/` id + 1.
5. The measurement script parses the `#wrong` column from the nccl-tests
   header (2.29 uses a 13-column `busbw #wrong` layout; 2.31 uses the
   15-column `error wrong` layout) so it runs unchanged on the L40S pair.
