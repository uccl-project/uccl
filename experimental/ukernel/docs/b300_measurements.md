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
