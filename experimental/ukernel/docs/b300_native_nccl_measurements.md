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
# kernel name / grid / block
ncu --set basic --replay-mode application -k "regex:ncclDevKernel" \
    --launch-skip 20 --launch-count 1 \
    ./all_reduce_perf -b 256M -e 256M -g 2 -n 30
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
- Threads/block: **not yet confirmed** — ncu replay failed with "Failed
  to save memory for replay"; retry uses `--replay-mode application`
  (see command above). Expect 512 for LL (default `NCCL_NTHREADS`).
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

- Confirm native threads/block via `ncu --replay-mode application`
- `NCCL_NTHREADS=256` at 32 channels (thread-count sensitivity)
- Record native AllGather / ReduceScatter baselines the same way
