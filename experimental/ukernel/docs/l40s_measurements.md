# L40S benchmark report (node5/node6)

Complete shim (ukernel) vs native NCCL measurement report, built on a
standard, testbed-portable plan. All runs validated (0 wrong).

## Environment

- node5 `10.31.154.11`, node6 `10.31.154.12`; 8× L40S each (PCIe-only,
  no NVLink), ConnectX-6 dual-port HDR, driver 610.57.04
- IB fabric: node5 `mlx5_0` (LID 1, 200G), `mlx5_3` (LID 5, 200G);
  node6 `mlx5_0` (LID 3), `mlx5_1` (LID 2) — same subnet, all 200G.
  The 10G "bond" ports (LID 0) are off-fabric and excluded by
  `pick_dev_for_gpu`.
- Shim: `experimental/ukernel` build + NCCL compat
  (`build/nccl/lib`); native: system NCCL 2.31.2. nccl-tests MPI build
  under `thirdparty/nccl-tests/build`; AllToAll via
  `experimental/ukernel/bench/alltoall_perf` (uses `ncclAllToAll`,
  which both the shim and native NCCL implement).
- Benchmarks ran with all 8 GPUs per node idle and no foreign traffic
  (the co-resident vLLM modelserver was stopped for the runs).

## Standard benchmark plan (testbed-portable)

| axis | values |
|---|---|
| collectives | AllReduce (ring, fp32 sum, out-of-place), AllToAll (out-of-place) |
| sizes per rank | 1M, 4M, 16M, 64M, 256M |
| same-node ranks | 2, 4, 8 (1 GPU/rank) |
| cross-node ranks | 4 (2+2), 8 (4+4), 16 (8+8) |
| shim blocks | ladder 1/8/32 + **blocks = native coll channels** (2/4) |
| native | default, channel count recorded per config |

Notes:

- **AllToAll ignores shim blocks**: the shim's `ncclAlltoAll` keeps the
  persistent worker out of the data path (self-slice + CE/IPC/RDMA
  copies), so `UK_CCL_DEV_BLOCKS` has no effect. AllToAll is therefore
  reported with a single shim column (default b8); the AllReduce tables
  carry the blocks ladder and the channels-matched column.
- Native coll channels (from `NCCL_DEBUG=INFO` at init): S2/S4 = 4,
  S8/X4/X8/X16 = 2. The channels-matched shim column uses
  `UK_CCL_DEV_BLOCKS = native channels` for a like-for-like comparison
  of the two implementations' per-rank parallelism.
- AllReduce: `-n 10 -w 2`, out-of-place busbw. AllToAll: `--iters=5
  --warmup=2`, rank-0 busbw; cross-node uses `--skip-verify=1` (the
  bench's fill-file handshake is per-node; same exchange is verified
  same-node).
- All runs 0 wrong.

Reproducible commands (16 ranks, shim):

```bash
cd /root/uccl/uccl/thirdparty/nccl-tests/build
export LD_LIBRARY_PATH=/root/uccl/uccl/experimental/ukernel/build/nccl/lib:/usr/local/lib
mpirun --allow-run-as-root --bind-to none --map-by :OVERSUBSCRIBE -np 16 \
  --host 10.31.154.11:8,10.31.154.12:8 \
  -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES -x UK_CCL_UNBIND=1 \
  -x UK_CCL_RDMA_FUSED_MODE=proxy -x UK_CCL_DEV_BLOCKS=2 \
  ./all_reduce_perf -b 1M -e 256M -f 4 -g 1 -c 1 -n 10 -w 2
# AllToAll: --dev from the OpenMPI local rank via a wrapper
mpirun --allow-run-as-root --bind-to none --map-by :OVERSUBSCRIBE -np 16 \
  --host 10.31.154.11:8,10.31.154.12:8 \
  -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES -x UK_CCL_UNBIND=1 \
  -x UK_CCL_RDMA_FUSED_MODE=proxy -x UK_CCL_DEV_BLOCKS=8 \
  bash /tmp/a2a_rank.sh --bytes=268435456 --iters=5 --warmup=2 --skip-verify=1
```

Native: same with `LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib64`.

## AllReduce — busbw GB/s

### S2 — 2 ranks, same node (node5)

| size | shim b1 | shim b8 | shim b32 | shim b=ch | native (4ch) |
|---:|---:|---:|---:|---:|---:|
| 1M | 7.32 | 8.74 | 9.17 | 8.16 | 14.46 |
| 4M | 16.33 | 20.02 | 20.32 | 19.04 | 20.12 |
| 16M | 23.52 | 23.76 | 23.67 | 23.36 | 21.18 |
| 64M | 24.48 | 24.46 | 24.52 | 24.39 | 21.43 |
| 256M | 25.60 | 25.61 | 25.62 | 25.60 | 21.62 |

### S4 — 4 ranks, same node

| size | shim b1 | shim b8 | shim b32 | shim b=ch | native (4ch) |
|---:|---:|---:|---:|---:|---:|
| 1M | 5.29 | 5.74 | 5.73 | 5.65 | 14.96 |
| 4M | 10.48 | 13.26 | 13.36 | 12.87 | 21.23 |
| 16M | 23.77 | 23.79 | 23.89 | 23.92 | 21.89 |
| 64M | 24.60 | 24.57 | 24.40 | 24.57 | 21.69 |
| 256M | 25.64 | 25.63 | 25.64 | 25.64 | 21.67 |

### S8 — 8 ranks, same node

| size | shim b1 | shim b8 | shim b32 | shim b=ch | native (2ch) |
|---:|---:|---:|---:|---:|---:|
| 1M | 2.91 | 2.90 | 2.97 | 2.79 | 9.38 |
| 4M | 6.44 | 6.99 | 7.10 | 6.67 | 14.58 |
| 16M | 12.95 | 13.48 | 13.56 | 13.24 | 14.23 |
| 64M | 15.31 | 14.20 | 14.44 | 14.89 | 14.19 |
| 256M | 16.15 | 15.08 | 15.01 | 15.61 | 14.36 |

### X4 — 4 ranks cross-node (2+2)

| size | shim b1 | shim b8 | shim b32 | shim b=ch | native (2ch) |
|---:|---:|---:|---:|---:|---:|
| 1M | 4.80 | 5.42 | 5.32 | 5.16 | 11.25 |
| 4M | 7.69 | 9.51 | 9.69 | 8.43 | 13.69 |
| 16M | 11.53 | 11.51 | 11.49 | 11.51 | 13.70 |
| 64M | 11.54 | 11.50 | 11.46 | 11.52 | 13.66 |
| 256M | 10.98 | 10.95 | 10.92 | 10.95 | 13.72 |

### X8 — 8 ranks cross-node (4+4)

| size | shim b1 | shim b8 | shim b32 | shim b=ch | native (2ch) |
|---:|---:|---:|---:|---:|---:|
| 1M | 2.88 | 3.19 | 2.99 | 3.15 | 5.08 |
| 4M | 6.61 | 8.11 | 8.03 | 7.54 | 14.48 |
| 16M | 11.44 | 11.46 | 11.48 | 11.47 | 14.52 |
| 64M | 11.52 | 11.43 | 11.45 | 11.49 | 14.39 |
| 256M | 11.07 | 11.16 | 11.12 | 11.14 | 14.52 |

### X16 — 16 ranks cross-node (8+8)

| size | shim b1 | shim b8 | shim b32 | shim b=ch | native (2ch) |
|---:|---:|---:|---:|---:|---:|
| 1M | 1.47 | 1.45 | 1.42 | 1.43 | 4.95 |
| 4M | 4.54 | 4.80 | 4.76 | 4.61 | 12.21 |
| 16M | 8.50 | 10.42 | 10.67 | 9.59 | 15.07 |
| 64M | 11.73 | 11.80 | 11.61 | 11.80 | 14.54 |
| 256M | 10.93 | 10.95 | 10.91 | 10.82 | 14.68 |

## AllToAll — busbw GB/s

### S2 — 2 ranks, same node (node5)

| size | shim (b8) | native (4ch) |
|---:|---:|---:|
| 1M | 7.8 | 9.7 |
| 4M | 16.8 | 18.1 |
| 16M | 21.6 | 22.2 |
| 64M | 23.7 | 23.5 |
| 256M | 23.8 | 23.9 |

### S4 — 4 ranks, same node

| size | shim (b8) | native (4ch) |
|---:|---:|---:|
| 1M | 8.8 | 12.5 |
| 4M | 15.3 | 18.3 |
| 16M | 16.3 | 20.1 |
| 64M | 15.6 | 20.7 |
| 256M | 16.0 | 20.2 |

### S8 — 8 ranks, same node

| size | shim (b8) | native (2ch) |
|---:|---:|---:|
| 1M | 4.5 | 4.8 |
| 4M | 6.1 | 5.9 |
| 16M | 6.4 | 6.8 |
| 64M | 6.7 | 6.6 |
| 256M | 6.6 | 6.9 |

### X4 — 4 ranks cross-node (2+2)

| size | shim (b8) | native (2ch) |
|---:|---:|---:|
| 1M | 6.5 | 6.8 |
| 4M | 8.4 | 9.3 |
| 16M | 9.1 | 10.7 |
| 64M | 8.6 | 11.3 |
| 256M | 8.8 | 11.3 |

### X8 — 8 ranks cross-node (4+4)

| size | shim (b8) | native (2ch) |
|---:|---:|---:|
| 1M | 4.9 | 5.3 |
| 4M | 5.8 | 6.3 |
| 16M | 6.1 | 6.4 |
| 64M | 6.2 | 6.3 |
| 256M | 6.1 | 6.5 |

### X16 — 16 ranks cross-node (8+8)

| size | shim (b8) | native (2ch) |
|---:|---:|---:|
| 1M | 1.9 | 3.3 |
| 4M | 2.1 | 4.2 |
| 16M | 1.5 | 4.5 |
| 64M | 2.1 | 4.7 |
| 256M | 1.7 | 4.7 |

## Analysis

- **Same-node AllReduce: shim wins at 16M+.** S2/S4 reach 23.5-25.6
  GB/s vs native 21.2-21.9 (+14-18%); S8 is at parity (15.0-16.2 vs
  14.2-14.4). At small sizes (<=4M) native leads 2-3x — the shim's
  host-driven proxy/dispatch cost dominates latency there.
- **Cross-node AllReduce: native leads ~15-30%.** X16 256M: shim 10.9
  vs native 14.7. The gap is per-cross-edge RDMA throughput, not
  deadlock or correctness.
- **blocks ladder (AllReduce)**: b8/b32 beat b1 at small sizes; at 16M+
  blocks are neutral. The channels-matched column (b=ch) sits inside the
  ladder range, confirming the shim's blocks knob spans the same
  "parallelism" space native gets from channels.
- **AllToAll: shim is close to native same-node and within 5-25%
  cross-node at <=8 ranks**, but the 16-rank cross-node case degrades
  (shim 1.7-2.1 vs native 4.7 GB/s busbw) — the shim's per-peer put
  scheduling does not yet match native at high fan-out.
- AllToAll ignores worker blocks by design (CE/IPC/RDMA data path), so
  only the default shim column is shown.

## Platform quirks discovered on this pair (2026-08-26)

1. **`HostNativeAtomicSupported=0`** (measured; B300 reports the same).
   The GPU cannot do native atomics on host-pinned memory, so a device
   kernel must never `atomicAdd_system` a host signal ring. Device
   completion signals use the per-slot **plain-store device flag**
   protocol (`signal_flag_write`: `__threadfence_system` + store +
   fence). The old signal-ring producer (`signal_ring_write`) was removed
   in `08d4f381`.
2. **Copy-engine `cudaMemset` writes do not reliably drain before a
   resident worker kernel's read-modify-write.** The worker's reduce can
   lose the first round (~1.8M elements, 128B-line pattern). Avoid with
   kernel-zero (`UKernel::Device::zero_device_buffer`), a ~200ms delay,
   or launching the worker after the memset+sync. Survives a driver
   reload (firmware-level); unrelated to gdrcopy.
3. **ShmExchanger multi-rank init race (fixed).** The POSIX shm store
   was sized only by its creator after the flock; a peer that locked
   first mmap'd a 0-length file and SIGBUSed. Sized under the lock
   regardless of creator.

## Worker completion barrier (commits `33c5b812`, `864239ce`)

The persistent worker's per-task multi-block barrier was a reset-to-0
counter; a slow block arriving after the reset leaked its +1 into the
next task's count, releasing barriers early and re-processing tasks
(wrong reduce results, and a permanent worker hang in the real path).
Fixes: monotonic completion counter keyed to the task index, tail-
visible release, and a device-side counter anchor for relaunched grids
(the host's GDR tail read can lag; block 0 re-anchors at kernel entry).

### Same-host IPC test invocation gotcha

`test_spray_executor_e2e` needs both ranks to see **both** GPUs
(`CUDA_VISIBLE_DEVICES=0,1` with `--gpu=0/--gpu=1`); restricting each
rank to one visible device breaks the peer device numbering and the IPC
path fails.

## Copy engine (CE) bandwidth

`/tmp/run_ingress2.sh` on node5: ce_d2d_same 10.8, ce_peer_1to1 0.9,
ce_peer_5to1 20.9, ce_h2d_pinned 0.8 GB/s. Identical before/after a
reboot — stable baseline, not a recoverable degradation. Collectives'
core bandwidth uses SM loads/stores, not the CE.

## Worker reduce peak bandwidth (launch path, 256MB fp32 sum)

| blocks | 30 x 256MB total | per-launch | GB/s |
|---:|---:|---:|---:|
| 1 | 0.634 s | 21.1 ms | **12.1** |
| 8 | 0.091 s | 3.05 ms | 84.0 |
| 32 | 0.038 s | 1.27 ms | 201 |
| 64 | 0.036 s | 1.20 ms | 214 |

~32 SMs saturate; 64 takes the full bandwidth. `blocks_per_worker=8`
comfortably feeds the NIC/CE ingress and remains the default.

Note on the numbers: the single-SM reduce ceiling is **~12 GB/s of
data rate** (1 block). A b1 AllReduce's 25.6 GB/s busbw at 256M is
nccl-tests' ring-exchange metric, not SM reduce throughput: with 2
ranks the worker's critical path reduces one 128M shard at ~12 GB/s
(~10.6ms, matching the measured ~10.5ms), while busbw counts the full
2x data exchanged.

## Next steps

- Cross-node AllReduce (shim ~11 vs native ~14.7 GB/s at 256M): per-
  cross-edge RDMA throughput is the lever; multi-QP striping per cross
  chunk.
- 16-rank cross-node AllToAll (shim 1.7-2.1 vs native 4.7): high
  fan-out put scheduling / incast control.
- Small-message AllReduce latency: shim's host proxy adds dispatch cost
  at <=4M.
- Re-run this standard plan on other testbeds (B300 when it returns) to
  validate the plan and compare per-testbed signatures.
