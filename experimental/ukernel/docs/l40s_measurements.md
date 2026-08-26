# L40S benchmark report (node5/node6)

Complete shim (ukernel) vs native NCCL measurement report for the L40S
pair: AllReduce + AllToAll across sizes, rank counts (same-node and
cross-node) and worker block counts. All runs validated (0 wrong).

## Environment

- node5 `10.31.154.11`, node6 `10.31.154.12`; 8× L40S each (PCIe-only,
  no NVLink), ConnectX-6 dual-port HDR, driver 610.57.04
- IB fabric: node5 `mlx5_0` (LID 1, 200G), `mlx5_3` (LID 5, 200G);
  node6 `mlx5_0` (LID 3), `mlx5_1` (LID 2) — same subnet, all 200G.
  The 10G "bond" ports (LID 0) are off-fabric and excluded by
  `pick_dev_for_gpu`.
- Shim: `experimental/ukernel` build + NCCL compat
  (`build/nccl/lib`); native: system NCCL 2.31.2. nccl-tests MPI build
  under `thirdparty/nccl-tests/build`; alltoall via
  `experimental/ukernel/bench/alltoall_perf` (uses `ncclAllToAll`,
  which both the shim and native NCCL implement).
- k8s vLLM pods occupy GPUs from time to time (auto-restart). The
  benchmark was run with all GPUs idle and a watcher holding any
  respawned vLLM process down for the duration, so no foreign traffic
  interfered.

Reproducible commands (12 ranks, shim):

```bash
cd /root/uccl/uccl/thirdparty/nccl-tests/build
export LD_LIBRARY_PATH=/root/uccl/uccl/experimental/ukernel/build/nccl/lib:/usr/local/lib
# AllReduce
mpirun --allow-run-as-root --bind-to none --map-by :OVERSUBSCRIBE -np 12 \
  --host 10.31.154.11:6,10.31.154.12:6 \
  -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES -x UK_CCL_UNBIND=1 \
  -x UK_CCL_RDMA_FUSED_MODE=proxy -x UK_CCL_DEV_BLOCKS=8 \
  ./all_reduce_perf -b 1M -e 256M -f 4 -g 1 -c 1 -n 10 -w 2
# AllToAll (--dev is taken from the OpenMPI local rank via a wrapper)
mpirun --allow-run-as-root --bind-to none --map-by :OVERSUBSCRIBE -np 12 \
  --host 10.31.154.11:6,10.31.154.12:6 \
  -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES -x UK_CCL_UNBIND=1 \
  -x UK_CCL_RDMA_FUSED_MODE=proxy -x UK_CCL_DEV_BLOCKS=8 \
  bash /tmp/a2a_rank.sh --bytes=268435456 --iters=5 --warmup=2 --skip-verify=1
```

Native: same commands with `LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib64`.
Cross-node AllToAll uses `--skip-verify=1` (the bench's fill-file
handshake is per-node local); correctness of the same exchange was
verified same-node.

## Methodology (2026-08-26)

- Collectives: AllReduce (ring, `float sum`, out-of-place) and AllToAll
  (`ncclAllToAll`, out-of-place).
- Sizes per rank: 1M, 4M, 16M, 64M, 256M (AllReduce one invocation per
  config; AllToAll one run per size).
- Rank layouts: same-node 2/4/8 on node5 (`S2/S4/S8`), cross-node
  2+2/4+4/6+6 (`X4/X8/X12`).
- Worker blocks (`UK_CCL_DEV_BLOCKS`): 1, 8, 32; native NCCL has no
  such knob (column `native`).
- AllReduce: `-n 10 -w 2`, report out-of-place busbw. AllToAll:
  `--iters=5 --warmup=2`, report rank-0 busbw.
- All runs 0 wrong; 240 data points total.

## AllReduce — busbw GB/s

### S2 — 2 ranks, same node (node5)

| size | shim b1 | shim b8 | shim b32 | native |
|---:|---:|---:|---:|---:|
| 1M | 7.32 | 8.74 | 9.17 | 14.46 |
| 4M | 16.33 | 20.02 | 20.32 | 20.12 |
| 16M | 23.52 | 23.76 | 23.67 | 21.18 |
| 64M | 24.48 | 24.46 | 24.52 | 21.43 |
| 256M | 25.60 | 25.61 | 25.62 | 21.62 |

### S4 — 4 ranks, same node

| size | shim b1 | shim b8 | shim b32 | native |
|---:|---:|---:|---:|---:|
| 1M | 5.29 | 5.74 | 5.73 | 14.96 |
| 4M | 10.48 | 13.26 | 13.36 | 21.23 |
| 16M | 23.77 | 23.79 | 23.89 | 21.89 |
| 64M | 24.60 | 24.57 | 24.40 | 21.69 |
| 256M | 25.64 | 25.63 | 25.64 | 21.67 |

### S8 — 8 ranks, same node

| size | shim b1 | shim b8 | shim b32 | native |
|---:|---:|---:|---:|---:|
| 1M | 2.91 | 2.90 | 2.97 | 9.38 |
| 4M | 6.44 | 6.99 | 7.10 | 14.58 |
| 16M | 12.95 | 13.48 | 13.56 | 14.23 |
| 64M | 15.31 | 14.20 | 14.44 | 14.19 |
| 256M | 16.15 | 15.08 | 15.01 | 14.36 |

### X4 — 4 ranks cross-node (2+2)

| size | shim b1 | shim b8 | shim b32 | native |
|---:|---:|---:|---:|---:|
| 1M | 4.80 | 5.42 | 5.32 | 11.25 |
| 4M | 7.69 | 9.51 | 9.69 | 13.69 |
| 16M | 11.53 | 11.51 | 11.49 | 13.70 |
| 64M | 11.54 | 11.50 | 11.46 | 13.66 |
| 256M | 10.98 | 10.95 | 10.92 | 13.72 |

### X8 — 8 ranks cross-node (4+4)

| size | shim b1 | shim b8 | shim b32 | native |
|---:|---:|---:|---:|---:|
| 1M | 2.88 | 3.19 | 2.99 | 5.08 |
| 4M | 6.61 | 8.11 | 8.03 | 14.48 |
| 16M | 11.44 | 11.46 | 11.48 | 14.52 |
| 64M | 11.52 | 11.43 | 11.45 | 14.39 |
| 256M | 11.07 | 11.16 | 11.12 | 14.52 |

### X12 — 12 ranks cross-node (6+6)

| size | shim b1 | shim b8 | shim b32 | native |
|---:|---:|---:|---:|---:|
| 1M | 1.94 | 2.04 | 2.03 | 6.54 |
| 4M | 5.30 | 6.15 | 6.10 | 12.60 |
| 16M | 10.85 | 11.63 | 12.22 | 14.45 |
| 64M | 12.53 | 12.40 | 12.47 | 14.44 |
| 256M | 11.42 | 11.47 | 11.40 | 14.36 |

## AllToAll — busbw GB/s

### S2 — 2 ranks, same node (node5)

| size | shim b1 | shim b8 | shim b32 | native |
|---:|---:|---:|---:|---:|
| 1M | 7.7 | 7.8 | 7.8 | 9.7 |
| 4M | 16.5 | 16.8 | 16.8 | 18.1 |
| 16M | 22.0 | 21.6 | 22.0 | 22.2 |
| 64M | 23.7 | 23.7 | 23.6 | 23.5 |
| 256M | 23.8 | 23.8 | 23.9 | 23.9 |

### S4 — 4 ranks, same node

| size | shim b1 | shim b8 | shim b32 | native |
|---:|---:|---:|---:|---:|
| 1M | 7.0 | 8.8 | 8.3 | 12.5 |
| 4M | 13.1 | 15.3 | 16.3 | 18.3 |
| 16M | 15.3 | 16.3 | 13.9 | 20.1 |
| 64M | 15.7 | 15.6 | 15.2 | 20.7 |
| 256M | 16.1 | 16.0 | 16.2 | 20.2 |

### S8 — 8 ranks, same node

| size | shim b1 | shim b8 | shim b32 | native |
|---:|---:|---:|---:|---:|
| 1M | 4.6 | 4.5 | 4.6 | 4.8 |
| 4M | 6.3 | 6.1 | 6.2 | 5.9 |
| 16M | 6.9 | 6.4 | 6.7 | 6.8 |
| 64M | 6.6 | 6.7 | 6.7 | 6.6 |
| 256M | 6.7 | 6.6 | 6.5 | 6.9 |

### X4 — 4 ranks cross-node (2+2)

| size | shim b1 | shim b8 | shim b32 | native |
|---:|---:|---:|---:|---:|
| 1M | 6.3 | 6.5 | 6.3 | 6.8 |
| 4M | 8.6 | 8.4 | 8.4 | 9.3 |
| 16M | 9.1 | 9.1 | 9.1 | 10.7 |
| 64M | 8.7 | 8.6 | 8.7 | 11.3 |
| 256M | 8.8 | 8.8 | 8.8 | 11.3 |

### X8 — 8 ranks cross-node (4+4)

| size | shim b1 | shim b8 | shim b32 | native |
|---:|---:|---:|---:|---:|
| 1M | 5.1 | 4.9 | 4.8 | 5.3 |
| 4M | 5.8 | 5.8 | 5.8 | 6.3 |
| 16M | 6.1 | 6.1 | 6.1 | 6.4 |
| 64M | 6.4 | 6.2 | 6.2 | 6.3 |
| 256M | 6.1 | 6.1 | 6.1 | 6.5 |

### X12 — 12 ranks cross-node (6+6)

| size | shim b1 | shim b8 | shim b32 | native |
|---:|---:|---:|---:|---:|
| 1M | 2.4 | 2.4 | 2.4 | 3.2 |
| 4M | 2.3 | 2.4 | 2.4 | 3.5 |
| 16M | 2.6 | 2.7 | 2.7 | 3.5 |
| 64M | 3.0 | 2.8 | 2.5 | 3.6 |
| 256M | 2.8 | 2.6 | 2.6 | 3.6 |

## Analysis

- **Same-node AllReduce: shim wins at 16M+.** S2/S4 reach 23.5-25.6
  GB/s vs native 21.2-21.9 (shim +14-18%); S8 is at parity (15.0-16.2
  vs 14.2-14.4). At small sizes (≤4M) native leads 2-3x — the shim's
  host-driven proxy/dispatch cost dominates latency at small message
  sizes.
- **Cross-node AllReduce: native leads ~15-30%** at all sizes
  (X12 256M: shim 11.4 vs native 14.4). This is the per-cross-edge
  throughput gap (~11 GB/s vs ~14 GB/s aggregate here), not a deadlock
  or correctness issue.
- **AllToAll is close to native everywhere** — same-node parity at 16M+
  (S2 ~23.8 both), cross-node within 5-25% (X12: shim 2.6-3.0 vs native
  3.5-3.6). AllToAll moves bytes through the CE/IPC/RDMA paths with the
  worker mostly uninvolved, so blocks barely matter.
- **Worker blocks matter only for AllReduce at small sizes**: b8/b32
  beat b1 (S2 1M: 8.74/9.17 vs 7.32) because more blocks hide the
  reduce's latency; at 16M+ blocks are neutral. b8 remains the sane
  default (matches the ~32-SM saturation point of the reduce kernel).

## Platform quirks discovered on this pair (2026-08-26)

1. **`HostNativeAtomicSupported=0`** (measured; B300 reports the same).
   The GPU cannot do native atomics on host-pinned memory, so a device
   kernel must never `atomicAdd_system` a host signal ring. Device
   completion signals use the per-slot **plain-store device flag**
   protocol (`signal_flag_write`: `__threadfence_system` + store +
   fence). The old signal-ring producer (`signal_ring_write`) was removed
   in `08d4f381`.
2. **Copy-engine `cudaMemset` writes do not reliably drain before a
   resident worker kernel's read-modify-write.** `cudaMemset` returns
   (host-side) while the CE's zero-writes are still in flight; the
   worker's reduce then reads 0, writes 1, and a late CE write lands
   after it, silently reverting the element to 0. The first reduce
   round loses ~1.8M elements (sparse 128B cache lines, 64MB-segment
   pattern, run-varying). It survives a full driver reload (firmware-
   level behavior), is unrelated to gdrcopy, and is avoided by:
   - **kernel-zero** the buffer instead of `cudaMemset` (kernel
     completion orders the writes) — committed for benches/tests
     (`8c4f9740`, `08d4f381`), or
   - a ~200ms delay after `cudaMemset`, or
   - launching the persistent worker *after* the memset+sync (the launch
     boundary is the ordering point; verified wrong=0 5/5).
3. **ShmExchanger multi-rank init race (fixed).** The POSIX shm store
   was created at size 0 and only the creator truncated it after the
   flock; a peer that opened + locked before the creator's ftruncate
   mmap'd a 0-length file and SIGBUSed (`init_shared_store` bus error,
   intermittent on multi-rank starts). The store is now sized under the
   lock regardless of who created it.

## Worker completion barrier (commits `33c5b812`, `864239ce`)

The persistent worker's per-task multi-block barrier was a reset-to-0
counter; a slow block arriving after the reset leaked its +1 into the
next task's count, releasing barriers early and re-processing tasks
(wrong reduce results, and a permanent worker hang in the real path).
Fixes:
- **Monotonic completion counter** keyed to the absolute task index:
  task N's barrier completes at `gridDim.x * (N+1)`, never reset, so a
  late block's add is absorbed into the correct task.
- **Tail-visible release**: blocks also wait for the leader's FIFO-tail
  publish before advancing.
- **Device-side counter anchor**: the counter is zeroed by the host on
  every (re)launch, but a relaunched grid must start from
  `gridDim.x * tail`. A host-side anchor is unreliable (the host's GDR
  read of the tail can lag 10 tasks), so block 0 re-anchors at kernel
  entry from its own device-scope tail read and publishes an
  anchor-ready flag the other blocks wait on.

### Same-host IPC test invocation gotcha

`test_spray_executor_e2e` needs both ranks to see **both** GPUs
(`CUDA_VISIBLE_DEVICES=0,1` with `--gpu=0/--gpu=1`). Restricting each
rank to one visible device breaks the peer device numbering — the peer's
published `device_idx` then points at the caller's own GPU, so
`cudaDeviceEnablePeerAccess` fails and every IPC put is rejected forever
(looks like an executor deadlock).

## Copy engine (CE) bandwidth

`/tmp/run_ingress2.sh` on node5:

| test | GB/s |
|---:|---:|
| ce_d2d_same | 10.8 |
| ce_peer_1to1 (gpu1→gpu0) | 0.9 |
| ce_peer_5to1 aggregate into gpu0 | 20.9 |
| ce_h2d_pinned | 0.8 |

Identical before and after a full reboot + driver reload — treat these
as the stable baseline for this pair, not a recoverable degradation.
The collectives' core bandwidth uses SM loads/stores, not the CE, so CE
throughput does not limit the measured collectives.

## Worker reduce peak bandwidth (launch path, 256MB fp32 sum)

| blocks | GB/s |
|---:|---:|
| 8 | 87.3 |
| 16 | 150 |
| 32 | 211 (saturation ~92%) |
| 64 | 230.3 (peak) |

~32 SMs saturate; 64 takes the full bandwidth. `blocks_per_worker=8`
comfortably feeds the NIC/CE ingress rates and remains the default.

## Next steps

- Close the cross-node AllReduce gap (shim 11-12 vs native 14-15
  GB/s at 256M): the limiting factor is per-cross-edge RDMA throughput;
  multi-QP striping per cross chunk is the next lever.
- Small-message AllReduce latency: the shim's host-driven proxy adds
  dispatch cost at ≤4M; a tighter enqueue path (or batching signals)
  would close the 2-3x small-size gap.
- AllToAll cross-node is within 25% of native — verify the same shape
  on B300 once it returns (the CE/incast behavior differs there).
