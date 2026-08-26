# L40S measurements (node5/node6)

Cross-node shim-vs-native measurements on the L40S pair, plus the
NIC-aware ring-order experiments. Follows the B300 report format: keep
commands reproducible, append new runs with a date section.

## Environment

- node5 `10.31.154.11`, node6 `10.31.154.12`; 8× L40S each (PCIe-only,
  no NVLink), ConnectX-6 dual-port HDR
- IB fabric: node5 `mlx5_0` (LID 1, 200G), `mlx5_3` (LID 5, 200G);
  node6 `mlx5_0` (LID 3), `mlx5_1` (LID 2) — same subnet, all 200G
- 10G "bond" ports (`mlx5_bond_0/1`, 1X QDR, **LID 0**) on both nodes
  are *not* on the fabric: cross-node QPs over them fail to establish.
  On node5, GPU1/GPU4 pick these ports, so ranks 1/4 cannot carry
  cross-node RDMA in any ring order.
- k8s pods (vLLM) occupy GPU1 on node5 and GPU6 on node6 (moved over
  time; check `nvidia-smi` before runs). Benchmarks use 6 free GPUs per
  node, e.g. `CUDA_VISIBLE_DEVICES=0,2,3,4,5,7` → 12 ranks cross-node.
- Shim: `experimental/ukernel` build (`libukernel.so` + NCCL compat);
  nccl-tests MPI build under `thirdparty/nccl-tests/build`.

Reproducible command (12 ranks, shim):

```bash
cd /root/uccl/uccl/thirdparty/nccl-tests/build
export LD_LIBRARY_PATH=/root/uccl/uccl/experimental/ukernel/build/nccl/lib:/usr/local/lib
mpirun --allow-run-as-root --bind-to none --map-by :OVERSUBSCRIBE -np 12 \
  --host 10.31.154.11:6,10.31.154.12:6 \
  -x LD_LIBRARY_PATH -x UK_CCL_UNBIND=1 -x UK_CCL_RDMA_FUSED_MODE=proxy \
  -x CUDA_VISIBLE_DEVICES=0,2,3,4,5,7 \
  -x UK_CCL_RUN_WATCHDOG_MS=30000 \
  ./all_reduce_perf -b 1M -e 8M -f 2 -g 1 -c 1 -n 10 -w 2
```

Native NCCL: point `LD_LIBRARY_PATH` at `/usr/local/cuda/lib64:/usr/lib64`
(NCCL 2.31.2) instead of the shim.

## Native vs shim, 12 ranks cross-node (2026-08-24)

AllReduce `float sum`, 6 GPUs/node, validation on, 0 wrong everywhere.
Busbw in GB/s.

| size | native time (us) | native busbw | shim identity time | shim identity busbw | shim interleave time | shim interleave busbw |
|---:|---:|---:|---:|---:|---:|---:|
| 1M | 296.8 | 6.48 | 993.5 | 1.93 | 1045.0 | 1.84 |
| 2M | 358.9 | 10.71 | 1091.7 | 3.52 | 1666.5 | 2.31 |
| 4M | 587.9 | 13.08 | 1295.9 | 5.93 | 3658.2 | 2.10 |
| 8M | 1020.9 | 15.06 | 1856.7 | 8.28 | 4944.9 | 3.11 |

With the multi-HCA device spread (2026-08-24, see below):

| size | shim identity | shim identity busbw | shim interleave | shim interleave busbw |
|---:|---:|---:|---:|---:|
| 1M | 887.9 | 2.17 | 910.0 | 2.11 |
| 2M | 1037.8 | 3.70 | 1054.5 | 3.65 |
| 4M | 1231.1 | 6.25 | 1362.0 | 5.65 |
| 8M | 1717.8 | 8.95 | 2240.1 | 6.87 |

AllGather / ReduceScatter 8M (avg busbw):

| coll | shim identity | shim interleave |
|---:|---:|---:|
| AG | 9.34 | 2.98 |
| RS | 7.66 | 2.85 |

## 16-rank cross-node baseline (2026-08-23 09:05–09:12, 8 GPUs/node)

32M out-of-place, busbw GB/s (before the k8s pods occupied GPUs 0/3):

| coll | shim | native NCCL |
|---:|---:|---:|
| AR | 6341 us / 9.9 | 4284 us / 14.7 |
| AG | 3192 us / 9.9 | 2094 us / 15.0 |
| RS | 3202 us / 9.8 | 2202 us / 14.3 |

Same-node 8-rank AG 32M: shim 1933 us, native 2076 us (shim wins
same-node). 2-rank cross-node 32M AR: shim 2511 us / 13.36 GB/s
(`uk2rank.log`); native cross-node 2-rank not yet measured.

## Ring-order (interleave) experiment (2026-08-24)

`UK_CCL_RING_INTERLEAVE=1` block-interleaves per-host rank groups;
`UK_CCL_RING_BLOCK` tunes the width. 12 ranks, 8M AllGather, avg busbw:

| block | cross-node edges | result |
|---:|---:|---|
| 1 | 12 | 0.87 GB/s, 0 wrong |
| 3 (default) | 4 | 3.07 GB/s, 0 wrong |
| 4 | 4 | 2.95 GB/s, 0 wrong |
| 6 (= identity) | 2 | 8.41 GB/s, 0 wrong |

Findings:

1. Interleave is a regression, not an optimization: every extra
   cross-node edge carries a full chunk per step, so doubling the edge
   count doubles total cross-node bytes while per-edge throughput stays
   fixed (~4.2 GB/s), so the shim spends ~2.6x longer in the RDMA
   phases. Both AG and RS degrade together, and HostProf shows host-side
   enqueue/dispatch cost is not the bottleneck (0.01 us/op).
2. Ring orders that make node5 ranks 1 or 4 cross-node used to crash at
   `transport put path is not established`: those GPUs picked the 10G
   bond ports (LID 0, off-fabric). Fixed 2026-08-24 by filtering
   off-fabric ports in `pick_dev_for_gpu` (IB ports need an SM LID;
   RoCE ports need a cluster-grade rate + valid GID) — GPU1/4 now fall
   back to the 200G port and every block width runs 0 wrong. The ring
   planner additionally validates that both ends of every cross edge
   report an on-fabric NIC and falls back to identity otherwise.
3. The interleave regression was a **device-selection artifact**:
   node5's two 200G HCAs (mlx5_0, mlx5_3) sit at the same PCIe
   distance, and the old name-order tie-break put every GPU on mlx5_0 —
   which measures slower anyway (p50 143 us, ~15 GB/s vs mlx5_3's
   71 us / 18.5-25 GB/s). Every concurrent cross flow shared one port,
   roughly halving per-flow wire bandwidth. `pick_dev_for_gpu` now
   round-robins GPUs across the nearest-distance fabric tier: interleave
   8M went 3.06 -> 6.87 GB/s and identity 8.28 -> 8.95 GB/s.
4. The remaining identity-vs-native gap (8.95 vs 15.06 at 8M, 9.9 vs
   14.7 at 16 ranks / 32M) is per-cross-edge throughput (~4.5 vs native
   ~7.4 GB/s per edge). Fixing it needs better per-edge pipelining /
   multi-rail splitting of each cross chunk, not more edges.

## Platform quirks discovered on this pair (2026-08-26)

Two platform-level characteristics shape how the shim must be written on
these L40S boxes:

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

## 12 ranks cross-node, post-fix (2026-08-26)

After the worker barrier + counter-anchor fixes (see below), the
cross-node AllReduce ring completes with 0 wrong (was a permanent
deadlock). 6 GPUs/node, `UK_CCL_RDMA_FUSED_MODE=proxy`, validation on:

| size | shim time (us) | shim busbw | native busbw (08-24) |
|---:|---:|---:|---:|
| 1M | 983.7 | 1.95 | 6.48 |
| 2M | 1075.3 | 3.58 | 10.71 |
| 4M | 1253.1 | 6.14 | 13.08 |
| 8M | 1694.4 | 9.08 | 15.06 |

The remaining gap is per-cross-edge throughput (~4.5 vs native ~7.4
GB/s/edge), not deadlock.

### Worker completion barrier (commit `33c5b812`, `864239ce`)

The persistent worker's per-task multi-block barrier was a
reset-to-0 counter; a slow block arriving after the reset leaked its +1
into the next task's count, releasing barriers early and re-processing
tasks (wrong reduce results, and a permanent worker hang in the real
path). Fixes:
- **Monotonic completion counter** keyed to the absolute task index:
  task N's barrier completes at `gridDim.x * (N+1)`, never reset, so a
  late block's add is absorbed into the correct task.
- **Tail-visible release**: blocks also wait for the leader's FIFO-tail
  publish before advancing (releasing on the counter alone let a block
  race ahead of the publish and re-process the task).
- **Device-side counter anchor**: the counter is zeroed by the host on
  every (re)launch, but a relaunched grid (worker idle-exits between
  ops) must start from `gridDim.x * tail`. A host-side anchor is
  unreliable — the host's GDR read of the tail can lag (measured 10
  tasks behind) — so block 0 re-anchors at kernel entry from its own
  device-scope tail read and publishes an anchor-ready flag the other
  blocks wait on.

Symptom the anchor fixed: `[dev-stall] fifo0 pending=1 head=2 tail=1`
with the worker spinning in the barrier's counter wait (`stuck=3`,
count 448 vs tail 66 → anchor 10 tasks stale).

### Same-host IPC test invocation gotcha

`test_spray_executor_e2e` needs both ranks to see **both** GPUs
(`CUDA_VISIBLE_DEVICES=0,1` with `--gpu=0/--gpu=1`). Restricting each
rank to one visible device breaks the peer device numbering — the peer's
published `device_idx` then points at the caller's own GPU, so
`cudaDeviceEnablePeerAccess` fails with "peer access is not supported"
and every IPC put is rejected forever (looks like an executor
deadlock). `cudaDeviceCanAccessPeer(0,1)=1` here; P2P enable works when
the device numbering is consistent across ranks.

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

~32 SMs saturate; 64 takes the full bandwidth. The worker default of 8
blocks comfortably feeds the NIC/CE ingress rates, so
`blocks_per_worker=8` remains a sensible default.

## Next steps

- Push per-cross-edge throughput: inspect WR posting / completion
  pacing per edge, multi-QP (multi-rail) split of each cross chunk; and
  extend the topology snapshot with link kind/speed for per-dst credit
  tuning once B300 returns (incast control).
- Re-run 16-rank interleave validation once all 8 GPUs per node are
  free to confirm the regression shape holds at 8 ranks/node.
- Keep the CE baseline in mind when reading any test that resets
  buffers with `cudaMemset` before a worker op; prefer the kernel-zero
  helper (`UKernel::Device::zero_device_buffer`).
