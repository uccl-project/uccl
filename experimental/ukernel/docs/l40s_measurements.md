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
- GPUs 0/3 on both nodes are occupied by k8s pods (vLLM + distkv);
  benchmarks run on `CUDA_VISIBLE_DEVICES=1,2,4,5,6,7` → 6 GPUs/node,
  12 ranks cross-node.
- Shim: `experimental/ukernel` build (`libukernel.so` + NCCL compat);
  nccl-tests MPI build under `thirdparty/nccl-tests/build`.

Reproducible command (12 ranks, shim):

```bash
cd /root/uccl/uccl/thirdparty/nccl-tests/build
export LD_LIBRARY_PATH=/root/uccl/uccl/experimental/ukernel/build/nccl/lib:/usr/local/lib
mpirun --allow-run-as-root --bind-to none --map-by :OVERSUBSCRIBE -np 12 \
  --host 10.31.154.11:6,10.31.154.12:6 \
  -x LD_LIBRARY_PATH -x UK_CCL_UNBIND=1 -x UK_CCL_RDMA_FUSED_MODE=proxy \
  -x CUDA_VISIBLE_DEVICES=1,2,4,5,6,7 \
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
3. The 16-rank gap (9.9 vs 14.7 GB/s) is per-cross-edge throughput
   (~4.2 vs native ~7.4 GB/s per edge), not edge count. Fixing it needs
   better per-edge pipelining / multi-rail splitting, not more edges.

## Next steps

- Push per-cross-edge throughput: inspect WR posting / completion
  pacing per edge, multi-QP (multi-rail) split of each cross chunk; and
  extend the topology snapshot with link kind/speed for per-dst credit
  tuning once B300 returns (incast control).
- Re-run 16-rank interleave validation once GPUs 0/3 are free (8
  GPUs/node) to confirm the regression shape holds at 8 ranks/node.
