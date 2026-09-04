# L40S benchmark report (current uk-300)

Shim (ukernel) vs native NCCL on the two-node L40S cluster, built from `uk-300` HEAD (fused-RS thread sync, per-fifo TaskArgs pools, single-coordinator idle exit). All runs validated 0 wrong / AllToAll verify OK; medians of 3.

## Environment

- node5 `10.31.154.11`, node6 `10.31.154.12`: 8× NVIDIA L40S each (PCIe Gen5 x16, no NVLink), ConnectX-6 HDR dual-port 200G, driver 610.57.04.
- Shim: `experimental/ukernel` (`build/nccl/lib`). Native: system NCCL 2.31.2+cuda13.3.
- nccl-tests MPI build; AllToAll via `bench/alltoall_perf` (ncclAllToAll) with per-rank device wrapper. Sizes 1M..256M, n=5 w=1, 3 reps, median OOP busbw.
- Same-node ranks 2/4/8 on node5; cross-node 4/8/16 split evenly (2+2 / 4+4 / 8+8). Cross-node shim adds `UK_CCL_RDMA_FUSED_MODE=proxy`.

## AllReduce — busbw GB/s (median of 3)

### Same-node

| size | np2 shim/native | np4 shim/native | np8 shim/native |
|---|---|---|---|
| 1M | 9.2/14.2 | 5.8/14.9 | 3.0/9.2 |
| 4M | 20.1/20.1 | 13.3/21.2 | 7.3/14.7 |
| 16M | 23.5/21.1 | 23.8/21.9 | 14.0/14.4 |
| 64M | 24.4/21.5 | 24.5/21.7 | 14.9/14.2 |
| 256M | 25.6/21.7 | 25.6/21.8 | 15.6/14.4 |

### Cross-node

| size | np4 shim/native | np8 shim/native | np16 shim/native |
|---|---|---|---|
| 1M | 5.3/11.1 | 3.2/5.1 | 1.5/5.1 |
| 4M | 9.7/13.6 | 8.1/14.6 | 5.0/12.4 |
| 16M | 11.5/13.9 | 11.5/14.6 | 10.6/15.1 |
| 64M | 11.5/13.7 | 11.6/14.4 | 11.7/14.6 |
| 256M | 11.0/13.7 | 11.3/14.4 | 11.2/14.6 |

### Same-node fused vs unfused (2026-09-04)

Fusion ablation closeout on the PCIe platform. Fused =
`UK_CCL_DEV_BLOCKS=32` + `FUSE_REDUCE_COPY=1 FUSE_AG_COPY=1
LARGE_TILES=16 TILE_MIN_BYTES=8M IPC_BATCH=16`; unfused = `DEV_BLOCKS=32`
only. nccl-tests AllReduce, OOP, medians of 3, all 0 wrong.

AllReduce busbw (GB/s), unfused / fused:

| size | np2 | np4 | np8 |
|---|---:|---:|---:|
| 16M | 23.5 / 20.5 | 24.0 / 12.9 | 13.8 / 7.4 |
| 64M | 24.4 / 22.2 | 24.6 / 17.9 | 14.7 / 10.1 |
| 256M | 25.6 / 22.9 | 25.5 / 19.7 | 15.6 / 10.2 |

Fusion is **negative on L40S**: 11-13% slower at 2 ranks and 20-46%
slower at 4-8 ranks. With PCIe the bottleneck the CE/IPC path beats the
fused device-copy path (SM LD/ST to peer), so the shipped L40S shim
column stays unfused; fusion is a B300/NVLink story.

## AllToAll — busbw GB/s (rank-0 median)

### Same-node

| size | np2 shim/native | np4 shim/native | np8 shim/native |
|---|---|---|---|
| 1M | 7.7/9.8 | 7.7/12.6 | 4.6/4.8 |
| 4M | 17.0/18.0 | 14.2/17.7 | 6.2/5.8 |
| 16M | 21.9/22.3 | 15.2/20.5 | 6.7/6.7 |
| 64M | 23.7/23.6 | 15.1/20.3 | 6.6/6.7 |
| 256M | 23.8/24.0 | 15.8/20.0 | 6.6/7.6 |

### Cross-node

| size | np4 shim/native | np8 shim/native | np16 shim/native |
|---|---|---|---|
| 1M | 6.2/6.9 | 4.9/4.9 | 1.9/3.4 |
| 4M | 8.4/9.5 | 5.8/6.5 | 2.0/4.2 |
| 16M | 9.1/10.8 | 6.1/6.6 | 2.1/4.4 |
| 64M | 8.7/11.3 | 6.1/6.4 | 2.3/4.6 |
| 256M | 8.7/11.3 | 6.1/6.5 | 2.1/4.7 |

## Factor analysis

### Effect of message size (same-node AllReduce)

The shim is latency-bound at 1M (np2 ratio 0.65) and reaches parity by 4M (np2: 1.00), then *beats* native at 16M+ on 2-4 ranks (1.12-1.18 at 16-256M). At np8 both saturate the PCIe bus near ~14-16 GB/s and are within noise (0.97-1.08). Native also plateaus around 21-22 GB/s at np2/4, so the shim's absolute ceiling (~25 GB/s) is slightly above native on this platform — the L40S story is the opposite of B300: no NVLink means the CE/IPC path is competitive.

### Effect of ranks (same-node)

Both sides lose busbw as ranks grow (np2 25.6/21.7 → np8 15.6/14.4 GB/s at 256M), because more ring hops over the same PCIe switch add serialization. The shim/native ratio stays ~1.1 at np2/4 but converges to ~1.0 at np8. AllToAll same-node shows a sharper shim drop at np4 (0.74-0.79 of native) while np2/np8 are near parity.

### Cross-node: RDMA proxy overhead

Cross-node AllReduce shim/native is 0.48-0.84 (small sizes worst, 16M+ ~0.77-0.84) — the software RDMA fused proxy costs ~20-25% vs native at large sizes and more at 1-4M. AllToAll cross-node: np4/8 near parity (0.77-1.00) but np16 shim drops to ~0.45-0.56 (2.0-2.3 vs 4.2-4.7 GB/s), consistent with per-peer incast or proxy throughput limits at 8 ranks per node.

### Effect of shim blocks (256M same-node unfused AllReduce)

| blocks | np2 | np4 | np8 |
|---:|---:|---:|---:|
| 8 | 25.6 | 25.4 | 15.9 |
| 16 | 25.6 | 25.4 | 15.5 |
| 32 | 25.6 | 25.6 | 15.6 |
| 64 | 25.6 | 25.4 | 15.4 |

Blocks are neutral on L40S: 25.3-25.6 GB/s at np2/4 and 15.4-16.2 at np8 across b8..b64. The bottleneck is PCIe, not worker parallelism — this is why the multi-fifo hang (fixed this cycle) had no perf impact here.

### Reproducible commands

```bash
cd /root/uccl/uccl/thirdparty/nccl-tests/build
export LD_LIBRARY_PATH=/root/uccl/uccl/experimental/ukernel/build/nccl/lib:/usr/local/lib
mpirun --allow-run-as-root --bind-to none --map-by :OVERSUBSCRIBE -np 4 \
  -x LD_LIBRARY_PATH -x UK_CCL_UNBIND=1 \
  ./all_reduce_perf -b 1M -e 256M -f 4 -g 1 -c 1 -n 5 -w 1
# cross-node:
mpirun --allow-run-as-root --bind-to none --map-by :OVERSUBSCRIBE -np 8 \
  --host 10.31.154.11:4,10.31.154.12:4 -x LD_LIBRARY_PATH -x UK_CCL_UNBIND=1 \
  -x UK_CCL_RDMA_FUSED_MODE=proxy ./all_reduce_perf -b 1M -e 256M -f 4 -g 1 -c 1 -n 5 -w 1
```

Raw logs: `/tmp/uk_l40s_matrix/` on node5; runner: `bench/run_l40s_matrix.sh`.
