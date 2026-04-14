# DeepEP-Lite L4 Performance Benchmarks

Low-latency mode on L4 GPUs (sm_89, PCIe Gen4 x16, no NVLink).

**Testbed**: 2 nodes (l40/l41), 4× L4 each, 1× mlx5 400Gbps RDMA NIC per node.

**Optimizations applied**: Redundant `__threadfence_system()` removal (+9.1%),
shared-memory intra-node (no RDMA loopback).

**Benchmark**: `bench/test_low_latency.py`, `num_topk=4`, `num_experts=8`,
`--disable-nvlink`. All experiments pass correctness verification.

**Note on GDR**: L4 supports GDR (nvidia_peermem), but with `NUM_MAX_NVL_PEERS=1`
and shared-memory intra-node, the RDMA buffer is always host-allocated
(`cudaMallocHost` via POSIX shm). GDR is effectively unused — `UCCL_FORCE_NO_GDR`
has no effect because the shared-buffer path is selected before GDR detection.

---

## Summary Table

RDMA buffers use `cudaMallocHost` (host pinned memory via POSIX shared memory).
Intra-node uses proxy memcpy between shm slices (no NIC involvement).
Inter-node uses host-staged RDMA WRITE.

| Setup | Tokens | Hidden | D+C BW (GB/s) | D BW (GB/s) | C BW (GB/s) | D+C Latency (μs) |
|-------|-------:|-------:|---------------:|------------:|------------:|------------------:|
| 1n×2g | 128 | 2048 | 9.81 | 6.23 | 9.97 | 318 |
| 1n×4g | 128 | 2048 | 7.80 | 5.80 | 7.10 | 400 |

### Detailed Timing

| Setup | D Send (μs) | D Recv (μs) | C Send (μs) | C Recv (μs) |
|-------|------------:|------------:|------------:|------------:|
| 1n×2g 128×2048 | 115 | 59 | 107 | 101 |
| 1n×4g 128×2048 | 175 | 60 | 114 | 105 |

## Analysis

### Bottleneck: PCIe Bandwidth

The system is PCIe-bandwidth-limited. FP8 cast uses HW intrinsics, PCIe writes
are coalesced, ring buffer atomics are pipelined. The dispatch send time (~115 μs
for 128×2048) is dominated by per-token overhead: ring buffer CAS, volatile
reads, and `__threadfence_system()` — all requiring PCIe round-trips.

See `optimization-attempts.md` for the full list of 20 optimization attempts.

### Why Not CUDA IPC P2P for Intra-Node?

Tested: `NUM_MAX_NVL_PEERS=4`, kernel writes directly to peer GPU memory via
PCIe P2P (like the NVLink path). Result: **1.54 GB/s** (−80% vs shared memory).
Each GPU `st.global` to remote GPU memory generates individual PCIe TLPs with
~2 μs per-access latency. CPU proxy batch memcpy is 5× faster on PCIe.

