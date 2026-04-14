# DeepEP-Lite

Fork of `uccl/ep` that runs DeepEP on hardware **without NVLink or GPUDirect
RDMA** — PCIe + standard RDMA only. Low-latency mode only.

| DeepEP Dependency | Replacement |
|-------------------|-------------|
| IBGDA (NVSHMEM) | uccl-ep CPU proxy (pre-existing) |
| GPUDirect RDMA | Host-staged: GPU↔Host PCIe + Host↔Host RDMA |
| NVLink (intra-node) | POSIX shared-memory + proxy memcpy |

Tested on 2 nodes × 4× L4 (sm_89, PCIe Gen4, mlx5 400Gbps NIC).

## Architecture

```
Inter-node:  GPU →(PCIe)→ host RDMA buf →(ibv RDMA WRITE)→ remote host buf →(PCIe)→ remote GPU
Intra-node:  GPU →(PCIe)→ POSIX shm slice →(proxy memcpy)→ peer shm slice →(PCIe)→ peer GPU
```

- `cudaMallocHost()` RDMA buffer is PCIe-mapped — accessible from GPU, CPU, and NIC.
  Auto-detected via `can_register_gpu_memory_for_rdma()`.
- QPs never created for same-node peers.
- CUDA IPC P2P (NVLink-style kernel writes over PCIe) is 5× slower than proxy
  memcpy — each `st.global` is an individual PCIe TLP. See `optimization-attempts.md`.

## Key Changes from uccl/ep

1. **`NUM_MAX_NVL_PEERS=1`** (`PCIE_INTRANODE` flag): all peers are RDMA peers.
2. **sm_89 fallbacks**: TMA/mbarrier/elect.sync gated with `__CUDA_ARCH__ >= 900`;
   sm_89 uses `__ldg`/global stores + `UNROLLED_WARP_COPY`.
3. **Shared PD**: proxy threads share `ibv_context`/PD/MR via `g_shared_rdma_cache`.
4. **RDMA atomic → RDMA WRITE**: `FETCH_AND_ADD` to `cudaHostAlloc` fails cross-node;
   replaced with `RDMA_WRITE` + `IBV_SEND_INLINE`.
5. **POSIX shm intra-node**: `/dev/shm/uccl_rdma_*`, proxy memcpy between slices.
6. **Redundant fence fix**: removed duplicate `__threadfence_system()` in
   `internode_ll.cu` (already in `ring_buffer.cuh::atomic_set_and_commit()`).

## Build & Test

```bash
make -j SM=89                    # produces ep.abi3.so
make install                     # install to site-packages
torchrun --nproc_per_node=4 \    # single-node test
  bench/test_low_latency.py --num-tokens=128 --hidden=2048 \
  --num-topk=4 --num-experts=8 --disable-nvlink
bash run_multinode.sh            # multi-node (see script for usage)
```

Constraint: `num_experts` must be divisible by total GPU count.

## Benchmark (L4, 128 tok × 7168 hid × top-8 × 64 exp)

### No GDR (`UCCL_FORCE_NO_GDR=1`)

| Setup | D+C BW (GB/s) | Dispatch | Combine | Latency (μs) |
|-------|:-------------:|:--------:|:-------:|:------------:|
| 1n×2g | 12.02 | 13.98 | 11.26 | 1,835 |
| 1n×4g | 6.95 | 4.98 | 8.74 | 3,172 |
| 2n×1g | 12.38 | 14.10 | 11.70 | 1,782 |
| 2n×4g | 5.96 | 5.28 | 6.42 | 3,702 |

### With GDR (default, nvidia_peermem)

| Setup | D+C BW (GB/s) | Dispatch | Combine | Latency (μs) |
|-------|:-------------:|:--------:|:-------:|:------------:|
| 1n×2g | 12.02 | 14.11 | 11.23 | 1,835 |
| 1n×4g | 6.91 | 5.07 | 8.70 | 3,192 |
| 2n×1g | **24.89** | 24.06 | 21.25 | 886 |
| 2n×4g | 6.01 | 5.34 | 6.36 | 3,668 |

**GDR impact**: Single-node and 2n×4g are identical (shared-memory intra-node
path is selected before GDR detection). 2n×1g sees **2× speedup** with GDR
because `local_world_size=1` bypasses shared buffer — NIC reads/writes GPU
VRAM directly, eliminating PCIe host staging.

Bottleneck: PCIe round-trip latency per token (ring buffer CAS, fences).
See `optimization-attempts.md` for 20 optimization attempts (only fence fix effective: +9.1%).

## Modification Guidelines

- **Kernels**: guard sm_90+ with `__CUDA_ARCH__ >= 900`.
- **NVL_PEERS**: must handle `== 1` in asserts and casts.
- **RDMA**: non-GDR auto-detected; no manual flags.
- **Includes**: `util/` headers from `../../../include/` (uccl repo root).
