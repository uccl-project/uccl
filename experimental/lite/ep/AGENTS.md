# AGENTS.md — DeepEP-Lite

## Project Identity

**DeepEP-Lite** runs DeepEP (expert-parallel MoE communication) in
environments **without NVLink and without GPUDirect RDMA (GDR)** — only PCIe
and standard (non-GDR) RDMA.

This codebase is a fork of `uccl/ep` (uccl-ep), placed in
`experimental/lite/ep/` to keep the mainline ep code untouched.

### What problems does DeepEP-Lite solve?

DeepEP assumes three hardware features:

| Feature | What it does | DeepEP-Lite replacement |
|---------|-------------|------------------------|
| **IBGDA** (NVSHMEM) | GPU-initiated RDMA | uccl-ep's CPU proxy (already solved) |
| **GPUDirect RDMA** | NIC reads/writes GPU VRAM directly | **Host-staged data path**: GPU↔Host DMA + Host↔Host RDMA |
| **NVLink** | High-bandwidth intranode GPU interconnect | **PCIe P2P** or host-staged memcpy fallback |

### Target Testbed

- **2 nodes**, each with **4× L4 GPUs** (PCIe, no NVLink, no GDR)
- **400 Gbps Mellanox ConnectX (mlx5)** RDMA NIC (non-GDR)
- IPs: see `experimental/lite/ip.txt` (4.14.153.89, 4.14.153.90)
- L4 = Ada Lovelace, sm_89, PCIe Gen4 x16 (~32 GB/s)

### Scope

- **Low-latency mode only** (per-expert routing, small RDMA writes)
- Normal (batched high-throughput) mode deferred
- Intranode: try CUDA P2P first, host-staged fallback

## Architecture

### Current Data Flow (GDR path — what uccl-ep does today)

```
GPU kernel → D2H cmd queue → CPU proxy → RDMA WRITE (NIC reads from GPU VRAM)
                                                      ↓
                                             Remote GPU VRAM (NIC writes directly)
```

### New Data Flow (Host-Staged path — what DeepEP-Lite does)

The key discovery: the existing uccl-ep code **already supports host-allocated
RDMA buffers**. When `can_register_gpu_memory_for_rdma()` returns false (as it
will on L4 without GDR), the buffer is allocated with `cudaMallocHost()`, which
is accessible from both CPU and GPU:

```
GPU kernel → writes data to RDMA buffer (host memory, PCIe-mapped)
          → issues TransferCmd via D2H queue
CPU proxy → reads TransferCmd, extracts offset
          → ibv_post_send(offset in host RDMA buf → remote host RDMA buf)
Remote NIC → writes to remote host RDMA buffer
Remote GPU → reads from host RDMA buffer (PCIe-mapped)
```

No explicit `cudaMemcpy` staging is needed for correctness — the GPU accesses
host memory directly via PCIe-mapped pointers. This is slower than GDR but
functional. Future optimization: batch PCIe transfers with explicit DMA.

### Layer Diagram

```
┌────────────────────────────────────────────────────────────┐
│  Python API (DeepEP-compatible)                            │
│  Buffer, low_latency_dispatch(), low_latency_combine()     │
├────────────────────────────────────────────────────────────┤
│  EP Runtime (CUDA kernels, sm_89 compatible)               │
│  Issue commands via D2H queue (same as uccl-ep)            │
│  TMA→global load/store fallback for sm_89                  │
├────────────────────────────────────────────────────────────┤
│  CPU Proxy (existing uccl-ep proxy, unmodified)            │
│  Sender:  poll D2H queue → post RDMA WRITE from host buf  │
│  Receiver: poll CQ → data already in host RDMA buf        │
├────────────────────────────────────────────────────────────┤
│  RDMA Transport (ibverbs)                                  │
│  ibv_reg_mr on host pinned memory — no GDR needed          │
│  Auto-detected via can_register_gpu_memory_for_rdma()      │
├────────────────────────────────────────────────────────────┤
│  Host RDMA Buffer (cudaMallocHost)                         │
│  PCIe-mapped: accessible from both GPU and CPU/NIC         │
│  No separate staging needed for correctness                │
└────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
experimental/lite/ep/
├── AGENTS.md              ← This file (AI-native project docs)
├── Makefile               ← Build system (PCIE_INTRANODE=1 by default)
├── include/
│   ├── ep_configs.cuh     ← NUM_MAX_NVL_PEERS (1 or 8)
│   ├── ep_utils.cuh       ← TMA/mbarrier fallbacks for sm_89
│   ├── buffer.cuh         ← Asym/SymBuffer templates
│   ├── proxy.hpp          ← Proxy interface
│   ├── rdma.hpp           ← RDMA abstractions
│   ├── uccl_ibgda.cuh     ← GPU command interface + get_ipc_p2p_ptr
│   ├── common.hpp         ← Constants, compile flags
│   └── ...
├── src/
│   ├── proxy.cpp          ← CPU proxy (sender/receiver)
│   ├── rdma.cpp           ← RDMA transport, host MR auto-detection
│   ├── internode_ll.cu    ← Low-latency kernel (sm_89 fallback added)
│   ├── internode.cu       ← Normal-mode kernel (compile-fixed for NVL=1)
│   ├── uccl_ep.cc         ← Python bindings (nanobind)
│   └── ...
├── deep_ep_wrapper/       ← DeepEP-compatible Python API
├── tests/                 ← Test scripts
├── bench/                 ← Benchmarks
└── doc/                   ← Design docs
```

## Building

```bash
# From experimental/lite/ep/
make -j            # Build the Python extension (ep.abi3.so)
make install       # Install to Python site-packages

# With Intel RDMA NIC flags (if needed):
USE_INTEL_RDMA_NIC=1 make -j
```

Toolchain: g++, nvcc (CUDA 12+), C++17, sm_89 (L4).

Dependencies: `nanobind`, `torch`, `libibverbs`, `libnuma`, `libnl-3`.

## Key Design Decisions

### 1. No separate host staging buffer needed

The existing `cudaMallocHost` RDMA buffer is PCIe-mapped and accessible from
both GPU and NIC. The GPU writes directly to this buffer via PCIe, and the NIC
reads from it for RDMA. No explicit `cudaMemcpy` staging is needed for
correctness. This eliminates the need for `host_staging.hpp/.cpp`.

Performance note: explicit DMA batching could improve throughput by reducing
per-access PCIe overhead, but this is deferred to Phase 2 optimization.

### 2. NUM_MAX_NVL_PEERS = 1

Set to 1 so every peer is treated as an RDMA peer. All communication goes
through the proxy RDMA path. No intranode P2P kernel code paths are used.
This is correct because L4s have no NVLink.

### 3. sm_89 kernel compatibility

TMA, mbarrier, and elect.sync are sm_90+ features. In `internode_ll.cu`:
- sm_90+: uses TMA for shared memory staging (original path)
- sm_89: falls back to regular `__ldg`/global stores + `UNROLLED_WARP_COPY`
- LogFMT transform: sm_89 path loads to registers, transforms, stores directly

In `ep_utils.cuh`: all TMA/mbarrier functions gated with `__CUDA_ARCH__ >= 900`.

### 4. Atomic buffer always on host

The atomic signaling buffer is allocated with `cudaHostAlloc` (not
`cudaMalloc`), so RDMA can target it without GDR, and the GPU reads it via
mapped pointer.

## Modification Guidelines

When modifying code:

1. **Kernel changes** (`internode_ll.cu`, `ep_utils.cuh`): guard sm_90+
   features with `__CUDA_ARCH__ >= 900`. Provide sm_89 fallbacks using
   regular global loads/stores. Use `UNROLLED_WARP_COPY` for simple data copies.

2. **RDMA changes** (`rdma.cpp`): the non-GDR path is auto-detected via
   `can_register_gpu_memory_for_rdma()`. No manual flags needed.

3. **NVL_PEERS changes** (`internode.cu`, `buffer.cuh`): code using
   `NUM_MAX_NVL_PEERS` must handle the case where it equals 1. Watch for
   `sizeof(bool) * NVL_PEERS == sizeof(uint64_t)` type assumptions.

4. **Python changes** (`buffer.py`, `utils.py`): the `rdma_buffer_is_host_allocated`
   flag flows from C++ to Python. The NVLink check is skipped for L4/T4/L40.

5. **Include paths**: `util/` headers come from `../../../include/` (the
   top-level uccl repo). Do not duplicate them.

## Compile Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `PCIE_INTRANODE` | **1** (on) | Sets `NUM_MAX_NVL_PEERS=1`, all peers are RDMA |
| `USE_DMABUF` | off | DMA-BUF GPU MR (not used — no GDR) |
| `INTEL_RDMA_NIC` | off | Intel irdma support (not needed for mlx5) |
| `ATOMICS_USE_HOST_MEMORY` | on | Atomic buffer on host (auto-enabled) |
| `SM` | auto | GPU architecture (89 for L4) |

## Testing

```bash
# PCIe DMA benchmark on single node:
cd tests && make && ./gpu_to_cpu_bench

# Internode test (2 nodes, 4 GPUs each):
# Node 0:
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
  --master_addr=4.14.153.89 --master_port=12355 \
  tests/test_internode_simple.py

# Node 1:
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 \
  --master_addr=4.14.153.89 --master_port=12355 \
  tests/test_internode_simple.py
```

## Performance Expectations

| Metric | NVLink+GDR (H100) | PCIe+non-GDR (L4) | Ratio |
|--------|-------------------|--------------------|-------|
| Intranode BW | 450 GB/s | ~25 GB/s | 18× |
| Internode BW | 50 GB/s | ~25 GB/s | 2× |
| LL dispatch latency | ~5 μs | ~30-60 μs | 6-12× |

The goal is **correctness first, performance second**. Getting DeepEP running
on commodity hardware at any speed is the primary milestone.

## Implementation Phases

### Phase 0: Setup ✅
- [x] Copy ep codebase to experimental/lite/ep
- [x] Write AGENTS.md
- [x] Verify baseline build (sm_89, `PCIE_INTRANODE=1`)

### Phase 1: Host-Staged Buffer Infrastructure ✅
- [x] Host staging: existing `rdma_buffer_is_host_allocated` path works
- [x] `can_register_gpu_memory_for_rdma()` auto-detects non-GDR and falls back
- [x] `cudaMallocHost` buffer is PCIe-mapped — GPU and NIC both access it

### Phase 2: Proxy Modifications (DEFERRED — not needed for correctness)
- [x] Sender proxy: works as-is (host buffer is directly NIC-readable)
- [x] Receiver proxy: works as-is (host buffer is PCIe-mapped to GPU)
- [ ] Pipeline optimization: future work for batching PCIe DMA

### Phase 3: Intranode + NVLink ✅
- [x] `PCIE_INTRANODE` compile flag → `NUM_MAX_NVL_PEERS=1`
- [x] All peers treated as RDMA peers (no intranode P2P used)
- [x] `check_nvlink_connections()` skips NVLink check for L4/T4/L40
- [x] Fixed `AsymBuffer` and static asserts for `NUM_MAX_NVL_PEERS=1`

### Phase 4: GPU Kernel Adaptations ✅
- [x] sm_89 fallback for TMA in `internode_ll.cu` (regular global loads/stores)
- [x] sm_89 fallback for TMA/mbarrier in `ep_utils.cuh`
- [x] `elect_one_sync` fallback (`lane_id == 0`)
- [x] `get_ipc_p2p_ptr` returns 0 for all cross-rank (correct for no NVLink)
- [x] Signaling via host-mapped atomics (existing path)

### Phase 4.5: Shared PD Fix ✅
- [x] Root cause: 4 proxy threads each opening independent ibv_context/PD/MR
      for the same GPU memory, causing nvidia_peermem rkey/PD mismatches
- [x] Fix: Made shared RDMA resource cache unconditional (not just DMABUF)
- [x] All proxy threads share one ibv_context, one PD, one MR, one rkey
- [x] Each thread still has its own CQ and QPs under the shared PD
- [x] Cleanup: `release_shared_rdma_resources()` called unconditionally
- [x] Declaration in `rdma.hpp` moved out of `#ifdef USE_DMABUF`

### Phase 4.6: RDMA Atomic Workaround ✅
- [x] Root cause: `IBV_WR_ATOMIC_FETCH_AND_ADD` to `cudaHostAlloc` memory
      fails with `IBV_WC_REM_ACCESS_ERR` (status=10) on mlx5 cross-node
- [x] Regular RDMA_WRITE works fine; only atomics fail
- [x] NIC capabilities and QP/MR access flags are all correct
- [x] Fix: Replace FETCH_AND_ADD with `IBV_WR_RDMA_WRITE` for signaling
- [x] Each atomic slot is written by exactly one sender per dispatch →
      no atomicity required, RDMA_WRITE is sufficient
- [x] `IBV_SEND_INLINE` is critical: without it, the scratch buffer gets
      reused before the NIC reads it, causing data corruption in ≥4 GPU configs
- [x] Applied to both fast-mode and normal-mode atomic paths

### Phase 5: Testing & Benchmarking ✅
- [x] Single-node 2-GPU test — PASSED ✅
- [x] Single-node 4-GPU test — PASSED ✅
- [x] Single-node benchmarks — PASSED ✅ (see results below)
- [x] Multi-node 2n×1g — PASSED ✅ (all 16 experiment configs)
- [x] Multi-node 2n×4g (8 GPUs) — PASSED ✅ (all 32 experiment configs)
- [x] Full 8-GPU benchmark with performance numbers

### Phase 6: Documentation ✅
- [x] AGENTS.md (this file)
- [x] Multi-node launch scripts (`run_multinode.sh`)
- [x] Architecture doc (`doc/architecture.md`)

## Benchmark Results (Single-Node, L4 GPUs)

All tests pass correctness verification. Clean exit (no segfaults).

### 4-GPU Benchmarks

| Config | Dispatch+Combine BW | Dispatch BW | Combine BW | D+C Latency |
|--------|---------------------|-------------|------------|-------------|
| 128 tok × 2048 hid × 16 exp | 3.32 GB/s | 3.09 GB/s | 3.46 GB/s | 942 μs |
| 256 tok × 4096 hid × 16 exp | 3.65 GB/s | 3.70 GB/s | 3.71 GB/s | 3,451 μs |
| 512 tok × 7168 hid × 16 exp | 3.63 GB/s | 3.97 GB/s | 3.54 GB/s | 12,224 μs |

### 2-GPU Benchmarks

| Config | Dispatch+Combine BW | Dispatch BW | Combine BW | D+C Latency |
|--------|---------------------|-------------|------------|-------------|
| 128 tok × 2048 hid × 8 exp | 5.49 GB/s | 3.78 GB/s | 4.65 GB/s | 279 μs |
| 256 tok × 7168 hid × 8 exp | 5.36 GB/s | 5.42 GB/s | 5.44 GB/s | 2,037 μs |

### Performance Notes

- 2-GPU achieves higher bandwidth per rank because less contention on the
  single PCIe root complex / RDMA NIC
- 4-GPU bandwidth is limited by shared PCIe Gen4 interconnect (~32 GB/s)
  and single mlx5 NIC
- Dispatch latency is dominated by GPU→host PCIe transfers (~80-150 μs send time)
- Combine latency includes recv wait time (varies by rank due to scheduling)

## Benchmark Results (Multi-Node, 2 × 4 L4 GPUs)

All correctness tests passed on all 8 ranks.

### 2n×1g (2 nodes, 1 GPU each)

| Config | Dispatch+Combine BW | Dispatch BW | Combine BW | D+C Latency |
|--------|---------------------|-------------|------------|-------------|
| 128 tok × 2048 hid × 8 exp | 14.81 GB/s | 6.20 GB/s | 13.68 GB/s | 211 μs |

### 2n×4g (2 nodes, 4 GPUs each = 8 GPUs total)

| Config | Dispatch+Combine BW | Dispatch BW | Combine BW | D+C Latency |
|--------|---------------------|-------------|------------|-------------|
| 128 tok × 2048 hid × 8 exp | 1.02 GB/s | ~0.96 GB/s | ~0.96 GB/s | 3,058 μs |

### Multi-Node Performance Notes

- 2n×1g achieves excellent bandwidth — limited mainly by PCIe latency
- 2n×4g bandwidth drops due to NIC contention: all 4 GPUs share one mlx5 NIC
- RDMA FETCH_AND_ADD was replaced with RDMA_WRITE for signaling (see Phase 4.6)
- Intra-node communication uses RDMA loopback (no NVLink)

## Quick Start

```bash
# Build
cd experimental/lite/ep
conda activate uccl
make -j SM=89

# Install
cp ep.abi3.so /home/yangz/nfs/miniconda3/envs/uccl/lib/python3.12/site-packages/uccl/

# Single-node test (4 GPUs)
torchrun --nproc_per_node=4 bench/test_internode_simple.py

# Multi-node test (requires SSH to l41)
# Step 1: On l41, run: bash /home/yangz/nfs/zhongjie/uccl/experimental/lite/ep/setup_l41_ssh.sh
# Step 2: On l40, run: bash run_multinode.sh
```

## Changes Summary

### Files Modified (from uccl/ep baseline)

| File | Change | Why |
|------|--------|-----|
| `Makefile` | Added `PCIE_INTRANODE` flag (default=1), `DISABLE_SM90_FEATURES`, fixed include paths | Build standalone in experimental/lite/ep |
| `include/ep_configs.cuh` | `NUM_MAX_NVL_PEERS` = 1 when `PCIE_INTRANODE`; always include `cuda_fp8.h` | All peers are RDMA peers; L4 has native FP8 |
| `include/ep_utils.cuh` | `__CUDA_ARCH__ >= 900` guards on TMA/mbarrier/elect | sm_89 (L4) compatibility |
| `include/ep_launch.cuh` | `DISABLE_SM90_FEATURES` path uses `cudaLaunchKernelEx` with cooperative attr | sm_89 host-side launch |
| `include/buffer.cuh` | Relaxed `kNumRanks > 1` asserts to `>= 1` | Support `NUM_MAX_NVL_PEERS=1` |
| `include/rdma.hpp` | Moved `release_shared_rdma_resources()` decl out of `#ifdef USE_DMABUF` | Shared PD cleanup for all builds |
| `src/internode.cu` | Fixed static asserts, guarded uint64_t NVL cast | Compile with `NVL_PEERS=1` |
| `src/internode_ll.cu` | Added sm_89 fallback path (global loads instead of TMA) | Low-latency kernel on L4 |
| `src/proxy.cpp` | Loopback RDMA for same-IP peers; unconditional `release_shared_rdma_resources()` | PCIE_INTRANODE same-node RDMA + cleanup fix |
| `src/rdma.cpp` | Shared PD/MR cache unconditional; FETCH_AND_ADD→RDMA_WRITE+INLINE for atomics | Fix rkey/PD mismatch + cross-node atomic failure |
| `src/uccl_ep.cc` | Added `get_num_max_nvl_peers()`; auto-detect GPU/host alloc | Runtime NVL_PEERS query + GDR detection |
| `bench/utils.py` | PCIE_INTRANODE-aware node_idx/num_nodes; Tensor→int fix | Correct proxy setup for NVL_PEERS=1 |
| `bench/buffer.py` | PCIE_INTRANODE-aware effective_local_world_size | Correct Buffer init for NVL_PEERS=1 |
| `bench/test_internode_simple.py` | Rewritten to use Buffer internal proxy mgmt | Avoid double-init |
| `deep_ep_wrapper/deep_ep/utils.py` | Skip NVLink check for L4/T4/L40 GPUs | No NVLink available |

## Multi-Node: How to Run

SSH to l41 is currently broken (uccl-dev key not in l41's `~/.ssh/authorized_keys`).

### Fix SSH

On l41 (requires physical or console access):
```bash
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOtluFz7491tDMxrGJpg7ViEFyoRhCOMSOOyfdVo5h2M yangz@mibura-sky-test-01" >> ~/.ssh/authorized_keys
```

Or run: `bash /home/yangz/nfs/zhongjie/fix_l41_ssh.sh`

### Run Multi-Node Test

```bash
cd experimental/lite/ep

# Simple test (1 GPU per node)
bash run_multinode.sh --gpus-per-node 1

# Full benchmark (4 GPUs per node)
bash run_multinode.sh --test bench/test_low_latency.py \
    --test-args "--num-tokens=128 --hidden=2048 --num-topk=4 --num-experts=8 --disable-nvlink"
```

### Manual Multi-Node Launch

```bash
# On l41 (via SSH or direct console):
source ~/zhongjie/zj_py/bin/activate
cd /home/yangz/nfs/zhongjie/uccl/experimental/lite/ep
cp ep.abi3.so ~/zhongjie/zj_py/lib/python3.13/site-packages/uccl/ep.abi3.so
NCCL_IB_HCA=mlx5_0 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nnodes=2 --nproc_per_node=4 --node_rank=1 \
    --master_addr=4.14.153.89 --master_port=12356 \
    bench/test_low_latency.py --num-tokens=128 --hidden=2048 --num-topk=4 --num-experts=8 --disable-nvlink

# On l40:
conda activate uccl
cd /home/yangz/nfs/zhongjie/uccl/experimental/lite/ep
cp ep.abi3.so /home/yangz/nfs/miniconda3/envs/uccl/lib/python3.12/site-packages/uccl/ep.abi3.so
NCCL_IB_HCA=mlx5_0 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nnodes=2 --nproc_per_node=4 --node_rank=0 \
    --master_addr=4.14.153.89 --master_port=12356 \
    bench/test_low_latency.py --num-tokens=128 --hidden=2048 --num-topk=4 --num-experts=8 --disable-nvlink
```
