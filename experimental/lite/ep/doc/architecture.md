# DeepEP-Lite Architecture

## Overview

DeepEP-Lite enables DeepEP (expert-parallel MoE communication) to run on
hardware without NVLink or GPUDirect RDMA. It is a modified fork of uccl-ep
(`ep/`) placed in `experimental/lite/ep/`.

## Key Insight: Minimal Changes Required

The existing uccl-ep codebase already supports host-allocated RDMA buffers.
The main changes for non-GDR, non-NVLink operation are:

1. **sm_89 kernel compatibility** — TMA/mbarrier fallbacks
2. **NUM_MAX_NVL_PEERS=1** — treat all peers as RDMA peers
3. **NVLink check bypass** — skip for GPUs without NVLink

No proxy modifications, no new buffer managers, no new data path code.

## Data Path

### How GPU Data Reaches Remote GPU (Non-GDR)

```
Sender GPU                              Receiver GPU
    │                                       ▲
    │ PCIe write (mapped ptr)               │ PCIe read (mapped ptr)
    ▼                                       │
Host RDMA Buffer (cudaMallocHost)    Host RDMA Buffer (cudaMallocHost)
    │                                       ▲
    │ ibv_post_send (RDMA WRITE)            │ (NIC writes directly)
    ▼                                       │
    ╌╌╌╌╌╌╌╌╌╌ RDMA Network ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
```

### Why This Works Without GDR

1. `cudaMallocHost()` creates pinned host memory accessible from both GPU and CPU
2. The GPU can read/write this memory via PCIe-mapped pointers (slow but correct)
3. The RDMA NIC can read/write host memory natively (no GDR needed)
4. The MR is registered with `ibv_reg_mr()` on host memory (standard verbs)

### Auto-Detection Flow

```
allocate_rdma_buffer_dlpack()
  ├─ can_register_gpu_memory_for_rdma()?
  │   ├─ YES → cudaMalloc + ibv_reg_mr (GDR/DMA-BUF path)
  │   └─ NO  → cudaMallocHost + ibv_reg_mr (host path) ← our path
  │
per_thread_rdma_init()
  ├─ is_cuda_host_pointer(gpu_buf)?
  │   ├─ YES → ibv_reg_mr (standard host MR) ← our path
  │   └─ NO  → reg_mr_gpu_dmabuf (GPU MR)
```

## sm_89 Kernel Compatibility

### TMA (Tensor Memory Accelerator) — sm_90+ only

TMA provides async bulk memory copy between global and shared memory.
On sm_89, we replace TMA with regular global memory loads/stores:

| sm_90+ (TMA) | sm_89 (fallback) |
|---|---|
| `tma_load_1d` → shared mem | `__ldg` → registers |
| `tma_store_1d` ← shared mem | global store ← registers |
| `mbarrier_wait` for sync | `__syncwarp` |
| `elect_one_sync` | `lane_id == 0` |

### internode_ll.cu Dispatch Kernel

The low-latency dispatch kernel copies token data to the RDMA send buffer,
then issues RDMA WRITEs via the D2H command queue:

```
sm_90+:
  for each token:
    TMA load src → shared mem → transform → TMA store to send buf
    issue RDMA WRITE cmd

sm_89:
  for each token:
    global load src → registers → transform → global store to send buf
    issue RDMA WRITE cmd
```

For the non-LogFMT case (most common), the sm_89 path uses
`UNROLLED_WARP_COPY` — a simple warp-cooperative global memcpy.

## NUM_MAX_NVL_PEERS = 1

With `PCIE_INTRANODE` defined, `NUM_MAX_NVL_PEERS` is set to 1. This means:

- Every GPU is its own "NVLink group" (single GPU per group)
- `num_rdma_ranks = num_ranks / 1 = num_ranks` (all peers are RDMA)
- `nvl_rank = rank % 1 = 0` for every GPU
- Intranode communication uses RDMA path (same as internode)
- `get_ipc_p2p_ptr()` returns 0 for all cross-rank communication

### Impact on Normal Mode (internode.cu)

The normal-mode kernel has deep dependencies on `NUM_MAX_NVL_PEERS=8`:
- `uint64_t` casts assuming 8 bool NVL ranks
- NVL barrier synchronization
- AsymBuffer with multiple NVL ranks

These are compile-fixed (guards and relaxed asserts) but the normal mode
is NOT the primary target. Low-latency mode is the focus.

## Atomic Signaling Path

The atomic buffer (for completion signaling) is always on host memory:

```
Sender: RDMA atomic (FAD) → remote host atomic buffer
Remote GPU: volatile read of host-mapped atomic pointer → sees completion
```

`ATOMICS_USE_HOST_MEMORY` is auto-enabled. The host atomic buffer is
allocated with `cudaHostAlloc` and mapped to GPU with `cudaHostGetDevicePointer`.
GPU volatile reads are coherent for host-mapped memory on L4.

## Performance Characteristics

| Component | Bandwidth | Latency |
|---|---|---|
| L4 PCIe Gen4 x16 | ~25 GB/s | ~1-2 μs |
| 400G RDMA (mlx5) | ~50 GB/s | ~1-2 μs |
| Host RDMA buffer access (GPU side) | ~25 GB/s | ~1-2 μs per access |

Bottleneck: PCIe bandwidth for GPU↔host buffer access.
For small messages (low-latency mode), latency dominates over bandwidth.

## Future Optimizations

1. **Explicit DMA staging**: Use `cudaMemcpyAsync` to batch data from GPU
   working memory to host RDMA buffer, reducing per-access PCIe overhead.
2. **Double-buffered pipeline**: Overlap PCIe DMA and RDMA transfers.
3. **PCIe P2P for intranode**: If L4s support `cudaDeviceCanAccessPeer`,
   use direct P2P instead of routing through host+RDMA.
