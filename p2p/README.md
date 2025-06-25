# UCCL GPU-Driven Peer-to-Peer Engine

A minimal yet performant prototype that demonstrates **end-to-end GPU-direct peer-to-peer (P2P) data movement** across machines using **GPUDirect RDMA** and a lightweight **CPU proxy**.  
The code base is intentionally simple so you can graft the core ideas—ring-buffer command queues, NUMA-aware RDMA queues, and CUDA bulk-copy kernels—into larger UCCL deployments or other distributed-GPU projects.

---

## Features

| Area                    | What’s Implemented |
| ----------------------- | ------------------ |
| **Transport**           | Infiniband verbs + GPUDirect RDMA (`nvidia_peermem` / `nv_peer_mem`) |
| **Memory Path**         | Zero-copy GPU→NIC DMA; optional staged copy via CPU proxy |
| **Concurrency Model**   | Per-thread QPs/CQs, lock-free ring buffers, NUMA pinning |
| **Copy Engine**         | Batched CUDA kernel (`peer_copy_kernel_vec_batched`) that overlaps copy and compute |
| **Benchmarks**          | Local loopback (`benchmark_local`) and two-node RDMA throughput test (`benchmark_remote`) |
| **Build System**        | One-line `make` (targets CUDA 12+, GCC/Clang ≥ 9) |

---

## Folder Structure

```text
p2p/                           # ← repo root
├── Makefile                   # standalone build
├── README.md                  # this file
├── benchmark_local.cu         # single-GPU smoke test
├── benchmark_remote.cu        # two-node RDMA benchmark (rank 0/1)
├── include/                   # public headers
│   ├── common.hpp
│   ├── copy_ring.hpp
│   ├── gpu_kernel.cuh
│   ├── peer_copy*.hpp|.cuh
│   ├── proxy.hpp
│   └── rdma.hpp
└── src/                       # implementation
    ├── common.cpp
    ├── gpu_kernel.cu
    ├── peer_copy*.cu
    ├── peer_copy_worker.cpp
    ├── proxy.cpp
    └── rdma.cpp
```

> **Tip:** All `.cuh` headers compile for both host and device by guarding CUDA-specific code inside `__CUDA_ARCH__` checks.

---

## Prerequisites

| Component | Requirement |
|-----------|-------------|
| **GPU**   | NVIDIA A100/H100 or any GPU that supports GPUDirect RDMA |
| **Driver**| NVIDIA 535+ with `nvidia_peermem` (or legacy `nv_peer_mem`) loaded |
| **NIC**   | Mellanox CX-5/6/7 or equivalent with RoCE/IB support |
| **CUDA**  | 12.2 or newer (tested on 12.5) |
| **OFED**  | MLNX_OFED 5.9+ |
| **OS**    | Linux 5.15+ |

---

## Build

```bash
make               # builds benchmarks and static libs
make clean         # remove objects and binaries
```

## Running benchmarks

### 1. Local Sanity Test (single machine)

```bash
./benchmark_local
```

### 2. Two-Node Test
```bash
# On **sender** node (rank 0)
./benchmark_remote 0 <receiver_ip>

# On **receiver** node (rank 1)
./benchmark_remote 1 <sender_ip>
```

1.	Each rank pins its GPU buffer with GPUDirect RDMA and exchanges RDMAConnectionInfo.
2.	Rank 0 writes batched copy commands into a host-mapped ring buffer.
3.	A CPU proxy polls that ring, posts IBV_WR_RDMA_WRITE_WITH_IMM, and recycles WQEs on completion.
4.	Rank 1’s proxy posts matching receives and funnels completed work into a peer-copy kernel (optional) that pushes data to additional GPUs.