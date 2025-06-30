# UCCL GPU-Driven Peer-to-Peer Engine

An efficient and simple prototype that demonstrates **end-to-end GPU-direct peer-to-peer (P2P) data communication** across machines using **GPUDirect RDMA** and a lightweight **CPU proxy**.  

For UCCL's host/CPU-driven P2P engine, see [p2p](../p2p/) folder.

## Overview
1.	Each rank pins its GPU buffer with GPUDirect RDMA and exchanges RDMAConnectionInfo.
2.	Rank 0 writes batched copy commands into a host-mapped ring buffer managed by local CPU proxy.
3.	The CPU proxy polls that ring, posts `IBV_WR_RDMA_WRITE_WITH_IMM`, and recycles WQEs on completion.
4.	Rank 1’s proxy (on the remote node) posts matching receives and funnels completed work into a peer-copy kernel (optional) that pushes data to additional GPUs through NVLink. This step mimicks the requirements in MoE models where a token can be routed to multiple experts on the remote node.

---

## Folder Structure

```text
p2p/                           # ← repo root
├── Makefile                   # standalone build
├── README.md                  
├── benchmark_local.cu         # single-GPU FIFO buffer test
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

---

## Prerequisites

| Component | Requirement |
|-----------|-------------|
| **GPU**   | NVIDIA A100/H100 or any GPU that supports GPUDirect RDMA |
| **NIC**   | Mellanox CX-5/6/7 or equivalent with RoCE/IB support |
| **CUDA**  | 12.2 or newer (tested on 12.4) |

---

## Build

```bash
make               # builds benchmarks and static libs
make clean         # remove objects and binaries
```

## Running benchmarks

### 1. Local single-machine Test

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
