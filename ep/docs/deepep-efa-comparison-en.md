# Comparison of 4 Approaches to Run DeepEP on EFA

## Background

[DeepEP](https://github.com/deepseek-ai/DeepEP) is DeepSeek's open-source GPU-initiated expert-parallel (EP) communication library, a key component for efficient large-scale MoE inference and training. However, DeepEP is tightly coupled with NVIDIA IBGDA (InfiniBand GPU Direct Async) and can only run on InfiniBand networks, making it incompatible with AWS EFA (Elastic Fabric Adapter) and other RDMA networks.

This document compares 4 approaches to enable DeepEP-equivalent functionality on EFA:

1. **NCCL GIN** — DeepEP PR #521, replacing NVSHMEM with NCCL Device API
2. **UCCL-EP** — A fully independent cross-platform EP communication library
3. **Modified NVSHMEM** — Adding EFA transport support to NVSHMEM
4. **pplx-garden** — Perplexity AI's open-source RDMA communication solution

## Overview

| | NCCL GIN (PR #521) | UCCL-EP | Modified NVSHMEM | pplx-garden |
|--|---|---|---|---|
| **GPU-to-NIC path** | NCCL Device API → CPU proxy | GPU→FIFO→CPU proxy→ibverbs | NVSHMEM API → UCX proxy | GPU kernel → CPU (Rust) → libfabric |
| **GPU-direct on EFA** | No (GDAKI requires IBGDA) | No | Potentially (via efa-dp-direct) | No |
| **EFA validated** | No (known compat bugs) | Yes, multi-platform | No | Yes, with benchmarks |
| **AMD GPU support** | No (NCCL limitation) | Yes | No (NVSHMEM limitation) | No |
| **DeepEP API compatible** | Yes (same codebase) | Yes (compatible interface) | Yes (same codebase) | No (independent API) |
| **Language** | C++/CUDA | C++/CUDA | C++/CUDA | Rust + CUDA |
| **Open-source status** | PR not merged | Released | Requires custom work | Released |

## Performance Comparison

### Low-Latency Mode (16 EP, 128 tokens, 7168 hidden, top-8, FP8 dispatch + BF16 combine)

| Approach | Dispatch Latency | Dispatch BW | Combine Latency | Combine BW | D+C Total | Hardware |
|----------|-----------------|-------------|-----------------|------------|-----------|----------|
| **DeepEP** (IBGDA) | 118 us | 63 GB/s | 195 us | 74 GB/s | ~313 us | H800 + CX7 IB |
| **pplx-garden** CX7 | 110 us | - | 186 us | - | ~296 us | H100 + CX7 IB |
| **pplx-garden** EFA | 215 us | - | 242 us | - | ~457 us | H100 + EFA |
| **pplx-garden** EFA (measured) | 145 us | 52.4 GB/s | 221 us | 66.7 GB/s | **~366 us** | B200 + 400G EFA |
| **UCCL-EP** (README) | 228 us | 33 GB/s | 318 us | 46 GB/s | ~546 us | B200 + 400G EFA |
| **UCCL-EP** (measured) | - | - | - | - | ~504 us | B200 + 400G EFA |

### Normal (High-Throughput) Mode (16 EP, 4096 tokens, 7168 hidden, top-8)

| Approach | Dispatch Latency | Dispatch BW | Combine Latency | Combine BW | D+C Total | Hardware |
|----------|-----------------|-------------|-----------------|------------|-----------|----------|
| **DeepEP** (IBGDA) | - | 43-58 GB/s | - | Similar | - | H800 + CX7 IB |
| **pplx-garden** EFA (README) | 3197 us | - | 5379 us | - | ~8576 us | H100 + EFA |
| **pplx-garden** EFA (measured) | 2905 us | 83.4 GB/s | 5193 us | 90.6 GB/s | **~8098 us** | B200 + 400G EFA |
| **UCCL-EP** (measured) | - | 49.7 GB/s | - | 57.7 GB/s | - | B200 + 400G EFA |

> In normal mode, pplx-garden achieves higher bandwidth (83-91 GB/s) than UCCL-EP (50-58 GB/s) on the same B200 hardware. This is likely due to pplx-garden's use of libfabric (native EFA) vs UCCL-EP's ibverbs path, and pplx-garden's NVLink-based intra-node transfer.
>
> Note: pplx-garden and UCCL-EP use different benchmark methodologies and parameters (e.g., 288 vs 256 experts), so direct bandwidth comparisons should be interpreted with caution.

## Detailed Analysis

### Approach 1: NCCL GIN (DeepEP PR #521)

**Mechanism**: Introduces a `CommunicationBackend` abstraction layer in DeepEP, mapping NVSHMEM's PGAS memory model to NCCL's window-based model. GPU kernels issue network operations through the NCCL Device API (`put()`, `signal()`, `flush()`, etc.).

**NCCL GIN has two modes**:

| GIN Mode | Mechanism | EFA Support |
|----------|-----------|-------------|
| `NCCL_GIN_TYPE=3` (GDAKI) | GPU directly operates NIC doorbell | Not supported (requires IBGDA) |
| `NCCL_GIN_TYPE=2` (Proxy) | GPU → CPU proxy → NIC | Theoretically supported |

**Issues on EFA**:
- GIN Proxy mode has known compatibility issues on EFA ([NCCL #1913](https://github.com/NVIDIA/nccl/issues/1913), [#1921](https://github.com/NVIDIA/nccl/issues/1921)): EFA multi-rail topology inconsistencies cause initialization failures
- Requires `OFI_NCCL_FORCE_NUM_RAILS=4` workaround ([aws-ofi-nccl #1061](https://github.com/aws/aws-ofi-nccl/issues/1061))
- PR was only tested on H100 + InfiniBand; completely unvalidated on EFA
- Requires NCCL 2.28.9+

**Verdict**: Theoretically possible but practically problematic. Only Proxy mode works on EFA, which won't outperform UCCL-EP. Compatibility issues remain unresolved.

### Approach 2: UCCL-EP

**Mechanism**: A fully independent EP communication library. GPU kernels write RDMA commands to a FIFO; CPU proxy threads read and issue RDMA operations via ibverbs. The proxy path is deeply optimized for EP communication patterns.

**Strengths**:
- **Broadest hardware support**: EFA, CX7 InfiniBand, Broadcom Thor-2, AMD Pollara
- **Only solution supporting AMD GPUs** (CUDA + HIP)
- DeepEP-compatible API, drop-in replacement
- Validated on p5en (H200), p6-b200 (B200), MI300X, and more
- Normal mode performance on par with DeepEP

**Limitations**:
- LL mode latency is ~1.6x higher than IBGDA due to CPU proxy overhead

**Verdict**: The most mature and reliable EFA solution today. Best cross-platform compatibility.

### Approach 3: Modified NVSHMEM for EFA

**Mechanism**: NVSHMEM includes a UCX transport (UCX → libfabric → EFA) and an ibrc transport alongside IBGDA. In theory, DeepEP could work on non-IBGDA networks through these transports.

**Possible paths**:

| Path | Description | Feasibility |
|------|-------------|-------------|
| **A. UCX/ibrc transport** | `nvshmem_put_nbi` calls from GPU kernels are routed through host proxy | Requires significant DeepEP kernel changes; performance degrades to CPU proxy level |
| **B. efa-dp-direct integration** | Amazon's [efa-dp-direct](https://github.com/amzn/efa-dp-direct) provides GPU-direct access to EFA queue pairs. If integrated into NVSHMEM as a new transport, it could enable true GPU-direct RDMA on EFA | efa-dp-direct is very early-stage (open-sourced Oct 2025, 3 stars) |

**Why efa-dp-direct matters**:
- Provides device-side APIs for CUDA kernels to directly post EFA work requests and poll completion queues
- If mature and integrated into NVSHMEM, it is the **only path that could match DeepEP's IBGDA-level LL performance on EFA**
- Discussed in NVSHMEM issue [#4](https://github.com/NVIDIA/nvshmem/issues/4)

**Verdict**: Not practical short-term (requires efa-dp-direct maturation + NVSHMEM integration + DeepEP adaptation), but **highest long-term potential**.

### Approach 4: pplx-garden (Perplexity AI)

**Mechanism**: Perplexity AI's open-source inference communication library with a Rust-based RDMA TransferEngine (supporting libfabric/EFA and libibverbs/CX7), paired with custom CUDA dispatch/combine kernels.

**Repository**: [github.com/perplexityai/pplx-garden](https://github.com/perplexityai/pplx-garden)

**Strengths**:
- Native EFA and CX7 support with complete benchmark data
- Best LL performance on EFA among existing solutions (16EP D+C ~366 us on B200, ~457 us on H100)
- Recommended by DeepEP maintainers as an EFA alternative ([issue #369](https://github.com/deepseek-ai/DeepEP/issues/369))
- Supports split send/recv and micro-batching; SM-free during RDMA transfers

**Limitations**:
- API incompatible with DeepEP; completely independent project
- Rust ecosystem requires additional integration work with existing Python/C++ inference frameworks
- No AMD GPU support
- Smaller community (370 stars)

**Verdict**: Best LL performance on EFA among existing open-source solutions, but poor ecosystem compatibility. Best suited for teams willing to integrate independently.

## Root Cause: Why is LL Latency Higher on EFA?

All EFA-based approaches show higher LL mode latency than DeepEP with IBGDA. The fundamental reason is:

```
DeepEP (IBGDA):  GPU kernel → NIC doorbell  (~118 us dispatch)
                 Zero CPU involvement

EFA approaches:  GPU kernel → CPU proxy → EFA NIC  (~215-228 us dispatch)
                 Extra GPU↔CPU communication hop
```

EFA does not support IBGDA/GDAKI, so GPUs cannot directly operate the NIC. All EFA solutions must route through a CPU proxy, adding an inherent GPU→CPU→NIC latency overhead.

**The only potential breakthrough** is Amazon's `efa-dp-direct` — if it matures and is integrated into a communication library (NVSHMEM or otherwise), EFA could support true GPU-direct RDMA, eliminating the CPU proxy bottleneck.

## Recommendations

| Use Case | Recommended Approach | Rationale |
|----------|---------------------|-----------|
| Need EP on EFA now | **UCCL-EP** | Most mature, DeepEP-compatible API |
| Lowest LL latency on EFA | **pplx-garden** | Best EFA LL numbers (~366 us vs ~504 us on B200) |
| Need AMD GPU support | **UCCL-EP** | Only solution supporting AMD |
| Keep DeepEP code unchanged | **UCCL-EP** > NCCL GIN | UCCL-EP is API-compatible; NCCL GIN has compat issues |
| Long-term GPU-direct on EFA | Watch **efa-dp-direct** + NVSHMEM | Only path to IBGDA-level LL performance |

## References

- [DeepEP](https://github.com/deepseek-ai/DeepEP)
- [UCCL-EP](https://github.com/uccl-project/uccl/tree/main/ep)
- [DeepEP NCCL PR #521](https://github.com/deepseek-ai/DeepEP/pull/521)
- [pplx-garden](https://github.com/perplexityai/pplx-garden)
- [efa-dp-direct](https://github.com/amzn/efa-dp-direct)
- [NVSHMEM](https://github.com/NVIDIA/nvshmem)
- [NCCL GIN + EFA Compatibility Issues](https://github.com/NVIDIA/nccl/issues/1913)
- [DeepEP EFA Discussion](https://github.com/deepseek-ai/DeepEP/issues/369)
- [GPU-Initiated Networking for NCCL (Paper)](https://arxiv.org/abs/2511.15076)
