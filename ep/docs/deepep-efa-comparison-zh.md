# DeepEP 在 EFA 上运行的 4 种技术方案对比

## 背景

[DeepEP](https://github.com/deepseek-ai/DeepEP) 是 DeepSeek 开源的 GPU 发起的专家并行（Expert Parallel）通信库，是高效大规模 MoE 推理/训练的关键组件。然而，DeepEP 深度依赖 NVIDIA IBGDA（InfiniBand GPU Direct Async），只能在 InfiniBand 网络上运行，无法在 AWS EFA（Elastic Fabric Adapter）等其他 RDMA 网络上使用。

本文对比 4 种让 DeepEP 功能在 EFA 上工作的技术方案：

1. **NCCL GIN** — DeepEP PR #521，用 NCCL Device API 替代 NVSHMEM
2. **UCCL-EP** — 完全自研的跨平台 EP 通信库
3. **修改 NVSHMEM** — 让 NVSHMEM 支持 EFA transport
4. **pplx-garden** — Perplexity 开源的 RDMA 通信方案

## 方案总览

| | NCCL GIN (PR #521) | UCCL-EP | NVSHMEM 改造 | pplx-garden |
|--|---|---|---|---|
| **GPU→NIC 通路** | NCCL Device API → CPU proxy | GPU→FIFO→CPU proxy→ibverbs | NVSHMEM API → UCX proxy | GPU kernel → CPU (Rust) → libfabric |
| **EFA 上能否 GPU-direct** | 不能 (GDAKI 需要 IBGDA) | 不能 | 未来可能 (efa-dp-direct) | 不能 |
| **EFA 已验证** | 未验证（有已知兼容性 bug） | 已验证，多平台 | 未验证 | 已验证，有性能数据 |
| **支持 AMD GPU** | 不支持 (NCCL 限制) | 支持 | 不支持 (NVSHMEM 限制) | 不支持 |
| **API 兼容 DeepEP** | 是（同一代码库） | 是（兼容接口） | 是（同一代码库） | 否（独立 API） |
| **实现语言** | C++/CUDA | C++/CUDA | C++/CUDA | Rust + CUDA |
| **开源状态** | PR 未合并 | 已开源 | 需自行改造 | 已开源 |

## 性能对比

### Low-Latency 模式 (16 EP, 128 tokens, 7168 hidden, top-8, FP8 dispatch + BF16 combine)

| 方案 | Dispatch 延迟 | Dispatch BW | Combine 延迟 | Combine BW | D+C 总延迟 | 硬件 |
|------|-------------|------------|-------------|------------|-----------|------|
| **DeepEP** (IBGDA) | 118 us | 63 GB/s | 195 us | 74 GB/s | ~313 us | H800 + CX7 IB |
| **pplx-garden** CX7 | 110 us | - | 186 us | - | ~296 us | H100 + CX7 IB |
| **pplx-garden** EFA | 215 us | - | 242 us | - | ~457 us | H100 + EFA |
| **pplx-garden** EFA (实测) | 145 us | 52.4 GB/s | 221 us | 66.7 GB/s | **~366 us** | B200 + 400G EFA |
| **UCCL-EP** (README) | 228 us | 33 GB/s | 318 us | 46 GB/s | ~546 us | B200 + 400G EFA |
| **UCCL-EP** (实测) | - | - | - | - | ~504 us | B200 + 400G EFA |

### Normal 模式 (16 EP, 4096 tokens, 7168 hidden, top-8)

| 方案 | Dispatch 延迟 | Dispatch BW | Combine 延迟 | Combine BW | D+C 总延迟 | 硬件 |
|------|-------------|------------|-------------|------------|-----------|------|
| **DeepEP** (IBGDA) | - | 43-58 GB/s | - | 类似 | - | H800 + CX7 IB |
| **pplx-garden** EFA (官方) | 3197 us | - | 5379 us | - | ~8576 us | H100 + EFA |
| **pplx-garden** EFA (实测) | 2905 us | 83.4 GB/s | 5193 us | 90.6 GB/s | **~8098 us** | B200 + 400G EFA |
| **UCCL-EP** (实测) | - | 49.7 GB/s | - | 57.7 GB/s | - | B200 + 400G EFA |

> Normal 模式下，pplx-garden 在相同 B200 硬件上实现了更高的带宽（83-91 GB/s）vs UCCL-EP（50-58 GB/s）。这可能是因为 pplx-garden 使用 libfabric（EFA 原生）而 UCCL-EP 使用 ibverbs 路径，以及 pplx-garden 的节点内 NVLink 传输。
>
> 注意：pplx-garden 和 UCCL-EP 使用不同的 benchmark 方法和参数（如 288 vs 256 专家数），带宽直接对比需谨慎解读。

## 各方案详细分析

### 方案 1：NCCL GIN（DeepEP PR #521）

**原理**：在 DeepEP 中引入 `CommunicationBackend` 抽象层，将 NVSHMEM 的 PGAS 内存模型映射到 NCCL 的 window-based 模型。GPU kernel 通过 NCCL Device API（`put()`、`signal()`、`flush()` 等）发起网络操作。

**NCCL GIN 有两种模式**：

| GIN 模式 | 机制 | EFA 支持 |
|---------|------|---------|
| `NCCL_GIN_TYPE=3` (GDAKI) | GPU 直接操作 NIC doorbell | 不支持（需要 IBGDA） |
| `NCCL_GIN_TYPE=2` (Proxy) | GPU→CPU proxy→NIC | 理论上支持 |

**EFA 上的问题**：
- GIN Proxy 模式在 EFA 上有已知兼容性问题（[NCCL #1913](https://github.com/NVIDIA/nccl/issues/1913)、[#1921](https://github.com/NVIDIA/nccl/issues/1921)）：EFA 多 rail 拓扑不一致导致初始化失败
- 需要设置 `OFI_NCCL_FORCE_NUM_RAILS=4` workaround（[aws-ofi-nccl #1061](https://github.com/aws/aws-ofi-nccl/issues/1061)）
- PR 只在 H100 + InfiniBand 上测试过，EFA 上完全未验证
- 依赖 NCCL 2.28.9+

**结论**：理论可行但实际坑多。EFA 上只能用 Proxy 模式，性能不会比 UCCL-EP 好。兼容性问题尚未解决。

### 方案 2：UCCL-EP

**原理**：完全自研的 EP 通信库，GPU kernel 将 RDMA 命令写入 FIFO，CPU proxy 线程读取并通过 ibverbs 发起 RDMA 操作。针对 EP 场景深度优化 proxy 路径。

**优势**：
- **最广泛的硬件支持**：EFA、CX7 InfiniBand、Broadcom Thor-2、AMD Pollara
- **唯一支持 AMD GPU** 的方案（CUDA + HIP）
- API 兼容 DeepEP，可直接替换
- 已在 p5en (H200)、p6-b200 (B200)、MI300X 等多平台验证
- Normal 模式性能与 DeepEP 原版持平

**不足**：
- LL 模式因 CPU proxy 开销，延迟比 IBGDA 方案高 ~1.6x

**结论**：当前最成熟、最可靠的 EFA 方案。跨平台兼容性最强。

### 方案 3：修改 NVSHMEM 支持 EFA

**原理**：NVSHMEM 除了 IBGDA transport 外，还有 UCX transport（UCX → libfabric → EFA）和 ibrc transport。理论上可以让 DeepEP 在非 IBGDA 环境下通过这些 transport 工作。

**可行路径**：

| 路径 | 描述 | 可行性 |
|------|------|--------|
| **A. UCX/ibrc transport** | DeepEP kernel 中的 `nvshmem_put_nbi` 等调用走 host proxy | 需要大改 DeepEP kernel，性能退化到 CPU proxy 水平 |
| **B. efa-dp-direct 集成** | Amazon 的 [efa-dp-direct](https://github.com/amzn/efa-dp-direct) 提供 GPU 直接操作 EFA queue pair 的能力，如果集成进 NVSHMEM 作为新 transport，可在 EFA 上实现真正的 GPU-direct | efa-dp-direct 还很早期（2025 年 10 月开源，3 star） |

**efa-dp-direct 的关键意义**：
- 提供了从 CUDA kernel 直接发送 EFA work request、轮询 completion queue 的 device API
- 如果成熟并集成进 NVSHMEM，是 **唯一可能在 EFA 上达到 DeepEP IBGDA 原版 LL 性能** 的路径
- NVSHMEM issue [#4](https://github.com/NVIDIA/nvshmem/issues/4) 中已有指向该项目的讨论

**结论**：短期不现实（需要 efa-dp-direct 成熟 + NVSHMEM 集成 + DeepEP 适配），但 **长期最有潜力**。

### 方案 4：pplx-garden（Perplexity AI）

**原理**：Perplexity AI 开源的推理通信库，用 Rust 实现的 RDMA TransferEngine（支持 libfabric/EFA 和 libibverbs/CX7），配合定制的 CUDA dispatch/combine kernel。

**项目地址**：[github.com/perplexityai/pplx-garden](https://github.com/perplexityai/pplx-garden)

**优势**：
- 原生支持 EFA 和 CX7，有完整的性能对比数据
- EFA 上 LL 性能在现有方案中最好（16EP D+C ~366 us on B200，~457 us on H100）
- DeepEP 官方维护者推荐作为 EFA 替代方案（[issue #369](https://github.com/deepseek-ai/DeepEP/issues/369)）
- 支持 split send/recv 以及 micro-batching，RDMA 传输期间 SM-free

**不足**：
- API 不兼容 DeepEP，是完全独立的项目
- Rust 生态，与现有 Python/C++ 推理框架集成需额外工作
- 不支持 AMD GPU
- 社区较小（370 star）

**结论**：EFA 上 LL 性能最好的现有开源方案，但生态兼容性差，适合愿意自行集成的团队。

## 根因分析：为什么 EFA 上 LL 延迟更高？

所有 EFA 方案在 LL 模式下都比 DeepEP IBGDA 慢，根因是：

```
DeepEP (IBGDA):  GPU kernel → NIC doorbell  (~118 us dispatch)
                 零 CPU 参与

EFA 方案:        GPU kernel → CPU proxy → EFA NIC  (~215-228 us dispatch)
                 额外一跳 GPU↔CPU 通信开销
```

EFA 不支持 IBGDA/GDAKI，GPU 无法直接操作 NIC，必须通过 CPU 转发。这个额外的 GPU→CPU→NIC 路径是所有 EFA 方案的固有瓶颈。

**唯一的破局点**是 Amazon 的 `efa-dp-direct`——如果它成熟并被集成到通信库（NVSHMEM 或其他）中，EFA 上也能实现 GPU-direct RDMA，从而消除 CPU proxy 瓶颈。

## 推荐

| 需求场景 | 推荐方案 | 理由 |
|---------|---------|------|
| 现在就要在 EFA 上跑 EP | **UCCL-EP** | 最成熟，API 兼容 DeepEP |
| 追求 EFA 上最低 LL 延迟 | **pplx-garden** | EFA LL 性能最好 (~366 us vs ~504 us on B200) |
| 需要 AMD GPU 支持 | **UCCL-EP** | 唯一支持 AMD 的方案 |
| 想保持 DeepEP 代码不改 | **UCCL-EP** > NCCL GIN | UCCL-EP API 兼容；NCCL GIN 兼容性问题多 |
| 长期投资 GPU-direct on EFA | 关注 **efa-dp-direct** + NVSHMEM | 唯一可能达到 IBGDA 级 LL 性能的路径 |

## 参考链接

- [DeepEP](https://github.com/deepseek-ai/DeepEP)
- [UCCL-EP](https://github.com/uccl-project/uccl/tree/main/ep)
- [DeepEP NCCL PR #521](https://github.com/deepseek-ai/DeepEP/pull/521)
- [pplx-garden](https://github.com/perplexityai/pplx-garden)
- [efa-dp-direct](https://github.com/amzn/efa-dp-direct)
- [NVSHMEM](https://github.com/NVIDIA/nvshmem)
- [NCCL GIN + EFA 兼容性问题](https://github.com/NVIDIA/nccl/issues/1913)
- [DeepEP EFA 讨论](https://github.com/deepseek-ai/DeepEP/issues/369)
- [GPU-Initiated Networking for NCCL (论文)](https://arxiv.org/abs/2511.15076)
