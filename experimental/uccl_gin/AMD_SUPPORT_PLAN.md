# UCCL-GIN AMD (ROCm/HIP) 支持计划

> 状态：计划草案，未开工，未 commit。
> 目标：让 `experimental/uccl_gin` 能在 AMD GPU（CDNA3/CDNA4，gfx942/gfx950）上
> 构建并跑通；参考 `ep/` 现有的 ROCm 支持机制。
> 非目标（本计划范围外，见 Phase 4 决策门）：AMD 上的全源语高性能 EFA 等价路径。

## 0. 先认清三层现状（决定计划的可行边界）

逐文件比对了 `ep/setup.py` 的 ROCm 分支、`include/util/gpu_rt.h` 共享 shim、
standalone 现有代码、以及非 EFA 的 atomic 路径。结论分三层：

### 层 1：transport 子目录 —— 基本 ready（好消息）
`experimental/uccl_gin/transport/*` 是从 `ep/` 逐字节拷来的，**HIP 支持原样保留**：
- `ring_buffer.cuh` 全部 `__HIP_DEVICE_COMPILE__` guard 在，HIP 下自动 include
  `amd_nanosleep.cuh`（`__nanosleep` 的 AMD 宏实现）。
- `common.hpp:4` 已 `#include "util/gpu_rt.h"`（CUDA/HIP runtime shim：
  `gpuMalloc`/`gpuSetDevice`/...），并带 `UCCL_EP_DEFAULT_AGGRESSIVE_ATOMIC` 宏。
- `uccl_proxy.cpp`/`rdma.cpp`/`fifo_util.hpp` 的 HIP guard、dmabuf MR 注册路径都在。
- 这些文件**不需要改源码**，只要 build 系统按 HIP 编它们。

### 层 2：standalone 新写的代码 —— CUDA-only（需要 shim 化）
本轮新写、没经过 ep 的代码用了裸 CUDA API，HIP 下编不过：
- `context.cpp` / `resources.cuh`：裸 `cudaMalloc`/`cudaSetDevice`/`cudaMemcpy`/
  `cudaMemset`（应改用 `util/gpu_rt.h` 的 `gpu*` shim，ep 全程用 shim）。
- `uccl_gin/uccl_gin.cuh` / `uccl_gin_rail.cuh` / `resources.cuh`：~十几处裸
  `__trap()`——HIP 没有这个 intrinsic（`exception.cuh` 已有 NV `asm("trap;")` /
  AMD `abort()` 的现成模式，抽成 `UCCL_GIN_TRAP()` 宏）。
- `uccl_gin.cuh:42` 无条件 `#include <nccl_device.h>` 取 `ncclTeamTagRail/Lsa`——
  RCCL 无 device API，此头在 ROCm 不存在。tag 仅用于 `if constexpr` 分发，定义
  `uccl_gin::RailTag/LsaTag`（NV 下 alias 到 nccl tag）即可解耦。
- `tests/microbench.cu`：顶部无条件 include `<nccl.h>`/`<nccl_device.h>`，
  `UCCL_GIN_WITH_NCCL_GIN=0` 也编不过；NCCL 参考路径要整段 gate。
- `clock64()` HIP 原生有，不是问题。

### 层 3：build 系统 —— 完全没有 AMD 路径（要新建）
`ep/` 怎么支持 AMD（参考实现）：
- **不是手写 hipify**。`ep/setup.py:12` 用 `torch.utils.cpp_extension` 的
  `BuildExtension` + `CUDAExtension`；torch 在 `torch.version.hip` 下**自动
  hipify** 源码（hipify_torch 内置）。`setup.py` 顶层 `if torch.version.cuda:
  ... else: # AMD 分支`。
- AMD 分支干的事（`setup.py:282-375`）：`PYTORCH_ROCM_ARCH`/`rocminfo` 检测
  gfx 架构 → `--offload-arch=gfx94x`；`-DDISABLE_SM90_FEATURES`；
  `-DUCCL_EP_DEFAULT_AGGRESSIVE_ATOMIC=1`（CDNA 上 vanilla acquire/release
  sys-scope atomic 不 drain vector writes，必须 `s_waitcnt vmcnt`）；
  `-DDISABLE_BUILTIN_SHLF_SYNC`；ROCm root 发现（HIP_HOME/rocm-sdk/opt/rocm）。
- **EFA 检测只在 CUDA 分支**——ep 的 AMD 构建本来就是非 EFA。
- standalone 现在是手写 `Makefile` + 裸 `nvcc`：写死 `nvcc -arch=sm_$(SM)`、
  `-DEFA`、`-lefa -lcudart -lcuda`、`nvidia.nccl` wheel。AMD 一条路径都没有。

### 层 4（真正的墙）：atomic 协议是 EFA-shaped
即便构建修通，非 EFA 构建（AMD 必然非 EFA）下：
- `rdma.cpp:2294` `#ifndef EFA assert(false && "Reorderable atomic operations
  should not be triggered")`——UCCL-GIN 的 ordered software atomic
  （PackAtomicWithSeq + receiver reorder buffer）是为 "EFA SRD 乱序 + 无 native
  atomic" 造的；非 EFA receiver 一遇到 reorderable imm 直接断言。
- 受影响源语：`red_add_rel`、`put_tail_add`（piggyback reorderable imm）、
  `put_value`（WRITE_VALUE 只在 EFA normal-mode 分支实现，fast/非 EFA 已 loud
  abort——上轮我们自己加的 guard）。
- 不受影响：`put`（非 EFA 分支是普通 RDMA WRITE）、`quiet`/`flush`（CQE drain
  平台无关）。
- 反观 ep：非 EFA 走 **native RDMA atomic**（`rdma.cpp:3130`
  `post_atomic_operations_native_rdma`，`FETCH_AND_ADD` 到远端 atomic buffer），
  RC QP 本身保序、硬件 atomic，根本不需要 seq reorder。这是和 EFA 完全不同的
  另一条后端。

**因此**：device API（put/quiet/D2H ring/proxy 框架）是平台无关的，但
**Rail 后端的 atomic 半边是 EFA 专属**。AMD 全源语 = 需要一个 IB/RoCE-native-atomic
flavor，这是 Phase 4 的独立子项目，不是移植补丁。

## 1. 关键前提：AMD 机器的网络栈是什么？（开工前必须定）
计划的形态完全取决于目标 AMD 机器：
- **若只有 xGMI/单节点**：只需 intranode，跨节点 RDMA 无关——但 UCCL-GIN 本就是
  scale-out 方案，单节点没意义。
- **若 MI300X + Broadcom/RoCE 或 IB**：走 native verbs + native atomic，对应 ep
  的非 EFA 路径。这是最可能的目标，Phase 4 按这个设计。
- **AMD 上没有 EFA**：EFA 是 AWS 专属，AMD 实例不会有。
→ **行动**：先确认目标机器型号、NIC、ROCm 版本、是否多机，再决定做到 Phase 几。
  （AGENTS 里 AMD 不在 p5en/EFA 验收路径上，这是好奇心/可移植性驱动，不挡主线。）

## 2. 分阶段

### Phase 0：边界冻结 + 目标机器确认
1. 确认 AMD 目标：GPU 型号/gfx 架构、ROCm 版本、NIC 类型、单机还是多机。
2. 决定构建方式二选一（见 Phase 1）。
3. 产出："AMD 目标环境清单" 写入 worklog；据此确定本计划做到 Phase 2（只 put/quiet）
   还是 Phase 4（全源语 native atomic）。

### Phase 1：build 系统支持 HIP（不改协议）
**方式 A（推荐，跟 ep 一致）**：放弃手写 Makefile，给 standalone 写一个
`setup.py`，照搬 ep 的 `BuildExtension`+`CUDAExtension`+`torch.version.hip` 分支。
torch 自动 hipify，省掉手写 hipify。代价：引入 torch 构建依赖（standalone 当前
刻意无 torch）。
**方式 B（保留 Makefile）**：Makefile 加 `UCCL_GIN_PLATFORM=rocm` 分支：
`hipcc`、`--offload-arch=$(GFX)`、`-D__HIP_PLATFORM_AMD__`、ROCm include/lib、
RCCL 替 NCCL；源码 hipify 靠 HIP headers 的 CUDA→HIP 宏映射（`hip_runtime.h` +
`cuda_compat` 风格）或显式 `hipify-perl` 预处理。代价：要自己维护 hipify。
→ 默认选 **A**，除非要保持 standalone 零 torch 依赖。

验收：HIP 构建产出 `uccl_gin_microbench` + `_uccl_gin.so`（先不要求跑通跨节点）。

### Phase 2：device 层 shim 化（层 2 的活）
1. 加 `UCCL_GIN_TRAP()` 宏（仿 `exception.cuh`：NV `__trap()` / AMD `abort()`），
   替换 `uccl_gin/*.cuh` + `resources.cuh` 全部裸 `__trap()`。
2. `context.cpp`/`resources.cuh` 裸 `cuda*` → `util/gpu_rt.h` 的 `gpu*` shim。
3. team tag 解耦：定义 `uccl_gin::RailTag`/`LsaTag`，NV 下 alias `ncclTeamTag*`；
   `uccl_gin.cuh` 不再无条件 include `<nccl_device.h>`（移到 `#if !ROCm` 或
   NCCL 参考路径里）。
4. `microbench.cu`：把 `<nccl.h>`/`<nccl_device.h>` 和 NCCL 参考 kernel 全部
   收进 `#if UCCL_GIN_WITH_NCCL_GIN`（NV-only），AMD 下只编 UCCL 路径。
5. 受 layer-4 限制，AMD 构建默认 `red_add_rel`/`put_tail_add`/`put_value` 在
   host 端或编译期 loud reject（`#error` 或 runtime abort），**只暴露 put/quiet**。

验收：AMD 单机能 import `_uccl_gin`、create/destroy Context（intranode 形态，
proxy 起停干净）；microbench `--only` 跑 put+quiet 的 correctness（完成信号改用
轮询 recv 数据或 host barrier，不用 red_add）。

### Phase 3：AMD 多机 put/quiet 跑通（非 EFA verbs 数据面）
1. 确认 standalone transport 的非 EFA 数据 WRITE 路径（`rdma.cpp` `#ifndef EFA`
   分支 + `SOFTWARE_ORDERING`）在目标 NIC（RoCE/IB）上能 post RC WRITE。
2. peer exchange / QP 建链按 ep 非 EFA 路径核对（RC QP vs EFA SRD）。
3. 验收：2 节点 AMD，put + quiet correctness（rank-tagged pattern 逐字节校验，
   quiet 后复用 source）。

### Phase 4（决策门，默认不做）：native-atomic Rail flavor
让 `red_add_rel`/`put_tail_add` 在非 EFA 上走 native RDMA atomic（参考
`rdma.cpp:3130` `post_atomic_operations_native_rdma`，FETCH_AND_ADD），
`put_tail_add` 退化为 WRITE + 顺序 WRITE_WITH_IMM/native atomic（RC 保序，无需
seq reorder）。`put_value` 在非 EFA 用普通 WRITE（无 bounce）。这是和 EFA 路径
并列的第二个后端 flavor，按 `#ifdef EFA` 选型。
- 需在本计划或专门设计文档写明：为何 AMD 路径和 EFA 路径的 atomic 实现不同
  （AGENTS 硬要求：偏离原 transport 方法须给理由——这里理由是 native atomic +
  RC ordering 使 EFA 的 software-seq 路径成为不必要）。
- 验收：AMD 2 节点全 5 源语 correctness（对齐现有 EFA microbench gate）。

## 3. 风险 / 决策点
1. **目标机器未知**：整个计划的范围（Phase 2 vs Phase 4）悬而未决，Phase 0 必须
   先定。
2. **方式 A 引入 torch 依赖**：与 standalone "零 torch、纯 Makefile" 的设计取向
   冲突；若坚持零依赖走方式 B，hipify 维护成本转移给我们。
3. **aggressive atomic**：ep 在 AMD 默认开 `UCCL_EP_DEFAULT_AGGRESSIVE_ATOMIC=1`
   是为 V1 kernel 的跨 GPU tail store；UCCL-GIN 的 tail 在 host atomic buffer，
   这个宏对 GIN 路径是否需要要实测（standalone common.hpp 已带宏，默认 0）。
4. **CDNA 内存序**：`__threadfence_system()` 在 `atomic_set_and_commit` 的 D2H
   commit 上，CDNA 的 sys-scope fence 语义与 Hopper 不同，多生产者并发下要专门
   验证 D2H ring 的可见性（和 NCCL-EP 计划的多生产者 stress gate 同源）。
5. **RCCL vs NCCL**：NCCL 参考对比路径（microbench 的 apples-to-apples）在 AMD
   上要么换 RCCL，要么直接关掉（`UCCL_GIN_WITH_NCCL_GIN=0`）；不影响 UCCL 主路径。

## 4. 最小可行路径（如果只想"先看到 AMD 上活着"）
Phase 0 定环境 → Phase 1 方式 A（torch setup.py，自动 hipify）→ Phase 2 shim 化
→ 只跑 put+quiet。约 2-3 天。全源语（Phase 4）是周级、独立后端，建议排在 NCCL-EP
集成、性能调优之后，且仅当确有 AMD 多机需求时立项。

## 5. 纪律（沿用 AGENTS.md）
- 不动 `ep/`、不动 `nccl/`；AMD 改动集中在 `experimental/uccl_gin/`。
- 未实现/不支持分支一律 loud（trap/`#error`/abort），不留 silent fallback。
- 每个 Phase 的服务器结果、根因、路线调整写 worklog；未上机的改动标注"仅本地"。
- 偏离原 EFA transport 的设计（Phase 4 native atomic）必须在文档写明理由。
