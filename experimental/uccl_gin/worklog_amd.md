# UCCL-GIN AMD 支持 worklog

记录在 AMD (MI325X / ROCm) 机器上让 `experimental/uccl_gin` 构建并跑通的工作。
配套计划见 `AMD_SUPPORT_PLAN.md`。

---

## 2026-06-12 Phase 0：目标机器环境确认（ssh amd0 / amd1）

### 主机
- `amd0` = `chi-mi325x-pod2-098`，mgmt IP `45.76.230.97`
- `amd1` = `chi-mi325x-pod2-099`，公网 `144.202.52.73`
- 互联：两台在 `enp49s0f1np1`（= mlx5_1 的 netdev）上有 `10.162.224.0/20` 内网
  - amd0: `10.162.224.145`，amd1: `10.162.224.146`（RoCE 可用网卡 + 有 IP，可做
    bootstrap + 一条 RDMA rail）
  - 注意：amd1 主机名不可直接 ping，但 `ssh amd1`（~/.ssh/config → 144.202.52.73）通。

### GPU / ROCm
- 每台 **8 × AMD Instinct MI325X**，GFX `gfx942`（CDNA3，sramecc+ xnack-）
- ROCm **6.4.2-120**，HIP **6.4.43484**，`hipcc` 在 `/usr/bin/hipcc`
- OS Ubuntu 24.04.4，kernel 6.8.0-117
- 探测时 8 卡全部 0% 占用（仅 root 的 Ray idle 集群 dashboard/gcs 在跑，无 GPU 负载）

### 网络栈（关键：RoCE，非 EFA）
- RDMA 网卡：
  - 2 × `mlx5_0/1`（Mellanox CX，link_layer **Ethernet** = RoCE）
  - 8 × `bnxt_re0..7`（Broadcom，link_layer **Ethernet** = RoCE，bnxt_re6 DOWN）
  - 全部 `transport: InfiniBand (verbs)`，`PORT_ACTIVE`
- **没有 EFA**（预期：EFA 是 AWS 专属）。
- → 对应 `ep/` 的**非 EFA 路径 = native RDMA verbs + RC QP + native atomic
  (FETCH_AND_ADD)**，不是 EFA 的 software-seq atomic。

### 软件
- 无系统 torch；有 miniconda3 + 多个用户的 uccl venv（NFS）。
- RCCL：`/opt/rocm/lib/librccl.so`。
- 有 UCCL 自己的 RCCL net plugin：
  `/home/yangzhou/.../uccl/lib/librccl-net-uccl.so`。
- MPI：OpenMPI 4.1.6（`/usr/bin/mpirun`）。
- 共享 NFS：`/home/yangzhou/nfs`（多人 uccl checkout，git remote 是公开
  `github.com/uccl-project/uccl.git`，不是本仓库 `DanielDanyang/uccl-danyang`）。
- 登录身份 `yangzhou`（**共享账号，多人使用——遵守 AGENTS：用隔离 venv/scratch、
  不动他人进程、重负载前查占用**）。

### Phase 0 结论
1. 机器是 **RoCE + native atomic**，落在 layer-4 的"墙"那一侧：
   - `put` / `quiet` / `flush`：走非 EFA RDMA WRITE 路径，**预期可直接工作**。
   - `red_add_rel` / `put_tail_add` / `put_value`：依赖 EFA-shaped software-seq
     atomic，非 EFA receiver 会 `assert(false && "Reorderable atomic...")`。
     需要 **native-atomic Rail flavor**（计划 Phase 4，独立后端，周级）。
2. 构建工具链齐全（hipcc / ROCm 6.4 / RCCL），可走 Makefile+hipcc（方式 B）或
   torch 自动 hipify（方式 A，但需装 rocm-torch）。
3. 代码上服务器：本仓库是 `DanielDanyang/uccl-danyang`（已 push 分支
   `uccl-gin-rail-primitives`），与服务器现有的公开 uccl checkout 不同源，需单独
   clone（私有 fork 需 auth）或从本地 rsync。

### 用户决策（2026-06-12）
- A. 范围：**先只做 put/quiet**，RoCE 上跑通后再决定要不要投 native-atomic 后端。
  red_add_rel / put_tail_add / put_value 在 AMD 构建下 loud reject。
- B. 构建：**torch setup.py 自动 hipify**（照搬 ep）。需要 ROCm torch venv。
- C. 代码：服务器 `~/nfs/danyang` 下 `git clone
  github.com/DanielDanyang/uccl-danyang` + checkout `uccl-gin-rail-primitives`
  （公开 repo，无需 git 凭证）。

## 2026-06-12 Phase 1：build 系统 + device shim（进行中）

### 已完成：device 层可移植性 shim（commit）
新增 `uccl_gin/platform.cuh`：
- `UCCL_GIN_TRAP()` 宏：NV `__trap()` / AMD `__builtin_trap()`。
- `UCCL_GIN_WITH_NCCL_GIN` 默认值 + `UCCL_GIN_HAVE_NCCL_DEVICE` 判定。
- 刻意**不** include `<nccl_device.h>`（resources.cuh 会被 host 的 context.cpp
  间接 include，nccl device 头必须留在 uccl_gin.cuh 一处）。

改动：
- `uccl_gin/{resources,uccl_gin_rail,uccl_gin}.cuh`：16 处 `__trap()` →
  `UCCL_GIN_TRAP()`，include platform.cuh。
- `uccl_gin.cuh`：team tag 在 `UCCL_GIN_HAVE_NCCL_DEVICE` 下 include
  `<nccl_device.h>`，否则定义 stand-in `ncclTeamTagRail/Lsa`（同名空 struct），
  同一 `gin.put<ncclTeamTagRail>()` call site 在 AMD 编译不变。
- `context.cpp`：`<cuda_runtime.h>` → `util/gpu_rt.h` 的 `gpu*` shim；
  `cudaMemset` 因 shim 缺失用 HIP/CUDA guard 内联。
- NV 行为等价（`UCCL_GIN_TRAP`==`__trap`，`gpu*`==`cuda*`，tag 仍来自 nccl），
  本机无 CUDA 未编译验证，逻辑等价。

### 关键发现（修正计划范围）
1. **microbench 是 nvcc 编的 MPI 可执行文件，不是 Python 扩展**——torch 的
   `BuildExtension`/`setup.py` 只能建 `_uccl_gin.so`，建不了 microbench。AMD 上
   microbench 必须由 Makefile+hipcc 编。→ 故 build 方式实际只能是 **Makefile+hipcc**；
   shim 做完后代码已无裸 cuda*，torch 自动 hipify 无收益，且 venv 是 rocm7.0 与
   系统 rocm6.4 错配。**建议放弃 torch setup.py 方案，改 Makefile+hipcc。**
2. **AMD 非 EFA 上只有 put + quiet 能工作**：
   - `red_add_rel`、`put_tail_add`：reorderable software atomic，非 EFA receiver
     `assert(false)`。
   - `put_value`：WRITE_VALUE 在 fast-mode / 非 EFA 分支被我们上轮加的 loud abort
     挡掉（虽然语义上它只是普通 4B WRITE，本可工作）。
   - 现有 microbench 的**所有** UCCL correctness gate（put-add / tail-add /
     quiet / red-add / put-value）都用 red_add 或 put_tail_add 做完成信号 →
     **移植整个 microbench 在 AMD 上跑不出有效 gate**。
   - → Phase 2 需要一个**新的 put/quiet-only smoke**：sender put(data) → quiet →
     MPI barrier → receiver 读 recv 校验（RC WRITE 的 CQE drain + barrier 往返
     保证数据已落远端）。不依赖任何 atomic 路径。

### 构建尝试（PLATFORM=rocm，hipcc）逐个暴露的问题
- 非 EFA、非 SOFTWARE_ORDERING → WRITE 走 rdma.cpp 第三个 `#else` 分支（标准 RC
  `ibv_post_send`），正是 RoCE 需要的，和 ep 的非 EFA 路径同源。
- Makefile `PLATFORM=rocm` 分支需显式 `-D__HIP_PLATFORM_AMD__`（否则 gpu_rt.h
  走 CUDA 分支找 cuda.h）。已加。
- 2 个 transport 头有裸 `#include <cuda.h>`（`fifo_util.hpp`、`ep_util.hpp`），
  ep 靠 hipify 翻译，standalone 手工加 `#if __HIP_PLATFORM_AMD__ → hip_runtime.h`
  guard。已改（限 experimental/uccl_gin/transport 拷贝）。

### ⛔ 当前硬卡点：gpu_rt.h 依赖 ROCm 6.4.2 缺失的 DMA-BUF 符号
- `include/util/gpu_rt.h` 的 HIP 分支无条件引用 `hipMemRangeHandleType` /
  `hipMemRangeHandleTypeDmaBufFd`（GPUDirect DMA-BUF handle export 用）。
- **系统 ROCm 6.4.2 头里这两个符号不存在**（`grep -rl` 在
  `/opt/rocm-6.4.2/include` 0 命中），编译 `error: unknown type name
  'hipMemRangeHandleType'`。
- 本机**只装了 ROCm 6.4.2**（无 7.0）。venv 里 torch 是 rocm7.0 wheel，但
  `ROCM_HOME` 仍解析到 `/opt/rocm-6.4.2`（编译用 6.4.2 头）。
- `ep.abi3.so` 的 mtime 是 2026-05-27、`/home/.../uccl/uccl/` 下，**是别处 CI
  编好放 NFS 的，不是在这台 6.4.2 上编的**。所以"ep 在本机能编"不成立——
  ep 的 gpu_rt.h 同样会在 6.4.2 上撞这个符号。
- 这个符号只服务 DMA-BUF GPUDirect 路径；uccl_gin RoCE 不用 USE_DMABUF，逻辑上
  可以 guard 掉。但 gpu_rt.h 是 **ep/uccl_gin 共享的外层 include**，改它要顾及 ep。

### 已解决的逐个 build 卡点（均已 commit + push）
1. `__HIP_PLATFORM_AMD__` 宏 → Makefile rocm 分支显式加。 ✓
2. transport 2 个裸 cuda include（fifo_util/ep_util）→ HIP guard。 ✓
3. gpu_rt.h 的 DMA-BUF typedef（hipMemRangeHandleType，6.4.2 缺）→ `#ifdef USE_DMABUF`
   gate（唯一使用者 rdma.cpp 也全在 USE_DMABUF 下）。 ✓

### ⛔ 新的、更根本的卡点：mscclpp FIFO 层被无条件织入，必须 hipify
- 编 `context.o` 现在卡在 `transport/fifo_util.hpp`：满是 `cudaError_t` /
  `CUresult` / `cuGetErrorString` / `cudaDeviceGetPCIBusId`（MSCCLPP 的
  `CudaError` 类 + `MSCCLPP_CUDATHROW`）。这正是 ep 靠 **hipify** 翻译的部分。
- 试图"gate 掉未使用的 FIFO 代码"走不通：standalone 默认用非-FIFO ring 路径，
  FIFO 的**使用**确实都在 `#ifdef USE_MSCCLPP_FIFO_BACKEND` 下，但
  **`transport/uccl_proxy.hpp` 无条件声明 mscclpp::Fifo 成员**（95
  `std::vector<std::unique_ptr<mscclpp::Fifo>> fifos;`、113 `set_fifo(mscclpp::Fifo*)`、
  130 `mscclpp::Fifo* fifo_;`），不在任何 guard 下。要 gate 干净需改 uccl_proxy
  的成员/方法 + uccl_proxy.cpp 定义 + 多个 include + 从 build 去掉 fifo.cpp，
  whack-a-mole，风险高。
- **结论**：mscclpp 那层 cuda 代码必须经 hipify 才能在 HIP 上编——这就是
  "用 ep 用的东西" = hipify。

### 推荐的下一步（待确认/继续）
按"用 ep 用的方式"，给 ROCm 构建引入 hipify：
- 方案 H1（Makefile + hipify-perl，无 torch）：服务器已有 `/usr/bin/hipify-perl`。
  Makefile rocm 分支加一步：把 transport/*.{cpp,cc,hpp} + *.cu 拷到
  `build/hipified/`，跑 `hipify-perl --inplace`，再用 hipcc 编 hipified 副本。
  保持 standalone 零 torch；能同时建 microbench/smoke 可执行文件。
- 方案 H2（torch BuildExtension，ep 原样）：照搬 ep setup.py 自动 hipify，但
  (a) 只能建 `_uccl_gin.so`，建不了 microbench/smoke 可执行文件（仍要 Makefile）；
  (b) venv torch 是 rocm7.0 而系统 6.4.2，ROCM_HOME=6.4.2。
- **我倾向 H1**：最接近 ep 的 hipify 语义，又能建可执行 smoke，零 torch。
- 已有的手工 shim（platform.cuh trap、gpu_rt 用法、2 个 include guard、gpu_rt
  dma-buf gate）与 hipify 不冲突：gpu* 不是 cuda 符号，hipify 不动；hipify 只翻
  mscclpp 那些裸 cuda*。

### venv
按你说的会建独立 venv 跑后续 python 测试。注意 hipcc 的 C++ smoke 本身不需要
torch；venv 主要给 `_uccl_gin.so` import 和 python gate。
