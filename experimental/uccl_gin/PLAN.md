# UCCL-GIN Standalone Plan

## 目标

在 `experimental/uccl_gin/` 里做一套干净、独立的 UCCL-GIN backend，提供和
NCCL-GIN 关键 device primitive 形状相同的接口，用 UCCL 的 D2H FIFO + CPU proxy +
EFA verbs 承载 `Rail` 通信。

这一步**不接 DeepEP**，只做 GIN primitive 的替换层和验证层。DeepEP 之后只应该依赖这套
standalone UCCL-GIN，而不是继续把代码写进 `ep/`。

## 硬约束

- `ep/` 是参考实现和代码来源，不再是开发目标。
- 不允许从 `experimental/uccl_gin` 直接 include `ep/...`。需要的代码直接拷贝进
  `experimental/uccl_gin`，再删掉 EP-specific 逻辑。
- 不影响原来的 `uccl/ep` 路径：最终验收时，除非明确拆出共享公共头，否则
  `git diff -- ep` 应为空。
- 不引入 DeepEP 依赖，不 include `deep_ep/...`，不依赖 DeepEP buffer/layout/JIT。
- Python 测试必须验证 primitive 的正确性和 ordering，不能只 benchmark。
- 所有从 `ep` 拷来的逻辑都要保留原有 transport 思想：D2H ring、proxy drain、
  EFA post、CQ poll、completion/ack、ordered software atomic。不要重新发明一套
  proxy 协议，除非 plan 里说明理由。

## 最小 Primitive 范围

先覆盖 DeepEP 后续需要的最小 GIN 子集，但本阶段测试不涉及 DeepEP：

| primitive | 语义 | UCCL-GIN 后端 |
|---|---|---|
| `put<Rail>(dst, src, bytes, peer, options)` | symmetric window 到 symmetric window 的 RDMA WRITE | D2H `TransferCmd` WRITE + CPU proxy post EFA WRITE |
| `red_add_rel<Rail>(ptr, value, peer)` | 远端 counter release add | WRITE_WITH_IMM + receiver software atomic + seq reorder |
| `quiet/flush<Rail>()` | 保证本 lane 之前的命令完成到指定语义点 | D2H QUIET/BARRIER + proxy CQ drain/ack |

扩展但不作为最小 correctness gate：

- `put_tail_add<Rail>`：payload WRITE piggyback atomic add，主要用于减少独立 tail WR。
- `put_value<Rail>`：单 word WRITE，可由 `put` 的小 payload path 表达，后续再决定是否单独暴露。
- `get_sym_ptr<Rail>`：EFA 不能直接 device dereference 远端 VA，standalone 层只返回本地 offset/metadata，不伪装成可直访指针。

## 目录结构

```text
experimental/uccl_gin/
  PLAN.md
  README.md                         # 使用说明和 server test 命令
  include/uccl_gin/
    uccl_gin.cuh                    # public device API: UCCLGin handle
    uccl_gin_rail.cuh               # Rail primitive implementation
    resources.cuh                   # POD resource bundle, lane mapping, window geometry
    transfer_cmd.cuh                # lean device-visible TransferCmd ABI
    d2h_queue.cuh                   # copied/minimized D2H queue device view
    ring_buffer.cuh                 # copied/minimized GPU->CPU ring ABI
  src/
    proxy.cpp                       # standalone proxy, copied then trimmed from ep/src/proxy.cpp
    rdma.cpp                        # EFA verbs + software atomic receiver path
    context.cpp                     # host init, peer exchange, resource construction
    bindings.cpp                    # Python binding
  tests/
    cpp/
      microbench.cu                 # C++/MPI microbench, no Python dependency
    python/
      test_put.py
      test_red_add.py
      test_quiet_ordering.py
      test_multilane.py
  python/uccl_gin/
    __init__.py
```

如果后续需要 build system，优先在 `experimental/uccl_gin` 内放独立 `CMakeLists.txt`
或 `setup.py`，不要复用 `ep/setup.py`。

## 从 `ep` 拷贝的代码清单

只拷贝 transport substrate，不拷贝 EP 语义层：

保留/精简：

- `ep/include/ring_buffer.cuh`
- `ep/include/d2h_queue_device.cuh`
- `ep/include/d2h_queue_host.hpp`
- `ep/include/common.hpp` 中 queue/proxy/command 相关常量
- `ep/include/rdma.hpp` 中 `TransferCmd`、software atomic packing、EFA helper
- `ep/src/proxy.cpp` 中 D2H drain、post send、CQ poll、ack/quiet、pending atomic 逻辑
- `ep/src/rdma.cpp` 中 MR/QP/CQ 和 receiver completion apply
- `ep/src/uccl_proxy.cpp` 中 peer metadata exchange 可复用部分

删除/禁止带入：

- `SourceMeta`、expert layout、dispatch/combine/internode/intranode kernel
- `DeepEP` wrapper、`BufferLayout`、`TokenLayout`
- EP benchmark/test 逻辑
- V1 packed token staging、low-latency semantic metadata
- 所有 `ep` Python package 入口

拷贝后要做一次 include 清理：`experimental/uccl_gin` 内部 include 只能指向本目录、
UCCL 公共 include、CUDA/NCCL/libibverbs/libfabric/MPI 标准依赖。

## 分阶段计划

### Phase 0: Baseline 和边界冻结

1. 从当前分支记录 baseline：`git diff ecba87ae...HEAD -- ep` 只作为参考，不继续在
   `ep` 上叠代码。
2. 在 `experimental/uccl_gin` 建独立 skeleton。
3. 写 `README.md`，说明目标、非目标、构建方式、server 环境变量。
4. 验收：`git diff -- ep` 为空或只包含进入本阶段前已有的用户改动。

### Phase 1: 拷贝并瘦身 transport substrate

1. 拷贝 D2H ring、TransferCmd、proxy、rdma receiver 相关最小文件。
2. 改 namespace，去掉 `ep` 专属类型和宏。
3. 保持 16B `TransferCmd` ABI，除非必须扩展；扩展需要写明兼容理由。
4. 保留 payload-before-tail 的正确性机制：WRITE 完成依赖、ordered software atomic、
   receiver seq reorder。
5. 验收：standalone C++ 能编译一个空 context + proxy 初始化，不启动 DeepEP。

### Phase 2: Device API 接口

1. 实现 `uccl_gin::UCCLGinResources`：
   - D2H queue device views
   - local/remote registered window base
   - atomic buffer base
   - rank/lane/topology 信息
2. 实现 `uccl_gin::UCCLGin`：
   - `put<Rail>`
   - `red_add_rel<Rail>`
   - `quiet/flush<Rail>`
   - loud trap 未实现分支，避免 silent fallback。
3. `Lsa`/NVLink 不在本阶段实现；standalone 测试只测 `Rail`。
4. 验收：一个 CUDA kernel 可以 include `uccl_gin.cuh` 并发 D2H command。

### Phase 3: Host context 和 Python binding

1. 实现 host `Context`：
   - 初始化 EFA devices / QP / CQ / MR
   - 分配并注册 symmetric GPU window 和 atomic buffer
   - 创建 D2H queues 和 proxy threads
   - 通过 MPI 或 torch.distributed 交换 peer metadata
2. Python binding 暴露：
   - `Context(rank, world_size, local_rank, ...)`
   - `alloc_window(bytes)`
   - `resources()` 返回可传给 kernel launcher 的 packed resource
   - `barrier()`, `close()`
3. Python binding 只负责测试和后续集成便利，不承载性能热路径。
4. 验收：Python 能创建/销毁 context，多 rank 下无泄漏、无残留 proxy thread。

当前状态：

- 已实现最小 C++ `uccl_gin::Context`，并由 `_uccl_gin` CPython 扩展暴露给 Python。
- `python -m uccl_gin.context_smoke` 已在 2-node x 8-rank GIN-only 形状下验证
  create/resources/close/MPI finalize。
- Python binding 当前只覆盖 host context 生命周期；device primitive correctness 仍由
  standalone C++ microbench kernel 覆盖。Phase 4 需要继续把 put/add/quiet 的 Python
  tests 拆成更细的 pytest gate。

### Phase 4: Primitive 正确性测试

Python tests 必须覆盖：

1. `put` correctness：
   - rank-tagged pattern
   - 1KB 到 8MB size sweep
   - 远端 buffer 逐字节/逐 word 校验
2. `red_add_rel` correctness：
   - 多 lane、多 rank 对同一 counter add
   - 验证最终 counter 精确等于期望
   - 验证 seq reorder 不丢、不重、不倒退
3. `quiet/flush` ordering：
   - payload 发出后调用 `quiet`
   - `quiet` 返回后立即复用/覆盖 source buffer
   - 用独立 completion signal 等待远端最终完成，再验证远端收到覆盖前的数据
   - 不把 `quiet` 错测成 remote visibility；NCCL-GIN 只承诺 source 可安全复用
4. multi-lane：
   - 1/2/4/8/16 lane sweep
   - 校验 lane mapping 不把所有流压到单 NIC
5. teardown：
   - 重复创建销毁 100 次，确认 proxy 退出、CQ/MR/QP 释放。

所有测试先在单机 loopback/single node 走最小路径，再跑双机 EFA。

### Phase 5: Microbench

1. C++ microbench 保留，作为不依赖 Python 的低层验证。
2. Python benchmark 用同一 correctness kernel 包住计时，避免只测假数据。
3. 指标：
   - per-rank GB/s
   - aggregate GB/s
   - WR/s
   - proxy CPU utilization
   - CQE count
   - per-lane bytes
4. 与 NCCL-GIN 做 apples-to-apples：
   - 同 rank/world
   - 同 rails/lanes
   - 同 payload size
   - 同 correctness gate

### Phase 6: 清理和回归

1. 删除 `ep/tests/uccl_gin_microbench` 中的旧 standalone 原型，或保留为历史但不作为主入口。
2. 确认 `ep/` 原测试仍能跑；若 `ep/` 完全无 diff，则只需要跑原 baseline smoke。
3. 文档写清楚从 `ep` 拷贝来的文件及后续同步策略。
4. commit 前检查：

```bash
git diff -- ep
git status --short
rg 'ep/include|ep/src|deep_ep' experimental/uccl_gin
```

`rg` 结果必须为空或在文档注释中明确说明。

## Server 验收顺序

1. 本地静态检查：

```bash
rg 'ep/include|ep/src|deep_ep' experimental/uccl_gin
git diff -- ep
```

2. `p5en_0` 单机 build + import：

```bash
cd /home/ubuntu/efs/yzhou/playground/daniel/uccl-danyang/uccl-danyang
git checkout <branch>
make -C experimental/uccl_gin -j
python -m pytest experimental/uccl_gin/tests/python -q -k single
```

3. 双机 correctness：

```bash
MASTER_ADDR=<p5en_0_ip> MASTER_PORT=<port> WORLD_SIZE=2 RANK=0 ...
MASTER_ADDR=<p5en_0_ip> MASTER_PORT=<port> WORLD_SIZE=2 RANK=1 ...
python -m pytest experimental/uccl_gin/tests/python -q -k efa
```

4. 双机 benchmark：

```bash
/opt/amazon/openmpi/bin/mpirun ... experimental/uccl_gin/tests/cpp/uccl_gin_microbench
```

5. 每次服务器运行后写 worklog：命令、env、日志路径、结果、是否有残留进程。

## 当前决策

- 现在先不把 DeepEP V2 接进来。
- 现在先不在 `ep/` 上继续开发。
- 现在先把 GIN primitive 独立化，并用 Python tests 把 correctness/order 站稳。
- 后续 DeepEP 集成时，只允许依赖 `experimental/uccl_gin` 的 public API；如果需要额外
  primitive，先在 standalone 层加测试，再接 DeepEP。

## 当前验证进度（2026-06-11）

- standalone build 保持 `UCCL_GIN_WITH_NCCL_GIN=1`，NCCL-GIN reference path 未被绕开。
- 修复 normal-mode topology 对 `MAX_NUM_GPUS=8` 的硬编码后，`2-node x 1-rank`
  GIN-only 不再在 proxy/QP 初始化或 RDMA destination validation 处失败。
- `TransferCmd.bytes` 是 24-bit；public `put` 现在自动分片超过 wire limit 的 payload。
  大 `put_tail_add` 使用 plain-WRITE dependency tracking + ordered ATOMIC，不能只把
  tail piggyback 到最后一个 chunk，因为 EFA SRD 不保证跨 WR 到达顺序。
- 2-rank UCCL-only、2-rank NCCL-GIN reference + UCCL-GIN、2-node x 8-rank GIN-only
  均通过 `1 KiB / 4 MiB(or 1 MiB) / 16 MiB` correctness。
- Python primitive gates 已拆开验证 `put+red_add`、`put_tail_add`、`quiet source-reuse`、
  `red_add counter` 和综合 size sweep；服务器结果 `6 passed`。
- 新增 context create/destroy stress gate；2-rank 默认完整验收 100 次已通过。
