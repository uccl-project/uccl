# NCCL-EP × UCCL-GIN 集成计划

> 状态：设计基线。Phase 1-2 adapter 准备代码已在 `b181a613` 落地，
> `3fb60814` 记录了待服务器验证项；本文仍作为集成 gate 和后续阶段定义。
> 对象：仓内 vendored `thirdparty/nccl-ep/`，来源为 NVIDIA NCCL 官方
> `contrib/nccl_ep` (NCCL 2.30.4)。外层 git-ignored 参考 clone 不直接修改或提交。
> 目标：把 nccl_ep 里走 GIN 的 RDMA 操作（`net.put` / `net.signal` / `net.waitSignal` /
> `net.readSignal` / `net.flush`）替换为 standalone UCCL-GIN，使
> `ncclEpDispatch` / `ncclEpCombine`（先 HT 模式）在 AWS EFA 上跑通并快于
> aws-ofi-nccl proxy GIN。LSA/NVLink 路径、NCCL communicator、layout 全部不动。

## 0. 已确认的事实（决定计划走向）

逐文件审过 `nccl/contrib/nccl_ep`（device/ ~7.6k 行，host nccl_ep.cc ~2.5k 行）：

1. **HT 模式的 RDMA 面是 rail 结构的**。`hybrid_ep.cuh:969`
   `remote_node_id = remote_idx < node_rank ? remote_idx : remote_idx + 1`、
   `:2245` `remote_global_rank = node_id`——dst 直接是节点索引，传给
   `net.put(world, remote_node_id, ...)`。即 nccl_ep 为 RDMA 建立了 per-rail
   communicator（同 local rank 跨节点，world = num_nodes）。这与 UCCL-GIN 当前的
   rail-only 拓扑（`validate_rail_dst` / `skip_normal_mode_peer`）**天然匹配**，
   HT 不需要动 transport 拓扑。⚠️ 开工前需在 nccl_ep.cc 里确认 comm split 的具体
   方式（Phase 0 任务）。
2. **LL 模式是全网状**（`low_latency.cu:240` `net.put(world, dstRank, ...)`，
   dstRank 任意）。UCCL transport 目前只给 rail peer 建 QP——LL 需要 full-mesh
   QP 拓扑，是独立的 transport 工作量 → 后置到 Phase 5，且默认**不承诺**。
3. **没有 `get`**。整个 nccl_ep device 代码无 `net.get`，UCCL-GIN 不需要 GET 原语。
4. **GIN 调用面很窄**（这是好消息）：
   - `ncclGin net(dcomms[comm_idx], ctx_idx, NCCL_GIN_RESOURCE_SHARING_CTA)`，
     `(comm_idx, ctx_idx)` 由 `get_comm_ctx(global_channel, ...)` 从 channel 推出
     （`hybrid_ep.cuh:950-955`）。
   - `net.put(world, dst, win, roff, win, soff, bytes, ncclGin_None, ...,
     ncclGinOptFlagsAggregateRequests)`——HT 数据 put 全部**不带 per-put signal**
     （`hybrid_ep.cuh:993/1001/1010/2296/2375`）。
   - `net.signal(world, dst, ncclGin_SignalAdd{tail_signal_id, 1}, ...)`——chunk
     批量 put 之后的独立 tail signal（`:1027/2335/2406`）。**依赖 NCCL "同
     (comm,ctx) 上 signal 在先前 puts 之后到达" 的保证**——这正是 UCCL-GIN
     sender-side dependency（red_add 等 payload CQE）重建的语义，同 lane 即可。
   - `net.waitSignal(ncclCoopThread(), id, expected)`（`:1122`）和
     `net.readSignal(id)`（`low_latency.cu:284/950`，HT 也有）。
   - `net.flush(ncclCoopWarp(), cuda::memory_order_acquire)`——只有这一处 flush，
     **warp-coop**（`:1037`）。
   - LL 专属：0 字节 put + `SignalAdd{id, numTokensSent+1}`（`low_latency.cu:240-248`），
     delta 可达数千。
5. **signal 数量**：`reqs.ginSignalCount = num_total_signals`
   （HT，nccl_ep.cc:844；dispatch_signals + combine_signals）；LL 是
   `2 * num_total_signals`（:1079）。UCCL 的 atomic slot 偏移是 13-bit/8B 对齐
   → **最多 1023 个可用 slot（slot 0 保留）**。HT 的 signal 是 per
   (channel × node) 量级，预计 ≪1023；LL 是 per (local expert × rank) 量级，
   可能超——又一个 LL 后置的理由。
6. kernel 是 **warp-specialized 多 warp 并发发 put** 的——standalone 至今所有
   测试都是单线程发命令。多生产者 D2H commit 必须先有独立 stress gate。

## 1. 语义映射表（adapter 的合同）

| nccl_ep 用法 | UCCL-GIN 映射 | 约束/备注 |
|---|---|---|
| `ncclGin net(dcomms[c], ctx, CTA)` | `UCCLGin gin(res)`，`lane_hint = global_channel` | (comm,ctx) 二维折叠为 lane；同一 channel 的 put+signal 必须同 lane（ordering 契约） |
| `net.put(world, node, win, roff, win, soff, bytes, None, ..., Aggregate)` | `gin.put<Rail>(recv_ptr, send_ptr, bytes, dst_rank, lane)` | dst_rank = node_idx × local_world + my_local_rank；Aggregate flag 对应 proxy 侧 coalescing（Phase 4 调优项，先忽略） |
| `net.signal(world, node, SignalAdd{id, 1})` | `gin.red_add_rel<Rail>(slot_ptr(id), 1, dst, 同 lane)` | 顺序性 = sender-side dependency；delta < 16383（kMaxSendAtomicValue 本身被保留作 kLargeAtomicValue 哨兵，不可用） |
| 0-byte put + `SignalAdd{id, delta}`（LL） | delta ≤ 255 且有 payload 时可用 `put_tail_add`；纯 signal 或大 delta 用 `red_add_rel` | LL 后置 |
| `net.waitSignal(coop, id, expected)` | 新增 device helper：`ld_acquire_sys` 自旋读 `atomic_tail_base + slot_off(id)` 直到 ≥ expected | atomic buffer 是 host-mapped、GPU 可读（外层 ep 的 hybrid 集成已用同一模式） |
| `net.readSignal(id)` | `ld_acquire_sys(slot_ptr(id))` | |
| `net.flush(ncclCoopWarp, acquire)` | 新增 warp-coop flush：`__activemask` 选举 lane0 → 对该 warp 用过的 lane 逐个 quiet → `__syncwarp` | 当前 `flush(coop_t)` 是 static_assert loud gap，本计划把 warp 特化补上 |
| signal 的 reset/单调性 | UCCL slot 是 int64 host 内存，由我们控制 reset 时机 | Phase 0 摸清 nccl_ep 的 signal 生命周期（每 iter reset 还是单调累加） |
| LSA（`ncclGetLsaPointer`、NVLink ld/st）、ncclComm、barrier | **完全不动**，仍走真 NCCL | UCCL 只接管网络面 |

## 2. 前置工作（standalone 侧，不碰 nccl_ep）

这些都是 standalone 自己的缺口，做完每一项都有独立 microbench gate，与 nccl_ep 解耦：

- **P1a 外部 window 注册**：`Context` 增加"注册调用方已分配的 buffer"构造路径
  （`UcclProxy` 本来就收 `gpu_buffer_addr` + `owns_gpu_buffer=false`，改动小）。
  nccl_ep 的 buffer 由 `ncclEpCreateGroup` 的 alloc_fn（ncclMemAlloc）分配并注册
  为 NCCL window，UCCL 必须注册**同一块**内存。
  验收：microbench 加 `--external-window` 模式，全 gate 复跑 PASS。
- **P1b signal slot 抽象**：host 侧 slot 分配表（id → 8B offset，≤1023 个，
  slot 0 保留），device 侧 `wait_signal(id, expected)` / `read_signal(id)` helper。
  验收：新增 microbench gate——多 lane put + signal + waitSignal 闭环。
- **P1c warp-coop flush**：实现 `flush(WarpCoop)` 特化，去掉对应 static_assert
  分支（保留 CTA 级的 loud gap）。
  验收：32 线程 warp 全员调 flush 的 microbench kernel。
- **P1d 多生产者 stress gate**：多 warp/多 CTA 并发对同一组 lane 发 put/red_add
  的 correctness kernel（这是给 warp-specialized kernel 铺路，也是上轮 review
  遗留项）。验收：2 节点 EP16，并发度 ≥8 warps，数据+counter 全对。
- **P1e 约束固化**：文档和测试固化“同一 counter slot 的 ordered add
  必须固定单 lane”，并验证 per-counter outstanding sequence 不超过
  `kReorderingBufferSize`。单独的 queue inflight cap `static_assert` 不能证明这个条件。

## 3. Seam 设计（nccl_ep 侧）

沿用 DeepEP 的 vendored minimal-patch 策略：

- **vendor**：把 `nccl/contrib/nccl_ep` 拷为仓库内受版本控制的副本（建议
  `thirdparty/nccl-ep-<上游sha>/`，附 VENDORED.md 记上游 commit 与改动清单）。
  绝不直接改 git-ignored 的 `nccl/`。
- **adapter 而不是散点替换**：新建 `uccl_gin_net.cuh`，定义
  `struct UcclGinNet`，方法面 = 第 1 节表格里 nccl_ep 实际用到的子集
  （构造、put、signal、waitSignal、readSignal、flush）。kernel 里的
  `ncclGin net(...)` 实例化点（HT 约 4 处、LL 约 6 处）通过
  `#if NCCL_EP_USE_UCCL_GIN` 选型——这是 vendored 副本里唯一的成片改动。
  禁止在 kernel 体内写 if/else 双路径（AGENTS：不留 silent fallback）。
- **host 插桩**：`ncclEpCreateGroup` 处（宏门控）：
  1. 在 NCCL comm/window 建好后，用 P1a 的外部注册 API 给同一 buffer 建
     `uccl_gin::Context`（per-rail：bootstrap 用 rail 子通信域的 rank/world）；
  2. 按 nccl_ep 的 signal 分配逻辑填 slot 表；
  3. 把 `UCCLGinResources` + slot 表加进 kernel launch 参数（dcomms 数组旁边）。
  UCCL 路径下 devComm reqs 的 `ginSignalCount` 等 GIN 项可保留（无害）或置零。
- **bootstrap**：standalone Context 现在用 MPI_Allgather 交换 PeerMeta；nccl_ep
  测试（ep_test）本来就在 mpirun 下跑，先沿用 MPI；如后续要纯 NCCL 启动，再加
  ncclComm out-of-band 交换（独立小项，不挡主线）。

## 4. 分阶段

### Phase 0：baseline + 摸底（不写集成代码）
1. 在 2×p5en 上用**原样 NCCL GIN**（aws-ofi-nccl master proxy GIN）build & run
   `contrib/nccl_ep` 的 ep_test/ep_bench，HT 模式 correctness + 性能基线
   （对照当年 DeepEP EP16 dispatch 5 GB/s 的量级，nccl_ep HT 预计同病）。
2. 读 nccl_ep.cc 确认：rail comm split 方式、signal 分配公式与生命周期
   （reset vs 单调）、HT 各 put 的典型 size 分布（决定小消息优化优先级）、
   buffer/window 布局。产出一页"调用面与资源清单"附在本计划后。
3. 验收：基线数字 + 清单进 worklog；若发现 HT 还有本计划未覆盖的 GIN 用法
   （如 counter、barrier、flushAsync），先回来改第 1 节映射表。

### Phase 1：standalone 前置项（P1a–P1e）
全部带独立 microbench gate，与 nccl_ep 无关，详见第 2 节。

### Phase 2：HT dispatch 接通
1. vendor + adapter + host 插桩（第 3 节）。
2. 只接 dispatch 路径（`hybrid_ep.cuh` 第一段：put×3 形态 + tail signal +
   waitSignal + warp flush）。combine 仍走 NCCL GIN？——**不行**，同一 build
   双 GIN 会引入 silent fallback 嫌疑；改为：combine 在 Phase 2 期间用
   `#error`/host 拒绝门控，测试只跑 dispatch-only 形态（nccl_ep 若无
   dispatch-only 测试入口，则 Phase 2 验收推迟到 Phase 3 一起）。
3. 验收：2×p5en EP16，ep_test HT dispatch correctness PASS。

### Phase 3：HT combine 接通
1. combine 的 put + signal + waitSignal（`hybrid_ep.cuh:2245-2420`）。
2. 验收：ep_test HT 全流程 correctness PASS（多 size、多 token 配置）。

### Phase 4：性能
1. 与 Phase 0 基线 apples-to-apples（同机、同配置、同 channel 数）。
2. 调优顺序：lane/channel 映射 → proxy 侧对 AggregateRequests 语义做
   coalescing（参考外层
   `ep/docs/uccl_gin_compact_staging.md`）→ inflight cap / proxy 线程数 sweep。
   只有在一个 logical chunk 已融成单个 payload WR，或 sender completion dependency
   仍能证明早先 payload 全部完成时，才允许将独立 signal 折叠进
   `put_tail_add`。EFA SRD 不保证多 WR 到达顺序，不能仅在“最后 post 的 WR”
   上 piggyback signal。
3. 验收：HT dispatch/combine 带宽 ≥ NCCL GIN 基线；目标量级写进 worklog 后再定
   （先拿到 Phase 0 数字）。**不达标不合并**（AGENTS：不留"只跑通 correctness、
   性能明显不对"的状态）。

### Phase 5（决策门，默认不做）：LL 模式
需要：full-mesh QP 拓扑（transport 级改动，需写设计理由）、signal slot 超 1023
的扩位方案（AtomicsImm 重新分 bit 或两级映射）、大 delta piggyback。只有 HT
达标且确有 LL 需求时再立项。

## 5. 风险与未决问题

1. **HT rail 假设**未在 host 代码层证实（Phase 0 第 2 条）。若 HT 某条路径用
   全局 rank 直发非 rail peer，工作量升级为 LL 同级。
2. **waitSignal 轮询 host 内存的延迟**：NCCL signal 在 GPU 可见内存，UCCL slot
   在 host pinned 内存，GPU 自旋读跨 PCIe（~1µs/读）。HT 的 waitSignal 是
   per-chunk 粒度，预计可接受；若成瓶颈，备选：proxy 把 signal 值镜像写回
   GPU 内存（一条 D2H 反向通路，transport 已有 completion 回传机制可复用）。
3. **性能命题本身**：standalone 4K put 仅 1.40 GB/s（rails=2，单线程驱动），
   而 nccl_ep HT 的 put 粒度未知。若 Phase 0 显示 HT put 普遍 ≤16KB，Phase 4
   的 coalescing 不是调优项而是前提。
4. **多 devComm/ctx 并行度**：NCCL 用多 comm 撑 QP 并行；UCCL 等价物是
   lane/proxy 数（当前 4 proxy × 8 ring）。channel 数 > 32 时映射要重新算。
5. 上游漂移：nccl_ep 是 contrib、迭代快；vendored 副本按 DeepEP 同款
   re-vendor 流程管理。
6. dst_rank uint8（≤256 rank）、window ≤16 GiB——nccl_ep HT 配置预计不触碰，
   Phase 0 清单里核一遍。

## 6. 纪律（沿用 AGENTS.md）

- 每个 Phase 的服务器结果、根因判断、路线调整必须进 worklog。
- vendored nccl_ep 改动保持最小、集中、可 re-apply；不动 `nccl/`、不动 `ep/` V1。
- 不写临时双路径/scaffold 进主代码；未实现分支一律 loud（trap/static_assert/abort）。
- 服务器验证统一用 `/home/ubuntu/.venvs/uccl-gin-cu13`，多机必须 GIN proxy 路径。
