# NCCL-EP × UCCL-GIN 工作日志

## 2026-06-12: Phase 0 — NCCL-EP baseline

### Build

编译 `nccl/contrib/nccl_ep`（NCCL 2.30.4 + CUDA 13.0 + aws-ofi-nccl master）。

关键坑：
- `nccl_ep.cc` 包含 `nccl_device.h`，host 编译需 `-DNCCL_HOSTLIB_ONLY`
- `.cu` 文件需 `-DNCCL_CHECK_CUDACC=1`
- ep_bench 需要 `cupti.h`（在 `/usr/local/cuda-13.0/extras/CUPTI/include`）
- 全部文件（含 `.cc`）用 nvcc 编译（cuda/atomic 等 header 只在 nvcc 下存在）

Build script: `/home/ubuntu/efs/yzhou/playground/daniel/nccl_ep_build/build_nccl_ep.sh`

### HT correctness

```
mpirun -np 16 -npernode 8 ... build/ep_test
```

全部 16 rank `Success`。Dispatch + Combine correctness PASS。

### HT performance baseline

```
build/ep_bench -a ht -t 5000 -d 7168 -k 8 -e 256 -w 5 -i 20 -V
```

| | kernel time | rdma_send BW | 说明 |
|---|---|---|---|
| Dispatch | 10.06 ms | 7.10 GB/s | 71.41 MB RDMA outbound per rank |
| Combine | 1.97 ms | — | BW 数字有公式 bug（recv_bytes÷combine_time），不可用 |

Correctness: Dispatch PASSED, Combine PASSED (calc_diff=3.2e-08)。

**关键结论**:
- NCCL-EP dispatch rdma_send 7.10 GB/s — 和 DeepEP V2 直接 NCCL GIN 的 5-6 GB/s 同量级。per-token RDMA WRITE 的 EFA proxy 开销是共同瓶颈。
- NCCL-EP 每个 dispatch put 走 `ncclGinOptFlagsAggregateRequests`（SKIP_DB_RINGING），不减少 WR 数。
- NCCL-EP HT combine 走 `STREAMING_BATCH=8` chunked RDMA，所以 combine 带宽远高于 dispatch。
- ep_bench 的 combine BW 计算公式有 bug（拿 dispatch 的 recv_bytes 除 combine kernel time）。

### 有用信息（给后续 Phase 用）

- HT 模式 dispatch 的 `net.put` 不带 per-put signal（只用 AggregateRequests），combine 也类似。
- Signal 是独立操作：`net.signal(world, node, SignalAdd{id, 1})` — chunk 批量 put 之后的 tail 通知。
- WaitSignal 是 `coop.waitSignal`（自旋读 host-mapped signal），UCCL 可通过 `ld_acquire_sys(atomic_tail_base)` 直接等价。
- Signal 数量：dispatch_signals + combine_signals，per (channel × node)。HT 模式预计 ≤1023 slots。
- 每个 `ncclGin` 对应一个 (comm, ctx) 对。UCCL 需要映射到 lane。

---

## Phase 1 plan

Per NCCL_EP_PLAN.md §2，5 个 standalone 前置项（P1a-P1e）。每个带独立 microbench gate，不碰 nccl_ep。

### P1a: 外部 window 注册

目标：`Context` 支持注册调用方已分配的 GPU buffer（而非 cudaMalloc）。

当前 `Context` 内部 `cudaMalloc` window。NCCL-EP 的 buffer 由 `ncclMemAlloc` 分配并注册
为 NCCL window，UCCL 必须注册同一块内存。

`UcclProxy` 构造函数已支持 `gpu_buffer_addr` + `owns_gpu_buffer=false`。改动：
- `Context` 增加 `ContextConfig::external_window` 参数（ptr + bytes）
- 跳过 cudaMalloc/cudaFree，直接使用外部 window
- 验收：microbench 加 `--external-window` flag，全 gate PASS

### P1b: signal slot 抽象

目标：host 侧 slot 分配表 + device 侧 waitSignal/readSignal。

当前 UCCL-GIN 的 counter 是 host-mapped atomic buffer。NCCL-EP 的 signal 是 per (channel × node) 的
monotonic counter。适配：
- Host 侧：slot 分配表（signal_id → 8B offset in atomic buffer，slot 0 reserved）
- Device 侧：`wait_signal(signal_id, expected)` → `ld_acquire_sys(atomic_tail_base + offset)`
  自旋直到 ≥ expected
- Device 侧：`read_signal(signal_id)` → `ld_acquire_sys(atomic_tail_base + offset)`
- 验收：microbench gate — multi-lane put + signal + waitSignal 闭环

### P1c: warp-coop flush

当前 `flush(coop_t)` 是 static_assert。NCCL-EP 用 `net.flush(ncclCoopWarp(), acquire)`。

实现 warp-coop 特化：
- `__activemask` 选举 lane0 → lane0 对所有活跃 warp 的 lane 发 quiet → `__syncwarp`
- 验收：32-thread warp 全员调 flush 的 microbench kernel

### P1d: 多生产者 stress gate

当前 standalone 所有测试是单线程驱动。NCCL-EP 是 warp-specialized 多 warp 并发 put。

验收：2-node EP16，≥8 warps 并发对同一组 lane 发 put/red_add，data+counter 全对。

### P1e: 约束固化

- `static_assert(kUCCLGinMaxInflightNormal < kReorderingBufferSize)`
- 文档写清：同一 counter slot 的 ordered add 必须固定单 lane

---

## 2026-06-12 (续): Phase 1-2 代码完成，待 server 验证

### Adapter 实现

不改 UCCL-GIN 一行代码。在 vendored `thirdparty/nccl-ep/adapter/` 下：

- `uccl_gin_net.cuh`：device 侧 adapter — `UcclGinNet` struct 包装 `UCCLGin`，
  暴露和 NCCL-EP HT kernel 兼容的 `put/signal/waitSignal/readSignal/flush` 方法面。
  额外的 NCCL 参数（world, ncclWindow, remote_action 等）通过模板 varargs 接受并忽略。
  Signal 映射：signal_id → atomic_tail_base[id * sizeof(int64_t)]。

- `uccl_gin_context.hpp/cpp`：host 侧 adapter — `UcclGinContext` 用外部 GPU buffer
  创建 UCCL transport（UcclProxy + D2H ring + peer exchange via MPI）。

### hybrid_ep.cuh 改动

宏 `NCCL_EP_USE_UCCL_GIN` 门控。定义了两个辅助宏：
- `NCCL_EP_UCCL_PARAM`：在 kernel 函数签名末尾添加 `uccl_resources` 参数
- `NCCL_EP_NET_CREATE(comm, ctx, chan)` / `NCCL_EP_NET_CREATE_SIMPLE(comm, ctx)`：
  替换原来的 `ncclGin net(...)` + `ncclTeam world = ...` 两行

改动覆盖全部 4 个 ncclGin 创建点（dispatch N2N、dispatch waitSignal、combine N2N、combine waitSignal）。

NCCL-EP kernel 内的 `net.put(world, ...)` / `net.signal(world, ...)` / `net.waitSignal(...)` /
`net.flush(...)` 调用面完全不变——adapter 接受相同参数列表。

### build_uccl_ep.sh

编译脚本：
- `NCCL_EP_USE_UCCL_GIN=0`（默认）：编 NCCL-only，和原版完全一致
- `NCCL_EP_USE_UCCL_GIN=1`：加 UCCL-GIN transport objects + adapter context

### 待验证（server 不可达）

1. UCCL=0 编译 → ep_test 跑通（验证宏不破坏 NCCL 路径）
2. UCCL=1 编译 → ep_test HT dispatch 跑通（Phase 2）
3. ep_test HT combine 跑通（Phase 3）
4. ep_bench 性能对比

---

## Phase 2-3 plan（概要）

- Phase 2：vendor nccl_ep → adapter (uccl_gin_net.cuh) → HT dispatch 接通
- Phase 3：HT combine 接通
- 详细计划见 NCCL_EP_PLAN.md §3-4

---

## 2026-06-22: standalone 代码指南与 NCCL-EP 计划收口

- 新增 `CODE_GUIDE.md`，按 resources → Rail command → device API → host
  context → microbench 的数据流记录当前 standalone 实现。
- 修正文档中已过时的 `value_staging_off`：当前 `WRITE_VALUE` 使用
  per-proxy host bounce MR，不再切 GPU window staging slot。
- 明确 `quiet` 的 gate 是本 lane 早先 WRITE/WRITE_VALUE 的 sender CQE，
  保证 source buffer 可复用；receiver visibility 仍需上层协议/测试 settle。
- 修正 ordered-atomic 文档：queue inflight cap 不能单独证明 4-bit seq
  不 alias，还需同 counter 固定 lane 和 per-counter outstanding 边界。
- 更新 `NCCL_EP_PLAN.md` 状态为已有 Phase 1-2 adapter 准备代码，并删除
  “直接把最后一个 WR 的 signal piggyback”这个不满足 EFA SRD 乱序语义的
  调优前提。

本轮只整理文档，transport/device/host 代码零改动，未重跑服务器测试。
