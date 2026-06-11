# UCCL-GIN 架构与实现

UCCL-GIN 是一套与 NCCL GIN 接口对齐的 device-side 通信原语，用 UCCL 的 D2H ring
+ CPU proxy + EFA verbs 底座承载 DeepEP V2 的跨节点 (Rail) 流量，节点内 (Lsa)
委托 NCCL/NVLink 不动。

## 1. 和原来 UCCL EP 的关系

UCCL EP（`ep/` 下的 `internode.cu` 等）是 DeepEP **V1** 时代的产物：
V1 kernel 直接嵌入 transport——`nvshmemi_ibgda_put_nbi_warp`、`PackAtomicWithSeq`
全部 inline 在 kernel 里。换一个 kernel 就要重抄一遍。

UCCL-GIN 把 transport 从 kernel 里**抽出来**，包装成 `handle::UCCLGin`——一个和
DeepEP V2 原生 `handle::NCCLGin` 签名相同的 C++ 模板类。V2 kernel 不再感知 transport
差异：`gin.put<Rail>(...)`、`gin.red_add_rel<Rail>(...)` 通过模板分发，编译时选
UCCLGin 还是 NCCLGin。

**UCCL EP 没有、UCCL-GIN 新增的：**

| UCCL EP (V1) | UCCL-GIN (V2) |
|---|---|
| transport 嵌入 kernel，一个 kernel 一个 fork | `handle::UCCLGin` 独立 API，多 kernel 复用 |
| coordinator warp 攒 chunk + 发 RDMA | `put<Rail>` 内部分片，调用方不用管 chunk 大小 |
| tail counter 在 GPU window 内 | `atomic_tail_base` 独立 compact buffer，用 compact-index API |
| 无 inflight cap | `kUCCLGinMaxInflightNormal` 背压 |
| 无 sender-side dependency | `PendingAtomicBatch` + `retire_inflight_write` |
| 只有 dispatch 路径 | dispatch + combine 共用同一套 handle |
| 无独立 microbench | `tests/microbench.cu`，每个源语独立 correctness gate |

**从 UCCL EP 继承的（没动）：**

- 16B `TransferCmd` ABI + D2H ring (ring_buffer.cuh, d2h_queue_device.cuh)
- CPU proxy + EFA verbs post (transport/proxy.cpp, transport/rdma.cpp)
- `PackAtomicWithSeq` + receiver 端 SeqBuf reorder (transport/rdma.cpp)
- WRITE_WITH_IMM + empty WRITE_WITH_IMM 用于软件 atomic
- CQE poll + completion ack + QUIET/BARRIER

## 2. 代码结构

```
experimental/uccl_gin/
├── uccl_gin/                  # 公共 API（DeepEP 将来 include 的目标）
│   ├── uccl_gin.cuh           # handle::UCCLGin — 镜像 NCCLGin 的方法面
│   ├── uccl_gin_rail.cuh      # rail_put / rail_put_tail_add / rail_red_add
│   └── resources.cuh          # UCCLGinResources POD + queue mapping + profile counters
│
├── transport/                 # 内部底座（从 ep/ 拷贝，不依赖 ep/）
│   ├── proxy.cpp / proxy.hpp  # CPU proxy：D2H drain、post send、CQ poll、ack、dependency
│   ├── rdma.cpp / rdma.hpp    # EFA verbs：MR/QP/CQ、WRITE_WITH_IMM、receiver atomic apply
│   ├── uccl_proxy.cpp/.hpp    # UcclProxy 封装：生命周期、peer exchange
│   ├── ring_buffer.cuh        # TransferCmd (16B)、CmdType、ring 操作
│   ├── d2h_queue_device.cuh   # GPU 侧 D2HHandle（atomic_set_and_commit）
│   ├── d2h_queue_host.hpp     # Host 侧 D2H 管理
│   ├── common.hpp / common.cpp
│   ├── fifo*                  # MSCCLPP FIFO 后端
│   ├── adaptive_sleeper*      # 自适应休眠
│   └── exception.cuh / ep_util.hpp / bench_utils.hpp / barrier_local.hpp
│
├── context.hpp / context.cpp  # Host Context：transport 初始化、资源分配、peer exchange
├── bindings.cpp               # CPython 扩展：_uccl_gin 模块
│
├── tests/
│   ├── microbench.cu          # C++/MPI 独立 microbench（每个源语独立 correctness gate）
│   ├── test_context.py        # Python context smoke test
│   ├── test_microbench.py     # Python microbench wrapper
│   └── test_primitives.py     # Per-primitive Python correctness tests
│
├── python/uccl_gin/           # Python 辅助
│   ├── __init__.py
│   ├── microbench.py
│   ├── context_smoke.py
│   └── context_stress.py
│
├── Makefile
├── PLAN.md
└── ARCHITECTURE.md            # 本文档
```

## 3. 核心 API：handle::UCCLGin

`uccl_gin/uccl_gin.cuh` 定义了 `struct UCCLGin`。它和 DeepEP V2 的
`deep_ep::elastic::handle::NCCLGin` 有相同的方法签名，只是 Rail 分支走 D2H+proxy+EFA。

关键设计决策：
- 不是 NCCLGin 的子类——是独立 struct，通过**模板分发** (`if constexpr (team_t == Rail)`)
  路由到不同后端。Lsa/World 委托组合的 NCCLGin（未在 standalone 实现，`__trap()`）。
- 所有方法都是 `const`——gin 是无状态 view，状态在 `UCCLGinResources` 里。
- 每个方法接受 `lane_hint` 参数，由 `lane()` 映射到具体 D2H ring。

### 3.1 UCCLGinResources

```cpp
struct UCCLGinResources {
    d2hq::D2HHandle** d2h_queues;  // device array of D2H handle pointers
    uint32_t num_queues;

    uint64_t window_base;           // registered window 基地址（put offset 原点）
    uint64_t atomic_tail_base;      // host-mapped atomic buffer 基地址
    uint64_t value_staging;         // per-lane put_value staging (one int per lane)

    int num_scaleout_ranks;
    int num_scaleup_ranks;
    int scaleout_rank;
    int scaleup_rank;
    uint32_t num_lanes;
};
```

这个 struct 就是 DeepEP kernel 启动时传给 handle 的全部状态。设计上故意 lean——
没有 EP layout、没有 token count、没有 expert metadata——因为 standalone UCCL-GIN
不耦合 DeepEP。

### 3.2 已实现源语

#### put<Rail> — 纯 payload WRITE

```cpp
template <typename team_t>
void put(void* recv_sym_ptr, void* send_sym_ptr, int num_bytes,
         int dst_rank, int lane_hint = 0) const;
```

内部逻辑：
1. 两个对称指针 `send_sym_ptr` / `recv_sym_ptr` 通过 `window_off()` 转换成 4-byte 移位 offset
2. 大 payload 自动分片：`kTransferCmdMaxBytes` (~256KB) 对齐切割，循环发多条 rail_put
3. 每条 `rail_put` 写一条 `TransferCmd{WRITE, atomic_val=0}` 到 D2H ring
4. Proxy 读到 `atomic_val == 0` → 发普通 `IBV_WR_RDMA_WRITE`
5. 这条 WRITE 进入 sender-side dependency tracking

对应 NCCL GIN: `gin.put<Rail>(recv, send, bytes, dst, AggregateRequests)`

#### put_tail_add<Rail> — payload + piggyback count (WRITE_WITH_IMM)

```cpp
template <typename team_t>
void put_tail_add(void* recv_sym_ptr, void* send_sym_ptr, int num_bytes,
                  int dst_rank, int count_delta, uint32_t atomic_byte_off,
                  int lane_hint = 0) const;
```

这是 UCCL-GIN 独有的 primitive——NCCL GIN 没有等价 API。NCCL 用独立的 `put` + `red_add_rel`
+ NIC FORCE_SO 保证顺序，我们把它合成一条 WR。

内部逻辑：
1. payload 和 tail counter 的信息**编入同一条 TransferCmd**：
   - `cmd.atomic_val = count_delta` (1..255)
   - `cmd.atomic_offset = atomic_byte_off` (非零触发 piggyback 路径)
2. Proxy 看到 `atomic_val > 0 && atomic_offset > 0` → 发 `IBV_WR_RDMA_WRITE_WITH_IMM`
   - imm = `PackAtomicWithSeq(count, offset, seq)`
3. 一条 EFA WR 完成两件事：payload DMA + receiver CPU tail counter advance

约束：
- `count_delta` 1..255（8-bit，受限于 TransferCmd 字段宽度）
- `atomic_byte_off > 0`（slot 0 reserved，用于兼容 V1 的 `atomic_offset > 0` trigger）
- payload ≤ `kTransferCmdMaxBytes`（不分片，因为 count 只能 attach 给一条 WR）

对应 V1 概念: `nvshmemi_ibgda_put_nbi_warp` + piggyback tail

#### red_add_rel<Rail> — 独立 ordered ATOMIC

```cpp
template <typename team_t>
void red_add_rel(void* sym_ptr, int value, int dst_rank,
                 int lane_hint = 0) const;
```

`sym_ptr` 是 `atomic_tail_base` 内的地址（不是 window 内的 sym_ptr——与 NCCL API 不同）。
内部 `rail_red_add`：
1. 编码 `TransferCmd{ATOMIC, value=delta, req_rptr=atomic_byte_off, atomic_offset=1}`
2. `atomic_offset = 1` 触发 proxy 的 `PackAtomicWithSeq` 有序路径
3. Proxy 把 ATOMIC cmd 入 `pending_atomics_` queue，依赖 plain WRITE CQE 全完成后才 post
4. Post 时发一条空 payload `WRITE_WITH_IMM`，imm = `PackAtomicWithSeq(delta, offset, seq)`
5. Receiver proxy 解码 imm，按 seq 排序（reorder buffer），`atomicAdd` 到 host buffer

约束：
- `delta` 范围 ±16383（15-bit signed，PackAtomicWithSeq 限制）
- `atomic_byte_off` ≤ 8191（13-bit）、必须 8-byte 对齐

对应 NCCL GIN: `gin.red_add_rel<Rail>(ptr, value, dst)`

#### put_value<Rail> — 单 word WRITE

```cpp
template <typename team_t>
void put_value(void* sym_ptr, int value, int dst_rank,
               int lane_hint = 0) const;
```

内部逻辑：
1. 把 4-byte value 写到 per-lane staging buffer (`resources_.value_staging + lane * 4`)
2. 调 `rail_put` 从 staging buffer 发 4-byte WRITE 到远端 `sym_ptr`

约束：TransferCmd 不携带 inline data，value 必须先 stage 到 GPU 内存再 RDMA。

对应 NCCL GIN: `gin.putValue<Rail>(ptr, value, dst)`

#### quiet — lane drain

```cpp
void quiet(int lane_hint = 0) const;
```

写一条 `QUIET` cmd 到 lane 的 D2H ring，自旋等待 proxy 消费（tail 越过该 slot）。
保证该 lane 上所有之前的 D2H 命令已被 proxy post。**不保证 EFA delivery 完成。**

对应 NCCL GIN: flush 语义的子集（NCCL flush 还保证远端可见）

#### flush

```cpp
void flush() const { quiet(); }
```

捷径——调 `quiet(0)`。未来如果需要多 lane flush 再扩展。

### 3.3 未实现（有意留 __trap）

- **Lsa/World 分支**：`put<Lsa>`、`red_add_rel<Lsa>` 等全部 `__trap()`。
  这是因为 standalone UCCL-GIN 不需要 NVLink 路径；DeepEP 集成时会组合一个 NCCLGin
  实例来委托这些调用。
- **`get_sym_ptr<Rail>`**：EFA 不能 device-dereference 远端指针。DeepEP 需要此 API
  时再决定返回 offset/metadata 还是报错。

## 4. 底层 rail 操作

`uccl_gin/uccl_gin_rail.cuh` 提供三个不依赖 handle 的裸函数。handle 调它们，DeepEP
kernel 也可以直接调（用于不能用模板封装的地方，如 tail API）。

| 函数 | 产生的 TransferCmd | atomic_val | atomic_offset | proxy 行为 |
|------|-------------------|------------|---------------|-----------|
| `rail_put(dst, bytes, loff, roff)` | WRITE | 0 | 0 | 普通 RDMA WRITE |
| `rail_put_tail_add(dst, bytes, loff, roff, count, tail_off)` | WRITE | count (1..255) | tail_off | WRITE_WITH_IMM (piggyback) |
| `rail_red_add(dst, delta, off)` | ATOMIC | — | 1 | 空 WRITE_WITH_IMM (ordered atomic) |

所有函数返回 `uint64_t slot`——D2H ring slot number。调用方不需要等 slot 完成
（那是 proxy 的事），但 return value 可用于 debug 或跟踪 inflight 数量。

## 5. TransferCmd ABI（16 bytes，继承自 UCCL EP）

```
Byte:  0        1        2      4        6        8       12
     ┌────────┬────────┬────────┬────────┬────────┬────────┐
     │cmd_type│dst_rank│bytes   │req_rptr│req_lptr│atomic  │
     │ WRITE  │        │(2B) +  │(4B)    │/value  │_offset │
     │ ATOMIC │        │atomic  │        │(4B)    │(2B)    │
     │ QUIET  │        │_val(1B)│        │        │        │
     └────────┴────────┴────────┴────────┴────────┴────────┘
```

编码规则（与 V1 兼容）：

- **WRITE cmd** (`cmd_type = WRITE`):
  - `bytes` = payload 字节数（低 2B）+ `atomic_val`（高 1B，0 或 1..255）
  - `req_lptr` = 本地 window offset，右移 `kWriteAddrShiftNormal`(=2) 位（4-byte 粒度）
  - `req_rptr` = 远端 window offset，同上移位
  - `atomic_offset` = 0（plain WRITE）或 >0（piggyback WRITE_WITH_IMM）

- **ATOMIC cmd** (`cmd_type = ATOMIC`):
  - `value` = 有符号 delta（15-bit）
  - `req_rptr` = atomic buffer 内的 RAW byte offset
  - `atomic_offset` = 1（触发 ordered/`PackAtomicWithSeq` 路径）

- **QUIET cmd** (`cmd_type = QUIET`):
  - 其他字段忽略。Proxy 消费后回写 ack，不产生 EFA 操作。

## 6. Proxy 侧新增机制

### 6.1 Sender-side async completion dependency

UCCL EP V1 不需要这个——V1 的 coordinator warp 在 channel 结束时没有独立的 finish
ATOMIC 需要等前面的 payload WRITE。DeepEP V2 有。

实现（`transport/proxy.cpp`）：

```
PendingAtomicBatch {
    vector<uint64_t> wrs;           // 要 post 的 ATOMIC WR
    vector<TransferCmd> cmds;       // 对应的 cmd
    int pending_writes = 0;         // 还有多少依赖 WRITE 未完成
};

post_gpu_commands_mixed():
    for each cmd:
        if cmd is WRITE && cmd.atomic_val == 0:
            wr_id → inflight_write_wrs_
            wr_id → atomic_dependency_wrs_
        // atomic_val > 0: piggyback WRITE_WITH_IMM — 不进 dependency
        // （receiver seq ordering 保证 count 在 finish 之前）

    for each ATOMIC cmd:
        batch = {wrs, cmds, pending_writes = |atomic_dependency_wrs_|}
        for each wr_id in dependency_wrs_:
            atomic_dep_by_wr_[wr_id] = &batch
        pending_atomic_batches_.push_back(batch)

retire_inflight_write(wr_id):
    inflight_write_wrs_.erase(wr_id)
    batch = atomic_dep_by_wr_[wr_id]
    if (--batch->pending_writes == 0):
        post the ATOMIC batch  // 所有 payload CQE 都回来了

progress_pending_atomics():
    while pending_atomic_batches_ not empty:
        if batch.pending_writes == 0:
            post & pop
```

### 6.2 Piggyback tail（WRITE_WITH_IMM）

V1 已有 `PackAtomicWithSeq` + WRITE_WITH_IMM 编码，但没有用在 payload WRITE 上。
UCCL-GIN 在 `rail_put_tail_add` 中复用了 `atomic_val` + `atomic_offset` 字段，
让 proxy 发 `IBV_WR_RDMA_WRITE_WITH_IMM` 同时带 payload 和 tail delta。

### 6.3 Receiver reorder buffer

V1 已有 SeqBuf —— `unordered_map<size_t, SeqBuf>`，按 (ring, tail_slot) 哈希。
每个 WRITE_WITH_IMM 携带 `PackAtomicWithSeq` 编码的 seq number，receiver proxy
按序 apply。乱序暂存，就绪 drain。UCCL-GIN 完全复用。

## 7. Host Context

`context.hpp / context.cpp` 封装 transport 初始化：

```
Context cfg(rank, world_size, local_world_size, max_message_bytes, ifname)
  → cudaMalloc GPU window (2 * max_bytes: send | recv)
  → 创建 kNumProxyThs 个 UcclProxy 实例
  → 每个 proxy 创建自己的 EFA QP/CQ/MR
  → MPI_Allgather PeerMeta（QPN, rkey, buffer addr, IP）
  → proxy.start_dual() 启动 drain/send 线程
  → 收集 D2H ring handle → 组装 UCCLGinResources
  → cudaMalloc value_staging
  → Python binding 暴露 Context + resources()
```

## 8. 测试架构

### C++ microbench (`tests/microbench.cu`)

2-node paired-remote workload。每个 rank 向对端 pair 发数据，然后用 counter/quiet
验证正确性和顺序。支持 `--only` 选测特定 primitive、`--correctness-only` 跳过 benchmark。

当前 coverage：

| 测试 | 源语 | 验证内容 |
|------|------|---------|
| UCCL-put/add | put + red_add_rel | 数据 integrity + put+tail 顺序 |
| UCCL-tail/q | put_tail_add + quiet | piggyback 正确性 + quiet drain |
| UCCL-put+q | put + quiet | quiet 的 happens-before 语义 |
| UCCL-red_add counter | red_add_rel | 多 lane counter 精确值 |
| NCCL (reference) | ncclGin.put | apples-to-apples baseline |

### Python tests (`tests/test_primitives.py`)

通过 subprocess 启动 mpirun + C++ microbench，pytest 风格断言输出中的 PASS/FAIL。
Gateway env var `UCCL_GIN_RUN_PRIMITIVES=1` 控制是否真正启动 MPI。

## 9. 关键设计决策

1. **不继承 NCCLGin**。独立 struct + `if constexpr` 模板分发。理由是 Lsa/World 只是委托
   NCCLGin 而已，不需要继承带来的耦合。而且 DeepEP 的 gin 类型是模板参数，不是虚函数。

2. **Tail 存储独立于 window**。NCCL `red_add_rel` 直接在 GPU window 内做 atomic add。
   EFA 不支持硬件 RDMA atomics，`PackAtomicWithSeq` 的 offset 只有 13-bit，装不下 window
   tail offset。所以 tail 迁移到 `atomic_tail_base`——host-mapped compact buffer，用
   (channel, src_rank) 索引访问。

3. **`put_tail_add` 是 UCCL-GIN 独有的源语**。NCCL 用 `put` + `red_add_rel` + NIC FORCE_SO
   保证顺序。我们把它合成一条 WRITE_WITH_IMM。这不是 code cleanup——是物理必要的：EFA 不保证
   独立 WR 和 ATOMIC 之间的顺序，合成一条就消除了这个 ordering gap。

4. **TransferCmd ABI 不变**。16B、CmdType 编码、offset 移位——全部和 V1 一样。改动的是
   GPU 侧如何填这些字段（`rail_put_tail_add` 复用 `atomic_val`），以及 proxy 侧如何理解
   （`atomic_val > 0` → piggyback path）。

5. **Sender-side dependency 只覆盖 finish**。Piggyback count delta 和 finish 打到同一个
   receiver tail counter，receiver seq ordering 保证 count 在 finish 之前。只有不带 count
   的 plain WRITE 才需要 sender 等 CQE。这使 dependency_max 从 72 降到 2。

6. **没有实现 Lsa**。Standalone 不接 DeepEP，不需要 NVLink 路径。集成时组合一个 NCCLGin
   实例委托所有 Lsa 调用即可。
