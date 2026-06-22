# UCCL-GIN 代码阅读指南

按数据流顺序，从 API 到 transport，逐层讲解。

## 文件阅读顺序

```
第一层   uccl_gin/resources.cuh       数据结构（5 分钟）
第二层   uccl_gin/uccl_gin_rail.cuh   底层原语（10 分钟）
第三层   uccl_gin/uccl_gin.cuh         API 面（15 分钟）
第四层   context.hpp / context.cpp     Host 初始化（10 分钟）
第五层   tests/microbench.cu           测试 & 全链路（15 分钟）
```

---

## 第一层：数据结构 `uccl_gin/resources.cuh`

```
┌──────────────────────────────────────────────────────────────┐
│ UCCLGinResources                                             │
│                                                              │
│  d2h_queues[] ──→ GPU 可访问的 D2H ring handle 数组         │
│  window_base   ──→ 所有 put offset 的零点                    │
│  atomic_tail_base → red_add_rel counter 的零点               │
│                                                              │
│  num_scaleout_ranks = 2  (节点数)                            │
│  num_scaleup_ranks  = 8  (每节点 GPU 数)                    │
│  num_lanes = proxy 线程数                                    │
│                                                              │
│  queue_index_from_hint():                                    │
│    hint → [round-robin across proxies] → D2H queue index     │
│                                                              │
│  validate_rail_dst():                                        │
│    校验 dst rank 是合法的 paired-remote 目标                 │
└──────────────────────────────────────────────────────────────┘
```

这个 struct 是 Host → GPU 的契约。所有内容都是 POD——可以直接 by-value 传给
kernel launch。它包含 device-visible 指针：D2H handle 和 `atomic_tail_base` 可以
指向 GPU 可见的 host-mapped memory，`window_base` 则指向注册的 payload window。
这里没有 EP layout/expert 语义。

**关键点**: `queue_index_from_hint` 的 round-robin 映射确保了流量均匀分布到所有
proxy 线程和 NIC，而不是全压在第一个 proxy 上。

---

## 第二层：底层原语 `uccl_gin/uccl_gin_rail.cuh`

四个函数，每个编码一种 TransferCmd：

```
                     GPU 侧                              Proxy 侧
                     ──────                              ────────
rail_put ──────────→ TransferCmd{WRITE} ─────────────→ 普通 RDMA WRITE
rail_put_tail_add ─→ TransferCmd{WRITE, atomic_val>0} → WRITE_WITH_IMM
rail_red_add ──────→ TransferCmd{ATOMIC} ─────────────→ 空 WRITE_WITH_IMM
rail_write_value ──→ TransferCmd{WRITE_VALUE} ────────→ host bounce + WRITE
```

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  GPU 虚拟地址                                                     │
│      │                                                           │
│      │ window_off(addr, window_base, window_bytes, bytes)        │
│      │   ├─ addr ∈ [base, base+bytes)?                          │
│      │   ├─ (addr-base) 4-byte 对齐?                            │
│      │   └─ return (addr - base) >> 2  // 4-byte 移位 offset    │
│      ▼                                                           │
│  32-bit 移位 offset                                               │
│      │                                                           │
│      │ 编码进 TransferCmd.req_lptr / req_rptr / ...              │
│      ▼                                                           │
│  TransferCmd (16B) ──→ atomic_set_and_commit() ──→ D2H ring     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**核心设计决策**:
- `rail_put_tail_add` 是 UCCL-GIN 独有的。它在一条 WR 里同时完成 payload 传输
  和 tail counter 更新——消除了 EFA 上独立 WR 和 ATOMIC 之间的 ordering gap。
- `rail_write_value` 也是独有的。解决了 GPU store → NIC DMA 之间的 visibility
  race：inline data 先拷到 host bounce buffer，再从 host MR 发 RDMA。
- 所有函数的 inflight cap 参数 `kUCCLGinMaxInflightNormal`(默认 8) 提供可控
  背压。它本身不足以证明 4-bit ordered-atomic sequence 永不 alias；同一
  counter 还必须固定到同一 lane，并对 per-counter outstanding 边界单独建立不变式。

---

## 第三层：API 面 `uccl_gin/uccl_gin.cuh`

`struct UCCLGin` — 所有方法都是 `const`、device-side、模板参数 `team_t` 分发：

```
                     team_t == Rail            team_t == Lsa
                     ─────────────             ────────────
put              →   rail_put (可分片)        __trap()
put_tail_add     →   rail_put_tail_add         __trap()
red_add_rel      →   rail_red_add              __trap()
put_value        →   rail_write_value          __trap()
quiet            →   QUIET cmd + spin-wait     —
flush            →   for each q: quiet(q)      —
```

### put 的大 payload 分片

```
                         num_bytes > kTransferCmdMaxBytes?
                                   │
                    ┌──────────────┴──────────────┐
                    │ YES                         │ NO
                    ▼                             ▼
          while (remaining > 0)           rail_put(全部 bytes)
            chunk = min(remaining, kTransferCmdMaxAlignedBytes)
            rail_put(dst, chunk, ...)
            remaining -= chunk
```

NCCL GIN 的 put 是一条 WR，NIC DMA 硬件内部管分片。UCCL-GIN 在软件层拆多 WR——
每条独立 RDMA WRITE、独立 completion。多个 chunk 在 EFA 上可能乱序到达（但写
不重叠偏移，数据正确）。DeepEP token ~14KB 不会触发。

### 完整的 put + red_add_rel 链路

```
GPU kernel (sender)                     CPU proxy (sender)          CPU proxy (receiver)
───────────────────                     ─────────────────          ───────────────────

① gin.put<Rail>(recv, send, bytes, dst)
   → rail_put → TransferCmd{WRITE}
   → D2H ring, slot N
                                         ③ drain ring
                                           cmd.atomic_val == 0
                                           → post RDMA WRITE
                                           → wr_id → inflight_write_wrs_
                                           → wr_id → atomic_dependency_wrs_

② gin.red_add_rel<Rail>(ptr, val, dst)
   → rail_red_add → TransferCmd{ATOMIC}
   → D2H ring, slot N+1
                                         ④ drain ring
                                           enqueue_pending_atomics:
                                             batch.pending_writes = |dep_wrs_|
                                             atomic_dep_by_wr_[wr_id] = &batch
                                                                         ⑤ CQE poll
                                                                            retire_inflight_write:
                                                                            batch.pending_writes--
                                                                            if 0 → post ATOMIC
                                                                            ↓
                                                                      ⑥ WRITE_WITH_IMM (空 payload)
                                                                         imm=PackAtomicWithSeq(seq,off,val)
                                                                             ↓
                                                                         ⑦ decode imm
                                                                            reorder buffer (按 seq)
                                                                            atomicAdd(tail, val)
```

---

## 第四层：Host 初始化 `context.hpp` + `context.cpp`

```
Context::setup(cfg)
  │
  ├─ ① cudaSetDevice(local_rank)
  ├─ ② cudaMalloc GPU window (2 × max_message_bytes: send | recv)
  ├─ ③ 创建 kNumProxyThs 个 UcclProxy 实例
  │     每个 proxy: EFA QP + CQ + MR + D2H ring + proxy thread
  ├─ ④ MPI_Allgather PeerMeta (QPN, rkey, buffer addr, IP, port)
  ├─ ⑤ proxy.start_dual() → 启动 drain/send 线程
  ├─ ⑥ 收集 D2H handle → cudaMemcpy 到 device array
  ├─ ⑦ 设置 shared atomic buffer (所有 proxy 共享 proxy[0] 的)
  ├─ ⑧ 组装 UCCLGinResources:
  │      d2h_queues, num_queues,
  │      window_base, window_bytes, atomic_tail_base,
  │      num_scaleout_ranks, num_scaleup_ranks,
  │      scaleout_rank, scaleup_rank, num_lanes
  └─ ⑨ 完成资源包；WRITE_VALUE host bounce slot 由各 proxy 初始化并注册
```

```
┌──────────┐           ┌──────────────┐          ┌──────────────┐
│ Rank 0   │           │ Rank 8       │          │ Rank ...      │
│ GPU 0    │           │ GPU 0        │          │               │
│ node 0   │           │ node 1       │          │               │
└────┬─────┘           └──────┬───────┘          └───────┬───────┘
     │                        │                          │
     │  ① 各自 cudaMalloc window                           │
     │  ② 各自创建 proxy (EFA QP/CQ/MR)                    │
     │  ③ MPI_Allgather PeerMeta                          │
     │  ④ 各自 start proxy threads                        │
     │  ⑤ 各自填充 UCCLGinResources                        │
     │                                                     │
     ▼                        ▼                          ▼
  UCCLGin(resources)     UCCLGin(resources)     UCCLGin(resources)
     │                        │                          │
     │  gin.put<Rail>(        │  gin.put<Rail>(          │
     │    recv, send,         │    recv, send,           │
     │    bytes, rank8)       │    bytes, rank0)         │
     ▼                        ▼                          ▼
  D2H ring → proxy → EFA WRITE → peer's GPU HBM
```

---

## 第五层：测试 `tests/microbench.cu`

```
main()
  │
  ├─ MPI_Init, cudaSetDevice
  ├─ NCCL-GIN setup (optional)
  ├─ Context setup (UCCL-GIN)
  │
  ├─ ─── correctness pass ───
  │   │
  │   ├─ verify_uccl_red_add()       # red_add_rel counter 精确值
  │   ├─ verify_uccl_put_value()     # put_value + red_add 完成信号
  │   │
  │   │  for each size:
  │   ├─ verify_nccl()               # NCCL put+signal 参考
  │   ├─ verify_uccl()               # UCCL put + red_add_rel
  │   ├─ verify_uccl_tailadd()       # UCCL put_tail_add + quiet
  │   └─ verify_uccl_put_quiet()     # UCCL put + quiet ordering
  │
  ├─ ─── BW sweep ───
  │   for each size:
  │     run_nccl_gin()  → timed NCCL kernel
  │     run_uccl_gin()  → timed UCCL kernel
  │
  └─ cleanup
```

### 每个 verify 函数测试什么

```
verify_uccl (put + red_add_rel):
  ┌────────┐  ① put (pattern data)     ┌────────┐
  │ rank r │ ─────────────────────────→ │ peer p │
  │        │  ② red_add_rel(counter,1) │        │
  │        │ ─────────────────────────→ │        │
  └────────┘                            └───┬────┘
                                            │ ③ spin counter==1
                                            │ ④ verify_recv == peer's pattern
                                            ▼
                                         PASS/FAIL

  验证：plain WRITE + ordered ATOMIC 的 payload-before-tail 顺序。
  如果 red_add 在 payload 之前到达 receiver，counter=1 但 recv 还是 poison → FAIL。

verify_uccl_tailadd (put_tail_add + quiet):
  ┌────────┐  ① put_tail_add (pat.+cnt)  ┌────────┐
  │ rank r │ ──────────────────────────→  │ peer p │
  │        │  ② quiet(lane)              │        │
  └────────┘                              └───┬────┘
                                              │ ③ wait counter==iters
                                              │ ④ verify_recv
                                              ▼
                                           PASS/FAIL

  验证：piggyback put 的正确性——payload 和 count 在同一条 WR 里，不可分。
  且 quiet() 不会死锁。

verify_uccl_put_quiet (put + quiet):
  ┌────────┐  ① put (pattern)       ┌────────┐
  │ rank r │ ──────────────────────→ │ peer p │
  │        │  ② quiet(lane)         │        │
  └────────┘                         └───┬────┘
                                         │ ③ MPI_Barrier settle
                                         │ ④ verify_recv
                                         ▼
                                      PASS/FAIL

  验证：quiet() 等到本 lane 之前 WRITE/WRITE_VALUE 的 sender CQE，因而
  source buffer 可安全复用。
  限制：sender completion 不单独构成 receiver visibility 协议；测试使用额外
  `MPI_Barrier` settle 后再校验远端。大 payload(>4MB) 当前 skip。

verify_uccl_red_add (纯 counter):
  ┌────────┐  red_add × iters × lanes  ┌────────┐
  │ rank r │ ─────────────────────────→ │ peer p │
  └────────┘                            └───┬────┘
                                            │ wait all slots == iters
                                            │ verify exact count per slot
                                            ▼
                                         PASS/FAIL

  验证：多 lane atomic add 的精确性——每个 lane 独立 counter，最终值 == iters。

verify_uccl_put_value (put_value + red_add):
  ┌────────┐  ① put_value × iters       ┌────────┐
  │ rank r │ ──────────────────────────→ │ peer p │
  │        │  ② red_add_rel(counter,1)  │        │
  └────────┘                             └───┬────┘
                                             │ ③ spin counter==1
                                             │ ④ verify each slot == 0xDEAD0000+i
                                             ▼
                                          PASS/FAIL

  验证：WRITE_VALUE 路径——inline data 正确地从 host bounce buffer 到达远端 HBM。
```

---

## 最小集成清单

如果要让 DeepEP V2 或 NCCL-EP 使用 UCCL-GIN，需要：

1. **Build 时**: `-I experimental/uccl_gin`（让 kernel 能 include `uccl_gin/uccl_gin.cuh`）
2. **Host 侧**: 创建 `uccl_gin::Context`，获取 `resources()`，传给 kernel launcher
3. **Kernel 侧**: 用 `UCCLGin` 替代 `NCCLGin`，调用面相同
4. **Tail 存储**: 接受 tail counter 在 `atomic_tail_base` 而非 window 内
5. **Ordering**: 用 `put_tail_add` 替代 `put` + `red_add_rel`（或接受 sender-side dep overhead）
