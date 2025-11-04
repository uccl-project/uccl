# Multiple QPs Under Same Context - Complete Guide

## 简答

**是的！同一个 `ibv_context` 下可以创建多个 QP 和 QP_EX。**

## 详细说明

### RDMA资源层次结构

```
┌─────────────────────────────────────┐
│   ibv_context (RDMA Device)         │
│   - 一个设备一个context              │
│   - 代表一个物理HCA/NIC              │
└─────────────────────────────────────┘
          │
          ├──────────────┬──────────────┬──────────────┐
          ▼              ▼              ▼              ▼
     ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
     │ PD #1  │    │ PD #2  │    │  CQ #1 │    │  CQ #2 │
     └────────┘    └────────┘    └────────┘    └────────┘
          │              │
          ├────┬────┬────┤
          ▼    ▼    ▼    ▼
       QP #1 QP #2 QP #3 QP #4  ... (可以创建多个)
```

### 资源限制

每个RDMA设备都有硬件限制，可以通过 `ibv_query_device()` 查询：

```cpp
struct ibv_device_attr attr;
ibv_query_device(ctx, &attr);

printf("Max QPs: %d\n", attr.max_qp);        // 通常: 数千到数万
printf("Max CQs: %d\n", attr.max_cq);        // 通常: 数千
printf("Max PDs: %d\n", attr.max_pd);        // 通常: 数百
printf("Max MRs: %d\n", attr.max_mr);        // 通常: 数万
```

### 典型硬件限制（参考值）

| 设备类型 | Max QPs | Max CQs | Max PDs |
|---------|---------|---------|---------|
| Mellanox ConnectX-4 | 262,144 | 16,384 | 32,768 |
| Mellanox ConnectX-5 | 524,288 | 32,768 | 65,536 |
| AWS EFA (Gen 1) | 2,048 | 1,024 | 256 |
| AWS EFA (Gen 2) | 8,192 | 4,096 | 1,024 |

## 使用场景

### 1. 多对多通信（All-to-All）

```cpp
// 在分布式训练中，每个节点与其他所有节点建立QP连接
int num_nodes = 16;
std::vector<ibv_qp*> qps;

for (int peer = 0; peer < num_nodes; peer++) {
    if (peer != my_rank) {
        ibv_qp* qp = create_qp(ctx, pd, cq);  // 同一个ctx
        qps.push_back(qp);
        // 与 peer 节点交换 QP 信息并连接
        connect_qp(qp, peer);
    }
}
// 结果：15个QP，全部在同一个ctx下
```

### 2. QoS优先级分级

```cpp
// 不同优先级的数据通道
struct {
    ibv_qp* control_qp;    // 控制信息（高优先级）
    ibv_qp* data_qp;       // 数据传输（中优先级）
    ibv_qp* background_qp; // 后台任务（低优先级）
} communication_channels;

// 全部使用同一个 ctx
communication_channels.control_qp = ibv_create_qp(pd, &attr);
communication_channels.data_qp = ibv_create_qp(pd, &attr);
communication_channels.background_qp = ibv_create_qp(pd, &attr);
```

### 3. 多轨并行传输（Multi-rail）

```cpp
// 使用多个QP并行传输，提高带宽利用率
const int NUM_RAILS = 4;
ibv_qp* rails[NUM_RAILS];

for (int i = 0; i < NUM_RAILS; i++) {
    rails[i] = ibv_create_qp(pd, &qp_attr);  // 同一个 pd, 同一个 ctx
}

// 大数据分片并行发送
void send_large_data(char* data, size_t size) {
    size_t chunk_size = size / NUM_RAILS;
    for (int i = 0; i < NUM_RAILS; i++) {
        size_t offset = i * chunk_size;
        rdma_post_send(rails[i], data + offset, chunk_size);
    }
}
```

### 4. 不同传输类型

```cpp
// 同一个ctx下创建不同类型的QP
struct ibv_qp_init_attr attr_rc = {...};
attr_rc.qp_type = IBV_QPT_RC;  // Reliable Connection
ibv_qp* qp_rc = ibv_create_qp(pd, &attr_rc);

struct ibv_qp_init_attr attr_uc = {...};
attr_uc.qp_type = IBV_QPT_UC;  // Unreliable Connection
ibv_qp* qp_uc = ibv_create_qp(pd, &attr_uc);

struct ibv_qp_init_attr attr_ud = {...};
attr_ud.qp_type = IBV_QPT_UD;  // Unreliable Datagram
ibv_qp* qp_ud = ibv_create_qp(pd, &attr_ud);
```

## 资源共享策略

### 策略1：共享PD和CQ（推荐用于简单场景）

```cpp
// 所有QP共享相同的PD和CQ
ibv_pd* shared_pd = ibv_alloc_pd(ctx);
ibv_cq* shared_cq = ibv_create_cq(ctx, 1024, NULL, NULL, 0);

for (int i = 0; i < num_qps; i++) {
    qp_attr.send_cq = shared_cq;
    qp_attr.recv_cq = shared_cq;
    qps[i] = ibv_create_qp(shared_pd, &qp_attr);
}

// 优点：节省资源
// 缺点：需要在poll CQ时区分不同QP的completion
```

### 策略2：独立CQ（推荐用于高性能场景）

```cpp
// 每个QP有独立的CQ，共享PD
ibv_pd* shared_pd = ibv_alloc_pd(ctx);

for (int i = 0; i < num_qps; i++) {
    ibv_cq* dedicated_cq = ibv_create_cq(ctx, 128, NULL, NULL, 0);
    qp_attr.send_cq = dedicated_cq;
    qp_attr.recv_cq = dedicated_cq;
    qps[i] = ibv_create_qp(shared_pd, &qp_attr);
}

// 优点：可以独立poll，减少contention
// 缺点：消耗更多硬件资源
```

### 策略3：独立PD（用于隔离安全）

```cpp
// 每个QP有独立的PD和CQ
for (int i = 0; i < num_qps; i++) {
    ibv_pd* dedicated_pd = ibv_alloc_pd(ctx);
    ibv_cq* dedicated_cq = ibv_create_cq(ctx, 128, NULL, NULL, 0);
    qp_attr.send_cq = dedicated_cq;
    qp_attr.recv_cq = dedicated_cq;
    qps[i] = ibv_create_qp(dedicated_pd, &qp_attr);
}

// 优点：不同PD的MR互相隔离，安全性高
// 缺点：消耗最多资源，无法跨PD共享MR
```

## 实际应用案例

### NCCL (NVIDIA Collective Communications Library)

NCCL在每个GPU之间会创建多个QP：
- 每个peer connection = 多个QP（用于不同的ring/tree算法）
- 8 GPUs x 7 peers x 4 QPs/peer = 224 QPs per node
- 全部在同一个 `ibv_context` 下

### MPI (Message Passing Interface)

Open MPI的OpenIB模块：
- 每个MPI rank pair之间可能有多个QP
- Eager protocol QP + Rendezvous protocol QP
- 64 ranks x 63 peers x 2 QPs = 8,064 QPs
- 全部共享同一个设备context

### RDMA-based Storage (NVMe-oF, SPDK)

- 每个I/O queue = 一个QP
- 一个NVMe controller可能有数十个I/O queues
- 多个controllers共享同一个RDMA设备
- 所有QPs在同一个context下

## QP_EX (Extended QP) 注意事项

### 创建 QP_EX

```cpp
// QP_EX也是通过同一个context创建
struct ibv_cq_init_attr_ex cq_attr = {};
cq_attr.cqe = 128;
ibv_cq_ex* cq_ex = ibv_create_cq_ex(ctx, &cq_attr);  // 同一个ctx

struct ibv_qp_init_attr_ex qp_attr = {};
qp_attr.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
qp_attr.pd = pd;
qp_attr.send_cq = ibv_cq_ex_to_cq(cq_ex);
qp_attr.recv_cq = ibv_cq_ex_to_cq(cq_ex);
qp_attr.qp_type = IBV_QPT_DRIVER;  // EFA uses DRIVER type
qp_attr.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE | IBV_QP_EX_WITH_SEND;

ibv_qp* qp_ex = ibv_create_qp_ex(ctx, &qp_attr);  // 同一个ctx
```

### QP vs QP_EX 混合使用

```cpp
// 可以在同一个context下混合使用传统QP和QP_EX
std::vector<ibv_qp*> regular_qps;
std::vector<ibv_qp*> extended_qps;

// 创建传统QP
for (int i = 0; i < 5; i++) {
    regular_qps.push_back(ibv_create_qp(pd, &qp_init_attr));
}

// 创建扩展QP
for (int i = 0; i < 5; i++) {
    extended_qps.push_back(ibv_create_qp_ex(ctx, &qp_init_attr_ex));
}

// 全部10个QP都在同一个ctx下
```

## 性能考虑

### 1. CQ竞争

```cpp
// 问题：多个QP共享一个CQ时，poll_cq会成为瓶颈
shared_cq = ibv_create_cq(ctx, 1024, ...);
for (qp : qps) {
    qp->send_cq = shared_cq;  // 所有QP竞争同一个CQ
}

// 解决方案：使用多个CQ
for (int i = 0; i < num_qps; i++) {
    cqs[i] = ibv_create_cq(ctx, 128, ...);
    qps[i]->send_cq = cqs[i];  // 每个QP独立CQ
}
```

### 2. 线程模型

```cpp
// 模型1：单线程轮询所有QP的CQ
void single_thread_poll() {
    while (running) {
        for (int i = 0; i < num_qps; i++) {
            ibv_poll_cq(qps[i]->send_cq, 1, &wc);
        }
    }
}

// 模型2：每个QP一个专用线程（高并发）
void per_qp_thread(int qp_id) {
    while (running) {
        ibv_poll_cq(qps[qp_id]->send_cq, 32, wcs);
        // 处理completions
    }
}
```

### 3. 内存注册

```cpp
// 同一个MR可以被多个QP使用（如果它们共享同一个PD）
ibv_mr* shared_mr = ibv_reg_mr(shared_pd, buf, size, access_flags);

// 所有QP都可以使用这个MR
for (int i = 0; i < num_qps; i++) {
    struct ibv_sge sge = {
        .addr = (uint64_t)buf,
        .length = size,
        .lkey = shared_mr->lkey  // 所有QP都用同一个lkey
    };
    // post send/recv with this sge
}
```

## 调试技巧

### 查看QP数量

```bash
# 查看当前系统的QP使用情况
cat /sys/class/infiniband/*/ports/*/counters/port_rcv_data
cat /sys/class/infiniband/*/ports/*/counters/port_xmit_data

# 使用 ibv_devinfo 查看设备能力
ibv_devinfo -v

# 使用 perfquery 查看端口统计（需要ibutils）
perfquery
```

### 代码中track QP

```cpp
struct qp_tracker {
    std::map<uint32_t, ibv_qp*> qp_map;  // qp_num -> qp
    std::map<ibv_qp*, std::string> qp_names;

    void add_qp(ibv_qp* qp, const std::string& name) {
        qp_map[qp->qp_num] = qp;
        qp_names[qp] = name;
        printf("Added QP: %s (num=%u, ctx=%p)\n",
               name.c_str(), qp->qp_num, qp->context);
    }

    void print_all() {
        printf("Total QPs under context: %zu\n", qp_map.size());
        for (auto& kv : qp_names) {
            printf("  - %s: QP#%u\n", kv.second.c_str(), kv.first->qp_num);
        }
    }
};
```

## 常见问题

### Q1: 一个context最多能创建多少个QP？

**A:** 取决于硬件限制，可以通过 `device_attr.max_qp` 查看。通常是数千到数万个。

### Q2: 多个QP会不会影响性能？

**A:** 主要看CQ的设计。如果合理分配CQ，多QP可以提高并发性能。如果所有QP共享一个CQ，poll_cq可能成为瓶颈。

### Q3: QP之间会互相影响吗？

**A:** 不同QP在逻辑上是独立的。但它们共享：
- 硬件资源（PCIe带宽、NIC缓冲区）
- CQ（如果配置为共享）
- 内存带宽

### Q4: 创建QP失败怎么办？

**A:** 常见原因：
1. 超过硬件限制 (`max_qp`)
2. 内存不足
3. 设备驱动问题

```cpp
ibv_qp* qp = ibv_create_qp(pd, &attr);
if (!qp) {
    if (errno == ENOMEM) {
        printf("Not enough memory or exceeded max_qp\n");
    } else if (errno == EINVAL) {
        printf("Invalid QP attributes\n");
    }
    // 查询当前QP使用情况
    ibv_query_device(ctx, &device_attr);
}
```

### Q5: 可以动态创建和销毁QP吗？

**A:** 可以，但要注意：
1. 确保QP上没有未完成的操作
2. 先销毁QP，再销毁依赖的CQ和PD
3. 正确的销毁顺序：`QP -> CQ -> MR -> PD -> Context`

## 示例代码编译和运行

```bash
# 编译示例
g++ -std=c++17 multiple_qp_example.cpp -o multiple_qp_example -libverbs -lefa

# 运行（需要有RDMA设备）
./multiple_qp_example

# 预期输出：
# - 设备信息
# - 成功创建5个QP
# - 所有QP的context指针相同
# - QP编号不同
```

## 总结

✅ **同一个 `ibv_context` 下可以创建多个 QP 和 QP_EX**

✅ **这是RDMA编程的标准做法**

✅ **常用于：多连接、QoS、多轨传输、分布式通信**

✅ **资源可以共享（PD、CQ）或独立，根据需求选择**

✅ **注意硬件限制和性能优化**
