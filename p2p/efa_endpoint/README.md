# EFA P2P RDMA 示例

这个项目演示了如何使用 AWS EFA (Elastic Fabric Adapter) 进行点对点 RDMA 通信，使用了清晰的架构和内存管理。

## 架构设计

### 核心组件

1. **MemoryPool** (`memory_pool.h`)
   - 使用 mmap 分配页对齐的内存，满足 RDMA 注册要求
   - 支持 HOST 和 GPU 内存类型
   - 内存复用和池化管理

2. **RdmaMemoryManager** (`rdma_memory_manager.h`)
   - 负责内存分配和 RDMA 注册
   - 两个核心功能：
     - `allocate_and_register()`: 分配新内存并注册
     - `register_memory()`: 注册已分配的内存
   - 可选使用 MemoryPool 进行内存管理

3. **RdmaContext** (`rdma_context.h`)
   - 简单的 RDMA 上下文封装
   - 不包含内存管理功能，保持职责单一
   - 提供设备、PD、GID 查询等基础功能

4. **EFAChannel** (`efa_channel.h`)
   - EFA 特定的通信通道
   - 支持 RDMA Write with Immediate
   - 需要外部提供 lkey（通过 RdmaMemoryManager 获取）

5. **MetadataExchanger** (`metadata_exchange.h`)
   - 基于 TCP 的 metadata 交换
   - 用于交换 QPN, GID, RKEY, ADDR 等信息
   - 支持多节点之间的连接建立

## 编译

```bash
# 确保先 deactivate conda 环境
conda deactivate

# 编译
make efa_p2p_example

# 清理
make clean
```

## 使用示例

### 两节点通信

在节点 0 上运行：
```bash
./efa_p2p_example 0 <node1_ip>
```

在节点 1 上运行：
```bash
./efa_p2p_example 1 <node0_ip>
```

### 参数说明

```
./efa_p2p_example <rank> <peer_ip> [peer_rank]
  rank: 本节点的 rank (0 或 1)
  peer_ip: 对端节点的 IP 地址
  peer_rank: 对端节点的 rank (可选，默认为 1-rank)
```

## 代码示例

```cpp
// 1. 创建内存池和管理器
auto mem_pool = std::make_shared<MemoryPool>();
auto mem_mgr = std::make_shared<RdmaMemoryManager>(ctx->getPD(), mem_pool);

// 2. 分配并注册内存
void* buf = mem_mgr->allocate_and_register(4096, MemoryType::HOST);
struct ibv_mr* mr = mem_mgr->get_mr(buf);

// 3. 准备 metadata
metadata local_meta = {
    .qpn = channel->getQP()->qp_num,
    .gid = /* GID */,
    .rkey = mr->rkey,
    .addr = (uint64_t)buf
};

// 4. 交换 metadata
MetadataExchanger exchanger(my_rank);
exchanger.init();
metadata remote_meta;
exchanger.exchange(peer_ip, peer_rank, local_meta, &remote_meta);

// 5. 建立连接并通信
channel->connect(remote_meta);
channel->send(sendReq, mr->lkey);
```

## 内存管理特性

### MemoryPool
- 使用 mmap 分配内存，确保页对齐 (4KB)
- 支持内存复用，提高性能
- 自动对齐到 RDMA 注册要求
- 支持 HOST 和 GPU 内存

### RdmaMemoryManager
- 统一的内存分配和注册接口
- 自动追踪已注册的内存
- 支持两种模式：
  1. 使用 MemoryPool 进行池化管理
  2. 直接分配（fallback 模式）
- 析构时自动清理所有注册的内存

## GPU 支持

要启用 GPU 内存支持，编译时需要定义 `HAVE_CUDA`：
```bash
g++ -DHAVE_CUDA ... main.cc -o efa_p2p_example
```

使用 GPU 内存：
```cpp
void* gpu_buf = mem_mgr->allocate_and_register(size, MemoryType::GPU);
```

## 注意事项

1. **页对齐**: 所有内存都会自动对齐到 4KB 页边界
2. **内存生命周期**: 使用 MemoryPool 时，内存会被复用而不是立即释放
3. **线程安全**: MemoryPool 和 RdmaMemoryManager 都是线程安全的
4. **端口配置**: 默认使用 12345 + rank 作为 TCP 端口，可以修改
5. **超时设置**: metadata 交换默认超时 10 秒，可调整

## 故障排除

### 编译错误
- 确保已安装 EFA 驱动和 libibverbs
- 检查 CUDA 路径是否正确（如果使用 GPU）

### 运行时错误
- 检查防火墙是否允许 TCP 端口（12345-12346）
- 确保两个节点都能访问 EFA 设备
- 检查 GID 是否正确配置

### 连接超时
- 确保对端节点已经启动
- 检查网络连接
- 增加超时时间

## 扩展

要支持更多节点或更复杂的拓扑，可以：
1. 扩展 MetadataExchanger 支持多对多连接
2. 使用集中式的 metadata 服务器
3. 集成到 MPI 或其他分布式框架中

## 许可证

SPDX-License-Identifier: (GPL-2.0 OR BSD-2-Clause)
