# TCPX Plugin API 设计文档

_为 NIXL 插件提供的 TCPX 传输层 API_

## 🎯 设计目标

1. **隐藏底层细节**：插件不需要知道 ChannelManager、SlidingWindow、UnpackLauncher 的存在
2. **符合 NIXL 契约**：API 设计与 nixlBackendEngine 接口对齐
3. **可测试**：API 可以独立于 NIXL 框架进行测试
4. **高性能**：零拷贝、异步、多通道并行

## 📐 核心抽象

### 1. `TcpxSession` - 会话管理

**职责**：管理与一个或多个远程节点的连接

```cpp
class TcpxSession {
public:
  // 构造函数
  TcpxSession(int gpu_id, int num_channels);
  ~TcpxSession();

  // 生命周期管理（完整的握手流程）
  std::string listen();  // Server: 创建 listen comms，返回序列化的 handles
  int accept(const std::string& remote_name);  // Server: accept 连接
  int loadRemoteConnInfo(const std::string& remote_name, const std::string& conn_info);
  int connect(const std::string& remote_name);  // Client: 连接到 server
  int disconnect(const std::string& remote_name);

  // 内存管理（支持多个独立的内存注册）
  struct MemoryHandle {
    void* buffer;
    size_t size;
    int ptr_type;
    bool is_recv;
    void* mhandle;  // TCPX memory handle
    uint64_t id;    // 唯一标识符
  };

  uint64_t registerMemory(void* buffer, size_t size, int ptr_type, bool is_recv);
  int deregisterMemory(uint64_t mem_id);
  MemoryHandle* getMemoryHandle(uint64_t mem_id);

  // 传输操作（返回 TcpxTransfer 对象）
  TcpxTransfer* createTransfer(const std::string& remote_name);

private:
  struct Impl;  // PIMPL 模式，隐藏实现细节
  std::unique_ptr<Impl> impl_;
};
```

**实现细节**（在 `TcpxSession::Impl` 中）：
- `ChannelManager* mgr_`
- `std::map<std::string, std::vector<ncclNetHandle_v7>> remote_handles_`（存储远程节点的 handles）
- `std::map<std::string, bool> remote_accepted_`（跟踪 accept 状态）
- `CUdevice cu_dev_`
- `CUcontext cu_ctx_`
- `cudaStream_t unpack_stream_`
- `tcpx::device::UnpackLauncher* launcher_`
- `std::map<uint64_t, MemoryHandle> registered_memory_`（多个内存注册）
- `uint64_t next_mem_id_ = 0`

### 2. `TcpxTransfer` - 传输请求

**职责**：表示一次传输操作（send 或 recv）

```cpp
class TcpxTransfer {
public:
  // 发起传输（需要提供 memory_id）
  int postSend(uint64_t mem_id, size_t offset, size_t size, int tag);
  int postRecv(uint64_t mem_id, size_t offset, size_t size, int tag);

  // 批量发起（多个 chunk）
  int postSendBatch(const std::vector<uint64_t>& mem_ids,
                    const std::vector<size_t>& offsets,
                    const std::vector<size_t>& sizes,
                    const std::vector<int>& tags);
  int postRecvBatch(const std::vector<uint64_t>& mem_ids,
                    const std::vector<size_t>& offsets,
                    const std::vector<size_t>& sizes,
                    const std::vector<int>& tags);

  // 轮询完成状态
  bool isComplete();
  int wait(int timeout_ms = -1);  // -1 = 无限等待

  // 清理传输资源
  // ⭐ send 请求：无需特殊处理（no-op）
  // ⭐ recv 请求：调用 tcpx_irecv_consumed() 释放 TCPX slots
  int release();

private:
  friend class TcpxSession;
  TcpxTransfer(TcpxSession::Impl* session, const std::string& remote_name);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};
```

**实现细节**（在 `TcpxTransfer::Impl` 中）：
- `TcpxSession::Impl* session_`（指向父 session）
- `std::string remote_name_`
- `std::vector<ChannelWindow> channel_windows_`（每个通道的滑动窗口）
- `std::vector<PostedChunk> all_posted_chunks_`（所有已发起的请求）
- `int total_send_chunks_`（总 send chunk 数）
- `int total_recv_chunks_`（总 recv chunk 数）
- `int completed_send_chunks_`（已完成 send chunk 数）
- `int completed_recv_chunks_`（已完成 recv chunk 数）
- `bool is_send_complete_`
- `bool is_recv_complete_`
- `int next_channel_`（Round-robin 通道选择）

### 3. `TcpxMemoryDescriptor` - 内存元数据

**职责**：序列化/反序列化内存信息（用于 NIXL 的 getPublicData/loadRemoteMD）

```cpp
struct TcpxMemoryDescriptor {
  uint64_t base_addr;   // GPU 内存基地址
  size_t size;          // 内存大小
  int ptr_type;         // NCCL_PTR_CUDA
  
  // 序列化为字符串（用于 NIXL 的 getPublicData）
  std::string serialize() const;
  
  // 从字符串反序列化（用于 NIXL 的 loadRemoteMD）
  static TcpxMemoryDescriptor deserialize(const std::string& str);
};
```

## 📂 文件结构

```
p2p/tcpx/
├── include/
│   ├── tcpx_types.h            # ⭐ 新增：核心类型（PostedChunk, ChannelWindow, 常量）
│   ├── tcpx_logging.h          # ⭐ 新增：日志宏（LOG_DEBUG, LOG_ERROR, getEnvInt）
│   ├── tcpx_session.h          # ⭐ 新增：TcpxSession 类声明
│   ├── tcpx_transfer.h         # ⭐ 新增：TcpxTransfer 类声明
│   └── tcpx_memory_desc.h      # ⭐ 新增：TcpxMemoryDescriptor 结构
├── src/
│   ├── tcpx_helpers.cc         # ⭐ 新增：辅助函数（event 管理、drainCompletedKernels）
│   ├── tcpx_session.cc         # ⭐ 新增：TcpxSession 实现
│   ├── tcpx_transfer.cc        # ⭐ 新增：TcpxTransfer 实现
│   └── tcpx_memory_desc.cc     # ⭐ 新增：序列化/反序列化实现
├── tests/
│   ├── test_tcpx_perf_multi.cc # 现有测试（保持不变）
│   └── test_tcpx_api.cc        # ⭐ 新增：API 单元测试
├── Makefile                    # 更新：编译 libtcpx_p2p.a 和 libtcpx_p2p.so
└── libtcpx_p2p.so              # ⭐ 产出：共享库（NIXL 插件用）
```

**注**：现有文件（channel_manager.h/cc, bootstrap.h/cc, sliding_window.h/cc, device/）保持不变，不在此列出。

## 🔧 实施步骤

### 步骤 1：提取核心逻辑到可复用函数（2 小时）

**从 `test_tcpx_perf_multi.cc` 提取**：
- `process_completed_chunk()` → `TcpxTransfer::Impl::drainCompletedKernels()`
- `wait_for_channel_capacity()` → `TcpxTransfer::Impl::waitForCapacity()`
- SERVER 初始化逻辑 → `TcpxSession::Impl::setupServer()`
- CLIENT 连接逻辑 → `TcpxSession::connect()`

**注意**：这一步**不需要**完整执行 REFACTOR_ROADMAP，只需要：
1. 移动 `PostedChunk` 和 `ChannelWindow` 到文件顶部（必须，否则无法编译）
2. 提取上述 4 个函数（为了在 API 层复用）

### 步骤 2：实现 `TcpxSession` 类（6 小时）

**文件**：`p2p/tcpx/src/tcpx_session.cc`

**实现要点**：
```cpp
struct TcpxSession::Impl {
  int gpu_id_;
  int num_channels_;
  ChannelManager* mgr_ = nullptr;

  CUdevice cu_dev_;
  CUcontext cu_ctx_;
  cudaStream_t unpack_stream_ = nullptr;
  tcpx::device::UnpackLauncher* launcher_ = nullptr;

  std::map<std::string, std::vector<ncclNetHandle_v7>> remote_handles_;
  std::map<std::string, bool> remote_accepted_;
  std::map<uint64_t, MemoryHandle> registered_memory_;
  uint64_t next_mem_id_ = 0;

  // 析构函数：完整的 RAII 清理
  ~Impl() {
    if (launcher_) delete launcher_;
    if (unpack_stream_) cudaStreamDestroy(unpack_stream_);

    // 注销所有内存
    for (auto& [id, mem] : registered_memory_) {
      if (mem.mhandle && mgr_) {
        auto& ch = mgr_->get_channel(0);
        void* comm = mem.is_recv ? ch.recv_comm : ch.send_comm;
        tcpx_dereg_mr(comm, mem.mhandle);
      }
    }

    if (cu_ctx_) cuDevicePrimaryCtxRelease(cu_dev_);
    if (mgr_) {
      mgr_->close_all(true);
      mgr_->close_all(false);
      delete mgr_;
    }
  }
};

// Server 端：listen 并返回序列化的 handles
std::string TcpxSession::listen() {
  std::vector<ncclNetHandle_v7> handles;
  if (impl_->mgr_->server_listen_all(handles) != 0) {
    return "";
  }

  // 序列化 handles（使用 bootstrap.h 的逻辑）
  std::ostringstream oss;
  for (const auto& h : handles) {
    oss.write(reinterpret_cast<const char*>(&h), sizeof(h));
  }
  return oss.str();
}

// Server 端：accept 连接
int TcpxSession::accept(const std::string& remote_name) {
  if (impl_->mgr_->server_accept_all() != 0) {
    return -1;
  }
  impl_->remote_accepted_[remote_name] = true;
  return 0;
}

// Client 端：加载 server 的 handles
int TcpxSession::loadRemoteConnInfo(const std::string& remote_name,
                                     const std::string& conn_info) {
  std::vector<ncclNetHandle_v7> handles(impl_->num_channels_);
  std::istringstream iss(conn_info);
  for (auto& h : handles) {
    iss.read(reinterpret_cast<char*>(&h), sizeof(h));
  }
  impl_->remote_handles_[remote_name] = handles;
  return 0;
}

// Client 端：连接到 server
int TcpxSession::connect(const std::string& remote_name) {
  auto it = impl_->remote_handles_.find(remote_name);
  if (it == impl_->remote_handles_.end()) return -1;
  return impl_->mgr_->client_connect_all(it->second);
}

// 注册内存（支持多个独立的注册）
uint64_t TcpxSession::registerMemory(void* buffer, size_t size,
                                      int ptr_type, bool is_recv) {
  uint64_t mem_id = impl_->next_mem_id_++;

  // 在所有通道上注册
  void* mhandle = nullptr;
  auto& ch = impl_->mgr_->get_channel(0);
  void* comm = is_recv ? ch.recv_comm : ch.send_comm;

  if (tcpx_reg_mr(comm, buffer, size, ptr_type, &mhandle) != 0) {
    return 0;  // 失败
  }

  MemoryHandle mem;
  mem.buffer = buffer;
  mem.size = size;
  mem.ptr_type = ptr_type;
  mem.is_recv = is_recv;
  mem.mhandle = mhandle;
  mem.id = mem_id;

  impl_->registered_memory_[mem_id] = mem;
  return mem_id;
}
```

### 步骤 3：实现 `TcpxTransfer` 类（8 小时）

**文件**：`p2p/tcpx/src/tcpx_transfer.cc`

**关键依赖**（从 test_tcpx_perf_multi.cc 提取）：
- `PostedChunk` 和 `ChannelWindow` 结构体（已在步骤 1 移到 tcpx_types.h）
- `MAX_INFLIGHT_PER_CHANNEL` 常量
- CUDA event 生命周期管理
- 日志宏（LOG_DEBUG, LOG_ERROR）

**实现要点**：
```cpp
struct TcpxTransfer::Impl {
  TcpxSession::Impl* session_;
  std::string remote_name_;

  // 每个通道的滑动窗口状态
  std::vector<ChannelWindow> channel_windows_;

  // 所有已发起的请求
  std::vector<PostedChunk> all_posted_chunks_;
  int total_chunks_ = 0;
  int completed_chunks_ = 0;

  bool completed_ = false;
  int next_channel_ = 0;  // Round-robin 通道选择

  // 从 test_tcpx_perf_multi.cc 提取的逻辑
  bool drainCompletedKernels(int channel_id);
  bool waitForCapacity(int channel_id);

  // 构造函数：初始化 CUDA events
  Impl(TcpxSession::Impl* session, const std::string& remote_name)
      : session_(session), remote_name_(remote_name) {
    channel_windows_.resize(session->num_channels_);
    for (int ch = 0; ch < session->num_channels_; ++ch) {
      channel_windows_[ch].events.resize(MAX_INFLIGHT_PER_CHANNEL);
      for (int i = 0; i < MAX_INFLIGHT_PER_CHANNEL; ++i) {
        cudaEventCreate(&channel_windows_[ch].events[i]);
      }
    }
  }

  // 析构函数：销毁 CUDA events
  ~Impl() {
    for (auto& win : channel_windows_) {
      for (auto& evt : win.events) {
        cudaEventDestroy(evt);
      }
    }
  }
};

int TcpxTransfer::postRecv(uint64_t mem_id, size_t offset, size_t size, int tag) {
  // 1. 获取内存句柄
  auto* mem = impl_->session_->getMemoryHandle(mem_id);
  if (!mem || !mem->is_recv) return -1;

  // 2. 选择通道（round-robin）
  int ch_id = impl_->next_channel_;
  impl_->next_channel_ = (impl_->next_channel_ + 1) % impl_->session_->num_channels_;

  auto& ch = impl_->session_->mgr_->get_channel(ch_id);
  auto& win = impl_->channel_windows_[ch_id];

  // 3. 等待容量
  if (!impl_->waitForCapacity(ch_id)) return -1;

  // 4. 发起 irecv
  void* request = nullptr;
  void* dst_ptr = (char*)mem->buffer + offset;
  void* dst_ptrs[1] = {dst_ptr};
  int sizes[1] = {(int)size};
  int tags[1] = {tag};
  void* mhandles[1] = {mem->mhandle};

  if (tcpx_irecv(ch.recv_comm, 1, dst_ptrs, sizes, tags, mhandles, &request) != 0) {
    return -1;
  }

  // 5. 记录请求
  PostedChunk chunk;
  chunk.request = request;
  chunk.dst_ptr = dst_ptr;
  chunk.bytes = size;
  chunk.offset = offset;
  chunk.tag = tag;
  chunk.global_idx = impl_->total_chunks_++;

  win.inflight_recvs.push_back(chunk);
  impl_->all_posted_chunks_.push_back(chunk);

  return 0;
}

bool TcpxTransfer::isComplete() {
  // 轮询所有通道，drain 已完成的 kernels
  for (int ch = 0; ch < impl_->session_->num_channels_; ++ch) {
    impl_->drainCompletedKernels(ch);
  }

  // 检查是否所有 chunks 都已完成
  impl_->completed_ = (impl_->completed_chunks_ >= impl_->total_chunks_);
  return impl_->completed_;
}

int TcpxTransfer::release() {
  // 只对 recv 请求调用 tcpx_irecv_consumed
  for (int ch = 0; ch < impl_->session_->num_channels_; ++ch) {
    auto& ch_res = impl_->session_->mgr_->get_channel(ch);
    auto& win = impl_->channel_windows_[ch];

    // 消费所有 pending 的 recv 请求（⭐ 只处理 recv）
    for (auto* req : win.pending_recv_reqs) {
      tcpx_irecv_consumed(ch_res.recv_comm, 1, req);
    }
    win.pending_recv_reqs.clear();

    // send 请求不需要 consumed，只需清理
    win.pending_send_reqs.clear();
  }

  return 0;
}

// drainCompletedKernels 实现（从 test_tcpx_perf_multi.cc 提取）
bool TcpxTransfer::Impl::drainCompletedKernels(int channel_id) {
  auto& win = channel_windows_[channel_id];
  auto& ch = session_->mgr_->get_channel(channel_id);

  // 检查 pending recv kernels
  for (size_t i = 0; i < win.pending_recv_reqs.size(); ) {
    cudaError_t err = cudaEventQuery(win.events[win.pending_recv_indices[i]]);

    if (err == cudaSuccess) {
      // Kernel 完成，消费 recv（⭐ 在这里调用 consumed）
      tcpx_irecv_consumed(ch.recv_comm, 1, win.pending_recv_reqs[i]);

      // 移除
      win.pending_recv_reqs.erase(win.pending_recv_reqs.begin() + i);
      win.pending_recv_indices.erase(win.pending_recv_indices.begin() + i);

      completed_recv_chunks_++;  // ⭐ 更新 recv 计数
    } else if (err == cudaErrorNotReady) {
      i++;  // 继续等待
    } else {
      return false;  // 错误
    }
  }

  // 检查 pending send kernels（不需要 consumed）
  for (size_t i = 0; i < win.pending_send_reqs.size(); ) {
    cudaError_t err = cudaEventQuery(win.events[win.pending_send_indices[i]]);

    if (err == cudaSuccess) {
      // Send 完成，不需要 consumed
      win.pending_send_reqs.erase(win.pending_send_reqs.begin() + i);
      win.pending_send_indices.erase(win.pending_send_indices.begin() + i);

      completed_send_chunks_++;  // ⭐ 更新 send 计数
    } else if (err == cudaErrorNotReady) {
      i++;  // 继续等待
    } else {
      return false;  // 错误
    }
  }

  return true;
}
```

### 步骤 4：实现 `TcpxMemoryDescriptor`（1 小时）

**文件**：`p2p/tcpx/src/tcpx_memory_desc.cc`

```cpp
std::string TcpxMemoryDescriptor::serialize() const {
  std::ostringstream oss;
  oss << base_addr << "," << size << "," << ptr_type;
  return oss.str();
}

TcpxMemoryDescriptor TcpxMemoryDescriptor::deserialize(const std::string& str) {
  TcpxMemoryDescriptor desc;
  std::istringstream iss(str);
  char comma;
  iss >> desc.base_addr >> comma >> desc.size >> comma >> desc.ptr_type;
  return desc;
}
```

### 步骤 5：更新 Makefile 编译静态库和共享库（2 小时）

**注意**：NIXL 插件需要 `.so` 共享库，不是 `.a` 静态库

```makefile
# C++ 编译选项（添加 -fPIC）
CXXFLAGS += -fPIC

# CUDA 编译选项（添加 -Xcompiler -fPIC）⭐ 关键：device 代码也需要 -fPIC
NVCCFLAGS += -Xcompiler -fPIC

# 编译 device 对象（必须使用 -fPIC）⭐
device/unpack_kernels.o: device/unpack_kernels.cu
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC -c $< -o $@

device/unpack_launch.o: device/unpack_launch.cc
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

# 编译静态库（用于测试）
libtcpx_p2p.a: src/tcpx_session.o src/tcpx_transfer.o src/tcpx_memory_desc.o \
               src/tcpx_helpers.o src/channel_manager.o src/bootstrap.o \
               src/sliding_window.o device/unpack_launch.o device/unpack_kernels.o
	ar rcs $@ $^

# 编译共享库（用于 NIXL 插件）⭐
libtcpx_p2p.so: src/tcpx_session.o src/tcpx_transfer.o src/tcpx_memory_desc.o \
                src/tcpx_helpers.o src/channel_manager.o src/bootstrap.o \
                src/sliding_window.o device/unpack_launch.o device/unpack_kernels.o
	$(CXX) -shared -o $@ $^ $(LDFLAGS) -lcuda -lcudart

# 验证共享库（确保没有 relocation 错误）⭐
verify-so: libtcpx_p2p.so
	@echo "Checking for TEXTREL (should be empty)..."
	@readelf -d libtcpx_p2p.so | grep TEXTREL || echo "✓ No TEXTREL found (good)"

# 更新测试依赖
tests/test_tcpx_perf_multi: tests/test_tcpx_perf_multi.cc libtcpx_p2p.a
	$(CXX) $(CXXFLAGS) $< -o $@ -L. -ltcpx_p2p $(LDFLAGS)

tests/test_tcpx_api: tests/test_tcpx_api.cc libtcpx_p2p.a
	$(CXX) $(CXXFLAGS) $< -o $@ -L. -ltcpx_p2p $(LDFLAGS)
```

### 步骤 6：编写 API 单元测试（3 小时）

**文件**：`p2p/tcpx/tests/test_tcpx_api.cc`

**注意**：需要真实的两端握手，不能在同一进程内测试（TCPX 需要网络通信）

```cpp
// 测试场景：两个进程，通过 bootstrap socket 交换连接信息
int main(int argc, char** argv) {
  bool is_server = (argc > 1 && strcmp(argv[1], "server") == 0);

  if (is_server) {
    // ========== SERVER 端 ==========
    TcpxSession server(0, 2);

    // 1. Listen 并获取连接信息
    std::string conn_info = server.listen();

    // 2. 通过 bootstrap socket 发送给 client
    int bootstrap_fd = socket(AF_INET, SOCK_STREAM, 0);
    // ... bind, listen, accept ...
    send(bootstrap_fd, conn_info.data(), conn_info.size(), 0);

    // 3. 接收 client 的确认
    char ack[4];
    recv(bootstrap_fd, ack, 4, 0);

    // 4. Accept 连接
    server.accept("client");

    // 5. 注册内存
    void* recv_buf = nullptr;
    cudaMalloc(&recv_buf, 1024);
    uint64_t mem_id = server.registerMemory(recv_buf, 1024, NCCL_PTR_CUDA, true);

    // 6. 发起传输
    auto* xfer = server.createTransfer("client");
    xfer->postRecv(mem_id, 0, 1024, 0);

    // 7. 等待完成
    xfer->wait();

    // 8. 清理
    xfer->release();
    delete xfer;

    server.deregisterMemory(mem_id);
    cudaFree(recv_buf);
    close(bootstrap_fd);

  } else {
    // ========== CLIENT 端 ==========
    TcpxSession client(0, 2);

    // 1. 连接到 server 的 bootstrap socket
    int bootstrap_fd = socket(AF_INET, SOCK_STREAM, 0);
    // ... connect to server ...

    // 2. 接收 server 的连接信息
    char conn_info_buf[1024];
    int len = recv(bootstrap_fd, conn_info_buf, 1024, 0);
    std::string conn_info(conn_info_buf, len);

    // 3. 加载并连接
    client.loadRemoteConnInfo("server", conn_info);
    client.connect("server");

    // 4. 发送确认
    send(bootstrap_fd, "ACK", 4, 0);

    // 5. 注册内存
    void* send_buf = nullptr;
    cudaMalloc(&send_buf, 1024);
    uint64_t mem_id = client.registerMemory(send_buf, 1024, NCCL_PTR_CUDA, false);

    // 6. 发起传输
    auto* xfer = client.createTransfer("server");
    xfer->postSend(mem_id, 0, 1024, 0);

    // 7. 等待完成
    xfer->wait();

    // 8. 清理
    xfer->release();
    delete xfer;

    client.deregisterMemory(mem_id);
    cudaFree(send_buf);
    close(bootstrap_fd);
  }

  return 0;
}
```

### 步骤 7：重构 `test_tcpx_perf_multi.cc` 使用新 API（2 小时）

**目标**：验证新 API 的性能与原实现一致

```cpp
int main() {
  // ... 参数解析 ...

  if (is_server) {
    TcpxSession session(gpu_id, num_channels);
    std::string conn_info = session.listen();  // ⭐ 使用 listen()

    // Bootstrap handshake（发送 conn_info 给 client）
    // ...

    session.accept("client");  // ⭐ 接受连接

    // 注册内存（使用 mem_id）
    uint64_t recv_mem_id = session.registerMemory(recv_buf, test_size, NCCL_PTR_CUDA, true);

    for (int iter = 0; iter < iterations; ++iter) {
      auto* xfer = session.createTransfer("client");

      // Post all recvs（使用 mem_id + offset）⭐
      for (size_t offset = 0; offset < test_size; offset += chunk_bytes) {
        xfer->postRecv(recv_mem_id, offset, chunk_bytes, tag++);
      }

      xfer->wait();
      xfer->release();
    }
  } else {
    TcpxSession session(gpu_id, num_channels);

    // Bootstrap handshake（接收 server 的 conn_info）
    // ...

    session.loadRemoteConnInfo("server", conn_info);
    session.connect("server");  // ⭐ 连接到 server

    // 注册内存（使用 mem_id）
    uint64_t send_mem_id = session.registerMemory(send_buf, test_size, NCCL_PTR_CUDA, false);

    for (int iter = 0; iter < iterations; ++iter) {
      auto* xfer = session.createTransfer("server");

      // Post all sends（使用 mem_id + offset）⭐
      for (size_t offset = 0; offset < test_size; offset += chunk_bytes) {
        xfer->postSend(send_mem_id, offset, chunk_bytes, tag++);
      }

      xfer->wait();
      xfer->release();
    }
  }

  return 0;
}
```

## 📊 时间估算（修正后）

| 步骤 | 任务 | 时间 | 关键修复 |
|------|------|------|---------|
| 1 | 提取核心逻辑 | 3 小时 | 包含日志宏、常量、event 管理 |
| 2 | 实现 TcpxSession | 6 小时 | 完整握手流程、多内存注册 |
| 3 | 实现 TcpxTransfer | 8 小时 | drainCompletedKernels、irecv_consumed |
| 4 | 实现 TcpxMemoryDescriptor | 1 小时 | - |
| 5 | 更新 Makefile | 2 小时 | 静态库 + 共享库 |
| 6 | 编写 API 单元测试 | 3 小时 | 真实两端握手 |
| 7 | 重构 test_tcpx_perf_multi.cc | 2 小时 | - |
| **总计** | | **25 小时（3-4 天）** | |

## ✅ 验证标准

- [ ] `libtcpx_p2p.a` 编译成功
- [ ] `test_tcpx_api` 通过（loopback 测试）
- [ ] `test_tcpx_perf_multi` 使用新 API 后性能保持 ~9 GB/s
- [ ] API 头文件清晰，无底层细节泄漏

## 🚀 下一步：NIXL 插件实现

完成上述 API 层后，就可以开始实现 NIXL 插件了：

```
thirdparty/nixl/src/plugins/tcpx/
├── tcpx_backend.h          # 继承 nixlBackendEngine
├── tcpx_backend.cpp        # 实现所有虚函数
├── tcpx_plugin.cpp         # 插件入口
└── meson.build             # 编译配置
```

**关键映射**：
- `nixlTcpxEngine::connect()` → `TcpxSession::connect()`
- `nixlTcpxEngine::registerMem()` → `TcpxSession::registerMemory()`
- `nixlTcpxEngine::postXfer()` → `TcpxTransfer::postSend/postRecv()`
- `nixlTcpxEngine::checkXfer()` → `TcpxTransfer::isComplete()`

## 📝 与 REFACTOR_ROADMAP 的关系

**需要执行的部分**：
- ✅ 步骤 2（移动结构体定义）- **必须**，否则无法编译
- ✅ 步骤 3（提取 lambda 函数）- **必须**，为了在 API 层复用
- ❌ 步骤 1（日志控制）- **可选**，不影响功能
- ❌ 步骤 4-8（提取 setup/run 函数）- **可选**，只是美化测试代码

**结论**：只需执行 REFACTOR_ROADMAP 的**核心部分**（步骤 2-3），不需要完整执行。

