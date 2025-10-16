# TCPX NIXL 插件开发路线图

_从 p2p/tcpx 到 thirdparty/nixl/src/plugins/tcpx 的完整路径_

## 🎯 最终目标

构建一个 NIXL 插件 `thirdparty/nixl/src/plugins/tcpx`，使 NIXL 能够通过 TCPX 进行 GPU-to-GPU 通信。

## 📋 当前状态

✅ **已完成**：
- p2p/tcpx 基础设施（ChannelManager, Bootstrap, SlidingWindow）
- TCPX 底层封装（tcpx_interface.h）
- GPU kernel unpack（device/unpack_kernels.cu）
- 性能验证（~9 GB/s，test_tcpx_perf_multi.cc）

❌ **缺失**：
- 面向插件的 C++ API 层
- 内存元数据序列化/反序列化
- NIXL 插件骨架

## 🚀 三阶段路线图

### 阶段 1：构建插件 API 层（2-3 天）⭐ **当前优先级**

**目标**：创建 `libtcpx_p2p.a` 静态库，提供清晰的 C++ API

#### 任务 1.1：提取核心逻辑（4 小时）⚠️ 依赖复杂，需要完整迁移

**从 `test_tcpx_perf_multi.cc` 提取**：
1. 移动 `PostedChunk` 和 `ChannelWindow` 到文件顶部（**必须**，否则无法编译）
2. 提取 `process_completed_chunk()` 函数
3. 提取 `wait_for_channel_capacity()` 函数

**完整依赖清单**（必须一起迁移，否则 helper 无法编译）：

**1. 常量定义**（移到 `include/tcpx_types.h`）：
```cpp
constexpr int MAX_INFLIGHT_PER_CHANNEL = 16;  // 或从环境变量读取
constexpr int DEFAULT_NUM_CHANNELS = 2;
```

**2. 日志宏**（移到 `include/tcpx_logging.h`）：
```cpp
#define LOG_DEBUG(fmt, ...) if (getEnvInt("TCPX_DEBUG", 0)) fprintf(stderr, "[DEBUG] " fmt "\n", ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)
#define LOG_PERF(fmt, ...)  if (getEnvInt("TCPX_PERF", 0)) fprintf(stderr, "[PERF] " fmt "\n", ##__VA_ARGS__)

inline int getEnvInt(const char* name, int default_val) {
  const char* val = getenv(name);
  return val ? atoi(val) : default_val;
}
```

**3. CUDA event 管理**（移到 `src/tcpx_helpers.cc`）：
```cpp
// 初始化 events（在 TcpxTransfer 构造时调用）
void initChannelEvents(std::vector<ChannelWindow>& windows, int num_channels) {
  windows.resize(num_channels);
  for (int ch = 0; ch < num_channels; ++ch) {
    windows[ch].events.resize(MAX_INFLIGHT_PER_CHANNEL);
    for (int i = 0; i < MAX_INFLIGHT_PER_CHANNEL; ++i) {
      cudaEventCreate(&windows[ch].events[i]);
    }
  }
}

// 销毁 events（在 TcpxTransfer 析构时调用）
void destroyChannelEvents(std::vector<ChannelWindow>& windows) {
  for (auto& win : windows) {
    for (auto& evt : win.events) {
      cudaEventDestroy(evt);
    }
  }
}
```

**4. 滑动窗口逻辑**（移到 `src/tcpx_helpers.cc`）：
```cpp
// drainCompletedKernels 实现（从 process_completed_chunk 提取）
bool drainCompletedKernels(ChannelWindow& win, void* recv_comm, int& completed_chunks);

// waitForCapacity 实现（从 wait_for_channel_capacity 提取）
bool waitForCapacity(ChannelWindow& win, int timeout_ms = 1000);
```

**输出文件**：
- `p2p/tcpx/include/tcpx_types.h`（PostedChunk, ChannelWindow, 常量）
- `p2p/tcpx/include/tcpx_logging.h`（日志宏、getEnvInt）
- `p2p/tcpx/src/tcpx_helpers.cc`（initChannelEvents, destroyChannelEvents, drainCompletedKernels, waitForCapacity）

**验证**：
```bash
# 编译 helper 库
make src/tcpx_helpers.o

# 检查未定义符号（应该只有 CUDA/TCPX 外部符号）
nm src/tcpx_helpers.o | grep " U "
# 预期输出：cudaEventCreate, cudaEventQuery, cudaEventDestroy,
#           tcpx_test, tcpx_irecv_consumed 等外部符号
# 不应该有：process_completed_chunk, MAX_INFLIGHT_PER_CHANNEL 等内部符号
```

#### 任务 1.2：实现 `TcpxSession` 类（6 小时）⚠️ 握手流程修复

**功能**：
- 会话管理（**完整握手**：listen → accept / loadRemoteConnInfo → connect）
- **多内存注册**（支持独立的 send/recv 缓冲区，返回 mem_id）
- 连接信息序列化/反序列化

**文件**：
- `p2p/tcpx/include/tcpx_session.h`
- `p2p/tcpx/src/tcpx_session.cc`

**关键点**：
- PIMPL 模式隐藏 ChannelManager 等实现细节
- RAII 清理（注销所有内存、primary context、关闭所有通道）
- **Server 端**：`listen()` → `accept(remote_name)`
- **Client 端**：`loadRemoteConnInfo()` → `connect(remote_name)`
- **内存管理**：`std::map<uint64_t, MemoryHandle>` 跟踪多个注册

#### 任务 1.3：实现 `TcpxTransfer` 类（8 小时）⚠️ 状态管理修复

**功能**：
- 发起 send/recv 操作（使用 mem_id + offset）
- 轮询完成状态（drainCompletedKernels）
- **正确清理**（调用 tcpx_irecv_consumed）
- 批量传输支持

**文件**：
- `p2p/tcpx/include/tcpx_transfer.h`
- `p2p/tcpx/src/tcpx_transfer.cc`

**关键状态字段**（必须维护）：
```cpp
struct TcpxTransfer::Impl {
  TcpxSession::Impl* session_;
  std::string remote_name_;

  std::vector<ChannelWindow> channel_windows_;  // 每个通道的滑动窗口

  // 传输状态
  int total_send_chunks_ = 0;    // 总 send chunk 数
  int total_recv_chunks_ = 0;    // 总 recv chunk 数
  int completed_send_chunks_ = 0;  // 已完成 send chunk 数
  int completed_recv_chunks_ = 0;  // 已完成 recv chunk 数

  bool is_send_complete_ = false;
  bool is_recv_complete_ = false;

  int next_channel_ = 0;  // Round-robin 通道选择
};
```

**完成检查逻辑**：
```cpp
bool TcpxTransfer::isComplete() {
  // 轮询所有通道，drain 已完成的 kernels
  for (int ch = 0; ch < impl_->session_->num_channels_; ++ch) {
    drainCompletedKernels(impl_->channel_windows_[ch],
                          impl_->session_->mgr_->get_channel(ch).recv_comm,
                          impl_->completed_recv_chunks_);  // ⭐ 更新计数
  }

  // 检查是否所有 chunks 都已完成
  impl_->is_send_complete_ = (impl_->completed_send_chunks_ >= impl_->total_send_chunks_);
  impl_->is_recv_complete_ = (impl_->completed_recv_chunks_ >= impl_->total_recv_chunks_);

  return impl_->is_send_complete_ && impl_->is_recv_complete_;
}
```

**清理逻辑**（区分 send/recv）：
```cpp
int TcpxTransfer::release() {
  // 只对 recv 请求调用 tcpx_irecv_consumed
  for (int ch = 0; ch < impl_->session_->num_channels_; ++ch) {
    auto& ch_res = impl_->session_->mgr_->get_channel(ch);
    auto& win = impl_->channel_windows_[ch];

    // 消费所有 pending 的 recv 请求
    for (auto* req : win.pending_recv_reqs) {  // ⭐ 只处理 recv
      tcpx_irecv_consumed(ch_res.recv_comm, 1, req);
    }
    win.pending_recv_reqs.clear();

    // send 请求不需要 consumed，只需清理
    win.pending_send_reqs.clear();
  }

  return 0;
}
```

**关键点**：
- 复用任务 1.1 提取的 `drainCompletedKernels()` 逻辑
- 管理每个通道的滑动窗口状态（`std::vector<ChannelWindow>`）
- **CUDA events 生命周期**：构造时调用 `initChannelEvents()`，析构时调用 `destroyChannelEvents()`
- **完成检查**：`isComplete()` 调用 `drainCompletedKernels()` 并更新 `completed_recv_chunks_`
- **清理**：`release()` **只对 recv 请求**调用 `tcpx_irecv_consumed()`，send 请求不需要

#### 任务 1.4：实现内存元数据（1 小时）

**功能**：
- 序列化内存信息（地址、大小、类型）
- 反序列化

**文件**：
- `p2p/tcpx/include/tcpx_memory_desc.h`
- `p2p/tcpx/src/tcpx_memory_desc.cc`

#### 任务 1.5：更新构建系统（3 小时）⚠️ 需要共享库 + device 代码 -fPIC

**更新 Makefile**：
- 编译 `libtcpx_p2p.a` 静态库（用于测试）
- 编译 `libtcpx_p2p.so` 共享库（**用于 NIXL 插件**）⭐
- **关键**：所有对象文件（包括 device 代码）都需要 `-fPIC`
- 更新测试依赖

**详细编译命令**：
```makefile
# C++ 编译选项（添加 -fPIC）
CXXFLAGS += -fPIC

# CUDA 编译选项（添加 -Xcompiler -fPIC）
NVCCFLAGS += -Xcompiler -fPIC

# 编译 device 对象（必须使用 -fPIC）
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

# 更新测试依赖
tests/test_tcpx_perf_multi: tests/test_tcpx_perf_multi.cc libtcpx_p2p.a
	$(CXX) $(CXXFLAGS) $< -o $@ -L. -ltcpx_p2p $(LDFLAGS)

tests/test_tcpx_api: tests/test_tcpx_api.cc libtcpx_p2p.a
	$(CXX) $(CXXFLAGS) $< -o $@ -L. -ltcpx_p2p $(LDFLAGS)
```

**验证**：
```bash
# 编译共享库
make libtcpx_p2p.so

# 检查是否有 relocation 错误
readelf -d libtcpx_p2p.so | grep TEXTREL
# 应该没有输出（如果有 TEXTREL，说明某些对象文件缺少 -fPIC）

# 检查符号
nm -D libtcpx_p2p.so | grep TcpxSession
# 应该看到 TcpxSession 的符号
```

#### 任务 1.6：编写 API 测试（3 小时）⚠️ 需要真实两端

**文件**：
- `p2p/tcpx/tests/test_tcpx_api.cc`（**真实两端测试**，不是 loopback）

**测试流程**：
- Server: `listen()` → 通过 bootstrap socket 发送 conn_info → `accept()` → `postRecv()` → `wait()`
- Client: 通过 bootstrap socket 接收 conn_info → `loadRemoteConnInfo()` → `connect()` → `postSend()` → `wait()`

**验证**：
- API 可以独立于 NIXL 使用
- 完整握手流程正确
- 性能保持 ~9 GB/s

#### 任务 1.7：重构性能测试（2 小时）

**更新 `test_tcpx_perf_multi.cc`**：
- 使用新 API 重写
- 验证性能不变

**阶段 1 产出**：
- ✅ `libtcpx_p2p.a` 静态库（测试用）
- ✅ `libtcpx_p2p.so` 共享库（**NIXL 插件用**）⭐
- ✅ 清晰的 C++ API（TcpxSession, TcpxTransfer）
- ✅ 完整握手流程（listen/accept, loadRemoteConnInfo/connect）
- ✅ 多内存注册支持
- ✅ 正确的资源清理（区分 send/recv，只对 recv 调用 tcpx_irecv_consumed）
- ✅ 完整的依赖迁移（常量、日志宏、event 管理）
- ✅ device 代码 -fPIC 支持
- ✅ API 单元测试通过
- ✅ 性能保持 ~9 GB/s

**时间**：27 小时（3-4 天）

---

### 阶段 2：实现 NIXL 插件（3-4 天）

**目标**：创建 `thirdparty/nixl/src/plugins/tcpx` 插件

#### 任务 2.1：创建插件骨架（2 小时）

**参考**：`thirdparty/nixl/src/plugins/mooncake`

**文件**：
```
thirdparty/nixl/src/plugins/tcpx/
├── tcpx_backend.h          # 继承 nixlBackendEngine
├── tcpx_backend.cpp        # 实现虚函数
├── tcpx_plugin.cpp         # 插件入口
├── meson.build             # 编译配置
└── README.md               # 文档
```

#### 任务 2.2：实现生命周期方法（4 小时）⚠️ 分 Server/Client 角色

**Server 端流程**：
```cpp
std::string getConnInfo() {
  return session_->listen();  // 创建 listen comms，返回序列化的 handles
}

int connect(const std::string& remote_name) {
  return session_->accept(remote_name);  // ⭐ Server 调用 accept()
}
```

**Client 端流程**：
```cpp
int loadRemoteConnInfo(const std::string& remote_name, const std::string& conn_info) {
  return session_->loadRemoteConnInfo(remote_name, conn_info);
}

int connect(const std::string& remote_name) {
  return session_->connect(remote_name);  // ⭐ Client 调用 connect()
}
```

**关键点**：
- NIXL 的 `connect()` 回调**不会同时扮演 server 和 client**
- 需要根据插件实例的角色（通过构造参数或 `getConnInfo()` 是否被调用）决定调用 `accept()` 还是 `connect()`
- 建议在插件内部维护 `bool is_server_` 标志

#### 任务 2.3：实现资源管理方法（4 小时）

**方法**：
- `registerMem()` → `TcpxSession::registerMemory()`
- `deregisterMem()` → `TcpxSession::deregisterMemory()`
- `getPublicData()` → `TcpxMemoryDescriptor::serialize()`
- `loadLocalMD()` / `loadRemoteMD()` → `TcpxMemoryDescriptor::deserialize()`

#### 任务 2.4：实现传输方法（8 小时）

**方法**：
- `prepXfer()` → 创建 `TcpxTransfer` 对象
- `postXfer()` → `TcpxTransfer::postSend/postRecv()`
- `checkXfer()` → `TcpxTransfer::isComplete()`
- `releaseReqH()` → `TcpxTransfer::release()`

**关键点**：
- 处理 NIXL 的 `nixl_meta_dlist_t`（多个内存段）
- 映射到 TCPX 的批量传输

#### 任务 2.5：更新 Meson 构建（2 小时）

**更新**：
- `thirdparty/nixl/src/plugins/meson.build`（添加 tcpx 子目录）
- `thirdparty/nixl/src/plugins/tcpx/meson.build`（链接 libtcpx_p2p.a）

#### 任务 2.6：编写插件测试（4 小时）

**测试场景**：
- 两个 NIXL agent 通过 TCPX 插件通信
- 验证 registerMem → connect → postXfer → checkXfer 流程

**阶段 2 产出**：
- ✅ NIXL 插件编译通过
- ✅ 插件可以加载
- ✅ 基本传输功能正常

**时间**：24 小时（3-4 天）

---

### 阶段 3：集成测试和优化（2-3 天）

**目标**：在真实环境中验证插件

#### 任务 3.1：端到端测试（8 小时）

**场景**：
- 两个 GCE H100 节点
- 使用 NIXL 框架 + TCPX 插件
- 传输大文件（GB 级别）

**验证**：
- 性能达到 ~9 GB/s
- 无资源泄漏（cuda-memcheck）
- 稳定性（长时间运行）

#### 任务 3.2：性能调优（4 小时）

**优化点**：
- 调整通道数（2/4/8）
- 调整 chunk 大小
- 调整滑动窗口大小

#### 任务 3.3：文档和示例（4 小时）

**文档**：
- `thirdparty/nixl/src/plugins/tcpx/README.md`（使用说明）
- `thirdparty/nixl/src/plugins/tcpx/ARCHITECTURE.md`（架构说明）
- API 文档（Doxygen 注释）

**示例**：
- 简单的 send/recv 示例
- 多节点传输示例

**阶段 3 产出**：
- ✅ 插件在生产环境可用
- ✅ 性能达标
- ✅ 文档完善

**时间**：16 小时（2-3 天）

---

## 📊 总体时间估算（修正后）

| 阶段 | 任务 | 时间 | 关键修复 |
|------|------|------|---------|
| 阶段 1 | 构建插件 API 层 | 3-4 天（27h） | 完整依赖迁移、状态管理、device -fPIC |
| 阶段 2 | 实现 NIXL 插件 | 3-4 天（24h） | Server/Client 角色区分 |
| 阶段 3 | 集成测试和优化 | 2-3 天（16h） | - |
| **总计** | | **8-9 天（67h）** | |

## 🎯 里程碑

### 里程碑 1：API 层完成（第 3 天）
- [ ] `libtcpx_p2p.a` 编译成功
- [ ] `test_tcpx_api` 通过
- [ ] `test_tcpx_perf_multi` 使用新 API 后性能保持

### 里程碑 2：插件可用（第 7 天）
- [ ] NIXL 插件编译成功
- [ ] 插件可以加载
- [ ] 基本传输功能正常

### 里程碑 3：生产就绪（第 10 天）
- [ ] 端到端测试通过
- [ ] 性能达到 ~9 GB/s
- [ ] 文档完善

## ⚠️ 关键决策

### 是否需要执行 REFACTOR_ROADMAP？

**答案**：**部分执行**

**需要执行的部分**：
- ✅ 步骤 2：移动结构体定义（**必须**，否则无法编译）
- ✅ 步骤 3：提取 lambda 函数（**必须**，为了在 API 层复用）

**不需要执行的部分**：
- ❌ 步骤 1：日志控制（可选，不影响功能）
- ❌ 步骤 4-8：提取 setup/run 函数（可选，只是美化测试代码）

**理由**：
- 目标是"提供清晰的库接口给插件调用"，不是"写完美的测试代码"
- 只需要把数据面逻辑收敛到可复用的类/函数
- 等插件跑通再考虑剩余的美化重构

## 📚 相关文档

- `PLUGIN_API_DESIGN.md` - API 详细设计
- `REFACTOR_PLAN_FIXES.md` - 重构计划修复说明
- `thirdparty/nixl/src/plugins/mooncake/ARCHITECTURE.md` - Mooncake 插件参考

## 🚦 下一步行动

**立即开始**：阶段 1 任务 1.1（提取核心逻辑）

**命令**：
```bash
cd p2p/tcpx

# 1. 创建新文件
touch include/tcpx_types.h
touch src/tcpx_helpers.cc

# 2. 移动结构体定义（参考 REFACTOR_PLAN_FIXES.md）
# 编辑 tests/test_tcpx_perf_multi.cc

# 3. 验证编译
make clean && make
```

**预期时间**：2 小时

**完成标志**：
- [ ] `PostedChunk` 和 `ChannelWindow` 在 `include/tcpx_types.h`
- [ ] `process_completed_chunk()` 在 `src/tcpx_helpers.cc`
- [ ] `make` 编译成功
- [ ] `test_tcpx_perf_multi` 运行正常

