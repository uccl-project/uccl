#pragma once

#include "tcpx/device/unpack_launch.h"
#include "tcpx/include/bootstrap.h"
#include "tcpx/include/tcpx_interface.h"
#include "tcpx/include/unpack_descriptor.h"
#include <array>
#include <atomic>
#include <deque>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

extern thread_local bool inside_python;

namespace tcpx {

// ============================================================================
// 锁使用总结（Lock Usage Summary）
// ============================================================================
//
// tcpx_engine 使用 4 个锁来保护不同的资源，避免数据竞争：
//
// 【锁 1】conn_mu_ (std::shared_mutex)
//   保护资源：conn_map_ (连接映射)
//   锁类型：读写锁（允许多个读者，一个写者）
//   使用场景：
//     - 读锁：send_async/recv_async/poll_async 查找连接
//     - 写锁：accept/connect 创建连接，close 清理连接
//   持锁时间：短（只做查找操作）
//
// 【锁 2】mr_mu_ (std::mutex)
//   保护资源：mr_map_ (内存注册映射)
//   锁类型：普通互斥锁
//   使用场景：
//     - reg/dereg: 注册/注销内存
//     - populate_conn_handles_: 缓存 TCPX 句柄
//     - free_conn_: 清理连接的句柄缓存
//   持锁时间：短（避免在持锁时调用 TCPX API）
//
// 【锁 3】transfer_mu_ (std::mutex)
//   保护资源：transfer_map_ (传输映射)
//   锁类型：普通互斥锁
//   使用场景：
//     - post_send_/post_recv_: 创建传输
//     - poll_async: 用户线程查询传输状态并推进传输（高频）
//   持锁时间：可能较长（推进整个传输）
//   性能优化：用户线程 busy-wait polling，最大化吞吐量
//
// 【锁 4】window_mu_ (std::mutex)
//   保护资源：send_inflight_chunks_, recv_inflight_chunks_ (窗口计数器)
//   锁类型：普通互斥锁
//   使用场景：
//     - reserve_*_slot: 调度前检查窗口配额
//     - release_*_slot: chunk 完成后释放配额
//   持锁时间：非常短（只做计数器操作）
//
// ============================================================================
// 锁顺序（避免死锁）
// ============================================================================
//
// 1. transfer_mu_ -> conn_mu_
//    - poll_async: 先锁 transfer_mu_，再锁 conn_mu_（嵌套）
//
// 2. 独立的锁（不会同时持有）
//    - window_mu_: 独立于其他锁
//    - mr_mu_: 独立于其他锁
//
// 3. 避免死锁的设计
//    - poll_async: 先释放 transfer_mu_，再调用 reset_conn_window_counters_（持 window_mu_）
//    - free_conn_: 先收集句柄（持 mr_mu_），释放锁后再调用 tcpx_dereg_mr
//
// ============================================================================
// 性能优化策略
// ============================================================================
//
// 1. 使用读写锁（conn_mu_）
//    - 允许多个线程并发读取 conn_map_
//    - 减少锁竞争，提高多线程性能
//
// 2. 减少持锁时间
//    - 嵌套作用域：查找完成后立即释放锁
//    - 先收集数据，释放锁后再调用可能阻塞的函数
//
// 3. 用户线程主动推进（poll_async）
//    - 用户线程 busy-wait polling，最快响应
//    - 无后台线程，避免上下文切换开销
//
// ============================================================================

// ============================================================================
// 连接状态结构 (Connection State)
// ============================================================================
// Conn 结构体封装了单个 TCPX 连接的所有状态，包括：
// - TCPX 通信句柄（send_comm/recv_comm）
// - CUDA 事件池（用于接收端 GPU unpack 操作的同步）
//
// 设计要点：
// 1. CUDA 事件池预分配并循环复用，避免热路径上的事件创建/销毁开销
// 2. 使用 enable_shared_from_this 支持安全的异步访问
class Endpoint;

// Note: shared_from_this is not needed; connections are always managed via
// std::shared_ptr in conn_map_.
struct Conn {
  Conn() {
    recv_dev_handle = recv_dev_handle_storage.data();
    send_dev_handle = send_dev_handle_storage.data();
    std::memset(recv_dev_handle_storage.data(), 0,
                recv_dev_handle_storage.size());
    std::memset(send_dev_handle_storage.data(), 0,
                send_dev_handle_storage.size());
  }

  // -------------------------------------------------------------------------
  // 连接基本信息
  // -------------------------------------------------------------------------
  uint64_t conn_id = 0;           // 连接唯一标识符
  std::string ip_addr;            // 远端 IP 地址
  int remote_gpu_idx = -1;        // 远端 GPU 索引
  int remote_port = -1;           // 远端控制端口
  int ctrl_sock_fd = -1;          // TCP 控制套接字文件描述符

  // -------------------------------------------------------------------------
  // CUDA 事件池 (Event Pool) - 用于接收端 GPU unpack 操作的同步
  // -------------------------------------------------------------------------
  // 设计理念：
  // 1. 预分配固定数量的 CUDA events，整个连接生命周期内循环复用
  // 2. 池大小 = UCCL_TCPX_MAX_RECV_INFLIGHT（默认 16）
  // 3. 使用 event_counter 进行轮询分配：event_idx = counter % recv_events.size()
  // 4. 避免热路径上的 cudaEventCreate/Destroy 开销
  //
  // 对齐参考实现：
  // - 原 TcpxSession 的 ChannelWindow::events 设计
  // - 在 connect/accept 时根据 UCCL_TCPX_MAX_RECV_INFLIGHT 初始化
  // - 在 free_conn_ 时统一销毁
  std::vector<cudaEvent_t> recv_events;  // 事件池数组
  uint64_t event_counter = 0;            // 事件分配计数器（用于轮询）

  // -------------------------------------------------------------------------
  // TCPX 通信句柄
  // -------------------------------------------------------------------------
  void* send_comm = nullptr;      // TCPX 发送通信句柄
  void* recv_comm = nullptr;      // TCPX 接收通信句柄
  // Note: send_dev_handle is not referenced by the engine today but is kept
  // to satisfy the tcpx_connect_v5 API contract (device-handle lifetime).
  void* send_dev_handle = nullptr;// 发送端设备句柄（指向 send_dev_handle_storage）
  void* recv_dev_handle = nullptr;// 接收端设备句柄（指向 recv_dev_handle_storage）

  // Host-side cached copy of the recv device handle to avoid per-chunk
  // cuMemcpyDtoH when launching unpack.
  tcpx::plugin::unpackNetDeviceHandle recv_dev_handle_host{};
  bool recv_dev_handle_cached = false;

  // 设备句柄存储（对齐要求：16 字节）
  alignas(16) std::array<uint8_t, 512> recv_dev_handle_storage;
  alignas(16) std::array<uint8_t, 512> send_dev_handle_storage;
};

// ============================================================================
// FIFO 元数据项 (FIFO Metadata Item)
// ============================================================================
// FifoItem 用于 READ 操作的元数据交换：
// 1. 被动端（服务器）调用 advertise() 生成 FifoItem
// 2. 通过 UCCL 控制套接字发送给主动端（客户端）
// 3. 主动端调用 queue_read_response() 解析 FifoItem 并发起传输
//
// 布局要求：固定 64 字节，与 UCCL listener 期望的格式对齐
struct FifoItem {
  uint64_t mr_id;    // 注册内存区域的唯一标识符
  uint32_t size;     // 需要传输的数据大小（字节）
  uint32_t tag;      // TCPX 标签（用于匹配 isend/irecv 操作）
  uint64_t offset;   // 在注册内存区域内的字节偏移量
  uint64_t token;    // 保留字段（用于未来扩展，当前保持对齐）
  char padding[32];  // 填充至 64 字节
};
static_assert(sizeof(struct FifoItem) == 64, "FifoItem size is not 64 bytes");

// ============================================================================
// 内存注册条目 (Memory Registration Entry)
// ============================================================================
// MrEntry 缓存了单个注册内存区域的信息及其在各连接上的 TCPX 句柄：
// 1. 每个 mr_id 对应一个 MrEntry
// 2. 每个连接首次使用该 MR 时，调用 tcpx_reg_mr 并缓存句柄
// 3. 后续使用直接复用缓存的句柄，避免重复注册
//
// 设计要点：
// - send_handles 和 recv_handles 分别缓存发送和接收方向的句柄
// - 在 dereg() 时遍历所有连接，调用 tcpx_dereg_mr 释放句柄
struct MrEntry {
  void* base = nullptr;           // 内存区域基地址
  size_t size = 0;                // 内存区域大小（字节）
  int ptr_type = NCCL_PTR_CUDA;   // 指针类型（CUDA 设备内存）

  // 每连接的 TCPX 注册句柄缓存
  // - 键：conn_id（连接唯一标识符）
  // - 值：mhandle（TCPX 内存句柄）
  std::unordered_map<uint64_t, void*> send_handles;  // 发送方向句柄
  std::unordered_map<uint64_t, void*> recv_handles;  // 接收方向句柄
};

// ============================================================================
// 待处理传输 (Pending Transfer)
// ============================================================================
// PendingTransfer 表示一个正在进行的传输操作，支持分块传输和流水线处理：
//
// 分块策略：
// - 大传输被切分为多个 chunk（大小由 UCCL_TCPX_CHUNK_BYTES 控制，默认 512KB）
// - 每个 chunk 独立调用 tcpx_isend/tcpx_irecv
// - 每个 chunk 有独立的 tag（base_tag + chunk_idx）
//
// 流水线处理（接收端）：
// - Stage 0：调度阶段 - schedule_recv_chunks_locked 提交 tcpx_irecv
// - Stage 1：网络完成 - poll_chunk_request_ 检查 tcpx_test，数据到达 bounce buffer
// - Stage 2：GPU 完成 - launch_chunk_unpack_ 启动 CUDA kernel，cudaEventQuery 等待完成
//
// 流水线处理（发送端）：
// - Stage 0：调度阶段 - schedule_send_chunks_locked 提交 tcpx_isend
// - Stage 1+2：网络完成 - poll_chunk_request_ 检查 tcpx_test，发送完成
//
// 滑动窗口控制：
// - 发送端：max_send_inflight_ 限制同时在途的发送 chunks
// - 接收端：max_recv_inflight_ 限制同时在途的接收 chunks
// - 通过 reserve_*_slot / release_*_slot 管理窗口配额
struct PendingTransfer {
  // --------------------------------------------------------------------------
  // 单个 Chunk 的状态
  // --------------------------------------------------------------------------
  struct ChunkState {
    size_t offset = 0;          // 在整个传输中的字节偏移量
    size_t bytes = 0;           // 该 chunk 的大小（字节）
    uint32_t tag = 0;           // TCPX 标签（base_tag + chunk_idx）
    void* request = nullptr;    // TCPX 请求句柄（tcpx_isend/irecv 返回）
    void* dst_ptr = nullptr;    // 目标地址（发送端：源地址；接收端：目标地址）
    bool needs_unpack = false;  // 是否需要 GPU unpack（接收端 bounce buffer -> 目标内存）
    bool stage1_done = false;   // Stage 1 是否完成（网络传输完成）
    bool stage2_done = false;   // Stage 2 是否完成（GPU unpack 完成或发送确认）
    bool posted = false;        // 是否已提交 TCPX 请求

    // Bounce-buffer 元数据（接收端使用）
    // - 在 Stage 1 完成后，从 TCPX request 中提取
    // - 用于启动 GPU unpack kernel
    rx::UnpackDescriptorBlock desc_block{};

    // CUDA 事件（接收端使用）
    // - 从 Conn::recv_events pool 中获取（不拥有，只是引用）
    // - 用于同步 GPU unpack kernel 的完成
    cudaEvent_t event = nullptr;

    // 事件在 pool 中的索引（用于调试）
    size_t event_idx = 0;
  };

  // --------------------------------------------------------------------------
  // 传输类型
  // --------------------------------------------------------------------------
  enum class Kind {
    kSend,  // 发送操作（send_async）
    kRecv,  // 接收操作（recv_async，需要 GPU unpack）
    kRead   // 读取操作（read_async，不需要 GPU unpack）
  };

  // --------------------------------------------------------------------------
  // 传输元数据
  // --------------------------------------------------------------------------
  Kind kind = Kind::kRecv;        // 传输类型
  uint64_t transfer_id = 0;       // 传输唯一标识符
  uint64_t conn_id = 0;           // 所属连接的 ID
  uint64_t mr_id = 0;             // 内存注册区域的 ID
  size_t total_bytes = 0;         // 总传输大小（字节）
  uint32_t base_tag = 0;          // 基础 TCPX 标签（每个 chunk 的 tag = base_tag + idx）
  size_t next_chunk_to_post = 0;  // 下一个待提交的 chunk 索引
  void* mhandle = nullptr;        // TCPX 内存句柄（从 MrEntry 缓存中获取）

  // --------------------------------------------------------------------------
  // Chunk 向量和进度跟踪
  // --------------------------------------------------------------------------
  // chunks 向量包含所有 chunk 的状态，每个 chunk 按流水线阶段推进：
  // - 发送端：send_queue（等待 tcpx_test 完成）
  // - 接收端：recv_stage1_queue（等待网络完成）-> recv_stage2_queue（等待 GPU 完成）
  std::vector<ChunkState> chunks;     // 所有 chunk 的状态数组
  size_t chunks_completed = 0;        // 已完成的 chunk 数量

  // --------------------------------------------------------------------------
  // 流水线队列
  // --------------------------------------------------------------------------
  // 发送端队列：
  std::deque<size_t> send_queue;          // 已提交但未完成的发送 chunk 索引

  // 接收端队列：
  std::deque<size_t> recv_stage1_queue;   // Stage 1：等待网络传输完成的 chunk 索引
  std::deque<size_t> recv_stage2_queue;   // Stage 2：等待 GPU unpack 完成的 chunk 索引
};

// ============================================================================
// TCPX 传输引擎 (TCPX Transport Engine)
// ============================================================================
// Endpoint 是 TCPX 传输引擎的核心类，提供以下功能：
//
// 1. 连接管理：
//    - connect/accept：建立 TCPX 连接（包括 TCP 控制握手和 TCPX 通道建立）
//
// 2. 内存注册：
//    - reg/dereg：注册/注销 CUDA 设备内存
//    - 每个 MR 在每个连接上按需注册，句柄缓存在 MrEntry 中
//
// 3. 异步传输：
//    - send_async/recv_async：发送/接收数据
//    - read_async：READ 操作（被动端通过 advertise 发布，主动端通过 queue_read_response 拉取）
//    - 所有传输自动分块并通过流水线处理
//
// 4. 进度推进：
//    - poll_async：轮询单个传输的完成状态（用户主动调用）
//    - progress_conn：推进某个连接上的所有传输
//
// 5. 设计模型：
//    - 用户主动推进：所有传输通过 poll_async 显式推进
//    - 无后台线程：避免线程开销和锁竞争
//    - 优势：简单高效，用户完全控制推进时机
class Endpoint {
 public:
  // ==========================================================================
  // 构造和析构
  // ==========================================================================

  /**
   * 创建 TCPX 传输引擎实例
   *
   * 初始化内容：
   * - 选择最佳 TCPX 设备（通过 find_best_dev 评分）
   * - 绑定 TCP 控制端口（支持端口重试，由 UCCL_TCPX_PORT_RETRIES 控制）
   * - 创建 TCPX listen comm（用于接受连接）
   * - 初始化 CUDA 上下文和 unpack stream
   * - 读取环境变量配置（chunk_bytes, max_send_inflight, max_recv_inflight, debug）
   *
   * @param local_gpu_idx 本地 GPU 索引（可通过 UCCL_TCPX_LOCAL_DEVICE 覆盖）
   * @param num_cpus CPU 数量（当前未使用）
   */
  explicit Endpoint(uint32_t const num_cpus);

  /**
   * 销毁 TCPX 传输引擎实例
   *
   * 清理顺序：
   * 1. 释放所有 TCPX 连接和内存注册
   * 2. 关闭 TCPX listen comm 和 TCP 控制套接字
   * 3. 销毁 CUDA stream 和释放 CUDA 上下文
   */
  ~Endpoint();

  // ==========================================================================
  // 连接管理
  // ==========================================================================

  /**
   * 主动连接到远端服务器
   *
   * 握手流程：
   * 1. 创建 TCP 控制套接字并连接到远端
   * 2. 交换 EndpointInfo（IP、端口、GPU 索引）
   * 3. 接收服务器的 TCPX listen handle
   * 4. 调用 tcpx_connect_v5 建立发送通道
   * 5. 创建本地 reverse listen handle 并发送给服务器
   * 6. 调用 tcpx_accept_v5 建立接收通道
   * 7. 初始化 CUDA 事件池
   *
   * @param ip_addr 远端 IP 地址
   * @param remote_gpu_idx 远端 GPU 索引
   * @param remote_port 远端控制端口（-1 表示使用默认端口）
   * @param conn_id 输出：连接 ID
   * @return 成功返回 true，失败返回 false
   */
  bool connect(std::string ip_addr, int remote_gpu_idx, int remote_port,
               uint64_t& conn_id);

  /**
   * 被动接受远端连接
   *
   * 握手流程（与 connect 对称）：
   * 1. 在 TCP 控制套接字上 accept 新连接
   * 2. 交换 EndpointInfo
   * 3. 发送本地 TCPX listen handle
   * 4. 调用 tcpx_accept_v5 建立接收通道
   * 5. 接收客户端的 reverse listen handle
   * 6. 调用 tcpx_connect_v5 建立发送通道
   * 7. 初始化 CUDA 事件池
   *
   * @param ip_addr 输出：远端 IP 地址
   * @param remote_gpu_idx 输出：远端 GPU 索引
   * @param conn_id 输出：连接 ID
   * @return 成功返回 true，失败返回 false
   */
  bool accept(std::string& ip_addr, int& remote_gpu_idx, uint64_t& conn_id);

  /**
   * 获取本地 endpoint 元数据（统一接口，与 RDMA 对齐）
   *
   * 元数据格式：
   * - IPv4：10 字节（4 字节 IP + 2 字节端口 + 4 字节 GPU 索引）
   * - IPv6：22 字节（16 字节 IP + 2 字节端口 + 4 字节 GPU 索引）
   * - 其他：sizeof(EndpointInfo)
   *
   * @return 元数据字节数组
   */
  std::vector<uint8_t> get_unified_metadata();

  /**
   * 解析 endpoint 元数据
   *
   * @param metadata 元数据字节数组
   * @return (IP 地址, 端口, GPU 索引) 元组
   */
  static std::tuple<std::string, uint16_t, int> parse_metadata(
      std::vector<uint8_t> const& metadata);

  // ==========================================================================
  // READ 操作支持（FIFO 元数据交换）
  // ==========================================================================

  /**
   * 发布内存区域供远端读取（被动端）
   *
   * 生成 FifoItem 并写入 out_buf，包含：
   * - mr_id：内存注册 ID
   * - size：数据大小
   * - tag：分配的 TCPX 标签
   * - offset：在 MR 内的偏移量
   *
   * @param conn_id 连接 ID
   * @param mr_id 内存注册 ID
   * @param addr 数据地址
   * @param len 数据长度
   * @param out_buf 输出缓冲区（至少 64 字节）
   * @return 成功返回 true，失败返回 false
   */
  bool advertise(uint64_t conn_id, uint64_t mr_id, void* addr, size_t len,
                 char* out_buf);

  /**
   * 响应 READ 请求（主动端）
   *
   * 解析 FifoItem，定位数据并调用 post_send_ 发送：
   * - 使用 FifoItem 中的 tag 确保匹配
   * - 传输完全异步，通过 poll_async 推进
   *
   * @param conn_id 连接 ID
   * @param fifo_item FIFO 元数据项
   * @return 成功返回 true，失败返回 false
   */
  bool queue_read_response(uint64_t conn_id, FifoItem const& fifo_item);

  /**
   * 分配新的 TCPX 标签（原子递增）
   *
   * @return 新的标签值
   */
  uint32_t allocate_tag() { return next_tag_.fetch_add(1); }

  /**
   * 获取连接的控制套接字文件描述符
   *
   * @param conn_id 连接 ID
   * @return 文件描述符，失败返回 -1
   */
  int get_sock_fd(uint64_t conn_id) const {
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    auto it = conn_map_.find(conn_id);
    if (it == conn_map_.end()) return -1;
    return it->second->ctrl_sock_fd;
  }

  // ==========================================================================
  // 内存注册
  // ==========================================================================

  /**
   * 注册 CUDA 设备内存
   *
   * 创建 MrEntry 并分配唯一的 mr_id：
   * - 实际的 TCPX 注册延迟到首次使用时（populate_conn_handles_）
   * - 句柄缓存在 MrEntry 的 send_handles/recv_handles 中
   *
   * @param data 内存地址
   * @param size 内存大小
   * @param mr_id 输出：内存注册 ID
   * @return 成功返回 true，失败返回 false
   */
  bool reg(void const* data, size_t size, uint64_t& mr_id);

  /**
   * 注销内存注册
   *
   * 遍历所有连接，调用 tcpx_dereg_mr 释放缓存的句柄：
   * - 先收集所有句柄（避免持锁时间过长）
   * - 再逐个调用 tcpx_dereg_mr
   *
   * @param mr_id 内存注册 ID
   * @return 成功返回 true，失败返回 false
   */
  bool dereg(uint64_t mr_id);

  /**
   * 根据地址查找内存注册 ID
   *
   * 遍历 mr_map_，检查 [addr, addr+size) 是否在某个 MR 范围内
   *
   * @param addr 地址
   * @param size 大小
   * @param mr_id 输出：内存注册 ID
   * @return 找到返回 true，否则返回 false
   */
  bool find_mr_by_addr(uintptr_t addr, size_t size, uint64_t* mr_id) const;

  // ==========================================================================
  // 异步传输 API
  // ==========================================================================

  /**
   * 异步读取数据（主动端）
   *
   * 根据 FifoItem 中的 tag 发起接收：
   * - 调用 recv_async_with_tag 使用指定的 tag
   * - 传输自动分块并通过 poll_async 推进
   *
   * @param conn_id 连接 ID
   * @param mr_id 内存注册 ID
   * @param dst 目标地址
   * @param size 数据大小
   * @param slot_item FIFO 元数据项
   * @param transfer_id 输出：传输 ID
   * @return 成功返回 true，失败返回 false
   */
  bool read_async(uint64_t conn_id, uint64_t mr_id, void* dst, size_t size,
                  FifoItem const& slot_item, uint64_t* transfer_id);

  /**
   * 异步发送数据
   *
   * 自动分配 tag 并调用 send_async_with_tag
   *
   * @param conn_id 连接 ID
   * @param mr_id 内存注册 ID
   * @param data 源数据地址
   * @param size 数据大小
   * @param transfer_id 输出：传输 ID
   * @return 成功返回 true，失败返回 false
   */
  bool send_async(uint64_t conn_id, uint64_t mr_id, void const* data,
                  size_t size, uint64_t* transfer_id);

  /**
   * 异步发送数据（指定 tag）
   *
   * 流程：
   * 1. 调用 post_send_ 创建 PendingTransfer 并分块
   * 2. 调用 schedule_send_chunks_locked 提交初始 chunks
   * 3. 用户通过 poll_async 推进传输
   *
   * @param conn_id 连接 ID
   * @param mr_id 内存注册 ID
   * @param data 源数据地址
   * @param size 数据大小
   * @param tag TCPX 标签
   * @param transfer_id 输出：传输 ID
   * @return 成功返回 true，失败返回 false
   */
  bool send_async_with_tag(uint64_t conn_id, uint64_t mr_id, void const* data,
                           size_t size, uint32_t tag, uint64_t* transfer_id);

  /**
   * 异步接收数据
   *
   * 自动分配 tag 并调用 recv_async_with_tag
   *
   * @param conn_id 连接 ID
   * @param mr_id 内存注册 ID
   * @param data 目标地址
   * @param size 数据大小
   * @param transfer_id 输出：传输 ID
   * @return 成功返回 true，失败返回 false
   */
  bool recv_async(uint64_t conn_id, uint64_t mr_id, void* data, size_t size,
                  uint64_t* transfer_id);

  /**
   * 异步接收数据（指定 tag）
   *
   * 流程：
   * 1. 调用 post_recv_ 创建 PendingTransfer 并分块
   * 2. 调用 schedule_recv_chunks_locked 提交初始 chunks
   * 3. 用户通过 poll_async 推进传输
   *
   * @param conn_id 连接 ID
   * @param mr_id 内存注册 ID
   * @param data 目标地址
   * @param size 数据大小
   * @param tag TCPX 标签
   * @param transfer_id 输出：传输 ID
   * @return 成功返回 true，失败返回 false
   */
  bool recv_async_with_tag(uint64_t conn_id, uint64_t mr_id, void* data,
                           size_t size, uint32_t tag, uint64_t* transfer_id);

  // ==========================================================================
  // 进度推进
  // ==========================================================================

  /**
   * 轮询传输完成状态
   *
   * 流程：
   * 1. 查找 transfer_map_ 中的传输
   * 2. 如果不存在，说明已完成
   * 3. 如果存在，调用 advance_transfer_locked 推进
   * 4. 如果完成，调用 finalize_transfer_locked 清理
   *
   * @param transfer_id 传输 ID
   * @param is_done 输出：是否完成
   * @return 成功返回 true，失败返回 false
   */
  bool poll_async(uint64_t transfer_id, bool* is_done);

  /**
   * 推进某个连接上的所有传输
   *
   * 遍历 transfer_map_，对所有属于该连接的传输调用 advance_transfer_locked
   *
   * @param conn_id 连接 ID
   * @return 是否有进度
   */
  bool progress_conn(uint64_t conn_id);

  // chunk_bytes() getter 移除：当前无外部调用者，如需可再添加

 private:
  // ==========================================================================
  // TCPX 和 CUDA 资源
  // ==========================================================================
  int dev_id_ = -1;                       // TCPX 设备 ID（通过 find_best_dev 选择）
  int ctrl_listen_fd_ = -1;               // TCP 控制套接字（用于 accept 新连接）
  void* listen_comms_ = nullptr;          // TCPX listen comm（用于接受 TCPX 连接）
  uint32_t local_gpu_idx_ = 0;            // 本地 GPU 索引
  int ctrl_port_ = 0;                     // TCP 控制端口（绑定成功后的实际端口）
  ncclNetHandle_v7 listen_handle_{};      // TCPX listen handle（用于 connect 端）

  // ==========================================================================
  // ID 生成器（原子递增）
  // ==========================================================================
  std::atomic<uint64_t> next_conn_id_ = 0;      // 连接 ID 生成器
  std::atomic<uint64_t> next_mr_id_{1};         // 内存注册 ID 生成器（从 1 开始）
  std::atomic<uint64_t> next_transfer_id_{1};   // 传输 ID 生成器（从 1 开始）
  std::atomic<uint32_t> next_tag_{1};           // TCPX 标签生成器（从 1 开始）

  // ==========================================================================
  // 连接管理
  // ==========================================================================
  // 【锁 1】conn_mu_: 连接映射的读写锁（shared_mutex）
  //
  // 保护资源：
  // - conn_map_: 连接 ID 到连接对象的映射
  //
  // 锁类型：std::shared_mutex（读写锁）
  // - 允许多个读者同时访问（std::shared_lock）
  // - 只允许一个写者独占访问（std::unique_lock）
  //
  // 使用场景：
  // - 读锁（shared_lock）：
  //   * send_async/recv_async/poll_async: 查找连接对象
  //   * get_sock_fd: 获取套接字文件描述符
  //   * 高频操作，需要并发读取
  // - 写锁（unique_lock）：
  //   * accept/connect: 创建新连接并加入 conn_map_
  //   * close: 析构时清理所有连接
  //   * 低频操作，需要独占访问
  //
  // 性能优化：
  // - 使用读写锁而非普通互斥锁，允许并发读取
  // - 减少锁竞争，提高多线程性能
  //
  // 锁顺序（避免死锁）：
  // - 如果需要同时持有 conn_mu_ 和 transfer_mu_，先获取 transfer_mu_，再获取 conn_mu_
  // - 例如：poll_async 先锁 transfer_mu_，再锁 conn_mu_
  //
  mutable std::shared_mutex conn_mu_;
  std::unordered_map<uint64_t, std::shared_ptr<Conn>> conn_map_;

  // ==========================================================================
  // 内存注册管理
  // ==========================================================================
  // 【锁 2】mr_mu_: 内存注册映射的互斥锁（mutex）
  //
  // 保护资源：
  // - mr_map_: 内存注册 ID 到 MrEntry 的映射
  // - MrEntry 包含：
  //   * base/size: 注册的内存区域
  //   * send_handles/recv_handles: 每个连接的 TCPX 句柄缓存
  //
  // 锁类型：std::mutex（普通互斥锁）
  // - 不使用读写锁，因为写操作（缓存句柄）也很频繁
  //
  // 使用场景：
  // - reg: 注册新的内存区域
  // - dereg: 注销内存区域并清理所有连接的句柄
  // - populate_conn_handles_: 为连接缓存 TCPX 句柄（写操作）
  // - find_mr_by_addr: 根据地址查找 MR（读操作）
  // - free_conn_: 清理连接的所有句柄缓存
  //
  // 持锁时间：
  // - 尽量短：只在访问 mr_map_ 时持锁
  // - 避免在持锁时调用 TCPX API（可能阻塞）
  // - 例如：dereg 先收集句柄，释放锁后再调用 tcpx_dereg_mr
  //
  // 锁顺序（避免死锁）：
  // - mr_mu_ 可以与其他锁同时持有，但要注意顺序
  // - 通常先获取 mr_mu_，再获取 conn_mu_（如果需要）
  //
  mutable std::mutex mr_mu_;
  std::unordered_map<uint64_t, MrEntry> mr_map_;

  // ==========================================================================
  // 传输管理
  // ==========================================================================
  // 【锁 3】transfer_mu_: 传输映射的互斥锁（mutex）
  //
  // 保护资源：
  // - transfer_map_: 传输 ID 到 PendingTransfer 的映射
  // - PendingTransfer 包含：
  //   * chunks: chunk 状态数组
  //   * send_queue/recv_stage1_queue/recv_stage2_queue: 流水线队列
  //   * chunks_completed: 已完成的 chunks 数量
  //
  // 锁类型：std::mutex（普通互斥锁）
  // - 不使用读写锁，因为写操作（推进传输）非常频繁
  //
  // 使用场景：
  // - post_send_/post_recv_: 创建新传输并加入 transfer_map_
  // - poll_async: 用户线程查询传输状态并推进传输（高频操作）
  // - finalize_transfer_locked: 完成传输并从 transfer_map_ 移除
  //
  // 持锁时间：
  // - 可能较长：advance_transfer_locked 需要持锁推进整个传输
  // - 包括调度 chunks、轮询 TCPX、启动 GPU kernels
  // - 但这是必要的，因为需要保证传输状态的一致性
  //
  // 锁顺序（避免死锁）：
  // - 如果需要同时持有 transfer_mu_ 和 conn_mu_，先获取 transfer_mu_
  // - 例如：poll_async 先锁 transfer_mu_，再锁 conn_mu_
  // - poll_async 推进完成后释放锁，再调用 reset_conn_window_counters_
  //
  // 性能考虑：
  // - transfer_mu_ 是热点锁（用户线程频繁访问）
  // - 优化策略：
  //   * 用户线程 busy-wait polling，最快响应
  //   * 无后台线程，避免上下文切换开销
  //
  mutable std::mutex transfer_mu_;
  std::unordered_map<uint64_t, PendingTransfer> transfer_map_;

  // ==========================================================================
  // CUDA 资源（用于接收端 GPU unpack）
  // ==========================================================================
  cudaStream_t unpack_stream_ = nullptr;                      // CUDA stream（用于 unpack kernel）
  std::unique_ptr<device::UnpackLauncher> unpack_launcher_;   // Unpack kernel 启动器

  // ==========================================================================
  // 配置参数（从环境变量读取）
  // ==========================================================================
  size_t chunk_bytes_ = 0;          // 每个 chunk 的大小（UCCL_TCPX_CHUNK_BYTES，默认 512KB）
  size_t max_send_inflight_ = 0;    // 发送端最大在途 chunks（UCCL_TCPX_MAX_SEND_INFLIGHT，默认 12）
  size_t max_recv_inflight_ = 0;    // 接收端最大在途 chunks（UCCL_TCPX_MAX_RECV_INFLIGHT，默认 16）
  bool debug_enabled_ = false;      // 调试日志开关（UCCL_TCPX_DEBUG）
  CUdevice cu_device_ = 0;          // CUDA 设备句柄
  CUcontext cu_context_ = nullptr;  // CUDA 上下文（primary context）

  // ==========================================================================
  // 滑动窗口控制（流控）
  // ==========================================================================
  // 设计要点：
  // - 每个连接独立计数发送和接收方向的在途 chunks
  // - reserve_*_slot：尝试获取窗口配额（如果已满则返回 false）
  // - release_*_slot：释放窗口配额
  //
  // 【锁 4】window_mu_: 窗口计数器的互斥锁（mutex）
  //
  // 保护资源：
  // - send_inflight_chunks_: 每个连接的发送在途 chunks 数量
  // - recv_inflight_chunks_: 每个连接的接收在途 chunks 数量
  //
  // 锁类型：std::mutex（普通互斥锁）
  // - 操作简单（递增/递减计数器），不需要读写锁
  //
  // 使用场景：
  // - reserve_send_slot/reserve_recv_slot: 调度前检查窗口是否有空闲配额
  // - release_send_slot/release_recv_slot: chunk 完成后释放配额
  // - reset_conn_window_counters_: 用于连接关闭及传输完成后的计数重置（全连接维度）
  //
  // 持锁时间：
  // - 非常短：只做简单的计数器操作
  // - 不会调用任何可能阻塞的函数
  //
  // 锁顺序（避免死锁）：
  // - window_mu_ 独立于其他锁，不会同时持有
  // - 例如：poll_async 先释放 transfer_mu_，再调用 reset_conn_window_counters_
  //
  // 性能考虑：
  // - 高频操作：每个 chunk 调度和完成都会访问
  // - 优化策略：持锁时间极短，只做计数器操作
  //
  mutable std::mutex window_mu_;
  std::unordered_map<uint64_t, size_t> send_inflight_chunks_;
  std::unordered_map<uint64_t, size_t> recv_inflight_chunks_;

  // ==========================================================================
  // 调度结果枚举
  // ==========================================================================
  enum class ScheduleOutcome {
    kNoProgress,  // 无进度（窗口已满或无待调度 chunks）
    kProgress,    // 有进度（成功调度了至少一个 chunk）
    kError        // 错误（TCPX 调用失败）
  };

  // ==========================================================================
  // 连接生命周期管理
  // ==========================================================================

  /**
   * 释放连接资源
   *
   * 清理顺序：
   * 1. 遍历 mr_map_，收集该连接的所有 TCPX 注册句柄
   * 2. 调用 tcpx_dereg_mr 释放所有句柄
   * 3. 销毁 CUDA 事件池（recv_events）
   * 4. 关闭 TCPX send_comm 和 recv_comm
   * 5. 关闭 TCP 控制套接字
   *
   * @param conn 连接对象
   */
  void free_conn_(std::shared_ptr<Conn> const& conn);

  // ==========================================================================
  // Chunk 调度（Stage 0：提交 TCPX 请求）
  // ==========================================================================

  /**
   * 调度发送 chunks（持锁调用）
   *
   * 流程：
   * 1. 检查滑动窗口是否有空闲配额（reserve_send_slot）
   * 2. 调用 tcpx_isend 提交 chunk
   * 3. 如果返回 kTcpxBusy，短暂休眠后重试（最多 512 次）
   * 4. 成功后将 chunk 索引加入 send_queue
   *
   * @param conn 连接对象
   * @param transfer 传输对象
   * @return 调度结果（kNoProgress/kProgress/kError）
   */
  ScheduleOutcome schedule_send_chunks_locked(Conn& conn,
                                              PendingTransfer& transfer);

  /**
   * 调度接收 chunks（持锁调用）
   *
   * 流程：
   * 1. 检查滑动窗口是否有空闲配额（reserve_recv_slot）
   * 2. 调用 tcpx_irecv 提交 chunk
   * 3. 如果返回 kTcpxBusy，短暂休眠后重试（最多 512 次）
   * 4. 成功后将 chunk 索引加入 recv_stage1_queue
   *
   * @param conn 连接对象
   * @param transfer 传输对象
   * @return 调度结果（kNoProgress/kProgress/kError）
   */
  ScheduleOutcome schedule_recv_chunks_locked(Conn& conn,
                                              PendingTransfer& transfer);

  // ==========================================================================
  // 滑动窗口流控
  // ==========================================================================

  /**
   * 尝试获取发送窗口配额
   *
   * @param conn_id 连接 ID
   * @param limit 窗口上限
   * @return 成功返回 true，窗口已满返回 false
   */
  bool reserve_send_slot(uint64_t conn_id, size_t limit);

  /**
   * 尝试获取接收窗口配额
   *
   * @param conn_id 连接 ID
   * @param limit 窗口上限
   * @return 成功返回 true，窗口已满返回 false
   */
  bool reserve_recv_slot(uint64_t conn_id, size_t limit);

  /**
   * 释放发送窗口配额
   *
   * @param conn_id 连接 ID
   */
  void release_send_slot(uint64_t conn_id);

  /**
   * 释放接收窗口配额
   *
   * @param conn_id 连接 ID
   */
  void release_recv_slot(uint64_t conn_id);

  // ==========================================================================
  // 内存注册辅助函数
  // ==========================================================================

  /**
   * 为连接填充 TCPX 内存句柄
   *
   * 流程：
   * 1. 检查 MrEntry 中是否已缓存该连接的句柄
   * 2. 如果未缓存，调用 tcpx_reg_mr 注册并缓存
   * 3. 返回句柄指针
   *
   * @param conn 连接对象
   * @param mr_id 内存注册 ID
   * @param is_recv 是否为接收方向
   * @param mhandle_out 输出：TCPX 内存句柄
   * @return 成功返回 true，失败返回 false
   */
  bool populate_conn_handles_(Conn& conn, uint64_t mr_id, bool is_recv,
                              void** mhandle_out);
  bool ensure_recv_dev_handle_cached_(Conn& conn);

  // ==========================================================================
  // 传输进度推进（Stage 1/2）
  // ==========================================================================

  /**
   * 推进传输的 Stage 1 和 Stage 2（持锁调用）
   *
   * 发送端流程：
   * - 遍历 send_queue，对每个 chunk 调用 poll_chunk_request_
   * - 如果 tcpx_test 返回完成，标记 stage2_done 并释放窗口配额
   *
   * 接收端流程：
   * - Stage 1：遍历 recv_stage1_queue，对每个 chunk 调用 poll_chunk_request_
   *   - 如果 tcpx_test 返回完成，启动 GPU unpack kernel
   *   - 将 chunk 索引移入 recv_stage2_queue
   * - Stage 2：遍历 recv_stage2_queue，对每个 chunk 调用 cudaEventQuery
   *   - 如果 event 已 ready，调用 finalize_recv_chunk_ 释放资源
   *   - 标记 stage2_done 并释放窗口配额
   *
   * @param conn 连接对象
   * @param transfer 传输对象
   * @param schedule_send 输出：是否需要调度更多发送 chunks
   * @param schedule_recv 输出：是否需要调度更多接收 chunks
   * @return 成功返回 true，失败返回 false
   */
  bool progress_transfer_locked(Conn& conn, PendingTransfer& transfer,
                                bool* schedule_send, bool* schedule_recv);

  /**
   * 统一的传输推进辅助函数（持锁调用）
   *
   * 流程：
   * 1. 调用 schedule_*_chunks_locked 调度新 chunks（Stage 0）
   * 2. 调用 progress_transfer_locked 推进已提交的 chunks（Stage 1/2）
   * 3. 如果有进度，再次尝试调度（充分利用窗口）
   * 4. 检查是否所有 chunks 都已完成
   *
   * @param conn 连接对象
   * @param transfer 传输对象
   * @param transfer_complete 输出：传输是否完成
   * @return 成功返回 true，失败返回 false
   */
  bool advance_transfer_locked(Conn& conn, PendingTransfer& transfer,
                               bool* transfer_complete);

  /**
   * 完成传输并清理资源（持锁调用）
   *
   * 从 transfer_map_ 中移除传输
   *
   * @param it 传输迭代器
   */
  void finalize_transfer_locked(
      std::unordered_map<uint64_t, PendingTransfer>::iterator it);

  /**
   * 重置连接的窗口计数器
   *
   * 清空 send_inflight_chunks_ 和 recv_inflight_chunks_ 中的条目
   *
   * @param conn_id 连接 ID
   */
  void reset_conn_window_counters_(uint64_t conn_id);

  // ==========================================================================
  // Chunk 级别的进度推进
  // ==========================================================================

  /**
   * 轮询单个 chunk 的 TCPX 请求状态
   *
   * 调用 tcpx_test 检查网络传输是否完成：
   * - rc == kTcpxBusy：仍在进行中
   * - rc == 2：连接关闭（如果 completed=1 则忽略）
   * - rc == 0 && completed == 1：完成
   *
   * @param transfer 传输对象
   * @param chunk Chunk 状态
   * @param done 输出：是否完成
   * @param received_size 输出：接收到的字节数
   * @return 成功返回 true，失败返回 false
   */
  bool poll_chunk_request_(PendingTransfer& transfer,
                           PendingTransfer::ChunkState& chunk, bool* done,
                           int* received_size);

  /**
   * 将接收 chunk 的 GPU unpack 操作加入队列
   *
   * 流程：
   * 1. 从 TCPX request 中提取 unpack 元数据（fragment 数量和偏移）
   * 2. 从 recv_dev_handle 中读取 bounce buffer 地址
   * 3. 构建 UnpackDescriptorBlock
   * 4. 调用 unpack_launcher_->launch 启动 CUDA kernel
   * 5. 从 Conn::recv_events pool 中获取 event 并 record
   *
   * @param transfer 传输对象
   * @param chunk Chunk 状态
   * @param request TCPX 请求对象
   * @param conn 连接对象
   * @return 成功返回 true，失败返回 false
   */
  bool enqueue_chunk_unpack_(PendingTransfer& transfer,
                             PendingTransfer::ChunkState& chunk,
                             tcpx::plugin::tcpxRequest* request, Conn& conn);

  /**
   * 完成接收 chunk 并释放资源
   *
   * 流程：
   * 1. 调用 tcpx_irecv_consumed 释放 bounce buffer
   * 2. 清空 chunk.event 和 chunk.request（不销毁 event，由 pool 拥有）
   *
   * @param conn 连接对象
   * @param chunk Chunk 状态
   * @return 成功返回 true，失败返回 false
   */
  bool finalize_recv_chunk_(Conn& conn, PendingTransfer::ChunkState& chunk);



  // ==========================================================================
  // 传输提交辅助函数
  // ==========================================================================

  /**
   * 提交发送传输
   *
   * 流程：
   * 1. 调用 populate_conn_handles_ 获取 TCPX 内存句柄
   * 2. 根据 chunk_bytes_ 计算 chunk 数量
   * 3. 创建 PendingTransfer 并填充 chunks 向量
   * 4. 调用 schedule_send_chunks_locked 提交初始 chunks
   * 5. 将传输加入 transfer_map_
   *
   * @param conn 连接对象
   * @param mr_id 内存注册 ID
   * @param mr 内存注册条目
   * @param data 源数据地址
   * @param size 数据大小
   * @param tag TCPX 标签
   * @param transfer_id 输出：传输 ID
   * @return 成功返回 true，失败返回 false
   */
  bool post_send_(Conn& conn, uint64_t mr_id, MrEntry const& mr,
                  void const* data, size_t size, int tag,
                  uint64_t& transfer_id);

  /**
   * 提交接收传输
   *
   * 流程：
   * 1. 调用 populate_conn_handles_ 获取 TCPX 内存句柄
   * 2. 根据 chunk_bytes_ 计算 chunk 数量
   * 3. 创建 PendingTransfer 并填充 chunks 向量
   * 4. 调用 schedule_recv_chunks_locked 提交初始 chunks
   * 5. 将传输加入 transfer_map_
   *
   * @param conn 连接对象
   * @param mr_id 内存注册 ID
   * @param mr 内存注册条目
   * @param data 目标地址
   * @param size 数据大小
   * @param tag TCPX 标签
   * @param transfer_id 输出：传输 ID
   * @param needs_unpack 是否需要 GPU unpack
   * @return 成功返回 true，失败返回 false
   */
  bool post_recv_(Conn& conn, uint64_t mr_id, MrEntry const& mr, void* data,
                  size_t size, int tag, uint64_t& transfer_id,
                  bool needs_unpack);
};

}  // namespace tcpx
