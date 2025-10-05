/**
 * @file test_tcpx_perf.cc
 * @brief TCPX GPU-to-GPU 性能基准测试程序
 *
 * 【程序目标】
 * 测量两个 H100 节点之间通过 TCPX (GPU Direct TCPX) 进行 GPU-to-GPU
 * 数据传输的性能。 基于 test_tcpx_transfer.cc 的逻辑，增加了迭代和计时功能。
 *
 * 【核心设计】
 * - Server: 接收数据，使用 GPU kernel 解包（从分散的 bounce buffer
 * 拷贝到连续的目标 GPU 内存）
 * - Client: 发送数据
 * - 使用滑动窗口机制避免耗尽 TCPX 请求池（每个 comm 最多 16 个并发请求）
 *
 * 【使用方法】
 *   # Server 端（10.65.74.150）
 *   UCCL_TCPX_PERF_SIZE=67108864 ./tests/test_tcpx_perf server 0
 *
 *   # Client 端（10.64.113.77）
 *   UCCL_TCPX_PERF_SIZE=67108864 ./tests/test_tcpx_perf client 10.65.74.150 0
 *
 * 【环境变量】
 *   UCCL_TCPX_PERF_SIZE: 每次迭代传输的总字节数（默认 4MB）
 *   UCCL_TCPX_PERF_ITERS: 迭代次数（默认 10）
 *   UCCL_TCPX_CHUNK_BYTES: 每个 chunk 的大小（默认 2MB，优化后从 512KB 增加）
 *   UCCL_TCPX_UNPACK_IMPL: 解包实现方式（kernel|d2d|host，默认 kernel）
 */

#include "../device/unpack_launch.h"    // GPU kernel 启动器
#include "../include/rx_descriptor.h"   // 接收描述符构建工具
#include "../include/tcpx_interface.h"  // TCPX API 封装
#include "../include/tcpx_structs.h"    // TCPX 内部结构定义
#include <arpa/inet.h>
#include <netinet/in.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

namespace {
// ============================================================================
// 常量定义
// ============================================================================

constexpr size_t kHandleBytes =
    128;  // NCCL handle 大小（用于 bootstrap 交换连接信息）
struct ncclNetHandle_v7 {
  char data[kHandleBytes];
};  // NCCL v7 handle 结构

constexpr int kBootstrapPort =
    12347;  // Bootstrap TCP 端口（用于交换 TCPX handle）
constexpr int kTransferTag = 99;  // 基础传输标签（每个 chunk 会加上偏移）
constexpr size_t kMaxSize = 256 * 1024 * 1024;  // 最大传输大小（256MB）
constexpr size_t kRegisteredBytes =
    kMaxSize + 4096;  // 注册内存大小（额外 4KB 用于对齐）

// ============================================================================
// 辅助函数：环境变量读取
// ============================================================================

// 读取整数类型环境变量
int getEnvInt(char const* name, int def) {
  char const* v = std::getenv(name);
  return v ? std::atoi(v) : def;
}

// 读取 size_t 类型环境变量（用于大尺寸配置）
size_t getEnvSize(char const* name, size_t def) {
  char const* v = std::getenv(name);
  return v ? static_cast<size_t>(std::atoll(v)) : def;
}

// ============================================================================
// Bootstrap 连接函数（用于交换 TCPX handle）
// ============================================================================

/**
 * @brief 创建 bootstrap server 并等待 client 连接
 * @return 已连接的 client socket fd（失败返回 -1）
 *
 * 【重要】这个函数内部已经调用了 accept()，返回的是已连接的 client_fd
 * 这与某些实现不同（有些只返回 listen_fd），需要注意！
 */
int create_bootstrap_server() {
  // 创建 TCP socket
  int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd < 0) return -1;

  // 设置 SO_REUSEADDR 允许快速重启（避免 TIME_WAIT 状态占用端口）
  int opt = 1;
  setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  // 绑定到所有网卡的 kBootstrapPort 端口
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;  // 监听所有网卡
  addr.sin_port = htons(kBootstrapPort);

  if (bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    close(listen_fd);
    return -1;
  }

  // 开始监听（backlog=1，只接受一个连接）
  if (listen(listen_fd, 1) < 0) {
    close(listen_fd);
    return -1;
  }

  // 【关键】立即 accept 并返回 client_fd（阻塞直到 client 连接）
  int client_fd = accept(listen_fd, nullptr, nullptr);
  close(listen_fd);  // 关闭 listen socket（不再需要）
  return client_fd;
}

/**
 * @brief 连接到 bootstrap server
 * @param ip Server IP 地址
 * @return 连接的 socket fd（失败返回 -1）
 *
 * 【重试机制】最多重试 30 次，每次间隔 100ms（总共 3 秒）
 */
int connect_bootstrap(char const* ip) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(kBootstrapPort);
  inet_aton(ip, &addr.sin_addr);

  // 重试连接（server 可能还没启动）
  for (int i = 0; i < 30; ++i) {
    if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0)
      return fd;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  close(fd);
  return -1;
}
}  // namespace

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char** argv) {
  // ============================================================================
  // 环境变量设置（TCPX 配置）
  // ============================================================================

  // 启用 zero-copy（从 4KB 开始使用 devmem-tcp）
  setenv("NCCL_MIN_ZCOPY_SIZE", "4096", 0);
  setenv("NCCL_GPUDIRECTTCPX_MIN_ZCOPY_SIZE", "4096", 0);

  // 启用接收同步（确保数据完整性）
  setenv("NCCL_GPUDIRECTTCPX_RECV_SYNC", "1", 0);

  // 启用 TCPX wrapper 调试日志（除非用户覆盖）
  setenv("UCCL_TCPX_DEBUG", "1", 0);

  // 默认关闭 kernel launch 调试日志（太详细）
  if (!std::getenv("UCCL_TCPX_LAUNCH_DEBUG"))
    setenv("UCCL_TCPX_LAUNCH_DEBUG", "0", 0);

  // ============================================================================
  // 命令行参数解析
  // ============================================================================

  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <server|client> <gpu_id|server_ip> [gpu_id]" << std::endl;
    return 1;
  }

  bool is_server = (std::string(argv[1]) == "server");
  int gpu_id = 0;
  std::string server_ip;

  if (is_server) {
    // Server 模式: ./test_tcpx_perf server <gpu_id>
    gpu_id = std::atoi(argv[2]);
  } else {
    // Client 模式: ./test_tcpx_perf client <server_ip> [gpu_id]
    server_ip = argv[2];
    gpu_id = (argc > 3) ? std::atoi(argv[3]) : 0;
  }

  // ============================================================================
  // 测试参数配置
  // ============================================================================

  // 每次迭代传输的总字节数（默认 4MB）
  size_t test_size = getEnvSize("UCCL_TCPX_PERF_SIZE", 4 * 1024 * 1024);

  // 迭代次数（用于计算平均性能）
  int iterations = getEnvInt("UCCL_TCPX_PERF_ITERS", 10);

  // Chunk 大小：优先使用 UCCL_TCPX_CHUNK_BYTES，其次
  // NCCL_P2P_NET_CHUNKSIZE，默认 2MB 【重要】Chunk 机制用于避免单次传输过大导致
  // TCPX bounce buffer 压力过大 【优化】从 512KB 增加到 2MB，减少 chunk
  // 数量，降低固定开销
  size_t chunk_bytes =
      getEnvSize("UCCL_TCPX_CHUNK_BYTES",
                 getEnvSize("NCCL_P2P_NET_CHUNKSIZE", 2 * 1024 * 1024));

  // 限制最大传输大小
  if (test_size > kMaxSize) test_size = kMaxSize;

  std::cout << "[PERF] Mode: " << (is_server ? "SERVER" : "CLIENT")
            << std::endl;
  std::cout << "[PERF] GPU: " << gpu_id << std::endl;
  std::cout << "[PERF] Size: " << (test_size / 1024 / 1024) << " MB"
            << std::endl;
  std::cout << "[PERF] Iterations: " << iterations << std::endl;

  // ============================================================================
  // TCPX 插件初始化
  // ============================================================================

  int ndev = tcpx_get_device_count();
  if (ndev <= 0 || gpu_id >= ndev) {
    std::cerr << "[ERROR] Invalid GPU" << std::endl;
    return 1;
  }

  // ============================================================================
  // SERVER 端逻辑
  // ============================================================================

  if (is_server) {
    // ==========================================================================
    // 步骤 1: 创建 TCPX listen comm 并生成 handle
    // ==========================================================================

    ncclNetHandle_v7 handle{};  // 用于存储连接信息的 handle
    void* listen_comm = nullptr;

    // 调用 TCPX 插件的 listen 接口，生成 handle（包含 IP、端口等信息）
    if (tcpx_listen(gpu_id, &handle, &listen_comm) != 0) {
      std::cerr << "[ERROR] tcpx_listen failed" << std::endl;
      return 1;
    }

    std::cout << "[PERF] Waiting for client..." << std::endl;

    // ==========================================================================
    // 步骤 2: 通过 bootstrap TCP 连接交换 handle
    // ==========================================================================

    // 【重要】create_bootstrap_server 内部已经调用了 accept()
    // 返回的是已连接的 client_fd，不是 listen_fd！
    int bootstrap_fd = create_bootstrap_server();
    if (bootstrap_fd < 0) {
      std::cerr << "[ERROR] bootstrap server failed" << std::endl;
      return 1;
    }

    std::cout << "[PERF] Bootstrap connection established, sending handle"
              << std::endl;

    // 通过 bootstrap 连接发送 TCPX handle 给 client
    // 【注意】使用循环发送，因为 send() 可能不会一次发送完所有数据
    size_t total_sent = 0;
    while (total_sent < kHandleBytes) {
      ssize_t sent = send(bootstrap_fd, handle.data + total_sent,
                          kHandleBytes - total_sent, 0);
      if (sent <= 0) {
        std::cerr << "[ERROR] Failed to send handle" << std::endl;
        close(bootstrap_fd);
        return 1;
      }
      total_sent += static_cast<size_t>(sent);
    }

    // ==========================================================================
    // 步骤 3: Accept TCPX 连接（client 会用 handle 连接过来）
    // ==========================================================================

    void* recv_comm = nullptr;  // 接收通信句柄

    // 【重要】recv_dev_handle 必须 16 字节对齐（TCPX 插件要求）
    alignas(16) unsigned char recv_dev_handle_storage[512] = {0};
    void* recv_dev_handle = recv_dev_handle_storage;

    // 【重试机制】tcpx_accept_v5 可能需要多次调用才能成功
    // 因为 client 的 connect 可能还没到达
    int attempts = 0;
    constexpr int kMaxRetries = 100;
    while (attempts < kMaxRetries) {
      int rc = tcpx_accept_v5(listen_comm, &recv_comm, &recv_dev_handle);
      if (rc != 0) {
        std::cerr << "[ERROR] tcpx_accept_v5 returned rc=" << rc << std::endl;
        close(bootstrap_fd);
        return 1;
      }
      if (recv_comm) break;  // 成功获取 recv_comm
      ++attempts;
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!recv_comm) {
      std::cerr << "[ERROR] Failed to obtain recv_comm after retries"
                << std::endl;
      close(bootstrap_fd);
      return 1;
    }

    // ==========================================================================
    // 步骤 4: 选择 Unpack 实现方式
    // ==========================================================================

    // 支持三种 unpack 实现：
    // - kernel: 使用 GPU kernel 解包（最快，推荐）
    // - d2d: 使用 cuMemcpyDtoD 逐个 fragment 拷贝（中等速度）
    // - host: 先 DtoH 到 host，gather，再 HtoD（最慢，仅用于调试）
    char const* impl_env = std::getenv("UCCL_TCPX_UNPACK_IMPL");
    std::string impl = impl_env ? std::string(impl_env) : std::string("kernel");
    std::transform(impl.begin(), impl.end(), impl.begin(), ::tolower);
    std::cout << "[PERF] Unpack impl: " << impl << std::endl;

    std::cout << "[PERF] TCPX connection established" << std::endl;

    // ==========================================================================
    // 步骤 5: 配置接收模式（GPU 或 Host）
    // ==========================================================================

    // Host-recv 调试模式：接收到 host 内存而不是 GPU 内存
    // 【注意】这会禁用 devmem-tcp zero-copy，仅用于调试
    bool use_host_recv = false;
    if (char const* env = std::getenv("UCCL_TCPX_HOST_RECV_DEBUG")) {
      use_host_recv = (std::string(env) != "0");
    }
    std::cout << "[PERF] Host recv debug mode: "
              << (use_host_recv ? "ON" : "OFF") << std::endl;

    // ==========================================================================
    // 步骤 6: CUDA 初始化
    // ==========================================================================

    CUdevice cuDev;   // CUDA 设备句柄
    CUcontext cuCtx;  // CUDA 上下文

    // 初始化 CUDA Driver API
    if (cuInit(0) != CUDA_SUCCESS ||
        cuDeviceGet(&cuDev, gpu_id) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&cuCtx, cuDev) != CUDA_SUCCESS ||
        cuCtxSetCurrent(cuCtx) != CUDA_SUCCESS) {
      std::cerr << "[ERROR] CUDA initialization failed" << std::endl;
      close(bootstrap_fd);
      return 1;
    }

    // 【重要】必须调用 cudaSetDevice，否则 CUDA Runtime API 会使用错误的设备
    if (cudaSetDevice(gpu_id) != cudaSuccess) {
      std::cerr << "[ERROR] cudaSetDevice failed" << std::endl;
      close(bootstrap_fd);
      return 1;
    }

    // ==========================================================================
    // 步骤 7: 分配接收缓冲区
    // ==========================================================================

    CUdeviceptr d_base = 0, d_aligned = 0;  // GPU 内存指针
    void* h_recv_base = nullptr;            // Host 内存指针
    void* recv_buf = nullptr;               // 实际使用的缓冲区指针
    int recv_ptr_type = NCCL_PTR_CUDA;      // 内存类型标志

    if (!use_host_recv) {
      // GPU 内存模式（正常模式）

      // 分配额外的 4096 字节用于对齐
      if (cuMemAlloc(&d_base, kRegisteredBytes + 4096) != CUDA_SUCCESS) {
        std::cerr << "[ERROR] CUDA allocation failed" << std::endl;
        close(bootstrap_fd);
        return 1;
      }

      // 【重要】对齐到 4KB 边界（devmem-tcp 要求）
      uintptr_t addr = static_cast<uintptr_t>(d_base);
      addr = (addr + 4095) & ~static_cast<uintptr_t>(4095);  // 向上对齐到 4KB
      d_aligned = static_cast<CUdeviceptr>(addr);
      recv_buf = reinterpret_cast<void*>(d_aligned);
      recv_ptr_type = NCCL_PTR_CUDA;
    } else {
      // Host 内存模式（调试模式）
      if (cudaMallocHost(&h_recv_base, kRegisteredBytes) != cudaSuccess) {
        std::cerr << "[ERROR] cudaMallocHost failed" << std::endl;
        close(bootstrap_fd);
        return 1;
      }
      recv_buf = h_recv_base;
      recv_ptr_type = NCCL_PTR_HOST;
    }

    // ==========================================================================
    // 步骤 8: 注册内存到 TCPX
    // ==========================================================================

    void* recv_mhandle = nullptr;
    std::cout << "[PERF][SERVER] Registering recv buffer: ptr=" << recv_buf
              << " size=" << kRegisteredBytes << " type="
              << (recv_ptr_type == NCCL_PTR_CUDA ? "NCCL_PTR_CUDA"
                                                 : "NCCL_PTR_HOST")
              << std::endl;

    // 注册内存：TCPX 插件会设置 devmem-tcp 映射（如果是 GPU 内存）
    if (tcpx_reg_mr(recv_comm, recv_buf, kRegisteredBytes, recv_ptr_type,
                    &recv_mhandle) != 0) {
      std::cerr << "[ERROR] tcpx_reg_mr failed" << std::endl;
      if (h_recv_base) cudaFreeHost(h_recv_base);
      if (d_base) cuMemFree(d_base);
      close(bootstrap_fd);
      return 1;
    }
    std::cout << "[PERF][SERVER] Recv buffer registered successfully, mhandle="
              << recv_mhandle << std::endl;

    // ==========================================================================
    // 步骤 9: 创建持久化的 Stream 和 Launcher（仅 kernel 模式）
    // ==========================================================================

    // 【关键优化】在循环外创建 stream 和 launcher，避免每个 chunk 都创建/销毁
    // 这是性能优化的核心：避免了每个 chunk ~4ms 的 stream 创建开销
    cudaStream_t unpack_stream = nullptr;
    tcpx::device::UnpackLauncher* launcher_ptr = nullptr;

    // ==========================================================================
    // 步骤 10: 滑动窗口机制（避免耗尽 TCPX 请求池）
    // ==========================================================================

    // 【核心问题】TCPX 插件每个 comm 只有 MAX_REQUESTS=16 个请求槽
    // 如果同时发起超过 16 个 irecv，会报错 "unable to allocate requests"
    //
    // 【解决方案】滑动窗口：
    // 1. 最多同时有 MAX_INFLIGHT (16) 个 chunk 在处理中
    // 2. 每个 chunk 的生命周期：irecv → kernel launch → event record → event
    // sync → irecv_consumed
    // 3. 当窗口满时，等待最老的 chunk 完成，释放其请求槽，再发起新的 irecv
    //
    // 【为什么需要 CUDA events】
    // - 不能在 irecv 完成后立即调用 irecv_consumed（数据还在 bounce
    // buffer，kernel 还没拷贝完）
    // - 必须等待 kernel 完成后才能调用 irecv_consumed
    // - 使用 CUDA event 跟踪每个 chunk 的 kernel 完成状态

    constexpr int MAX_INFLIGHT =
        16;  // 最大并发 chunk 数（等于 TCPX 请求池大小）
    std::vector<cudaEvent_t> events;  // CUDA events（用于跟踪 kernel 完成）
    std::vector<void*> pending_reqs;  // 待完成的 TCPX 请求
    std::vector<int> pending_indices;  // 待完成的 chunk 索引（用于选择 event）

    if (!use_host_recv && impl == "kernel") {
      // 创建 CUDA stream（所有 kernel 都在这个 stream 上异步执行）
      if (cudaStreamCreate(&unpack_stream) != cudaSuccess) {
        std::cerr << "[ERROR] Failed to create unpack stream" << std::endl;
        if (h_recv_base) cudaFreeHost(h_recv_base);
        if (d_base) cuMemFree(d_base);
        tcpx_dereg_mr(recv_comm, recv_mhandle);
        tcpx_close_recv(recv_comm);
        tcpx_close_listen(listen_comm);
        close(bootstrap_fd);
        return 1;
      }

      // 创建 UnpackLauncher（管理 kernel 启动）
      tcpx::device::UnpackLaunchConfig cfg;
      cfg.stream = unpack_stream;
      cfg.enable_profiling = false;  // 关闭性能分析（减少开销）
      cfg.use_small_kernel =
          true;  // 使用优化的小 kernel（warp-per-descriptor）
      launcher_ptr = new tcpx::device::UnpackLauncher(cfg);

      // 预创建 CUDA events（避免运行时创建开销）
      events.resize(MAX_INFLIGHT);
      for (int i = 0; i < MAX_INFLIGHT; ++i) {
        if (cudaEventCreate(&events[i]) != cudaSuccess) {
          std::cerr << "[ERROR] Failed to create CUDA event " << i << std::endl;
          return 1;
        }
      }

      // 预分配 vector 容量（避免动态扩容）
      pending_reqs.reserve(MAX_INFLIGHT);
      pending_indices.reserve(MAX_INFLIGHT);

      std::cout << "[PERF][SERVER] Created persistent stream, launcher, and "
                << MAX_INFLIGHT << " events for async kernel mode" << std::endl;
    }

    // ==========================================================================
    // 步骤 11: 性能测试主循环
    // ==========================================================================

    double total_time_ms = 0.0;  // 累计所有迭代的时间

    for (int iter = 0; iter < iterations; ++iter) {
      std::cout << "[PERF] Iteration " << iter << ": total bytes=" << test_size
                << ", chunk_bytes=" << chunk_bytes << std::endl;

      auto start = std::chrono::high_resolution_clock::now();

      // 每次迭代开始时重置滑动窗口状态
      if (!use_host_recv && impl == "kernel") {
        // [DEBUG] Iteration start: clearing sliding window (removed for
        // performance)
        pending_reqs.clear();
        pending_indices.clear();
      }

      // ========================================================================
      // Chunk 循环：将大消息分成多个 chunk 接收
      // ========================================================================

      size_t offset = 0;      // 当前接收偏移量
      int chunk_counter = 0;  // Chunk 计数器（用于选择 event）

      while (offset < test_size) {
        // 计算当前 chunk 的大小（最后一个 chunk 可能小于 chunk_bytes）
        const size_t this_chunk = std::min(chunk_bytes, test_size - offset);

        // 计算目标地址（recv_buf + offset）
        void* dst_ptr = reinterpret_cast<void*>(
            reinterpret_cast<uintptr_t>(recv_buf) + offset);

        // 计算 chunk 索引和唯一 tag
        const size_t chunk_idx = offset / chunk_bytes;

        // 【重要】每个 chunk 使用唯一的 tag，避免 TCPX 插件混淆不同的请求
        // tag = 基础 tag + 迭代编号*10000 + chunk 索引
        int const tag = kTransferTag + static_cast<int>(iter) * 10000 +
                        static_cast<int>(chunk_idx);

        std::cout << "[PERF][SERVER] chunk_idx=" << chunk_idx << " tag=" << tag
                  << " size=" << this_chunk << " offset=" << offset
                  << std::endl;

        // ======================================================================
        // 【修复】滑动窗口检查 - 必须在 tcpx_irecv 之前！
        // ======================================================================

        // 【问题】TCPX 插件每个 comm 只有 16 个请求槽
        // 如果同时有超过 16 个 irecv 请求未调用 irecv_consumed，会报错
        //
        // 【解决方案】在发起新的 irecv 之前，检查滑动窗口是否已满
        // 如果满了，先等待最老的 chunk 完成并释放请求槽

        if (!use_host_recv && impl == "kernel") {
          if (pending_reqs.size() >= MAX_INFLIGHT) {
            // 获取最老的 chunk 的索引和 event
            int oldest_idx = pending_indices.front();
            cudaEvent_t oldest_event = events[oldest_idx % MAX_INFLIGHT];

            // [DEBUG] Sliding window FULL (removed for performance)

            // 【关键】等待最老的 chunk 的 kernel 完成
            cudaError_t err = cudaEventSynchronize(oldest_event);
            if (err != cudaSuccess) {
              std::cerr << "[ERROR] cudaEventSynchronize (pre-irecv) failed: "
                        << cudaGetErrorString(err) << std::endl;
              break;
            }

            // 【关键】释放最老的 chunk 的 TCPX 请求槽
            void* oldest_req = pending_reqs.front();
            // int oldest_chunk_idx = pending_indices.front();  // Unused after
            // removing debug logs

            // [DEBUG] Releasing TCPX request (removed for performance)

            tcpx_irecv_consumed(recv_comm, 1, oldest_req);

            // 从滑动窗口中移除最老的 chunk
            pending_reqs.erase(pending_reqs.begin());
            pending_indices.erase(pending_indices.begin());

            // [DEBUG] Request released (removed for performance)
          }
        }

        // ======================================================================
        // 发起异步接收（tcpx_irecv）
        // ======================================================================

        // TCPX irecv 参数（支持批量接收，这里只接收 1 个）
        void* recv_data[1] = {dst_ptr};  // 目标地址数组
        int recv_sizes[1] = {static_cast<int>(this_chunk)};  // 大小数组
        int recv_tags[1] = {tag};                            // Tag 数组
        void* recv_mhandles[1] = {recv_mhandle};  // 内存句柄数组
        void* recv_request = nullptr;             // 输出：请求句柄

        int irecv_rc = tcpx_irecv(recv_comm, 1, recv_data, recv_sizes,
                                  recv_tags, recv_mhandles, &recv_request);
        if (irecv_rc != 0) {
          std::cerr << "[ERROR] tcpx_irecv failed: rc=" << irecv_rc
                    << " chunk_idx=" << chunk_idx << " iter=" << iter
                    << " offset=" << offset << " tag=" << tag << std::endl;
          std::cerr.flush();  // 强制刷新缓冲区到日志文件
          break;
        }

        // [DEBUG] tcpx_irecv success (removed for performance)

        // ======================================================================
        // 轮询等待接收完成（tcpx_test）
        // ======================================================================

        int done = 0, received_size = 0;

        // 【修复】移除超时限制，持续轮询直到接收完成
        // 原因：
        // 1. tcpxTest 本身没有超时机制，只是检查请求是否完成
        // 2. 之前的 10 秒超时导致 Server 端提前退出（只处理了 17 个 chunks）
        // 3. 性能测试中，我们期望所有数据都能到达
        // 4. 如果真的有问题（如网络断开），程序会卡住，用户可以手动中断
        while (!done) {
          tcpx_test(recv_request, &done, &received_size);
          if (!done) std::this_thread::sleep_for(std::chrono::microseconds(10));
        }

        // ======================================================================
        // Unpack 逻辑：将分散的 bounce buffer 数据拷贝到连续的目标内存
        // ======================================================================

        if (use_host_recv) {
          // Host-recv 调试模式：数据已经在 host buffer，无需 unpack
          std::cout << "[PERF][SERVER] host-recv completed size="
                    << received_size << " (skip unpack)" << std::endl;
          offset += this_chunk;
        } else {
          // ====================================================================
          // GPU 接收模式：需要 unpack（从 bounce buffer 拷贝到目标 GPU 内存）
          // ====================================================================

          // 【关键】将 recv_request 转换为 TCPX 内部结构
          // 这是一个黑盒操作，依赖于 TCPX 插件的内部实现
          auto* rx_req =
              reinterpret_cast<tcpx::plugin::tcpxRequest*>(recv_request);
          auto* dev_handle_struct =
              reinterpret_cast<tcpx::plugin::NcclNetDeviceHandle*>(
                  recv_dev_handle);

          // 验证 TCPX 元数据是否有效
          if (!rx_req || !dev_handle_struct || !rx_req->unpack_slot.mem ||
              !rx_req->unpack_slot.cnt) {
            std::cerr << "[ERROR] Missing TCPX metadata for unpack"
                      << std::endl;
            break;
          }

          // 【重要】读取 fragment 数量（从 GPU 内存读取到 host）
          // cnt 是一个 device 指针，指向一个 uint64_t 计数器
          // TCPX 插件在接收完成后会更新这个计数器
          uint64_t frag_count = *(rx_req->unpack_slot.cnt);
          std::cout << "[PERF][SERVER] frag_count=" << frag_count << std::endl;

          // 验证 fragment 数量是否合理
          if (frag_count == 0 || frag_count > MAX_UNPACK_DESCRIPTORS) {
            std::cerr << "[ERROR] Invalid fragment count: " << frag_count
                      << std::endl;
            break;
          }

          // 【关键】读取 device handle（包含 bounce buffer 地址）
          // dev_handle_struct->handle 是一个 device 指针，需要 DtoH 拷贝
          tcpx::plugin::unpackNetDeviceHandle dev_handle{};
          if (cuMemcpyDtoH(
                  &dev_handle,
                  reinterpret_cast<CUdeviceptr>(dev_handle_struct->handle),
                  sizeof(dev_handle)) != CUDA_SUCCESS) {
            std::cerr << "[ERROR] Failed to read device handle" << std::endl;
            break;
          }

          // 【关键】构建 unpack descriptor block
          // meta_entries: 指向 loadMeta 数组（scatter-gather 列表）
          // 每个 loadMeta 描述一个 fragment：{src_off, len, dst_off}
          auto* meta_entries =
              static_cast<tcpx::plugin::loadMeta*>(rx_req->unpack_slot.mem);
          tcpx::rx::UnpackDescriptorBlock desc_block;
          tcpx::rx::buildDescriptorBlock(
              meta_entries, static_cast<uint32_t>(frag_count),
              dev_handle.bounce_buf, dst_ptr, desc_block);

          // 设置 ready_flag（用于 GPU kernel 的 visibility barrier）
          desc_block.ready_flag = rx_req->unpack_slot.cnt;
          desc_block.ready_threshold = frag_count;

          // ====================================================================
          // 执行 Unpack（根据选择的实现方式）
          // ====================================================================

          int lrc = 0;  // Launch return code

          if (impl == "kernel") {
            // ==================================================================
            // Kernel 模式：使用 GPU kernel 解包（推荐，性能最好）
            // ==================================================================

            // ----------------------------------------------------------------
            // 【注意】滑动窗口检查已经移到 tcpx_irecv 之前（第 530-565 行）
            // ----------------------------------------------------------------
            // 这样可以确保在发起新的 irecv 之前，TCPX 请求池有可用的槽位

            // ----------------------------------------------------------------
            // 启动当前 chunk 的 unpack kernel
            // ----------------------------------------------------------------

            // 【关键】使用异步 launch（不是 launchSync）
            // 这样 kernel 会在 GPU 上异步执行，CPU 可以继续处理下一个 chunk
            lrc = launcher_ptr->launch(desc_block);
            if (lrc != 0) {
              std::cerr << "[ERROR] Unpack kernel launch failed: " << lrc
                        << std::endl;
              break;
            }

            // ----------------------------------------------------------------
            // 记录 CUDA event（用于跟踪 kernel 完成）
            // ----------------------------------------------------------------

            // 【重要】使用 chunk_counter % MAX_INFLIGHT 循环使用 events
            // 因为最多同时有 MAX_INFLIGHT 个 chunk 在处理中
            int event_idx = chunk_counter % MAX_INFLIGHT;
            cudaError_t err = cudaEventRecord(events[event_idx], unpack_stream);
            if (err != cudaSuccess) {
              std::cerr << "[ERROR] cudaEventRecord failed: "
                        << cudaGetErrorString(err) << std::endl;
              break;
            }

            // ----------------------------------------------------------------
            // 将当前 chunk 加入滑动窗口
            // ----------------------------------------------------------------

            pending_reqs.push_back(recv_request);
            pending_indices.push_back(chunk_counter);

          } else if (impl == "d2d") {
            // ==================================================================
            // D2D 模式：使用 cuMemcpyDtoD 逐个 fragment 拷贝
            // ==================================================================

            // 【特点】
            // - 简单直接，不需要 kernel
            // - 性能中等（比 kernel 慢，因为每个 fragment 都是单独的拷贝操作）
            // - 适合调试和验证

            for (uint32_t i = 0; i < desc_block.count; ++i) {
              auto const& meta = desc_block.descriptors[i];

              // 计算源地址（bounce buffer + src_off）
              CUdeviceptr src_ptr = static_cast<CUdeviceptr>(
                  reinterpret_cast<uintptr_t>(desc_block.bounce_buffer) +
                  meta.src_off);

              // 计算目标地址（dst_buffer + dst_off）
              CUdeviceptr dst_ptr = static_cast<CUdeviceptr>(
                  reinterpret_cast<uintptr_t>(desc_block.dst_buffer) +
                  meta.dst_off);

              // 执行 GPU-to-GPU 拷贝
              if (cuMemcpyDtoD(dst_ptr, src_ptr, meta.len) != CUDA_SUCCESS) {
                std::cerr << "[ERROR] D2D copy failed at descriptor " << i
                          << std::endl;
                lrc = -1;
                break;
              }
            }
            if (lrc != 0) break;

            // D2D 模式是同步的，拷贝完成后立即释放请求
            tcpx_irecv_consumed(recv_comm, 1, recv_request);

          } else {  // host gather
            // ==================================================================
            // Host Gather 模式：DtoH → gather → HtoD（最慢，仅用于调试）
            // ==================================================================

            // 【流程】
            // 1. 将所有 fragments 从 GPU 拷贝到 host（DtoH）
            // 2. 在 host 上 gather 成连续内存
            // 3. 将连续内存拷贝回 GPU（HtoD）

            std::vector<unsigned char> tmp(desc_block.total_bytes);
            size_t off = 0;

            // 步骤 1: DtoH（逐个 fragment）
            for (uint32_t i = 0; i < desc_block.count; ++i) {
              auto const& meta = desc_block.descriptors[i];
              CUdeviceptr src_ptr = static_cast<CUdeviceptr>(
                  reinterpret_cast<uintptr_t>(desc_block.bounce_buffer) +
                  meta.src_off);
              if (cuMemcpyDtoH(tmp.data() + off, src_ptr, meta.len) !=
                  CUDA_SUCCESS) {
                std::cerr << "[ERROR] Host gather DtoH failed at descriptor "
                          << i << std::endl;
                lrc = -1;
                break;
              }
              off += meta.len;
            }
            if (lrc != 0) break;

            // 步骤 2: HtoD（连续内存）
            if (cuMemcpyHtoD(
                    static_cast<CUdeviceptr>(
                        reinterpret_cast<uintptr_t>(desc_block.dst_buffer)),
                    tmp.data(), tmp.size()) != CUDA_SUCCESS) {
              std::cerr << "[ERROR] Host gather HtoD failed" << std::endl;
              break;
            }

            // Host gather 是同步的，拷贝完成后立即释放请求
            tcpx_irecv_consumed(recv_comm, 1, recv_request);
          }

          // 更新偏移量和计数器
          offset += this_chunk;
          chunk_counter++;
        }
      }  // end of chunk loop

      // ========================================================================
      // 迭代结束：排空滑动窗口中剩余的 chunks（仅 kernel 模式）
      // ========================================================================

      // 【重要】在迭代结束时，滑动窗口中可能还有未完成的 chunks
      // 必须等待它们全部完成并释放请求槽，否则会泄漏资源

      if (!use_host_recv && impl == "kernel") {
        // [DEBUG] Draining sliding window (removed for performance)

        while (!pending_reqs.empty()) {
          // 获取最老的 chunk
          int oldest_idx = pending_indices.front();
          cudaEvent_t oldest_event = events[oldest_idx % MAX_INFLIGHT];

          // [DEBUG] Waiting for chunk (removed for performance)

          // 等待 kernel 完成
          cudaError_t err = cudaEventSynchronize(oldest_event);
          if (err != cudaSuccess) {
            std::cerr << "[ERROR] cudaEventSynchronize (drain) failed: "
                      << cudaGetErrorString(err) << std::endl;
            break;
          }

          // 释放 TCPX 请求槽
          void* oldest_req = pending_reqs.front();
          tcpx_irecv_consumed(recv_comm, 1, oldest_req);

          // 从窗口中移除
          pending_reqs.erase(pending_reqs.begin());
          pending_indices.erase(pending_indices.begin());
        }

        // [DEBUG] Sliding window drained (removed for performance)
      }

      // ========================================================================
      // 计算本次迭代的性能
      // ========================================================================

      auto end = std::chrono::high_resolution_clock::now();
      double iter_time_ms =
          std::chrono::duration<double, std::milli>(end - start).count();
      total_time_ms += iter_time_ms;
      std::cout << "[PERF] Iter " << iter << " time=" << iter_time_ms << "ms"
                << std::endl;
    }  // end of iteration loop

    // ==========================================================================
    // 计算并输出平均性能
    // ==========================================================================

    double avg_ms = total_time_ms / iterations;
    double bw_gbps =
        (test_size / (1024.0 * 1024.0 * 1024.0)) / (avg_ms / 1000.0);

    std::cout << "[PERF] Avg: " << std::fixed << std::setprecision(3) << avg_ms
              << " ms, "
              << "BW: " << std::fixed << std::setprecision(2) << bw_gbps
              << " GB/s" << std::endl;

    // ==========================================================================
    // 清理资源
    // ==========================================================================

    // 清理持久化的 launcher 和 stream（仅 kernel 模式）
    if (launcher_ptr) {
      delete launcher_ptr;
      launcher_ptr = nullptr;
    }
    if (unpack_stream) {
      cudaStreamDestroy(unpack_stream);
      unpack_stream = nullptr;
    }

    // 清理 CUDA events
    for (auto& evt : events) {
      cudaEventDestroy(evt);
    }

    // 清理 TCPX 资源
    tcpx_dereg_mr(recv_comm, recv_mhandle);

    // 清理内存
    if (h_recv_base) cudaFreeHost(h_recv_base);
    if (d_base) cuMemFree(d_base);

    // 清理 CUDA 上下文
    cuDevicePrimaryCtxRelease(cuDev);

    // 关闭 TCPX 连接
    tcpx_close_recv(recv_comm);
    tcpx_close_listen(listen_comm);

    // 关闭 bootstrap 连接
    close(bootstrap_fd);

    // ============================================================================
    // CLIENT 端逻辑
    // ============================================================================

  } else {
    // ==========================================================================
    // 步骤 1: 连接到 server 的 bootstrap 端口
    // ==========================================================================

    int bootstrap_fd = connect_bootstrap(server_ip.c_str());
    if (bootstrap_fd < 0) {
      std::cerr << "[ERROR] bootstrap connect failed" << std::endl;
      return 1;
    }

    // ==========================================================================
    // 步骤 2: 从 server 接收 TCPX handle
    // ==========================================================================

    ncclNetHandle_v7 handle{};
    size_t total_received = 0;

    // 【注意】使用循环接收，因为 recv() 可能不会一次接收完所有数据
    while (total_received < kHandleBytes) {
      ssize_t r = recv(bootstrap_fd, handle.data + total_received,
                       kHandleBytes - total_received, 0);
      if (r <= 0) {
        std::cerr << "[ERROR] Failed to receive handle" << std::endl;
        close(bootstrap_fd);
        return 1;
      }
      total_received += static_cast<size_t>(r);
    }

    // ==========================================================================
    // 步骤 3: 使用 handle 连接到 server 的 TCPX 端口
    // ==========================================================================

    void* send_comm = nullptr;

    // 【重要】send_dev_handle 必须 16 字节对齐（TCPX 插件要求）
    alignas(16) unsigned char send_dev_handle_storage[512] = {0};
    void* send_dev_handle = send_dev_handle_storage;

    if (tcpx_connect_v5(gpu_id, &handle, &send_comm, &send_dev_handle) != 0 ||
        !send_comm) {
      std::cerr << "[ERROR] tcpx_connect_v5 failed" << std::endl;
      close(bootstrap_fd);
      return 1;
    }

    std::cout << "[PERF] TCPX connection established" << std::endl;

    // ==========================================================================
    // 步骤 4: CUDA 初始化和内存分配
    // ==========================================================================

    CUdevice cuDev;
    CUcontext cuCtx;
    CUdeviceptr d_base, d_aligned;

    // 初始化 CUDA Driver API 并分配 GPU 内存
    if (cuInit(0) != CUDA_SUCCESS ||
        cuDeviceGet(&cuDev, gpu_id) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&cuCtx, cuDev) != CUDA_SUCCESS ||
        cuCtxSetCurrent(cuCtx) != CUDA_SUCCESS ||
        cuMemAlloc(&d_base, kRegisteredBytes + 4096) != CUDA_SUCCESS) {
      std::cerr << "[ERROR] CUDA initialization or allocation failed"
                << std::endl;
      close(bootstrap_fd);
      return 1;
    }

    // 【重要】必须调用 cudaSetDevice，否则 CUDA Runtime API 会使用错误的设备
    if (cudaSetDevice(gpu_id) != cudaSuccess) {
      std::cerr << "[ERROR] cudaSetDevice failed" << std::endl;
      cuMemFree(d_base);
      close(bootstrap_fd);
      return 1;
    }

    // 对齐到 4KB 边界（devmem-tcp 要求）
    uintptr_t addr = static_cast<uintptr_t>(d_base);
    addr = (addr + 4095) & ~static_cast<uintptr_t>(4095);
    d_aligned = static_cast<CUdeviceptr>(addr);
    void* send_buf = reinterpret_cast<void*>(d_aligned);

    // ==========================================================================
    // 步骤 5: 注册发送缓冲区到 TCPX
    // ==========================================================================

    void* send_mhandle = nullptr;
    std::cout << "[PERF][CLIENT] Registering send buffer: ptr=" << send_buf
              << " size=" << kRegisteredBytes << " type=NCCL_PTR_CUDA"
              << std::endl;

    if (tcpx_reg_mr(send_comm, send_buf, kRegisteredBytes, NCCL_PTR_CUDA,
                    &send_mhandle) != 0) {
      std::cerr << "[ERROR] tcpx_reg_mr failed" << std::endl;
      cuMemFree(d_base);
      close(bootstrap_fd);
      return 1;
    }
    std::cout << "[PERF][CLIENT] Send buffer registered successfully, mhandle="
              << send_mhandle << std::endl;

    // ==========================================================================
    // 步骤 6: 配置发送端滑动窗口
    // ==========================================================================

    // 【核心问题】TCPX 插件每个 comm 只有 MAX_REQUESTS=16 个请求槽
    // 如果同时发起超过 16 个 isend，会报错 "unable to allocate requests"
    //
    // 【解决方案】滑动窗口：
    // 1. 最多同时有 MAX_INFLIGHT_SEND (12) 个 send 请求在处理中
    // 2. 当窗口满时，等待最老的 send 完成，再发起新的 isend
    // 3. 使用 12 而不是 16，留一些余量（避免边界情况）
    //
    // 【与 Server 端的区别】
    // - Server: 需要 CUDA events 跟踪 kernel 完成，因为 irecv_consumed 必须在
    // kernel 完成后调用
    // - Client: 不需要 events，因为 send 请求在 tcpx_test 返回 done=1
    // 时自动释放

    constexpr int MAX_INFLIGHT_SEND =
        12;  // 最大并发 send 请求数（< 16，留余量）
    std::vector<void*> pending_send_reqs;  // 待完成的 send 请求
    pending_send_reqs.reserve(MAX_INFLIGHT_SEND);

    std::cout << "[PERF][CLIENT] Using sliding window with MAX_INFLIGHT_SEND="
              << MAX_INFLIGHT_SEND << std::endl;

    // ==========================================================================
    // 步骤 7: 性能测试主循环
    // ==========================================================================

    double total_time_ms = 0.0;

    for (int iter = 0; iter < iterations; ++iter) {
      std::cout << "[PERF] Iteration " << iter << ": total bytes=" << test_size
                << ", chunk_bytes=" << chunk_bytes << std::endl;
      auto start = std::chrono::high_resolution_clock::now();

      // 每次迭代开始时重置滑动窗口
      pending_send_reqs.clear();

      // ========================================================================
      // Chunk 循环：将大消息分成多个 chunk 发送
      // ========================================================================

      size_t offset = 0;
      int chunk_counter = 0;

      while (offset < test_size) {
        // 计算当前 chunk 的大小
        const size_t this_chunk = std::min(chunk_bytes, test_size - offset);

        // 计算源地址（send_buf + offset）
        void* src_ptr = reinterpret_cast<void*>(
            reinterpret_cast<uintptr_t>(send_buf) + offset);

        // 计算 chunk 索引和唯一 tag（必须与 server 端一致）
        const size_t chunk_idx = offset / chunk_bytes;
        int const tag = kTransferTag + static_cast<int>(iter) * 10000 +
                        static_cast<int>(chunk_idx);

        std::cout << "[PERF][CLIENT] chunk_idx=" << chunk_idx << " tag=" << tag
                  << " size=" << this_chunk << " offset=" << offset
                  << std::endl;

        // ======================================================================
        // 滑动窗口：如果窗口满，等待最老的 send 完成
        // ======================================================================

        if (pending_send_reqs.size() >= MAX_INFLIGHT_SEND) {
          void* oldest_req = pending_send_reqs.front();
          int done = 0, sent_size = 0;

          // 【修复】移除超时限制，持续轮询直到发送完成
          // 原因：与 Server 端相同，tcpxTest 本身没有超时机制
          while (!done) {
            tcpx_test(oldest_req, &done, &sent_size);
            if (!done)
              std::this_thread::sleep_for(std::chrono::microseconds(10));
          }

          // 【重要】Send 请求在 tcpx_test 返回 done=1 时自动释放
          // 不需要像 recv 那样调用 irecv_consumed
          pending_send_reqs.erase(pending_send_reqs.begin());
        }

        // ======================================================================
        // 发起异步发送（tcpx_isend）
        // ======================================================================

        void* send_request = nullptr;
        if (tcpx_isend(send_comm, src_ptr, static_cast<int>(this_chunk), tag,
                       send_mhandle, &send_request) != 0) {
          std::cerr << "[ERROR] tcpx_isend failed (chunk " << chunk_idx << ")"
                    << std::endl;
          break;
        }

        // 将当前 send 请求加入滑动窗口
        pending_send_reqs.push_back(send_request);
        offset += this_chunk;
        chunk_counter++;
      }

      // ========================================================================
      // 迭代结束：排空滑动窗口中剩余的 send 请求
      // ========================================================================

      // 【重要】在迭代结束时，滑动窗口中可能还有未完成的 send 请求
      // 必须等待它们全部完成，确保所有数据都已发送

      // [DEBUG] Draining client sliding window (removed for performance)

      while (!pending_send_reqs.empty()) {
        void* oldest_req = pending_send_reqs.front();
        int done = 0, sent_size = 0;

        // 【修复】移除超时限制，持续轮询直到发送完成
        while (!done) {
          tcpx_test(oldest_req, &done, &sent_size);
          if (!done) std::this_thread::sleep_for(std::chrono::microseconds(10));
        }

        // Send 请求自动释放，只需从窗口中移除
        pending_send_reqs.erase(pending_send_reqs.begin());
      }

      // [DEBUG] Client sliding window drained (removed for performance)

      // ========================================================================
      // 计算本次迭代的性能
      // ========================================================================

      auto end = std::chrono::high_resolution_clock::now();
      double iter_time_ms =
          std::chrono::duration<double, std::milli>(end - start).count();
      total_time_ms += iter_time_ms;
      std::cout << "[PERF] Iter " << iter << " time=" << iter_time_ms << "ms"
                << std::endl;
    }  // end of iteration loop

    // ==========================================================================
    // 计算并输出平均性能
    // ==========================================================================

    double avg_ms = total_time_ms / iterations;
    double bw_gbps =
        (test_size / (1024.0 * 1024.0 * 1024.0)) / (avg_ms / 1000.0);

    std::cout << "[PERF] Avg: " << std::fixed << std::setprecision(3) << avg_ms
              << " ms, "
              << "BW: " << std::fixed << std::setprecision(2) << bw_gbps
              << " GB/s" << std::endl;

    // ==========================================================================
    // 清理资源
    // ==========================================================================

    // 注销内存
    tcpx_dereg_mr(send_comm, send_mhandle);

    // 释放 GPU 内存
    cuMemFree(d_base);

    // 释放 CUDA 上下文
    cuDevicePrimaryCtxRelease(cuDev);

    // 关闭 TCPX 连接
    tcpx_close_send(send_comm);

    // 关闭 bootstrap 连接
    close(bootstrap_fd);
  }  // end of client logic

  return 0;
}  // end of main
