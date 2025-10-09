/**
 * @file test_tcpx_perf_multi.cc
 * @brief Multi-Channel TCPX GPU-to-GPU 性能基准测试程序
 *
 * 【程序目标】
 * 测量两个 H100 节点之间通过多个 TCPX channels 进行 GPU-to-GPU 数据传输的性能。
 * 基于 test_tcpx_perf.cc 的逻辑，增加了多 channel 支持以利用多个 NIC (eth1-4)。
 *
 * 【核心设计】
 * - Server: 接收数据，使用 GPU kernel 解包（从分散的 bounce buffer
 * 拷贝到连续的目标 GPU 内存）
 * - Client: 发送数据
 * - 使用 ChannelManager 管理多个 TCPX channels
 * - Round-robin 分配 chunks 到不同的 channels
 * - 每个 channel 独立的滑动窗口机制（避免耗尽 TCPX 请求池）
 *
 * 【使用方法】
 *   # Server 端（10.65.74.150）
 *   UCCL_TCPX_NUM_CHANNELS=4 UCCL_TCPX_PERF_SIZE=67108864
 * ./tests/test_tcpx_perf_multi server 0
 *
 *   # Client 端（10.64.113.77）
 *   UCCL_TCPX_NUM_CHANNELS=4 UCCL_TCPX_PERF_SIZE=67108864
 * ./tests/test_tcpx_perf_multi client 10.65.74.150 0
 *
 * 【环境变量】
 *   UCCL_TCPX_NUM_CHANNELS: Channel 数量（默认 1，推荐 4）
 *   UCCL_TCPX_PERF_SIZE: 每次迭代传输的总字节数（默认 4MB）
 *   UCCL_TCPX_PERF_ITERS: 迭代次数（默认 10）
 *   UCCL_TCPX_CHUNK_BYTES: 每个 chunk 的大小（默认 2MB）
 *   UCCL_TCPX_UNPACK_IMPL: 解包实现方式（kernel|d2d|host，默认 kernel）
 *
 * 【关键血泪经验】（从 test_tcpx_perf.cc 继承）
 * 1. 滑动窗口检查必须在 tcpx_irecv 之前，不是之后！
 * 2. tcpx_irecv_consumed 必须在 kernel 完成后调用（使用 CUDA events 跟踪）
 * 3. Device handle 必须 16 字节对齐
 * 4. Accept 可能返回 nullptr，需要重试
 * 5. 每个 chunk 使用唯一的 tag（避免 TCPX 混淆）
 * 6. 移除超时限制，持续轮询直到完成
 * 7. 在循环外创建 stream 和 launcher（避免每个 chunk 4ms 开销）
 * 8. Client 滑动窗口设为 12（< 16，留余量）
 * 9. Chunk size 保持 512KB（默认值）
 */

/*
======================== 全文件阅读指南（2025-10-08 更新）
========================
本文件是“多进程基线”的权威参考实现。接下来我们将基于它推进：
- 放弃单进程 orchestrator；每个 GPU 的独立进程上，打开 4 条 TCPX 连接（可通过
UCCL_TCPX_NUM_CHANNELS=4 配置）
- 一条 channel ≈ 一条 TCPX 连接；在本测试中 channel 的概念仅作为“连接个数”的载体
- 供应商建议：单 200Gbps NIC + ~8 条连接是 per-NIC 的上限基线（~21.26
GB/s）；不追求 send/recv 对称
- NUMA 建议：每 GPU 绑定到其 NUMA-local 的
NIC；当前可用静态数组硬编码映射（本测试不强制，运行脚本可设置 IFNAME）
- 线程：保持 NCCL 的线程机制；无需应用层 ACK，简单 send/recv 即可

核心语义与约束（务必理解）：
1) 进度与队列顺序
   - 每次 tcpx_test() 内部会驱动
tcpxCommProgress()；需要“持续”调用以保持后台状态机前进
   - 必须严格 FIFO：只对每通道队首请求调用 tcpx_test（对应 TCPX
rq.next_transmitting）
   - rc!=0 在本基线被视为“真错误”（通常意味着不是你的请求或状态非法）；done=0
则表示未完成（非错误） 2) 窗口与上限
   - TCPX 每个 comm 有 MAX_REQUESTS=16 个槽位；我们在 server recv 侧允许 16，在
client send 侧使用 12（留安全余量）
   - 窗口满时：server 调用 wait_for_channel_capacity / client 轮询最老的 send
完成
   - 非阻塞时机：每次 post 后立即调用
process_completed_chunk(blocking=false)，并“顺手”轮询其它通道（opportunistic
drain） 3) Recv 生命周期
   - irecv → tcpx_test(done=1) →（若 kernel 路径则等待 CUDA event）→
tcpx_irecv_consumed()
   - send 生命周期更简单：done=1 自动释放，无 consumed
4) Tag 唯一性
   - 每个 chunk 必须有全局唯一 tag（本文件以迭代号+全局 chunk
序号编码），确保不会交叉匹配 5) 性能计时口径（不对称是正常的）
   - server 吞吐：包含 recv 侧 drain（和可能的 unpack 工作），窗口深度 16；有
consumed；工作量偏重
   - client 吞吐：只计发送流水线，窗口 12，自释放；口径更“轻”
   - 进度节奏、NUMA/NIC 绑定也会造成两端读数差异

代码导读（主要结构）：
- server 路径：
  1. ChannelManager.listen → bootstrap 发送 handles → accept → CUDA init →
注册接收缓冲
  2. 为每通道构建“滑动窗口”与（kernel 模式）持久化 CUDA stream/launcher
  3. 主循环：
     - 按 chunk 划分，round-robin 分配到不同通道
     - post tcpx_irecv 后立即 process_completed_chunk(false) + 顺手轮询其它通道
     - 窗口达到上限时阻塞式释放（等待 kernel event 或完成）
     - 迭代末尾排空 inflight/pending（kernel 模式下）
- client 路径：
  1. bootstrap 接收 handles → 连接所有通道 → CUDA init → 注册发送缓冲
  2. 为每通道创建发送侧滑动窗口（12）
  3. 主循环：
     - 按 chunk 划分并 round-robin 选择通道
     - 窗口满时等待队首 send 完成（tcpx_test 轮询+sleep）
     - post tcpx_isend 并记录 pending，迭代末尾排空

如何将“每 GPU 4 连接”落地：
- 运行时设置环境变量：UCCL_TCPX_NUM_CHANNELS=4（server 与 client 需一致）
- 如需脚本化：在 run 脚本中导出该变量；若需单 NIC 压测，可设置 IFNAME 只暴露一个
NIC

常见陷阱：
- 遗漏“每次 post 后的非阻塞进度驱动” → 第一批请求卡死
- 测队首之外的 request → 触发 TCPX 的 next_transmitting 保护（多为错误）
- Recv 未在 done=1 后调用 consumed → 槽位泄露导致后续阻塞
- 追求 send/recv 对称带宽 → 不必要（口径不同且流水线不同）
===============================================================================
*/

#include "../device/unpack_launch.h"     // GPU kernel 启动器
#include "../include/bootstrap.h"        // Bootstrap protocol
#include "../include/channel_manager.h"  // Multi-channel manager
#include "../include/rx_descriptor.h"    // 接收描述符构建工具
#include "../include/tcpx_interface.h"   // TCPX API 封装
#include "../include/tcpx_structs.h"     // TCPX 内部结构定义
#include <arpa/inet.h>
#include <netinet/in.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iomanip>
#include <iostream>
#include <thread>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
// 供应商要点（2025-10-08）：
// - 一条 channel ≈ 一条 TCPX 连接；本项目不需在 NIXL 里过度强调 channel 概念
// - 每 GPU 绑定其 NUMA-local NIC；当前可静态映射（脚本/环境变量控制 IFNAME）
// - vLLM 模式是一进程一 GPU；每进程仅操作一块 NIC（本测试仍可多连接以压测 NIC）

#include <algorithm>
#include <vector>

namespace {
// ============================================================================
// 常量定义
// ============================================================================

constexpr int kTransferTag = 99;                // 基础传输标签
constexpr size_t kMaxSize = 256 * 1024 * 1024;  // 最大传输大小（256MB）
constexpr size_t kRegisteredBytes = kMaxSize + 4096;  // 注册内存大小

// ============================================================================
// 辅助函数
// ============================================================================

int getEnvInt(char const* name, int def) {
  char const* v = std::getenv(name);
  return v ? std::atoi(v) : def;
}

size_t getEnvSize(char const* name, size_t def) {
  char const* v = std::getenv(name);
  return v ? static_cast<size_t>(std::atoll(v)) : def;
}

// CUDA 错误检查辅助函数
bool cuda_check(CUresult res, char const* msg) {
  if (res != CUDA_SUCCESS) {
    char const* err_str = nullptr;
    cuGetErrorString(res, &err_str);
    std::cerr << "[ERROR] " << msg
              << " failed: " << (err_str ? err_str : "unknown") << std::endl;
    return false;
  }
  return true;
}

bool cuda_check(cudaError_t err, char const* msg) {
  if (err != cudaSuccess) {
    std::cerr << "[ERROR] " << msg << " failed: " << cudaGetErrorString(err)
              << std::endl;
    return false;
  }
  return true;
}

}  // namespace

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char** argv) {
  // ============================================================================
  // 环境变量设置（TCPX 配置）
  // ============================================================================

  // 运行建议（2025-10-08 更新）：
  // - 放弃单进程 orchestrator；本测试作为主路径推进
  // - 推荐配置：UCCL_TCPX_NUM_CHANNELS=2, NCCL_NSOCKS_PERTHREAD=2
  //   * 2 个 channels per GPU
  //   * 每个 channel 2 个 sockets
  //   * 总共：4 sockets per GPU
  //   * 2 个 GPUs 共享 1 个 NIC = 8 sockets per NIC (MAX_SOCKETS=8)
  // - 不必追求 send/recv 对称带宽；关注单向上限与总体规模化行为

  // 【关键】启用 zero-copy（从 4KB 开始使用 devmem-tcp）
  setenv("NCCL_MIN_ZCOPY_SIZE", "4096", 0);
  setenv("NCCL_GPUDIRECTTCPX_MIN_ZCOPY_SIZE", "4096", 0);

  // 【关键】启用接收同步（确保数据完整性）
  setenv("NCCL_GPUDIRECTTCPX_RECV_SYNC", "1", 0);

  // 启用 TCPX wrapper 调试日志
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
    gpu_id = std::atoi(argv[2]);
  } else {
    server_ip = argv[2];
    gpu_id = (argc > 3) ? std::atoi(argv[3]) : 0;
  }

  // ============================================================================
  // 测试参数配置
  // ============================================================================

  // Channel 数量（默认 2）
  int num_channels = getEnvInt("UCCL_TCPX_NUM_CHANNELS", 2);
  int nsocks_per_thread = getEnvInt("NCCL_NSOCKS_PERTHREAD", 2);
  int nthreads = getEnvInt("NCCL_SOCKET_NTHREADS", 1);
  int sockets_per_channel = nsocks_per_thread * nthreads;
  int total_sockets_per_gpu = num_channels * sockets_per_channel;

  std::cout << "[PERF] ========================================" << std::endl;
  std::cout << "[PERF] TCPX Connection Configuration:" << std::endl;
  std::cout << "[PERF]   GPU ID: " << gpu_id << std::endl;
  std::cout << "[PERF]   Channels per GPU: " << num_channels << std::endl;
  std::cout << "[PERF]   Sockets per channel: " << sockets_per_channel << " ("
            << nsocks_per_thread << " × " << nthreads << ")" << std::endl;
  std::cout << "[PERF]   Total sockets per GPU: " << total_sockets_per_gpu
            << std::endl;
  std::cout << "[PERF]   Note: 2 GPUs share 1 NIC → "
            << (total_sockets_per_gpu * 2) << " sockets per NIC (MAX_SOCKETS=8)"
            << std::endl;
  std::cout
      << "[PERF]   Target: Pipeline parallelism via multiple channels & sockets"
      << std::endl;
  std::cout << "[PERF] ========================================" << std::endl;

  // === 路径总览（SERVER）===
  // 步骤 1: listen 并生成 handles（后续 bootstrap 发给 client）
  // 步骤 2: bootstrap：server_send_handles（一次性发完所有通道句柄）
  // 步骤 3: accept 所有通道（生成 recv_comm）
  // 步骤 4: 选择 unpack 实现（kernel/d2d/host）
  // 步骤 5: CUDA 初始化 + 分配 4KB 对齐缓冲（devmem-tcp 要求）
  // 步骤 6: 将接收缓冲注册到所有通道
  // 步骤 7:（kernel 模式）创建持久化 stream/launcher，预建 events
  // 步骤 8: 为每通道创建滑动窗口（MAX_INFLIGHT_PER_CHANNEL=16）
  // 步骤 9: 循环按 chunk post irecv，post 后立刻 process_completed_chunk(false)
  //         并顺手轮询其它通道；窗口满时阻塞等待 capacity（kernel: 等 event；非
  //         kernel: 轮询 done）
  // 步骤 10: 迭代尾部排空 inflight（以及 kernel 模式下 pending）并统计性能

  // 每次迭代传输的总字节数（默认 4MB）
  size_t test_size = getEnvSize("UCCL_TCPX_PERF_SIZE", 4 * 1024 * 1024);

  // 迭代次数
  int iterations = getEnvInt("UCCL_TCPX_PERF_ITERS", 10);

  // Chunk 大小：默认 512KB
  size_t chunk_bytes =
      getEnvSize("UCCL_TCPX_CHUNK_BYTES",
                 getEnvSize("NCCL_P2P_NET_CHUNKSIZE", 512 * 1024));

  // 限制最大传输大小
  if (test_size > kMaxSize) test_size = kMaxSize;

  std::cout << "[PERF] Mode: " << (is_server ? "SERVER" : "CLIENT")
            << std::endl;
  std::cout << "[PERF] GPU: " << gpu_id << std::endl;
  std::cout << "[PERF] Channels: " << num_channels << std::endl;
  std::cout << "[PERF] Size: " << (test_size / 1024 / 1024) << " MB"
            << std::endl;
  std::cout << "[PERF] Iterations: " << iterations << std::endl;
  std::cout << "[PERF] Chunk size: " << (chunk_bytes / 1024 / 1024) << " MB"
            << std::endl;

  // ============================================================================
  // TCPX 插件初始化
  // ============================================================================

  int ndev = tcpx_get_device_count();
  if (ndev <= 0) {
    std::cerr << "[ERROR] No TCPX net devices available" << std::endl;
    return 1;
  }

  int cuda_device_count = 0;
  cudaError_t cuda_count_err = cudaGetDeviceCount(&cuda_device_count);
  if (cuda_count_err != cudaSuccess) {
    std::cerr << "[ERROR] Unable to query CUDA device count: "
              << cudaGetErrorString(cuda_count_err) << std::endl;
    return 1;
  }

  if (gpu_id < 0 || gpu_id >= cuda_device_count) {
    std::cerr << "[ERROR] Invalid GPU id " << gpu_id
              << " (available GPUs: " << cuda_device_count << ")" << std::endl;
    return 1;
  }

  std::cout << "[PERF] TCPX devices: " << ndev << std::endl;
  std::cout << "[PERF] CUDA devices: " << cuda_device_count << std::endl;

  int bootstrap_port_base =
      getEnvInt("UCCL_TCPX_BOOTSTRAP_PORT_BASE", ::kBootstrapPort);
  int bootstrap_port = bootstrap_port_base + gpu_id;
  if (bootstrap_port <= 0 || bootstrap_port >= 65535) {
    std::cerr << "[ERROR] Computed bootstrap port " << bootstrap_port
              << " out of range (base=" << bootstrap_port_base << ")"
              << std::endl;
    return 1;
  }
  std::cout << "[PERF] Bootstrap port: " << bootstrap_port << std::endl;

  // ============================================================================
  // SERVER 端逻辑
  // ============================================================================

  if (is_server) {
    std::cout << "[PERF] Starting SERVER mode" << std::endl;

    // ==========================================================================
    // 步骤 1: 创建 ChannelManager 并 listen
    // ==========================================================================

    ChannelManager mgr(num_channels, gpu_id);
    std::vector<ncclNetHandle_v7> handles;

    if (mgr.server_listen_all(handles) != 0) {
      std::cerr << "[ERROR] server_listen_all failed" << std::endl;
      return 1;
    }

    // CRITICAL: Update num_channels to actual count after clamping
    num_channels = mgr.get_num_channels();
    std::cout << "[PERF] Listening on " << num_channels
              << " channels (after clamping to available TCPX devices)"
              << std::endl;

    // ==========================================================================
    // 步骤 2: Bootstrap 握手（发送 handles 给 client）
    // ==========================================================================

    int bootstrap_fd = -1;
    if (bootstrap_server_create(bootstrap_port, &bootstrap_fd) != 0) {
      std::cerr << "[ERROR] bootstrap_server_create failed" << std::endl;
      return 1;
    }

    std::cout << "[PERF] Bootstrap connection established" << std::endl;

    if (bootstrap_server_send_handles(bootstrap_fd, handles) != 0) {
      std::cerr << "[ERROR] bootstrap_server_send_handles failed" << std::endl;
      close(bootstrap_fd);
      return 1;
    }

    std::cout << "[PERF] Sent " << handles.size() << " handles to client"
              << std::endl;

    // ==========================================================================
    // 步骤 3: Accept 所有 channels
    // ==========================================================================

    if (mgr.server_accept_all() != 0) {
      std::cerr << "[ERROR] server_accept_all failed" << std::endl;
      close(bootstrap_fd);
      return 1;
    }

    std::cout << "[PERF] All " << num_channels << " channels accepted"
              << std::endl;

    // ==========================================================================
    // 步骤 4: 选择 Unpack 实现方式
    // ==========================================================================

    char const* impl_env = std::getenv("UCCL_TCPX_UNPACK_IMPL");
    std::string impl = impl_env ? std::string(impl_env) : std::string("kernel");
    std::transform(impl.begin(), impl.end(), impl.begin(), ::tolower);
    std::cout << "[PERF] Unpack impl: " << impl << std::endl;

    // ==========================================================================
    // 步骤 5: CUDA 初始化
    // ==========================================================================

    CUdevice cuDev;
    CUcontext cuCtx;

    if (!cuda_check(cuInit(0), "cuInit") ||
        !cuda_check(cuDeviceGet(&cuDev, gpu_id), "cuDeviceGet") ||
        !cuda_check(cuDevicePrimaryCtxRetain(&cuCtx, cuDev),
                    "cuDevicePrimaryCtxRetain") ||
        !cuda_check(cuCtxSetCurrent(cuCtx), "cuCtxSetCurrent") ||
        !cuda_check(cudaSetDevice(gpu_id), "cudaSetDevice")) {
      close(bootstrap_fd);
      return 1;
    }

    // ==========================================================================
    // 步骤 6: 分配接收缓冲区
    // ==========================================================================

    CUdeviceptr d_base = 0, d_aligned = 0;

    if (!cuda_check(cuMemAlloc(&d_base, kRegisteredBytes + 4096),
                    "cuMemAlloc")) {
      close(bootstrap_fd);
      return 1;
    }

    // 【关键】对齐到 4KB 边界（devmem-tcp 要求）
    uintptr_t addr = static_cast<uintptr_t>(d_base);
    addr = (addr + 4095) & ~static_cast<uintptr_t>(4095);
    d_aligned = static_cast<CUdeviceptr>(addr);
    void* recv_buf = reinterpret_cast<void*>(d_aligned);

    std::cout << "[PERF] Allocated recv buffer: " << recv_buf << std::endl;

    // ==========================================================================
    // 步骤 7: 注册内存到所有 channels（共享内存方式）
    // ==========================================================================

    if (mgr.register_memory(recv_buf, kRegisteredBytes, NCCL_PTR_CUDA, true) !=
        0) {
      std::cerr << "[ERROR] register_memory failed" << std::endl;
      cuMemFree(d_base);
      close(bootstrap_fd);
      return 1;
    }

    std::cout << "[PERF] Registered recv buffer with " << num_channels
              << " channels" << std::endl;
    // 进度引擎（关键）：总是测试 FIFO 队首
    // - rc!=0 视为真错（非队首/无效状态）；done=0 表示未完成（非错）
    // - done=1 后读取 unpack 元数据并执行 kernel/d2d/host 路径，最后
    // irecv_consumed()
    // - 该函数在 post
    // 后以非阻塞模式调用；窗口满时则以阻塞模式调用，保证持续推进

    // ==========================================================================
    // 步骤 8: 创建持久化的 Stream 和 Launcher（仅 kernel 模式）
    // ==========================================================================

    // 【关键优化】在循环外创建，避免每个 chunk ~4ms 的创建开销
    cudaStream_t unpack_stream = nullptr;
    tcpx::device::UnpackLauncher* launcher_ptr = nullptr;

    if (impl == "kernel") {
      if (!cuda_check(cudaStreamCreate(&unpack_stream), "cudaStreamCreate")) {
        cuMemFree(d_base);
        mgr.deregister_memory(true);
        mgr.close_all(true);
        close(bootstrap_fd);
        return 1;
      }

      tcpx::device::UnpackLaunchConfig cfg;
      cfg.stream = unpack_stream;
      cfg.enable_profiling = false;
      cfg.use_small_kernel = true;
      launcher_ptr = new tcpx::device::UnpackLauncher(cfg);

      std::cout
          << "[PERF] Created persistent stream and launcher for kernel mode"
          << std::endl;
    }

    // ==========================================================================
    // 步骤 9: 为每个 channel 创建滑动窗口
    // ==========================================================================

    // 【核心问题】TCPX 插件每个 comm 只有 MAX_REQUESTS=16 个请求槽
    // 【解决方案】每个 channel 独立的滑动窗口
    constexpr int MAX_INFLIGHT_PER_CHANNEL = 16;

    // 每个 channel 的滑动窗口状态
    struct PostedChunk {
      void* request = nullptr;
      void* dst_ptr = nullptr;
      size_t bytes = 0;
      size_t offset = 0;
      int tag = 0;
      int global_idx = 0;
    };

    struct ChannelWindow {
      std::vector<cudaEvent_t> events;  // CUDA events（用于跟踪 kernel 完成）
      std::vector<void*>
          pending_reqs;  // Kernel 已提交但尚未 consumed 的 TCPX 请求
      std::vector<int> pending_indices;  // 待完成的 chunk 索引
      int chunk_counter = 0;             // 该 channel 处理的 chunk 计数
      std::deque<PostedChunk> inflight_recvs;  // 已经 post 但尚未解包的 chunk
    };

    std::vector<ChannelWindow> channel_windows(num_channels);

    if (impl == "kernel") {
      // 为每个 channel 预创建 CUDA events
      for (int ch = 0; ch < num_channels; ++ch) {
        channel_windows[ch].events.resize(MAX_INFLIGHT_PER_CHANNEL);
        for (int i = 0; i < MAX_INFLIGHT_PER_CHANNEL; ++i) {
          if (!cuda_check(cudaEventCreate(&channel_windows[ch].events[i]),
                          "cudaEventCreate")) {
            std::cerr << "[ERROR] Failed to create event for channel " << ch
                      << std::endl;
            return 1;
          }
        }
        channel_windows[ch].pending_reqs.reserve(MAX_INFLIGHT_PER_CHANNEL);
        channel_windows[ch].pending_indices.reserve(MAX_INFLIGHT_PER_CHANNEL);
      }
      std::cout << "[PERF] Created " << MAX_INFLIGHT_PER_CHANNEL
                << " events per channel for async kernel mode" << std::endl;
    }

    // PROGRESS CORE (server/recv) — this is the canonical reference we mimic in
    // single-process orchestrator:
    // - Always test only the FIFO head (matches TCPX rq.next_transmitting)
    // - If test_rc != 0: treat as a real error (wrong request or invalid state)
    // - If done == 0: in blocking mode, sleep and keep polling; in
    // non-blocking, return to caller
    // - When done == 1: read unpack metadata, run kernel/memcpy path, then call
    // tcpx_irecv_consumed()

    auto process_completed_chunk = [&](int channel_id, ChannelResources& ch,
                                       ChannelWindow& win,
                                       bool blocking) -> bool {
      const int kSleepMicros = 10;

      while (!win.inflight_recvs.empty()) {
        auto& entry = win.inflight_recvs.front();
        int done = 0;
        int received_size = 0;
        int test_rc = tcpx_test(entry.request, &done, &received_size);

        // Handle errors
        if (test_rc != 0) {
          // Special case: rc=2 (connection closed by peer)
          // This can happen when client finishes all sends and closes the
          // connection while server is still draining the last few chunks.
          if (test_rc == 2) {
            if (done == 1) {
              // Data was received successfully before connection closed - this
              // is OK
              std::cout << "[INFO] Connection closed by peer after chunk "
                        << entry.global_idx << " completed on channel "
                        << channel_id << " (expected at end of transfer)"
                        << std::endl;
              // Continue processing this chunk - data was received successfully
            } else {
              // Connection closed but data not yet complete (done=0)
              // This is a transient state - the data may still be in flight.
              // In blocking mode, continue polling; in non-blocking mode,
              // return to retry later.
              std::cout << "[WARN] Connection closed by peer while chunk "
                        << entry.global_idx << " on channel " << channel_id
                        << " still in progress (done=0, will retry)"
                        << std::endl;
              if (blocking) {
                // Continue polling - data may complete in next iteration
                std::this_thread::sleep_for(
                    std::chrono::microseconds(kSleepMicros));
                continue;
              } else {
                // Non-blocking: return to let caller retry later
                break;
              }
            }
          } else {
            // Other errors (rc != 0 and rc != 2) are real errors
            std::cerr << "[ERROR] tcpx_test failed (rc=" << test_rc
                      << ", done=" << done << ") for channel " << channel_id
                      << " chunk " << entry.global_idx << std::endl;
            return false;
          }
        }

        // If not done yet, handle based on blocking mode
        if (!done) {
          if (blocking) {
            std::this_thread::sleep_for(
                std::chrono::microseconds(kSleepMicros));
            continue;
          }
          break;
        }

        std::cout << "[DEBUG][SERVER] Chunk " << entry.global_idx
                  << " recv completed (received_size=" << received_size << ")"
                  << std::endl;

        auto* rx_req =
            reinterpret_cast<tcpx::plugin::tcpxRequest*>(entry.request);
        auto* dev_handle_struct =
            reinterpret_cast<tcpx::plugin::NcclNetDeviceHandle*>(
                ch.recv_dev_handle);

        if (!rx_req || !dev_handle_struct || !rx_req->unpack_slot.mem ||
            !rx_req->unpack_slot.cnt) {
          std::cerr << "[ERROR] Missing TCPX metadata for unpack (channel "
                    << channel_id << ", chunk " << entry.global_idx << ")"
                    << std::endl;
          return false;
        }

        uint64_t frag_count = *(rx_req->unpack_slot.cnt);
        std::cout << "[DEBUG][SERVER] Chunk " << entry.global_idx << " has "
                  << frag_count << " fragments" << std::endl;

        if (frag_count == 0 || frag_count > MAX_UNPACK_DESCRIPTORS) {
          std::cerr << "[ERROR] Invalid fragment count: " << frag_count
                    << " (cnt_cache=" << rx_req->unpack_slot.cnt_cache << ")"
                    << std::endl;
          return false;
        }

        tcpx::plugin::unpackNetDeviceHandle dev_handle{};
        if (!cuda_check(cuMemcpyDtoH(&dev_handle,
                                     reinterpret_cast<CUdeviceptr>(
                                         dev_handle_struct->handle),
                                     sizeof(dev_handle)),
                        "cuMemcpyDtoH device handle")) {
          return false;
        }

        auto* meta_entries =
            static_cast<tcpx::plugin::loadMeta*>(rx_req->unpack_slot.mem);
        tcpx::rx::UnpackDescriptorBlock desc_block;
        tcpx::rx::buildDescriptorBlock(
            meta_entries, static_cast<uint32_t>(frag_count),
            dev_handle.bounce_buf, entry.dst_ptr, desc_block);
        desc_block.ready_flag = rx_req->unpack_slot.cnt;
        desc_block.ready_threshold = frag_count;

        int lrc = 0;

        if (impl == "kernel") {
          std::cout << "[DEBUG][SERVER] Launching unpack kernel for chunk "
                    << entry.global_idx << " (channel " << channel_id
                    << ", descriptors=" << desc_block.count << ")" << std::endl;

          lrc = launcher_ptr->launch(desc_block);
          if (lrc != 0) {
            std::cerr << "[ERROR] Unpack kernel launch failed: " << lrc
                      << std::endl;
            return false;
          }

          int event_idx = win.chunk_counter % MAX_INFLIGHT_PER_CHANNEL;
          if (!cuda_check(cudaEventRecord(win.events[event_idx], unpack_stream),
                          "cudaEventRecord")) {
            return false;
          }

          win.pending_reqs.push_back(entry.request);
          win.pending_indices.push_back(win.chunk_counter);
          win.chunk_counter++;

        } else if (impl == "d2d") {
          for (uint32_t i = 0; i < desc_block.count; ++i) {
            const auto& meta = desc_block.descriptors[i];
            CUdeviceptr src_ptr = static_cast<CUdeviceptr>(
                reinterpret_cast<uintptr_t>(desc_block.bounce_buffer) +
                meta.src_off);
            CUdeviceptr dst_ptr_frag = static_cast<CUdeviceptr>(
                reinterpret_cast<uintptr_t>(desc_block.dst_buffer) +
                meta.dst_off);

            if (!cuda_check(cuMemcpyDtoD(dst_ptr_frag, src_ptr, meta.len),
                            "cuMemcpyDtoD")) {
              lrc = -1;
              break;
            }
          }
          if (lrc != 0) return false;

          tcpx_irecv_consumed(ch.recv_comm, 1, entry.request);

        } else {  // host gather
          std::vector<unsigned char> tmp(desc_block.total_bytes);
          size_t host_off = 0;
          for (uint32_t i = 0; i < desc_block.count; ++i) {
            const auto& meta = desc_block.descriptors[i];
            CUdeviceptr src_ptr = static_cast<CUdeviceptr>(
                reinterpret_cast<uintptr_t>(desc_block.bounce_buffer) +
                meta.src_off);
            if (!cuda_check(
                    cuMemcpyDtoH(tmp.data() + host_off, src_ptr, meta.len),
                    "cuMemcpyDtoH")) {
              lrc = -1;
              break;
            }
            host_off += meta.len;
          }
          if (lrc != 0) return false;

          if (!cuda_check(cuMemcpyHtoD(static_cast<CUdeviceptr>(
                                           reinterpret_cast<uintptr_t>(
                                               desc_block.dst_buffer)),
                                       tmp.data(), tmp.size()),
                          "cuMemcpyHtoD")) {
            return false;
          }

          tcpx_irecv_consumed(ch.recv_comm, 1, entry.request);
        }

        win.inflight_recvs.pop_front();
      }

      return true;
    };

    auto wait_for_channel_capacity = [&](int channel_id, ChannelResources& ch,
                                         ChannelWindow& win) -> bool {
      while (win.pending_reqs.size() + win.inflight_recvs.size() >=
             MAX_INFLIGHT_PER_CHANNEL) {
        if (!win.pending_reqs.empty()) {
          int oldest_idx = win.pending_indices.front();
          void* oldest_req = win.pending_reqs.front();
          cudaEvent_t oldest_event =
              win.events[oldest_idx % MAX_INFLIGHT_PER_CHANNEL];

          std::cout << "[DEBUG][SERVER] Channel " << channel_id
                    << " sliding window FULL (" << win.pending_reqs.size()
                    << "+" << win.inflight_recvs.size() << "/"
                    << MAX_INFLIGHT_PER_CHANNEL << "), waiting for chunk "
                    << oldest_idx << " kernel to complete..." << std::endl;

          if (!cuda_check(cudaEventSynchronize(oldest_event),
                          "cudaEventSynchronize")) {
            return false;
          }

          tcpx_irecv_consumed(ch.recv_comm, 1, oldest_req);
          win.pending_reqs.erase(win.pending_reqs.begin());
          win.pending_indices.erase(win.pending_indices.begin());

          std::cout << "[DEBUG][SERVER] Channel " << channel_id
                    << " window now has " << win.pending_reqs.size() << "+"
                    << win.inflight_recvs.size() << " outstanding" << std::endl;
          continue;
        }

        if (!win.inflight_recvs.empty()) {
          if (!process_completed_chunk(channel_id, ch, win,
                                       /*blocking=*/true)) {
            return false;
          }
          continue;
        }

        break;
      }
      return true;
    };

    // ==========================================================================
    // 步骤 10: 性能测试主循环
    // ==========================================================================

    double total_time_ms = 0.0;
    int completed_iters = 0;
    bool abort_benchmark = false;

    for (int iter = 0; iter < iterations; ++iter) {
      std::cout << "[PERF] Iteration " << iter << ": total bytes=" << test_size
                << ", chunk_bytes=" << chunk_bytes << std::endl;

      auto start = std::chrono::high_resolution_clock::now();
      bool iteration_failed = false;

      // 每次迭代开始时重置所有 channel 的滑动窗口状态
      if (impl == "kernel") {
        std::cout << "[DEBUG] Iteration " << iter
                  << " start: clearing sliding windows for " << num_channels
                  << " channels" << std::endl;
        for (int ch = 0; ch < num_channels; ++ch) {
          channel_windows[ch].pending_reqs.clear();
          // 将当前 irecv 记录到该通道的 FIFO
          // 后，立刻触发一次非阻塞进度（避免首批请求积压）
          // 同时“顺手”对其它通道做非阻塞进度，缩短 metadata/backlog，降低 HOL
          // 风险

          channel_windows[ch].pending_indices.clear();
          channel_windows[ch].chunk_counter = 0;
        }
      }

      // ========================================================================
      // Chunk 循环：将大消息分成多个 chunk，round-robin 分配到不同 channels
      // ========================================================================

      size_t offset = 0;
      int global_chunk_idx = 0;

      while (offset < test_size) {
        const size_t this_chunk = std::min(chunk_bytes, test_size - offset);
        void* dst_ptr = reinterpret_cast<void*>(
            reinterpret_cast<uintptr_t>(recv_buf) + offset);

        // 【关键】Round-robin 选择 channel
        int channel_id = global_chunk_idx % num_channels;
        ChannelResources& ch = mgr.get_channel(channel_id);
        ChannelWindow& win = channel_windows[channel_id];

        // 【关键】每个 chunk 使用唯一的 tag
        int const tag = kTransferTag + iter * 10000 + global_chunk_idx;

        std::cout << "[DEBUG][SERVER] chunk=" << global_chunk_idx
                  << " channel=" << channel_id << " tag=" << tag
                  << " size=" << this_chunk << " offset=" << offset
                  << " outstanding="
                  << (win.pending_reqs.size() + win.inflight_recvs.size())
                  << "/" << MAX_INFLIGHT_PER_CHANNEL << std::endl;

        if (impl == "kernel") {
          if (!wait_for_channel_capacity(channel_id, ch, win)) {
            std::cerr << "[ERROR] Failed while waiting for channel capacity"
                      << std::endl;
            iteration_failed = true;
            break;
          }
        } else {
          while (win.inflight_recvs.size() >= MAX_INFLIGHT_PER_CHANNEL) {
            if (!process_completed_chunk(channel_id, ch, win,
                                         /*blocking=*/true)) {
              std::cerr
                  << "[ERROR] Failed while waiting for inflight recv to drain"
                  << std::endl;
              iteration_failed = true;
              break;
            }
          }
          if (iteration_failed) break;
        }

        // ======================================================================
        // 发起异步接收（tcpx_irecv）
        // ======================================================================

        void* recv_data[1] = {dst_ptr};
        int recv_sizes[1] = {static_cast<int>(this_chunk)};
        int recv_tags[1] = {tag};
        void* recv_mhandles[1] = {ch.mhandle};
        void* recv_request = nullptr;

        std::cout << "[DEBUG][SERVER] Calling tcpx_irecv for chunk "
                  << global_chunk_idx << " on channel " << channel_id
                  << std::endl;

        int irecv_rc = tcpx_irecv(ch.recv_comm, 1, recv_data, recv_sizes,
                                  recv_tags, recv_mhandles, &recv_request);
        if (irecv_rc != 0) {
          std::cerr << "[ERROR] tcpx_irecv failed: rc=" << irecv_rc
                    << " chunk=" << global_chunk_idx
                    << " channel=" << channel_id << std::endl;
          iteration_failed = true;
          break;
        }

        PostedChunk posted;
        posted.request = recv_request;
        posted.dst_ptr = dst_ptr;
        posted.bytes = this_chunk;
        posted.offset = offset;
        posted.tag = tag;
        posted.global_idx = global_chunk_idx;
        // Immediately record the posted recv into per-channel FIFO; this
        // ensures the head we poll is always the oldest request, matching TCPX
        // expectations.

        win.inflight_recvs.push_back(std::move(posted));

        bool ok =
            process_completed_chunk(channel_id, ch, win, /*blocking=*/false);
        if (!ok) {
          std::cerr << "[ERROR] Failed to process completed chunks"
                    << std::endl;
          iteration_failed = true;
          // Opportunistic drain: after each post, try to progress other
          // channels in non-blocking mode to keep TCPX queues moving and reduce
          // HOL.
          break;
        }

        // Opportunistically drain other channels to keep metadata queues short
        for (int other = 0; other < num_channels; ++other) {
          if (other == channel_id) continue;
          ChannelWindow& other_win = channel_windows[other];
          if (other_win.inflight_recvs.empty()) continue;
          ChannelResources& other_ch = mgr.get_channel(other);
          if (!process_completed_chunk(other, other_ch, other_win,
                                       /*blocking=*/false)) {
            std::cerr
                << "[ERROR] Failed to process completed chunks for channel "
                << other << std::endl;
            ok = false;
            break;
          }
        }

        if (!ok) {
          iteration_failed = true;
          break;
        }

        offset += this_chunk;
        global_chunk_idx++;
      }  // end of chunk loop

      if (iteration_failed) {
        abort_benchmark = true;
      }

      // ========================================================================
      // 迭代结束：排空所有 channels 的滑动窗口（仅 kernel 模式）
      // ========================================================================

      if (impl == "kernel") {
        for (int ch = 0; ch < num_channels; ++ch) {
          ChannelResources& channel = mgr.get_channel(ch);
          ChannelWindow& win = channel_windows[ch];
          while (!win.inflight_recvs.empty()) {
            if (!process_completed_chunk(ch, channel, win, /*blocking=*/true)) {
              std::cerr << "[ERROR] Failed to drain inflight recvs for channel "
                        << ch << std::endl;
              abort_benchmark = true;
              break;
            }
          }
          if (abort_benchmark) break;
        }

        if (!abort_benchmark) {
          for (int ch = 0; ch < num_channels; ++ch) {
            ChannelResources& channel = mgr.get_channel(ch);
            ChannelWindow& win = channel_windows[ch];

            while (!win.pending_reqs.empty()) {
              int oldest_idx = win.pending_indices.front();
              cudaEvent_t oldest_event =
                  win.events[oldest_idx % MAX_INFLIGHT_PER_CHANNEL];

              if (!cuda_check(cudaEventSynchronize(oldest_event),
                              "cudaEventSynchronize drain")) {
                abort_benchmark = true;
                break;
              }

              void* oldest_req = win.pending_reqs.front();
              tcpx_irecv_consumed(channel.recv_comm, 1, oldest_req);

              win.pending_reqs.erase(win.pending_reqs.begin());
              win.pending_indices.erase(win.pending_indices.begin());
            }

            // === 路径总览（CLIENT）===
            // 步骤 1: bootstrap 连接并接收 handles（获取通道数）→ 以
            // handles.size() 构造 ChannelManager 步骤 2: 连接所有通道（生成
            // send_comm） 步骤 3: CUDA 初始化 + 分配 4KB 对齐发送缓冲 +
            // 注册到所有通道 步骤 4:
            // 为每通道创建发送侧窗口（MAX_INFLIGHT_SEND_PER_CHANNEL=12） 步骤
            // 5: 主循环按 chunk 发送；窗口满时阻塞等待最老 send 完成（tcpx_test
            // + sleep） 步骤 6: 迭代尾部排空 pending 发送请求并统计平均性能

            if (abort_benchmark) break;
          }
        }
      }

      // ========================================================================
      // 计算本次迭代的性能
      // ========================================================================

      auto end = std::chrono::high_resolution_clock::now();
      double iter_time_ms =
          std::chrono::duration<double, std::milli>(end - start).count();
      if (!iteration_failed && !abort_benchmark) {
        total_time_ms += iter_time_ms;
        ++completed_iters;
        std::cout << "[PERF] Iter " << iter << " time=" << iter_time_ms << "ms"
                  << std::endl;
      } else {
        std::cerr << "[ERROR] Iter " << iter << " aborted after "
                  << iter_time_ms << "ms" << std::endl;
      }

      if (abort_benchmark) {
        std::cerr << "[ERROR] Aborting benchmark due to previous errors"
                  << std::endl;
        break;
      }
    }  // end of iteration loop

    // ==========================================================================
    // NOTE on measurement asymmetry (server vs client):
    // - Server average includes recv-side drain (plus optional unpack path) and
    //   uses MAX_INFLIGHT_PER_CHANNEL=16; client send uses 12 by design.
    // - Send requests auto-release on completion; recv requires
    // tcpx_irecv_consumed()
    //   after done=1. That lifecycle difference shifts where overhead is paid.
    // - Progress cadence and NUMA/NIC mapping can also skew the two readings.
    // - Per vendor guidance (2025-10-08): measuring per-NIC max should use one
    //   200Gbps NIC with ~8 TCPX connections; do not expect symmetric GB/s.

    // 计算并输出平均性能
    // ==========================================================================

    if (completed_iters > 0) {
      double avg_ms = total_time_ms / completed_iters;
      double bw_gbps =
          (test_size / (1024.0 * 1024.0 * 1024.0)) / (avg_ms / 1000.0);

      std::cout << "[PERF] Avg (" << completed_iters << " iter): " << std::fixed
                << std::setprecision(3) << avg_ms << " ms, "
                << "BW: " << std::fixed << std::setprecision(2) << bw_gbps
                << " GB/s" << std::endl;
    } else {
      std::cerr << "[ERROR] No successful iterations recorded" << std::endl;
    }

    // ==========================================================================
    // 清理资源
    // ==========================================================================

    // 清理 launcher 和 stream
    if (launcher_ptr) {
      delete launcher_ptr;
      launcher_ptr = nullptr;
    }
    if (unpack_stream) {
      cudaStreamDestroy(unpack_stream);
      unpack_stream = nullptr;
    }

    // 清理 CUDA events
    if (impl == "kernel") {
      for (int ch = 0; ch < num_channels; ++ch) {
        for (auto& evt : channel_windows[ch].events) {
          cudaEventDestroy(evt);
        }
      }
    }

    // 清理 TCPX 资源
    mgr.deregister_memory(true);
    cuMemFree(d_base);
    cuDevicePrimaryCtxRelease(cuDev);
    mgr.close_all(true);
    close(bootstrap_fd);

    // ============================================================================
    // CLIENT 端逻辑
    // ============================================================================

  } else {
    std::cout << "[PERF] Starting CLIENT mode" << std::endl;

    // ==========================================================================
    // 步骤 1: Bootstrap 连接（接收 handles）
    // ==========================================================================

    int bootstrap_fd = -1;
    if (bootstrap_client_connect(server_ip.c_str(), bootstrap_port,
                                 &bootstrap_fd) != 0) {
      std::cerr << "[ERROR] bootstrap_client_connect failed" << std::endl;
      return 1;
    }

    std::cout << "[PERF] Bootstrap connected to " << server_ip << std::endl;

    std::vector<ncclNetHandle_v7> handles;
    if (bootstrap_client_recv_handles(bootstrap_fd, handles) != 0) {
      std::cerr << "[ERROR] bootstrap_client_recv_handles failed" << std::endl;
      close(bootstrap_fd);
      return 1;
    }

    std::cout << "[PERF] Received " << handles.size() << " handles from server"
              << std::endl;

    // ==========================================================================
    // 步骤 2: 创建 ChannelManager 并连接所有 channels
    // ==========================================================================

    // CRITICAL: Use handles.size() instead of env value to match server's
    // actual channel count
    ChannelManager mgr(static_cast<int>(handles.size()), gpu_id);

    if (mgr.client_connect_all(handles) != 0) {
      std::cerr << "[ERROR] client_connect_all failed" << std::endl;
      close(bootstrap_fd);
      return 1;
    }

    // CRITICAL: Update num_channels to actual count (must match server)
    num_channels = mgr.get_num_channels();
    std::cout << "[PERF] All " << num_channels
              << " channels connected (matched to server's count)" << std::endl;

    // ==========================================================================
    // 步骤 3: CUDA 初始化和内存分配
    // ==========================================================================

    CUdevice cuDev;
    CUcontext cuCtx;
    CUdeviceptr d_base, d_aligned;

    if (!cuda_check(cuInit(0), "cuInit") ||
        // send 窗口满：等待队首发送完成（tcpx_test + sleep），避免占满 TCPX
        // 请求槽 注意：send 完成后自动释放，仅需从窗口 vector 中移除指针

        !cuda_check(cuDeviceGet(&cuDev, gpu_id), "cuDeviceGet") ||
        !cuda_check(cuDevicePrimaryCtxRetain(&cuCtx, cuDev),
                    "cuDevicePrimaryCtxRetain") ||
        !cuda_check(cuCtxSetCurrent(cuCtx), "cuCtxSetCurrent") ||
        !cuda_check(cuMemAlloc(&d_base, kRegisteredBytes + 4096),
                    "cuMemAlloc") ||
        !cuda_check(cudaSetDevice(gpu_id), "cudaSetDevice")) {
      close(bootstrap_fd);
      return 1;
    }

    // 对齐到 4KB 边界
    uintptr_t addr = static_cast<uintptr_t>(d_base);
    addr = (addr + 4095) & ~static_cast<uintptr_t>(4095);
    d_aligned = static_cast<CUdeviceptr>(addr);
    void* send_buf = reinterpret_cast<void*>(d_aligned);

    std::cout << "[PERF] Allocated send buffer: " << send_buf << std::endl;

    // ==========================================================================
    // 步骤 4: 注册发送缓冲区到所有 channels（共享内存方式）
    // ==========================================================================

    if (mgr.register_memory(send_buf, kRegisteredBytes, NCCL_PTR_CUDA, false) !=
        0) {
      std::cerr << "[ERROR] register_memory failed" << std::endl;
      cuMemFree(d_base);
      close(bootstrap_fd);
      return 1;
    }

    std::cout << "[PERF] Registered send buffer with " << num_channels
              << " channels" << std::endl;

    // ==========================================================================
    // 步骤 5: 配置发送端滑动窗口（每个 channel 独立）
    // ==========================================================================

    // 【关键】Client 使用 12 而不是 16，留余量避免边界情况
    constexpr int MAX_INFLIGHT_SEND_PER_CHANNEL = 12;

    struct SendChannelWindow {
      std::vector<void*> pending_reqs;
      int chunk_counter = 0;
    };

    std::vector<SendChannelWindow> send_windows(num_channels);
    for (int ch = 0; ch < num_channels; ++ch) {
      send_windows[ch].pending_reqs.reserve(MAX_INFLIGHT_SEND_PER_CHANNEL);
    }

    std::cout << "[PERF] Using sliding window with MAX_INFLIGHT_SEND="
              << MAX_INFLIGHT_SEND_PER_CHANNEL << " per channel" << std::endl;

    // ==========================================================================
    // 步骤 6: 性能测试主循环
    // ==========================================================================

    double total_time_ms = 0.0;

    for (int iter = 0; iter < iterations; ++iter) {
      std::cout << "[PERF] Iteration " << iter << ": total bytes="
                << test_size
                // 迭代结束：对每个通道排空 pending 发送请求；此处只需轮询 done
                // 并从窗口移除（无 consumed）

                << ", chunk_bytes=" << chunk_bytes << std::endl;

      auto start = std::chrono::high_resolution_clock::now();

      // 每次迭代开始时重置所有 channel 的滑动窗口
      std::cout << "[DEBUG] Iteration " << iter
                << " start: clearing send windows for " << num_channels
                << " channels" << std::endl;
      for (int ch = 0; ch < num_channels; ++ch) {
        send_windows[ch].pending_reqs.clear();
        send_windows[ch].chunk_counter = 0;
      }

      // ========================================================================
      // Chunk 循环：将大消息分成多个 chunk，round-robin 分配到不同 channels
      // ========================================================================

      size_t offset = 0;
      int global_chunk_idx = 0;

      while (offset < test_size) {
        const size_t this_chunk = std::min(chunk_bytes, test_size - offset);
        void* src_ptr = reinterpret_cast<void*>(
            reinterpret_cast<uintptr_t>(send_buf) + offset);

        // 【关键】Round-robin 选择 channel
        int channel_id = global_chunk_idx % num_channels;
        ChannelResources& ch = mgr.get_channel(channel_id);
        SendChannelWindow& win = send_windows[channel_id];

        // 【关键】每个 chunk 使用唯一的 tag（必须与 server 端一致）
        int const tag = kTransferTag + iter * 10000 + global_chunk_idx;

        std::cout << "[DEBUG][CLIENT] chunk=" << global_chunk_idx
                  << " channel=" << channel_id << " tag=" << tag
                  << " size=" << this_chunk << " offset=" << offset
                  << " pending=" << win.pending_reqs.size() << "/"
                  << MAX_INFLIGHT_SEND_PER_CHANNEL << std::endl;

        // ======================================================================
        // 滑动窗口：如果该 channel 的窗口满，等待最老的 send 完成
        // ======================================================================

        if (win.pending_reqs.size() >= MAX_INFLIGHT_SEND_PER_CHANNEL) {
          std::cout << "[DEBUG][CLIENT] Channel " << channel_id
                    << " send window FULL (" << win.pending_reqs.size() << "/"
                    << MAX_INFLIGHT_SEND_PER_CHANNEL
                    << "), waiting for oldest send" << std::endl;

          void* oldest_req = win.pending_reqs.front();
          int done = 0, sent_size = 0;

          // 【修复】移除超时限制，持续轮询直到发送完成
          int poll_count = 0;
          while (!done) {
            tcpx_test(oldest_req, &done, &sent_size);
            if (!done) {
              std::this_thread::sleep_for(std::chrono::microseconds(10));
              poll_count++;
              if (poll_count % 1000 == 0) {
                std::cout << "[DEBUG][CLIENT] Still waiting for oldest send "
                             "(poll_count="
                          << poll_count << ")" << std::endl;
              }
            }
          }

          std::cout << "[DEBUG][CLIENT] Oldest send completed after "
                    << poll_count << " polls, sent_size=" << sent_size
                    << std::endl;

          // Send 请求自动释放，只需从窗口中移除
          win.pending_reqs.erase(win.pending_reqs.begin());

          std::cout << "[DEBUG][CLIENT] Channel " << channel_id
                    << " window now has " << win.pending_reqs.size()
                    << " pending sends" << std::endl;
        }

        // ======================================================================
        // 发起异步发送（tcpx_isend）
        // ======================================================================

        std::cout << "[DEBUG][CLIENT] Calling tcpx_isend for chunk "
                  << global_chunk_idx << " on channel " << channel_id
                  << std::endl;

        void* send_request = nullptr;
        if (tcpx_isend(ch.send_comm, src_ptr, static_cast<int>(this_chunk), tag,
                       ch.mhandle, &send_request) != 0) {
          std::cerr << "[ERROR] tcpx_isend failed (chunk " << global_chunk_idx
                    << " channel " << channel_id << ")" << std::endl;
          break;
        }

        std::cout << "[DEBUG][CLIENT] tcpx_isend returned, request="
                  << send_request << std::endl;

        // 将当前 send 请求加入该 channel 的滑动窗口
        win.pending_reqs.push_back(send_request);
        win.chunk_counter++;

        std::cout << "[DEBUG][CLIENT] Chunk " << global_chunk_idx
                  << " added to window (counter=" << win.chunk_counter
                  << ", pending=" << win.pending_reqs.size() << ")"
                  << std::endl;

        offset += this_chunk;
        global_chunk_idx++;
      }

      // ========================================================================
      // 迭代结束：排空所有 channels 的滑动窗口
      // ========================================================================

      std::cout << "[DEBUG] Iteration " << iter
                << " end: draining send windows for " << num_channels
                << " channels" << std::endl;

      for (int ch = 0; ch < num_channels; ++ch) {
        SendChannelWindow& win = send_windows[ch];

        std::cout << "[DEBUG][CLIENT] Draining channel " << ch << " ("
                  << win.pending_reqs.size() << " pending sends)" << std::endl;

        while (!win.pending_reqs.empty()) {
          void* oldest_req = win.pending_reqs.front();
          int done = 0, sent_size = 0;

          while (!done) {
            tcpx_test(oldest_req, &done, &sent_size);
            if (!done)
              std::this_thread::sleep_for(std::chrono::microseconds(10));
          }

          win.pending_reqs.erase(win.pending_reqs.begin());
        }

        std::cout << "[DEBUG][CLIENT] Channel " << ch << " drained"
                  << std::endl;
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

    mgr.deregister_memory(false);
    cuMemFree(d_base);
    cuDevicePrimaryCtxRelease(cuDev);
    mgr.close_all(false);
    close(bootstrap_fd);
  }  // end of client logic

  return 0;
}  // end of main