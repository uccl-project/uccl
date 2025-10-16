/**
 * @file tcpx_perf_runner.cc
 * @brief Implementation of TCPX performance benchmark runner
 */

#include "tcpx_perf_runner.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>

#include "../include/bootstrap.h"
#include "../include/tcpx_session.h"
#include "../include/tcpx_transfer.h"
#include "../include/tcpx_interface.h"

namespace tcpx {

constexpr int kTransferTag = 99;

// ============================================================================
// Utility Functions
// ============================================================================

int getEnvInt(const char* name, int def) {
  const char* v = std::getenv(name);
  return v ? std::atoi(v) : def;
}

size_t getEnvSize(const char* name, size_t def) {
  const char* v = std::getenv(name);
  return v ? static_cast<size_t>(std::atoll(v)) : def;
}

void printPerfStats(const PerfStats& stats, const PerfConfig& config) {
  std::cout << "\n";
  std::cout << "========================================\n";
  std::cout << "TCPX Performance Benchmark Results\n";
  std::cout << "========================================\n";
  std::cout << "Mode:           " << (config.is_server ? "SERVER" : "CLIENT") << "\n";
  std::cout << "GPU ID:         " << config.gpu_id << "\n";
  std::cout << "Channels:       " << stats.num_channels << "\n";
  std::cout << "Iterations:     " << stats.iterations << "\n";
  std::cout << "Total bytes:    " << (stats.total_bytes / 1024 / 1024) << " MB\n";
  std::cout << "Total time:     " << std::fixed << std::setprecision(2) 
            << stats.total_time_ms << " ms\n";
  std::cout << "Bandwidth:      " << std::fixed << std::setprecision(2) 
            << stats.bandwidth_gbps << " GB/s\n";
  std::cout << "========================================\n";
  std::cout << std::endl;
}

// ============================================================================
// PerfRunner::Impl
// ============================================================================

struct PerfRunner::Impl {
  // Configuration
  PerfConfig config;

  // CUDA resources
  CUdevice cu_device;
  CUcontext cu_context;
  CUdeviceptr d_base;
  CUdeviceptr d_aligned;
  void* buffer;

  // TCPX resources
  TcpxSession* session;
  int bootstrap_fd;

  // Memory registration
  uint64_t mem_id;
  static constexpr size_t kMaxSize = 256 * 1024 * 1024;
  static constexpr size_t kRegisteredBytes = kMaxSize + 4096;

  // State
  bool initialized;

  Impl(const PerfConfig& cfg)
      : config(cfg),
        cu_device(0),
        cu_context(nullptr),
        d_base(0),
        d_aligned(0),
        buffer(nullptr),
        session(nullptr),
        bootstrap_fd(-1),
        mem_id(1),
        initialized(false) {}

  ~Impl() {
    cleanup();
  }

  void cleanup() {
    if (session) {
      delete session;
      session = nullptr;
    }

    if (d_base) {
      cuMemFree(d_base);
      d_base = 0;
      d_aligned = 0;
      buffer = nullptr;
    }

    if (cu_context) {
      cuCtxSetCurrent(nullptr);
      cuDevicePrimaryCtxRelease(cu_device);
      cu_context = nullptr;
    }

    if (bootstrap_fd >= 0) {
      close(bootstrap_fd);
      bootstrap_fd = -1;
    }

    initialized = false;
  }

  int initCuda() {
    // Initialize CUDA
    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
      const char* err_str = nullptr;
      cuGetErrorString(res, &err_str);
      std::cerr << "[ERROR] cuInit failed: " << (err_str ? err_str : "unknown") << std::endl;
      return -1;
    }

    // Get device
    res = cuDeviceGet(&cu_device, config.gpu_id);
    if (res != CUDA_SUCCESS) {
      const char* err_str = nullptr;
      cuGetErrorString(res, &err_str);
      std::cerr << "[ERROR] cuDeviceGet failed: " << (err_str ? err_str : "unknown") << std::endl;
      return -1;
    }

    // Get context
    res = cuDevicePrimaryCtxRetain(&cu_context, cu_device);
    if (res != CUDA_SUCCESS) {
      const char* err_str = nullptr;
      cuGetErrorString(res, &err_str);
      std::cerr << "[ERROR] cuDevicePrimaryCtxRetain failed: " << (err_str ? err_str : "unknown") << std::endl;
      return -1;
    }

    // Set context
    res = cuCtxSetCurrent(cu_context);
    if (res != CUDA_SUCCESS) {
      const char* err_str = nullptr;
      cuGetErrorString(res, &err_str);
      std::cerr << "[ERROR] cuCtxSetCurrent failed: " << (err_str ? err_str : "unknown") << std::endl;
      return -1;
    }

    // Set device
    cudaError_t cuda_err = cudaSetDevice(config.gpu_id);
    if (cuda_err != cudaSuccess) {
      std::cerr << "[ERROR] cudaSetDevice failed: " << cudaGetErrorString(cuda_err) << std::endl;
      return -1;
    }

    std::cout << "[PERF] CUDA initialized (GPU " << config.gpu_id << ")" << std::endl;
    return 0;
  }

  int allocateMemory() {
    // Allocate buffer with extra space for alignment
    CUresult res = cuMemAlloc(&d_base, kRegisteredBytes + 4096);
    if (res != CUDA_SUCCESS) {
      const char* err_str = nullptr;
      cuGetErrorString(res, &err_str);
      std::cerr << "[ERROR] cuMemAlloc failed: " << (err_str ? err_str : "unknown") << std::endl;
      return -1;
    }

    // Align to 4KB boundary (devmem-tcp requirement)
    uintptr_t addr = static_cast<uintptr_t>(d_base);
    addr = (addr + 4095) & ~static_cast<uintptr_t>(4095);
    d_aligned = static_cast<CUdeviceptr>(addr);
    buffer = reinterpret_cast<void*>(d_aligned);

    std::cout << "[PERF] Allocated buffer: " << buffer 
              << " (" << (kRegisteredBytes / 1024 / 1024) << " MB)" << std::endl;
    return 0;
  }

  int bootstrapServer() {
    // Create bootstrap server
    if (bootstrap_server_create(config.bootstrap_port, &bootstrap_fd) != 0) {
      std::cerr << "[ERROR] bootstrap_server_create failed" << std::endl;
      return -1;
    }

    std::cout << "[PERF] Bootstrap server listening on port " << config.bootstrap_port << std::endl;

    // Get connection info (JSON string with handles)
    std::string conn_info = session->listen();
    if (conn_info.empty()) {
      std::cerr << "[ERROR] session->listen failed" << std::endl;
      return -1;
    }

    // Parse handles from JSON (simplified - just send the string)
    // In real implementation, we'd parse JSON and extract handles
    // For now, we'll use a simple approach: send conn_info length + conn_info
    uint32_t info_len = static_cast<uint32_t>(conn_info.size());
    if (write(bootstrap_fd, &info_len, sizeof(info_len)) != sizeof(info_len)) {
      std::cerr << "[ERROR] Failed to send conn_info length" << std::endl;
      return -1;
    }
    if (write(bootstrap_fd, conn_info.data(), info_len) != static_cast<ssize_t>(info_len)) {
      std::cerr << "[ERROR] Failed to send conn_info" << std::endl;
      return -1;
    }

    std::cout << "[PERF] Sent connection info to client (" << info_len << " bytes)" << std::endl;
    return 0;
  }

  int bootstrapClient() {
    // Connect to server
    if (bootstrap_client_connect(config.server_ip.c_str(), config.bootstrap_port, &bootstrap_fd) != 0) {
      std::cerr << "[ERROR] bootstrap_client_connect failed" << std::endl;
      return -1;
    }

    std::cout << "[PERF] Bootstrap client connected to " << config.server_ip
              << ":" << config.bootstrap_port << std::endl;

    // Receive connection info from server
    uint32_t info_len = 0;
    if (read(bootstrap_fd, &info_len, sizeof(info_len)) != sizeof(info_len)) {
      std::cerr << "[ERROR] Failed to receive conn_info length" << std::endl;
      return -1;
    }

    std::vector<char> buffer(info_len);
    if (read(bootstrap_fd, buffer.data(), info_len) != static_cast<ssize_t>(info_len)) {
      std::cerr << "[ERROR] Failed to receive conn_info" << std::endl;
      return -1;
    }

    std::string conn_info(buffer.begin(), buffer.end());
    std::cout << "[PERF] Received connection info from server (" << info_len << " bytes)" << std::endl;

    // Load remote connection info
    std::string remote_name = "server";
    if (session->loadRemoteConnInfo(remote_name, conn_info) != 0) {
      std::cerr << "[ERROR] loadRemoteConnInfo failed" << std::endl;
      return -1;
    }

    // Connect to all channels
    if (session->connect(remote_name) != 0) {
      std::cerr << "[ERROR] session->connect failed" << std::endl;
      return -1;
    }

    std::cout << "[PERF] Connected to server" << std::endl;
    return 0;
  }
};

// ============================================================================
// PerfRunner - Public API
// ============================================================================

PerfRunner::PerfRunner(const PerfConfig& config)
    : impl_(new Impl(config)) {
}

PerfRunner::~PerfRunner() {
  delete impl_;
}

int PerfRunner::initialize() {
  std::cout << "[PERF] Initializing " << (impl_->config.is_server ? "SERVER" : "CLIENT") 
            << " mode..." << std::endl;

  // Check TCPX plugin
  int ndev = tcpx_get_device_count();
  if (ndev <= 0) {
    std::cerr << "[ERROR] No TCPX devices available" << std::endl;
    return -1;
  }
  std::cout << "[PERF] TCPX devices: " << ndev << std::endl;

  // Create session (gpu_id, num_channels)
  impl_->session = new TcpxSession(impl_->config.gpu_id, impl_->config.num_channels);
  if (!impl_->session) {
    std::cerr << "[ERROR] Failed to create TcpxSession" << std::endl;
    return -1;
  }

  // Bootstrap handshake (includes listen/connect)
  if (impl_->config.is_server) {
    if (impl_->bootstrapServer() != 0) {
      return -1;
    }
    std::cout << "[PERF] Listening on " << impl_->session->getNumChannels() << " channels" << std::endl;
  } else {
    if (impl_->bootstrapClient() != 0) {
      return -1;
    }
  }

  // Server: accept
  if (impl_->config.is_server) {
    std::string remote_name = "client";
    if (impl_->session->accept(remote_name) != 0) {
      std::cerr << "[ERROR] session->accept failed" << std::endl;
      return -1;
    }
    std::cout << "[PERF] Accepted " << impl_->session->getNumChannels() << " channels" << std::endl;
  }

  // Initialize CUDA
  if (impl_->initCuda() != 0) {
    return -1;
  }

  // Allocate memory
  if (impl_->allocateMemory() != 0) {
    return -1;
  }

  // Register memory
  bool is_recv = impl_->config.is_server;
  int ptr_type = NCCL_PTR_CUDA;
  impl_->mem_id = impl_->session->registerMemory(impl_->buffer,
                                                  Impl::kRegisteredBytes,
                                                  ptr_type, is_recv);
  if (impl_->mem_id == 0) {
    std::cerr << "[ERROR] registerMemory failed" << std::endl;
    return -1;
  }

  std::cout << "[PERF] Registered memory (mem_id=" << impl_->mem_id
            << ", " << (is_recv ? "recv" : "send") << ")" << std::endl;

  impl_->initialized = true;
  std::cout << "[PERF] Initialization complete" << std::endl;
  return 0;
}

int PerfRunner::run(PerfStats* stats) {
  if (!impl_->initialized) {
    std::cerr << "[ERROR] Runner not initialized" << std::endl;
    return -1;
  }

  if (!stats) {
    std::cerr << "[ERROR] stats pointer is null" << std::endl;
    return -1;
  }

  std::cout << "[PERF] Starting benchmark..." << std::endl;
  std::cout << "[PERF]   Size per iteration: " << (impl_->config.test_size / 1024 / 1024) << " MB" << std::endl;
  std::cout << "[PERF]   Chunk size: " << (impl_->config.chunk_bytes / 1024 / 1024) << " MB" << std::endl;
  std::cout << "[PERF]   Iterations: " << impl_->config.iterations << std::endl;

  // Calculate chunks per iteration
  size_t chunks_per_iter = (impl_->config.test_size + impl_->config.chunk_bytes - 1) / impl_->config.chunk_bytes;
  std::cout << "[PERF]   Chunks per iteration: " << chunks_per_iter << std::endl;

  // Remote name for transfer
  std::string remote_name = impl_->config.is_server ? "client" : "server";

  // Start timing
  auto start_time = std::chrono::high_resolution_clock::now();

  // Run iterations
  for (int iter = 0; iter < impl_->config.iterations; ++iter) {
    std::cout << "[PERF] Iteration " << (iter + 1) << "/" << impl_->config.iterations << std::endl;

    // Create transfer
    TcpxTransfer* transfer = impl_->session->createTransfer(remote_name);
    if (!transfer) {
      std::cerr << "[ERROR] createTransfer failed" << std::endl;
      return -1;
    }

    // Post chunks
    size_t remaining = impl_->config.test_size;
    size_t offset = 0;
    for (size_t chunk_idx = 0; chunk_idx < chunks_per_iter; ++chunk_idx) {
      size_t chunk_size = std::min(remaining, impl_->config.chunk_bytes);
      int tag = kTransferTag + iter * 10000 + static_cast<int>(chunk_idx);

      int rc;
      if (impl_->config.is_server) {
        // Server: post recv
        rc = transfer->postRecv(impl_->mem_id, offset, chunk_size, tag);
        if (rc != 0) {
          std::cerr << "[ERROR] postRecv failed (chunk " << chunk_idx << ")" << std::endl;
          delete transfer;
          return -1;
        }
      } else {
        // Client: post send
        rc = transfer->postSend(impl_->mem_id, offset, chunk_size, tag);
        if (rc != 0) {
          std::cerr << "[ERROR] postSend failed (chunk " << chunk_idx << ")" << std::endl;
          delete transfer;
          return -1;
        }
      }

      offset += chunk_size;
      remaining -= chunk_size;
    }

    // Wait for completion
    if (transfer->wait() != 0) {
      std::cerr << "[ERROR] transfer->wait failed" << std::endl;
      delete transfer;
      return -1;
    }

    // Get stats before release
    int completed_chunks = transfer->getCompletedChunks();
    int total_chunks = transfer->getTotalChunks();

    // Release transfer
    if (transfer->release() != 0) {
      std::cerr << "[ERROR] transfer->release failed" << std::endl;
      delete transfer;
      return -1;
    }

    delete transfer;

    std::cout << "[PERF]   Iteration " << (iter + 1) << " complete ("
              << completed_chunks << "/" << total_chunks
              << " chunks)" << std::endl;
  }

  // End timing
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  double total_time_ms = duration.count() / 1000.0;

  // Calculate statistics
  size_t total_bytes = impl_->config.test_size * impl_->config.iterations;
  double bandwidth_gbps = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (total_time_ms / 1000.0);

  stats->total_time_ms = total_time_ms;
  stats->bandwidth_gbps = bandwidth_gbps;
  stats->total_bytes = total_bytes;
  stats->iterations = impl_->config.iterations;
  stats->num_channels = impl_->session->getNumChannels();

  std::cout << "[PERF] Benchmark complete" << std::endl;
  return 0;
}

void PerfRunner::cleanup() {
  impl_->cleanup();
}

}  // namespace tcpx

