// tests/test_tcpx_plugin_api.cc
#include "tcpx_plugin_api.h"
#include "bootstrap.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>

using namespace tcpx_plugin;

static inline int getEnvInt(char const* name, int def) {
  char const* v = std::getenv(name);
  return v ? std::atoi(v) : def;
}

static inline size_t getEnvSize(char const* name, size_t def) {
  char const* v = std::getenv(name);
  return v ? static_cast<size_t>(std::atoll(v)) : def;
}

static void* cuda_alloc_aligned(size_t bytes, void** base_out) {
  // 4KB alignment
  CUdeviceptr d_base = 0;
  size_t alloc = bytes + 4096;
  if (cuMemAlloc(&d_base, alloc) != CUDA_SUCCESS) return nullptr;
  uintptr_t addr = static_cast<uintptr_t>(d_base);
  addr = (addr + 4095) & ~static_cast<uintptr_t>(4095);
  if (base_out) *base_out = reinterpret_cast<void*>(d_base);
  return reinterpret_cast<void*>(addr);
}

int main(int argc, char** argv) {
  // Default env settings for testing
  setenv("NCCL_MIN_ZCOPY_SIZE", "4096", 0);
  setenv("NCCL_GPUDIRECTTCPX_MIN_ZCOPY_SIZE", "4096", 0);
  setenv("NCCL_GPUDIRECTTCPX_RECV_SYNC", "1", 0);
  setenv("UCCL_TCPX_DEBUG", "1", 0);
  setenv("UCCL_TCPX_KERNEL_DEBUG", "0", 0);

  if (argc < 2) {
    std::cerr
        << "Usage:\n"
        << "  Server: " << argv[0] << " server [gpu_id]\n"
        << "  Client: " << argv[0] << " client <server_ip> [gpu_id]\n";
    return 1;
  }

  std::string mode = argv[1];
  bool is_server = (mode == "server");
  std::string server_ip;
  int gpu_id = 0;

  if (is_server) {
    gpu_id = (argc > 2) ? std::atoi(argv[2]) : 0;
  } else {
    if (argc < 3) {
      std::cerr << "client mode requires <server_ip>\n";
      return 1;
    }
    server_ip = argv[2];
    gpu_id    = (argc > 3) ? std::atoi(argv[3]) : 0;
  }

  // Config via env
  int    num_channels  = getEnvInt("UCCL_TCPX_NUM_CHANNELS", 1);
  size_t total_bytes   = getEnvSize("UCCL_TCPX_PERF_SIZE", 32 * 1024 * 1024);
  size_t chunk_bytes   = getEnvSize("UCCL_TCPX_CHUNK_BYTES", 512 * 1024);
  int    port_base     = getEnvInt("UCCL_TCPX_BOOTSTRAP_PORT_BASE", 12345);
  int    port          = port_base + gpu_id;
  int    iterations    = getEnvInt("UCCL_TCPX_PERF_ITERS", 1);  
  std::string remote_name = is_server ? "client" : "server";

  std::cout << "=========== tcpx_plugin_api smoke test ===========\n";
  std::cout << "Mode          : " << (is_server ? "SERVER" : "CLIENT") << "\n";
  std::cout << "GPU           : " << gpu_id << "\n";
  std::cout << "Channels      : " << num_channels << "\n";
  std::cout << "Total bytes   : " << (total_bytes / 1024 / 1024) << " MB\n";
  std::cout << "Chunk bytes   : " << (chunk_bytes / 1024) << " KB\n";
  std::cout << "Iterations    : " << iterations << "\n";
  std::cout << "Bootstrap port: " << port << "\n";
  if (!is_server) std::cout << "Server IP    : " << server_ip << "\n";
  std::cout << "===================================================\n";

  // 1) Init plugin
  InitOptions opts{};
  if (init(opts) != Status::kOk) {
    std::cerr << "tcpx_plugin::init failed\n";
    return 2;
  }

  // 2) CUDA init
  if (cuInit(0) != CUDA_SUCCESS) {
    std::cerr << "cuInit failed\n"; return 3;
  }
  CUdevice cu_dev = 0;
  if (cuDeviceGet(&cu_dev, gpu_id) != CUDA_SUCCESS) {
    std::cerr << "cuDeviceGet failed\n"; return 3;
  }
  CUcontext cu_ctx = nullptr;
  if (cuDevicePrimaryCtxRetain(&cu_ctx, cu_dev) != CUDA_SUCCESS) {
    std::cerr << "cuDevicePrimaryCtxRetain failed\n"; return 3;
  }
  if (cuCtxSetCurrent(cu_ctx) != CUDA_SUCCESS) {
    std::cerr << "cuCtxSetCurrent failed\n"; return 3;
  }
  cudaSetDevice(gpu_id);

  // 3) Create session
  Session sess(gpu_id, num_channels, "", -1);

  // 4) TCP bootstrap
  int sock_fd = -1;
  if (is_server) {
    if (bootstrap_server_create(port, &sock_fd) != 0) {
      std::cerr << "bootstrap_server_create failed\n"; return 4;
    }
    if (server_send_conn_json(&sess, sock_fd) != Status::kOk) {
      std::cerr << "server_send_conn_json failed\n"; return 4;
    }
  } else {
    if (bootstrap_client_connect(server_ip.c_str(), port, &sock_fd) != 0) {
      std::cerr << "bootstrap_client_connect failed\n"; return 4;
    }
    if (client_recv_conn_json(&sess, sock_fd, remote_name) != Status::kOk) {
      std::cerr << "client_recv_conn_json failed\n"; return 4;
    }
    if (sess.connect(remote_name) != Status::kOk) {
      std::cerr << "sess.connect failed\n"; return 4;
    }
  }
  if (is_server) {
    if (sess.accept(remote_name) != Status::kOk) {
      std::cerr << "sess.accept failed\n"; return 4;
    }
  }

  // 5) Allocate device memory
  void* base_ptr = nullptr;
  void* aligned  = cuda_alloc_aligned(total_bytes, &base_ptr);
  if (!aligned) {
    std::cerr << "device alloc failed\n"; return 5;
  }

  if (!is_server) {
    cudaMemset(aligned, 0xAB, total_bytes);
    cudaDeviceSynchronize();
  } else {
    cudaMemset(aligned, 0x00, total_bytes);
    cudaDeviceSynchronize();
  }

  // 6) Register memory
  uint64_t mem_id = sess.register_memory(aligned, total_bytes, is_server);
  if (mem_id == 0) {
    std::cerr << "register_memory failed\n"; return 6;
  }

  // 7) Create transfer
  ConnID cid{remote_name, -1};
  Transfer* xfer = sess.create_transfer(cid);
  if (!xfer) {
    std::cerr << "create_transfer failed\n"; return 7;
  }

  // 8) Send/recv
  int tag_base = 100;
  auto t0 = std::chrono::high_resolution_clock::now();

  for (int it = 0; it < iterations; ++it) {
    if (is_server) {
      if (xfer->recv_all(mem_id, total_bytes, 0, chunk_bytes, tag_base + it*1000)
          != Status::kOk) {
        std::cerr << "recv_all failed\n"; return 8;
      }
    } else {
      if (xfer->send_all(mem_id, total_bytes, 0, chunk_bytes, tag_base + it*1000)
          != Status::kOk) {
        std::cerr << "send_all failed\n"; return 8;
      }
    }
    if (xfer->wait() != Status::kOk) {
      std::cerr << "xfer->wait failed\n"; return 8;
    }
    xfer->release();
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;

  // 9) Server-side check
  if (is_server) {
    unsigned char h0 = 0;
    cudaMemcpy(&h0, aligned, 1, cudaMemcpyDeviceToHost);
    std::cout << "[SERVER] first byte after recv: 0x" << std::hex
              << (int)h0 << std::dec << "\n";
  }

  double gb = (double)total_bytes * iterations / (1024.0 * 1024.0 * 1024.0);
  std::cout << (is_server ? "[SERVER] " : "[CLIENT] ")
            << "Done. time=" << ms << " ms, size="
            << gb << " GB, bw=" << (gb / (ms/1000.0)) << " GB/s\n";

  // 10) Cleanup
  delete xfer;
  sess.deregister_memory(mem_id);

  if (base_ptr) cuMemFree(reinterpret_cast<CUdeviceptr>(base_ptr));
  if (sock_fd >= 0) close(sock_fd);

  cuCtxSetCurrent(nullptr);
  cuDevicePrimaryCtxRelease(cu_dev);
  std::cout << "exit ok\n";
  return 0;
}
