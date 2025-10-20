#include "tcpx_plugin_api.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <string>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>

using namespace tcpx_plugin;

// -------------------------------
// Env helpers
// -------------------------------
static int getEnvInt(const char* name, int defaultVal) {
  const char* val = std::getenv(name);
  return val ? std::atoi(val) : defaultVal;
}
static size_t getEnvSize(const char* name, size_t defaultVal) {
  const char* val = std::getenv(name);
  return val ? static_cast<size_t>(std::strtoull(val, nullptr, 10)) : defaultVal;
}

// -------------------------------
// Alignment helpers
// -------------------------------
static inline size_t align_up_4k(size_t x)   { return (x + 4095ull) & ~4095ull; }
static inline size_t align_down_4k(size_t x) { return (x & ~4095ull); }

// -------------------------------
// Aligned CUDA allocation (4KB)
// -------------------------------
static bool cuda_alloc_4k_aligned(size_t want_bytes, void** out_aligned_ptr,
                                  CUdeviceptr* out_base, size_t* out_usable_bytes) {
  if (!out_aligned_ptr || !out_base || !out_usable_bytes) return false;
  *out_aligned_ptr = nullptr; *out_base = 0; *out_usable_bytes = 0;

  // Over-allocate +4096 to guarantee we can align up to 4KB.
  size_t alloc_bytes = want_bytes + 4096ull;
  CUdeviceptr d_base = 0;
  CUresult rc = cuMemAlloc(&d_base, alloc_bytes);
  if (rc != CUDA_SUCCESS) return false;

  uintptr_t addr = static_cast<uintptr_t>(d_base);
  uintptr_t aligned = (addr + 4095ull) & ~4095ull;
  size_t usable = alloc_bytes - (aligned - addr);

  *out_base = d_base;
  *out_aligned_ptr = reinterpret_cast<void*>(aligned);
  *out_usable_bytes = usable;
  return true;
}

// -------------------------------
// Pretty print Status
// -------------------------------
static const char* status_str(Status s) {
  switch (s) {
    case Status::kOk:          return "kOk";
    case Status::kUnavailable: return "kUnavailable";
    case Status::kInvalidArg:  return "kInvalidArg";
    case Status::kInternal:    return "kInternal";
    case Status::kTimeout:     return "kTimeout";
  }
  return "Unknown";
}

int main(int argc, char** argv) {
  // Recommended defaults for devmem TCP
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

  const std::string mode = argv[1];
  const bool is_server = (mode == "server");
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
    gpu_id = (argc > 3) ? std::atoi(argv[3]) : 0;
  }

  // Basic configuration via env
  const int    num_channels     = getEnvInt("UCCL_TCPX_NUM_CHANNELS", 1);
  const size_t total_bytes_raw  = getEnvSize("UCCL_TCPX_PERF_SIZE",   32ull * 1024 * 1024);
  const size_t chunk_bytes_raw  = getEnvSize("UCCL_TCPX_CHUNK_BYTES", 512ull * 1024);
  const int    port_base        = getEnvInt("UCCL_TCPX_BOOTSTRAP_PORT_BASE", 12345);
  const int    port             = port_base + gpu_id;
  const int    iterations       = getEnvInt("UCCL_TCPX_PERF_ITERS", 1);
  const std::string remote_name = is_server ? "client" : "server";

  // Async test knobs (env)
  const int    async_inflight_max = std::max(1, getEnvInt("UCCL_TCPX_ASYNC_INFLIGHT", 4));
  const int    async_poll_us      = std::max(1, getEnvInt("UCCL_TCPX_ASYNC_POLL_US", 200)); // microseconds
  const int    do_async_test      = getEnvInt("UCCL_TCPX_DO_ASYNC", 1); // set 0 to skip async test

  // Force 4KB alignment for chunk size (safer with device recv scatter layout)
  size_t chunk_bytes = align_up_4k(chunk_bytes_raw);
  if (chunk_bytes != chunk_bytes_raw) {
    std::cout << "[WARN] UCCL_TCPX_CHUNK_BYTES (" << chunk_bytes_raw
              << ") is not 4KB aligned. Using " << chunk_bytes << " instead.\n";
  }

  // We'll align the registered size down to 4KB.
  size_t total_bytes = total_bytes_raw;

  std::cout << "=========== tcpx new API smoke test ===========\n";
  std::cout << "Mode          : " << (is_server ? "SERVER" : "CLIENT") << "\n";
  std::cout << "GPU           : " << gpu_id << "\n";
  std::cout << "Channels      : " << num_channels << "\n";
  std::cout << "Total bytes   : " << (total_bytes / (1024.0 * 1024.0)) << " MB\n";
  std::cout << "Chunk bytes   : " << (chunk_bytes / 1024.0) << " KB\n";
  std::cout << "Iterations    : " << iterations << "\n";
  std::cout << "Bootstrap port: " << port << "\n";
  if (!is_server) std::cout << "Server IP    : " << server_ip << "\n";
  std::cout << "Async inflight: " << async_inflight_max << "\n";
  std::cout << "Async poll(us): " << async_poll_us << "\n";
  std::cout << "Run async test: " << (do_async_test ? "yes" : "no") << "\n";
  std::cout << "================================================\n";

  // Transport configuration
  Config cfg;
  cfg.plugin_path = nullptr;     // use default loader path
  cfg.gpu_id      = gpu_id;
  cfg.channels    = num_channels;
  cfg.chunk_bytes = chunk_bytes; // already 4KB aligned
  cfg.enable      = true;

  Transport t(cfg);

  // CUDA init
  if (cuInit(0) != CUDA_SUCCESS) {
    std::cerr << "cuInit failed\n";
    return 3;
  }
  CUdevice cu_dev = 0;
  if (cuDeviceGet(&cu_dev, gpu_id) != CUDA_SUCCESS) {
    std::cerr << "cuDeviceGet failed\n";
    return 3;
  }
  CUcontext cu_ctx = nullptr;
  if (cuDevicePrimaryCtxRetain(&cu_ctx, cu_dev) != CUDA_SUCCESS) {
    std::cerr << "cuDevicePrimaryCtxRetain failed\n";
    return 3;
  }
  if (cuCtxSetCurrent(cu_ctx) != CUDA_SUCCESS) {
    std::cerr << "cuCtxSetCurrent failed\n";
    return 3;
  }
  cudaSetDevice(gpu_id);

  // Bootstrap + connect
  ConnHandle conn = 0;
  Status st = Status::kOk;
  if (is_server) {
    st = t.accept(port, remote_name, conn);
  } else {
    st = t.connect(server_ip, port, remote_name, conn);
  }
  if (st != Status::kOk) {
    std::cerr << "connect/accept failed, status=" << static_cast<int>(st)
              << " (" << status_str(st) << ")\n";
    cuCtxSetCurrent(nullptr);
    cuDevicePrimaryCtxRelease(cu_dev);
    return 4;
  }

  // Device buffer (4KB aligned)
  void* aligned_ptr = nullptr;
  CUdeviceptr d_base = 0;
  size_t usable = 0;

  if (!cuda_alloc_4k_aligned(total_bytes, &aligned_ptr, &d_base, &usable) || aligned_ptr == nullptr) {
    std::cerr << "cuMemAlloc (aligned) failed\n";
    cuCtxSetCurrent(nullptr);
    cuDevicePrimaryCtxRelease(cu_dev);
    return 5;
  }

  // Registered size must be <= usable and 4KB aligned down.
  size_t reg_bytes = align_down_4k(std::min(usable, total_bytes));
  if (reg_bytes == 0) {
    std::cerr << "aligned usable size is too small after 4KB trim\n";
    cuMemFree(d_base);
    cuCtxSetCurrent(nullptr);
    cuDevicePrimaryCtxRelease(cu_dev);
    return 5;
  }

  // Initialize device buffer contents
  if (!is_server) {
    cudaMemset(aligned_ptr, 0xAB, reg_bytes);
  } else {
    cudaMemset(aligned_ptr, 0x00, reg_bytes);
  }
  cudaDeviceSynchronize();

  // Register memory (recv side uses is_recv=true)
  MrHandle mr = 0;
  st = t.register_memory(aligned_ptr, reg_bytes, /*is_recv*/ is_server, mr);
  if (st != Status::kOk || mr == 0) {
    std::cerr << "register_memory failed, status=" << static_cast<int>(st)
              << " (" << status_str(st) << ")\n";
    cuMemFree(d_base);
    cuCtxSetCurrent(nullptr);
    cuDevicePrimaryCtxRelease(cu_dev);
    return 6;
  }

  // =============== Blocking test ===============
  {
    const int tag_base = 100;
    auto t0 = std::chrono::high_resolution_clock::now();
    Status bst = Status::kOk;

    for (int it = 0; it < iterations; ++it) {
      if (is_server) {
        bst = t.recv(conn, mr, reg_bytes, tag_base + it * 1000);
      } else {
        bst = t.send(conn, mr, reg_bytes, tag_base + it * 1000);
      }
      if (bst != Status::kOk) {
        std::cerr << (is_server ? "recv" : "send")
                  << " (blocking) failed at iter " << it
                  << ", status=" << static_cast<int>(bst)
                  << " (" << status_str(bst) << ")\n";
        st = bst;
        break;
      }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    double gb = (double)reg_bytes * iterations / (1024.0 * 1024.0 * 1024.0);

    if (bst == Status::kOk) {
      std::cout << (is_server ? "[SERVER] " : "[CLIENT] ")
                << "[BLOCK] time=" << ms << " ms, size=" << gb
                << " GB, bw=" << (gb / (ms / 1000.0)) << " GB/s\n";
      if (is_server) {
        unsigned char h0 = 0;
        cudaMemcpy(&h0, aligned_ptr, 1, cudaMemcpyDeviceToHost);
        std::cout << "[SERVER][BLOCK] first byte: 0x" << std::hex << (int)h0 << std::dec << "\n";
      }
    }
  }

  // =============== Async test ===============
  if (do_async_test) {
    // Re-initialize server buffer to a known value; client keeps pattern 0xAB.
    if (is_server) {
      cudaMemset(aligned_ptr, 0x00, reg_bytes);
      cudaDeviceSynchronize();
    }

    const int tag_base = 50000; // separate tag space from blocking test
    auto t0 = std::chrono::high_resolution_clock::now();

    // We will queue up to async_inflight_max transfers; each iteration is one full reg_bytes transfer.
    // For client: send_async; for server: recv_async.
    struct Inflight {
      TxHandle tx{0};
      int      iter{-1};
    };
    std::vector<Inflight> inflight;

    int next_iter_to_post = 0;
    int completed = 0;
    Status ast = Status::kOk;

    auto post_one = [&](int it)->Status {
      TxHandle tx = 0;
      Status pst;
      if (is_server) {
        pst = t.recv_async(conn, mr, reg_bytes, tag_base + it * 1000, tx);
      } else {
        pst = t.send_async(conn, mr, reg_bytes, tag_base + it * 1000, tx);
      }
      if (pst == Status::kOk) {
        inflight.push_back({tx, it});
      }
      return pst;
    };

    // Initially fill the window up to inflight cap or total iterations
    for (; next_iter_to_post < iterations && (int)inflight.size() < async_inflight_max; ++next_iter_to_post) {
      Status pst = post_one(next_iter_to_post);
      if (pst != Status::kOk) {
        std::cerr << "[ASYNC] post failed at iter " << next_iter_to_post
                  << ", status=" << static_cast<int>(pst)
                  << " (" << status_str(pst) << ")\n";
        ast = pst;
        break;
      }
    }

    // Poll loop: progress until all iterations complete
    while (ast == Status::kOk && completed < iterations) {
      // Poll each inflight tx
      for (size_t i = 0; i < inflight.size();) {
        bool done = false;
        Status pst = t.poll_transfer(inflight[i].tx, done);
        if (pst != Status::kOk) {
          std::cerr << "[ASYNC] poll_transfer failed for iter " << inflight[i].iter
                    << ", status=" << static_cast<int>(pst)
                    << " (" << status_str(pst) << ")\n";
          ast = pst;
          break;
        }
        if (done) {
          // remove this entry
          inflight.erase(inflight.begin() + i);
          ++completed;

          // Try to post a new one to keep the window full
          if (next_iter_to_post < iterations) {
            Status npst = post_one(next_iter_to_post);
            if (npst != Status::kOk) {
              std::cerr << "[ASYNC] post failed at iter " << next_iter_to_post
                        << ", status=" << static_cast<int>(npst)
                        << " (" << status_str(npst) << ")\n";
              ast = npst;
              break;
            }
            ++next_iter_to_post;
          }
        } else {
          ++i; // not done, move to next inflight
        }
      }

      if (ast != Status::kOk) break;

      // Sleep a bit to avoid busy-wait (poll interval from env)
      if (completed < iterations) {
        usleep(async_poll_us);
      }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    double gb = (double)reg_bytes * iterations / (1024.0 * 1024.0 * 1024.0);

    if (ast == Status::kOk) {
      std::cout << (is_server ? "[SERVER] " : "[CLIENT] ")
                << "[ASYNC] time=" << ms << " ms, size=" << gb
                << " GB, bw=" << (gb / (ms / 1000.0)) << " GB/s"
                << " (inflight=" << async_inflight_max << ", poll_us=" << async_poll_us << ")\n";
      if (is_server) {
        unsigned char h0 = 0;
        cudaMemcpy(&h0, aligned_ptr, 1, cudaMemcpyDeviceToHost);
        std::cout << "[SERVER][ASYNC] first byte: 0x" << std::hex << (int)h0 << std::dec << "\n";
      }
    } else {
      st = ast;
    }
  }

  // Cleanup
  t.deregister_memory(mr);
  cuMemFree(d_base);

  cuCtxSetCurrent(nullptr);
  cuDevicePrimaryCtxRelease(cu_dev);

  std::cout << "exit " << ((st == Status::kOk) ? "ok" : "with error") << "\n";
  return (st == Status::kOk) ? 0 : 8;
}
