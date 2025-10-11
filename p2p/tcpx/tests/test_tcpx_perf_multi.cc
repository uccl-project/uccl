/**
 * @file test_tcpx_perf_multi.cc
 * @brief Multi-channel TCPX GPU-to-GPU performance benchmark
 *
 * Goal:
 * Measure GPU-to-GPU throughput between two H100 nodes over multiple TCPX
 * channels. Extends test_tcpx_perf.cc with multi-channel support to utilize
 * multiple NICs (e.g., eth1-4).
 *
 * Design:
 * - Server: receives data; unpacks with a GPU kernel (gathers from a scattered
 *   bounce buffer into a contiguous GPU destination).
 * - Client: sends data.
 * - ChannelManager manages multiple TCPX channels.
 * - Chunks are dispatched round-robin to channels.
 * - Each channel has its own sliding window to avoid exhausting request slots.
 *
 * Usage:
 *   # Server (example: 10.65.74.150)
 *   UCCL_TCPX_NUM_CHANNELS=4 UCCL_TCPX_PERF_SIZE=67108864 \
 *   ./tests/test_tcpx_perf_multi server 0
 *
 *   # Client (example: 10.64.113.77)
 *   UCCL_TCPX_NUM_CHANNELS=4 UCCL_TCPX_PERF_SIZE=67108864 \
 *   ./tests/test_tcpx_perf_multi client 10.65.74.150 0
 *
 * Env vars:
 *   UCCL_TCPX_NUM_CHANNELS: number of channels (default 1, recommend 4)
 *   UCCL_TCPX_PERF_SIZE: total bytes per iteration (default 4MB)
 *   UCCL_TCPX_PERF_ITERS: iteration count (default 10)
 *   UCCL_TCPX_CHUNK_BYTES: chunk size (default 2MB)
 *   UCCL_TCPX_UNPACK_IMPL: unpack impl (kernel|d2d|host, default kernel)
 *
 * Key lessons (from test_tcpx_perf.cc):
 * 1) Check window capacity BEFORE tcpx_irecv.
 * 2) Call tcpx_irecv_consumed after the kernel completes (track via events).
 * 3) Device handle must be 16-byte aligned.
 * 4) Accept may return nullptr; retry.
 * 5) Use a unique tag per chunk.
 * 6) Remove timeouts; poll until completion.
 * 7) Create stream/launcher outside loops (~4ms saved per chunk).
 * 8) Client send window = 12 (<16 for headroom).
 * 9) Default chunk size = 512KB.
 */

/*
======================== File Reading Guide (2025-10-08)
======================== This file is the reference implementation for the
multi-process baseline. Each GPU runs its own process and opens multiple TCPX
connections (e.g., UCCL_TCPX_NUM_CHANNELS=4). A channel ≈ a TCPX connection.

Vendor guidance: a single 200Gbps NIC with ~8 connections is a practical
per-NIC upper baseline (~21.26 GB/s). Do not chase symmetric send/recv numbers.

NUMA guidance: bind each GPU to its NUMA-local NIC (configure IFNAME via
script/env). Keep NCCL threading; no app-layer ACKs, plain send/recv.

Core semantics and constraints:
1) Progress & FIFO: tcpx_test() drives tcpxCommProgress() internally; call it
   repeatedly. Always test only the per-channel FIFO head (matches
   rq.next_transmitting). Nonzero rc is a real error; done=0 means not finished.
2) Windows & limits: each comm has MAX_REQUESTS=16. Server recv uses 16; client
   send uses 12 for safety. When full: server waits via
wait_for_channel_capacity; client polls oldest send. 3) Recv lifecycle: irecv →
tcpx_test(done=1) → (kernel waits for event) → tcpx_irecv_consumed(). Send is
simpler: done=1 auto-releases. 4) Unique tags: each chunk must have a globally
unique tag (iter + global chunk index) to avoid cross-match. 5) Measurement
asymmetry is expected: server includes recv drain (and unpack), window=16, has
consumed; client measures only send pipeline (window=12, auto-release). Progress
cadence and NUMA/NIC placement affect readings.

Structure overview:
- Server path:
  1) ChannelManager.listen → bootstrap send handles → accept → CUDA init →
     register recv buffer
  2) Per-channel sliding window and (kernel mode) persistent stream/launcher
  3) Main loop: split chunks, round-robin channels, post tcpx_irecv then
     process_completed_chunk(false) and opportunistically poll others; when
     window full, block (event or completion); end-of-iter drain
- Client path:
  1) Bootstrap receive handles → connect all channels → CUDA init → register
     send buffer
  2) Per-channel send window (12)
  3) Main loop: chunk split, round-robin channel selection; when full, wait for
     oldest send (tcpx_test + sleep); post tcpx_isend and track pending; drain
     at iteration end.

Operational notes:
- Set UCCL_TCPX_NUM_CHANNELS=4 on both server and client. To stress a single
  NIC, set IFNAME to expose one NIC.

Common pitfalls:
- Missing post-irecv non-blocking progress → first batch stalls
- Testing beyond FIFO head → triggers next_transmitting guard
- Not calling consumed after done=1 on recv → slot leak and stalls
- Chasing symmetric send/recv bandwidth → unnecessary
===============================================================================
*/

#include "../device/unpack_launch.h"     // GPU kernel launcher
#include "../include/bootstrap.h"        // Bootstrap protocol
#include "../include/channel_manager.h"  // Multi-channel manager
#include "../include/rx_descriptor.h"    // Receive descriptor builder
#include "../include/tcpx_interface.h"   // TCPX API wrapper
#include "../include/tcpx_structs.h"     // TCPX internal structs
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
// Vendor notes (2025-10-08):
// - One channel ≈ one TCPX connection; do not overemphasize channels in NIXL
// - Bind each GPU to its NUMA-local NIC (IFNAME via scripts/env is fine)
// - vLLM: one process per GPU; a process uses one NIC (multiple connections
//   still OK for NIC stress testing)

#include <algorithm>
#include <vector>

namespace {
// ============================================================================
// Constants
// ============================================================================

constexpr int kTransferTag = 99;                // base transfer tag
constexpr size_t kMaxSize = 256 * 1024 * 1024;  // max transfer size (256MB)
constexpr size_t kRegisteredBytes = kMaxSize + 4096;  // registered size

// ============================================================================
// Helpers
// ============================================================================

int getEnvInt(char const* name, int def) {
  char const* v = std::getenv(name);
  return v ? std::atoi(v) : def;
}

size_t getEnvSize(char const* name, size_t def) {
  char const* v = std::getenv(name);
  return v ? static_cast<size_t>(std::atoll(v)) : def;
}

// CUDA error checking helpers
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
// Main
// ============================================================================

int main(int argc, char** argv) {
  // ============================================================================
  // Env overrides (TCPX config)
  // ============================================================================

  // Recommendations (2025-10-08):
  // - Drop single-process orchestrator; use this test as the main path
  // - Example: UCCL_TCPX_NUM_CHANNELS=2, NCCL_NSOCKS_PERTHREAD=2
  //   * 2 channels per GPU; 2 sockets per channel → 4 sockets per GPU
  //   * 2 GPUs share 1 NIC → 8 sockets per NIC (MAX_SOCKETS=8)
  // - Focus on one-way max and scaling; symmetry is unnecessary

  // Enable zero-copy (devmem-tcp from 4KB)
  setenv("NCCL_MIN_ZCOPY_SIZE", "4096", 0);
  setenv("NCCL_GPUDIRECTTCPX_MIN_ZCOPY_SIZE", "4096", 0);

  // Enable recv sync (data integrity)
  setenv("NCCL_GPUDIRECTTCPX_RECV_SYNC", "1", 0);

  // Enable TCPX wrapper debug logs
  setenv("UCCL_TCPX_DEBUG", "1", 0);

  // Disable kernel-launch debug by default
  if (!std::getenv("UCCL_TCPX_LAUNCH_DEBUG"))
    setenv("UCCL_TCPX_LAUNCH_DEBUG", "0", 0);

  // ============================================================================
  // CLI parsing
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
  // Benchmark parameters
  // ============================================================================

  // Number of channels (default 2)
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

  // === SERVER path overview ===
  // 1) listen and generate handles (then send to client via bootstrap)
  // 2) accept all channels (create recv_comm)
  // 3) choose unpack impl (kernel/d2d/host)
  // 4) CUDA init + 4KB aligned buffer; register buffer across channels
  // 5) (kernel) persistent stream/launcher; pre-create events
  // 6) per-channel sliding window (MAX_INFLIGHT_PER_CHANNEL=16)
  // 7) post irecv per chunk; then non-blocking progress; block when full
  // 8) drain inflight/pending at iteration end and record performance

  // Total bytes per iteration (default 4MB)
  size_t test_size = getEnvSize("UCCL_TCPX_PERF_SIZE", 4 * 1024 * 1024);

  // Iteration count
  int iterations = getEnvInt("UCCL_TCPX_PERF_ITERS", 10);

  // Chunk size (default 512KB)
  size_t chunk_bytes =
      getEnvSize("UCCL_TCPX_CHUNK_BYTES",
                 getEnvSize("NCCL_P2P_NET_CHUNKSIZE", 512 * 1024));

  // Cap maximum transfer size
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
  // TCPX plugin initialization
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
  // SERVER logic
  // ============================================================================

  if (is_server) {
    std::cout << "[PERF] Starting SERVER mode" << std::endl;

    // ==========================================================================
    // Step 1: create ChannelManager and listen
    // ==========================================================================

    ChannelManager mgr(num_channels, gpu_id);
    std::vector<ncclNetHandle_v7> handles;

    if (mgr.server_listen_all(handles) != 0) {
      std::cerr << "[ERROR] server_listen_all failed" << std::endl;
      return 1;
    }

    // Update num_channels to actual count after clamping
    num_channels = mgr.get_num_channels();
    std::cout << "[PERF] Listening on " << num_channels
              << " channels (after clamping to available TCPX devices)"
              << std::endl;

    // ==========================================================================
    // Step 2: bootstrap handshake (send handles to client)
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
    // Step 3: accept all channels
    // ==========================================================================

    if (mgr.server_accept_all() != 0) {
      std::cerr << "[ERROR] server_accept_all failed" << std::endl;
      close(bootstrap_fd);
      return 1;
    }

    std::cout << "[PERF] All " << num_channels << " channels accepted"
              << std::endl;

    // ==========================================================================
    // Step 4: choose unpack implementation
    // ==========================================================================

    char const* impl_env = std::getenv("UCCL_TCPX_UNPACK_IMPL");
    std::string impl = impl_env ? std::string(impl_env) : std::string("kernel");
    std::transform(impl.begin(), impl.end(), impl.begin(), ::tolower);
    std::cout << "[PERF] Unpack impl: " << impl << std::endl;

    // ==========================================================================
    // Step 5: CUDA initialization
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
    // Step 6: allocate recv buffer
    // ==========================================================================

    CUdeviceptr d_base = 0, d_aligned = 0;

    if (!cuda_check(cuMemAlloc(&d_base, kRegisteredBytes + 4096),
                    "cuMemAlloc")) {
      close(bootstrap_fd);
      return 1;
    }

    // Align to 4KB boundary (devmem-tcp requirement)
    uintptr_t addr = static_cast<uintptr_t>(d_base);
    addr = (addr + 4095) & ~static_cast<uintptr_t>(4095);
    d_aligned = static_cast<CUdeviceptr>(addr);
    void* recv_buf = reinterpret_cast<void*>(d_aligned);

    std::cout << "[PERF] Allocated recv buffer: " << recv_buf << std::endl;

    // ==========================================================================
    // Step 7: register memory to all channels (shared)
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
    // Progress engine (recv): always test FIFO head
    // - rc!=0 means real error; done=0 means not finished
    // - When done=1: read unpack metadata, run kernel/d2d/host path, then call
    //   irecv_consumed()
    // - Call non-blocking after post; block when window is full

    // ==========================================================================
    // Step 8: create persistent stream and launcher (kernel only)
    // ==========================================================================

    // Create outside loops to avoid ~4ms per chunk overhead
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
    // Step 9: create per-channel sliding windows
    // ==========================================================================

    // Each comm has MAX_REQUESTS=16; use per-channel independent sliding
    // windows
    constexpr int MAX_INFLIGHT_PER_CHANNEL = 16;

    // Per-channel sliding window state
    struct PostedChunk {
      void* request = nullptr;
      void* dst_ptr = nullptr;
      size_t bytes = 0;
      size_t offset = 0;
      int tag = 0;
      int global_idx = 0;
    };

    struct ChannelWindow {
      std::vector<cudaEvent_t> events;  // CUDA events (track kernel completion)
      std::vector<void*> pending_reqs;  // Kernel submitted but not yet consumed
      std::vector<int> pending_indices;        // Pending chunk indices
      int chunk_counter = 0;                   // Chunks handled by this channel
      std::deque<PostedChunk> inflight_recvs;  // Posted but not yet unpacked
    };

    std::vector<ChannelWindow> channel_windows(num_channels);

    if (impl == "kernel") {
      // Pre-create CUDA events for each channel
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

        // Ablation mode: skip any unpack work and immediately consume the recv.
        if (impl == "none") {
          tcpx_irecv_consumed(ch.recv_comm, 1, entry.request);
          win.inflight_recvs.pop_front();
          // Proceed to next FIFO head (if any)
          continue;
        }

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
    // Step 10: main benchmark loop
    // ==========================================================================

    double total_time_ms = 0.0;
    int completed_iters = 0;
    bool abort_benchmark = false;

    for (int iter = 0; iter < iterations; ++iter) {
      std::cout << "[PERF] Iteration " << iter << ": total bytes=" << test_size
                << ", chunk_bytes=" << chunk_bytes << std::endl;

      auto start = std::chrono::high_resolution_clock::now();
      bool iteration_failed = false;

      // Reset per-channel sliding windows at the start of each iteration
      if (impl == "kernel") {
        std::cout << "[DEBUG] Iteration " << iter
                  << " start: clearing sliding windows for " << num_channels
                  << " channels" << std::endl;
        for (int ch = 0; ch < num_channels; ++ch) {
          channel_windows[ch].pending_reqs.clear();
          // After recording the irecv to the per-channel FIFO, trigger
          // non-blocking progress immediately to avoid initial backlog. Also
          // opportunistically progress other channels to reduce HOL risk.

          channel_windows[ch].pending_indices.clear();
          channel_windows[ch].chunk_counter = 0;
        }
      }

      // ========================================================================
      // Chunk loop: split message into chunks and round-robin across channels
      // ========================================================================

      size_t offset = 0;
      int global_chunk_idx = 0;

      while (offset < test_size) {
        const size_t this_chunk = std::min(chunk_bytes, test_size - offset);
        void* dst_ptr = reinterpret_cast<void*>(
            reinterpret_cast<uintptr_t>(recv_buf) + offset);

        // Round-robin channel selection
        int channel_id = global_chunk_idx % num_channels;
        ChannelResources& ch = mgr.get_channel(channel_id);
        ChannelWindow& win = channel_windows[channel_id];

        // Unique tag per chunk
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
        // Post async recv (tcpx_irecv)
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
      // End of iteration: drain per-channel windows (kernel mode only)
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

            // === CLIENT path overview ===
            // 1) Bootstrap and receive handles → ChannelManager(handles.size())
            // 2) Connect all channels (send_comm)
            // 3) CUDA init + 4KB-aligned send buffer + register to channels
            // 4) Per-channel send window (MAX_INFLIGHT_SEND_PER_CHANNEL=12)
            // 5) Send chunks; if full, wait on oldest send (tcpx_test+sleep)
            // 6) Drain pending sends and compute averages at iteration end

            if (abort_benchmark) break;
          }
        }
      }

      // ========================================================================
      // Compute this iteration's performance
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

    // Compute and print average
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
    // Cleanup
    // ==========================================================================

    // Destroy launcher and stream
    if (launcher_ptr) {
      delete launcher_ptr;
      launcher_ptr = nullptr;
    }
    if (unpack_stream) {
      cudaStreamDestroy(unpack_stream);
      unpack_stream = nullptr;
    }

    // Destroy CUDA events
    if (impl == "kernel") {
      for (int ch = 0; ch < num_channels; ++ch) {
        for (auto& evt : channel_windows[ch].events) {
          cudaEventDestroy(evt);
        }
      }
    }

    // Release TCPX/CUDA resources
    mgr.deregister_memory(true);
    cuMemFree(d_base);
    cuDevicePrimaryCtxRelease(cuDev);
    mgr.close_all(true);
    close(bootstrap_fd);

    // ============================================================================
    // CLIENT logic
    // ============================================================================

  } else {
    std::cout << "[PERF] Starting CLIENT mode" << std::endl;

    // ==========================================================================
    // Step 1: bootstrap connect (receive handles)
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
    // Step 2: create ChannelManager and connect all channels
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
    // Step 3: CUDA initialization and memory allocation
    // ==========================================================================

    CUdevice cuDev;
    CUcontext cuCtx;
    CUdeviceptr d_base, d_aligned;

    if (!cuda_check(cuInit(0), "cuInit") ||
        // When send window is full: wait for oldest send (tcpx_test + sleep)
        // Send auto-releases; just remove from the window vector

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

    // Align to 4KB boundary
    uintptr_t addr = static_cast<uintptr_t>(d_base);
    addr = (addr + 4095) & ~static_cast<uintptr_t>(4095);
    d_aligned = static_cast<CUdeviceptr>(addr);
    void* send_buf = reinterpret_cast<void*>(d_aligned);

    std::cout << "[PERF] Allocated send buffer: " << send_buf << std::endl;

    // ==========================================================================
    // Step 4: register send buffer to all channels (shared)
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
    // Step 5: configure per-channel send windows
    // ==========================================================================

    // Client uses 12 instead of 16 to keep headroom
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
    // Step 6: main benchmark loop
    // ==========================================================================

    double total_time_ms = 0.0;

    for (int iter = 0; iter < iterations; ++iter) {
      std::cout << "[PERF] Iteration " << iter << ": total bytes="
                << test_size
                // End-of-iteration: drain pending sends per channel (poll done
                // and remove from window; no consumed step needed)

                << ", chunk_bytes=" << chunk_bytes << std::endl;

      auto start = std::chrono::high_resolution_clock::now();

      // Reset all per-channel send windows at iteration start
      std::cout << "[DEBUG] Iteration " << iter
                << " start: clearing send windows for " << num_channels
                << " channels" << std::endl;
      for (int ch = 0; ch < num_channels; ++ch) {
        send_windows[ch].pending_reqs.clear();
        send_windows[ch].chunk_counter = 0;
      }

      // ========================================================================
      // Chunk loop: split into chunks; round-robin to channels
      // ========================================================================

      size_t offset = 0;
      int global_chunk_idx = 0;

      while (offset < test_size) {
        const size_t this_chunk = std::min(chunk_bytes, test_size - offset);
        void* src_ptr = reinterpret_cast<void*>(
            reinterpret_cast<uintptr_t>(send_buf) + offset);

        // Round-robin channel selection
        int channel_id = global_chunk_idx % num_channels;
        ChannelResources& ch = mgr.get_channel(channel_id);
        SendChannelWindow& win = send_windows[channel_id];

        // Unique tag per chunk (must match server)
        int const tag = kTransferTag + iter * 10000 + global_chunk_idx;

        std::cout << "[DEBUG][CLIENT] chunk=" << global_chunk_idx
                  << " channel=" << channel_id << " tag=" << tag
                  << " size=" << this_chunk << " offset=" << offset
                  << " pending=" << win.pending_reqs.size() << "/"
                  << MAX_INFLIGHT_SEND_PER_CHANNEL << std::endl;

        // ======================================================================
        // Sliding window: if full, wait for oldest send
        // ======================================================================

        if (win.pending_reqs.size() >= MAX_INFLIGHT_SEND_PER_CHANNEL) {
          std::cout << "[DEBUG][CLIENT] Channel " << channel_id
                    << " send window FULL (" << win.pending_reqs.size() << "/"
                    << MAX_INFLIGHT_SEND_PER_CHANNEL
                    << "), waiting for oldest send" << std::endl;

          void* oldest_req = win.pending_reqs.front();
          int done = 0, sent_size = 0;

          // Remove timeout; poll until send completes
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

          // Send auto-releases; just remove from the window
          win.pending_reqs.erase(win.pending_reqs.begin());

          std::cout << "[DEBUG][CLIENT] Channel " << channel_id
                    << " window now has " << win.pending_reqs.size()
                    << " pending sends" << std::endl;
        }

        // ======================================================================
        // Post async send (tcpx_isend)
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

        // Track the send request in channel window
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
      // End of iteration: drain all channel windows
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
      // Per-iteration performance
      // ========================================================================

      auto end = std::chrono::high_resolution_clock::now();
      double iter_time_ms =
          std::chrono::duration<double, std::milli>(end - start).count();
      total_time_ms += iter_time_ms;
      std::cout << "[PERF] Iter " << iter << " time=" << iter_time_ms << "ms"
                << std::endl;
    }  // end of iteration loop

    // ==========================================================================
    // Compute and print average
    // ==========================================================================

    double avg_ms = total_time_ms / iterations;
    double bw_gbps =
        (test_size / (1024.0 * 1024.0 * 1024.0)) / (avg_ms / 1000.0);

    std::cout << "[PERF] Avg: " << std::fixed << std::setprecision(3) << avg_ms
              << " ms, "
              << "BW: " << std::fixed << std::setprecision(2) << bw_gbps
              << " GB/s" << std::endl;

    // ==========================================================================
    // Cleanup resources
    // ==========================================================================

    mgr.deregister_memory(false);
    cuMemFree(d_base);
    cuDevicePrimaryCtxRelease(cuDev);
    mgr.close_all(false);
    close(bootstrap_fd);
  }  // end of client logic

  return 0;
}  // end of main