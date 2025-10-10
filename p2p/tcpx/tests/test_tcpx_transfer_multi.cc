/**
 * @file test_tcpx_transfer_multi.cc
 * @brief Multi-channel TCPX GPU-to-GPU end-to-end transfer validation
 *
 * This is a refactored version of test_tcpx_transfer.cc that uses the new
 * multi-channel infrastructure (ChannelManager, Bootstrap, SlidingWindow).
 *
 * Key improvements over original:
 * - Supports multiple TCPX channels (one per NIC)
 * - Uses ChannelManager for connection lifecycle
 * - Uses Bootstrap for multi-handle exchange
 * - Maintains all debugging experience from original
 *
 * This test validates the complete TCPX transfer pipeline including:
 * - Multi-channel TCPX connection establishment
 * - CUDA device buffer registration (shared across channels)
 * - Async send/receive operations
 * - RX metadata parsing (CMSG scatter-gather lists)
 * - Device unpack implementations (D2D and host-mediated)
 *
 * Server workflow:
 *   1. Create ChannelManager with N channels
 *   2. Listen on all channels, publish handles via bootstrap TCP
 *   3. Accept connections on all channels
 *   4. Register shared CUDA buffer, submit irecv on channel 0
 *   5. Poll for completion, parse RX metadata, execute unpack
 *   6. Validate received data against expected payload
 *
 * Client workflow:
 *   1. Connect to bootstrap TCP, receive handles
 *   2. Create ChannelManager, connect all channels
 *   3. Register shared CUDA buffer, write test payload
 *   4. Submit isend on channel 0
 *   5. Wait for completion and cleanup
 *
 * Environment variables:
 *   UCCL_TCPX_NUM_CHANNELS: Number of channels (default: 1 for compatibility)
 *   UCCL_TCPX_UNPACK_IMPL: Select unpack implementation (d2d|host|kernel)
 *   UCCL_TCPX_HOST_RECV_DEBUG: Use host buffer for recv (debugging)
 *   UCCL_TCPX_PAYLOAD_BYTES: Payload size (default: sizeof(kTestMessage))
 *
 * IMPORTANT DEBUGGING NOTES (from original test_tcpx_transfer.cc):
 * - Always set NCCL_MIN_ZCOPY_SIZE=4096 to avoid errqueue flakiness
 * - Always set NCCL_GPUDIRECTTCPX_RECV_SYNC=1 for deterministic recv
 * - tcpx_test returns size=0 for GPU path (this is expected)
 * - Must call tcpx_irecv_consumed after recv completes
 * - Accept may return nullptr initially (retry with backoff)
 * - Device handle must be 16-byte aligned
 * - Fragment count can be 0 if data not yet arrived (check cnt_cache)
 */

#include "../device/unpack_launch.h"
#include "../include/bootstrap.h"
#include "../include/channel_manager.h"
#include "../include/rx_descriptor.h"
#include "../include/tcpx_handles.h"
#include "../include/tcpx_interface.h"
#include "../include/tcpx_structs.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

namespace {
constexpr size_t kRegisteredBytes = 4096;  // 4 KB aligned allocation.

constexpr char kTestMessage[] = "Hello from TCPX client!";
constexpr size_t kDefaultPayloadBytes = sizeof(kTestMessage) - 1;
constexpr int kTransferTag = 42;

using NcclNetDeviceHandle = tcpx::plugin::NcclNetDeviceHandle;
using DevmemToken = tcpx::plugin::DevmemToken;
using TcpxUnpackSlot = tcpx::plugin::unpackSlot;
using TcpxRequest = tcpx::plugin::tcpxRequest;
using UnpackNetDeviceHandle = tcpx::plugin::unpackNetDeviceHandle;
using LoadMetaEntry = tcpx::plugin::loadMeta;
static_assert(sizeof(LoadMetaEntry) == 16, "loadMeta layout must match plugin");

constexpr int kNetDeviceUnpackMaxQueueDepth =
    tcpx::plugin::kNetDeviceUnpackMaxQueueDepth;
constexpr int kNetUnpackMaxSliceSize = tcpx::plugin::kNetUnpackMaxSliceSize;
constexpr int kSlicePageSize = tcpx::plugin::kSlicePageSize;
constexpr int kNetUnpackMaxSlicePages = tcpx::plugin::kNetUnpackMaxSlicePages;
struct NetUnpackMeta {
  LoadMetaEntry mem[kNetDeviceUnpackMaxQueueDepth][kNetUnpackMaxSlicePages];
  uint64_t cnt[kNetDeviceUnpackMaxQueueDepth];
};

void dump_hex(void const* data, size_t bytes) {
  unsigned char const* ptr = static_cast<unsigned char const*>(data);
  size_t limit = std::min<size_t>(bytes, 32);
  for (size_t i = 0; i < limit; ++i) {
    if (i && i % 16 == 0) std::cout << '\n';
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(ptr[i]) << ' ';
  }
  std::cout << std::dec << std::endl;
}

bool cuda_check(CUresult res, char const* what) {
  if (res == CUDA_SUCCESS) return true;

  char const* name = nullptr;
  char const* desc = nullptr;
  cuGetErrorName(res, &name);
  cuGetErrorString(res, &desc);

  std::cout << "[DEBUG] CUDA error at " << what << ": " << (name ? name : "?")
            << " - " << (desc ? desc : "") << std::endl;
  return false;
}

// Get number of channels from environment (default: 1 for backward
// compatibility)
int get_num_channels() {
  char const* env = std::getenv("UCCL_TCPX_NUM_CHANNELS");
  if (!env) return 1;

  char* endp = nullptr;
  long val = std::strtol(env, &endp, 10);
  if (endp && *endp == '\0' && val > 0 && val <= 64) {
    return static_cast<int>(val);
  }

  std::cerr << "[WARN] Invalid UCCL_TCPX_NUM_CHANNELS=" << env
            << ", using default=1" << std::endl;
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  // CRITICAL: Debug/stability knobs from original test
  // These settings are ESSENTIAL to avoid errqueue flakiness and timing issues
  setenv("UCCL_TCPX_DEBUG", "1", 0);
  setenv("NCCL_MIN_ZCOPY_SIZE", "4096", 0);
  setenv("NCCL_GPUDIRECTTCPX_MIN_ZCOPY_SIZE", "4096", 0);
  // Optional: make receive path more deterministic during debug
  setenv("NCCL_GPUDIRECTTCPX_RECV_SYNC", "1", 0);

  std::cout << "[DEBUG] === Multi-Channel TCPX GPU-to-GPU transfer test ==="
            << std::endl;

  if (argc < 2) {
    std::cout << "[DEBUG] Usage: " << argv[0] << " <server|client> [remote_ip]"
              << std::endl;
    std::cout << "[DEBUG] Environment variables:" << std::endl;
    std::cout
        << "[DEBUG]   UCCL_TCPX_NUM_CHANNELS: Number of channels (default: 1)"
        << std::endl;
    std::cout
        << "[DEBUG]   UCCL_TCPX_UNPACK_IMPL: d2d|host|kernel (default: d2d)"
        << std::endl;
    std::cout << "[DEBUG]   UCCL_TCPX_HOST_RECV_DEBUG: 1 to use host buffer"
              << std::endl;
    std::cout << "[DEBUG]   UCCL_TCPX_PAYLOAD_BYTES: Payload size (default: "
              << kDefaultPayloadBytes << ")" << std::endl;
    return 1;
  }

  int device_count = tcpx_get_device_count();
  if (device_count <= 0) {
    std::cout << "[DEBUG] ERROR: no TCPX devices detected" << std::endl;
    return 1;
  }

  int num_channels = get_num_channels();
  if (num_channels > device_count) {
    std::cout << "[DEBUG] WARNING: Requested " << num_channels
              << " channels but only " << device_count
              << " TCPX devices available. Clamping to " << device_count
              << std::endl;
    num_channels = device_count;
  }

  std::cout << "[DEBUG] Using " << num_channels << " channel(s)" << std::endl;

  int dev_id = 0;  // GPU device ID (not TCPX device ID)
  bool is_server = std::strcmp(argv[1], "server") == 0;

  size_t payload_bytes = kDefaultPayloadBytes;
  if (char const* p = std::getenv("UCCL_TCPX_PAYLOAD_BYTES")) {
    char* endp = nullptr;
    unsigned long v = std::strtoul(p, &endp, 10);
    if (endp && *endp == '\0' && v > 0) {
      if (v > kRegisteredBytes) v = kRegisteredBytes;
      payload_bytes = static_cast<size_t>(v);
    }
  }
  std::cout << "[DEBUG] Using payload_bytes=" << payload_bytes << std::endl;

  if (is_server) {
    std::cout << "[DEBUG] Running in SERVER mode" << std::endl;

    // Create ChannelManager for multi-channel support
    ChannelManager mgr(num_channels, dev_id);
    if (mgr.get_num_channels() == 0) {
      std::cout << "[DEBUG] ERROR: ChannelManager initialization failed"
                << std::endl;
      return 1;
    }

    std::cout << "[DEBUG] ChannelManager created with "
              << mgr.get_num_channels() << " channels" << std::endl;

    // Listen on all channels
    std::vector<ncclNetHandle_v7> handles;
    if (mgr.server_listen_all(handles) != 0) {
      std::cout << "[DEBUG] ERROR: server_listen_all failed" << std::endl;
      return 1;
    }
    std::cout << "[DEBUG] Listening on " << handles.size() << " channels"
              << std::endl;

    // Bootstrap: send handles to client
    int client_fd = -1;
    if (bootstrap_server_create(kBootstrapPort, &client_fd) != 0) {
      std::cout << "[DEBUG] ERROR: bootstrap server creation failed"
                << std::endl;
      mgr.close_all(true);
      return 1;
    }
    std::cout << "[DEBUG] Bootstrap connection established" << std::endl;

    if (bootstrap_server_send_handles(client_fd, handles) != 0) {
      std::cout << "[DEBUG] ERROR: failed to send handles" << std::endl;
      close(client_fd);
      mgr.close_all(true);
      return 1;
    }
    std::cout << "[DEBUG] Sent " << handles.size() << " handles to client"
              << std::endl;

    // Accept connections on all channels
    if (mgr.server_accept_all() != 0) {
      std::cout << "[DEBUG] ERROR: server_accept_all failed" << std::endl;
      close(client_fd);
      mgr.close_all(true);
      return 1;
    }
    std::cout << "[DEBUG] All channels accepted" << std::endl;

    // CUDA initialization
    CUdevice cuDev = 0;
    CUcontext cuCtx = nullptr;
    CUdeviceptr d_base = 0;
    CUdeviceptr d_aligned = 0;
    void* recv_buf = nullptr;
    void* recv_request = nullptr;
    bool use_host_recv = false;
    bool request_posted = false;
    bool request_consumed = false;
    int done = 0;  // IMPORTANT: int not bool for tcpx_test
    int received_size = 0;

    auto cleanup_server = [&]() {
      // CRITICAL: Must call tcpx_irecv_consumed after recv completes
      // Get channel 0 (we use channel 0 for this single-transfer test)
      ChannelResources& ch = mgr.get_channel(0);

      if (request_posted && recv_request && done && !request_consumed) {
        int rc_consumed = tcpx_irecv_consumed(ch.recv_comm, 1, recv_request);
        std::cout << "[DEBUG] tcpx_irecv_consumed(rc=" << rc_consumed
                  << ") request=" << recv_request << std::endl;
        if (rc_consumed == 0) {
          request_consumed = true;
        }
      } else if (request_posted && !done) {
        std::cout
            << "[DEBUG] Skipping irecv_consumed because request not complete"
            << std::endl;
      }

      mgr.deregister_memory(true);

      if (use_host_recv && recv_buf) {
        cudaFreeHost(recv_buf);
        recv_buf = nullptr;
      }
      if (d_base) {
        cuMemFree(d_base);
        d_base = 0;
      }

      mgr.close_all(true);

      if (client_fd >= 0) {
        close(client_fd);
        client_fd = -1;
      }
      if (cuCtx) {
        cuDevicePrimaryCtxRelease(cuDev);
        cuCtx = nullptr;
      }
    };

    auto fail_server = [&](char const* message) -> int {
      if (message && *message) {
        std::cout << message << std::endl;
      }
      cleanup_server();
      return 1;
    };

    if (!cuda_check(cuInit(0), "cuInit") ||
        !cuda_check(cuDeviceGet(&cuDev, dev_id), "cuDeviceGet") ||
        !cuda_check(cuDevicePrimaryCtxRetain(&cuCtx, cuDev),
                    "cuDevicePrimaryCtxRetain") ||
        !cuda_check(cuCtxSetCurrent(cuCtx), "cuCtxSetCurrent") ||
        !cuda_check(cuMemAlloc(&d_base, kRegisteredBytes + 4096),
                    "cuMemAlloc")) {
      return fail_server(
          "[DEBUG] ERROR: failed to allocate server receive buffer");
    }

    if (cudaSetDevice(dev_id) != cudaSuccess) {
      return fail_server("[DEBUG] ERROR: cudaSetDevice failed for server");
    }

    uintptr_t addr = static_cast<uintptr_t>(d_base);
    addr = (addr + 4095) & ~static_cast<uintptr_t>(4095);
    d_aligned = static_cast<CUdeviceptr>(addr);
    std::cout << "[DEBUG] Server GPU buffer aligned to 0x" << std::hex
              << d_aligned << " (base 0x" << d_base << ")" << std::dec
              << std::endl;

    int recv_ptr_type = NCCL_PTR_CUDA;
    if (char const* env = std::getenv("UCCL_TCPX_HOST_RECV_DEBUG")) {
      char flag = env[0];
      if (flag == '1' || flag == 't' || flag == 'T' || flag == 'y' ||
          flag == 'Y') {
        use_host_recv = true;
      }
    }

    if (use_host_recv) {
      void* host_aligned = nullptr;
      if (cudaMallocHost(&host_aligned, kRegisteredBytes) != cudaSuccess ||
          !host_aligned) {
        return fail_server(
            "[DEBUG] ERROR: cudaMallocHost failed for server receive buffer");
      }
      std::memset(host_aligned, 0, kRegisteredBytes);
      recv_buf = host_aligned;
      recv_ptr_type = NCCL_PTR_HOST;
      std::cout << "[DEBUG] Host fallback enabled; pinned buffer=" << recv_buf
                << std::endl;
    } else {
      recv_buf = reinterpret_cast<void*>(d_aligned);
    }

    // Register memory with all channels (shared buffer approach)
    if (mgr.register_memory(recv_buf, kRegisteredBytes, recv_ptr_type, true) !=
        0) {
      return fail_server(
          "[DEBUG] ERROR: register_memory failed for server receive buffer");
    }
    std::cout << "[DEBUG] Registered server receive buffer ptr=" << recv_buf
              << ", bytes=" << kRegisteredBytes << " with "
              << mgr.get_num_channels() << " channels" << std::endl;

    // For this simple test, we use channel 0 for the transfer
    // (Multi-channel data distribution will be in test_tcpx_perf)
    ChannelResources& ch0 = mgr.get_channel(0);

    void* recv_data[1] = {recv_buf};
    int recv_sizes[1] = {static_cast<int>(payload_bytes)};
    int recv_tags[1] = {kTransferTag};
    void* recv_mhandles[1] = {ch0.mhandle};

    if (tcpx_irecv(ch0.recv_comm, 1, recv_data, recv_sizes, recv_tags,
                   recv_mhandles, &recv_request) != 0) {
      return fail_server("[DEBUG] ERROR: tcpx_irecv failed on server");
    }
    request_posted = true;
    std::cout << "[DEBUG] Waiting for client data on channel 0, expected bytes="
              << payload_bytes << std::endl;

    // Poll for completion (same logic as original)
    int poll_iterations = 0;
    for (; poll_iterations < 200000 && !done; ++poll_iterations) {
      int rc_test = tcpx_test(recv_request, &done, &received_size);
      if (rc_test != 0) {
        std::cout << "[DEBUG] ERROR: tcpx_test returned " << rc_test
                  << std::endl;
        break;
      }
      if (!done) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
      }
      if (poll_iterations > 0 && poll_iterations % 1000 == 0) {
        std::cout << "[DEBUG] Polling progress: iteration=" << poll_iterations
                  << " done=" << done << " received_size=" << received_size
                  << std::endl;
      }
    }
    std::cout << "[DEBUG] tcpx_test poll loop iterations=" << poll_iterations
              << " done=" << done << " received_size=" << received_size
              << std::endl;

    // IMPORTANT: tcpx_test returns size=0 for GPU path (this is expected)
    if (done && received_size == 0) {
      std::cout << "[DEBUG] tcpx_test reported size=0 (expected for GPU path)"
                << std::endl;
    }

    std::vector<unsigned char> host(payload_bytes, 0);
    bool success = false;
    bool copy_ok = false;
    size_t bytes_copied = 0;

    // Unpack logic (same as original test_tcpx_transfer.cc)
    if (!done) {
      std::cout << "[DEBUG] ERROR: receive timed out" << std::endl;
    } else if (use_host_recv) {
      std::memcpy(host.data(), recv_buf, payload_bytes);
      copy_ok = true;
      bytes_copied = payload_bytes;
    } else {
      // GPU path: parse RX metadata and unpack scattered buffers
      auto* rx_req = reinterpret_cast<TcpxRequest*>(recv_request);
      auto* dev_handle_struct =
          reinterpret_cast<NcclNetDeviceHandle*>(ch0.recv_dev_handle);
      uint64_t frag_count =
          (rx_req && rx_req->unpack_slot.cnt) ? *(rx_req->unpack_slot.cnt) : 0;

      std::cout << "[DEBUG] Request metadata: request_ptr=" << rx_req
                << " active=" << (rx_req ? rx_req->unpack_slot.active : 0)
                << " idx=" << (rx_req ? rx_req->unpack_slot.idx : 0)
                << " cnt_cache=" << (rx_req ? rx_req->unpack_slot.cnt_cache : 0)
                << " cnt_ptr="
                << (rx_req && rx_req->unpack_slot.cnt ? rx_req->unpack_slot.cnt
                                                      : nullptr)
                << " meta_ptr=" << (rx_req ? rx_req->unpack_slot.mem : nullptr)
                << " size=" << (rx_req ? rx_req->size : 0)
                << " size_pending=" << (rx_req ? rx_req->size_pending : 0)
                << " frag_count=" << frag_count << std::endl;

      if (!rx_req || !dev_handle_struct || !rx_req->unpack_slot.mem ||
          !rx_req->unpack_slot.cnt) {
        std::cout << "[DEBUG] ERROR: missing TCPX metadata for device unpack"
                  << std::endl;
      } else if (frag_count == 0) {
        std::cout << "[DEBUG] ERROR: unpack metadata contains zero fragments "
                     "(cnt_cache="
                  << rx_req->unpack_slot.cnt_cache << ")" << std::endl;
      } else if (frag_count > MAX_UNPACK_DESCRIPTORS) {
        std::cout << "[DEBUG] ERROR: fragment count " << frag_count
                  << " exceeds descriptor capacity" << std::endl;
      } else {
        auto* meta_entries =
            static_cast<LoadMetaEntry*>(rx_req->unpack_slot.mem);
        UnpackNetDeviceHandle dev_handle{};
        if (!cuda_check(cuMemcpyDtoH(&dev_handle,
                                     reinterpret_cast<CUdeviceptr>(
                                         dev_handle_struct->handle),
                                     sizeof(dev_handle)),
                        "cuMemcpyDtoH(device_handle)")) {
          std::cout << "[DEBUG] ERROR: failed to read device handle metadata"
                    << std::endl;
        } else {
          std::cout << "[DEBUG] Device handle: meta=" << dev_handle.meta
                    << " bounce_buf=" << dev_handle.bounce_buf
                    << " head=" << dev_handle.head << std::endl;

          // Build descriptor block using simplified utility function
          tcpx::rx::UnpackDescriptorBlock desc_block;
          tcpx::rx::buildDescriptorBlock(
              meta_entries, static_cast<uint32_t>(frag_count),
              dev_handle.bounce_buf, reinterpret_cast<void*>(d_aligned),
              desc_block);

          // Set optional ready flag for device-side visibility barrier
          desc_block.ready_flag = rx_req->unpack_slot.cnt;
          desc_block.ready_threshold = frag_count;

          // Print descriptor details
          for (uint32_t i = 0; i < desc_block.count; ++i) {
            std::cout << "[DEBUG] descriptor[" << i
                      << "] src_off=" << desc_block.descriptors[i].src_off
                      << " len=" << desc_block.descriptors[i].len
                      << " dst_off=" << desc_block.descriptors[i].dst_off
                      << std::endl;
          }

          bool device_copy_ok = true;

          // Select unpack implementation: kernel | d2d | host
          char const* impl_env = std::getenv("UCCL_TCPX_UNPACK_IMPL");
          std::string impl =
              impl_env ? std::string(impl_env) : std::string("d2d");
          std::transform(impl.begin(), impl.end(), impl.begin(), ::tolower);

          if (impl == "kernel") {
            std::cout
                << "[DEBUG] Launching device unpack (kernel), total_bytes="
                << desc_block.total_bytes << std::endl;

            cudaStream_t dedicated_stream;
            if (cudaStreamCreate(&dedicated_stream) != cudaSuccess) {
              std::cout << "[DEBUG] ERROR: Failed to create dedicated stream"
                        << std::endl;
              device_copy_ok = false;
            } else {
              tcpx::device::UnpackLaunchConfig cfg;
              cfg.stream = dedicated_stream;
              cfg.enable_profiling = false;
              cfg.use_small_kernel = true;

              tcpx::device::UnpackLauncher launcher(cfg);
              int lrc = launcher.launchSync(desc_block);

              if (lrc != 0) {
                std::cout << "[DEBUG] ERROR: Unpack kernel failed rc=" << lrc
                          << std::endl;
                device_copy_ok = false;
              } else {
                std::cout << "[DEBUG] Unpack kernel completed successfully"
                          << std::endl;
              }

              cudaStreamDestroy(dedicated_stream);
            }
          } else if (impl == "host") {
            std::cout << "[DEBUG] Launching host gather (DtoH+memcpy+HtoD), "
                         "total_bytes="
                      << desc_block.total_bytes << std::endl;
            // Slow but robust path for debugging
            std::vector<unsigned char> tmp(desc_block.total_bytes);
            size_t off = 0;
            for (uint32_t i = 0; i < desc_block.count; ++i) {
              auto const& meta = desc_block.descriptors[i];
              CUdeviceptr src_ptr = static_cast<CUdeviceptr>(
                  reinterpret_cast<uintptr_t>(desc_block.bounce_buffer) +
                  meta.src_off);
              if (!cuda_check(cuMemcpyDtoH(tmp.data() + off, src_ptr, meta.len),
                              "cuMemcpyDtoH(fragment)")) {
                device_copy_ok = false;
                break;
              }
              off += meta.len;
            }
            if (device_copy_ok) {
              if (!cuda_check(cuMemcpyHtoD(static_cast<CUdeviceptr>(
                                               reinterpret_cast<uintptr_t>(
                                                   desc_block.dst_buffer)),
                                           tmp.data(), tmp.size()),
                              "cuMemcpyHtoD(dst)")) {
                device_copy_ok = false;
              }
            }
          } else {
            std::cout
                << "[DEBUG] Launching device unpack (D2D copies), total_bytes="
                << desc_block.total_bytes << std::endl;
            // Default: D2D copy per fragment (current fallback)
            for (uint32_t i = 0; i < desc_block.count; ++i) {
              auto const& meta = desc_block.descriptors[i];
              CUdeviceptr src_ptr = static_cast<CUdeviceptr>(
                  reinterpret_cast<uintptr_t>(desc_block.bounce_buffer) +
                  meta.src_off);
              CUdeviceptr dst_ptr = static_cast<CUdeviceptr>(
                  reinterpret_cast<uintptr_t>(desc_block.dst_buffer) +
                  meta.dst_off);
              if (!cuda_check(cuMemcpyDtoD(dst_ptr, src_ptr, meta.len),
                              "cuMemcpyDtoD(fragment)")) {
                device_copy_ok = false;
                break;
              }
            }
          }

          // Post-copy verification: bring dst buffer back to host
          if (!device_copy_ok) {
            std::cout << "[DEBUG] ERROR: device copy failed" << std::endl;
          } else if (!cuda_check(
                         cuMemcpyDtoH(host.data(), d_aligned, payload_bytes),
                         "cuMemcpyDtoH(dst)")) {
            std::cout
                << "[DEBUG] ERROR: failed to copy unpacked payload to host"
                << std::endl;
            device_copy_ok = false;
          } else {
            copy_ok = true;
            bytes_copied = desc_block.total_bytes;
          }
        }
      }
    }

    if (copy_ok) {
      dump_hex(host.data(), std::min<size_t>(payload_bytes, 32));
      size_t prefix = std::min(payload_bytes, sizeof(kTestMessage) - 1);
      bool prefix_ok = std::memcmp(host.data(), kTestMessage, prefix) == 0;
      bool tail_ok =
          (payload_bytes <= prefix) || host[payload_bytes - 1] == 0xAB;
      success = prefix_ok && tail_ok;
      std::cout << "[DEBUG] Receive completed, bytes=" << bytes_copied
                << std::endl;
      if (success) {
        std::cout << "[DEBUG] SUCCESS: payload matches expected string"
                  << std::endl;
        if (client_fd >= 0) {
          char ack = 1;
          (void)send(client_fd, &ack, 1, 0);
        }
      } else {
        std::cout << "[DEBUG] ERROR: payload mismatch" << std::endl;
      }
    } else if (done) {
      std::cout << "[DEBUG] ERROR: device unpack failed" << std::endl;
    }

    cleanup_server();
    return success ? 0 : 1;

  } else {
    // CLIENT MODE
    if (argc < 3) {
      std::cout << "[DEBUG] ERROR: client mode requires <remote_ip>"
                << std::endl;
      return 1;
    }
    char const* remote_ip = argv[2];
    std::cout << "[DEBUG] Running in CLIENT mode, connecting to " << remote_ip
              << std::endl;

    // Bootstrap: connect and receive handles
    int bootstrap_fd = -1;
    if (bootstrap_client_connect(remote_ip, kBootstrapPort, &bootstrap_fd) !=
        0) {
      std::cout << "[DEBUG] ERROR: bootstrap connect failed" << std::endl;
      return 1;
    }
    std::cout << "[DEBUG] Bootstrap connected" << std::endl;

    std::vector<ncclNetHandle_v7> handles;
    if (bootstrap_client_recv_handles(bootstrap_fd, handles) != 0) {
      std::cout << "[DEBUG] ERROR: failed to receive handles" << std::endl;
      close(bootstrap_fd);
      return 1;
    }
    std::cout << "[DEBUG] Received " << handles.size() << " handles from server"
              << std::endl;

    // Create ChannelManager and connect all channels
    ChannelManager mgr(static_cast<int>(handles.size()), dev_id);
    if (mgr.get_num_channels() == 0) {
      std::cout << "[DEBUG] ERROR: ChannelManager initialization failed"
                << std::endl;
      close(bootstrap_fd);
      return 1;
    }

    if (mgr.client_connect_all(handles) != 0) {
      std::cout << "[DEBUG] ERROR: client_connect_all failed" << std::endl;
      close(bootstrap_fd);
      mgr.close_all(false);
      return 1;
    }
    std::cout << "[DEBUG] All " << mgr.get_num_channels()
              << " channels connected" << std::endl;

    // CUDA initialization
    CUdevice cuDev = 0;
    CUcontext cuCtx = nullptr;
    CUdeviceptr d_base = 0;
    CUdeviceptr d_aligned = 0;
    void* send_request = nullptr;
    int done = 0;  // IMPORTANT: int not bool for tcpx_test
    int sent_size = 0;

    auto cleanup_client = [&]() {
      mgr.deregister_memory(false);

      if (d_base) {
        cuMemFree(d_base);
        d_base = 0;
      }

      mgr.close_all(false);

      if (bootstrap_fd >= 0) {
        close(bootstrap_fd);
        bootstrap_fd = -1;
      }
      if (cuCtx) {
        cuDevicePrimaryCtxRelease(cuDev);
        cuCtx = nullptr;
      }
    };

    auto fail_client = [&](char const* message) -> int {
      if (message && *message) {
        std::cout << message << std::endl;
      }
      cleanup_client();
      return 1;
    };

    if (!cuda_check(cuInit(0), "cuInit") ||
        !cuda_check(cuDeviceGet(&cuDev, dev_id), "cuDeviceGet") ||
        !cuda_check(cuDevicePrimaryCtxRetain(&cuCtx, cuDev),
                    "cuDevicePrimaryCtxRetain") ||
        !cuda_check(cuCtxSetCurrent(cuCtx), "cuCtxSetCurrent") ||
        !cuda_check(cuMemAlloc(&d_base, kRegisteredBytes + 4096),
                    "cuMemAlloc")) {
      return fail_client(
          "[DEBUG] ERROR: failed to allocate client send buffer");
    }

    if (cudaSetDevice(dev_id) != cudaSuccess) {
      return fail_client("[DEBUG] ERROR: cudaSetDevice failed for client");
    }

    uintptr_t addr = static_cast<uintptr_t>(d_base);
    addr = (addr + 4095) & ~static_cast<uintptr_t>(4095);
    d_aligned = static_cast<CUdeviceptr>(addr);

    // Prepare payload
    std::vector<unsigned char> host_payload(payload_bytes, 0);
    size_t prefix = std::min(payload_bytes, sizeof(kTestMessage) - 1);
    if (prefix > 0) {
      std::memcpy(host_payload.data(), kTestMessage, prefix);
    }
    if (payload_bytes > prefix) {
      host_payload[payload_bytes - 1] = 0xAB;
    }

    cuda_check(cuMemcpyHtoD(d_aligned, host_payload.data(), payload_bytes),
               "cuMemcpyHtoD");
    cuCtxSynchronize();

    // Register memory with all channels (shared buffer approach)
    if (mgr.register_memory(reinterpret_cast<void*>(d_aligned),
                            kRegisteredBytes, NCCL_PTR_CUDA, false) != 0) {
      return fail_client(
          "[DEBUG] ERROR: register_memory failed for client send buffer");
    }
    std::cout << "[DEBUG] Registered client send buffer ptr="
              << reinterpret_cast<void*>(d_aligned)
              << ", bytes=" << kRegisteredBytes << " with "
              << mgr.get_num_channels() << " channels" << std::endl;

    // For this simple test, we use channel 0 for the transfer
    ChannelResources& ch0 = mgr.get_channel(0);

    if (tcpx_isend(ch0.send_comm, reinterpret_cast<void*>(d_aligned),
                   static_cast<int>(payload_bytes), kTransferTag, ch0.mhandle,
                   &send_request) != 0) {
      return fail_client("[DEBUG] ERROR: tcpx_isend failed");
    }

    std::cout << "[DEBUG] tcpx_isend returned rc=0, request=" << send_request
              << std::endl;

    // Poll for completion (same logic as original)
    int send_iterations = 0;
    for (; send_iterations < 200000 && !done; ++send_iterations) {
      int rc_test = tcpx_test(send_request, &done, &sent_size);
      if (rc_test != 0) {
        std::cout << "[DEBUG] ERROR: tcpx_test (client) returned " << rc_test
                  << std::endl;
        break;
      }
      if (!done) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
      }
      if (send_iterations > 0 && send_iterations % 1000 == 0) {
        std::cout << "[DEBUG] Send polling: iteration=" << send_iterations
                  << " done=" << done << " sent_size=" << sent_size
                  << std::endl;
      }
    }
    std::cout << "[DEBUG] Send loop iterations=" << send_iterations
              << " done=" << done << " sent_size=" << sent_size << std::endl;

    if (done) {
      // IMPORTANT: tcpx_test returns size=0 for GPU path (this is expected)
      if (sent_size == 0) {
        std::cout
            << "[DEBUG] tcpx_test reported size=0 for send (expected when GPU"
               " completions are handled internally)"
            << std::endl;
      }
      std::cout << "[DEBUG] Send completed (payload=" << payload_bytes
                << " bytes)" << std::endl;
    } else {
      std::cout << "[DEBUG] WARNING: send did not complete before timeout"
                << std::endl;
    }

    // Wait for server ACK
    if (bootstrap_fd >= 0) {
      timeval tv{};
      tv.tv_sec = 2;
      tv.tv_usec = 0;
      setsockopt(bootstrap_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
      char ack = 0;
      ssize_t r = recv(bootstrap_fd, &ack, 1, 0);
      if (r == 1 && ack == 1) {
        std::cout << "[DEBUG] Received server ACK" << std::endl;
      } else {
        std::cout << "[DEBUG] WARNING: did not receive server ACK (recv=" << r
                  << ", ack=" << static_cast<int>(ack) << ")" << std::endl;
      }
    }

    cleanup_client();
    return done ? 0 : 1;
  }
}
