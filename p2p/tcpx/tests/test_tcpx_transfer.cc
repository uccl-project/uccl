/**
 * @file test_tcpx_transfer.cc
 * @brief TCPX GPU-to-GPU end-to-end transfer validation
 *
 * This test validates the complete TCPX transfer pipeline including:
 * - TCPX connection establishment (listen/accept/connect)
 * - CUDA device buffer registration
 * - Async send/receive operations
 * - RX metadata parsing (CMSG scatter-gather lists)
 * - Device unpack implementations (D2D and host-mediated)
 *
 * Server workflow:
 *   1. Listen on device 0, publish NCCL handle via bootstrap TCP
 *   2. Accept connection, register 4KB CUDA buffer, submit irecv
 *   3. Poll for completion, parse RX metadata, execute unpack
 *   4. Validate received data against expected payload
 *
 * Client workflow:
 *   1. Connect to server via bootstrap TCP and TCPX
 *   2. Register 4KB CUDA buffer, write test payload, submit isend
 *   3. Wait for completion and cleanup
 *
 * Environment variables:
 *   UCCL_TCPX_UNPACK_IMPL: Select unpack implementation (d2d|host|kernel)
 *     - d2d (default): Device-to-device memcpy per fragment
 *     - host: Host-mediated gather (DtoH + memcpy + HtoD)
 *     - kernel: CUDA kernel-based unpack (experimental, not in this PR)
 */

#include "../device/unpack_launch.h"
#include "../include/rx_descriptor.h"
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
constexpr size_t kHandleBytes = 128;
struct ncclNetHandle_v7 {
  char data[kHandleBytes];
};

constexpr int kBootstrapPort = 12345;
constexpr size_t kRegisteredBytes = 4096;  // 4 KB aligned allocation.

constexpr char kTestMessage[] = "Hello from TCPX client!";
constexpr size_t kDefaultPayloadBytes = sizeof(kTestMessage) - 1;
constexpr int kTransferTag = 42;
constexpr int kAcceptMaxRetries = 120;  // ~12 s (100 ms per retry).

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
int create_bootstrap_server() {
  int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd < 0) return -1;
  int opt = 1;
  setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(kBootstrapPort);
  if (bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    close(listen_fd);
    return -1;
  }
  if (listen(listen_fd, 1) < 0) {
    close(listen_fd);
    return -1;
  }
  int client_fd = accept(listen_fd, nullptr, nullptr);
  close(listen_fd);
  return client_fd;
}

int connect_to_bootstrap_server(char const* server_ip) {
  int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (sock_fd < 0) return -1;

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(kBootstrapPort);
  inet_aton(server_ip, &addr.sin_addr);

  for (int retry = 0; retry < 10; ++retry) {
    if (connect(sock_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) ==
        0) {
      return sock_fd;
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  close(sock_fd);
  return -1;
}

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
}  // namespace

int main(int argc, char** argv) {
  // Debug/stability knobs: avoid zero-copy for tiny payloads to reduce errqueue
  // flakiness.
  setenv("UCCL_TCPX_DEBUG", "1", 0);
  setenv("NCCL_MIN_ZCOPY_SIZE", "4096", 0);
  setenv("NCCL_GPUDIRECTTCPX_MIN_ZCOPY_SIZE", "4096", 0);
  // Optional: make receive path more deterministic during debug.
  setenv("NCCL_GPUDIRECTTCPX_RECV_SYNC", "1", 0);

  std::cout << "[DEBUG] === TCPX GPU-to-GPU transfer test ===" << std::endl;
  if (argc < 2) {
    std::cout << "[DEBUG] Usage: " << argv[0] << " <server|client> [remote_ip]"
              << std::endl;
    return 1;
  }
  int device_count = tcpx_get_device_count();
  if (device_count <= 0) {
    std::cout << "[DEBUG] ERROR: no TCPX devices detected" << std::endl;
    return 1;
  }
  int dev_id = 0;
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

    ncclNetHandle_v7 handle{};
    void* listen_comm = nullptr;
    if (tcpx_listen(dev_id, &handle, &listen_comm) != 0) {
      std::cout << "[DEBUG] ERROR: tcpx_listen failed" << std::endl;
      return 1;
    }
    std::cout << "[DEBUG] Listening on device " << dev_id << std::endl;

    int bootstrap_fd = create_bootstrap_server();
    if (bootstrap_fd < 0) {
      std::cout << "[DEBUG] ERROR: bootstrap server creation failed"
                << std::endl;
      tcpx_close_listen(listen_comm);
      return 1;
    }
    std::cout << "[DEBUG] Bootstrap connection established, sending handle"
              << std::endl;

    size_t total_sent = 0;
    while (total_sent < kHandleBytes) {
      ssize_t sent = send(bootstrap_fd, handle.data + total_sent,
                          kHandleBytes - total_sent, 0);
      if (sent <= 0) {
        std::cout << "[DEBUG] ERROR: failed to send NCCL handle" << std::endl;
        close(bootstrap_fd);
        tcpx_close_listen(listen_comm);
        return 1;
      }
      total_sent += static_cast<size_t>(sent);
    }

    void* recv_comm = nullptr;
    alignas(16) unsigned char recv_dev_handle_storage[512] = {0};
    void* recv_dev_handle = recv_dev_handle_storage;

    int attempts = 0;
    while (attempts < kAcceptMaxRetries) {
      int rc = tcpx_accept_v5(listen_comm, &recv_comm, &recv_dev_handle);
      if (rc != 0) {
        std::cout << "[DEBUG] ERROR: tcpx_accept_v5 returned rc=" << rc
                  << std::endl;
        tcpx_close_listen(listen_comm);
        close(bootstrap_fd);
        return 1;
      }
      if (recv_comm) break;
      ++attempts;
      if (attempts % 10 == 0) {
        std::cout << "[DEBUG] INFO: waiting for peer handshake (attempt "
                  << attempts << ")" << std::endl;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!recv_comm) {
      std::cout << "[DEBUG] ERROR: failed to obtain recv_comm after retries"
                << std::endl;
      tcpx_close_listen(listen_comm);
      close(bootstrap_fd);
      return 1;
    }
    std::cout << "[DEBUG] Connection accepted; recv_comm=" << recv_comm
              << std::endl;

    CUdevice cuDev = 0;
    CUcontext cuCtx = nullptr;
    CUdeviceptr d_base = 0;
    CUdeviceptr d_aligned = 0;
    void* recv_buf = nullptr;
    void* recv_mhandle = nullptr;
    void* recv_request = nullptr;
    bool use_host_recv = false;
    bool request_posted = false;
    bool request_consumed = false;
    int done = 0;  // Changed from bool to int for tcpx_test
    int received_size = 0;

    auto cleanup_server = [&]() {
      if (request_posted && recv_request && done && !request_consumed) {
        int rc_consumed = tcpx_irecv_consumed(recv_comm, 1, recv_request);
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
      if (recv_mhandle) {
        tcpx_dereg_mr(recv_comm, recv_mhandle);
        recv_mhandle = nullptr;
      }
      if (use_host_recv && recv_buf) {
        cudaFreeHost(recv_buf);
        recv_buf = nullptr;
      }
      if (d_base) {
        cuMemFree(d_base);
        d_base = 0;
      }
      if (recv_comm) {
        tcpx_close_recv(recv_comm);
        recv_comm = nullptr;
      }
      if (listen_comm) {
        tcpx_close_listen(listen_comm);
        listen_comm = nullptr;
      }
      if (bootstrap_fd >= 0) {
        close(bootstrap_fd);
        bootstrap_fd = -1;
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

    if (tcpx_reg_mr(recv_comm, recv_buf, kRegisteredBytes, recv_ptr_type,
                    &recv_mhandle) != 0) {
      return fail_server(
          "[DEBUG] ERROR: tcpx_reg_mr failed for server receive buffer");
    }

    std::cout << "[DEBUG] Registered server receive buffer ptr=" << recv_buf
              << ", bytes=" << kRegisteredBytes << std::endl;
    void* recv_data[1] = {recv_buf};
    int recv_sizes[1] = {static_cast<int>(payload_bytes)};
    int recv_tags[1] = {kTransferTag};
    void* recv_mhandles[1] = {recv_mhandle};
    if (tcpx_irecv(recv_comm, 1, recv_data, recv_sizes, recv_tags,
                   recv_mhandles, &recv_request) != 0) {
      return fail_server("[DEBUG] ERROR: tcpx_irecv failed on server");
    }
    request_posted = true;
    std::cout << "[DEBUG] Waiting for client data, expected bytes="
              << payload_bytes << std::endl;
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
    if (done && received_size == 0) {
      std::cout << "[DEBUG] tcpx_test reported size=0 (expected for GPU path)"
                << std::endl;
    }

    std::vector<unsigned char> host(payload_bytes, 0);
    bool success = false;
    bool copy_ok = false;
    size_t bytes_copied = 0;

    if (!done) {
      std::cout << "[DEBUG] ERROR: receive timed out" << std::endl;
    } else if (use_host_recv) {
      std::memcpy(host.data(), recv_buf, payload_bytes);
      copy_ok = true;
      bytes_copied = payload_bytes;
    } else {
      auto* rx_req = reinterpret_cast<TcpxRequest*>(recv_request);
      auto* dev_handle_struct =
          reinterpret_cast<NcclNetDeviceHandle*>(recv_dev_handle);
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
        if (bootstrap_fd >= 0) {
          char ack = 1;
          (void)send(bootstrap_fd, &ack, 1, 0);
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
    if (argc < 3) {
      std::cout << "[DEBUG] ERROR: client mode requires <remote_ip>"
                << std::endl;
      return 1;
    }
    char const* remote_ip = argv[2];
    std::cout << "[DEBUG] Running in CLIENT mode, connecting to " << remote_ip
              << std::endl;

    int bootstrap_fd = connect_to_bootstrap_server(remote_ip);
    if (bootstrap_fd < 0) {
      std::cout << "[DEBUG] ERROR: bootstrap connect failed" << std::endl;
      return 1;
    }

    ncclNetHandle_v7 handle{};
    size_t total_received = 0;
    while (total_received < kHandleBytes) {
      ssize_t r = recv(bootstrap_fd, handle.data + total_received,
                       kHandleBytes - total_received, 0);
      if (r <= 0) {
        std::cout << "[DEBUG] ERROR: failed to receive NCCL handle"
                  << std::endl;
        close(bootstrap_fd);
        return 1;
      }
      total_received += static_cast<size_t>(r);
    }

    void* send_comm = nullptr;
    alignas(16) unsigned char send_dev_handle_storage[512] = {0};
    void* send_dev_handle = send_dev_handle_storage;
    if (tcpx_connect_v5(dev_id, &handle, &send_comm, &send_dev_handle) != 0 ||
        !send_comm) {
      std::cout << "[DEBUG] ERROR: tcpx_connect_v5 connection failed"
                << std::endl;
      close(bootstrap_fd);
      return 1;
    }
    std::cout << "[DEBUG] TCPX connection established; send_comm=" << send_comm
              << ", send_dev_handle=" << send_dev_handle << std::endl;

    CUdevice cuDev = 0;
    CUcontext cuCtx = nullptr;
    CUdeviceptr d_base = 0;
    CUdeviceptr d_aligned = 0;
    void* send_mhandle = nullptr;
    void* send_request = nullptr;
    int done = 0;  // Changed from bool to int for tcpx_test
    int sent_size = 0;

    auto cleanup_client = [&]() {
      if (send_mhandle) {
        tcpx_dereg_mr(send_comm, send_mhandle);
        send_mhandle = nullptr;
      }
      if (d_base) {
        cuMemFree(d_base);
        d_base = 0;
      }
      if (send_comm) {
        tcpx_close_send(send_comm);
        send_comm = nullptr;
      }
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

    void* send_mhandle_local = nullptr;
    if (tcpx_reg_mr(send_comm, reinterpret_cast<void*>(d_aligned),
                    kRegisteredBytes, NCCL_PTR_CUDA,
                    &send_mhandle_local) != 0) {
      return fail_client(
          "[DEBUG] ERROR: tcpx_reg_mr failed for client send buffer");
    }
    send_mhandle = send_mhandle_local;
    std::cout << "[DEBUG] Registered client send buffer ptr="
              << reinterpret_cast<void*>(d_aligned)
              << ", bytes=" << kRegisteredBytes << std::endl;

    if (tcpx_isend(send_comm, reinterpret_cast<void*>(d_aligned),
                   static_cast<int>(payload_bytes), kTransferTag, send_mhandle,
                   &send_request) != 0) {
      return fail_client("[DEBUG] ERROR: tcpx_isend failed");
    }

    std::cout << "[DEBUG] tcpx_isend returned rc=0, request=" << send_request
              << std::endl;
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
