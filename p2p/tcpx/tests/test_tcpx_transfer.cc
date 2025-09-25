/**
 * @file test_tcpx_transfer_clean.cc
 * @brief TCPX GPU-to-GPU end-to-end transfer validation
 *
 * This test uses nccl-plugin-gpudirecttcpx APIs to transfer data between CUDA device memory
 * on two nodes. It builds upon test_connection.cc (handshake only) by adding CUDA buffer
 * registration and data validation.
 *
 * Server steps:
 *   1. Listen for TCPX connections on device 0, publish NCCL handle via bootstrap TCP socket.
 *   2. Accept TCPX connection, register 4KB CUDA buffer, submit async receive request.
 *   3. Poll for completion, copy data back to host memory and validate payload content.
 *
 * Client steps:
 *   1. Get NCCL handle via bootstrap TCP socket.
 *   2. Connect to server via TCPX, register 4KB CUDA buffer, write test payload and send.
 *   3. Wait for completion and cleanup resources.
 */

#include "../tcpx_interface.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr size_t kHandleBytes = 128;
struct ncclNetHandle_v7 {
  char data[kHandleBytes];
};

constexpr int kBootstrapPort = 12345;
constexpr size_t kRegisteredBytes = 4096;      // 4 KB aligned allocation.

// PROBLEM BACKGROUND MARKER:
// Previously using sizeof(kTestMessage) directly as payload size would include trailing '\0',
// causing client to think 25B while logs/transport layer only handled 24B visible chars,
// creating 25 vs 24B inconsistency, further triggering small packet zero-copy (MSG_ZEROCOPY)
// kernel errqueue flakiness, server only seeing 16B control message, no payload received.
// Fix: Use strlen semantics (sizeof-1), force <4KB to copy path at runtime to avoid small packet zero-copy.

constexpr char kTestMessage[] = "Hello from TCPX client!";
// Default payload length (visible chars only). Can be overridden by env
// UCCL_TCPX_PAYLOAD_BYTES up to kRegisteredBytes.
constexpr size_t kDefaultPayloadBytes = sizeof(kTestMessage) - 1;
constexpr int kTransferTag = 42;  // Payload tag
constexpr int kAcceptMaxRetries = 120;         // ~12 s (100 ms per retry).

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

int connect_to_bootstrap_server(const char* server_ip) {
  int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (sock_fd < 0) return -1;

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(kBootstrapPort);
  inet_aton(server_ip, &addr.sin_addr);

  for (int retry = 0; retry < 10; ++retry) {
    if (connect(sock_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
      return sock_fd;
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  close(sock_fd);
  return -1;
}

void dump_hex(const void* data, size_t bytes) {
  const unsigned char* p = static_cast<const unsigned char*>(data);
  size_t limit = std::min<size_t>(bytes, 32);
  for (size_t i = 0; i < limit; ++i) {
    if (i && i % 16 == 0) std::cout << "\n";
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(p[i]) << ' ';
  }
  std::cout << std::dec << std::endl;
}

bool cuda_check(CUresult res, const char* what) {
  if (res == CUDA_SUCCESS) return true;

  const char* name = nullptr;
  const char* desc = nullptr;
  cuGetErrorName(res, &name);
  cuGetErrorString(res, &desc);

  std::cout << "[DEBUG] CUDA error at " << what << ": "
            << (name ? name : "?") << " - " << (desc ? desc : "")
            << std::endl;
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  // Debug/stability knobs: avoid zero-copy for tiny payloads to reduce errqueue flakiness
  setenv("UCCL_TCPX_DEBUG", "1", 0);
  setenv("NCCL_MIN_ZCOPY_SIZE", "4096", 0);  // Force <4KB to copy path
  setenv("NCCL_GPUDIRECTTCPX_MIN_ZCOPY_SIZE", "4096", 0);  // Plugin-specific backup
  // Optional: make receive path more deterministic during debug
  setenv("NCCL_GPUDIRECTTCPX_RECV_SYNC", "1", 0);

  std::cout << "[DEBUG] === TCPX GPU-to-GPU transfer test ===" << std::endl;
  if (argc < 2) {
    std::cout << "[DEBUG] Usage: " << argv[0] << " <server|client> [remote_ip]" << std::endl;
    return 1;
  }
  int device_count = tcpx_get_device_count();
  if (device_count <= 0) {
    std::cout << "[DEBUG] ERROR: no TCPX devices detected" << std::endl;
    return 1;
  }
  int dev_id = 0;
  bool is_server = std::strcmp(argv[1], "server") == 0;

  // Resolve payload size (can be overridden via env UCCL_TCPX_PAYLOAD_BYTES)
  size_t payload_bytes = kDefaultPayloadBytes;
  if (const char* p = std::getenv("UCCL_TCPX_PAYLOAD_BYTES")) {
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
      std::cout << "[DEBUG] ERROR: bootstrap server creation failed" << std::endl;
      tcpx_close_listen(listen_comm);
      return 1;
    }
    std::cout << "[DEBUG] Bootstrap connection established, sending handle" << std::endl;

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
    // Keep bootstrap_fd open to optionally send a 1-byte ACK after payload verification

    void* recv_comm = nullptr;
    alignas(16) unsigned char recv_dev_handle_storage[512] = {0};
    void* recv_dev_handle = recv_dev_handle_storage;

    int attempts = 0;
    while (attempts < kAcceptMaxRetries) {
      int rc = tcpx_accept_v5(listen_comm, &recv_comm, &recv_dev_handle);
      if (rc != 0) {
        std::cout << "[DEBUG] ERROR: tcpx_accept_v5 returned error rc=" << rc << std::endl;
        tcpx_close_listen(listen_comm);
        return 1;
      }
      if (recv_comm) break;  // Successfully established
      ++attempts;
      if (attempts % 10 == 0) {
        std::cout << "[DEBUG] INFO: tcpx_accept_v5 rc=0 but recv_comm still null (attempt "
                  << attempts << "), continuing to wait for peer handshake..." << std::endl;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!recv_comm) {
      std::cout << "[DEBUG] ERROR: failed to get valid recv_comm after retries" << std::endl;
      tcpx_close_listen(listen_comm);
      return 1;
    }
    std::cout << "[DEBUG] Connection accepted; recv_comm=" << recv_comm << std::endl;

    CUdevice cuDev = 0;
    CUcontext cuCtx = nullptr;
    CUdeviceptr d_base = 0;
    CUdeviceptr d_aligned = 0;

    if (!cuda_check(cuInit(0), "cuInit") ||
        !cuda_check(cuDeviceGet(&cuDev, dev_id), "cuDeviceGet") ||
        !cuda_check(cuDevicePrimaryCtxRetain(&cuCtx, cuDev), "cuDevicePrimaryCtxRetain") ||
        !cuda_check(cuCtxSetCurrent(cuCtx), "cuCtxSetCurrent") ||
        !cuda_check(cuMemAlloc(&d_base, kRegisteredBytes + 4096), "cuMemAlloc")) {
      if (d_base) cuMemFree(d_base);
      if (cuCtx) cuDevicePrimaryCtxRelease(cuDev);
      tcpx_close_recv(recv_comm);
      tcpx_close_listen(listen_comm);
      return 1;
    }

    uintptr_t addr = static_cast<uintptr_t>(d_base);
    addr = (addr + 4095) & ~static_cast<uintptr_t>(4095);
    d_aligned = static_cast<CUdeviceptr>(addr);
    std::cout << "[DEBUG] Registered GPU buffer at 0x" << std::hex << d_aligned
              << std::dec << std::endl;

    // Allow host receive debug path via env to isolate GPUDirect issues
    const char* host_recv_dbg = std::getenv("UCCL_TCPX_HOST_RECV_DEBUG");
    void* recv_mhandle = nullptr;
    void* recv_buf = nullptr;
    int recv_ptr_type = NCCL_PTR_CUDA;
    if (host_recv_dbg && host_recv_dbg[0] == '1') {
      // Allocate page-aligned host buffer
      size_t page = 4096;
      void* host_aligned = nullptr;
      if (posix_memalign(&host_aligned, page, kRegisteredBytes) != 0 || !host_aligned) {
        std::cout << "[DEBUG] ERROR: posix_memalign failed for host recv" << std::endl;
        cuMemFree(d_base);
        tcpx_close_recv(recv_comm);
        tcpx_close_listen(listen_comm);
        cuDevicePrimaryCtxRelease(cuDev);
        return 1;
      }
      std::memset(host_aligned, 0, kRegisteredBytes);
      recv_buf = host_aligned;
      recv_ptr_type = NCCL_PTR_HOST;
      std::cout << "[DEBUG] Host-recv debug enabled; host buffer=" << recv_buf << std::endl;
    } else {
      recv_buf = reinterpret_cast<void*>(d_aligned);
      recv_ptr_type = NCCL_PTR_CUDA;
    }

    if (tcpx_reg_mr(recv_comm, recv_buf, kRegisteredBytes, recv_ptr_type, &recv_mhandle) != 0) {
      std::cout << "[DEBUG] ERROR: tcpx_reg_mr (recv) failed" << std::endl;
      cuMemFree(d_base);
      tcpx_close_recv(recv_comm);
      tcpx_close_listen(listen_comm);
      cuDevicePrimaryCtxRelease(cuDev);
      return 1;
    }

    void* recv_data[1] = {recv_buf};
    int recv_sizes[1] = {static_cast<int>(payload_bytes)};
    int recv_tags[1] = {kTransferTag};
    void* recv_mhandles[1] = {recv_mhandle};
    void* recv_request = nullptr;

    if (tcpx_irecv(recv_comm, 1, recv_data, recv_sizes, recv_tags,
                   recv_mhandles, &recv_request) != 0) {
      std::cout << "[DEBUG] ERROR: tcpx_irecv failed" << std::endl;
      tcpx_dereg_mr(recv_comm, recv_mhandle);
      cuMemFree(d_base);
      tcpx_close_recv(recv_comm);
      tcpx_close_listen(listen_comm);
      cuDevicePrimaryCtxRelease(cuDev);
      return 1;
    }

    // PROBLEM CODE LOCATION (Server side):
    // Historical issue: Server only received 16B control message, then peer closed, payload never arrived (timeout).
    // Root cause: Client small packet zero-copy caused errqueue flakiness + client closed too early without waiting.
    // Fix approach: Sender uses strlen length; <4KB forced to copy path; this side keeps short polling tcpx_test.

    std::cout << "[DEBUG] Waiting for payload..." << std::endl;
    int done = 0;
    int received_size = 0;
    for (int i = 0; i < 200000 && !done; ++i) {
      int rc_test = tcpx_test(recv_request, &done, &received_size);
      if (rc_test != 0) {
        std::cout << "[DEBUG] ERROR: tcpx_test returned " << rc_test << std::endl;
        break;
      }
      if (!done) std::this_thread::sleep_for(std::chrono::microseconds(10));
    }

    std::vector<unsigned char> host(payload_bytes, 0);
    bool success = false;
    // Ensure NIC writes are visible before copying back (for CUDA path)
    if (done && recv_ptr_type == NCCL_PTR_CUDA) cuCtxSynchronize();
    bool copy_ok = false;
    if (done && recv_ptr_type == NCCL_PTR_CUDA) {
      copy_ok = cuda_check(cuMemcpyDtoH(host.data(), d_aligned, payload_bytes), "cuMemcpyDtoH");
    } else if (done && recv_ptr_type == NCCL_PTR_HOST) {
      std::memcpy(host.data(), recv_buf, payload_bytes);
      copy_ok = true;
    }
    if (done && copy_ok) {
      std::cout << "[DEBUG] Receive completed, bytes=" << received_size << std::endl;
      dump_hex(host.data(), std::min<size_t>(payload_bytes, 32));
      size_t prefix = std::min(payload_bytes, sizeof(kTestMessage) - 1);
      bool prefix_ok = std::memcmp(host.data(), kTestMessage, prefix) == 0;
      bool tail_ok = true;
      if (payload_bytes > prefix) tail_ok = host[payload_bytes - 1] == 0xAB;
      success = prefix_ok && tail_ok;
      if (success) {
        std::cout << "[DEBUG] SUCCESS: payload matches expected string" << std::endl;
        // Notify client it is safe to close by sending a 1-byte ACK on bootstrap TCP
        if (bootstrap_fd >= 0) {
          char ack = 1;
          (void)send(bootstrap_fd, &ack, 1, 0);
        }
      } else {
        std::cout << "[DEBUG] ERROR: payload mismatch" << std::endl;
      }
    } else {
      std::cout << "[DEBUG] ERROR: receive timed out" << std::endl;
    }

    if (recv_request && done) {
      tcpx_irecv_consumed(recv_comm, 1, recv_request);
    }

    if (recv_mhandle) tcpx_dereg_mr(recv_comm, recv_mhandle);
    cuMemFree(d_base);
    tcpx_close_recv(recv_comm);
    tcpx_close_listen(listen_comm);
    if (bootstrap_fd >= 0) close(bootstrap_fd);
    cuDevicePrimaryCtxRelease(cuDev);
    return success ? 0 : 1;

  } else {
    if (argc < 3) {
      std::cout << "[DEBUG] ERROR: client mode requires <remote_ip>" << std::endl;
      return 1;
    }
    const char* remote_ip = argv[2];
    std::cout << "[DEBUG] Running in CLIENT mode, connecting to " << remote_ip << std::endl;

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
        std::cout << "[DEBUG] ERROR: failed to receive NCCL handle" << std::endl;
        close(bootstrap_fd);
        return 1;
      }
      total_received += static_cast<size_t>(r);
    }
    // Keep bootstrap_fd open to wait for server's 1-byte ACK after send

    void* send_comm = nullptr;
    // Pre-allocate device handle storage (some implementations require caller to provide buffer)
    alignas(16) unsigned char send_dev_handle_storage[512] = {0};
    void* send_dev_handle = send_dev_handle_storage;
    if (tcpx_connect_v5(dev_id, &handle, &send_comm, &send_dev_handle) != 0 ||
        !send_comm) {
      std::cout << "[DEBUG] ERROR: tcpx_connect_v5 connection failed" << std::endl;
      return 1;
    }
    std::cout << "[DEBUG] TCPX connection established; send_comm=" << send_comm
              << ", send_dev_handle=" << send_dev_handle << std::endl;

    CUdevice cuDev = 0;
    CUcontext cuCtx = nullptr;
    CUdeviceptr d_base = 0;
    CUdeviceptr d_aligned = 0;

    if (!cuda_check(cuInit(0), "cuInit") ||
        !cuda_check(cuDeviceGet(&cuDev, dev_id), "cuDeviceGet") ||
        !cuda_check(cuDevicePrimaryCtxRetain(&cuCtx, cuDev), "cuDevicePrimaryCtxRetain") ||
        !cuda_check(cuCtxSetCurrent(cuCtx), "cuCtxSetCurrent") ||
        !cuda_check(cuMemAlloc(&d_base, kRegisteredBytes + 4096), "cuMemAlloc")) {
      if (d_base) cuMemFree(d_base);
      if (cuCtx) cuDevicePrimaryCtxRelease(cuDev);
      tcpx_close_send(send_comm);
      return 1;
    }

    // Prepare payload in host memory according to payload_bytes

    uintptr_t addr = static_cast<uintptr_t>(d_base);
    addr = (addr + 4095) & ~static_cast<uintptr_t>(4095);
    d_aligned = static_cast<CUdeviceptr>(addr);
    std::vector<unsigned char> host_payload(payload_bytes, 0);
    size_t prefix = std::min(payload_bytes, sizeof(kTestMessage) - 1);
    if (prefix) std::memcpy(host_payload.data(), kTestMessage, prefix);
    if (payload_bytes > prefix) host_payload[payload_bytes - 1] = 0xAB;  // sentinel
    cuda_check(cuMemcpyHtoD(d_aligned, host_payload.data(), payload_bytes),
               "cuMemcpyHtoD");
    // Make sure device buffer contents are visible before zero-copy send kicks in
    cuCtxSynchronize();
    // Debug: verify device buffer content before send
    {
      size_t dump = std::min<size_t>(payload_bytes, 32);
      std::vector<unsigned char> verify(dump, 0);
      if (cuda_check(cuMemcpyDtoH(verify.data(), d_aligned, dump), "pre-send cuMemcpyDtoH")) {
        std::cout << "[DEBUG] Client GPU buffer prefix (" << dump << "):" << std::endl;
        dump_hex(verify.data(), dump);
      }
    }

    void* send_mhandle = nullptr;
    if (tcpx_reg_mr(send_comm, reinterpret_cast<void*>(d_aligned),
                    kRegisteredBytes, NCCL_PTR_CUDA, &send_mhandle) != 0) {
      std::cout << "[DEBUG] ERROR: tcpx_reg_mr (send) failed" << std::endl;
      cuMemFree(d_base);
      tcpx_close_send(send_comm);
      cuDevicePrimaryCtxRelease(cuDev);
      return 1;
    }

    void* send_request = nullptr;
    if (tcpx_isend(send_comm, reinterpret_cast<void*>(d_aligned),
                   static_cast<int>(payload_bytes), kTransferTag, send_mhandle,
                   &send_request) != 0) {
      std::cout << "[DEBUG] ERROR: tcpx_isend failed" << std::endl;
      tcpx_dereg_mr(send_comm, send_mhandle);
      cuMemFree(d_base);
      tcpx_close_send(send_comm);
      cuDevicePrimaryCtxRelease(cuDev);
      return 1;
    }

    int done = 0;
    int sent_size = 0;
    for (int i = 0; i < 200000 && !done; ++i) {
      int rc_test = tcpx_test(send_request, &done, &sent_size);
      if (rc_test != 0) {
        std::cout << "[DEBUG] ERROR: tcpx_test returned " << rc_test << std::endl;
        break;
      }
      if (!done) std::this_thread::sleep_for(std::chrono::microseconds(10));
    }

    if (done) {
      std::cout << "[DEBUG] Send completed, bytes=" << sent_size << std::endl;
    } else {
      std::cout << "[DEBUG] WARNING: send did not complete before timeout" << std::endl;
    }
    // Wait for 1-byte ACK from server on the bootstrap TCP socket to avoid early close
    if (bootstrap_fd >= 0) {
      timeval tv{}; tv.tv_sec = 2; tv.tv_usec = 0;
      setsockopt(bootstrap_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
      char ack = 0;
      ssize_t r = recv(bootstrap_fd, &ack, 1, 0);
      if (r == 1 && ack == 1) {
        std::cout << "[DEBUG] Server ACK received" << std::endl;
      } else {
        std::cout << "[DEBUG] WARNING: did not receive server ACK" << std::endl;
      }
    }

    tcpx_dereg_mr(send_comm, send_mhandle);
    cuMemFree(d_base);
    tcpx_close_send(send_comm);
    if (bootstrap_fd >= 0) close(bootstrap_fd);
    cuDevicePrimaryCtxRelease(cuDev);
    return done ? 0 : 1;
  }
}
