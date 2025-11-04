#include "uccl_engine.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

constexpr int kHandshakePortOffset = 1;
constexpr int kPointerPortOffset = 2;
constexpr int kMaxConnectRetries = 50;
constexpr int kConnectSleepMs = 100;
constexpr size_t kMaxChunkBytes = 4 * 1024 * 1024;

struct Options {
  std::string mode;
  std::string ip{"127.0.0.1"};
  int port = 28900;
  int gpu = 0;
  size_t size = 1 << 20;  // 1 MiB
  int iters = 1;
};

bool parse_args(int argc, char** argv, Options& out) {
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    auto pos = arg.find('=');
    std::string key = arg.substr(0, pos);
    std::string value = pos == std::string::npos ? "" : arg.substr(pos + 1);
    if (key == "--mode") {
      out.mode = value;
    } else if (key == "--ip") {
      out.ip = value;
    } else if (key == "--port") {
      out.port = std::stoi(value);
    } else if (key == "--gpu") {
      out.gpu = std::stoi(value);
    } else if (key == "--size") {
      out.size = static_cast<size_t>(std::stoull(value));
    } else if (key == "--iters") {
      int parsed = std::stoi(value);
      out.iters = std::max(parsed, 1);
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return false;
    }
  }
  if (out.mode != "server" && out.mode != "client") {
    std::cerr << "Missing or invalid --mode=server|client" << std::endl;
    return false;
  }
  return true;
}

bool checkCuda(cudaError_t err, char const* where) {
  if (err != cudaSuccess) {
    std::cerr << "[CUDA] " << where << " failed: " << cudaGetErrorString(err)
              << std::endl;
    return false;
  }
  return true;
}

std::vector<uint8_t> make_pattern(size_t n) {
  std::vector<uint8_t> v(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = static_cast<uint8_t>(i & 0xffu);
  }
  return v;
}

bool validate_pattern(uint8_t const* data, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    uint8_t expect = static_cast<uint8_t>(i & 0xffu);
    if (data[i] != expect) {
      std::cerr << "[UCCL Smoke] mismatch @" << i
                << " got=" << static_cast<int>(data[i])
                << " expect=" << static_cast<int>(expect) << std::endl;
      return false;
    }
  }
  return true;
}

size_t get_env_size(char const* name, size_t def_value) {
  char const* v = std::getenv(name);
  if (!v || !*v) return def_value;
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(v, &end, 10);
  if (end == v) return def_value;
  return static_cast<size_t>(parsed);
}

size_t resolve_chunk_bytes(size_t total_size) {
  size_t chunk = get_env_size("UCCL_TCPX_CHUNK_BYTES", 0);
  if (chunk == 0) chunk = get_env_size("NCCL_P2P_NET_CHUNKSIZE", 0);
  if (chunk == 0) chunk = get_env_size("NCCL_DYNAMIC_CHUNK_SIZE", 0);
  if (chunk == 0) chunk = 512 * 1024;
  chunk = std::max<size_t>(1, std::min(chunk, total_size));
  chunk = std::min(chunk, kMaxChunkBytes);
  return chunk;
}

struct SlotBuf {
  alignas(16) unsigned char bytes[64];
};

bool send_all(int fd, void const* buf, size_t len) {
  auto* ptr = static_cast<char const*>(buf);
  size_t sent = 0;
  while (sent < len) {
    ssize_t n = ::send(fd, ptr + sent, len - sent, 0);
    if (n <= 0) return false;
    sent += static_cast<size_t>(n);
  }
  return true;
}

bool recv_all(int fd, void* buf, size_t len) {
  auto* ptr = static_cast<char*>(buf);
  size_t recvd = 0;
  while (recvd < len) {
    ssize_t n = ::recv(fd, ptr + recvd, len - recvd, 0);
    if (n <= 0) return false;
    recvd += static_cast<size_t>(n);
  }
  return true;
}

struct HandshakePayload {
  uint64_t payload_bytes;
  uint32_t iterations;
  uint32_t reserved;
};

bool handshake_server(Options& opts) {
  int listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd < 0) {
    std::cerr << "[UCCL Smoke] handshake server socket() failed: "
              << std::strerror(errno) << std::endl;
    return false;
  }

  int reuse = 1;
  setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port =
      htons(static_cast<uint16_t>(opts.port + kHandshakePortOffset));
  if (bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    std::cerr << "[UCCL Smoke] handshake bind() failed: "
              << std::strerror(errno) << std::endl;
    ::close(listen_fd);
    return false;
  }
  if (listen(listen_fd, 1) < 0) {
    std::cerr << "[UCCL Smoke] handshake listen() failed: "
              << std::strerror(errno) << std::endl;
    ::close(listen_fd);
    return false;
  }

  int conn_fd = ::accept(listen_fd, nullptr, nullptr);
  if (conn_fd < 0) {
    std::cerr << "[UCCL Smoke] handshake accept() failed: "
              << std::strerror(errno) << std::endl;
    ::close(listen_fd);
    return false;
  }

  HandshakePayload payload{};
  if (!recv_all(conn_fd, &payload, sizeof(payload))) {
    std::cerr << "[UCCL Smoke] handshake recv() failed: "
              << std::strerror(errno) << std::endl;
    ::close(conn_fd);
    ::close(listen_fd);
    return false;
  }

  if (payload.payload_bytes > 0)
    opts.size = static_cast<size_t>(payload.payload_bytes);
  if (payload.iterations > 0) opts.iters = static_cast<int>(payload.iterations);

  uint32_t ack = 1;
  send_all(conn_fd, &ack, sizeof(ack));

  ::close(conn_fd);
  ::close(listen_fd);
  return true;
}

bool handshake_client(Options const& opts) {
  int sock_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (sock_fd < 0) {
    std::cerr << "[UCCL Smoke] handshake client socket() failed: "
              << std::strerror(errno) << std::endl;
    return false;
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port =
      htons(static_cast<uint16_t>(opts.port + kHandshakePortOffset));
  if (inet_pton(AF_INET, opts.ip.c_str(), &addr.sin_addr) != 1) {
    std::cerr << "[UCCL Smoke] handshake invalid IP " << opts.ip << std::endl;
    ::close(sock_fd);
    return false;
  }

  bool connected = false;
  for (int attempt = 0; attempt < kMaxConnectRetries; ++attempt) {
    if (::connect(sock_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) ==
        0) {
      connected = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kConnectSleepMs));
  }
  if (!connected) {
    std::cerr << "[UCCL Smoke] handshake connect() failed: "
              << std::strerror(errno) << std::endl;
    ::close(sock_fd);
    return false;
  }

  HandshakePayload payload{};
  payload.payload_bytes = static_cast<uint64_t>(opts.size);
  payload.iterations = static_cast<uint32_t>(opts.iters);
  if (!send_all(sock_fd, &payload, sizeof(payload))) {
    std::cerr << "[UCCL Smoke] handshake send() failed: "
              << std::strerror(errno) << std::endl;
    ::close(sock_fd);
    return false;
  }

  uint32_t ack = 0;
  if (!recv_all(sock_fd, &ack, sizeof(ack)) || ack != 1) {
    std::cerr << "[UCCL Smoke] handshake ack failed" << std::endl;
    ::close(sock_fd);
    return false;
  }

  ::close(sock_fd);
  return true;
}

struct PointerPayload {
  uint64_t device_ptr;
  uint64_t bytes;
};

int pointer_server_stage(Options const& opts, PointerPayload const& payload) {
  int listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd < 0) {
    std::cerr << "[UCCL Smoke] pointer server socket() failed: "
              << std::strerror(errno) << std::endl;
    return -1;
  }
  int reuse = 1;
  setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(static_cast<uint16_t>(opts.port + kPointerPortOffset));
  if (bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    std::cerr << "[UCCL Smoke] pointer bind() failed: " << std::strerror(errno)
              << std::endl;
    ::close(listen_fd);
    return -1;
  }
  if (listen(listen_fd, 1) < 0) {
    std::cerr << "[UCCL Smoke] pointer listen() failed: "
              << std::strerror(errno) << std::endl;
    ::close(listen_fd);
    return -1;
  }

  int conn_fd = ::accept(listen_fd, nullptr, nullptr);
  if (conn_fd < 0) {
    std::cerr << "[UCCL Smoke] pointer accept() failed: "
              << std::strerror(errno) << std::endl;
    ::close(listen_fd);
    return -1;
  }
  ::close(listen_fd);

  if (!send_all(conn_fd, &payload, sizeof(payload))) {
    std::cerr << "[UCCL Smoke] pointer send() failed: " << std::strerror(errno)
              << std::endl;
    ::close(conn_fd);
    return -1;
  }
  return conn_fd;
}

bool pointer_server_wait_ack(int conn_fd) {
  uint32_t ack = 0;
  bool ok = recv_all(conn_fd, &ack, sizeof(ack));
  ::close(conn_fd);
  return ok && ack == 1;
}

int pointer_client_stage(Options const& opts, PointerPayload& out) {
  int sock_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (sock_fd < 0) {
    std::cerr << "[UCCL Smoke] pointer client socket() failed: "
              << std::strerror(errno) << std::endl;
    return -1;
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(opts.port + kPointerPortOffset));
  if (inet_pton(AF_INET, opts.ip.c_str(), &addr.sin_addr) != 1) {
    std::cerr << "[UCCL Smoke] pointer invalid IP " << opts.ip << std::endl;
    ::close(sock_fd);
    return -1;
  }

  bool connected = false;
  for (int attempt = 0; attempt < kMaxConnectRetries; ++attempt) {
    if (::connect(sock_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) ==
        0) {
      connected = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kConnectSleepMs));
  }
  if (!connected) {
    std::cerr << "[UCCL Smoke] pointer connect() failed: "
              << std::strerror(errno) << std::endl;
    ::close(sock_fd);
    return -1;
  }

  if (!recv_all(sock_fd, &out, sizeof(out))) {
    std::cerr << "[UCCL Smoke] pointer recv() failed: " << std::strerror(errno)
              << std::endl;
    ::close(sock_fd);
    return -1;
  }

  return sock_fd;
}

bool pointer_client_send_ack(int sock_fd) {
  uint32_t ack = 1;
  bool ok = send_all(sock_fd, &ack, sizeof(ack));
  ::close(sock_fd);
  return ok;
}

struct DeviceDeleter {
  void operator()(uint8_t* ptr) const {
    if (ptr) cudaFree(ptr);
  }
};

using DeviceBuffer = std::unique_ptr<uint8_t, DeviceDeleter>;

void apply_env_defaults(int port) {
  std::string port_str = std::to_string(port);
  setenv("UCCL_TCPX_OOB_PORT", port_str.c_str(), 1);
  setenv("NCCL_GPUDIRECTTCPX_CTRL_DEV", "eth0", 0);
  setenv("NCCL_NSOCKS_PERTHREAD", "2", 0);
  setenv("NCCL_SOCKET_NTHREADS", "1", 0);
  setenv("NCCL_MIN_ZCOPY_SIZE", "4096", 0);
  setenv("NCCL_GPUDIRECTTCPX_MIN_ZCOPY_SIZE", "4096", 0);
  setenv("NCCL_GPUDIRECTTCPX_RECV_SYNC", "1", 0);
  setenv("NCCL_DYNAMIC_CHUNK_SIZE", "524288", 0);
  setenv("NCCL_P2P_NET_CHUNKSIZE", "524288", 0);
  setenv("NCCL_P2P_PCI_CHUNKSIZE", "524288", 0);
  setenv("NCCL_P2P_NVL_CHUNKSIZE", "1048576", 0);
  setenv("NCCL_BUFFSIZE", "8388608", 0);
  setenv("NCCL_SOCKET_IFNAME", "eth0", 0);
  setenv("NCCL_CROSS_NIC", "0", 0);
  setenv("NCCL_NET_GDR_LEVEL", "PIX", 0);
  setenv("NCCL_P2P_PXN_LEVEL", "0", 0);
  setenv("NCCL_ALGO", "Ring", 0);
  setenv("NCCL_PROTO", "Simple", 0);
  setenv("NCCL_MAX_NCHANNELS", "8", 0);
  setenv("NCCL_MIN_NCHANNELS", "8", 0);
  setenv("NCCL_DEBUG", "INFO", 0);
  setenv("NCCL_DEBUG_SUBSYS", "ENV", 0);
  setenv("UCCL_TCPX_DEBUG", "1", 0);
}

int run_server(Options opts) {
  if (!checkCuda(cudaSetDevice(opts.gpu), "cudaSetDevice(server)")) return 1;
  apply_env_defaults(opts.port);
#ifdef USE_TCPX
  {
    std::string gpu_env = std::to_string(opts.gpu);
    setenv("UCCL_TCPX_LOCAL_DEVICE", gpu_env.c_str(), 1);
  }
#endif

  uccl_engine_t* eng =
      uccl_engine_create(/*num_cpus=*/1, /*in_python=*/false);
  if (!eng) {
    std::cerr << "[UCCL Smoke] server: uccl_engine_create failed" << std::endl;
    return 1;
  }

  std::cout << "[UCCL Smoke] Waiting for client on port " << opts.port
            << "...\n";
  char peer_ip[128] = {0};
  int peer_gpu = -1;
  uccl_conn_t* conn =
      uccl_engine_accept(eng, peer_ip, sizeof(peer_ip), &peer_gpu);
  if (!conn) {
    std::cerr << "[UCCL Smoke] server: accept failed" << std::endl;
    uccl_engine_destroy(eng);
    return 1;
  }
  std::cout << "[UCCL Smoke] Accepted connection from " << peer_ip
            << " (peer GPU " << peer_gpu << ")\n";

  if (uccl_engine_start_listener(conn) != 0) {
    std::cerr << "[UCCL Smoke] server: start listener failed" << std::endl;
    uccl_engine_conn_destroy(conn);
    uccl_engine_destroy(eng);
    return 1;
  }

  uint8_t* raw_dev = nullptr;
  if (!checkCuda(cudaMalloc(&raw_dev, opts.size), "cudaMalloc(server)"))
    return 1;
  DeviceBuffer dev_buf(raw_dev);

  auto host_pattern = make_pattern(opts.size);
  if (!checkCuda(cudaMemcpy(raw_dev, host_pattern.data(), opts.size,
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy H2D (server)")) {
    return 1;
  }

  uccl_mr_t* mr =
      uccl_engine_reg(eng, reinterpret_cast<uintptr_t>(raw_dev), opts.size);
  if (!mr) {
    std::cerr << "[UCCL Smoke] server: uccl_engine_reg failed" << std::endl;
    return 1;
  }

  PointerPayload payload{reinterpret_cast<uint64_t>(raw_dev),
                         static_cast<uint64_t>(opts.size)};
  int ptr_fd = pointer_server_stage(opts, payload);
  if (ptr_fd < 0) {
    uccl_engine_mr_destroy(mr);
    return 1;
  }
  std::cout << "[UCCL Smoke] Advertised device buffer ptr=0x" << std::hex
            << payload.device_ptr << std::dec << " bytes=" << payload.bytes
            << std::endl;

  if (!pointer_server_wait_ack(ptr_fd)) {
    std::cerr << "[UCCL Smoke] server: pointer ack failed" << std::endl;
    uccl_engine_mr_destroy(mr);
    return 1;
  }

  std::cout << "[UCCL Smoke] Server received completion ack" << std::endl;

  std::vector<uint8_t> verify(opts.size);
  if (!checkCuda(
          cudaMemcpy(verify.data(), raw_dev, opts.size, cudaMemcpyDeviceToHost),
          "cudaMemcpy D2H (server)")) {
    return 1;
  }
  if (!validate_pattern(verify.data(), opts.size)) {
    std::cerr << "[UCCL Smoke] server: validation mismatch" << std::endl;
    return 1;
  }

  uccl_engine_mr_destroy(mr);
  uccl_engine_conn_destroy(conn);
  uccl_engine_destroy(eng);
  std::cout << "[UCCL Smoke] Server PASS" << std::endl;
  return 0;
}

int run_client(Options opts) {
  if (!checkCuda(cudaSetDevice(opts.gpu), "cudaSetDevice(client)")) return 1;
  apply_env_defaults(opts.port);
#ifdef USE_TCPX
  {
    std::string gpu_env = std::to_string(opts.gpu);
    setenv("UCCL_TCPX_LOCAL_DEVICE", gpu_env.c_str(), 1);
  }
#endif

  uccl_engine_t* eng =
      uccl_engine_create(/*num_cpus=*/1, /*in_python=*/false);
  if (!eng) {
    std::cerr << "[UCCL Smoke] client: uccl_engine_create failed" << std::endl;
    return 1;
  }

  std::cout << "[UCCL Smoke] Connecting to " << opts.ip << ":" << opts.port
            << "...\n";
  uccl_conn_t* conn = uccl_engine_connect(eng, opts.ip.c_str(),
                                          /*remote_gpu_idx=*/0, opts.port);
  if (!conn) {
    std::cerr << "[UCCL Smoke] client: connect failed" << std::endl;
    uccl_engine_destroy(eng);
    return 1;
  }

  if (uccl_engine_start_listener(conn) != 0) {
    std::cerr << "[UCCL Smoke] client: start listener failed" << std::endl;
    uccl_engine_conn_destroy(conn);
    uccl_engine_destroy(eng);
    return 1;
  }

  PointerPayload payload{};
  int ptr_fd = pointer_client_stage(opts, payload);
  if (ptr_fd < 0) {
    return 1;
  }
  uint64_t remote_ptr = payload.device_ptr;
  size_t remote_bytes = static_cast<size_t>(payload.bytes);
  if (remote_bytes < opts.size) {
    std::cerr << "[UCCL Smoke] client: remote advertised only " << remote_bytes
              << " bytes" << std::endl;
    return 1;
  }
  std::cout << "[UCCL Smoke] Received remote ptr=0x" << std::hex << remote_ptr
            << std::dec << " bytes=" << remote_bytes << std::endl;

  uint8_t* raw_dev = nullptr;
  if (!checkCuda(cudaMalloc(&raw_dev, opts.size), "cudaMalloc(client)"))
    return 1;
  DeviceBuffer dev_buf(raw_dev);

  uccl_mr_t* mr =
      uccl_engine_reg(eng, reinterpret_cast<uintptr_t>(raw_dev), opts.size);
  if (!mr) {
    std::cerr << "[UCCL Smoke] client: uccl_engine_reg failed" << std::endl;
    return 1;
  }

  size_t chunk_bytes = resolve_chunk_bytes(opts.size);
  size_t chunks_per_iter = (opts.size + chunk_bytes - 1) / chunk_bytes;
  std::vector<size_t> chunk_sizes(chunks_per_iter);
  std::vector<md_t> md_vec(chunks_per_iter);
  {
    size_t offset = 0;
    for (size_t chunk = 0; chunk < chunks_per_iter; ++chunk) {
      size_t this_bytes = std::min(chunk_bytes, opts.size - offset);
      chunk_sizes[chunk] = this_bytes;
      md_vec[chunk].op = UCCL_READ;
      md_vec[chunk].data.tx_data.data_ptr = remote_ptr + offset;
      md_vec[chunk].data.tx_data.data_size = this_bytes;
      offset += this_bytes;
    }
  }

  std::cout << "[UCCL Smoke] Client chunk size " << chunk_bytes << " bytes ("
            << chunks_per_iter << " chunks per iteration)" << std::endl;

  std::vector<SlotBuf> slots(chunks_per_iter);
  std::vector<uint8_t> host(opts.size);

  int warmup_iters = opts.iters > 1 ? 1 : 0;
  std::chrono::duration<double> total_recv_time(0);

  for (int iter = 0; iter < opts.iters; ++iter) {
    if (!checkCuda(cudaMemset(raw_dev, 0, opts.size),
                   "cudaMemset(client loop)"))
      return 1;

    auto iter_start = std::chrono::high_resolution_clock::now();

    if (uccl_engine_send_tx_md_vector(conn, md_vec.data(), md_vec.size()) < 0) {
      std::cerr << "[UCCL Smoke] client: send_tx_md_vector failed" << std::endl;
      return 1;
    }

    size_t offset = 0;
    for (size_t chunk = 0; chunk < chunks_per_iter; ++chunk) {
      bool got_fifo = false;
      for (int spin = 0; spin < 20000; ++spin) {
        if (uccl_engine_get_fifo_item(conn, static_cast<int>(chunk),
                                      slots[chunk].bytes) == 0) {
          got_fifo = true;
          break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
      if (!got_fifo) {
        std::cerr << "[UCCL Smoke] client: timeout waiting for FIFO item "
                  << chunk << std::endl;
        return 1;
      }

      size_t this_bytes = chunk_sizes[chunk];
      uint8_t* dst_ptr = raw_dev + offset;
      uint64_t transfer_id = 0;
      if (uccl_engine_read(conn, mr, dst_ptr, this_bytes, slots[chunk].bytes,
                           &transfer_id) != 0) {
        std::cerr << "[UCCL Smoke] client: uccl_engine_read failed for chunk "
                  << chunk << std::endl;
        return 1;
      }

      bool done = false;
      for (int spin = 0; spin < 60000; ++spin) {
        done = uccl_engine_xfer_status(conn, transfer_id);
        if (done) break;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
      if (!done) {
        std::cerr << "[UCCL Smoke] client: transfer timeout for chunk " << chunk
                  << std::endl;
        return 1;
      }

      offset += this_bytes;
    }

    auto iter_end = std::chrono::high_resolution_clock::now();
    if (iter >= warmup_iters) {
      total_recv_time +=
          std::chrono::duration_cast<std::chrono::duration<double>>(iter_end -
                                                                    iter_start);
    }

    bool verify_this_iter =
        (opts.iters == 1) || (iter == 0) || (iter == opts.iters - 1);
    if (verify_this_iter) {
      if (!checkCuda(cudaMemcpy(host.data(), raw_dev, opts.size,
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy D2H (client)")) {
        return 1;
      }
      if (!validate_pattern(host.data(), opts.size)) {
        std::cerr << "[UCCL Smoke] client: validation failed on iteration "
                  << (iter + 1) << std::endl;
        return 1;
      }
    }

    if (iter < warmup_iters) {
      std::cout << "[UCCL Smoke] Client warmup iteration " << (iter + 1) << "/"
                << opts.iters << " complete" << std::endl;
    } else {
      std::cout << "[UCCL Smoke] Client iteration " << (iter + 1) << "/"
                << opts.iters << " complete" << std::endl;
    }
  }

  int measured_iters = opts.iters - warmup_iters;
  if (measured_iters > 0) {
    double seconds = total_recv_time.count();
    double total_bytes = static_cast<double>(opts.size) * measured_iters;
    double gib = total_bytes / static_cast<double>(1ull << 30);
    double gib_per_s = seconds > 0.0 ? gib / seconds : 0.0;
    std::cout << "[UCCL Smoke] Client measured " << measured_iters
              << " iteration(s) in " << seconds << " s, bandwidth " << gib_per_s
              << " GiB/s" << std::endl;
  } else {
    std::cout << "[UCCL Smoke] Client ran warmup iteration only; no bandwidth "
                 "reported"
              << std::endl;
  }

  pointer_client_send_ack(ptr_fd);

  uccl_engine_mr_destroy(mr);
  uccl_engine_conn_destroy(conn);
  uccl_engine_destroy(eng);
  std::cout << "[UCCL Smoke] Client PASS" << std::endl;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options opts;
  if (!parse_args(argc, argv, opts)) {
    std::cerr << "Usage: " << argv[0]
              << " --mode=server|client [--ip=ADDR] [--port=NUM] [--gpu=ID]"
                 " [--size=BYTES] [--iters=N]\n";
    return 1;
  }

  if (opts.mode == "server") {
    if (!handshake_server(opts)) return 1;
    return run_server(opts);
  }
  if (!handshake_client(opts)) return 1;
  return run_client(opts);
}
