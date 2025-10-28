#include "tcpx_engine.h"

#include <cuda_runtime.h>

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
#include <type_traits>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>

namespace {

constexpr size_t kMaxChunkBytes = 4 * 1024 * 1024;  // Device-side descriptor limit (~8MB slack)

bool checkCuda(cudaError_t err, char const* where) {
  if (err != cudaSuccess) {
    std::cerr << "[CUDA] " << where << " failed: " << cudaGetErrorString(err)
              << std::endl;
    return false;
  }
  return true;
}

bool poll_transfer(tcpx::Endpoint& endpoint, uint64_t transfer_id) {
  constexpr int kSleepMicros = 100;
  bool done = false;
  while (!done) {
    if (!endpoint.poll_async(transfer_id, &done)) {
      std::cerr << "[TCPX] poll_async failed" << std::endl;
      return false;
    }
    if (!done) {
      std::this_thread::sleep_for(std::chrono::microseconds(kSleepMicros));
    }
  }
  return true;
}

struct Options {
  std::string mode;
  std::string ip{"127.0.0.1"};
  int port = 9999;
  int gpu = 0;
  size_t size = 1 << 26;  // 64 MiB default payload
  int iters = 10;         // includes one warmup iteration (if >1)
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
      out.size = static_cast<size_t>(std::stoul(value));
    } else if (key == "--iters") {
      int parsed = std::stoi(value);
      out.iters = parsed < 1 ? 1 : parsed;
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

std::vector<uint8_t> make_pattern(size_t n) {
  std::vector<uint8_t> v(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = static_cast<uint8_t>(i & 0xff);
  }
  return v;
}

bool validate_pattern(uint8_t const* data, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    uint8_t expect = static_cast<uint8_t>(i & 0xff);
    if (data[i] != expect) {
      std::cerr << "[TCPX Smoke] Data mismatch at index " << i
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
  if (chunk == 0) {
    chunk = get_env_size("NCCL_P2P_NET_CHUNKSIZE", 512 * 1024);
  }
  if (chunk == 0) chunk = 512 * 1024;
  chunk = std::max<size_t>(1, std::min(chunk, total_size));
  chunk = std::min(chunk, kMaxChunkBytes);
  return chunk;
}

struct CudaDeleter {
  void operator()(uint8_t* ptr) const {
    if (ptr) cudaFree(ptr);
  }
};

using DeviceBuffer = std::unique_ptr<uint8_t, CudaDeleter>;

struct MrGuard {
  tcpx::Endpoint* endpoint = nullptr;
  uint64_t id = 0;
  ~MrGuard() {
    if (endpoint && id) {
      endpoint->dereg(id);
    }
  }
};

struct HandshakePayload {
  uint64_t payload_bytes;
  uint32_t iterations;
  uint32_t reserved;
};
static_assert(std::is_trivially_copyable<HandshakePayload>::value,
              "HandshakePayload must be POD");

constexpr int kHandshakePortOffset = 1;

bool handshake_server(Options& opts) {
  int listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd < 0) {
    std::cerr << "[TCPX Smoke] handshake server socket() failed: "
              << std::strerror(errno) << std::endl;
    return false;
  }

  int reuse = 1;
  setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(static_cast<uint16_t>(opts.port + kHandshakePortOffset));
  if (bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    std::cerr << "[TCPX Smoke] handshake bind() failed: "
              << std::strerror(errno) << std::endl;
    ::close(listen_fd);
    return false;
  }
  if (listen(listen_fd, 1) < 0) {
    std::cerr << "[TCPX Smoke] handshake listen() failed: "
              << std::strerror(errno) << std::endl;
    ::close(listen_fd);
    return false;
  }

  int conn_fd = ::accept(listen_fd, nullptr, nullptr);
  if (conn_fd < 0) {
    std::cerr << "[TCPX Smoke] handshake accept() failed: "
              << std::strerror(errno) << std::endl;
    ::close(listen_fd);
    return false;
  }

  HandshakePayload payload{};
  ssize_t want = sizeof(payload);
  ssize_t got = ::recv(conn_fd, &payload, want, MSG_WAITALL);
  if (got != want) {
    std::cerr << "[TCPX Smoke] handshake recv() failed: "
              << std::strerror(errno) << std::endl;
    ::close(conn_fd);
    ::close(listen_fd);
    return false;
  }

  if (payload.payload_bytes > 0) {
    opts.size = static_cast<size_t>(payload.payload_bytes);
  }
  if (payload.iterations > 0) {
    opts.iters = static_cast<int>(payload.iterations);
  }

  uint32_t ack = 1;
  ::send(conn_fd, &ack, sizeof(ack), 0);

  ::close(conn_fd);
  ::close(listen_fd);
  return true;
}

bool handshake_client(Options const& opts) {
  int sock_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (sock_fd < 0) {
    std::cerr << "[TCPX Smoke] handshake client socket() failed: "
              << std::strerror(errno) << std::endl;
    return false;
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(opts.port + kHandshakePortOffset));
  if (inet_pton(AF_INET, opts.ip.c_str(), &addr.sin_addr) != 1) {
    std::cerr << "[TCPX Smoke] handshake invalid IP " << opts.ip << std::endl;
    ::close(sock_fd);
    return false;
  }

  constexpr int kMaxConnectRetries = 50;
  constexpr int kConnectSleepMs = 100;
  bool connected = false;
  for (int attempt = 0; attempt < kMaxConnectRetries; ++attempt) {
    if (connect(sock_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
      connected = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kConnectSleepMs));
  }
  if (!connected) {
    std::cerr << "[TCPX Smoke] handshake connect() failed after retries: "
              << std::strerror(errno) << std::endl;
    ::close(sock_fd);
    return false;
  }

  HandshakePayload payload{};
  payload.payload_bytes = static_cast<uint64_t>(opts.size);
  payload.iterations = static_cast<uint32_t>(std::max(opts.iters, 1));

  ssize_t want = sizeof(payload);
  ssize_t sent = ::send(sock_fd, &payload, want, 0);
  if (sent != want) {
    std::cerr << "[TCPX Smoke] handshake send() failed: "
              << std::strerror(errno) << std::endl;
    ::close(sock_fd);
    return false;
  }

  uint32_t ack = 0;
  ssize_t got = ::recv(sock_fd, &ack, sizeof(ack), MSG_WAITALL);
  if (got != sizeof(ack) || ack != 1) {
    std::cerr << "[TCPX Smoke] handshake ack failed" << std::endl;
    ::close(sock_fd);
    return false;
  }

  ::close(sock_fd);
  return true;
}

int run_server(tcpx::Endpoint& endpoint, Options opts) {
  std::cout << "[TCPX Smoke] Waiting for client on port " << opts.port
            << "...\n";
  std::string peer_ip;
  int peer_gpu = -1;
  uint64_t conn_id = 0;
  if (!endpoint.accept(peer_ip, peer_gpu, conn_id)) {
    std::cerr << "[TCPX] accept failed" << std::endl;
    return 1;
  }

  std::cout << "[TCPX Smoke] Accepted connection from " << peer_ip
            << " (peer GPU " << peer_gpu << "), expecting " << opts.iters
            << " iteration(s) of " << opts.size << " bytes\n";

  uint8_t* raw_dev = nullptr;
  if (!checkCuda(cudaMalloc(&raw_dev, opts.size), "cudaMalloc")) return 1;
  DeviceBuffer dev_buf(raw_dev);

  uint64_t mr_id = 0;
  if (!endpoint.reg(raw_dev, opts.size, mr_id)) {
    std::cerr << "[TCPX] reg failed" << std::endl;
    return 1;
  }
  MrGuard mr_guard{&endpoint, mr_id};

  std::vector<uint8_t> host_buf(opts.size);
  std::chrono::duration<double> total_unpack_time(0);
  int const warmup_iters = opts.iters > 1 ? 1 : 0;
  size_t const chunk_bytes = resolve_chunk_bytes(opts.size);
  size_t const chunks_per_iter =
      (opts.size + chunk_bytes - 1) / chunk_bytes;
  std::cout << "[TCPX Smoke] Server chunk size " << chunk_bytes
            << " bytes (" << chunks_per_iter << " chunks per iteration)"
            << std::endl;

  for (int iter = 0; iter < opts.iters; ++iter) {
    if (!checkCuda(cudaMemset(raw_dev, 0, opts.size), "cudaMemset")) return 1;

    auto iter_start = std::chrono::high_resolution_clock::now();
    size_t offset = 0;
    for (size_t chunk = 0; chunk < chunks_per_iter; ++chunk) {
      size_t const this_bytes =
          std::min(chunk_bytes, opts.size - offset);
      uint8_t* dst_ptr = raw_dev + offset;
      uint64_t transfer_id = 0;
      if (!endpoint.recv_async(conn_id, mr_id, dst_ptr, this_bytes,
                               &transfer_id)) {
        std::cerr << "[TCPX] recv_async failed" << std::endl;
        return 1;
      }
      if (!poll_transfer(endpoint, transfer_id)) return 1;
      offset += this_bytes;
    }
    auto iter_end = std::chrono::high_resolution_clock::now();
    if (iter >= warmup_iters) {
      total_unpack_time +=
          std::chrono::duration_cast<std::chrono::duration<double>>(iter_end -
                                                                    iter_start);
    }

    bool verify_this_iter =
        (opts.iters == 1) || (iter == 0) || (iter == opts.iters - 1);
    if (verify_this_iter) {
      // Spot-check the first and last iterations to ensure payload integrity
      // without incurring the full host validation cost on every loop.
      if (!checkCuda(cudaMemcpy(host_buf.data(), raw_dev, opts.size,
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy D2H")) {
        return 1;
      }
      if (!validate_pattern(host_buf.data(), opts.size)) {
        std::cerr << "[TCPX Smoke] Validation failed on iteration "
                  << iter + 1 << std::endl;
        return 1;
      }
    }
    if (iter < warmup_iters) {
      std::cout << "[TCPX Smoke] Warmup iteration " << (iter + 1)
                << "/" << opts.iters << " complete\n";
    } else {
      std::cout << "[TCPX Smoke] Iteration " << iter + 1 << "/"
                << opts.iters << " complete\n";
    }
  }

  std::cout << "[TCPX Smoke] Receive verification passed (" << opts.size
            << " bytes x " << opts.iters << " iteration(s))" << std::endl;
  int const measured_iters = opts.iters - warmup_iters;
  if (measured_iters > 0) {
    double server_seconds = total_unpack_time.count();
    double server_total_bytes =
        static_cast<double>(opts.size) * static_cast<double>(measured_iters);
    double server_gib =
        server_total_bytes / static_cast<double>(1ull << 30);
    double server_gbps =
        server_seconds > 0.0 ? server_gib / server_seconds : 0.0;
    std::cout << "[TCPX Smoke] Server observed throughput: " << server_gbps
              << " GiB/s (" << measured_iters << " measured iteration(s) in "
              << server_seconds << " s)" << std::endl;
  } else {
    std::cout << "[TCPX Smoke] Only warmup iteration executed on server; no "
                 "bandwidth reported.\n";
  }
  return 0;
}

int run_client(tcpx::Endpoint& endpoint, Options opts) {
  uint8_t* raw_dev = nullptr;
  if (!checkCuda(cudaMalloc(&raw_dev, opts.size), "cudaMalloc")) return 1;
  DeviceBuffer dev_buf(raw_dev);

  uint64_t mr_id = 0;
  if (!endpoint.reg(raw_dev, opts.size, mr_id)) {
    std::cerr << "[TCPX] reg failed" << std::endl;
    return 1;
  }
  MrGuard mr_guard{&endpoint, mr_id};

  auto payload = make_pattern(opts.size);
  if (!checkCuda(cudaMemcpy(raw_dev, payload.data(), opts.size,
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy H2D"))
    return 1;

  std::cout << "[TCPX Smoke] Connecting to " << opts.ip << ":" << opts.port
            << "...\n";
  uint64_t conn_id = 0;
  if (!endpoint.connect(opts.ip, /*remote_gpu_idx=*/0, opts.port, conn_id)) {
    std::cerr << "[TCPX] connect failed" << std::endl;
    return 1;
  }

  std::chrono::duration<double> total_send_time(0);
  int const warmup_iters = opts.iters > 1 ? 1 : 0;
  size_t const chunk_bytes = resolve_chunk_bytes(opts.size);
  size_t const chunks_per_iter =
      (opts.size + chunk_bytes - 1) / chunk_bytes;
  std::cout << "[TCPX Smoke] Client chunk size " << chunk_bytes
            << " bytes (" << chunks_per_iter << " chunks per iteration)"
            << std::endl;

  for (int iter = 0; iter < opts.iters; ++iter) {
    auto iter_start = std::chrono::high_resolution_clock::now();
    size_t offset = 0;
    for (size_t chunk = 0; chunk < chunks_per_iter; ++chunk) {
      size_t const this_bytes =
          std::min(chunk_bytes, opts.size - offset);
      uint8_t* src_ptr = raw_dev + offset;
      uint64_t transfer_id = 0;
      if (!endpoint.send_async(conn_id, mr_id, src_ptr, this_bytes,
                               &transfer_id)) {
        std::cerr << "[TCPX] send_async failed" << std::endl;
        return 1;
      }
      if (!poll_transfer(endpoint, transfer_id)) return 1;
      offset += this_bytes;
    }
    auto iter_end = std::chrono::high_resolution_clock::now();
    if (iter >= warmup_iters) {
      total_send_time +=
          std::chrono::duration_cast<std::chrono::duration<double>>(iter_end -
                                                                    iter_start);
    }
    if (iter < warmup_iters) {
      std::cout << "[TCPX Smoke] Warmup send complete (" << opts.size
                << " bytes) iteration " << iter + 1 << "/" << opts.iters
                << std::endl;
    } else {
      std::cout << "[TCPX Smoke] Payload sent (" << opts.size
                << " bytes) iteration " << iter + 1 << "/" << opts.iters
                << std::endl;
    }
  }
  int const measured_iters = opts.iters - warmup_iters;
  if (measured_iters > 0) {
    double seconds = total_send_time.count();
    double total_bytes = static_cast<double>(opts.size) * measured_iters;
    double gib =
        total_bytes / static_cast<double>(1ull << 30);  // binary GiB
    double gbps = seconds > 0.0 ? gib / seconds : 0.0;
    std::cout << "[TCPX Smoke] Completed " << measured_iters
              << " measured iteration(s) in " << seconds
              << " s, throughput " << gbps << " GiB/s" << std::endl;
  } else {
    std::cout << "[TCPX Smoke] Only warmup iteration executed on client; no "
                 "bandwidth reported.\n";
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options opts;
  if (!parse_args(argc, argv, opts)) {
    std::cerr
        << "Usage: " << argv[0]
        << " --mode=server|client [--ip=ADDR] [--port=NUM] [--gpu=ID] "
           "[--size=BYTES] [--iters=N]\n";
    return 1;
  }

  // Ensure both sides use the same TCP control port.
  if (opts.mode == "server") {
    if (!handshake_server(opts)) return 1;
  } else {
    if (!handshake_client(opts)) return 1;
  }

  std::string port_str = std::to_string(opts.port);
  setenv("UCCL_TCPX_OOB_PORT", port_str.c_str(), /*overwrite=*/1);
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
  setenv("NCCL_GPUDIRECTTCPX_TX_BINDINGS",
         "eth1:8-21,112-125;eth2:8-21,112-125;eth3:60-73,164-177;eth4:60-73,164-177", 0);
  setenv("NCCL_GPUDIRECTTCPX_RX_BINDINGS",
         "eth1:22-35,126-139;eth2:22-35,126-139;eth3:74-87,178-191;eth4:74-87,178-191", 0);
  setenv("NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS", "50000", 0);
  setenv("NCCL_GPUDIRECTTCPX_FORCE_ACK", "0", 0);
  setenv("NCCL_GPUDIRECTTCPX_TX_COMPLETION_NANOSLEEP", "100", 0);
  setenv("NCCL_GPUDIRECTTCPX_RX_COMPLETION_NANOSLEEP", "100", 0);
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
  setenv("UCCL_TCPX_KERNEL_DEBUG", "0", 0);

  if (!checkCuda(cudaSetDevice(opts.gpu), "cudaSetDevice")) return 1;

  tcpx::Endpoint endpoint(static_cast<uint32_t>(opts.gpu), /*num_cpus=*/1);

  return (opts.mode == "server") ? run_server(endpoint, opts)
                                 : run_client(endpoint, opts);
}
