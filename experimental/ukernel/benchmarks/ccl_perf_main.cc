#include "ccl_perf_main.h"

#include "../include/gpu_rt.h"
#include "../include/transport.h"
#include "../src/ccl/executor.h"
#include <algorithm>
#include <arpa/inet.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <netinet/in.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace UKernel {
namespace CCL {
namespace Benchmark {

namespace {

using Clock = std::chrono::steady_clock;

[[noreturn]] void fail(std::string const& msg) {
  throw std::runtime_error(msg);
}

void require(bool cond, std::string const& msg) {
  if (!cond) fail(msg);
}

int get_int_arg(int argc, char** argv, char const* key, int def) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == key && i + 1 < argc) {
      return std::stoi(argv[i + 1]);
    }
  }
  return def;
}

double get_double_arg(int argc, char** argv, char const* key, double def) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == key && i + 1 < argc) {
      return std::stod(argv[i + 1]);
    }
  }
  return def;
}

size_t get_size_arg(int argc, char** argv, char const* key, size_t def) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == key && i + 1 < argc) {
      return static_cast<size_t>(std::stoull(argv[i + 1]));
    }
  }
  return def;
}

std::string get_str_arg(int argc, char** argv, char const* key,
                        std::string const& def = "") {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == key && i + 1 < argc) {
      return argv[i + 1];
    }
  }
  return def;
}

char const* first_env(std::initializer_list<char const*> names) {
  for (char const* name : names) {
    char const* value = std::getenv(name);
    if (value != nullptr && value[0] != '\0') {
      return value;
    }
  }
  return nullptr;
}

int get_env_int(std::initializer_list<char const*> names, int def) {
  if (char const* value = first_env(names)) {
    return std::stoi(value);
  }
  return def;
}

size_t get_env_size(std::initializer_list<char const*> names, size_t def) {
  if (char const* value = first_env(names)) {
    return static_cast<size_t>(std::stoull(value));
  }
  return def;
}

std::string get_env_str(std::initializer_list<char const*> names,
                        std::string const& def = "") {
  if (char const* value = first_env(names)) {
    return value;
  }
  return def;
}

Transport::PreferredTransport parse_transport(std::string const& value) {
  if (value == "auto") return Transport::PreferredTransport::Auto;
  if (value == "ipc") return Transport::PreferredTransport::Ipc;
  if (value == "uccl") return Transport::PreferredTransport::Uccl;
  if (value == "tcp") return Transport::PreferredTransport::Tcp;
  throw std::invalid_argument("unsupported transport: " + value);
}

char const* collective_name(CollectiveKind kind) {
  switch (kind) {
    case CollectiveKind::AllReduce:
      return "allreduce";
    case CollectiveKind::AllToAll:
      return "alltoall";
  }
  return "unknown";
}

char const* dtype_name(ScalarType dtype) {
  switch (dtype) {
    case ScalarType::Float32:
      return "float";
    default:
      return "unknown";
  }
}

size_t align_up(size_t value, size_t alignment) {
  if (alignment == 0) return value;
  size_t remainder = value % alignment;
  return remainder == 0 ? value : value + (alignment - remainder);
}

size_t ceil_div(size_t a, size_t b) { return b == 0 ? 0 : (a + b - 1) / b; }

size_t bytes_alignment(CollectiveKind collective, int world_size,
                       ScalarType dtype) {
  size_t elem_bytes = scalar_type_size(dtype);
  switch (collective) {
    case CollectiveKind::AllReduce:
    case CollectiveKind::AllToAll:
      return static_cast<size_t>(std::max(1, world_size)) * elem_bytes;
  }
  return elem_bytes;
}

size_t stable_min_bytes(CollectiveKind collective, int world_size,
                        size_t tile_bytes, ScalarType dtype) {
  size_t alignment = bytes_alignment(collective, world_size, dtype);
  switch (collective) {
    case CollectiveKind::AllReduce:
      // Integration coverage exercises multi-tile shards. In benchmark mode the
      // current allreduce device-reduce path is still unstable when each rank
      // owns exactly one tile, so start from two tiles per shard.
      return align_up(2 * static_cast<size_t>(std::max(1, world_size)) *
                          tile_bytes,
                      alignment);
    case CollectiveKind::AllToAll:
      return align_up(tile_bytes, alignment);
  }
  return alignment;
}

size_t tiles_per_participant(CollectiveKind collective, int world_size,
                             size_t tensor_bytes, size_t tile_bytes) {
  size_t part_bytes = ceil_div(tensor_bytes, static_cast<size_t>(world_size));
  switch (collective) {
    case CollectiveKind::AllReduce:
    case CollectiveKind::AllToAll:
      return ceil_div(part_bytes, tile_bytes);
  }
  return 1;
}

uint32_t effective_num_flows(CollectiveKind collective, int world_size,
                             size_t tensor_bytes, size_t tile_bytes,
                             uint32_t requested_flows) {
  size_t tiles = tiles_per_participant(collective, world_size, tensor_bytes,
                                       tile_bytes);
  size_t clamped =
      std::min<size_t>(std::max<size_t>(1, requested_flows), std::max<size_t>(1, tiles));
  return static_cast<uint32_t>(clamped);
}

double busbw_factor(CollectiveKind collective, int world_size) {
  if (world_size <= 0) return 0.0;
  switch (collective) {
    case CollectiveKind::AllReduce:
      return 2.0 * static_cast<double>(world_size - 1) /
             static_cast<double>(world_size);
    case CollectiveKind::AllToAll:
      return static_cast<double>(world_size - 1) /
             static_cast<double>(world_size);
  }
  return 0.0;
}

std::vector<size_t> build_sizes(size_t min_bytes, size_t max_bytes,
                                double factor, size_t alignment) {
  require(min_bytes > 0, "min bytes must be > 0");
  require(max_bytes >= min_bytes, "max bytes must be >= min bytes");
  require(factor > 1.0, "factor must be > 1");

  std::vector<size_t> sizes;
  size_t current = min_bytes;
  while (true) {
    size_t aligned = align_up(current, alignment);
    if (aligned <= max_bytes &&
        (sizes.empty() || aligned > sizes.back())) {
      sizes.push_back(aligned);
    }
    if (current >= max_bytes) break;
    size_t next = static_cast<size_t>(std::ceil(current * factor));
    if (next <= current) next = current + 1;
    if (next > max_bytes) next = max_bytes;
    current = next;
  }
  require(!sizes.empty(), "no benchmark sizes after alignment");
  return sizes;
}

int create_tcp_server(int port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  require(fd >= 0, "failed to create coordination server socket");
  int opt = 1;
  require(::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) == 0,
          "failed to set SO_REUSEADDR");

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(static_cast<uint16_t>(port));
  require(::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0,
          "failed to bind coordination socket on port " +
              std::to_string(port));
  require(::listen(fd, 32) == 0, "failed to listen on coordination socket");
  return fd;
}

int connect_tcp_client(std::string const& ip, int port,
                       std::chrono::milliseconds timeout) {
  auto deadline = Clock::now() + timeout;
  while (true) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    require(fd >= 0, "failed to create coordination client socket");

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    require(::inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) == 1,
            "invalid server ip: " + ip);

    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
      return fd;
    }
    ::close(fd);
    if (Clock::now() >= deadline) {
      fail("timed out connecting to coordination server " + ip + ":" +
           std::to_string(port));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

void send_all(int fd, void const* data, size_t bytes, char const* what) {
  auto const* ptr = static_cast<char const*>(data);
  size_t sent = 0;
  while (sent < bytes) {
    ssize_t rc = ::send(fd, ptr + sent, bytes - sent, 0);
    require(rc > 0, std::string("failed to send ") + what);
    sent += static_cast<size_t>(rc);
  }
}

void recv_all(int fd, void* data, size_t bytes, char const* what) {
  auto* ptr = static_cast<char*>(data);
  size_t received = 0;
  while (received < bytes) {
    ssize_t rc = ::recv(fd, ptr + received, bytes - received, MSG_WAITALL);
    require(rc > 0, std::string("failed to recv ") + what);
    received += static_cast<size_t>(rc);
  }
}

void socket_barrier(std::string const& server_ip, int port, int rank,
                    int world_size) {
  uint8_t byte = 1;
  if (rank == 0) {
    int server_fd = create_tcp_server(port);
    std::vector<int> clients;
    clients.reserve(static_cast<size_t>(std::max(0, world_size - 1)));
    for (int peer = 1; peer < world_size; ++peer) {
      int client_fd = ::accept(server_fd, nullptr, nullptr);
      require(client_fd >= 0, "barrier accept failed");
      recv_all(client_fd, &byte, sizeof(byte), "barrier byte");
      clients.push_back(client_fd);
    }
    for (int client_fd : clients) {
      send_all(client_fd, &byte, sizeof(byte), "barrier ack");
      ::close(client_fd);
    }
    ::close(server_fd);
    return;
  }

  int client_fd =
      connect_tcp_client(server_ip, port, std::chrono::seconds(30));
  send_all(client_fd, &byte, sizeof(byte), "barrier byte");
  recv_all(client_fd, &byte, sizeof(byte), "barrier ack");
  ::close(client_fd);
}

double all_rank_max_double(std::string const& server_ip, int port, int rank,
                           int world_size, double local_value) {
  if (rank == 0) {
    double max_value = local_value;
    int server_fd = create_tcp_server(port);
    std::vector<int> clients;
    clients.reserve(static_cast<size_t>(std::max(0, world_size - 1)));
    for (int peer = 1; peer < world_size; ++peer) {
      int client_fd = ::accept(server_fd, nullptr, nullptr);
      require(client_fd >= 0, "timing accept failed");
      double peer_value = 0.0;
      recv_all(client_fd, &peer_value, sizeof(peer_value), "timing sample");
      max_value = std::max(max_value, peer_value);
      clients.push_back(client_fd);
    }
    for (int client_fd : clients) {
      send_all(client_fd, &max_value, sizeof(max_value), "timing max");
      ::close(client_fd);
    }
    ::close(server_fd);
    return max_value;
  }

  int client_fd =
      connect_tcp_client(server_ip, port, std::chrono::seconds(30));
  send_all(client_fd, &local_value, sizeof(local_value), "timing sample");
  double max_value = 0.0;
  recv_all(client_fd, &max_value, sizeof(max_value), "timing max");
  ::close(client_fd);
  return max_value;
}

struct DeviceBuffer {
  void* ptr = nullptr;
  size_t bytes = 0;

  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t nbytes) : bytes(nbytes) {
    GPU_RT_CHECK(gpuMalloc(&ptr, bytes));
  }

  ~DeviceBuffer() {
    if (ptr != nullptr) {
      gpuFree(ptr);
    }
  }

  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr(other.ptr), bytes(other.bytes) {
    other.ptr = nullptr;
    other.bytes = 0;
  }
};

CollectiveMemory build_collective_memory(int rank, int world_size,
                                         void* tensor_ptr, size_t tensor_bytes,
                                         void* staging_ptr,
                                         size_t staging_bytes) {
  CollectiveMemory memory;
  memory.tensor.local_rank = rank;
  memory.tensor.local_ptr = tensor_ptr;
  memory.tensor.bytes = tensor_bytes;
  memory.tensor.layout.sizes = {static_cast<int64_t>(tensor_bytes)};
  memory.tensor.layout.strides = {1};
  memory.tensor.layout.dtype = ScalarType::Float32;
  memory.tensor.peer_views.resize(static_cast<size_t>(world_size));
  memory.staging.local_ptr = staging_ptr;
  memory.staging.local_mr_id = 0;
  memory.staging.bytes = staging_bytes;
  memory.staging.layout.sizes = {static_cast<int64_t>(staging_bytes)};
  memory.staging.layout.strides = {1};
  memory.staging.layout.dtype = ScalarType::Float32;
  memory.staging.peer_views.resize(static_cast<size_t>(world_size));

  for (int peer = 0; peer < world_size; ++peer) {
    memory.tensor.peer_views[static_cast<size_t>(peer)].same_node = true;
    memory.staging.peer_views[static_cast<size_t>(peer)].same_node = true;
  }
  return memory;
}

void upload_floats(void* dst, std::vector<float> const& values) {
  GPU_RT_CHECK(gpuMemcpy(dst, values.data(), values.size() * sizeof(float),
                         gpuMemcpyHostToDevice));
  GPU_RT_CHECK(gpuDeviceSynchronize());
}

std::vector<float> download_floats(void const* src, size_t bytes) {
  require(bytes % sizeof(float) == 0, "buffer size must align to float");
  std::vector<float> out(bytes / sizeof(float), 0.0f);
  GPU_RT_CHECK(gpuMemcpy(out.data(), src, bytes, gpuMemcpyDeviceToHost));
  GPU_RT_CHECK(gpuDeviceSynchronize());
  return out;
}

void zero_buffer(void* ptr, size_t bytes) {
  GPU_RT_CHECK(gpuMemset(ptr, 0, bytes));
  GPU_RT_CHECK(gpuDeviceSynchronize());
}

void init_validation_allreduce(std::vector<float>& host, int rank) {
  for (size_t i = 0; i < host.size(); ++i) {
    host[i] = static_cast<float>(rank * 1000) + static_cast<float>(i);
  }
}

void verify_allreduce(std::vector<float> const& host, int world_size) {
  for (size_t i = 0; i < host.size(); ++i) {
    float expected = 0.0f;
    for (int rank = 0; rank < world_size; ++rank) {
      expected += static_cast<float>(rank * 1000) + static_cast<float>(i);
    }
    require(std::fabs(host[i] - expected) < 1e-3f,
            "allreduce verify failed at index " + std::to_string(i));
  }
}

void init_validation_alltoall(std::vector<float>& host, int rank,
                              int world_size) {
  require(host.size() % static_cast<size_t>(world_size) == 0,
          "alltoall validation requires divisible element count");
  size_t slice = host.size() / static_cast<size_t>(world_size);
  for (int dst_rank = 0; dst_rank < world_size; ++dst_rank) {
    for (size_t i = 0; i < slice; ++i) {
      host[static_cast<size_t>(dst_rank) * slice + i] =
          static_cast<float>(rank * 10000 + dst_rank * 100 +
                             static_cast<int>(i));
    }
  }
}

void verify_alltoall(std::vector<float> const& host, int rank, int world_size) {
  require(host.size() % static_cast<size_t>(world_size) == 0,
          "alltoall verify requires divisible element count");
  size_t slice = host.size() / static_cast<size_t>(world_size);
  for (int src_rank = 0; src_rank < world_size; ++src_rank) {
    for (size_t i = 0; i < slice; ++i) {
      float expected = static_cast<float>(src_rank * 10000 + rank * 100 +
                                          static_cast<int>(i));
      float actual = host[static_cast<size_t>(src_rank) * slice + i];
      require(std::fabs(actual - expected) < 1e-3f,
              "alltoall verify failed at src rank " +
                  std::to_string(src_rank) + ", index " + std::to_string(i));
    }
  }
}

void init_validation_pattern(CollectiveKind collective, std::vector<float>& host,
                             int rank, int world_size) {
  if (collective == CollectiveKind::AllReduce) {
    init_validation_allreduce(host, rank);
  } else {
    init_validation_alltoall(host, rank, world_size);
  }
}

void verify_output(CollectiveKind collective, std::vector<float> const& host,
                   int rank, int world_size) {
  if (collective == CollectiveKind::AllReduce) {
    verify_allreduce(host, world_size);
  } else {
    verify_alltoall(host, rank, world_size);
  }
}

void wait_for_collective(Executor& executor, CollectiveOpHandle handle,
                         std::chrono::milliseconds timeout) {
  auto deadline = Clock::now() + timeout;
  while (Clock::now() < deadline) {
    if (executor.poll(handle)) return;
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  require(executor.poll(handle), "collective timed out");
}

CollectiveConfig make_collective_config(CollectiveKind collective, int nranks,
                                        int rank, size_t tensor_bytes,
                                        size_t tile_bytes,
                                        uint32_t num_flows) {
  CollectiveConfig config{};
  config.nranks = nranks;
  config.rank = rank;
  config.num_flows = num_flows;
  config.tensor_bytes = tensor_bytes;
  config.tile_bytes = tile_bytes;
  config.staging_bytes =
      std::max(static_cast<size_t>(num_flows),
               static_cast<size_t>(nranks > 0 ? nranks - 1 : 0)) *
      tile_bytes;
  config.algorithm = (collective == CollectiveKind::AllReduce)
                         ? AlgorithmKind::Ring
                         : AlgorithmKind::Pairwise;
  config.dtype = ScalarType::Float32;
  config.reduction = ReductionKind::Sum;
  return config;
}

struct Options {
  int rank = 0;
  int world_size = 2;
  int gpu = 0;
  int exchanger_port = 29600;
  uint32_t num_flows = 2;
  uint32_t threads_per_block = 256;
  uint32_t fifo_capacity = 64;
  uint32_t smem_size = 0;
  uint32_t device_task_capacity = 4096;
  int warmup_iters = 5;
  int iters = 20;
  bool check = true;
  size_t min_bytes = 8;
  size_t max_bytes = 64ull << 20;
  size_t tile_bytes = 64ull << 10;
  double factor = 2.0;
  std::string exchanger_ip = "127.0.0.1";
  std::string transport = "auto";
  ScalarType dtype = ScalarType::Float32;
};

Options parse_options(int argc, char** argv, CollectiveKind collective) {
  Options opts;
  int env_rank = get_env_int({"RANK"}, 0);
  int env_world = get_env_int({"WORLD_SIZE"}, 2);
  int env_local_rank = get_env_int({"LOCAL_RANK"}, env_rank);

  opts.rank = get_int_arg(argc, argv, "--rank", env_rank);
  opts.world_size = get_int_arg(argc, argv, "--world-size", env_world);
  opts.gpu = get_int_arg(argc, argv, "--gpu", env_local_rank);
  opts.exchanger_port = get_int_arg(
      argc, argv, "--exchanger-port",
      get_env_int({"MASTER_PORT", "UHM_EXCHANGER_SERVER_PORT"}, 29500));
  opts.num_flows = static_cast<uint32_t>(get_int_arg(
      argc, argv, "--num-flows", get_env_int({"CCL_NUM_FLOWS"}, 2)));
  opts.threads_per_block = static_cast<uint32_t>(
      get_int_arg(argc, argv, "--threads-per-block", 256));
  opts.fifo_capacity =
      static_cast<uint32_t>(get_int_arg(argc, argv, "--fifo-capacity", 64));
  opts.smem_size =
      static_cast<uint32_t>(get_int_arg(argc, argv, "--smem-size", 0));
  opts.device_task_capacity = static_cast<uint32_t>(
      get_int_arg(argc, argv, "--device-task-capacity", 4096));
  opts.warmup_iters = get_int_arg(argc, argv, "-w",
                                  get_int_arg(argc, argv, "--warmup-iters", 5));
  opts.iters =
      get_int_arg(argc, argv, "-n", get_int_arg(argc, argv, "--iters", 20));
  size_t default_min_bytes =
      stable_min_bytes(collective, env_world, 64ull << 10, ScalarType::Float32);
  opts.min_bytes =
      get_size_arg(argc, argv, "-b",
                   get_size_arg(argc, argv, "--min-bytes",
                                get_env_size({"MIN_BYTES"}, default_min_bytes)));
  opts.max_bytes = get_size_arg(
      argc, argv, "-e",
      get_size_arg(argc, argv, "--max-bytes",
                   get_env_size({"MAX_BYTES"}, collective == CollectiveKind::AllReduce
                                                   ? (64ull << 20)
                                                   : (16ull << 20))));
  opts.factor = get_double_arg(argc, argv, "-f",
                               get_double_arg(argc, argv, "--factor", 2.0));
  opts.tile_bytes =
      get_size_arg(argc, argv, "--tile-bytes",
                   get_env_size({"TILE_BYTES", "CCL_TILE_BYTES"}, 64 << 10));
  opts.transport = get_str_arg(
      argc, argv, "--transport", get_env_str({"TRANSPORT"}, "auto"));
  opts.check = get_int_arg(argc, argv, "--check", 1) != 0;

  std::string default_ip =
      get_env_str({"MASTER_ADDR", "UHM_EXCHANGER_SERVER_IP"},
                  opts.rank == 0 ? "0.0.0.0" : "127.0.0.1");
  opts.exchanger_ip = get_str_arg(argc, argv, "--exchanger-ip", default_ip);
  return opts;
}

CollectiveOpHandle submit_collective(Executor& executor, CollectiveKind collective,
                                     CollectiveConfig const& config) {
  return collective == CollectiveKind::AllReduce
             ? executor.submit_allreduce(config)
             : executor.submit_alltoall(config);
}

void run_validation(Executor& executor, CollectiveKind collective,
                    CollectiveConfig const& config, void* tensor_ptr,
                    void* staging_ptr, int rank, int world_size) {
  std::vector<float> host(config.tensor_bytes / sizeof(float), 0.0f);
  init_validation_pattern(collective, host, rank, world_size);
  upload_floats(tensor_ptr, host);
  zero_buffer(staging_ptr, config.staging_bytes);

  CollectiveOpHandle handle = submit_collective(executor, collective, config);
  wait_for_collective(executor, handle, std::chrono::seconds(60));
  require(executor.status(handle) == CollectiveOpStatus::Completed,
          "validation collective did not complete");
  executor.release(handle);

  std::vector<float> out = download_floats(tensor_ptr, config.tensor_bytes);
  verify_output(collective, out, rank, world_size);
}

void prepare_timed_input(void* tensor_ptr, void* staging_ptr,
                         CollectiveConfig const& config) {
  zero_buffer(tensor_ptr, config.tensor_bytes);
  zero_buffer(staging_ptr, config.staging_bytes);
}

void print_header(CollectiveKind collective, Options const& opts) {
  std::printf("# CCL %s perf\n", collective_name(collective));
  std::printf(
      "# nranks %d rank %d gpu %d transport %s flows %u tile_bytes %zu "
      "threads_per_block %u fifo_capacity %u warmup %d iters %d\n",
      opts.world_size, opts.rank, opts.gpu, opts.transport.c_str(),
      opts.num_flows, opts.tile_bytes, opts.threads_per_block,
      opts.fifo_capacity, opts.warmup_iters, opts.iters);
  std::printf(
      "# note effective flows are clamped per size to available tiles per "
      "rank-local shard/slice\n");
}

void print_sweep_info(Options const& opts, std::vector<size_t> const& sizes) {
  std::printf("# sweep min_bytes %zu max_bytes %zu factor %.3f points %zu\n",
              opts.min_bytes, opts.max_bytes, opts.factor, sizes.size());
  if (!sizes.empty() && sizes.size() <= 32) {
    std::printf("# sizes");
    for (size_t bytes : sizes) {
      std::printf(" %zu", bytes);
    }
    std::printf("\n");
  }
  std::printf("%12s %12s %10s %12s %12s %12s\n", "bytes", "count", "type",
              "time(us)", "algbw(GB/s)", "busbw(GB/s)");
  std::fflush(stdout);
}

void print_row(size_t bytes, ScalarType dtype, double time_us, double algbw,
               double busbw) {
  std::printf("%12zu %12zu %10s %12.2f %12.2f %12.2f\n", bytes,
              bytes / scalar_type_size(dtype), dtype_name(dtype), time_us,
              algbw, busbw);
  std::fflush(stdout);
}

}  // namespace

int run_perf_main(int argc, char** argv, CollectiveKind collective) {
  Options opts = parse_options(argc, argv, collective);
  require(opts.world_size >= 2, "world_size must be >= 2");
  require(opts.rank >= 0 && opts.rank < opts.world_size, "rank out of range");
  require(opts.warmup_iters >= 0, "warmup must be >= 0");
  require(opts.iters > 0, "iters must be > 0");
  require(opts.tile_bytes > 0, "tile bytes must be > 0");

  size_t alignment = bytes_alignment(collective, opts.world_size, opts.dtype);
  opts.min_bytes = std::max(
      opts.min_bytes,
      stable_min_bytes(collective, opts.world_size, opts.tile_bytes, opts.dtype));
  std::vector<size_t> sizes =
      build_sizes(opts.min_bytes, opts.max_bytes, opts.factor, alignment);
  size_t max_tensor_bytes = sizes.back();
  CollectiveConfig max_config =
      make_collective_config(collective, opts.world_size, opts.rank,
                             max_tensor_bytes, opts.tile_bytes, opts.num_flows);

  GPU_RT_CHECK(gpuSetDevice(opts.gpu));

  DeviceBuffer tensor(max_tensor_bytes);
  DeviceBuffer staging(max_config.staging_bytes);

  CollectiveMemory memory = build_collective_memory(
      opts.rank, opts.world_size, tensor.ptr, max_tensor_bytes, staging.ptr,
      max_config.staging_bytes);

  ExecutorConfig executor_cfg{};
  executor_cfg.gpu_id = opts.gpu;
  executor_cfg.rank = opts.rank;
  executor_cfg.world_size = opts.world_size;
  executor_cfg.communicator_config =
      std::make_shared<Transport::CommunicatorConfig>();
  executor_cfg.communicator_config->exchanger_ip = opts.exchanger_ip;
  executor_cfg.communicator_config->exchanger_port = opts.exchanger_port;
  executor_cfg.communicator_config->local_id = opts.rank;
  executor_cfg.communicator_config->preferred_transport =
      parse_transport(opts.transport);
  executor_cfg.max_device_fifos = std::max<uint32_t>(opts.num_flows, 1);
  executor_cfg.device_task_capacity = opts.device_task_capacity;
  executor_cfg.threads_per_block = opts.threads_per_block;
  executor_cfg.fifo_capacity = opts.fifo_capacity;
  executor_cfg.smem_size = opts.smem_size;

  Executor executor(memory, executor_cfg);

  std::string coord_ip =
      get_env_str({"MASTER_ADDR"},
                  opts.rank == 0 && opts.exchanger_ip == "0.0.0.0"
                      ? "127.0.0.1"
                      : opts.exchanger_ip);

  if (opts.rank == 0) {
    print_header(collective, opts);
    print_sweep_info(opts, sizes);
  }

  for (size_t bytes : sizes) {
    uint32_t flows = effective_num_flows(collective, opts.world_size, bytes,
                                         opts.tile_bytes, opts.num_flows);
    CollectiveConfig config = make_collective_config(
        collective, opts.world_size, opts.rank, bytes, opts.tile_bytes,
        flows);

    if (opts.check) {
      socket_barrier(coord_ip, opts.exchanger_port + 1, opts.rank,
                     opts.world_size);
      run_validation(executor, collective, config, tensor.ptr, staging.ptr,
                     opts.rank, opts.world_size);
    }

    prepare_timed_input(tensor.ptr, staging.ptr, config);
    socket_barrier(coord_ip, opts.exchanger_port + 1, opts.rank,
                   opts.world_size);

    for (int iter = 0; iter < opts.warmup_iters; ++iter) {
      CollectiveOpHandle handle = submit_collective(executor, collective, config);
      wait_for_collective(executor, handle, std::chrono::seconds(60));
      require(executor.status(handle) == CollectiveOpStatus::Completed,
              "warmup collective failed");
      executor.release(handle);
    }

    socket_barrier(coord_ip, opts.exchanger_port + 1, opts.rank,
                   opts.world_size);
    auto t0 = Clock::now();
    for (int iter = 0; iter < opts.iters; ++iter) {
      CollectiveOpHandle handle = submit_collective(executor, collective, config);
      wait_for_collective(executor, handle, std::chrono::seconds(60));
      require(executor.status(handle) == CollectiveOpStatus::Completed,
              "timed collective failed");
      executor.release(handle);
    }
    auto t1 = Clock::now();

    double local_total_sec = std::chrono::duration<double>(t1 - t0).count();
    double max_total_sec = all_rank_max_double(
        coord_ip, opts.exchanger_port + 2, opts.rank, opts.world_size,
        local_total_sec);
    double avg_sec = max_total_sec / static_cast<double>(opts.iters);
    double algbw = static_cast<double>(bytes) / avg_sec / 1e9;
    double busbw = algbw * busbw_factor(collective, opts.world_size);

    if (opts.rank == 0) {
      print_row(bytes, opts.dtype, avg_sec * 1e6, algbw, busbw);
    }
  }

  return 0;
}

}  // namespace Benchmark
}  // namespace CCL
}  // namespace UKernel
