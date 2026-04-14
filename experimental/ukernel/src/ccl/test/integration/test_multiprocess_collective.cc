#include "backend_test_utils.h"
#include "gpu_rt.h"
#include "transport.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <sys/socket.h>
#include <unistd.h>

namespace UKernel {
namespace CCL {

namespace {

constexpr BufferId kTestInputBufferId = 7;
constexpr BufferId kTestScratchBufferId = 11;
constexpr CollectiveBufferRoles kTestRoles{
    kTestInputBufferId, kTestInputBufferId, kTestScratchBufferId};

size_t ceil_div(size_t a, size_t b) { return b == 0 ? 0 : (a + b - 1) / b; }

size_t default_test_bytes_per_rank(int world_size) {
  size_t base = 1u << 20;
  size_t alignment =
      static_cast<size_t>(std::max(1, world_size)) * sizeof(float);
  return base - (base % alignment);
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

[[noreturn]] void fail(std::string const& msg) {
  throw std::runtime_error(msg);
}

void require(bool cond, std::string const& msg) {
  if (!cond) fail(msg);
}

std::string preview_window(std::vector<float> const& host, size_t index,
                           size_t radius = 2) {
  if (host.empty()) return "[]";
  size_t begin = index > radius ? index - radius : 0;
  size_t end = std::min(host.size(), index + radius + 1);
  std::string out = "[";
  for (size_t i = begin; i < end; ++i) {
    if (i != begin) out += ", ";
    out += std::to_string(host[i]);
    if (i == index) out += "*";
  }
  out += "]";
  return out;
}

int create_tcp_server(int port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  require(fd >= 0, "failed to create barrier server socket");
  int opt = 1;
  require(::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) == 0,
          "failed to set SO_REUSEADDR on barrier socket");

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(static_cast<uint16_t>(port));
  require(::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0,
          "failed to bind barrier socket on port " + std::to_string(port));
  require(::listen(fd, 16) == 0, "failed to listen on barrier socket");
  return fd;
}

int connect_tcp_client(std::string const& ip, int port,
                       std::chrono::milliseconds timeout) {
  auto deadline = std::chrono::steady_clock::now() + timeout;
  while (true) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    require(fd >= 0, "failed to create barrier client socket");

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    require(::inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) == 1,
            "invalid barrier server ip: " + ip);

    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
      return fd;
    }
    ::close(fd);
    if (std::chrono::steady_clock::now() >= deadline) {
      fail("timed out connecting to barrier server " + ip + ":" +
           std::to_string(port));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

void write_barrier_byte(int fd) {
  char byte = 1;
  ssize_t rc = ::send(fd, &byte, sizeof(byte), 0);
  require(rc == static_cast<ssize_t>(sizeof(byte)),
          "failed to send barrier byte");
}

void read_barrier_byte(int fd) {
  char byte = 0;
  ssize_t rc = ::recv(fd, &byte, sizeof(byte), MSG_WAITALL);
  require(rc == static_cast<ssize_t>(sizeof(byte)),
          "failed to receive barrier byte");
}

void socket_barrier(std::string const& server_ip, int port, int rank,
                    int world_size, char const* label) {
  if (rank == 0) {
    int server_fd = create_tcp_server(port);
    std::vector<int> client_fds;
    client_fds.reserve(static_cast<size_t>(world_size - 1));
    for (int i = 1; i < world_size; ++i) {
      int client_fd = ::accept(server_fd, nullptr, nullptr);
      require(client_fd >= 0,
              std::string("barrier accept failed for ") + label);
      read_barrier_byte(client_fd);
      client_fds.push_back(client_fd);
    }
    for (int client_fd : client_fds) {
      write_barrier_byte(client_fd);
      ::close(client_fd);
    }
    ::close(server_fd);
    return;
  }

  int client_fd = connect_tcp_client(server_ip, port, std::chrono::seconds(30));
  write_barrier_byte(client_fd);
  read_barrier_byte(client_fd);
  ::close(client_fd);
}

int get_int_arg(int argc, char** argv, char const* key, int def) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == key && i + 1 < argc) {
      return std::stoi(argv[i + 1]);
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

CollectiveKind parse_collective(std::string const& value) {
  if (value == "allreduce") return CollectiveKind::AllReduce;
  if (value == "alltoall") return CollectiveKind::AllToAll;
  throw std::invalid_argument("unsupported collective: " + value);
}

Transport::PreferredTransport parse_transport(std::string const& value) {
  if (value == "auto") return Transport::PreferredTransport::Auto;
  if (value == "ipc") return Transport::PreferredTransport::Ipc;
  if (value == "rdma") return Transport::PreferredTransport::Rdma;
  if (value == "uccl") return Transport::PreferredTransport::Uccl;
  if (value == "tcp") return Transport::PreferredTransport::Tcp;
  throw std::invalid_argument("unsupported transport: " + value);
}

void gpu_malloc_bytes(void** ptr, size_t bytes) {
  GPU_RT_CHECK(gpuMalloc(ptr, bytes));
}

struct DeviceBuffer {
  void* ptr = nullptr;
  size_t bytes = 0;

  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t nbytes) : bytes(nbytes) {
    gpu_malloc_bytes(&ptr, bytes);
  }

  ~DeviceBuffer() {
    if (ptr != nullptr) {
      gpuFree(ptr);
    }
  }

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr(other.ptr), bytes(other.bytes) {
    other.ptr = nullptr;
    other.bytes = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this == &other) return *this;
    if (ptr != nullptr) {
      gpuFree(ptr);
    }
    ptr = other.ptr;
    bytes = other.bytes;
    other.ptr = nullptr;
    other.bytes = 0;
    return *this;
  }

  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;
};

CollectiveBinding build_collective_memory(int rank, int world_size,
                                          void* tensor_ptr, size_t tensor_bytes,
                                          void* staging_ptr,
                                          size_t staging_bytes) {
  CollectiveBinding binding;
  binding.registry = std::make_shared<BufferRegistry>();
  binding.registry->local_rank = rank;
  binding.roles = kTestRoles;
  RegisteredBuffer& tensor =
      binding.ensure_buffer(binding.buffer_id(CollectiveBufferRole::Input));
  tensor.local_ptr = tensor_ptr;
  tensor.bytes = tensor_bytes;
  tensor.layout.sizes = {static_cast<int64_t>(tensor_bytes)};
  tensor.layout.strides = {1};
  tensor.layout.dtype = ScalarType::Float32;
  tensor.peer_views.resize(static_cast<size_t>(world_size));
  RegisteredBuffer& staging =
      binding.ensure_buffer(binding.buffer_id(CollectiveBufferRole::Scratch));
  staging.local_ptr = staging_ptr;
  staging.local_mr_id = 0;
  staging.bytes = staging_bytes;
  staging.layout.sizes = {static_cast<int64_t>(staging_bytes)};
  staging.layout.strides = {1};
  staging.layout.dtype = ScalarType::Float32;
  staging.peer_views.resize(static_cast<size_t>(world_size));

  for (int peer = 0; peer < world_size; ++peer) {
    tensor.peer_views[static_cast<size_t>(peer)].same_node = true;
    staging.peer_views[static_cast<size_t>(peer)].same_node = true;
  }

  return binding;
}

void init_allreduce_input(std::vector<float>& host, int rank) {
  for (size_t i = 0; i < host.size(); ++i) {
    host[i] = static_cast<float>(rank * 1000) + static_cast<float>(i);
  }
}

void verify_allreduce_output(std::vector<float> const& host, int nranks) {
  for (size_t i = 0; i < host.size(); ++i) {
    float expected = 0.0f;
    for (int rank = 0; rank < nranks; ++rank) {
      expected += static_cast<float>(rank * 1000) + static_cast<float>(i);
    }
    require(std::fabs(host[i] - expected) < 1e-3f,
            "allreduce output mismatch at index " + std::to_string(i) +
                ", got=" + std::to_string(host[i]) +
                ", expected=" + std::to_string(expected) +
                ", window=" + preview_window(host, i));
  }
}

void init_alltoall_input(std::vector<float>& host, int rank, int nranks) {
  require(host.size() % static_cast<size_t>(nranks) == 0,
          "alltoall test requires evenly divisible element count");
  size_t slice_elems = host.size() / static_cast<size_t>(nranks);
  for (int dst_rank = 0; dst_rank < nranks; ++dst_rank) {
    for (size_t i = 0; i < slice_elems; ++i) {
      host[static_cast<size_t>(dst_rank) * slice_elems + i] =
          static_cast<float>(rank * 10000 + dst_rank * 100 +
                             static_cast<int>(i));
    }
  }
}

void verify_alltoall_output(std::vector<float> const& host, int rank,
                            int nranks) {
  require(host.size() % static_cast<size_t>(nranks) == 0,
          "alltoall test requires evenly divisible element count");
  size_t slice_elems = host.size() / static_cast<size_t>(nranks);
  for (int src_rank = 0; src_rank < nranks; ++src_rank) {
    for (size_t i = 0; i < slice_elems; ++i) {
      float expected = static_cast<float>(src_rank * 10000 + rank * 100 +
                                          static_cast<int>(i));
      float actual = host[static_cast<size_t>(src_rank) * slice_elems + i];
      require(std::fabs(actual - expected) < 1e-3f,
              "alltoall output mismatch at src rank " +
                  std::to_string(src_rank) + ", index " + std::to_string(i));
    }
  }
}

void upload_tensor(void* dst, std::vector<float> const& host) {
  GPU_RT_CHECK(gpuMemcpy(dst, host.data(), host.size() * sizeof(float),
                         gpuMemcpyHostToDevice));
  GPU_RT_CHECK(gpuDeviceSynchronize());
}

std::vector<float> download_tensor(void const* src, size_t bytes) {
  require(bytes % sizeof(float) == 0, "test expects float-sized tensors");
  std::vector<float> host(bytes / sizeof(float), 0.0f);
  GPU_RT_CHECK(gpuMemcpy(host.data(), src, bytes, gpuMemcpyDeviceToHost));
  GPU_RT_CHECK(gpuDeviceSynchronize());
  return host;
}

void wait_for_collective(Executor& executor, CollectiveOpHandle handle,
                         std::chrono::milliseconds timeout) {
  auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (executor.poll(handle)) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  require(executor.poll(handle), "collective timed out");
}

struct Options {
  int rank = 0;
  int world_size = 2;
  int gpu = 0;
  int exchanger_port = 6979;
  uint32_t num_flows = 2;
  size_t bytes_per_rank = default_test_bytes_per_rank(2);
  size_t tile_bytes = 64 << 10;
  std::string exchanger_ip = "127.0.0.1";
  std::string transport = "auto";
  CollectiveKind collective = CollectiveKind::AllReduce;
};

Options parse_options(int argc, char** argv) {
  Options opts;
  int env_rank = get_env_int({"RANK"}, 0);
  int env_world_size = get_env_int({"WORLD_SIZE"}, 2);
  int env_local_rank = get_env_int({"LOCAL_RANK"}, env_rank);

  opts.rank = get_int_arg(argc, argv, "--rank", env_rank);
  opts.world_size = get_int_arg(argc, argv, "--world-size", env_world_size);
  opts.gpu = get_int_arg(argc, argv, "--gpu", env_local_rank);
  opts.exchanger_port = get_int_arg(
      argc, argv, "--exchanger-port",
      get_env_int({"MASTER_PORT", "UHM_EXCHANGER_SERVER_PORT"}, 29500));
  opts.num_flows = static_cast<uint32_t>(get_int_arg(
      argc, argv, "--num-flows", get_env_int({"CCL_NUM_FLOWS"}, 2)));
  opts.bytes_per_rank =
      get_size_arg(argc, argv, "--bytes-per-rank",
                   get_env_size({"BYTES_PER_RANK", "CCL_BYTES_PER_RANK"},
                                default_test_bytes_per_rank(opts.world_size)));
  opts.tile_bytes =
      get_size_arg(argc, argv, "--tile-bytes",
                   get_env_size({"TILE_BYTES", "CCL_TILE_BYTES"}, 64 << 10));

  std::string default_ip =
      get_env_str({"MASTER_ADDR", "UHM_EXCHANGER_SERVER_IP"},
                  opts.rank == 0 ? "0.0.0.0" : "127.0.0.1");
  opts.exchanger_ip = get_str_arg(argc, argv, "--exchanger-ip", default_ip);
  opts.transport = get_str_arg(argc, argv, "--transport",
                               get_env_str({"TRANSPORT"}, "auto"));
  opts.collective = parse_collective(
      get_str_arg(argc, argv, "--collective",
                  get_env_str({"COLLECTIVE", "CCL_COLLECTIVE"}, "allreduce")));
  return opts;
}

int run_rank(Options const& opts) {
  require(opts.world_size >= 2, "world_size must be >= 2");
  require(opts.rank >= 0 && opts.rank < opts.world_size, "rank out of range");
  require(opts.bytes_per_rank > 0, "bytes_per_rank must be > 0");
  require(opts.tile_bytes > 0, "tile_bytes must be > 0");
  require(opts.bytes_per_rank % sizeof(float) == 0,
          "bytes_per_rank must be a multiple of sizeof(float)");
  if (opts.collective == CollectiveKind::AllToAll) {
    require(opts.bytes_per_rank %
                    (static_cast<size_t>(opts.world_size) * sizeof(float)) ==
                0,
            "alltoall test requires bytes_per_rank to be a multiple of "
            "world_size * sizeof(float)");
  }

  GPU_RT_CHECK(gpuSetDevice(opts.gpu));

  DeviceBuffer tensor(opts.bytes_per_rank);
  DeviceBuffer staging(opts.bytes_per_rank);
  GPU_RT_CHECK(gpuMemset(tensor.ptr, 0, opts.bytes_per_rank));
  GPU_RT_CHECK(gpuMemset(staging.ptr, 0, opts.bytes_per_rank));

  std::vector<float> input(opts.bytes_per_rank / sizeof(float), 0.0f);
  if (opts.collective == CollectiveKind::AllReduce) {
    init_allreduce_input(input, opts.rank);
  } else {
    init_alltoall_input(input, opts.rank, opts.world_size);
  }
  upload_tensor(tensor.ptr, input);

  auto memory = std::make_shared<CollectiveBinding>(build_collective_memory(
      opts.rank, opts.world_size, tensor.ptr, opts.bytes_per_rank, staging.ptr,
      opts.bytes_per_rank));

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
  executor_cfg.device_task_capacity = 4096;
  executor_cfg.threads_per_block = 256;
  executor_cfg.fifo_capacity = 64;
  std::fprintf(stderr, "[rank %d] executor init\n", opts.rank);
  Executor executor(executor_cfg);

  CollectiveConfig config =
      Testing::make_test_config(opts.world_size, opts.rank, opts.bytes_per_rank,
                                opts.tile_bytes, opts.num_flows);
  config.dtype = ScalarType::Float32;
  config.reduction = ReductionKind::Sum;
  if (opts.collective == CollectiveKind::AllToAll) {
    config.algorithm = AlgorithmKind::Pairwise;
  }
  std::string barrier_ip = get_env_str(
      {"MASTER_ADDR"}, opts.rank == 0 && opts.exchanger_ip == "0.0.0.0"
                           ? "127.0.0.1"
                           : opts.exchanger_ip);
  socket_barrier(barrier_ip, opts.exchanger_port + 1, opts.rank,
                 opts.world_size, "pre-submit");
  std::fprintf(stderr, "[rank %d] submit %s\n", opts.rank,
               collective_name(opts.collective));
  CollectiveOpHandle handle = (opts.collective == CollectiveKind::AllReduce)
                                  ? executor.submit_allreduce(config, memory)
                                  : executor.submit_alltoall(config, memory);
  wait_for_collective(executor, handle, std::chrono::seconds(60));
  if (executor.status(handle) != CollectiveOpStatus::Completed) {
    std::string error = executor.error_message(handle);
    if (error.empty()) {
      error = "unknown executor failure";
    }
    fail("collective did not complete successfully: " + error);
  }
  executor.release(handle);

  std::vector<float> output = download_tensor(tensor.ptr, opts.bytes_per_rank);
  if (opts.collective == CollectiveKind::AllReduce) {
    verify_allreduce_output(output, opts.world_size);
  } else {
    verify_alltoall_output(output, opts.rank, opts.world_size);
  }

  std::printf(
      "[rank %d] %s verified, bytes_per_rank=%zu, tile_bytes=%zu, "
      "num_flows=%u\n",
      opts.rank, collective_name(opts.collective), opts.bytes_per_rank,
      opts.tile_bytes, opts.num_flows);
  return 0;
}

}  // namespace

}  // namespace CCL
}  // namespace UKernel

int main(int argc, char** argv) {
  try {
    return UKernel::CCL::run_rank(UKernel::CCL::parse_options(argc, argv));
  } catch (std::exception const& e) {
    std::fprintf(stderr, "[ccl multiprocess] fatal: %s\n", e.what());
    return 2;
  }
}
