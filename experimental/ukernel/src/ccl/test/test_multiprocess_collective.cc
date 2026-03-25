#include "../../include/gpu_rt.h"
#include "../backend/transport_backend.h"
#include "backend_test_utils.h"
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
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "../../include/transport.h"

namespace UKernel {
namespace CCL {

namespace {

size_t ceil_div(size_t a, size_t b) { return b == 0 ? 0 : (a + b - 1) / b; }

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

  int client_fd =
      connect_tcp_client(server_ip, port, std::chrono::seconds(30));
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
  if (value == "uccl") return Transport::PreferredTransport::Uccl;
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

class EmulatedDeviceBackend final : public Backend {
 public:
  explicit EmulatedDeviceBackend(CollectiveMemory memory)
      : memory_(std::move(memory)) {}

  char const* name() const override { return "emulated-device"; }

  void validate(ExecutionPlan const& plan) const override {
    if (plan.staging_bytes_required != 0 &&
        memory_.staging.local_ptr == nullptr) {
      throw std::invalid_argument("emulated device staging buffer is missing");
    }
    if (plan.staging_bytes_required > memory_.staging.bytes) {
      throw std::invalid_argument(
          "emulated device staging capacity is insufficient");
    }
  }

  bool supports(ExecOpKind kind) const override {
    return kind == ExecOpKind::DeviceCopy || kind == ExecOpKind::DeviceReduce;
  }

  BackendToken submit(ExecOp const& op) override {
    if (!supports(op.kind)) {
      throw std::invalid_argument(
          "emulated device backend does not support this op");
    }
    if (op.kind == ExecOpKind::DeviceCopy) {
      apply_copy(op);
    } else {
      apply_sum_reduce(op);
    }
    BackendToken token{next_token_++};
    pending_.push_back(token.value);
    return token;
  }

  bool poll(BackendToken token) override { return is_completed(token.value); }

  bool try_pop_completed(BackendToken& token) override {
    if (pending_.empty()) return false;
    token.value = pending_.front();
    pending_.erase(pending_.begin());
    completed_.push_back(token.value);
    return true;
  }

  void release(BackendToken token) override {
    pending_.erase(std::remove(pending_.begin(), pending_.end(), token.value),
                   pending_.end());
    completed_.erase(
        std::remove(completed_.begin(), completed_.end(), token.value),
        completed_.end());
  }

 private:
  static void validate_span(char const* what, size_t offset, size_t bytes,
                            size_t capacity) {
    if (offset > capacity || bytes > capacity - offset) {
      throw std::invalid_argument(std::string(what) + " out of range");
    }
  }

  void* resolve_mutable(BufferRef const& ref, size_t bytes) const {
    switch (ref.kind) {
      case BufferKind::Staging:
        require(memory_.staging.local_ptr != nullptr, "missing staging");
        validate_span("staging", ref.offset_bytes, bytes,
                      memory_.staging.bytes);
        return static_cast<char*>(memory_.staging.local_ptr) + ref.offset_bytes;
      case BufferKind::Tensor:
        require(memory_.tensor.local_ptr != nullptr, "missing local tensor");
        validate_span("local tensor", ref.offset_bytes, bytes,
                      memory_.tensor.bytes);
        return static_cast<char*>(memory_.tensor.local_ptr) + ref.offset_bytes;
      case BufferKind::PeerTensor:
        break;
    }
    throw std::invalid_argument("unknown mutable ref");
  }

  void const* resolve_const(BufferRef const& ref, size_t bytes) const {
    switch (ref.kind) {
      case BufferKind::Staging:
        require(memory_.staging.local_ptr != nullptr, "missing staging");
        validate_span("staging", ref.offset_bytes, bytes,
                      memory_.staging.bytes);
        return static_cast<char const*>(memory_.staging.local_ptr) +
               ref.offset_bytes;
      case BufferKind::Tensor:
        require(memory_.tensor.local_ptr != nullptr, "missing local tensor");
        validate_span("local tensor", ref.offset_bytes, bytes,
                      memory_.tensor.bytes);
        return static_cast<char const*>(memory_.tensor.local_ptr) +
               ref.offset_bytes;
      case BufferKind::PeerTensor:
        break;
    }
    throw std::invalid_argument("unknown const ref");
  }

  bool is_completed(uint64_t token) const {
    for (uint64_t value : completed_) {
      if (value == token) return true;
    }
    for (uint64_t value : pending_) {
      if (value == token) return true;
    }
    return false;
  }

  void apply_copy(ExecOp const& op) const {
    void const* src = resolve_const(op.src, op.tile.size_bytes);
    void* dst = resolve_mutable(op.dst, op.tile.size_bytes);
    GPU_RT_CHECK(
        gpuMemcpy(dst, src, op.tile.size_bytes, gpuMemcpyDeviceToDevice));
    GPU_RT_CHECK(gpuDeviceSynchronize());
  }

  void apply_sum_reduce(ExecOp const& op) const {
    require(op.tile.size_bytes % sizeof(float) == 0,
            "allreduce test expects float-sized reductions");
    void const* src_dev = resolve_const(op.src, op.tile.size_bytes);
    void* dst_dev = resolve_mutable(op.dst, op.tile.size_bytes);

    size_t elems = op.tile.size_bytes / sizeof(float);
    std::vector<float> src(elems, 0.0f);
    std::vector<float> dst(elems, 0.0f);
    GPU_RT_CHECK(gpuMemcpy(src.data(), src_dev, op.tile.size_bytes,
                           gpuMemcpyDeviceToHost));
    GPU_RT_CHECK(gpuMemcpy(dst.data(), dst_dev, op.tile.size_bytes,
                           gpuMemcpyDeviceToHost));
    for (size_t i = 0; i < elems; ++i) {
      dst[i] += src[i];
    }
    GPU_RT_CHECK(gpuMemcpy(dst_dev, dst.data(), op.tile.size_bytes,
                           gpuMemcpyHostToDevice));
    GPU_RT_CHECK(gpuDeviceSynchronize());
  }

  CollectiveMemory memory_{};
  uint64_t next_token_ = 1;
  std::vector<uint64_t> pending_;
  std::vector<uint64_t> completed_;
};

CollectiveMemory build_collective_memory(Transport::Communicator& comm,
                                         int rank, int world_size,
                                         void* tensor_ptr, size_t tensor_bytes,
                                         void* staging_ptr,
                                         size_t staging_bytes) {
  CollectiveMemory memory;
  memory.tensor.local_rank = rank;
  memory.tensor.local_ptr = tensor_ptr;
  memory.tensor.bytes = tensor_bytes;
  memory.tensor.layout.sizes = {static_cast<int64_t>(tensor_bytes)};
  memory.tensor.layout.strides = {1};
  memory.tensor.peer_views.resize(static_cast<size_t>(world_size));
  memory.staging.local_ptr = staging_ptr;
  memory.staging.bytes = staging_bytes;
  memory.staging.layout.sizes = {static_cast<int64_t>(staging_bytes)};
  memory.staging.layout.strides = {1};

  Transport::MR local_mr = comm.reg_mr(tensor_ptr, tensor_bytes);
  memory.tensor.local_mr_id = local_mr.id;

  for (int peer = 0; peer < world_size; ++peer) {
    auto& view = memory.tensor.peer_views[static_cast<size_t>(peer)];
    view.same_node = true;
    view.peer_accessible = false;
  }

  for (int peer = 0; peer < world_size; ++peer) {
    if (peer == rank) continue;
    require(comm.notify_mr(peer, local_mr),
            "notify_mr failed for peer " + std::to_string(peer));
  }

  for (int peer = 0; peer < world_size; ++peer) {
    if (peer == rank) continue;
    Transport::MR remote_mr{};
    require(comm.wait_mr_notify(peer, remote_mr),
            "wait_mr_notify failed for peer " + std::to_string(peer));
    memory.tensor.peer_views[static_cast<size_t>(peer)].mr_id = remote_mr.id;
  }

  return memory;
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
            "allreduce output mismatch at index " + std::to_string(i));
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
  size_t bytes_per_rank = 1 << 20;
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
  opts.bytes_per_rank = get_size_arg(
      argc, argv, "--bytes-per-rank",
      get_env_size({"BYTES_PER_RANK", "CCL_BYTES_PER_RANK"}, 1 << 20));
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

  GPU_RT_CHECK(gpuSetDevice(opts.gpu));

  auto cfg = std::make_shared<Transport::CommunicatorConfig>();
  cfg->exchanger_ip = opts.exchanger_ip;
  cfg->exchanger_port = opts.exchanger_port;
  cfg->local_id = opts.rank;
  cfg->preferred_transport = parse_transport(opts.transport);

  std::fprintf(stderr, "[rank %d] communicator init\n", opts.rank);
  Transport::Communicator comm(opts.gpu, opts.rank, opts.world_size, cfg);

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

  std::fprintf(stderr, "[rank %d] exchange MRs\n", opts.rank);
  CollectiveMemory memory = build_collective_memory(
      comm, opts.rank, opts.world_size, tensor.ptr, opts.bytes_per_rank,
      staging.ptr, opts.bytes_per_rank);
  std::fprintf(stderr, "[rank %d] MR exchange ready\n", opts.rank);

  CommunicatorTransportBackend transport_backend(comm, memory);
  EmulatedDeviceBackend device_backend(memory);

  ExecutorBackends backends{};
  backends.transport = &transport_backend;
  backends.device = &device_backend;
  Executor executor(backends);

  CollectiveConfig config =
      Testing::make_test_config(opts.world_size, opts.rank, opts.bytes_per_rank,
                                opts.tile_bytes, opts.num_flows);
  if (opts.collective == CollectiveKind::AllToAll) {
    config.algorithm = AlgorithmKind::Pairwise;
  }
  std::string barrier_ip =
      get_env_str({"MASTER_ADDR"},
                  opts.rank == 0 && opts.exchanger_ip == "0.0.0.0"
                      ? "127.0.0.1"
                      : opts.exchanger_ip);
  socket_barrier(barrier_ip, opts.exchanger_port + 1, opts.rank,
                 opts.world_size, "pre-submit");
  std::fprintf(stderr, "[rank %d] submit %s\n", opts.rank,
               collective_name(opts.collective));
  CollectiveOpHandle handle = (opts.collective == CollectiveKind::AllReduce)
                                  ? executor.submit_allreduce(config)
                                  : executor.submit_alltoall(config);
  wait_for_collective(executor, handle, std::chrono::seconds(60));

  require(executor.status(handle) == CollectiveOpStatus::Completed,
          "collective did not complete successfully");
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
