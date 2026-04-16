#include "backend/transport_backend.h"
#include "config.h"
#include "gpu_rt.h"
#include "transport.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

[[noreturn]] void fail(std::string const& msg) {
  throw std::runtime_error(msg);
}

void require(bool cond, std::string const& msg) {
  if (!cond) fail(msg);
}

struct DeviceBuffer {
  void* ptr = nullptr;
  size_t bytes = 0;

  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t nbytes) : bytes(nbytes) {
    GPU_RT_CHECK(gpuMalloc(&ptr, bytes));
  }
  ~DeviceBuffer() {
    if (ptr != nullptr) gpuFree(ptr);
  }
  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;
};

int get_env_int(std::initializer_list<char const*> keys, int fallback) {
  for (char const* key : keys) {
    char const* value = std::getenv(key);
    if (value != nullptr && value[0] != '\0') {
      return std::atoi(value);
    }
  }
  return fallback;
}

std::string get_env_str(std::initializer_list<char const*> keys,
                        std::string fallback) {
  for (char const* key : keys) {
    char const* value = std::getenv(key);
    if (value != nullptr && value[0] != '\0') {
      return std::string(value);
    }
  }
  return fallback;
}

int get_int_arg(int argc, char** argv, char const* name, int fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) {
      return std::atoi(argv[i + 1]);
    }
  }
  return fallback;
}

std::string get_str_arg(int argc, char** argv, char const* name,
                        std::string fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) {
      return std::string(argv[i + 1]);
    }
  }
  return fallback;
}

int create_tcp_server(int port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  require(fd >= 0, "failed to create barrier socket");
  int opt = 1;
  require(::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) == 0,
          "failed to set SO_REUSEADDR");
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(static_cast<uint16_t>(port));
  require(::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0,
          "failed to bind barrier socket");
  require(::listen(fd, 16) == 0, "failed to listen on barrier socket");
  return fd;
}

int connect_tcp_client(std::string const& ip, int port,
                       std::chrono::milliseconds timeout) {
  auto deadline = std::chrono::steady_clock::now() + timeout;
  while (true) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    require(fd >= 0, "failed to create client socket");
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    require(::inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) == 1,
            "invalid barrier ip");
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
      return fd;
    }
    ::close(fd);
    if (std::chrono::steady_clock::now() >= deadline) {
      fail("failed to connect barrier client");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void write_exact(int fd, void const* buf, size_t bytes) {
  char const* p = static_cast<char const*>(buf);
  while (bytes != 0) {
    ssize_t n = ::write(fd, p, bytes);
    require(n > 0, "barrier write failed");
    p += n;
    bytes -= static_cast<size_t>(n);
  }
}

void read_exact(int fd, void* buf, size_t bytes) {
  char* p = static_cast<char*>(buf);
  while (bytes != 0) {
    ssize_t n = ::read(fd, p, bytes);
    require(n > 0, "barrier read failed");
    p += n;
    bytes -= static_cast<size_t>(n);
  }
}

void socket_barrier(std::string const& ip, int port, int rank, int nranks) {
  constexpr uint64_t kToken = 0xC011BACCULL;
  if (rank == 0) {
    int server = create_tcp_server(port);
    std::vector<int> clients;
    clients.reserve(static_cast<size_t>(nranks - 1));
    for (int i = 1; i < nranks; ++i) {
      int fd = ::accept(server, nullptr, nullptr);
      require(fd >= 0, "barrier accept failed");
      uint64_t token = 0;
      read_exact(fd, &token, sizeof(token));
      require(token == kToken, "barrier token mismatch");
      clients.push_back(fd);
    }
    for (int fd : clients) {
      write_exact(fd, &kToken, sizeof(kToken));
      ::close(fd);
    }
    ::close(server);
  } else {
    int fd = connect_tcp_client(ip, port, std::chrono::seconds(5));
    write_exact(fd, &kToken, sizeof(kToken));
    uint64_t token = 0;
    read_exact(fd, &token, sizeof(token));
    require(token == kToken, "barrier ack mismatch");
    ::close(fd);
  }
}

void upload_floats(void* dst, std::vector<float> const& values) {
  GPU_RT_CHECK(gpuMemcpy(dst, values.data(), values.size() * sizeof(float),
                         gpuMemcpyHostToDevice));
  GPU_RT_CHECK(gpuDeviceSynchronize());
}

std::vector<float> download_floats(void const* src, size_t bytes) {
  require(bytes % sizeof(float) == 0, "transport test requires float bytes");
  std::vector<float> out(bytes / sizeof(float), 0.0f);
  GPU_RT_CHECK(gpuMemcpy(out.data(), src, bytes, gpuMemcpyDeviceToHost));
  GPU_RT_CHECK(gpuDeviceSynchronize());
  return out;
}

void wait_for_token(CommunicatorTransportBackend& backend, BackendToken token,
                    std::chrono::milliseconds timeout) {
  auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (backend.poll(token)) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  require(backend.poll(token), "transport backend token timed out");
}

ExecutionPlan make_single_op_plan(int rank, int nranks, ExecOp const& op) {
  ExecutionPlan plan;
  plan.rank = rank;
  plan.nranks = nranks;
  plan.num_flows = 1;
  plan.tensor_bytes = op.tile.size_bytes;
  plan.tile_bytes = op.tile.size_bytes;
  plan.ops.push_back(op);
  return plan;
}

void verify_values(std::vector<float> const& actual,
                   std::vector<float> const& expected,
                   std::string const& what) {
  require(actual.size() == expected.size(), what + " size mismatch");
  for (size_t i = 0; i < actual.size(); ++i) {
    if (std::fabs(actual[i] - expected[i]) > 1e-6f) {
      fail(what + " mismatch at index " + std::to_string(i));
    }
  }
}

struct Options {
  int rank = 0;
  int world_size = 2;
  int gpu = 0;
  int exchanger_port = 29700;
  size_t bytes = 4096;
  std::string exchanger_ip = "127.0.0.1";
  std::string transport = "auto";
};

Options parse_options(int argc, char** argv) {
  Options opts;
  int env_rank = get_env_int({"RANK"}, 0);
  int env_world = get_env_int({"WORLD_SIZE"}, 2);
  int env_local_rank = get_env_int({"LOCAL_RANK"}, env_rank);
  opts.rank = get_int_arg(argc, argv, "--rank", env_rank);
  opts.world_size = get_int_arg(argc, argv, "--world-size", env_world);
  opts.gpu = get_int_arg(argc, argv, "--gpu", env_local_rank);
  opts.exchanger_port = get_int_arg(argc, argv, "--exchanger-port",
                                    get_env_int({"MASTER_PORT"}, 29700));
  opts.bytes = static_cast<size_t>(
      get_int_arg(argc, argv, "--bytes", static_cast<int>(opts.bytes)));
  opts.exchanger_ip = get_str_arg(argc, argv, "--exchanger-ip",
                                  get_env_str({"MASTER_ADDR"}, "127.0.0.1"));
  opts.transport = get_str_arg(argc, argv, "--transport", "auto");
  return opts;
}

Transport::PreferredTransport parse_transport(std::string const& value) {
  if (value == "auto") return Transport::PreferredTransport::Auto;
  if (value == "ipc") return Transport::PreferredTransport::Ipc;
  if (value == "rdma") return Transport::PreferredTransport::Rdma;
  if (value == "uccl") return Transport::PreferredTransport::Uccl;
  if (value == "tcp") return Transport::PreferredTransport::Tcp;
  fail("unknown transport: " + value);
  return Transport::PreferredTransport::Auto;
}

int run_rank(Options const& opts) {
  require(opts.world_size == 2, "transport backend test requires world_size=2");
  require(opts.rank == 0 || opts.rank == 1,
          "transport backend test requires ranks 0/1");
  require(opts.bytes % sizeof(float) == 0,
          "transport backend bytes must align to float");

  GPU_RT_CHECK(gpuSetDevice(opts.gpu));

  DeviceBuffer tensor(opts.bytes);
  DeviceBuffer staging(opts.bytes);
  GPU_RT_CHECK(gpuMemset(tensor.ptr, 0, opts.bytes));
  GPU_RT_CHECK(gpuMemset(staging.ptr, 0, opts.bytes));

  auto memory = std::make_shared<CollectiveBinding>();
  memory->registry = std::make_shared<BufferRegistry>();
  memory->registry->local_rank = opts.rank;
  memory->roles.input_buffer_id = 7;
  memory->roles.scratch_buffer_id = 11;
  memory->roles.validate();
  BufferId input_id = memory->buffer_id(CollectiveBufferRole::Input);
  BufferId scratch_id = memory->buffer_id(CollectiveBufferRole::Scratch);
  RegisteredBuffer& tensor_buf = memory->ensure_buffer(input_id);
  tensor_buf.local_ptr = tensor.ptr;
  tensor_buf.bytes = opts.bytes;
  tensor_buf.layout.dtype = ScalarType::Float32;
  tensor_buf.peer_views.resize(static_cast<size_t>(opts.world_size));
  RegisteredBuffer& staging_buf = memory->ensure_buffer(scratch_id);
  staging_buf.local_ptr = staging.ptr;
  staging_buf.bytes = opts.bytes;
  staging_buf.layout.dtype = ScalarType::Float32;
  staging_buf.peer_views.resize(static_cast<size_t>(opts.world_size));

  TransportBackendConfig cfg{};
  cfg.gpu_id = opts.gpu;
  cfg.rank = opts.rank;
  cfg.world_size = opts.world_size;
  cfg.communicator_config = std::make_shared<Transport::CommunicatorConfig>();
  cfg.communicator_config->exchanger_ip = opts.exchanger_ip;
  cfg.communicator_config->exchanger_port = opts.exchanger_port;
  cfg.communicator_config->local_id = opts.rank;
  cfg.communicator_config->preferred_transport =
      parse_transport(opts.transport);

  CommunicatorTransportBackend backend(cfg);

  std::vector<float> tensor_values(opts.bytes / sizeof(float), 0.0f);
  std::vector<float> staging_values(opts.bytes / sizeof(float), 0.0f);

  for (size_t i = 0; i < tensor_values.size(); ++i) {
    tensor_values[i] =
        static_cast<float>(1000 * opts.rank + static_cast<int>(i));
    staging_values[i] =
        static_cast<float>(2000 * opts.rank + static_cast<int>(i) * 2);
  }

  // Case 1: tensor -> peer staging
  if (opts.rank == 0) {
    upload_floats(tensor.ptr, tensor_values);
    GPU_RT_CHECK(gpuMemset(staging.ptr, 0, opts.bytes));
  } else {
    GPU_RT_CHECK(gpuMemset(tensor.ptr, 0, opts.bytes));
    GPU_RT_CHECK(gpuMemset(staging.ptr, 0, opts.bytes));
  }
  socket_barrier(opts.exchanger_ip, opts.exchanger_port + 1, opts.rank,
                 opts.world_size);

  ExecOp op1;
  op1.op_id = 0;
  op1.kind =
      (opts.rank == 0) ? ExecOpKind::TransportSend : ExecOpKind::TransportRecv;
  op1.tile.flow_index = 0;
  op1.tile.size_bytes = opts.bytes;
  op1.src = (opts.rank == 0) ? local_buffer_ref(input_id, 0)
                             : remote_buffer_ref(scratch_id, 0, 0);
  op1.dst = (opts.rank == 0) ? remote_buffer_ref(scratch_id, 1, 0)
                             : local_buffer_ref(scratch_id, 0);
  backend.validate(make_single_op_plan(opts.rank, opts.world_size, op1),
                   *memory);
  BackendToken token1 = backend.submit(op1, *memory);
  wait_for_token(backend, token1, std::chrono::seconds(10));
  backend.release(token1);
  socket_barrier(opts.exchanger_ip, opts.exchanger_port + 2, opts.rank,
                 opts.world_size);

  // Rank 1 should receive rank 0 tensor payload into local staging.
  if (opts.rank == 1) {
    std::vector<float> expected(opts.bytes / sizeof(float), 0.0f);
    for (size_t i = 0; i < expected.size(); ++i) {
      expected[i] = static_cast<float>(static_cast<int>(i));
    }
    verify_values(download_floats(staging.ptr, opts.bytes), expected,
                  "transport tensor->peer staging");
  }

  // Case 2: staging -> peer tensor
  if (opts.rank == 1) {
    upload_floats(staging.ptr, staging_values);
    GPU_RT_CHECK(gpuMemset(tensor.ptr, 0, opts.bytes));
  } else {
    GPU_RT_CHECK(gpuMemset(tensor.ptr, 0, opts.bytes));
    GPU_RT_CHECK(gpuMemset(staging.ptr, 0, opts.bytes));
  }
  socket_barrier(opts.exchanger_ip, opts.exchanger_port + 3, opts.rank,
                 opts.world_size);

  ExecOp op2;
  op2.op_id = 0;
  op2.kind =
      (opts.rank == 1) ? ExecOpKind::TransportSend : ExecOpKind::TransportRecv;
  op2.tile.flow_index = 0;
  op2.tile.size_bytes = opts.bytes;
  op2.src = (opts.rank == 1) ? local_buffer_ref(scratch_id, 0)
                             : remote_buffer_ref(input_id, 1, 0);
  op2.dst = (opts.rank == 1) ? remote_buffer_ref(input_id, 0, 0)
                             : local_buffer_ref(input_id, 0);
  backend.validate(make_single_op_plan(opts.rank, opts.world_size, op2),
                   *memory);
  BackendToken token2 = backend.submit(op2, *memory);
  wait_for_token(backend, token2, std::chrono::seconds(10));
  backend.release(token2);
  socket_barrier(opts.exchanger_ip, opts.exchanger_port + 4, opts.rank,
                 opts.world_size);

  if (opts.rank == 0) {
    std::vector<float> expected(opts.bytes / sizeof(float), 0.0f);
    for (size_t i = 0; i < expected.size(); ++i) {
      expected[i] = static_cast<float>(2000 + static_cast<int>(i) * 2);
    }
    verify_values(download_floats(tensor.ptr, opts.bytes), expected,
                  "transport staging->peer tensor");
  }

  std::printf("[rank %d] transport backend verified\n", opts.rank);
  return 0;
}

}  // namespace

}  // namespace CCL
}  // namespace UKernel

int main(int argc, char** argv) {
  try {
    return UKernel::CCL::run_rank(UKernel::CCL::parse_options(argc, argv));
  } catch (std::exception const& ex) {
    std::fprintf(stderr, "[transport backend test] fatal: %s\n", ex.what());
    return 2;
  }
}
