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

CollKind parse_collective(std::string const& value) {
  if (value == "allreduce") return CollKind::AllReduceRing;
  if (value == "alltoall") return CollKind::AllToAllPairwise;
  throw std::invalid_argument("unsupported collective: " + value);
}

Transport::PreferredTransport parse_transport(std::string const& value) {
  if (value == "auto") return Transport::PreferredTransport::Auto;
  if (value == "ipc") return Transport::PreferredTransport::Ipc;
  if (value == "uccl") return Transport::PreferredTransport::Uccl;
  if (value == "tcp") return Transport::PreferredTransport::Tcp;
  if (value == "rdma") return Transport::PreferredTransport::Rdma;
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

struct TestMemory {
  void* input_ptr;
  void* output_ptr;
  void* scratch_ptr;
};

TestMemory build_test_memory(void* tensor_ptr, void* staging_ptr) {
  TestMemory mem;
  mem.input_ptr = tensor_ptr;
  mem.output_ptr = tensor_ptr;  // in-place
  mem.scratch_ptr = staging_ptr;
  return mem;
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
  size_t bytes_per_rank = default_test_bytes_per_rank(2);
  size_t tile_bytes = 64 << 10;
  std::string exchanger_ip = "127.0.0.1";
  std::string transport = "auto";
  CollKind kind = CollKind::AllReduceRing;
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
  8 = static_cast<uint32_t>(get_int_arg(
       argc, argv, "--num-streams", get_env_int({"CCL_NUM_STREAMS"}, 2)));
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
  opts.kind = parse_collective(
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
  if (opts.kind == CollKind::AllToAllPairwise) {
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
  if (opts.kind == CollKind::AllReduceRing) {
    init_allreduce_input(input, opts.rank);
  } else {
    init_alltoall_input(input, opts.rank, opts.world_size);
  }
  upload_tensor(tensor.ptr, input);

  TestMemory mem = build_test_memory(tensor.ptr, staging.ptr);

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
  executor_cfg.max_device_fifos = std::max<uint32_t>(8, 1);
  executor_cfg.device_task_capacity = 4096;
  executor_cfg.threads_per_block = 256;
  executor_cfg.fifo_capacity = 64;
  std::fprintf(stderr, "[rank %d] executor init\n", opts.rank);
  Executor executor(executor_cfg);

  CollectiveConfig config =
      Testing::make_test_config(opts.world_size, opts.rank, opts.bytes_per_rank,
                                opts.tile_bytes);
  config.dtype = ScalarType::Float32;
  config.reduction = ReductionKind::Sum;
  if (opts.kind == CollKind::AllToAllPairwise) {
    config.kind = CollKind::AllToAllPairwise;
  }
  std::string barrier_ip = get_env_str(
      {"MASTER_ADDR"}, opts.rank == 0 && opts.exchanger_ip == "0.0.0.0"
                           ? "127.0.0.1"
                           : opts.exchanger_ip);
  socket_barrier(barrier_ip, opts.exchanger_port + 1, opts.rank,
                 opts.world_size, "pre-submit");
  std::fprintf(stderr, "[rank %d] submit %s\n", opts.rank,
               collective_name(opts.kind));
  CollectiveOpHandle handle = (opts.kind == CollKind::AllReduceRing)
                                  ? executor.submit_allreduce(config, mem.input_ptr, mem.output_ptr, mem.scratch_ptr)
                                  : executor.submit_alltoall(config, mem.input_ptr, mem.output_ptr, mem.scratch_ptr);
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
  if (opts.kind == CollKind::AllReduceRing) {
    verify_allreduce_output(output, opts.world_size);
  } else {
    verify_alltoall_output(output, opts.rank, opts.world_size);
  }

  std::printf(
      "[rank %d] %s verified, bytes_per_rank=%zu, tile_bytes=%zu\n",
      opts.rank, collective_name(opts.kind), opts.bytes_per_rank,
      opts.tile_bytes);
  std::printf(
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
