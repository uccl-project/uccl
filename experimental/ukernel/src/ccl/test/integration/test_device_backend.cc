#include "backend/device_backend.h"
#include "gpu_rt.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace UKernel {
namespace CCL {

namespace {

constexpr BufferId kTestInputBufferId = 7;
constexpr BufferId kTestScratchBufferId = 11;
constexpr CollectiveBufferRoles kTestRoles{
    kTestInputBufferId, kTestInputBufferId, kTestScratchBufferId};

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
};

void upload_floats(void* dst, std::vector<float> const& values) {
  GPU_RT_CHECK(gpuMemcpy(dst, values.data(), values.size() * sizeof(float),
                         gpuMemcpyHostToDevice));
  GPU_RT_CHECK(gpuDeviceSynchronize());
}

std::vector<float> download_floats(void const* src, size_t bytes) {
  require(bytes % sizeof(float) == 0, "float buffer size must align");
  std::vector<float> out(bytes / sizeof(float), 0.0f);
  GPU_RT_CHECK(gpuMemcpy(out.data(), src, bytes, gpuMemcpyDeviceToHost));
  GPU_RT_CHECK(gpuDeviceSynchronize());
  return out;
}

void zero_buffer(void* dst, size_t bytes) {
  GPU_RT_CHECK(gpuMemset(dst, 0, bytes));
}

std::string preview(std::vector<float> const& values, size_t count = 4) {
  std::string out = "[";
  size_t const limit = std::min(count, values.size());
  for (size_t i = 0; i < limit; ++i) {
    if (i != 0) out += ", ";
    out += std::to_string(values[i]);
  }
  if (values.size() > limit) out += ", ...";
  out += "]";
  return out;
}

void drain_all(DeviceBackend& backend, std::vector<BackendToken> const& tokens,
               std::chrono::milliseconds timeout) {
  std::vector<bool> seen(tokens.size(), false);
  size_t remaining = tokens.size();
  auto deadline = std::chrono::steady_clock::now() + timeout;
  BackendToken done_buf[64];
  while (remaining > 0 && std::chrono::steady_clock::now() < deadline) {
    size_t n = backend.drain(done_buf, 64);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < tokens.size(); ++j) {
        if (!seen[j] && done_buf[i].value == tokens[j].value) {
          seen[j] = true;
          --remaining;
          break;
        }
      }
    }
    if (remaining == 0) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  require(remaining == 0, "device backend token timed out");
}

void submit_and_drain(DeviceBackend& backend, Op const& op,
                      CollectiveBinding& binding,
                      std::chrono::milliseconds timeout) {
  OpBindings bind;
  bind.stream_index = 0;
  std::vector<BackendToken> tokens = {backend.submit(op, bind, binding)};
  drain_all(backend, tokens, timeout);
}

Op make_device_op(OpKind kind, size_t offset_bytes, size_t size_bytes) {
  Op op;
  op.kind = kind;
  op.bytes = size_bytes;
  op.src_off = offset_bytes;
  op.dst_off = offset_bytes;
  return op;
}

void verify_copy(std::vector<float> const& actual,
                 std::vector<float> const& expected, char const* label) {
  for (size_t i = 0; i < expected.size(); ++i) {
    require(std::fabs(actual[i] - expected[i]) < 1e-6f,
            std::string(label) + " mismatch at index " + std::to_string(i) +
                ", got=" + std::to_string(actual[i]) +
                ", expected=" + std::to_string(expected[i]));
  }
}

void verify_sum_reduce(std::vector<float> const& out,
                       std::vector<float> const& dst_init,
                       std::vector<float> const& src, char const* label,
                       std::vector<float> const* staging_after = nullptr) {
  for (size_t i = 0; i < dst_init.size(); ++i) {
    float expected = dst_init[i] + src[i];
    std::string msg = std::string(label) + " mismatch at index " +
                      std::to_string(i) + ", got=" + std::to_string(out[i]) +
                      ", expected=" + std::to_string(expected);
    if (staging_after != nullptr) {
      msg += ", dst_init=" + preview(dst_init) + ", src=" + preview(src) +
             ", staging_after=" + preview(*staging_after) +
             ", out=" + preview(out);
    }
    require(std::fabs(out[i] - expected) < 1e-5f, msg);
  }
}

CollectiveBinding make_memory(int rank, void* tensor_ptr, size_t tensor_bytes,
                              void* staging_ptr, size_t staging_bytes) {
  static auto s_registry = std::make_shared<BufferRegistry>();
  s_registry->local_rank = rank;
  CollectiveBinding binding;
  binding.registry = s_registry.get();
  binding.roles = kTestRoles;
  RegisteredBuffer& tensor =
      binding.ensure_buffer(binding.buffer_id(CollectiveBufferRole::Input));
  tensor.local_ptr = tensor_ptr;
  tensor.bytes = tensor_bytes;
  tensor.layout.dtype = ScalarType::Float32;
  tensor.peer_views.resize(1);
  RegisteredBuffer& staging =
      binding.ensure_buffer(binding.buffer_id(CollectiveBufferRole::Scratch));
  staging.local_ptr = staging_ptr;
  staging.bytes = staging_bytes;
  staging.layout.dtype = ScalarType::Float32;
  staging.peer_views.resize(1);
  return binding;
}

void validate_backend_for_op(DeviceBackend& backend, Op const& op,
                             CollectiveBinding& binding) {
  CollectivePlan plan;
  plan.nranks = 1;
  plan.rank = 0;
  plan.tile_bytes = op.bytes;
  plan.ops.push_back(op);
  backend.validate(plan, binding);
}

void test_device_copy() {
  std::printf("[test] device backend copy...\n");
  constexpr size_t kElems = 257;
  constexpr size_t kBytes = kElems * sizeof(float);

  DeviceBuffer tensor(kBytes);
  DeviceBuffer staging(kBytes);
  std::vector<float> src(kElems, 0.0f);
  for (size_t i = 0; i < src.size(); ++i) {
    src[i] = static_cast<float>(i * 3 + 1);
  }
  upload_floats(tensor.ptr, src);
  zero_buffer(staging.ptr, kBytes);

  auto memory = make_memory(0, tensor.ptr, kBytes, staging.ptr, kBytes);
  DeviceBackend backend;

  Op op = make_device_op(OpKind::DeviceCopy, 0, 0, kBytes, kTestInputBufferId,
                         kTestScratchBufferId);
  validate_backend_for_op(backend, op, memory);
  submit_and_drain(backend, op, memory, std::chrono::seconds(5));
  backend.stop(0);

  verify_copy(download_floats(staging.ptr, kBytes), src, "device copy");
}

void test_device_reduce_sum() {
  std::printf("[test] device backend reduce...\n");
  constexpr size_t kElems = 513;
  constexpr size_t kBytes = kElems * sizeof(float);

  DeviceBuffer tensor(kBytes);
  DeviceBuffer staging(kBytes);
  std::vector<float> dst_init(kElems, 0.0f);
  std::vector<float> src(kElems, 0.0f);
  for (size_t i = 0; i < kElems; ++i) {
    dst_init[i] = static_cast<float>(1000 + i);
    src[i] = static_cast<float>(i + 1);
  }
  upload_floats(tensor.ptr, dst_init);
  upload_floats(staging.ptr, src);

  auto memory = make_memory(0, tensor.ptr, kBytes, staging.ptr, kBytes);
  DeviceBackend backend;

  Op op =
      make_device_op(OpKind::DeviceReduce, 0, 0, kBytes, kTestScratchBufferId,
                     kTestInputBufferId, ReductionKind::Sum);
  validate_backend_for_op(backend, op, memory);
  submit_and_drain(backend, op, memory, std::chrono::seconds(5));
  backend.stop(0);
  std::vector<float> out = download_floats(tensor.ptr, kBytes);
  std::vector<float> staging_after = download_floats(staging.ptr, kBytes);
  verify_sum_reduce(out, dst_init, src, "device reduce", &staging_after);
}

void test_device_reduce_pipeline_same_flow() {
  std::printf("[test] device backend reduce pipeline on one stream...\n");
  constexpr size_t kTileElems = (64 << 10) / sizeof(float);
  constexpr size_t kTiles = 4;
  constexpr size_t kElems = kTileElems * kTiles;
  constexpr size_t kBytes = kElems * sizeof(float);

  DeviceBuffer tensor(kBytes);
  DeviceBuffer staging(kBytes);
  std::vector<float> dst_init(kElems, 0.0f);
  std::vector<float> src(kElems, 0.0f);
  for (size_t i = 0; i < kElems; ++i) {
    dst_init[i] = static_cast<float>(2000 + i);
    src[i] = static_cast<float>(1000 + 2 * i);
  }
  upload_floats(tensor.ptr, dst_init);
  upload_floats(staging.ptr, src);

  auto memory = make_memory(0, tensor.ptr, kBytes, staging.ptr, kBytes);
  DeviceBackend backend;

  std::vector<BackendToken> tokens;
  tokens.reserve(kTiles);
  bool validated = false;
  for (size_t tile = 0; tile < kTiles; ++tile) {
    size_t offset = tile * kTileElems * sizeof(float);
    Op op =
        make_device_op(static_cast<uint32_t>(tile), OpKind::DeviceReduce, 0,
                       offset, kTileElems * sizeof(float), kTestScratchBufferId,
                       kTestInputBufferId, ReductionKind::Sum);
    if (!validated) {
      validate_backend_for_op(backend, op, memory);
      validated = true;
    }
    tokens.push_back(backend.submit(op, memory));
  }
  drain_all(backend, tokens, std::chrono::seconds(5));
  backend.stop(0);

  verify_sum_reduce(download_floats(tensor.ptr, kBytes), dst_init, src,
                    "device reduce pipeline");
}

}  // namespace

}  // namespace CCL
}  // namespace UKernel

int main() {
  try {
    GPU_RT_CHECK(gpuSetDevice(0));
    std::printf("=== Device Backend Tests ===\n\n");
    UKernel::CCL::test_device_copy();
    UKernel::CCL::test_device_reduce_sum();
    UKernel::CCL::test_device_reduce_pipeline_same_flow();
    std::printf("\n=== Device backend tests PASSED ===\n");
    return 0;
  } catch (std::exception const& ex) {
    std::fprintf(stderr, "[device backend test] fatal: %s\n", ex.what());
    return 2;
  }
}
