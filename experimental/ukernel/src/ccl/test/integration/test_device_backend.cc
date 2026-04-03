#include "backend/device_backend.h"
#include "gpu_rt.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace UKernel {
namespace CCL {

namespace {

constexpr BufferId kTestInputBufferId = 7;
constexpr BufferId kTestScratchBufferId = 11;
constexpr CollectiveBufferRoles kTestRoles{kTestInputBufferId,
                                           kTestInputBufferId,
                                           kTestScratchBufferId};

[[noreturn]] void fail(std::string const& msg) { throw std::runtime_error(msg); }

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

void wait_for_token(DeviceBackend& backend, BackendToken token,
                    std::chrono::milliseconds timeout) {
  auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (backend.poll(token)) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  require(backend.poll(token), "device backend token timed out");
}

BackendToken submit_and_wait(DeviceBackend& backend, ExecOp const& op,
                             CollectiveBinding& binding,
                             std::chrono::milliseconds timeout) {
  BackendToken token = backend.submit(op, binding);
  wait_for_token(backend, token, timeout);
  return token;
}

ExecOp make_device_op(uint32_t op_id, ExecOpKind kind, uint32_t flow_index,
                      size_t offset_bytes, size_t size_bytes, BufferId src,
                      BufferId dst,
                      ReductionKind reduction = ReductionKind::None) {
  ExecOp op;
  op.op_id = op_id;
  op.kind = kind;
  op.tile.flow_index = flow_index;
  op.tile.offset_bytes = offset_bytes;
  op.tile.size_bytes = size_bytes;
  op.src.kind = BufferKind::Local;
  op.src.buffer_id = src;
  op.src.offset_bytes = offset_bytes;
  op.dst.kind = BufferKind::Local;
  op.dst.buffer_id = dst;
  op.dst.offset_bytes = offset_bytes;
  op.dtype = ScalarType::Float32;
  op.reduction = reduction;
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

std::shared_ptr<CollectiveBinding> make_memory(int rank, void* tensor_ptr,
                                               size_t tensor_bytes,
                                               void* staging_ptr,
                                               size_t staging_bytes) {
  auto binding = std::make_shared<CollectiveBinding>();
  binding->registry = std::make_shared<BufferRegistry>();
  binding->registry->local_rank = rank;
  binding->roles = kTestRoles;
  RegisteredBuffer& tensor =
      binding->ensure_buffer(binding->buffer_id(CollectiveBufferRole::Input));
  tensor.local_ptr = tensor_ptr;
  tensor.bytes = tensor_bytes;
  tensor.layout.sizes = {static_cast<int64_t>(tensor_bytes)};
  tensor.layout.strides = {1};
  tensor.layout.dtype = ScalarType::Float32;
  RegisteredBuffer& staging =
      binding->ensure_buffer(binding->buffer_id(CollectiveBufferRole::Scratch));
  staging.local_ptr = staging_ptr;
  staging.bytes = staging_bytes;
  staging.layout.sizes = {static_cast<int64_t>(staging_bytes)};
  staging.layout.strides = {1};
  staging.layout.dtype = ScalarType::Float32;
  return binding;
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

  ExecOp op = make_device_op(0, ExecOpKind::DeviceCopy, 0, 0, kBytes,
                             kTestInputBufferId, kTestScratchBufferId);
  BackendToken token = submit_and_wait(backend, op, *memory, std::chrono::seconds(5));
  backend.release(token);
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

  ExecOp op = make_device_op(0, ExecOpKind::DeviceReduce, 0, 0, kBytes,
                             kTestScratchBufferId, kTestInputBufferId,
                             ReductionKind::Sum);
  BackendToken token = submit_and_wait(backend, op, *memory, std::chrono::seconds(5));
  backend.release(token);
  backend.stop(0);
  std::vector<float> out = download_floats(tensor.ptr, kBytes);
  std::vector<float> staging_after = download_floats(staging.ptr, kBytes);
  verify_sum_reduce(out, dst_init, src, "device reduce", &staging_after);
}

void test_device_reduce_pipeline_same_flow() {
  std::printf("[test] device backend reduce pipeline on one flow...\n");
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
  for (size_t tile = 0; tile < kTiles; ++tile) {
    size_t offset = tile * kTileElems * sizeof(float);
    ExecOp op = make_device_op(static_cast<uint32_t>(tile),
                               ExecOpKind::DeviceReduce, 0, offset,
                               kTileElems * sizeof(float),
                               kTestScratchBufferId, kTestInputBufferId,
                               ReductionKind::Sum);
    tokens.push_back(backend.submit(op, *memory));
  }

  for (BackendToken token : tokens) {
    wait_for_token(backend, token, std::chrono::seconds(5));
    backend.release(token);
  }
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
