#include "../backend/device_backend.h"
#include "../../include/gpu_rt.h"
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

std::shared_ptr<CollectiveMemory> make_memory(int rank, void* tensor_ptr,
                                              size_t tensor_bytes,
                                              void* staging_ptr,
                                              size_t staging_bytes) {
  auto memory = std::make_shared<CollectiveMemory>();
  memory->tensor.local_rank = rank;
  memory->tensor.local_ptr = tensor_ptr;
  memory->tensor.bytes = tensor_bytes;
  memory->tensor.layout.sizes = {static_cast<int64_t>(tensor_bytes)};
  memory->tensor.layout.strides = {1};
  memory->tensor.layout.dtype = ScalarType::Float32;
  memory->staging.local_ptr = staging_ptr;
  memory->staging.bytes = staging_bytes;
  memory->staging.layout.sizes = {static_cast<int64_t>(staging_bytes)};
  memory->staging.layout.strides = {1};
  memory->staging.layout.dtype = ScalarType::Float32;
  return memory;
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
  GPU_RT_CHECK(gpuMemset(staging.ptr, 0, kBytes));

  auto memory = make_memory(0, tensor.ptr, kBytes, staging.ptr, kBytes);
  DeviceBackend backend(memory);

  ExecOp op;
  op.op_id = 0;
  op.kind = ExecOpKind::DeviceCopy;
  op.tile.flow_index = 0;
  op.tile.offset_bytes = 0;
  op.tile.size_bytes = kBytes;
  op.src.kind = BufferKind::Tensor;
  op.dst.kind = BufferKind::Staging;
  op.dtype = ScalarType::Float32;

  BackendToken token = backend.submit(op);
  wait_for_token(backend, token, std::chrono::seconds(5));
  backend.release(token);
  backend.stop(0);

  std::vector<float> dst = download_floats(staging.ptr, kBytes);
  for (size_t i = 0; i < src.size(); ++i) {
    require(std::fabs(dst[i] - src[i]) < 1e-6f,
            "device copy mismatch at index " + std::to_string(i) +
                ", got=" + std::to_string(dst[i]) +
                ", expected=" + std::to_string(src[i]));
  }
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
  DeviceBackend backend(memory);

  ExecOp op;
  op.op_id = 0;
  op.kind = ExecOpKind::DeviceReduce;
  op.tile.flow_index = 0;
  op.tile.offset_bytes = 0;
  op.tile.size_bytes = kBytes;
  op.src.kind = BufferKind::Staging;
  op.dst.kind = BufferKind::Tensor;
  op.dtype = ScalarType::Float32;
  op.reduction = ReductionKind::Sum;

  BackendToken token = backend.submit(op);
  wait_for_token(backend, token, std::chrono::seconds(5));
  backend.release(token);
  backend.stop(0);
  std::vector<float> out = download_floats(tensor.ptr, kBytes);
  std::vector<float> staging_after = download_floats(staging.ptr, kBytes);
  for (size_t i = 0; i < kElems; ++i) {
    float expected = dst_init[i] + src[i];
    require(std::fabs(out[i] - expected) < 1e-5f,
            "device reduce mismatch at index " + std::to_string(i) +
                ", got=" + std::to_string(out[i]) +
                ", expected=" + std::to_string(expected) +
                ", dst_init=" + preview(dst_init) + ", src=" + preview(src) +
                ", staging_after=" + preview(staging_after) +
                ", out=" + preview(out));
  }
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
  DeviceBackend backend(memory);

  std::vector<BackendToken> tokens;
  tokens.reserve(kTiles);
  for (size_t tile = 0; tile < kTiles; ++tile) {
    ExecOp op;
    op.op_id = static_cast<uint32_t>(tile);
    op.kind = ExecOpKind::DeviceReduce;
    op.tile.flow_index = 0;
    op.tile.offset_bytes = tile * kTileElems * sizeof(float);
    op.tile.size_bytes = kTileElems * sizeof(float);
    op.src.kind = BufferKind::Staging;
    op.src.offset_bytes = op.tile.offset_bytes;
    op.dst.kind = BufferKind::Tensor;
    op.dst.offset_bytes = op.tile.offset_bytes;
    op.dtype = ScalarType::Float32;
    op.reduction = ReductionKind::Sum;
    tokens.push_back(backend.submit(op));
  }

  for (BackendToken token : tokens) {
    wait_for_token(backend, token, std::chrono::seconds(5));
    backend.release(token);
  }
  backend.stop(0);

  std::vector<float> out = download_floats(tensor.ptr, kBytes);
  for (size_t i = 0; i < kElems; ++i) {
    float expected = dst_init[i] + src[i];
    require(std::fabs(out[i] - expected) < 1e-5f,
            "device reduce pipeline mismatch at index " + std::to_string(i) +
                ", got=" + std::to_string(out[i]) +
                ", expected=" + std::to_string(expected));
  }
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
