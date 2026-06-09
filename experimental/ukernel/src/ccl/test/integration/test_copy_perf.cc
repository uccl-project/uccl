#include "../../include/gpu_rt.h"
#include "../backend/backend.h"
#include "../backend/device_backend.h"
#include "../backend/rdma_local_copy_backend.h"
#include "../coll_types.h"
#include "../lower.h"
#include "../utils.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace UKernel {
namespace CCL {
namespace {

using Clock = std::chrono::steady_clock;

void fail(std::string const& msg) { throw std::runtime_error(msg); }

struct DeviceBuffer {
  void* ptr = nullptr;
  size_t bytes = 0;
  explicit DeviceBuffer(size_t nbytes) : bytes(nbytes) {
    GPU_RT_CHECK(gpuMalloc(&ptr, bytes));
  }
  ~DeviceBuffer() {
    if (ptr != nullptr) gpuFree(ptr);
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
    if (ptr != nullptr) gpuFree(ptr);
    ptr = other.ptr;
    bytes = other.bytes;
    other.ptr = nullptr;
    other.bytes = 0;
    return *this;
  }
};

struct BenchSize {
  size_t bytes;
  int iters;
};

static constexpr BenchSize kBenchSizes[] = {
    {4, 100000},        {16, 50000},       {64, 50000},
    {256, 20000},       {1024, 10000},     {4096, 5000},
    {16384, 2000},      {65536, 500},      {262144, 200},
    {1048576, 100},     {4194304, 50},     {16777216, 20},
    {67108864, 5},      {134217728, 3},
};

static constexpr size_t kMaxBuf = 512ULL * 1024 * 1024;
static constexpr int kWarmupFactor = 10;

DeviceBuffer g_src_buf(0);
DeviceBuffer g_dst_buf(0);

double bench_cuda_memcpy_d2d(size_t bytes, int iters) {
  int warmup = std::max(1, iters / kWarmupFactor);
  for (int i = 0; i < warmup; ++i) {
    GPU_RT_CHECK(gpuMemcpy(g_dst_buf.ptr, g_src_buf.ptr, bytes,
                           gpuMemcpyDeviceToDevice));
  }
  GPU_RT_CHECK(gpuDeviceSynchronize());

  auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i) {
    GPU_RT_CHECK(gpuMemcpy(g_dst_buf.ptr, g_src_buf.ptr, bytes,
                           gpuMemcpyDeviceToDevice));
  }
  GPU_RT_CHECK(gpuDeviceSynchronize());
  auto t1 = Clock::now();

  return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

double bench_device_backend_copy(size_t bytes, int iters) {
  TiledResult tiled;
  tiled.input_bytes = bytes;
  tiled.output_bytes = bytes;
  tiled.rank = 0;
  tiled.nranks = 1;
  Op op;
  op.kind = OpKind::Copy;
  op.bytes = bytes;
  op.src_off = 0;
  op.dst_off = 0;
  tiled.ops.push_back(op);
  tiled.chunk_of.push_back(0);
  tiled.layers = {{0}};

  DeviceBackend backend;
  backend.validate(tiled, g_src_buf.ptr, g_dst_buf.ptr, nullptr);

  int warmup = std::max(1, iters / kWarmupFactor);
  for (int i = 0; i < warmup; ++i) {
    OpBindings bind;
    bind.stream_index = 0;
    bind.resolved_src = g_src_buf.ptr;
    bind.resolved_dst = g_dst_buf.ptr;
    BackendToken tok;
    while (true) {
      tok = backend.submit(tiled.ops[0], bind, g_src_buf.ptr, g_dst_buf.ptr,
                           nullptr);
      if (tok.value != 0) break;
      BackendToken tmp;
      backend.drain(&tmp, 1);
    }
    BackendToken out;
    while (backend.drain(&out, 1) != 1 || out.value != tok.value) {
    }
  }
  backend.stop(0);

  backend.validate(tiled, g_src_buf.ptr, g_dst_buf.ptr, nullptr);

  auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i) {
    OpBindings bind;
    bind.stream_index = 0;
    bind.resolved_src = g_src_buf.ptr;
    bind.resolved_dst = g_dst_buf.ptr;
    BackendToken tok;
    while (true) {
      tok = backend.submit(tiled.ops[0], bind, g_src_buf.ptr, g_dst_buf.ptr,
                           nullptr);
      if (tok.value != 0) break;
      BackendToken tmp;
      backend.drain(&tmp, 1);
    }
    BackendToken out;
    while (backend.drain(&out, 1) != 1 || out.value != tok.value) {
    }
  }
  auto t1 = Clock::now();

  backend.stop(0);
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

double bench_rdma_local_copy(size_t bytes, int iters) {
  RdmaLocalCopyBackendConfig cfg;
  cfg.gpu_id = 0;
  RdmaLocalCopyBackend backend(cfg);
  if (strcmp(backend.name(), "degraded") == 0) return -1.0;

  TiledResult tiled;
  tiled.input_bytes = bytes;
  tiled.output_bytes = bytes;
  tiled.rank = 0;
  tiled.nranks = 1;
  Op op;
  op.kind = OpKind::Copy;
  op.bytes = bytes;
  op.src_off = 0;
  op.dst_off = 0;
  tiled.ops.push_back(op);
  tiled.chunk_of.push_back(0);
  tiled.layers = {{0}};

  backend.validate(tiled, g_src_buf.ptr, g_dst_buf.ptr, nullptr);
  if (strcmp(backend.name(), "degraded") == 0) return -1.0;

  int warmup = std::max(1, iters / kWarmupFactor);
  for (int i = 0; i < warmup; ++i) {
    OpBindings bind;
    bind.stream_index = 0;
    bind.resolved_src = g_src_buf.ptr;
    bind.resolved_dst = g_dst_buf.ptr;
    BackendToken tok;
    while (true) {
      tok = backend.submit(tiled.ops[0], bind, g_src_buf.ptr, g_dst_buf.ptr,
                           nullptr);
      if (tok.value != 0) break;
      if (strcmp(backend.name(), "degraded") == 0) return -1.0;
      BackendToken tmp;
      backend.drain(&tmp, 1);
    }
    BackendToken out;
    while (backend.drain(&out, 1) != 1 || out.value != tok.value) {
    }
  }

  backend.validate(tiled, g_src_buf.ptr, g_dst_buf.ptr, nullptr);
  if (strcmp(backend.name(), "degraded") == 0) return -1.0;

  auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i) {
    OpBindings bind;
    bind.stream_index = 0;
    bind.resolved_src = g_src_buf.ptr;
    bind.resolved_dst = g_dst_buf.ptr;
    BackendToken tok;
    while (true) {
      tok = backend.submit(tiled.ops[0], bind, g_src_buf.ptr, g_dst_buf.ptr,
                           nullptr);
      if (tok.value != 0) break;
      if (strcmp(backend.name(), "degraded") == 0) return -1.0;
      BackendToken tmp;
      backend.drain(&tmp, 1);
    }
    BackendToken out;
    while (backend.drain(&out, 1) != 1 || out.value != tok.value) {
    }
  }
  auto t1 = Clock::now();

  return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

const char* fmt_size(size_t bytes) {
  static char buf[32];
  if (bytes < 1024)
    snprintf(buf, sizeof(buf), "%6zu B", bytes);
  else if (bytes < 1024 * 1024)
    snprintf(buf, sizeof(buf), " %5.1f KB", bytes / 1024.0);
  else if (bytes < 1024ULL * 1024 * 1024)
    snprintf(buf, sizeof(buf), " %5.1f MB", bytes / (1024.0 * 1024.0));
  else
    snprintf(buf, sizeof(buf), " %5.1f GB",
             bytes / (1024.0 * 1024.0 * 1024.0));
  return buf;
}

void print_results() {
  std::printf("\n=== G2G Copy Performance Benchmark ===\n\n");
  std::printf("%-9s | %8s  %7s | %8s  %7s | %8s  %7s\n",
              "Size", "cudaMemcpy", "", "DeviceBackend", "", "RdmaLocal", "");
  std::printf("%-9s | %8s  %7s | %8s  %7s | %8s  %7s\n",
              "", "lat(us)", "GB/s", "lat(us)", "GB/s", "lat(us)", "GB/s");
  std::printf("----------|--------------------|---------------------|---------------------\n");

  g_src_buf = DeviceBuffer(kMaxBuf);
  g_dst_buf = DeviceBuffer(kMaxBuf);

  for (auto& bs : kBenchSizes) {
    double cm_us = bench_cuda_memcpy_d2d(bs.bytes, bs.iters);
    double cm_gbs = (bs.bytes * 1e-9) / (cm_us * 1e-6);
    double dev_us = bench_device_backend_copy(bs.bytes, bs.iters);
    double dev_gbs = (bs.bytes * 1e-9) / (dev_us * 1e-6);
    double rdma_us = bench_rdma_local_copy(bs.bytes, bs.iters);

    std::printf("%-9s | %8.2f  %7.2f | %8.2f  %7.2f | ",
                fmt_size(bs.bytes), cm_us, cm_gbs, dev_us, dev_gbs);
    if (rdma_us < 0)
      std::printf("          degraded\n");
    else {
      double rdma_gbs = (bs.bytes * 1e-9) / (rdma_us * 1e-6);
      std::printf("%8.2f  %7.2f\n", rdma_us, rdma_gbs);
    }
  }
}

}  // namespace
}  // namespace CCL
}  // namespace UKernel

int main() {
  try {
    GPU_RT_CHECK(gpuSetDevice(0));
    UKernel::CCL::print_results();
    std::printf("\n=== Copy perf benchmark DONE ===\n");
    return 0;
  } catch (std::exception const& ex) {
    std::fprintf(stderr, "[copy perf] fatal: %s\n", ex.what());
    return 2;
  }
}
