#include "uccl_lite/hardware_calibration.hpp"
#include "uccl_lite/runtime/build_identity.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <elf.h>
#include <link.h>

namespace uccl_lite {
namespace {
void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
__global__ void copyVectors(uint4 const* input, uint4* output, size_t count) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(blockDim.x) * gridDim.x)
    output[i] = input[i];
}
__global__ void verify(uint4 const* output, size_t count, unsigned expected,
                       int* errors) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(blockDim.x) * gridDim.x) {
    auto x = output[i];
    if (x.x != expected || x.y != expected || x.z != expected ||
        x.w != expected)
      atomicAdd(errors, 1);
  }
}
struct Buffers {
  int device = -1;
  uint4 *input = nullptr, *output = nullptr;
  int* errors = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr, stop = nullptr;
  ~Buffers() {
    if (device < 0) return;
    cudaSetDevice(device);
    if (stream) cudaStreamSynchronize(stream);
    if (input) cudaFree(input);
    if (output) cudaFree(output);
    if (errors) cudaFree(errors);
    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
    if (stream) cudaStreamDestroy(stream);
  }
};
struct DeviceGuard {
  int device;
  DeviceGuard() { check(cudaGetDevice(&device)); }
  ~DeviceGuard() { cudaSetDevice(device); }
};
struct PeerAccess {
  std::vector<std::pair<int, int>> owned;
  void enable(int device, int peer) {
    check(cudaSetDevice(device));
    auto status = cudaDeviceEnablePeerAccess(peer, 0);
    if (status == cudaErrorPeerAccessAlreadyEnabled)
      cudaGetLastError();
    else {
      check(status);
      owned.emplace_back(device, peer);
    }
  }
  ~PeerAccess() {
    for (auto const& pair : owned) {
      cudaSetDevice(pair.first);
      cudaDeviceDisablePeerAccess(pair.second);
    }
  }
};
double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  size_t n = values.size();
  return n % 2 ? values[n / 2] : (values[n / 2] + values[n / 2 - 1]) / 2;
}
}  // namespace
HardwareProfile probeGpuResources(Topology const& t,
                                  GpuProbeOptions const& options) {
  if (t.identity.empty() || t.devices.empty() ||
      t.gpuNodes.size() != t.devices.size() || options.bytes < 1024 ||
      options.bytes % 16 || options.warmups < 0 || options.iterations < 1 ||
      options.repetitions < 3)
    throw std::invalid_argument("invalid GPU calibration configuration");
  DeviceGuard restore;
  PeerAccess peers;
  std::vector<std::unique_ptr<Buffers>> buffers;
  std::map<size_t, size_t> rankForNode;
  for (size_t r = 0; r < t.devices.size(); ++r) {
    if (t.gpuNodes[r] >= t.hardware.size() ||
        !rankForNode.emplace(t.gpuNodes[r], r).second)
      throw std::invalid_argument("invalid calibration GPU mapping");
    buffers.push_back(std::make_unique<Buffers>());
    auto& b = *buffers.back();
    b.device = t.devices[r];
    check(cudaSetDevice(b.device));
    // This is an explicit collective idle boundary, outside user graph capture.
    check(cudaDeviceSynchronize());
    check(cudaStreamCreateWithFlags(&b.stream, cudaStreamNonBlocking));
    check(cudaEventCreate(&b.start));
    check(cudaEventCreate(&b.stop));
    check(cudaMalloc(&b.input, options.bytes));
    check(cudaMalloc(&b.output, options.bytes));
    check(cudaMalloc(&b.errors, sizeof(int)));
    check(cudaMemsetAsync(b.input, 0x40 + r, options.bytes, b.stream));
    check(cudaStreamSynchronize(b.stream));
  }
  HardwareProfile profile;
  profile.calibration.topologyIdentity =
      t.hardwareIdentity.empty() ? t.identity : t.hardwareIdentity;
  std::ostringstream identity;
  identity << "uccl-lite-gpu-resource-probe-v1;build="
           << runtime::artifactIdentity(
                  reinterpret_cast<uintptr_t>(&probeGpuResources))
           << ";bytes=" << options.bytes << ";warmups=" << options.warmups
           << ";iterations=" << options.iterations
           << ";repetitions=" << options.repetitions
           << ";scope=gpu-events-per-call;isolated-directions;dtype=uint4";
  profile.calibration.identity = identity.str();
  auto measure = [&](std::string const& resource, size_t source, size_t dest,
                     bool ce, bool memory, bool write = false) {
    auto& from = *buffers[source];
    auto& to = *buffers[dest];
    auto& executor = (ce || write) ? from : to;
    auto sample = [&](size_t bytes) {
      check(cudaSetDevice(to.device));
      check(cudaMemsetAsync(to.output, 0, bytes, to.stream));
      check(cudaStreamSynchronize(to.stream));
      check(cudaSetDevice(executor.device));
      auto launch = [&] {
        if (ce)
          check(cudaMemcpyPeerAsync(to.output, to.device, from.input,
                                    from.device, bytes, executor.stream));
        else {
          unsigned blocks =
              unsigned(std::min(size_t(4096), (bytes / 16 + 255) / 256));
          copyVectors<<<blocks, 256, 0, executor.stream>>>(
              from.input, to.output, bytes / 16);
          check(cudaGetLastError());
        }
      };
      ProbeObservation observation{
          resource,
          memory  ? "gpu-vector-copy"
          : ce    ? "cuda-peer-copy"
          : write ? "gpu-peer-write"
                  : "gpu-peer-read",
          "GPU events, mean per call; allocation/warmup/verification excluded; "
          "isolated direction",
          bytes,
          options.warmups,
          options.iterations,
          {}};
      for (int r = 0; r < options.repetitions; ++r) {
        for (int i = 0; i < options.warmups; ++i) launch();
        check(cudaEventRecord(executor.start, executor.stream));
        for (int i = 0; i < options.iterations; ++i) launch();
        check(cudaEventRecord(executor.stop, executor.stream));
        check(cudaEventSynchronize(executor.stop));
        float ms = 0;
        check(cudaEventElapsedTime(&ms, executor.start, executor.stop));
        observation.microseconds.push_back(ms * 1000.0 / options.iterations);
      }
      check(cudaSetDevice(to.device));
      check(cudaMemsetAsync(to.errors, 0, sizeof(int), to.stream));
      verify<<<256, 256, 0, to.stream>>>(to.output, bytes / 16,
                                         unsigned(0x40 + source) * 0x01010101U,
                                         to.errors);
      check(cudaGetLastError());
      check(cudaStreamSynchronize(to.stream));
      int errors = 0;
      check(cudaMemcpy(&errors, to.errors, sizeof(errors),
                       cudaMemcpyDeviceToHost));
      if (errors)
        throw std::runtime_error("GPU calibration data check failed: " +
                                 resource);
      double us = median(observation.microseconds);
      profile.observations.push_back(std::move(observation));
      return us;
    };
    double small = sample(16), large = sample(options.bytes);
    if (!(large > small && small > 0))
      throw std::runtime_error("nonpositive calibration slope: " + resource);
    double factor = memory ? 2 : 1;
    double bandwidth =
        factor * double(options.bytes - 16) / (large - small) / 1000;
    double latency = std::max(0.0, small - factor * 16 / bandwidth / 1000);
    profile.calibration.resources.push_back(
        {resource,
         {bandwidth, latency, Evidence::Measured},
         options.repetitions});
  };
  for (size_t rank = 0; rank < buffers.size(); ++rank)
    measure("memory:" + t.hardware[t.gpuNodes[rank]].id, rank, rank, false,
            true);
  for (auto const& path : t.paths) {
    auto source = rankForNode.find(path.source),
         dest = rankForNode.find(path.destination);
    if (source == rankForNode.end() || dest == rankForNode.end()) continue;
    bool ce = path.kind == PathKind::PeerCopy;
    peers.enable(
        buffers[path.kind != PathKind::PeerRead ? source->second : dest->second]
            ->device,
        buffers[path.kind != PathKind::PeerRead ? dest->second : source->second]
            ->device);
    measure(path.resource, source->second, dest->second, ce, false,
            path.kind == PathKind::PeerWrite);
  }
  return profile;
}
}  // namespace uccl_lite
