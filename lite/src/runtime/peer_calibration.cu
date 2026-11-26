#include "uccl_lite/peer_calibration.hpp"
#include "uccl_lite/peer_profile.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <set>
#include <stdexcept>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace uccl_lite {
namespace {
void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct DeviceGuard {
  int device;
  DeviceGuard() { check(cudaGetDevice(&device)); }
  ~DeviceGuard() { cudaSetDevice(device); }
};
__host__ __device__ float value(size_t index, int rank) {
  return float(rank + 1 + int(index % 4));
}
__global__ void initialize(__nv_bfloat16* input, size_t count, int rank) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x)
    input[i] = __float2bfloat16(value(i, rank));
}
__global__ void verify(unsigned short const* input,
                       unsigned short const* output, size_t inputElements,
                       size_t outputElements, size_t offset, size_t count,
                       int rank, int ranks, Collective op, int* errors) {
  size_t extent = max(inputElements, outputElements) + 256;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < extent;
       i += size_t(gridDim.x) * blockDim.x) {
    if (i < inputElements + 256) {
      unsigned short expected = 0x7f7f;
      if (i >= offset && i < offset + inputElements)
        expected =
            __bfloat16_as_ushort(__float2bfloat16(value(i - offset, rank)));
      if (input[i] != expected) atomicAdd(errors, 1);
    }
    if (i < outputElements + 256) {
      unsigned short expected = 0x7f7f;
      if (i >= offset && i < offset + outputElements) {
        size_t index = i - offset;
        float sum = 0;
        if (op == Collective::AllGather)
          sum = value(index % count, index / count);
        else
          for (int source = 0; source < ranks; ++source)
            sum += value(op == Collective::ReduceScatter
                             ? size_t(rank) * count + index
                             : index,
                         source);
        expected = __bfloat16_as_ushort(__float2bfloat16(sum));
      }
      if (output[i] != expected) atomicAdd(errors, 1);
    }
  }
}
struct Buffers {
  int device = -1;
  void *input = nullptr, *output = nullptr;
  int* errors = nullptr;
  cudaStream_t stream = nullptr;
  ~Buffers() {
    if (device < 0) return;
    cudaSetDevice(device);
    if (stream) cudaStreamSynchronize(stream);
    if (input) cudaFree(input);
    if (output) cudaFree(output);
    if (errors) cudaFree(errors);
    if (stream) cudaStreamDestroy(stream);
  }
};

}  // namespace
PeerQualification qualifyPeerPlans(
    Topology const& topology, Workload const& w,
    std::vector<PeerFamily> const& families,
    std::function<void(CollectiveCall const&, PeerFamily)> const& enqueue) {
  size_t ranks = topology.devices.size();
  if ((!ranks || ranks > kPeerRankCapacity) ||
      w.dtype != ScalarType::BFloat16 || w.inPlace || !w.bytes || w.bytes % 2 ||
      (w.collective != Collective::AllReduce &&
       w.collective != Collective::AllGather &&
       w.collective != Collective::ReduceScatter) ||
      (w.collective != Collective::AllReduce && w.bytes % (2 * ranks)) ||
      w.alignment < 2 || w.alignment > 256 ||
      (w.alignment & (w.alignment - 1)) || !enqueue || families.empty())
    throw std::invalid_argument("unsupported peer qualification workload");
  std::set<PeerFamily> unique;
  for (auto family : families)
    if (int(family) < 0 || int(family) > 2 || !unique.insert(family).second)
      throw std::invalid_argument("invalid peer qualification family list");
  if (!unique.count(PeerFamily::Direct))
    throw std::invalid_argument("qualification requires direct baseline");
  DeviceGuard restore;
  size_t count =
      w.bytes / 2 / (w.collective == Collective::AllReduce ? 1 : ranks);
  size_t inputs =
      count * (w.collective == Collective::ReduceScatter ? ranks : 1);
  size_t outputs = count * (w.collective == Collective::AllGather ? ranks : 1);
  if (std::max(inputs, outputs) > (SIZE_MAX - 512) / 2)
    throw std::invalid_argument("qualification extent overflow");
  // Offset yields exactly the requested alignment, even for scalar tails.
  size_t offset = w.alignment == 256 ? 256 : w.alignment;
  CollectiveCall call;
  call.collective = w.collective;
  call.dtype = w.dtype;
  call.count = count;
  std::vector<std::unique_ptr<Buffers>> storage;
  for (size_t rank = 0; rank < ranks; ++rank) {
    storage.push_back(std::make_unique<Buffers>());
    auto& b = *storage.back();
    b.device = topology.devices[rank];
    check(cudaSetDevice(b.device));
    check(cudaStreamCreateWithFlags(&b.stream, cudaStreamNonBlocking));
    check(cudaMalloc(&b.input, inputs * 2 + 512));
    check(cudaMalloc(&b.output, outputs * 2 + 512));
    check(cudaMalloc(&b.errors, sizeof(int)));
    check(cudaMemsetAsync(b.input, 0x7f, inputs * 2 + 512, b.stream));
    auto input = static_cast<char*>(b.input) + offset;
    auto output = static_cast<char*>(b.output) + offset;
    initialize<<<256, 256, 0, b.stream>>>(
        reinterpret_cast<__nv_bfloat16*>(input), inputs, rank);
    check(cudaGetLastError());
    call.ports.push_back({input, output, b.stream, b.device});
  }
  auto synchronize = [&] {
    for (auto const& b : storage) {
      check(cudaSetDevice(b->device));
      check(cudaStreamSynchronize(b->stream));
    }
  };
  synchronize();
  PeerQualification result;
  for (auto family : families) result.trials.push_back({family, {}});
  // Rotate order across repetitions rather than always warming one candidate
  // behind another. Report the complete range; no best-repetition selection.
  for (size_t repetition = 0; repetition < 3; ++repetition)
    for (size_t slot = 0; slot < families.size(); ++slot) {
      auto& trial = result.trials[(slot + repetition) % families.size()];
      for (auto const& b : storage) {
        check(cudaSetDevice(b->device));
        check(cudaMemsetAsync(b->output, 0x7f, outputs * 2 + 512, b->stream));
      }
      synchronize();
      for (int warmup = 0; warmup < 2; ++warmup) enqueue(call, trial.family);
      synchronize();
      // Warmup output must not make a broken timed call look correct.
      for (auto const& b : storage) {
        check(cudaSetDevice(b->device));
        check(cudaMemsetAsync(b->output, 0x7f, outputs * 2 + 512, b->stream));
      }
      synchronize();
      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < 3; ++i) enqueue(call, trial.family);
      synchronize();
      double us = std::chrono::duration<double, std::micro>(
                      std::chrono::steady_clock::now() - start)
                      .count() /
                  3;
      if (!std::isfinite(us) || us <= 0)
        throw std::runtime_error("invalid peer qualification timing");
      trial.microseconds.push_back(us);
      for (size_t rank = 0; rank < ranks; ++rank) {
        auto& b = *storage[rank];
        check(cudaSetDevice(b.device));
        check(cudaMemsetAsync(b.errors, 0, sizeof(int), b.stream));
        verify<<<256, 256, 0, b.stream>>>(
            static_cast<unsigned short*>(b.input),
            static_cast<unsigned short*>(b.output), inputs, outputs, offset / 2,
            count, rank, ranks, w.collective, b.errors);
        check(cudaGetLastError());
        check(cudaStreamSynchronize(b.stream));
        int errors = 0;
        check(cudaMemcpy(&errors, b.errors, sizeof(errors),
                         cudaMemcpyDeviceToHost));
        if (errors)
          throw std::runtime_error(
              "peer calibration output/input/guard check failed");
      }
    }
  std::array<std::vector<double>, 3> samples;
  for (auto const& trial : result.trials)
    samples[size_t(trial.family)] = trial.microseconds;
  result.family = qualifiedPeerFamily(samples);
  return result;
}
}  // namespace uccl_lite
