#include "uccl_lite/local_collectives.hpp"
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
using uccl_lite::Collective;
struct Device {
  int id;
  __nv_bfloat16 *input = nullptr, *output = nullptr;
  cudaStream_t stream = nullptr;
  Device(int device, size_t bytes) : id(device) {
    check(cudaSetDevice(id));
    check(cudaMalloc(&input, bytes));
    check(cudaMalloc(&output, bytes));
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
  Device(Device const&) = delete;
  Device& operator=(Device const&) = delete;
  Device(Device&& other) noexcept
      : id(other.id),
        input(std::exchange(other.input, nullptr)),
        output(std::exchange(other.output, nullptr)),
        stream(std::exchange(other.stream, nullptr)) {}
  ~Device() {
    cudaSetDevice(id);
    if (stream) cudaStreamDestroy(stream);
    if (output) cudaFree(output);
    if (input) cudaFree(input);
  }
};
float sourceValue(Collective op, int rank, size_t index, int generation) {
  if (op == Collective::AllGather)
    return float((rank + 1) * (1 + index % 7) + generation);
  return float((rank + 1) * (int(index % 5) - 2) + generation);
}
float expectedValue(Collective op, int rank, size_t index, size_t contribution,
                    int generation) {
  if (op == Collective::AllGather)
    return sourceValue(op, int(index / contribution), index % contribution,
                       generation);
  size_t whole = op == Collective::ReduceScatter
                     ? size_t(rank) * contribution + index
                     : index;
  return float(36 * (int(whole % 5) - 2) + 8 * generation);
}
}  // namespace

int main() {
  try {
    constexpr size_t maxBytes = (4 << 20) + 128;
    std::vector<int> ids{0, 1, 2, 3, 4, 5, 6, 7};
    auto topology = uccl_lite::inspectTopology(ids);
    uccl_lite::LocalCollectives local(topology, maxBytes);
    std::vector<Device> devices;
    devices.reserve(8);
    for (int id : ids) devices.emplace_back(id, maxBytes);
    uccl_lite::BackendPolicy policy;
    policy.mode = uccl_lite::RouterMode::Off;
    policy.fallbackIdentity = uccl_lite::LocalCollectives::identity();
    uccl_lite::CollectiveDispatcher dispatch(
        topology, policy,
        [&](uccl_lite::CollectiveCall const& call) { local.enqueue(call); });
    int cases = 0;
    for (size_t bytes : {size_t(128), size_t(8192), size_t((1 << 20) + 128),
                         size_t((2 << 20) + 128), maxBytes})
      for (auto op : {Collective::AllReduce, Collective::AllGather,
                      Collective::ReduceScatter})
        for (bool inPlace : {false, true})
          for (int generation = 0; generation < 2; ++generation) {
            size_t contribution =
                bytes / 2 / (op == Collective::AllReduce ? 1 : ids.size());
            size_t inputCount = op == Collective::ReduceScatter
                                    ? contribution * ids.size()
                                    : contribution;
            size_t outputCount = op == Collective::AllGather
                                     ? contribution * ids.size()
                                     : contribution;
            uccl_lite::CollectiveCall call;
            call.collective = op;
            call.count = contribution;
            call.inPlace = inPlace;
            for (int rank = 0; rank < 8; ++rank) {
              auto& device = devices[rank];
              check(cudaSetDevice(device.id));
              check(cudaMemset(device.output, 0x7f, maxBytes));
              auto* input = inPlace ? device.output : device.input;
              if (inPlace && op == Collective::AllGather)
                input += rank * contribution;
              std::vector<__nv_bfloat16> host(inputCount);
              for (size_t i = 0; i < inputCount; ++i)
                host[i] =
                    __float2bfloat16(sourceValue(op, rank, i, generation));
              check(cudaMemcpy(input, host.data(), inputCount * 2,
                               cudaMemcpyHostToDevice));
              auto* output = device.output;
              if (inPlace && op == Collective::ReduceScatter)
                output += rank * contribution;
              call.ports.push_back({input, output, device.stream, device.id});
            }
            local.prepare();
            auto decision = dispatch.prepare(call);
            if (decision.cpuRouter || dispatch.routerInitialized())
              throw std::runtime_error("unexpected router initialization");
            dispatch.enqueue(call);
            for (int rank = 0; rank < 8; ++rank) {
              auto& device = devices[rank];
              check(cudaSetDevice(device.id));
              std::vector<__nv_bfloat16> result(outputCount);
              check(cudaMemcpy(result.data(), call.ports[rank].output,
                               outputCount * 2, cudaMemcpyDeviceToHost));
              for (size_t i = 0; i < outputCount; ++i) {
                float actual = __bfloat162float(result[i]);
                float expected =
                    expectedValue(op, rank, i, contribution, generation);
                if (actual != expected)
                  throw std::runtime_error(
                      "local collective mismatch: " + std::to_string(cases) +
                      " rank " + std::to_string(rank) + " index " +
                      std::to_string(i));
              }
            }
            ++cases;
          }
    uccl_lite::CollectiveCall rounded;
    rounded.collective = Collective::AllReduce;
    rounded.count = 64;
    std::vector<std::vector<__nv_bfloat16>> values(
        8, std::vector<__nv_bfloat16>(rounded.count));
    for (int rank = 0; rank < 8; ++rank) {
      auto& device = devices[rank];
      for (size_t i = 0; i < rounded.count; ++i) {
        float value = float((rank + 1) * (1 + i % 23)) / 1000.0f;
        if (i == 0 && rank == 0)
          value = std::numeric_limits<float>::quiet_NaN();
        if (i == 1 && rank == 0) value = std::numeric_limits<float>::infinity();
        if (i == 1 && rank == 1)
          value = -std::numeric_limits<float>::infinity();
        values[rank][i] = __float2bfloat16(value);
      }
      check(cudaSetDevice(device.id));
      check(cudaMemcpy(device.input, values[rank].data(), rounded.count * 2,
                       cudaMemcpyHostToDevice));
      rounded.ports.push_back(
          {device.input, device.output, device.stream, device.id});
    }
    dispatch.enqueue(rounded);
    for (int rank = 0; rank < 8; ++rank) {
      check(cudaSetDevice(devices[rank].id));
      std::vector<__nv_bfloat16> actual(rounded.count);
      check(cudaMemcpy(actual.data(), devices[rank].output, rounded.count * 2,
                       cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < rounded.count; ++i) {
        float sum = __bfloat162float(values[0][i]);
        for (int source = 1; source < 8; ++source)
          sum += __bfloat162float(values[source][i]);
        float got = __bfloat162float(actual[i]);
        float expected = __bfloat162float(__float2bfloat16_rn(sum));
        if ((std::isnan(expected) && !std::isnan(got)) ||
            (!std::isnan(expected) && got != expected))
          throw std::runtime_error("rounded local AllReduce mismatch");
      }
    }
    ++cases;
    uccl_lite::CollectiveCall captured;
    captured.collective = Collective::AllReduce;
    captured.count = 64;
    for (auto& device : devices)
      captured.ports.push_back(
          {device.input, device.output, device.stream, device.id});
    check(cudaSetDevice(devices[0].id));
    check(
        cudaStreamBeginCapture(devices[0].stream, cudaStreamCaptureModeGlobal));
    bool rejected = false;
    try {
      local.enqueue(captured);
    } catch (std::invalid_argument const&) {
      rejected = true;
    }
    cudaGraph_t graph = nullptr;
    check(cudaStreamEndCapture(devices[0].stream, &graph));
    check(cudaGraphDestroy(graph));
    if (!rejected)
      throw std::runtime_error("local path accepted graph capture");
    std::printf(
        "PASS: %d eight-GPU BF16 local-fallback cases, including tails and "
        "both placements, BF16 rounding, and capture rejection\n",
        cases);
  } catch (std::exception const& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
