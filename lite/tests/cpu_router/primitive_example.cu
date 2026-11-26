// Direct AR/AG/RS programs composed from forward, process and multicast.
// PacketPipeline overlaps these operations using persistent packet credits.
#include "cpu_router.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
using namespace uccl_lite::cpu_router;
namespace {
constexpr int ranks = 2;
enum class Collective { AllReduce, AllGather, ReduceScatter };
char const* name(Collective op) {
  return op == Collective::AllReduce   ? "AllReduce"
         : op == Collective::AllGather ? "AllGather"
                                       : "ReduceScatter";
}
void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
uint16_t bf16(float value) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}
uint16_t inputValue(int rank, size_t index) {
  return bf16(float((rank + 1) * (1 + index % 13)) / 16.0f);
}
void wait(std::vector<ChannelCompletion> const& tickets) {
  for (auto const& ticket : tickets) ticket.wait();
}

// Uploaded inputs are immutable until every returned reader ticket completes.
std::vector<ChannelCompletion> allGather(
    std::array<MemorySpan, ranks> const& sources,
    std::array<MemorySpan, ranks> const& outputs,
    std::array<std::unique_ptr<Channel>, ranks> const& channels,
    std::array<cudaStream_t, ranks> const& streams, size_t bytes) {
  std::vector<ChannelCompletion> readers;
  for (int source = 0; source < ranks; ++source) {
    std::vector<RouterDestination> destinations;
    for (int destination = 0; destination < ranks; ++destination)
      destinations.push_back(
          {channels[destination].get(),
           outputs[destination].subspan(source * bytes, bytes),
           streams[destination]});
    // Rank-ordered output views perform concatenation without CPU payload
    // packing.
    auto tickets = CpuRouter::multicast(sources[source], destinations);
    readers.insert(readers.end(), tickets.begin(), tickets.end());
  }
  return readers;
}

std::vector<ChannelCompletion> reduceScatter(
    std::array<MemorySpan, ranks> const& sources,
    std::array<MemorySpan, ranks> const& outputs,
    std::array<RegisteredMemory, ranks> const& results,
    std::array<std::unique_ptr<Channel>, ranks> const& channels,
    std::array<cudaStream_t, ranks> const& streams, size_t bytes) {
  std::vector<ChannelCompletion> readers;
  for (int destination = 0; destination < ranks; ++destination) {
    std::array<MemorySpan, ranks> shard;
    for (int source = 0; source < ranks; ++source)
      shard[source] = sources[source].subspan(destination * bytes, bytes);
    auto result = results[destination].span().subspan(0, bytes);
    CpuRouter::process(shard.data(), shard.size(), result);
    // Only the owner receives this result shard.
    readers.push_back(CpuRouter::forward(*channels[destination],
                                         outputs[destination], result,
                                         streams[destination]));
  }
  return readers;
}

void run(Collective op, TransportBackend backend, bool inPlace, size_t count,
         std::array<int, ranks> const& devices) {
  size_t bytes = count * 2;
  size_t inputBytes = op == Collective::ReduceScatter ? ranks * bytes : bytes;
  size_t outputBytes = op == Collective::AllGather ? ranks * bytes : bytes;
  std::array<RegisteredMemory, ranks> gpuInputs, gpuOutputs, host, results;
  std::array<MemorySpan, ranks> inputs, outputs, sources;
  std::array<cudaStream_t, ranks> streams;
  std::array<std::unique_ptr<Channel>, ranks> channels;
  std::vector<ChannelCompletion> uploads;
  for (int rank = 0; rank < ranks; ++rank) {
    check(cudaSetDevice(devices[rank]));
    check(cudaStreamCreateWithFlags(&streams[rank], cudaStreamNonBlocking));
    gpuOutputs[rank] =
        RegisteredMemory::gpu(std::max(inputBytes, outputBytes), devices[rank]);
    auto storage = gpuOutputs[rank].span();
    outputs[rank] = storage.subspan(
        inPlace && op == Collective::ReduceScatter ? rank * bytes : 0,
        outputBytes);
    if (inPlace)
      inputs[rank] = storage.subspan(
          op == Collective::AllGather ? rank * bytes : 0, inputBytes);
    else {
      gpuInputs[rank] = RegisteredMemory::gpu(inputBytes, devices[rank]);
      inputs[rank] = gpuInputs[rank].span();
    }
    int node = gpuNumaNode(devices[rank]);
    host[rank] = RegisteredMemory::pinned(inputBytes, node);
    results[rank] = RegisteredMemory::pinned(bytes, node);
    // Pinned allocations are page-rounded; use only the logical payload.
    sources[rank] = host[rank].span().subspan(0, inputBytes);
    std::vector<uint16_t> values(inputBytes / 2);
    for (size_t i = 0; i < values.size(); ++i) values[i] = inputValue(rank, i);
    check(cudaMemcpy(inputs[rank].address, values.data(), inputBytes,
                     cudaMemcpyHostToDevice));
    channels[rank] = std::make_unique<Channel>(
        ChannelOptions{backend, devices[rank], 32, 256, 8});
    uploads.push_back(CpuRouter::forward(*channels[rank], sources[rank],
                                         inputs[rank], streams[rank]));
  }
  // Snapshot all sources before overwriting any overlapping GPU output.
  wait(uploads);
  if (op == Collective::AllGather)
    wait(allGather(sources, outputs, channels, streams, bytes));
  else if (op == Collective::ReduceScatter)
    wait(reduceScatter(sources, outputs, results, channels, streams, bytes));
  else {
    auto packet = results[0].span().subspan(0, bytes);
    CpuRouter::process(sources.data(), sources.size(), packet);
    std::vector<RouterDestination> destinations;
    for (int rank = 0; rank < ranks; ++rank)
      destinations.push_back(
          {channels[rank].get(), outputs[rank], streams[rank]});
    wait(CpuRouter::multicast(packet, destinations));
  }
  for (int rank = 0; rank < ranks; ++rank) {
    check(cudaSetDevice(devices[rank]));
    std::vector<uint16_t> values(outputBytes / 2);
    check(cudaMemcpy(values.data(), outputs[rank].address, outputBytes,
                     cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < values.size(); ++i) {
      uint16_t expected;
      if (op == Collective::AllGather)
        expected = inputValue(int(i / count), i % count);
      else {
        size_t index = op == Collective::ReduceScatter ? rank * count + i : i;
        expected = bf16(float(3 * (1 + index % 13)) / 16.0f);
      }
      if (values[i] != expected)
        throw std::runtime_error(std::string(name(op)) + " mismatch at rank " +
                                 std::to_string(rank) + " index " +
                                 std::to_string(i));
    }
    check(cudaStreamDestroy(streams[rank]));
  }
}
}  // namespace
int main() {
  try {
    int available;
    check(cudaGetDeviceCount(&available));
    if (available < ranks)
      throw std::runtime_error("primitive examples require two GPUs");
    std::array<int, ranks> devices{0, available > 4 ? 4 : 1};
    int cases = 0;
    for (auto backend : {TransportBackend::GpuKernel, TransportBackend::Dma})
      for (bool inPlace : {false, true})
        for (size_t count : {size_t(1), size_t(17), size_t(4097)})
          for (auto op : {Collective::AllReduce, Collective::AllGather,
                          Collective::ReduceScatter}) {
            run(op, backend, inPlace, count, devices);
            ++cases;
          }
    std::printf(
        "Primitive AR/AG/RS: %d cases passed (kernel/DMA, tails, "
        "in-place/out-of-place, "
        "devices %d,%d)\n",
        cases, devices[0], devices[1]);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
