#include "uccl_lite/collective_dispatch.hpp"
#include <array>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace uccl_lite;
namespace {
void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
std::vector<int> testDevices() {
  int count = 0;
  check(cudaGetDeviceCount(&count));
  if (count < 4) throw std::runtime_error("dispatcher check requires four GPUs");
  return count >= 8 ? std::vector<int>{0, 1, 4, 5}
                    : std::vector<int>{0, 1, 2, 3};
}
void run(bool legacy, int launchWorkers, bool smallGpu) {
  auto devices = testDevices();
  constexpr size_t maxBytes = 65552;
  CollectiveCall call;
  call.collective = Collective::ReduceScatter;
  call.count = 128;
  std::array<cudaStream_t, 4> captureStreams{};
  for (int device : devices) {
    check(cudaSetDevice(device));
    void *input, *output;
    check(cudaMalloc(&input, maxBytes));
    check(cudaMalloc(&output, maxBytes / devices.size()));
    cudaStream_t stream = launchWorkers ? cudaStreamLegacy : nullptr;
    if (!legacy)
      check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(cudaMemset(input, 0, maxBytes));
    call.ports.push_back({input, output, stream, device});
  }
  BackendPolicy policy;
  policy.mode = RouterMode::Force;
  cpu_router::CpuRouterOptions options;
  options.workersPerNode = 2;
  options.maxTensorBytes = maxBytes;
  options.gpuLaunchWorkersPerNode = launchWorkers;
  options.smallReduceScatterGpuProgressMaxBytes = smallGpu ? 65536 : 0;
  if (launchWorkers) {
    for (cudaStream_t stream : {cudaStreamPerThread, cudaStream_t(nullptr)}) {
      auto perThreadPorts = call.ports;
      perThreadPorts[0].stream = stream;
      bool rejected = false;
      try {
        cpu_router::PacketPipeline invalid(options);
        invalid.prepare(perThreadPorts);
      } catch (std::invalid_argument const&) {
        rejected = true;
      }
      if (!rejected)
        throw std::runtime_error(
            "parallel launch accepted thread-relative stream");
    }
  }
  CollectiveDispatcher dispatcher(
      inspectTopology(devices), policy,
      [](CollectiveCall const&) {
        throw std::runtime_error("unexpected fallback");
      },
      options);
  dispatcher.prepare(call);
  auto verify = [&] {
    dispatcher.enqueue(call);
    for (auto const& port : call.ports) {
      check(cudaSetDevice(port.device));
      check(cudaStreamSynchronize(port.stream));
      std::vector<uint16_t> result(call.count, 1);
      check(cudaMemcpy(result.data(), port.output, call.count * 2,
                       cudaMemcpyDeviceToHost));
      for (auto value : result)
        if (value)
          throw std::runtime_error("incorrect result after capture rejection");
    }
  };
  verify();
  for (size_t rank = 0; rank < devices.size(); ++rank) {
    check(cudaSetDevice(devices[rank]));
    if (legacy)
      // Capture a blocking stream: querying this device's legacy stream must
      // reject implicit capture even if another device is current on entry.
      check(cudaStreamCreate(&captureStreams[rank]));
    else
      captureStreams[rank] = call.ports[rank].stream;
    check(cudaStreamBeginCapture(captureStreams[rank],
                                 cudaStreamCaptureModeGlobal));
    check(cudaSetDevice(devices[(rank + 1) % devices.size()]));
    bool rejected = false;
    try {
      dispatcher.enqueue(call);
    } catch (std::exception const&) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("dispatcher accepted external capture");
    check(cudaSetDevice(devices[rank]));
    cudaGraph_t graph = nullptr;
    check(cudaStreamEndCapture(captureStreams[rank], &graph));
    size_t nodes = 0;
    check(cudaGraphGetNodes(graph, nullptr, &nodes));
    check(cudaGraphDestroy(graph));
    if (nodes) throw std::runtime_error("rejected capture enqueued GPU work");
    // A rejected call must not advance CPU jobs, GPU tickets or generations.
    verify();
  }
  if (smallGpu) {
    call.count = 65536 / (devices.size() * 2);
    dispatcher.prepare(call);
    verify();
    for (size_t bytes : {maxBytes, size_t(1024)}) {
      call.count = bytes / (devices.size() * 2);
      bool rejected = false;
      try {
        dispatcher.enqueue(call);
      } catch (std::logic_error const&) {
        rejected = true;
      }
      if (!rejected)
        throw std::runtime_error(
            "changed reduction mode did not require prepare");
      dispatcher.prepare(call);
      verify();
    }
  }
  for (size_t rank = 0; rank < devices.size(); ++rank) {
    check(cudaSetDevice(devices[rank]));
    if (legacy)
      check(cudaStreamDestroy(captureStreams[rank]));
    else
      check(cudaStreamDestroy(call.ports[rank].stream));
    check(cudaFree(const_cast<void*>(call.ports[rank].input)));
    check(cudaFree(call.ports[rank].output));
  }
}
// Nonzero data across both placement boundaries, with aliasing and repeated
// operation changes. A mode change must reject enqueue until prepare is called.
void runArAgProgressBoundaries() {
  auto devices = testDevices();
  constexpr size_t maxBytes = 65552;
  constexpr uint16_t values[] = {0x3f80, 0x4000, 0x4040, 0x4080};
  CollectiveCall call;
  std::vector<void*> inputs, outputs;
  for (int device : devices) {
    check(cudaSetDevice(device));
    void *input, *output;
    cudaStream_t stream;
    check(cudaMalloc(&input, maxBytes));
    check(cudaMalloc(&output, maxBytes));
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    inputs.push_back(input);
    outputs.push_back(output);
    call.ports.push_back({input, output, stream, device});
  }
  BackendPolicy policy;
  policy.mode = RouterMode::Force;
  cpu_router::CpuRouterOptions options;
  options.workersPerNode = 2;
  options.maxTensorBytes = maxBytes;
  options.batchCpuJobs = true;
  options.gpuLaunchWorkersPerNode = 1;
  options.smallAllReduceGpuProgressMaxBytes = 8192;
  options.smallAllGatherGpuProgressMaxBytes = 65536;
  // Equal packet geometry isolates progress-mode changes from packet changes.
  options.smallAllReducePacketBytes = 0;
  CollectiveDispatcher dispatcher(
      inspectTopology(devices), policy,
      [](CollectiveCall const&) {
        throw std::runtime_error("unexpected AR/AG fallback");
      },
      options);
  for (bool inPlace : {false, true})
    for (Collective op : {Collective::AllReduce, Collective::AllGather}) {
      size_t limit = op == Collective::AllReduce ? 8192 : 65536;
      for (size_t bytes : {limit, limit + 16, size_t(1024)}) {
        call.collective = op;
        call.inPlace = inPlace;
        call.count = bytes / 2 / (op == Collective::AllGather ? 4 : 1);
        for (size_t rank = 0; rank < devices.size(); ++rank) {
          auto& port = call.ports[rank];
          check(cudaSetDevice(port.device));
          port.output = outputs[rank];
          port.input =
              inPlace
                  ? static_cast<uint16_t*>(outputs[rank]) +
                        (op == Collective::AllGather ? rank * call.count : 0)
                  : inputs[rank];
          std::vector<uint16_t> data(call.count, values[rank]);
          check(cudaMemcpy(const_cast<void*>(port.input), data.data(),
                           data.size() * 2, cudaMemcpyHostToDevice));
        }
        if (bytes == limit + 16 ||
            (op == Collective::AllGather && bytes == limit)) {
          bool rejected = false;
          try {
            dispatcher.enqueue(call);
          } catch (std::logic_error const&) {
            rejected = true;
          }
          if (!rejected)
            throw std::runtime_error("AR/AG mode change skipped prepare");
        }
        dispatcher.prepare(call);
        dispatcher.enqueue(call);
        for (auto const& port : call.ports) {
          check(cudaSetDevice(port.device));
          check(cudaStreamSynchronize(port.stream));
          std::vector<uint16_t> result(bytes / 2);
          check(cudaMemcpy(result.data(), port.output, bytes,
                           cudaMemcpyDeviceToHost));
          for (size_t i = 0; i < result.size(); ++i) {
            uint16_t expected = op == Collective::AllGather
                                    ? values[i / call.count]
                                    : 0x4120;  // 1 + 2 + 3 + 4 = 10.
            if (result[i] != expected)
              throw std::runtime_error("incorrect AR/AG boundary result");
          }
        }
      }
    }
  for (size_t rank = 0; rank < devices.size(); ++rank) {
    check(cudaSetDevice(devices[rank]));
    check(cudaStreamDestroy(call.ports[rank].stream));
    check(cudaFree(inputs[rank]));
    check(cudaFree(outputs[rank]));
  }
}
}  // namespace
int main() {
  try {
    runArAgProgressBoundaries();
    for (int workers : {0, 1, 4}) {
      for (bool smallGpu : {false, true}) {
        run(false, workers, smallGpu);
        run(true, workers, smallGpu);
      }
    }
    std::puts(
        "PASS: dispatcher rejects external/implicit capture on every rank; "
        "eager recovery; AR/AG progress boundaries and in-place results");
    return 0;
  } catch (std::exception const& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
