// Standalone peer consumer: deliberately includes/links no router or NCCL API.
#include "uccl_lite/collective_call.hpp"
#include "uccl_lite/collectives/peer_collectives.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
using namespace uccl_lite;
void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
int main(int argc, char** argv) {
  try {
    std::vector<int> devices{2, 0, 3, 1};
    std::array<__nv_bfloat16*, 4> inputs{}, outputs{};
    std::array<cudaStream_t, 4> streams{};
    constexpr size_t count = 257, extent = count * 4;
    for (size_t r = 0; r < devices.size(); ++r) {
      check(cudaSetDevice(devices[r]));
      check(cudaMalloc(&inputs[r], extent * 2));
      check(cudaMalloc(&outputs[r], extent * 2));
      check(cudaStreamCreateWithFlags(&streams[r], cudaStreamNonBlocking));
      std::vector<__nv_bfloat16> values(extent, __float2bfloat16(float(r + 1)));
      check(cudaMemcpy(inputs[r], values.data(), extent * 2,
                       cudaMemcpyHostToDevice));
    }
    {
      PeerTuning tuning;
      tuning.automaticCalibration = true;
      PeerCollectives peer(inspectTopology(devices), tuning);
      for (auto op : {Collective::AllReduce, Collective::AllGather,
                      Collective::ReduceScatter}) {
        CollectiveCall call;
        call.collective = op;
        call.count = count;
        for (size_t r = 0; r < devices.size(); ++r)
          call.ports.push_back({inputs[r], outputs[r], streams[r], devices[r]});
        peer.prepare(call);
        if (!peer.plan(call).qualified)
          throw std::runtime_error(
              "standalone automatic qualification missing");
        if (argc > 1 &&
            peer.plan(call).fromCache != (std::string(argv[1]) == "cached"))
          throw std::runtime_error("unexpected persisted-plan cache behavior");
        peer.enqueue(call);
        for (size_t r = 0; r < devices.size(); ++r) {
          check(cudaSetDevice(devices[r]));
          check(cudaStreamSynchronize(streams[r]));
          size_t elements = op == Collective::AllGather ? extent : count;
          std::vector<__nv_bfloat16> actual(elements);
          check(cudaMemcpy(actual.data(), outputs[r], elements * 2,
                           cudaMemcpyDeviceToHost));
          for (size_t i = 0; i < elements; ++i)
            if (__bfloat162float(actual[i]) !=
                (op == Collective::AllGather ? float(i / count + 1) : 10.0f))
              throw std::runtime_error("standalone peer result mismatch");
        }
      }
    }
    for (size_t r = 0; r < devices.size(); ++r) {
      check(cudaSetDevice(devices[r]));
      check(cudaStreamDestroy(streams[r]));
      check(cudaFree(inputs[r]));
      check(cudaFree(outputs[r]));
    }
    std::cout << "PASS: standalone peer AR/AG/RS with shuffled devices and "
                 "scalar tails\n";
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
