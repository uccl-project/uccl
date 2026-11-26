// Synthetic timings test admission/plan switching, not performance claims.
#include "uccl_lite/collectives/peer_collectives.hpp"
#include "uccl_lite/router_plan.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
using namespace uccl_lite;
void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
int main() {
  try {
    auto t = inspectTopology({2, 0, 3, 1});
    std::array<void*, 4> input{}, output{};
    std::array<cudaStream_t, 4> streams{};
    CollectiveCall call;
    call.count = 64;
    for (int r = 0; r < 4; ++r) {
      check(cudaSetDevice(t.devices[r]));
      check(cudaMalloc(&input[r], 4096));
      check(cudaMalloc(&output[r], 4096));
      check(cudaStreamCreateWithFlags(&streams[r], cudaStreamNonBlocking));
      std::vector<__nv_bfloat16> data(2048, __float2bfloat16(float(r + 1)));
      check(cudaMemcpy(input[r], data.data(), 4096, cudaMemcpyHostToDevice));
      call.ports.push_back({input[r], output[r], streams[r], t.devices[r]});
    }
    {
      PeerTuning tuning;
      tuning.automaticCalibration = false;
      PeerCollectives peer(t, tuning);
      int fallbackCalls = 0;
      auto fallback = [&](CollectiveCall const& c) {
        ++fallbackCalls;
        peer.enqueue(c);
      };
      cpu_router::CpuRouterOptions base;
      base.maxTensorBytes = 4096;
      auto configurations = routerWorkerConfigurations(t, base, {1, 2});
      BackendPolicy policy;
      policy.fallbackIdentity = "synthetic-test-peer";
      RouterPlanDispatcher descriptors(t, policy, configurations, fallback);
      for (size_t count : {64UL, 128UL}) {
        call.count = count;
        for (size_t i = 0; i < configurations.size(); ++i) {
          bool fast = (count == 64) == (i == 0);
          policy.measurements.push_back(
              {descriptors.candidateTopology(i).identity,
               policy.fallbackIdentity,
               call.workload(),
               fast ? TimingRange{1, 2, 3} : TimingRange{4, 5, 6},
               {20, 21, 22},
               3,
               true});
        }
      }
      RouterPlanDispatcher runtime(t, policy, configurations, fallback);
      auto verify = [&] {
        runtime.enqueue(call);
        for (int r = 0; r < 4; ++r) {
          check(cudaSetDevice(t.devices[r]));
          check(cudaStreamSynchronize(streams[r]));
          std::vector<__nv_bfloat16> values(call.count);
          check(cudaMemcpy(values.data(), output[r], call.count * 2,
                           cudaMemcpyDeviceToHost));
          for (auto x : values)
            if (__bfloat162float(x) != 10)
              throw std::runtime_error("worker-plan result mismatch");
        }
      };
      call.count = 64;
      if (runtime.prepare(call).configuration != 0 ||
          !runtime.routerInitialized())
        throw std::runtime_error("first configuration not prepared");
      verify();
      call.count = 128;
      check(cudaSetDevice(t.devices[0]));
      check(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal));
      bool rejected = false;
      try {
        runtime.prepare(call);
      } catch (std::invalid_argument const&) {
        rejected = true;
      }
      if (!rejected || !runtime.routerInitialized())
        throw std::runtime_error("capture changed active worker plan");
      cudaGraph_t graph;
      check(cudaStreamEndCapture(streams[0], &graph));
      check(cudaGraphDestroy(graph));
      if (runtime.prepare(call).configuration != 1)
        throw std::runtime_error("worker plan did not switch");
      verify();
      call.count = 192;
      peer.prepare(call);
      if (runtime.prepare(call).configuration || runtime.routerInitialized())
        throw std::runtime_error(
            "fallback retained active router after idle prepare");
      verify();
      if (fallbackCalls != 1)
        throw std::runtime_error("fallback was not called exactly once");
      call.count = 64;
      runtime.prepare(call);
      verify();
    }
    for (int r = 0; r < 4; ++r) {
      check(cudaSetDevice(t.devices[r]));
      check(cudaStreamDestroy(streams[r]));
      check(cudaFree(input[r]));
      check(cudaFree(output[r]));
    }
    std::cout << "PASS: admitted worker configurations execute, switch at "
                 "prepare, reject capture and release for fallback\n";
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
