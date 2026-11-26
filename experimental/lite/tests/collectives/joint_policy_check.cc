#include "uccl_lite/joint_collectives.hpp"
#include <iostream>
#include <stdexcept>

using namespace uccl_lite;

int main() {
  try {
    for (int ranks : {1, 2, 3, 4, 8}) {
      Topology topology;
      topology.identity = "synthetic-no-peer";
      topology.avx2 = true;
      topology.peerAccess.resize(ranks * ranks, false);
      topology.permittedCpusPerNode = {{0, 4}};
      topology.permittedCpuIds = {{0, {0, 1, 2, 3}}};
      CollectiveCall call;
      call.count = 64;
      for (int rank = 0; rank < ranks; ++rank) {
        topology.devices.push_back(rank);
        topology.gpuNames.push_back("synthetic GPU");
        topology.numaNodes.push_back(0);
        topology.mappedHost.push_back(true);
        // decision() is CPU-only; no dereference or GPU allocation is allowed.
        call.ports.push_back({reinterpret_cast<void*>(0x1000),
                              reinterpret_cast<void*>(0x2000), nullptr, rank});
      }
      JointCollectives automatic(topology, JointMode::Auto);
      bool rejected = false, routed = false;
      try {
        routed = automatic.decision(call).cpuRouter;
      } catch (std::invalid_argument const&) {
        rejected = true;
      }
      if ((ranks == 1 ? (rejected || routed) : !rejected) ||
          automatic.routerInitialized())
        throw std::runtime_error("Auto forced an uncalibrated router");
      JointCollectives forced(topology, JointMode::Router);
      if (!forced.decision(call).cpuRouter || forced.routerInitialized())
        throw std::runtime_error("explicit validation mode changed");
    }
    std::cout
        << "Joint policy: missing peer support never forces Auto routing\n";
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
