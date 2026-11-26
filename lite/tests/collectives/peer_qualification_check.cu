#include "uccl_lite/peer_calibration.hpp"
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>
int main() {
  try {
    using namespace uccl_lite;
    auto t = inspectTopology({0, 1});
    Workload w{Collective::AllGather, ScalarType::BFloat16, 1024, false, 2};
    if (cudaSetDevice(1) != cudaSuccess)
      throw std::runtime_error("cannot select test GPU");
    int calls = 0;
    bool rejected = false;
    try {
      qualifyPeerPlans(t, w, {PeerFamily::Direct},
                       [&](CollectiveCall const&, PeerFamily) { ++calls; });
    } catch (std::runtime_error const& e) {
      rejected =
          std::string(e.what()).find("output/input/guard") != std::string::npos;
    }
    if (!rejected || calls != 5)
      throw std::runtime_error(
          "fast no-op candidate escaped full-buffer validation");
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device != 1)
      throw std::runtime_error("failed qualification changed caller device");
    calls = 0;
    rejected = false;
    try {
      qualifyPeerPlans(t, w, {PeerFamily::Bulk},
                       [&](CollectiveCall const&, PeerFamily) { ++calls; });
    } catch (std::invalid_argument const&) {
      rejected = true;
    }
    if (!rejected || calls)
      throw std::runtime_error(
          "qualification without a baseline started trials");
    std::cout << "PASS: no-op timing cannot qualify, missing baseline rejects, "
                 "failed trials restore the caller device\n";
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
