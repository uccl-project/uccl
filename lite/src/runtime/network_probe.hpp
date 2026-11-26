#pragma once
#include "uccl_lite/network_plan.hpp"
#include <mscclpp/core.hpp>

namespace uccl_lite::runtime {
class NetworkCalibrationFailure : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};
// Internal collective probe, called only after live endpoint/algorithm contract
// agreement, idle and outside capture. No MPI, native NCCL, router or files.
// Private buffers and connections are gone before this returns. Both directed
// paths and complete concurrent cohorts are measured; unknown physical rates
// remain unknown. Returns the same raw observations on every participant.
NetworkProfile probeNetworkResources(
    mscclpp::Bootstrap& control, std::vector<NetworkEndpoint> const& endpoints,
    std::vector<std::pair<size_t, size_t>> const& peers,
    std::vector<int> const& physicalRails, size_t maximum);
}  // namespace uccl_lite::runtime
