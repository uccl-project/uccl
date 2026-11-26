#pragma once
#include "collective_call.hpp"
#include "peer_plan.hpp"
#include <functional>

namespace uccl_lite {
struct PeerTrial {
  PeerFamily family = PeerFamily::Direct;
  std::vector<double> microseconds;
};
struct PeerQualification {
  std::vector<PeerTrial> trials;
  PeerFamily family = PeerFamily::Direct;
};
// Uses private guarded BF16 buffers with the caller's exact extent/alignment.
// All candidate timing is outside the caller's buffers, streams and hot path.
// Checks guards, inputs and every output after each timed repetition.
PeerQualification qualifyPeerPlans(
    Topology const& topology, Workload const& workload,
    std::vector<PeerFamily> const& families,
    std::function<void(CollectiveCall const&, PeerFamily)> const& enqueue);
}  // namespace uccl_lite
