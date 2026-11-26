#pragma once

#include "uccl_lite/peer_plan.hpp"
#include <memory>

namespace uccl_lite {
struct CollectiveCall;
struct Topology;

// Internal prepared BF16 peer executor shared by all-pairs-P2P platforms.
// JointCollectives owns validation, capture rejection and backend policy. This
// executor owns only peer streams, events, scratch and collective scheduling.
// Its caller must validate positive, out-of-place BF16 calls on this topology,
// serialize host calls, and prepare each exact workload with idle streams.
// Plans use an immutable topology/calibration snapshot, so enqueue needs no
// allocation or model evaluation. No CPU-router resources are allocated here;
// unsupported topologies reject.
class PeerCollectives {
 public:
  explicit PeerCollectives(Topology const& topology, PeerTuning tuning = {});
  ~PeerCollectives();
  PeerCollectives(PeerCollectives const&) = delete;
  PeerCollectives& operator=(PeerCollectives const&) = delete;

  void prepare(CollectiveCall const& call);
  void enqueue(CollectiveCall const& call);
  PeerSelection const& plan(CollectiveCall const& call) const;
  static bool supported(Topology const& topology);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace uccl_lite
