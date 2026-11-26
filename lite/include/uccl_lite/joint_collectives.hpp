#pragma once

#include "collective_dispatch.hpp"
#include "peer_plan.hpp"
#include <memory>
#include <string>

namespace uccl_lite {

enum class JointMode { Peer, Router, Auto };

// Single process, capability-checked BF16 out-of-place collectives. Calls are
// host serialized. prepare() is an explicit initialization boundary with idle
// caller streams. enqueue() respects producers on each caller stream and queues
// output completion AND input-reader completion on every caller stream. The
// initial peer adapter is eager-only; graph capture is rejected before side
// effects.
//
// The peer implementation adapts peer_ag_rs.cu from upstream 5b9a7f86:
// direct pulls, large AG peer-copy fan-out, large RS peer copies into private
// destination scratch, and large AR reduce-scatter followed by peer allgather.
// Peer paths support one through kPeerRankCapacity ranks; ordered FP32 sums
// convert once to BF16 before final output distribution. These are adaptations,
// not the unchanged FP32 benchmark envelope. Every exact workload must be
// prepared before enqueue; the resulting plan is cached for this immutable
// topology/calibration. Scratch growth is confined to prepare. Incomplete
// performance data retains the direct baseline; it never imports a size
// threshold from a different machine. Diagnostic family forcing is available
// through PeerTuning for validation and calibration only. Peer rank capacity is
// a kernel argument limit; Auto never forces CPU routing.
class JointCollectives {
 public:
  JointCollectives(Topology topology, JointMode mode, BackendPolicy policy = {},
                   cpu_router::CpuRouterOptions options = {},
                   PeerTuning peerTuning = {});
  ~JointCollectives();
  JointCollectives(JointCollectives const&) = delete;
  JointCollectives& operator=(JointCollectives const&) = delete;

  BackendDecision prepare(CollectiveCall const& call);
  BackendDecision enqueue(CollectiveCall const& call);
  BackendDecision decision(CollectiveCall const& call) const;
  Topology const& topology() const;
  bool routerInitialized() const;
  PeerSelection const& peerPlan(CollectiveCall const& call) const;
  static std::string peerIdentity();
  static bool peerSupported(Topology const& topology);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace uccl_lite
