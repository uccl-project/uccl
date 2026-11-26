#pragma once

#include "planner.hpp"
#include <array>

namespace uccl_lite {
// Device argument storage capacity, not a topology-specific algorithm switch.
inline constexpr int kPeerRankCapacity = 4;
enum class PeerFamily { Direct = 0, Coordinator = 1, Bulk = 2 };
struct PeerTuning {
  // Automatic probes/qualification run only at an idle prepare boundary.
  bool automaticCalibration = true;
  // Optional persisted local hardware profile; otherwise consult
  // UCCL_LITE_HARDWARE_PROFILE. Missing data stays unknown.
  std::string hardwareProfile;
  // Optional exact-workload cache; also configurable with
  // UCCL_LITE_PEER_PLAN_CACHE.
  std::string planCache;
  // Per-family overhead priors for supplied models. Automatic qualification
  // fits a per-workload residual including unmodeled costs; it is not a pure
  // launch-latency measurement and is never extrapolated. Unknown stays -1.
  std::array<double, 3> overheadUs{{-1, -1, -1}};
  // Explicit diagnostic selection for correctness/calibration, never an
  // automatic performance claim. Capability checks still apply.
  std::optional<PeerFamily> validationFamily;
};
struct PeerSelection {
  PeerFamily family = PeerFamily::Direct;
  ExecutionPlan plan;
  bool modeled = false;
  std::string reason;
  bool qualified = false;
  std::array<std::vector<double>, 3> localTimings;
  std::string calibrationIdentity;
  bool fromCache = false;
};
std::vector<PlanCandidate> peerCandidates(Topology const& topology,
                                          Workload const& workload,
                                          PeerTuning const& tuning);
PeerSelection selectPeerPlan(Topology const& topology, Workload const& workload,
                             PeerTuning const& tuning);
}  // namespace uccl_lite
