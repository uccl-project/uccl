#pragma once
#include "peer_plan.hpp"

namespace uccl_lite {
struct PeerPlanRecord {
  std::string topologyIdentity, executorIdentity, calibrationIdentity;
  Workload workload;
  std::array<std::vector<double>, 3> timings;
  bool correct = false;
};
// Same admission rule for fresh qualification and persisted records.
PeerFamily qualifiedPeerFamily(
    std::array<std::vector<double>, 3> const& timings);
std::vector<PeerPlanRecord> loadPeerPlanRecords(std::string const& path);
void updatePeerPlanRecord(std::string const& path,
                          PeerPlanRecord const& record);
std::optional<PeerSelection> matchPeerPlanRecord(
    Topology const& topology, Workload const& workload,
    std::string const& executorIdentity,
    std::vector<PeerPlanRecord> const& records);
}  // namespace uccl_lite
