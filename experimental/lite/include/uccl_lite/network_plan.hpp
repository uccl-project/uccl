#pragma once
#include "network_calibration.hpp"
#include "topology.hpp"
#include <map>

namespace uccl_lite {
struct RailDevice {
  std::string name;
  int transportIndex = -1;
};
struct RailAssignment {
  int device = -1;
  int transportIndex = -1;
  std::string nic;
  int port = -1;
  double capacityGBs = 0;
  std::vector<std::string> resources;
};
struct NetworkRailCandidate {
  RailAssignment assignment;
  // Directed physical capacity bounds, scoped to this host by the joint
  // planner.
  std::map<std::string, double> transmit, receive;
  // Physical CPU/PCI bounds are incomplete. This candidate may execute only
  // with matching directed path AND exact complete concurrent-flow evidence.
  bool requiresMeasurement = false;
};
// Every physically supported bidirectional GPU/NIC candidate, in logical rank
// order. Missing reverse-link capacities do not become zero-cost paths.
// The optional catalog expansion retains unknown physical bandwidth and marks
// provisional active-port candidates as requiring path/cohort measurements.
std::vector<std::vector<NetworkRailCandidate>> networkRailCandidates(
    Topology const& topology, std::vector<RailDevice> const& available,
    bool includeUnmeasuredPhysicalPaths = false);
struct NetworkEndpoint {
  NetworkHostIdentity host;
  std::string gpu;
  int device = -1;
  std::vector<NetworkRailCandidate> candidates;
};
struct MeasuredNetworkPlan {
  bool calibrated = false;
  std::vector<RailAssignment> assignments;
  size_t basisBytes = 0;
  double isolatedPathUs = 0, capacityBoundUs = 0;
  bool concurrentMeasured = false;
  double concurrentUs = -1;
  std::string concurrentGroup;
  std::string reason;
};
// Coupled source/destination rail prior for equal-size bidirectional flows.
// Complete live identity/evidence coverage is required; otherwise return no
// calibrated plan. Isolated path medians plus shared physical bounds are an
// advisory cost model, not measured concurrent throughput or backend admission.
MeasuredNetworkPlan planMeasuredNetworkRails(
    NetworkProfile const& profile,
    std::vector<NetworkEndpoint> const& endpoints,
    std::vector<std::pair<size_t, size_t>> const& peers);
// Equal per-rank traffic: minimize projected shared-resource load using known
// GPU--NIC PCI paths and NIC line-rate/measurement bounds. Device names are
// supplied by the transport, so HCA enumeration order is never guessed from a
// CUDA ordinal. This chooses rails within an admitted backend, not GDR support.
// The current MSCCL++ adapter supports a single active port per HCA; ambiguous
// multi-port HCAs are excluded until explicit port selection is integrated.
std::vector<RailAssignment> planNetworkRails(
    Topology const& topology, std::vector<RailDevice> const& available);
}  // namespace uccl_lite
