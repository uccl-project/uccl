#pragma once
#include "uccl_lite/network_calibration.hpp"
#include "uccl_lite/planner.hpp"
#include "uccl_lite/runtime/collective_backend.hpp"

namespace uccl_lite::runtime {
enum class NetworkBackend { Caller, Hierarchy, DirectExchange };
struct DirectNetworkEvidence {
  std::map<std::string, size_t> traffic;
  PlanEstimate model;
  std::string modelIdentity;
  size_t modelObservations = 0;
  std::vector<std::string> unmodeledResources;
  bool eligible = false, correct = false;
  std::string reason;
  std::vector<double> samples, baselineSamples;
  std::vector<std::string> configurationIdentities;
  bool fromCache = false;
  std::string cacheStatus;
  std::vector<double> validationSamples, validationBaselineSamples;
};
// Only candidates already qualified against their own caller baseline compete.
// A cache winner must supply its fresh checkpoint samples, not historic timing.
NetworkBackend chooseNetworkCandidate(
    bool hierarchyQualified, std::vector<double> const& hierarchyCurrent,
    bool directQualified, std::vector<double> const& directCurrent);
struct NetworkSelection {
  CollectiveRequest request;
  NetworkBackend backend = NetworkBackend::Caller;
  bool candidateEligible = false, candidateCorrect = false;
  std::vector<double> candidateSamples, baselineSamples;
  std::vector<int> executionRankOrder, rails;
  std::string topologyIdentity, executorIdentity, baselineIdentity, reason;
  std::shared_ptr<NetworkProfile const> calibration;
  DirectNetworkEvidence direct;
  // Final simultaneous-resource validation, when multiple executor kinds are
  // retained. Preparation may demote earlier plans to Caller after this check.
  std::vector<double> poolCandidateSamples, poolBaselineSamples;
  // Hierarchy checkpoint metadata. Direct has independent checkpoint fields.
  // On cache hits, main samples retain the original three-round evidence.
  // Fresh checkpoint samples are separate: one ABBA round for a hierarchy
  // winner, or one baseline block for a cached caller decision.
  bool fromCache = false;
  std::string cacheStatus;
  std::vector<double> validationCandidateSamples, validationBaselineSamples;
};
struct NetworkPreparationOptions {
  ScalarType dtype = ScalarType::Float32;
  CallerFallback fallback;
  bool allowNetwork = true, fallbackOnly = false;
  bool allowHierarchy = true, allowDirect = true;
  bool probeResources = true;
  // Rank zero owns this optional atomic snapshot; no shared filesystem needed.
  // Also configurable with UCCL_LITE_NETWORK_PLAN_CACHE at construction.
  std::string planCache;
};
// Caller-baseline selection across hosts, one process per rank, 1..32 ranks.
// createUniqueId is a CPU bootstrap ID, distinct from the caller's native ID.
// prepare is collective, idle and outside native groups/capture. It compares
// eligible network execution with the caller baseline on private guarded data.
// Unsupported layouts/dtypes/extents retain the baseline without initializing
// network transport. Operational preparation failures are errors, never local
// enqueue fallback. Enqueue only executes a collectively prepared handle.
class PreparedNetworkRuntime {
 public:
  static NetworkId createUniqueId();
  PreparedNetworkRuntime(BackendContext context,
                         NetworkPreparationOptions options);
  ~PreparedNetworkRuntime();
  PreparedNetworkRuntime(PreparedNetworkRuntime const&) = delete;
  PreparedNetworkRuntime& operator=(PreparedNetworkRuntime const&) = delete;
  size_t prepare(CollectiveRequest const& request);
  void enqueue(size_t plan, void const* input, void* output,
               cudaStream_t stream);
  void synchronize();
  bool idle() const;
  bool networkInitialized() const;
  NetworkSelection const& selection(size_t plan) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace uccl_lite::runtime
