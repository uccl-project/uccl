#pragma once
#include "uccl_lite/network_calibration.hpp"
#include "uccl_lite/runtime/collective_backend.hpp"

namespace uccl_lite::runtime {
struct RuntimeContext {
  NetworkId id{};
  int rank = 0, device = 0, ranks = 0;
  size_t capacity = 0;
  BackendContext backendContext() const {
    BackendContext c;
    c.id = id;
    c.rank = rank;
    c.device = device;
    c.ranks = ranks;
    c.capacity = capacity;
    return c;
  }
};
enum class RuntimeDomain { Local, Distributed, CallerOnly };
enum class RuntimeBackend {
  Caller,
  Peer,
  Host,
  PeerPacket,
  NetworkHierarchy,
  PeerTree,
  DirectExchange
};
struct RankParticipant {
  std::string host, gpu;
  int device = -1;
};
struct RuntimePlacement {
  RuntimeDomain domain = RuntimeDomain::CallerOnly;
  std::vector<RankParticipant> ranks;
};
struct CandidateEvidence {
  std::map<std::string, size_t> traffic;
  std::string resourceModelIdentity, resourceModelReason;
  size_t resourceModelObservations = 0;
  std::vector<std::string> unmodeledResources;
  RuntimeBackend backend;
  bool eligible = false, correct = false, fromCache = false;
  std::string cacheStatus;
  std::vector<double> samples, baselineSamples;
  std::vector<int> executionRankOrder, rails;
  std::vector<int> peerTreeParents;
  double resourceEstimateUs = -1;
  double resourceSerialUs = -1, resourceDependencyUs = -1, resourceLoadUs = -1;
  std::shared_ptr<NetworkProfile const> networkCalibration;
  std::vector<double> validationSamples, validationBaselineSamples;
  std::vector<std::string> configurationIdentities;
};
struct PreparedSelection {
  CollectiveRequest request;
  RuntimeBackend backend = RuntimeBackend::Caller;
  bool graph = false;
  bool fromCache = false;
  std::string cacheStatus;
  std::vector<CandidateEvidence> candidates;
  std::vector<double> poolCandidateSamples, poolBaselineSamples;
  std::string topologyIdentity, executorIdentity, baselineIdentity, reason;
};
struct PreparationOptions {
  ScalarType dtype = ScalarType::BFloat16;
  CallerFallback fallback;
  bool fallbackOnly = false;
  bool allowPeer = true, allowHost = true, allowPeerPacket = true,
       allowNetwork = true;
  bool probeResources = true;
  bool allowPeerDirect = true, allowPeerTree = true;
  std::string hardwareProfile;
  std::string networkPlanCache;
};
// One process per participating GPU, 1..32 logical ranks. The caller supplies
// its ready baseline and actual rank/device, not host counts or GPU groups.
// Collective prepare discovers placement and dispatches to the appropriate
// qualified selector. Multi-host partial views may qualify direct exchange;
// single-host partial views retain the caller baseline.
// All calls are host-serialized. Preparation/release require idle execution,
// closed native groups and no external capture. Enqueue never discovers/tunes.
class PreparedRuntime {
 public:
  static NetworkId createUniqueId();
  PreparedRuntime(RuntimeContext context, PreparationOptions options);
  ~PreparedRuntime();
  PreparedRuntime(PreparedRuntime const&) = delete;
  PreparedRuntime& operator=(PreparedRuntime const&) = delete;
  size_t prepare(CollectiveRequest const& request);
  void enqueue(size_t plan, void const* input, void* output,
               cudaStream_t stream);
  size_t prepareGraph(CollectiveRequest const& request, void const* input,
                      void* output);
  void launchGraph(size_t graph, cudaStream_t stream);
  void releaseGraph(size_t graph);
  void synchronize();
  bool idle() const;
  bool acceleratedResourcesInitialized() const;
  RuntimePlacement const& placement() const;
  PreparedSelection const& selection(size_t plan) const;
  PreparedSelection const& graphSelection(size_t graph) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace uccl_lite::runtime
