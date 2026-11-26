#pragma once
#include "uccl_lite/collective_types.hpp"
#include "uccl_lite/collectives.h"
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace uccl_lite {
struct DirectModelCalibration;
}
namespace uccl_lite::runtime {
struct RankBufferLayout {
  bool inPlace = false;
  // Exact minimum of the two pointer alignments, capped at 256 bytes.
  unsigned alignment = 16;
};
struct LocalRequest {
  NetworkOperation operation = NetworkOperation::AllReduce;
  size_t count = 0;
  // In logical source-rank order, identical on every rank at prepare.
  std::vector<RankBufferLayout> layouts;
  // Explicit collective policy for this handle, not a local enqueue fallback.
  bool fallbackOnly = false;
};
using CollectiveRequest = LocalRequest;
struct BackendContext {
  NetworkId id{};
  int rank = 0, device = 0, rail = -1, ranks = 0;
  size_t capacity = 0;
  // Local providers use logical rank order. Distributed providers resolve
  // ownership collectively and may use a stable per-host device inventory.
  std::vector<int> localDevices;
  std::vector<LocalRequest> requests;
};
// Legacy experimental specializations declare their actual contract here.
// The ABI frontend has no compile-time topology or dtype selection. Matching
// this descriptor is necessary, not sufficient: exact profile/identity and
// runtime transport capabilities must also agree before construction.
struct BackendSpec {
  std::string name, legacyProfileMagic;
  std::vector<int> ranksPerHost;
  ScalarType dtype = ScalarType::Float32;
  bool peerAccess = false, mappedHost = false;
  int minLocalRanks = 0, maxLocalRanks = 0;
  bool automatic = false;
  // Optional dynamic host/total bounds. These describe capacity only; they
  // never infer physical ownership, transport support, or admission evidence.
  // When enabled, min/maxLocalRanks apply to each participating host.
  int minHosts = 0, maxHosts = 0, minTotalRanks = 0, maxTotalRanks = 0;
  // Complete per-rank byte extent granularity; legacy specializations retain
  // 128.
  size_t extentMultiple = 128;
  bool rankUuidProfiles = false;
  bool acceptsLayout(std::vector<int> const& layout) const;
  int ranks() const;
  int elementBytes() const;
  int hostForRank(int rank) const;
  int localRank(int rank) const;
};
class CollectiveBackend {
 public:
  virtual ~CollectiveBackend() = default;
  virtual void prepare() = 0;
  virtual void enqueue(NetworkOperation op, void const* input, void* output,
                       size_t count, cudaStream_t stream) = 0;
  virtual void synchronize() = 0;
  virtual bool initialized() const = 0;
  virtual bool idle() const = 0;
  // Immutable prepared transport/placement identity, when supplied by a
  // provider.
  virtual std::string configurationIdentity() const { return {}; }
  // Global logical traffic/operation counts for the exact prepared topology.
  // Diagnostic only; bytes do not constitute measured bandwidth or admission.
  virtual std::map<std::string, size_t> traffic(NetworkOperation,
                                                size_t) const {
    return {};
  }
  // Owned, fixed-binding graphs prepared outside capture. This does not
  // authorize external capture of enqueue. Preparation/launch are collective.
  virtual bool supportsGraphs() const { return false; }
  virtual size_t prepareGraph(NetworkOperation, void const*, void*, size_t) {
    throw std::logic_error("backend has no owned graph implementation");
  }
  virtual void launchGraph(size_t, cudaStream_t) {
    throw std::logic_error("backend has no owned graph implementation");
  }
  virtual void releaseGraph(size_t) {
    throw std::logic_error("backend has no owned graph implementation");
  }
};
// Caller-owned baseline. Identity must include implementation and
// configuration; the runtime also binds the request to live
// rank/device/placement identity.
struct CallerFallback {
  std::shared_ptr<CollectiveBackend> backend;
  ScalarType dtype = ScalarType::BFloat16;
  std::string identity;
  std::string localIdentity;
};
struct BackendRegistration {
  BackendSpec spec;
  std::function<NetworkId()> uniqueId;
  std::function<std::unique_ptr<CollectiveBackend>(BackendContext const&)>
      create;
};
std::vector<BackendRegistration> const& availableBackends();
BackendRegistration peerPacketBackend();
BackendRegistration peerStagedBackend();
BackendRegistration peerStagedBackend(ScalarType dtype);
BackendRegistration peerTreeBackend(ScalarType dtype, std::vector<int> parents);
BackendRegistration preparedLocalBackend();
BackendRegistration preparedLocalBackend(ScalarType dtype);
BackendRegistration hostPacketBackend();
BackendRegistration networkHierarchyBackend();
BackendRegistration preparedDistributedBackend(ScalarType dtype);
// Generic executor factory. Automatic admission belongs to the exact-workload
// caller-baseline selector, not to the registration itself.
BackendRegistration directExchangeBackend(ScalarType dtype);
std::shared_ptr<DirectModelCalibration const> calibrateDirectExchange(
    CollectiveBackend& backend);
}  // namespace uccl_lite::runtime
