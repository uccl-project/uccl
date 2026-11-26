#pragma once
#include "uccl_lite/local_plan.hpp"
#include "uccl_lite/runtime/collective_backend.hpp"
#include <array>

namespace uccl_lite::runtime {
struct LocalSelection {
  LocalBackend backend = LocalBackend::Peer;
  LocalRequest request;
  // Graph timings and eager timings are separate admission/cache domains.
  bool graph = false;
  // Timings indexed by LocalBackend, in collection order, microseconds.
  // Three ABBA rounds, two warmups and three checked timed calls per block.
  std::array<std::vector<double>, 5> samples;
  // Paired samples/correctness indexed by kLocalAlternatives, not LocalBackend.
  std::array<std::vector<double>, 4> fallbackSamples;
  std::array<bool, 4> candidateCorrect{};
  bool peerEligible = false;
  std::string fallbackIdentity;
  bool hostEligible = false, hostCorrect = false;
  bool peerPacketEligible = false;
  bool peerTreeEligible = false;
  std::vector<int> peerTreeParents;
  double peerTreeResourceEstimateUs = -1;
  double peerTreeSerialUs = -1, peerTreeDependencyUs = -1,
         peerTreeResourceLoadUs = -1;
  // Execution-rank -> caller-rank mapping, empty when packet is ineligible.
  std::vector<int> peerPacketRankOrder;
  double peerResourceEstimateUs = -1;
  std::string topologyIdentity, executorIdentity, reason;
};
struct LocalPreparationOptions {
  ScalarType dtype = ScalarType::BFloat16;
  // Resource probing is collective, idle and outside any caller timing.
  bool probeResources = true;
  std::string hardwareProfile;
  CallerFallback fallback;
  // Collective policy choice, never inferred independently during enqueue.
  // This preparation creates no peer/host resources and performs no probes.
  bool fallbackOnly = false;
  bool allowPeer = true, allowHost = true;
  bool allowPeerPacket = true;
  // Family controls for diagnostics/policy; allowPeer gates both families.
  bool allowPeerDirect = true, allowPeerTree = true;
};
// BF16/FP32 sum and bitwise gather, one process per GPU. FP16 currently uses a
// caller fallback only. One instance has one immutable dtype. Direct peer
// execution needs a complete graph; caller-baseline preparation additionally
// qualifies source-ordered forwarding on a connected peer tree. Native/host
// plans remain available when peers are unavailable. This core API has no
// NCCL/MPI dependency. The caller supplies one shared random session ID and the
// same ordered local device list to every rank. Construction has no GPU/worker
// side effects. prepare is collective and must run on idle ranks outside
// capture. Each returned handle binds all ranks' layouts and one common
// backend. Serialized enqueue never probes, creates workers, or makes routing
// decisions. After a failed prepare, another successful collective prepare is
// required before any handle can execute. Native integrations require closed
// group scopes.
class PreparedLocalRuntime {
 public:
  PreparedLocalRuntime(BackendContext context,
                       LocalPreparationOptions options = {});
  ~PreparedLocalRuntime();
  PreparedLocalRuntime(PreparedLocalRuntime const&) = delete;
  PreparedLocalRuntime& operator=(PreparedLocalRuntime const&) = delete;
  size_t prepare(LocalRequest const& request);
  // Collective fixed-binding preparation, independently qualified in graph
  // mode. Buffers must remain valid until collective releaseGraph or teardown.
  // Old graphs can be rebuilt during later idle preparation.
  size_t prepareGraph(LocalRequest const& request, void const* input,
                      void* output);
  void launchGraph(size_t graph, cudaStream_t stream);
  void releaseGraph(size_t graph);
  LocalSelection const& graphSelection(size_t graph) const;
  void enqueue(size_t plan, void const* input, void* output,
               cudaStream_t stream);
  void synchronize();
  LocalSelection const& selection(size_t plan) const;
  bool hostInitialized() const;
  bool peerInitialized() const;
  bool peerPacketInitialized() const;
  bool peerTreeInitialized() const;
  bool idle() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
  size_t preparePlan(LocalRequest const& request, bool graph,
                     void const* input = nullptr, void* output = nullptr);
};
}  // namespace uccl_lite::runtime
