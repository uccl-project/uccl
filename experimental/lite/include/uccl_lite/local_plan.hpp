#pragma once
#include "planner.hpp"
#include <array>

namespace uccl_lite {
enum class LocalBackend { Peer, Host, Fallback, PeerPacket, PeerTree };
inline constexpr std::array<LocalBackend, 4> kLocalAlternatives{
    {LocalBackend::Peer, LocalBackend::Host, LocalBackend::PeerPacket,
     LocalBackend::PeerTree}};
// Minimum spanning tree on bidirectional CUDA-access edges. Measured read and
// GPU-memory costs guide the prior; unknown edges remain unestimated. Root=0,
// parent[0]=-1; an empty vector means no supported connected graph.
std::vector<int> peerTreeParents(Topology const& topology, size_t basisBytes);
// Compare the measured-edge tree and deduplicated BFS trees using the task
// graph cost. If none is estimated, retain the capability-aware edge prior.
std::vector<int> selectPeerTreeParents(Topology const& topology,
                                       Workload const& basis);
PlanCandidate treePeerCandidate(Topology const& topology,
                                Workload const& workload,
                                std::vector<int> const& parents);
// Resource-only estimate for the staged peer implementation. Zero stage
// overhead is intentional: launch, arithmetic and semaphore cost are omitted.
// This advisory estimate never admits an alternative without exact timings.
PlanCandidate stagedPeerCandidate(Topology const& topology,
                                  Workload const& workload);
bool peerPacketEligible(Topology const& topology, Workload const& workload);
// Execution-rank -> caller-rank permutation for two physical NUMA pairs.
// Empty means the retained algorithm cannot represent this topology.
std::vector<int> pairedNumaRankOrder(Topology const& topology);
// Exact-workload admission from complete ABBA samples. The alternative must
// be correct, beat its paired baseline median by 3%, and have nonoverlapping
// sample ranges. The legacy two-way helper returns Host to denote a win.
LocalBackend qualifiedLocalBackend(std::vector<double> const& peer,
                                   std::vector<double> const& host,
                                   bool hostCorrect);
LocalBackend qualifiedFallbackAlternative(
    std::array<std::vector<double>, 2> const& pairedBaselines,
    std::array<std::vector<double>, 2> const& candidates,
    std::array<bool, 2> const& correct);
LocalBackend qualifiedFallbackAlternative(
    std::array<std::vector<double>, 3> const& pairedBaselines,
    std::array<std::vector<double>, 3> const& candidates,
    std::array<bool, 3> const& correct);
LocalBackend qualifiedFallbackAlternative(
    std::array<std::vector<double>, 4> const& pairedBaselines,
    std::array<std::vector<double>, 4> const& candidates,
    std::array<bool, 4> const& correct);
}  // namespace uccl_lite
