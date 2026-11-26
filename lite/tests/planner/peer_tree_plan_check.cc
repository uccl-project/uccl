#include "uccl_lite/local_plan.hpp"
#include "uccl_lite/peer_tree.hpp"
#include <cmath>
#include <iostream>
#include <numeric>
using namespace uccl_lite;
void require(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
Topology topology(int n, bool complete) {
  Topology t;
  t.identity = "tree-fixture";
  t.devices.resize(n);
  std::iota(t.devices.begin(), t.devices.end(), 10);
  t.gpuNodes.resize(n);
  std::iota(t.gpuNodes.begin(), t.gpuNodes.end(), 0);
  t.peerAccess.resize(n * n);
  for (int r = 0; r < n; ++r) {
    t.hardware.push_back({"gpu" + std::to_string(r),
                          HardwareKind::Gpu,
                          t.devices[r],
                          r % 2,
                          {100, 0, Evidence::Measured}});
    for (int p = 0; p < n; ++p)
      if (r != p && (complete || std::abs(r - p) == 1)) {
        t.peerAccess[r * n + p] = true;
        t.paths.push_back({size_t(r),
                           size_t(p),
                           PathKind::PeerRead,
                           "read" + std::to_string(r) + ":" + std::to_string(p),
                           {20, 1, Evidence::Measured}});
      }
  }
  return t;
}
int main() {
  try {
    Workload w{Collective::AllReduce, ScalarType::Float32, 4096, false, 4};
    for (int n : {2, 3, 4, 8, 32}) {
      auto t = topology(n, false);
      auto parents = peerTreeParents(t, w.bytes);
      require(parents.size() == size_t(n) && parents[0] == -1,
              "connected sparse graph rejected");
      for (int r = 1; r < n; ++r)
        require(parents[r] == r - 1, "path parent mismatch");
      for (int source = 0; source < n; ++source) {
        auto route = peerTreeRoute(parents, source);
        for (int r = 0; r < n; ++r)
          require(route.depth[r] == std::abs(r - source),
                  "rooting reordered or skipped logical rank");
      }
      auto candidate = treePeerCandidate(t, w, parents);
      auto e = choosePlan(t, w, false, {candidate}).candidates[0];
      require(
          candidate.plan.stages.empty() && candidate.plan.tasks.size() % n == 0,
          "tree still uses global serial phases");
      for (size_t i = 0; i < candidate.plan.tasks.size(); ++i) {
        auto const& task = candidate.plan.tasks[i];
        size_t phase = i / n, rank = i % n;
        std::vector<size_t> expected;
        if (phase) {
          expected.push_back(i - n);
          if (task.id.find("zero-accumulator:") != 0) {
            if (rank) expected.push_back((phase - 1) * n + rank - 1);
            if (rank + 1 < size_t(n))
              expected.push_back((phase - 1) * n + rank + 1);
          }
        }
        auto actual = task.dependencies;
        std::sort(actual.begin(), actual.end());
        std::sort(expected.begin(), expected.end());
        require(actual == expected,
                "tree model does not match neighbor retirement/stream "
                "dependencies");
      }
      require(e.eligible && e.estimated &&
                  candidate.plan.peerEdges.size() == size_t(2 * (n - 1)),
              "tree resource model unavailable");
      if (n > 2)
        require(!choosePlan(t, w, false, {stagedPeerCandidate(t, w)})
                     .candidates[0]
                     .eligible,
                "sparse fixture accidentally full mesh");
      t.peerAccess[1] = false;
      require(peerTreeParents(t, w.bytes).empty(),
              "one-way/disconnected tree admitted");
      require(!choosePlan(t, w, false, {candidate}).candidates[0].eligible,
              "cached required edge not revalidated");
    }
    auto t = topology(4, true);
    auto first = peerTreeParents(t, w.bytes);
    require(std::count(first.begin(), first.end(), 0) == 3,
            "equal-cost deterministic star changed");
    t.hardware[0].memory.bandwidthGBs = 0.01;
    auto memory = peerTreeParents(t, w.bytes);
    require(std::count(memory.begin(), memory.end(), 0) == 1,
            "GPU memory bottleneck did not change tree");
    t = topology(4, true);
    for (auto& path : t.paths)
      if (path.source == 0 || path.destination == 0)
        path.performance.bandwidthGBs = 0.01;
    auto bandwidth = peerTreeParents(t, w.bytes);
    require(std::count(bandwidth.begin(), bandwidth.end(), 0) == 1,
            "peer bandwidth bottleneck did not change tree");
    t.paths.clear();
    for (auto& node : t.hardware) node.memory = {};
    auto unknown = treePeerCandidate(t, w, peerTreeParents(t, w.bytes));
    auto e = choosePlan(t, w, false, {unknown}).candidates[0];
    require(e.eligible && !e.estimated,
            "unknown bandwidth became a prediction");
    require(selectPeerTreeParents(t, w) == peerTreeParents(t, w.bytes),
            "unknown overlap costs replaced the capability prior");
    auto chainPrior = topology(4, true);
    for (auto& path : chainPrior.paths)
      if (std::abs(int(path.source) - int(path.destination)) == 1)
        path.performance.bandwidthGBs = 40;
    auto edgeTree = peerTreeParents(chainPrior, w.bytes);
    auto overlapTree = selectPeerTreeParents(chainPrior, w);
    auto edgeCost = choosePlan(chainPrior, w, false,
                               {treePeerCandidate(chainPrior, w, edgeTree)})
                        .candidates[0];
    auto overlapCost =
        choosePlan(chainPrior, w, false,
                   {treePeerCandidate(chainPrior, w, overlapTree)})
            .candidates[0];
    require(edgeTree != overlapTree && overlapCost.estimated &&
                overlapCost.microseconds < edgeCost.microseconds,
            "dependency/overlap model did not change the execution tree");
    bool bad = false;
    try {
      peerTreeRoute({-1, 2, 1}, 0);
    } catch (std::invalid_argument const&) {
      bad = true;
    }
    require(bad, "disconnected parent cycle accepted");
    std::array<std::vector<double>, 4> base{
        {{10, 10, 10}, {10, 10, 10}, {10, 10, 10}, {10, 10, 10}}};
    auto candidate = base;
    candidate[3] = {5, 5, 5};
    require(qualifiedFallbackAlternative(base, candidate,
                                         {true, true, true, true}) ==
                LocalBackend::PeerTree,
            "fourth candidate ignored by exact qualification");
    require(qualifiedFallbackAlternative(base, candidate,
                                         {true, true, true, false}) ==
                LocalBackend::Fallback,
            "incorrect tree admitted");
    std::cout
        << "PASS: sparse 2/3/4/8/32-rank trees, exact edges, "
           "bandwidth/memory-driven priors, unknown rates and qualification\n";
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
