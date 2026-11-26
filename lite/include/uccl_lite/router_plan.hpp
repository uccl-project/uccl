#pragma once
#include "collective_dispatch.hpp"
#include <map>
#include <optional>
#include <tuple>

namespace uccl_lite {
struct RouterConfiguration {
  std::string name;
  cpu_router::CpuRouterOptions options;
};
// CPU-only candidate generation. Exact CPU IDs enter the normal dispatcher
// identity and must therefore have matching paired collective measurements.
std::vector<RouterConfiguration> routerWorkerConfigurations(
    Topology const& topology, cpu_router::CpuRouterOptions const& base,
    std::vector<int> const& workersPerNode);
struct RouterPlanChoice {
  std::optional<size_t> configuration;
  BackendDecision admission;
  // Advisory partial-reduction memory traffic only, not compute/merge/transport
  // latency and never sufficient for admission.
  std::optional<double> memoryPriorUs;
};
// Only the winning, already-admitted configuration is prepared. Metadata
// candidates never create GPU buffers or worker threads. Changing active
// configurations requires an explicit idle prepare boundary.
class RouterPlanDispatcher {
 public:
  using Fallback = CollectiveDispatcher::Fallback;
  RouterPlanDispatcher(Topology topology, BackendPolicy policy,
                       std::vector<RouterConfiguration> configurations,
                       Fallback fallback);
  RouterPlanChoice decision(CollectiveCall const& call) const;
  RouterPlanChoice prepare(CollectiveCall const& call);
  RouterPlanChoice enqueue(CollectiveCall const& call);
  Topology const& candidateTopology(size_t index) const;
  bool routerInitialized() const;

 private:
  Topology topology_;
  BackendPolicy policy_;
  std::vector<RouterConfiguration> configurations_;
  Fallback fallback_;
  std::unique_ptr<CollectiveDispatcher> validator_, active_;
  std::vector<std::unique_ptr<CollectiveDispatcher>> candidates_;
  std::optional<size_t> activeIndex_;
  using Key = std::tuple<Collective, ScalarType, size_t, bool, unsigned>;
  std::map<Key, RouterPlanChoice> prepared_;
  static Key key(CollectiveCall const& call);
  std::optional<double> memoryPrior(CollectiveCall const& call,
                                    size_t index) const;
};
}  // namespace uccl_lite
