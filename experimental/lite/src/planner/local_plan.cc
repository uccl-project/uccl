#include "uccl_lite/local_plan.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace uccl_lite {
std::vector<int> pairedNumaRankOrder(Topology const& t) {
  if (t.devices.size() != 4 || t.numaNodes.size() != 4) return {};
  std::map<int, std::vector<int>> groups;
  for (int rank = 0; rank < 4; ++rank) {
    if (t.numaNodes[rank] < 0) return {};
    groups[t.numaNodes[rank]].push_back(rank);
  }
  if (groups.size() != 2) return {};
  std::vector<int> order;
  for (auto const& group : groups) {
    if (group.second.size() != 2) return {};
    order.insert(order.end(), group.second.begin(), group.second.end());
  }
  return order;
}
bool peerPacketEligible(Topology const& t, Workload const& w) {
  auto order = pairedNumaRankOrder(t);
  if (order.empty()) return false;
  PlanCandidate c;
  c.workload = w;
  c.topologyIdentity = t.identity;
  c.capability.collectives = {Collective::AllReduce, Collective::AllGather,
                              Collective::ReduceScatter};
  c.capability.dtypes = {ScalarType::Float32};
  c.capability.minRanks = c.capability.maxRanks = 4;
  c.capability.maxBytes = 1UL << 30;
  c.capability.extentMultiple = 128;
  c.capability.alignment = 4;
  c.capability.allPairsPeer = true;
  c.capability.mappedHost = true;
  c.capability.inPlace = true;
  c.plan.id = "peer-packet-capability-v1";
  c.plan.rankOrder = order;
  c.plan.groups = {{order[0], order[1]}, {order[2], order[3]}};
  return choosePlan(t, w, false, {c}).candidates[0].eligible;
}
LocalBackend qualifiedLocalBackend(std::vector<double> const& peer,
                                   std::vector<double> const& host,
                                   bool correct) {
  auto valid = [](auto const& values) {
    return values.size() >= 3 &&
           std::all_of(values.begin(), values.end(), [](double value) {
             return std::isfinite(value) && value > 0;
           });
  };
  if (!valid(peer))
    throw std::invalid_argument("invalid peer baseline samples");
  if (!correct || !valid(host)) return LocalBackend::Peer;
  auto median = [](auto values) {
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
  };
  return median(host) < 0.97 * median(peer) &&
                 *std::max_element(host.begin(), host.end()) <
                     *std::min_element(peer.begin(), peer.end())
             ? LocalBackend::Host
             : LocalBackend::Peer;
}
template <size_t N>
LocalBackend fallbackAlternative(
    std::array<std::vector<double>, N> const& baselines,
    std::array<std::vector<double>, N> const& candidates,
    std::array<bool, N> const& correct) {
  LocalBackend winner = LocalBackend::Fallback;
  double best = INFINITY;
  for (size_t i = 0; i < N; ++i) {
    if (!correct[i] || qualifiedLocalBackend(baselines[i], candidates[i],
                                             true) != LocalBackend::Host)
      continue;
    auto values = candidates[i];
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    double median =
        n % 2 ? values[n / 2] : values[n / 2 - 1] / 2 + values[n / 2] / 2;
    if (median < best) {
      best = median;
      winner = kLocalAlternatives[i];
    }
  }
  return winner;
}
LocalBackend qualifiedFallbackAlternative(
    std::array<std::vector<double>, 2> const& b,
    std::array<std::vector<double>, 2> const& c,
    std::array<bool, 2> const& good) {
  return fallbackAlternative(b, c, good);
}
LocalBackend qualifiedFallbackAlternative(
    std::array<std::vector<double>, 3> const& b,
    std::array<std::vector<double>, 3> const& c,
    std::array<bool, 3> const& good) {
  return fallbackAlternative(b, c, good);
}
LocalBackend qualifiedFallbackAlternative(
    std::array<std::vector<double>, 4> const& b,
    std::array<std::vector<double>, 4> const& c,
    std::array<bool, 4> const& good) {
  return fallbackAlternative(b, c, good);
}
}  // namespace uccl_lite
