#include "uccl_lite/network_plan.hpp"
#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>

namespace uccl_lite {
MeasuredNetworkPlan planMeasuredNetworkRails(
    NetworkProfile const& p, std::vector<NetworkEndpoint> const& endpoints,
    std::vector<std::pair<size_t, size_t>> const& peers) {
  MeasuredNetworkPlan result;
  auto unavailable = [&](char const* reason) {
    result.reason = reason;
    return result;
  };
  if (endpoints.empty() || endpoints.size() > 64 ||
      peers.size() * 2 != endpoints.size())
    throw std::invalid_argument("invalid measured network peer layout");
  std::array<NetworkHostIdentity, 2> current{};
  std::vector<int> hosts;
  std::set<std::pair<int, std::string>> gpus;
  for (auto const& e : endpoints) {
    int host = e.host.hostname == p.hosts[0].hostname   ? 0
               : e.host.hostname == p.hosts[1].hostname ? 1
                                                        : -1;
    if (host < 0 || e.host.topologyIdentity != p.hosts[host].topologyIdentity ||
        e.host.transportIdentity != p.hosts[host].transportIdentity)
      return unavailable(
          "live host/transport identity does not match network profile");
    if (e.gpu.empty() || e.device < 0 || !gpus.emplace(host, e.gpu).second)
      throw std::invalid_argument("invalid or duplicate network GPU endpoint");
    current[host] = e.host;
    hosts.push_back(host);
    if (e.candidates.empty())
      return unavailable("no complete physical bidirectional path");
  }
  std::vector<bool> used(endpoints.size());
  struct PairChoice {
    size_t a, b;
    double pathUs = 0;
    NetworkPathKey forward, reverse;
  };
  std::vector<std::vector<PairChoice>> choices(peers.size());
  std::set<size_t> commonSizes;
  bool first = true;
  for (size_t i = 0; i < peers.size(); ++i) {
    auto [a, b] = peers[i];
    if (a >= endpoints.size() || b >= endpoints.size() || a == b || used[a] ||
        used[b] || hosts[a] == hosts[b])
      throw std::invalid_argument("invalid cross-host network peer matching");
    used[a] = used[b] = true;
    for (size_t ca = 0; ca < endpoints[a].candidates.size(); ++ca)
      for (size_t cb = 0; cb < endpoints[b].candidates.size(); ++cb) {
        auto const& ra = endpoints[a].candidates[ca].assignment;
        auto const& rb = endpoints[b].candidates[cb].assignment;
        PairChoice choice{ca, cb};
        choice.forward = {hosts[a],         hosts[b], endpoints[a].gpu,
                          endpoints[b].gpu, ra.nic,   rb.nic,
                          ra.port,          rb.port};
        choice.reverse = {hosts[b],         hosts[a], endpoints[b].gpu,
                          endpoints[a].gpu, rb.nic,   ra.nic,
                          rb.port,          ra.port};
        for (auto const& key : {choice.forward, choice.reverse}) {
          auto it =
              std::find_if(p.paths.begin(), p.paths.end(),
                           [&](auto const& path) { return path.key == key; });
          if (it == p.paths.end() || !it->rejection.empty())
            return unavailable(
                "incomplete directed path evidence; retain physical plan");
          std::set<size_t> sizes;
          for (auto const& observation : it->sizes)
            sizes.insert(observation.bytes);
          if (first) {
            commonSizes = std::move(sizes);
            first = false;
          } else {
            std::set<size_t> intersection;
            std::set_intersection(
                commonSizes.begin(), commonSizes.end(), sizes.begin(),
                sizes.end(), std::inserter(intersection, intersection.end()));
            commonSizes = std::move(intersection);
          }
        }
        choices[i].push_back(std::move(choice));
      }
  }
  if (commonSizes.empty()) return unavailable("no common observed byte extent");
  size_t bytes = *commonSizes.rbegin();
  for (auto& pair : choices)
    for (auto& choice : pair) {
      auto a = estimateNetworkPath(p, current, choice.forward, bytes);
      auto b = estimateNetworkPath(p, current, choice.reverse, bytes);
      if (!a || !b) return unavailable("network measurements are unavailable");
      choice.pathUs = std::max(a->microseconds, b->microseconds);
    }
  std::map<std::string, double> capacities;
  auto resources = [&](size_t endpoint, size_t candidate) {
    std::vector<std::string> ids;
    auto const& e = endpoints[endpoint];
    auto const& c = e.candidates[candidate];
    if (c.assignment.device != e.device || c.assignment.transportIndex < 0 ||
        c.assignment.nic.empty() || c.assignment.port < 1 ||
        (!c.requiresMeasurement && (!std::isfinite(c.assignment.capacityGBs) ||
                                    c.assignment.capacityGBs <= 0)) ||
        c.transmit.empty() || c.receive.empty())
      throw std::invalid_argument("invalid measured rail candidate");
    for (auto const* map : {&c.transmit, &c.receive})
      for (auto const& [name, rate] : *map) {
        if (name.empty() || !std::isfinite(rate) || rate <= 0)
          throw std::invalid_argument("invalid network capacity bound");
        auto id = e.host.hostname + '/' + name;
        auto [it, added] = capacities.emplace(id, rate);
        if (!added && it->second != rate)
          throw std::invalid_argument("inconsistent shared network capacity");
        ids.push_back(std::move(id));
      }
    return ids;
  };
  std::vector<std::vector<std::vector<std::string>>> ids(endpoints.size());
  for (size_t i = 0; i < endpoints.size(); ++i)
    for (size_t c = 0; c < endpoints[i].candidates.size(); ++c)
      ids[i].push_back(resources(i, c));
  std::vector<size_t> order(peers.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return choices[a].size() < choices[b].size();
  });
  std::map<std::string, size_t> load;
  result.assignments.resize(endpoints.size());
  for (auto i : order) {
    auto [a, b] = peers[i];
    bool physicalPair = std::any_of(
        choices[i].begin(), choices[i].end(), [&](auto const& choice) {
          return !endpoints[a].candidates[choice.a].requiresMeasurement &&
                 !endpoints[b].candidates[choice.b].requiresMeasurement;
        });
    double best = std::numeric_limits<double>::infinity(), bound = 0;
    PairChoice const* selected = nullptr;
    std::map<std::string, size_t> selectedLoad;
    for (auto const& choice : choices[i]) {
      // Unknown bounds can compete through complete concurrent observations;
      // keep a physically supported prior when the pair has one.
      if (physicalPair &&
          (endpoints[a].candidates[choice.a].requiresMeasurement ||
           endpoints[b].candidates[choice.b].requiresMeasurement))
        continue;
      auto proposed = load;
      for (auto const& resource : ids[a][choice.a]) proposed[resource] += bytes;
      for (auto const& resource : ids[b][choice.b]) proposed[resource] += bytes;
      double physical = 0;
      for (auto const& [resource, volume] : proposed)
        physical =
            std::max(physical, double(volume) / capacities.at(resource) / 1000);
      double cost = std::max({result.isolatedPathUs, choice.pathUs, physical});
      if (cost < best) {
        best = cost;
        bound = physical;
        selected = &choice;
        selectedLoad = std::move(proposed);
      }
    }
    if (!selected) return unavailable("no supported measured peer path");
    result.assignments[a] = endpoints[a].candidates[selected->a].assignment;
    result.assignments[b] = endpoints[b].candidates[selected->b].assignment;
    result.isolatedPathUs = std::max(result.isolatedPathUs, selected->pathUs);
    result.capacityBoundUs = bound;
    load = std::move(selectedLoad);
  }
  result.calibrated = true;
  result.basisBytes = bytes;
  result.reason =
      "isolated path evidence plus shared physical bounds; concurrent "
      "performance unqualified";
  auto flowsFor = [&](std::vector<RailAssignment> const& assignments) {
    std::vector<NetworkPathKey> flows;
    for (auto [a, b] : peers) {
      auto const& ra = assignments[a];
      auto const& rb = assignments[b];
      flows.push_back({hosts[a], hosts[b], endpoints[a].gpu, endpoints[b].gpu,
                       ra.nic, rb.nic, ra.port, rb.port});
      flows.push_back({hosts[b], hosts[a], endpoints[b].gpu, endpoints[a].gpu,
                       rb.nic, ra.nic, rb.port, ra.port});
    }
    std::sort(flows.begin(), flows.end());
    return flows;
  };
  auto baselineFlows = flowsFor(result.assignments);
  auto baseline = estimateNetworkGroup(p, current, baselineFlows, bytes);
  if (!baseline) {
    for (size_t i = 0; i < endpoints.size(); ++i)
      for (auto const& candidate : endpoints[i].candidates)
        if (candidate.assignment.transportIndex ==
                result.assignments[i].transportIndex &&
            candidate.requiresMeasurement) {
          result.calibrated = false;
          result.assignments.clear();
          result.reason =
              "incomplete physical path requires exact concurrent-flow "
              "evidence";
          return result;
        }
    return result;
  }
  NetworkGroupObservation const* baselineGroup = nullptr;
  for (auto const& group : p.groups) {
    auto flows = group.flows;
    std::sort(flows.begin(), flows.end());
    if (flows == baselineFlows) baselineGroup = &group;
  }
  if (!baselineGroup)
    throw std::logic_error("matched concurrent baseline absent");
  auto baselineSamples = networkGroupSamples(*baselineGroup, bytes);
  double minimumBaseline =
      *std::min_element(baselineSamples.begin(), baselineSamples.end());
  result.concurrentMeasured = true;
  result.concurrentUs = baseline->microseconds;
  result.concurrentGroup = baselineGroup->label;
  for (auto const& group : p.groups) {
    if (!group.rejection.empty() || group.flows.size() != endpoints.size())
      continue;
    std::vector<int> chosen(endpoints.size(), -1);
    std::set<std::pair<size_t, size_t>> required;
    for (auto [a, b] : peers) {
      required.emplace(a, b);
      required.emplace(b, a);
    }
    bool supported = true;
    for (auto const& flow : group.flows) {
      auto endpoint = [&](int host, std::string const& gpu) {
        for (size_t i = 0; i < endpoints.size(); ++i)
          if (hosts[i] == host && endpoints[i].gpu == gpu) return i;
        return endpoints.size();
      };
      size_t a = endpoint(flow.sourceHost, flow.sourceGpu),
             b = endpoint(flow.destinationHost, flow.destinationGpu);
      if (a == endpoints.size() || b == endpoints.size() ||
          !required.erase({a, b})) {
        supported = false;
        break;
      }
      for (auto [index, nic, port] :
           {std::tuple<size_t, std::string, int>{a, flow.sourceNic,
                                                 flow.sourcePort},
            {b, flow.destinationNic, flow.destinationPort}}) {
        auto const& candidates = endpoints[index].candidates;
        auto found = std::find_if(
            candidates.begin(), candidates.end(), [&](auto const& c) {
              return c.assignment.nic == nic && c.assignment.port == port;
            });
        if (found == candidates.end()) {
          supported = false;
          break;
        }
        int c = found - candidates.begin();
        if (chosen[index] >= 0 && chosen[index] != c) {
          supported = false;
          break;
        }
        chosen[index] = c;
      }
      if (!supported) break;
    }
    if (!supported || !required.empty() ||
        std::find(chosen.begin(), chosen.end(), -1) != chosen.end())
      continue;
    auto measured = estimateNetworkGroup(p, current, group.flows, bytes);
    auto samples = networkGroupSamples(group, bytes);
    if (!measured || measured->microseconds >= result.concurrentUs ||
        measured->microseconds >= 0.97 * baseline->microseconds ||
        *std::max_element(samples.begin(), samples.end()) >= minimumBaseline)
      continue;
    for (size_t i = 0; i < endpoints.size(); ++i)
      result.assignments[i] = endpoints[i].candidates[chosen[i]].assignment;
    result.concurrentUs = measured->microseconds;
    result.concurrentGroup = group.label;
  }
  // Recompute the advisory diagnostics for the selected complete assignment;
  // do not leave the old isolated-plan numbers attached to a measured winner.
  result.isolatedPathUs = 0;
  result.capacityBoundUs = 0;
  std::map<std::string, size_t> selectedLoad;
  for (size_t i = 0; i < endpoints.size(); ++i) {
    auto const& candidates = endpoints[i].candidates;
    auto chosen =
        std::find_if(candidates.begin(), candidates.end(), [&](auto const& c) {
          return c.assignment.nic == result.assignments[i].nic &&
                 c.assignment.port == result.assignments[i].port;
        });
    for (auto const& id : ids[i][chosen - candidates.begin()])
      selectedLoad[id] += bytes;
  }
  for (auto const& [id, volume] : selectedLoad)
    result.capacityBoundUs = std::max(
        result.capacityBoundUs, double(volume) / capacities.at(id) / 1000);
  for (auto const& flow : flowsFor(result.assignments))
    result.isolatedPathUs =
        std::max(result.isolatedPathUs,
                 estimateNetworkPath(p, current, flow, bytes)->microseconds);
  result.reason =
      "exact concurrent flow cohort measured; GPU-driven collective "
      "performance remains unqualified";
  return result;
}
}  // namespace uccl_lite
