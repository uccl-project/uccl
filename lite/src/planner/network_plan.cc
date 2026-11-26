#include "uccl_lite/network_plan.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace uccl_lite {
namespace {
bool known(Performance const& p) {
  return (p.evidence == Evidence::Advertised ||
          p.evidence == Evidence::Measured) &&
         std::isfinite(p.bandwidthGBs) && p.bandwidthGBs > 0;
}
struct Route {
  RailAssignment assignment;
  std::map<std::string, double> capacities;
};
Route route(Topology const& t, size_t gpu, NetworkPort const& port, int rail) {
  size_t n = t.hardware.size();
  if (gpu >= n || port.node >= n || t.hardware[gpu].kind != HardwareKind::Gpu ||
      t.hardware[port.node].kind != HardwareKind::Nic)
    throw std::invalid_argument("invalid GPU/NIC hardware mapping");
  std::vector<double> width(n, 0);
  std::vector<size_t> hops(n, n), previous(n, t.links.size());
  std::vector<bool> done(n, false);
  width[gpu] = port.transmit.bandwidthGBs;
  hops[gpu] = 0;
  // Widest path, with shortest-hop tie breaking. Do not traverse other GPU or
  // NIC endpoints; a missing CPU/NUMA link stays missing.
  for (size_t step = 0; step < n; ++step) {
    size_t node = n;
    for (size_t i = 0; i < n; ++i)
      if (!done[i] && width[i] > 0 &&
          (node == n || width[i] > width[node] ||
           (width[i] == width[node] && hops[i] < hops[node])))
        node = i;
    if (node == n || node == port.node) break;
    done[node] = true;
    for (size_t i = 0; i < t.links.size(); ++i) {
      auto const& link = t.links[i];
      if (link.source >= n || link.destination >= n)
        throw std::invalid_argument("invalid PCI link endpoint");
      if (link.source != node || done[link.destination] ||
          !known(link.performance))
        continue;
      auto kind = t.hardware[link.destination].kind;
      if (link.destination != port.node &&
          (kind == HardwareKind::Gpu || kind == HardwareKind::Nic))
        continue;
      double candidate = std::min(width[node], link.performance.bandwidthGBs);
      if (candidate > width[link.destination] ||
          (candidate == width[link.destination] &&
           hops[node] + 1 < hops[link.destination])) {
        width[link.destination] = candidate;
        hops[link.destination] = hops[node] + 1;
        previous[link.destination] = i;
      }
    }
  }
  Route result;
  if (width[port.node] == 0) return result;
  result.assignment = {t.hardware[gpu].device, rail, port.name, port.port,
                       width[port.node],       {}};
  auto nic = "nic:" + port.name + ":" + std::to_string(port.port);
  result.capacities[nic] = port.transmit.bandwidthGBs;
  result.assignment.resources.push_back(nic);
  for (auto node = port.node; node != gpu;) {
    auto const& link = t.links.at(previous[node]);
    result.capacities[link.resource] = link.performance.bandwidthGBs;
    result.assignment.resources.push_back(link.resource);
    node = link.source;
  }
  return result;
}
}  // namespace
static void validateRails(Topology const& t,
                          std::vector<RailDevice> const& available) {
  if (t.devices.empty() || t.gpuNodes.size() != t.devices.size() ||
      std::set<int>(t.devices.begin(), t.devices.end()).size() !=
          t.devices.size())
    throw std::invalid_argument("invalid participating GPU set");
  for (size_t rank = 0; rank < t.devices.size(); ++rank)
    if (t.gpuNodes[rank] >= t.hardware.size() ||
        t.hardware[t.gpuNodes[rank]].device != t.devices[rank])
      throw std::invalid_argument(
          "GPU hardware mapping does not match participating devices");
  std::map<std::string, double> shared;
  for (auto const& link : t.links) {
    auto [it, added] =
        shared.emplace(link.resource, link.performance.bandwidthGBs);
    if (link.resource.empty() ||
        (!added && it->second != link.performance.bandwidthGBs))
      throw std::invalid_argument("inconsistent physical link capacity");
  }
  std::map<std::string, int> rails;
  std::set<int> indices;
  for (auto const& dev : available)
    if (dev.name.empty() || dev.transportIndex < 0 ||
        !rails.emplace(dev.name, dev.transportIndex).second ||
        !indices.insert(dev.transportIndex).second)
      throw std::invalid_argument("ambiguous transport rail mapping");
}
std::vector<RailAssignment> planNetworkRails(
    Topology const& t, std::vector<RailDevice> const& available) {
  validateRails(t, available);
  std::map<std::string, int> rails;
  for (auto const& dev : available) rails.emplace(dev.name, dev.transportIndex);
  std::map<std::string, size_t> active;
  for (auto const& port : t.networkPorts)
    if (port.active) ++active[port.name];
  std::vector<std::vector<Route>> routes(t.devices.size());
  for (size_t rank = 0; rank < t.devices.size(); ++rank) {
    for (auto const& port : t.networkPorts) {
      if (!port.active || active[port.name] != 1 || !known(port.transmit) ||
          !rails.count(port.name))
        continue;
      auto candidate = route(t, t.gpuNodes[rank], port, rails.at(port.name));
      if (candidate.assignment.transportIndex >= 0)
        routes[rank].push_back(std::move(candidate));
    }
    if (routes[rank].empty())
      throw std::invalid_argument(
          "no supported active GPU-to-NIC bandwidth path for device " +
          std::to_string(t.devices[rank]));
  }
  std::vector<size_t> order(t.devices.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  // Assign constrained GPUs first, preserving caller order for ties.
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return routes[a].size() < routes[b].size();
  });
  std::map<std::string, size_t> load;
  std::vector<RailAssignment> result(t.devices.size());
  for (auto rank : order) {
    Route const* best = nullptr;
    double bestCost = std::numeric_limits<double>::infinity();
    for (auto const& candidate : routes[rank]) {
      double cost = 0;
      for (auto const& [resource, bandwidth] : candidate.capacities)
        cost = std::max(cost, double(load[resource] + 1) / bandwidth);
      if (cost < bestCost ||
          (cost == bestCost && (!best || candidate.assignment.capacityGBs >
                                             best->assignment.capacityGBs))) {
        best = &candidate;
        bestCost = cost;
      }
    }
    for (auto const& [resource, bandwidth] : best->capacities) ++load[resource];
    result[rank] = best->assignment;
  }
  return result;
}

std::vector<std::vector<NetworkRailCandidate>> networkRailCandidates(
    Topology const& t, std::vector<RailDevice> const& available,
    bool includeUnmeasuredPhysicalPaths) {
  validateRails(t, available);
  if (!includeUnmeasuredPhysicalPaths) (void)planNetworkRails(t, available);
  std::map<std::string, int> active;
  for (auto const& port : t.networkPorts)
    if (port.active) ++active[port.name];
  std::vector<std::vector<NetworkRailCandidate>> result(t.devices.size());
  for (size_t rank = 0; rank < t.devices.size(); ++rank) {
    for (auto const& dev : available)
      for (auto const& port : t.networkPorts) {
        if (dev.name != port.name || !port.active || active[port.name] != 1 ||
            !known(port.transmit))
          continue;
        auto forward = route(t, t.gpuNodes[rank], port, dev.transportIndex);
        bool missing = forward.assignment.transportIndex < 0;
        if (missing && !includeUnmeasuredPhysicalPaths) continue;
        NetworkRailCandidate candidate;
        candidate.requiresMeasurement = missing;
        candidate.assignment = missing ? RailAssignment{t.devices[rank],
                                                        dev.transportIndex,
                                                        port.name,
                                                        port.port,
                                                        -1,
                                                        {}}
                                       : forward.assignment;
        auto nic = "nic:" + port.name + ":" + std::to_string(port.port);
        candidate.transmit[nic + ":tx"] = port.transmit.bandwidthGBs;
        candidate.receive[nic + ":rx"] = port.transmit.bandwidthGBs;
        bool valid = true;
        for (auto const& [resource, capacity] : forward.capacities) {
          if (resource == nic) continue;
          auto edge = std::find_if(
              t.links.begin(), t.links.end(),
              [&](auto const& link) { return link.resource == resource; });
          auto reverse = std::find_if(
              t.links.begin(), t.links.end(), [&](auto const& link) {
                return edge != t.links.end() &&
                       link.source == edge->destination &&
                       link.destination == edge->source &&
                       known(link.performance);
              });
          if (reverse == t.links.end()) {
            valid = false;
            break;
          }
          candidate.transmit[resource] = capacity;
          candidate.receive[reverse->resource] =
              reverse->performance.bandwidthGBs;
        }
        if (!valid && includeUnmeasuredPhysicalPaths) {
          candidate.requiresMeasurement = true;
          candidate.assignment.capacityGBs = -1;
          candidate.assignment.resources.clear();
          candidate.transmit = {{nic + ":tx", port.transmit.bandwidthGBs}};
          candidate.receive = {{nic + ":rx", port.transmit.bandwidthGBs}};
          valid = true;
        }
        if (valid) result[rank].push_back(std::move(candidate));
      }
    std::sort(result[rank].begin(), result[rank].end(),
              [](auto const& a, auto const& b) {
                return std::tie(a.assignment.nic, a.assignment.port) <
                       std::tie(b.assignment.nic, b.assignment.port);
              });
  }
  return result;
}
}  // namespace uccl_lite
