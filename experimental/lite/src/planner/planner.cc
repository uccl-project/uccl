#include "uccl_lite/planner.hpp"
#include "overlap_model.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace uccl_lite {
namespace {
using Resources = std::map<std::string, Performance>;
bool valid(Performance const& p, bool allowAdvertised) {
  return (p.evidence == Evidence::Measured ||
          (allowAdvertised && p.evidence == Evidence::Advertised)) &&
         std::isfinite(p.bandwidthGBs) && p.bandwidthGBs > 0 &&
         std::isfinite(p.latencyUs) && p.latencyUs >= 0;
}
Resources resources(Topology const& t) {
  Resources result;
  auto add = [&](std::string const& id, Performance const& p) {
    auto [it, inserted] = result.emplace(id, p);
    if (id.empty() ||
        (!inserted && (p.bandwidthGBs != it->second.bandwidthGBs ||
                       p.latencyUs != it->second.latencyUs ||
                       p.evidence != it->second.evidence)))
      throw std::invalid_argument("inconsistent shared resource: " + id);
  };
  for (auto const& link : t.links) {
    if (link.source >= t.hardware.size() ||
        link.destination >= t.hardware.size())
      throw std::invalid_argument("link endpoint outside topology");
    add(link.resource, link.performance);
  }
  for (auto const& path : t.paths) {
    if (path.source >= t.hardware.size() ||
        path.destination >= t.hardware.size())
      throw std::invalid_argument("path endpoint outside topology");
    add(path.resource, path.performance);
  }
  for (auto const& node : t.hardware) add("memory:" + node.id, node.memory);
  for (auto const& port : t.networkPorts)
    add("nic:" + port.name + ":" + std::to_string(port.port),
        port.active ? port.transmit : Performance{});
  return result;
}
template <class T>
bool contains(std::vector<T> const& list, T const& value) {
  return std::find(list.begin(), list.end(), value) != list.end();
}
std::string supported(Topology const& t, Workload const& w, bool capture,
                      PlanCandidate const& candidate) {
  auto const& c = candidate.capability;
  size_t n = t.devices.size();
  if (!n || std::set<int>(t.devices.begin(), t.devices.end()).size() != n)
    return "invalid participating devices";
  if (!contains(c.collectives, w.collective) || !contains(c.dtypes, w.dtype))
    return "collective or dtype unsupported";
  if (n < c.minRanks || n > c.maxRanks) return "rank extent unsupported";
  size_t scalar = w.dtype == ScalarType::Float32 ? 4 : 2;
  size_t shards = w.collective == Collective::AllReduce ? 1 : n;
  if (!w.bytes || w.bytes % scalar || (w.bytes / scalar) % shards ||
      w.bytes < c.minBytes || w.bytes > c.maxBytes || !c.extentMultiple ||
      w.bytes % c.extentMultiple)
    return "message extent unsupported";
  if (!w.alignment || !c.alignment || (w.alignment & (w.alignment - 1)) ||
      (c.alignment & (c.alignment - 1)) || w.alignment < c.alignment)
    return "buffer alignment unsupported";
  if (w.inPlace && !c.inPlace) return "in-place unsupported";
  if (capture && !c.capture) return "capture unsupported";
  if (c.allPairsPeer) {
    if (t.peerAccess.size() != n * n) return "peer matrix missing";
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < n; ++j)
        if (i != j && !t.peerAccess[i * n + j]) return "peer path unavailable";
  }
  for (auto [accessor, peer] : candidate.plan.peerEdges) {
    if (accessor < 0 || peer < 0 || accessor == peer || size_t(accessor) >= n ||
        size_t(peer) >= n || t.peerAccess.size() != n * n ||
        !t.peerAccess[size_t(accessor) * n + peer])
      return "required peer edge unavailable";
  }
  if (c.mappedHost && (t.mappedHost.size() != n || t.numaNodes.size() != n ||
                       std::find(t.mappedHost.begin(), t.mappedHost.end(),
                                 false) != t.mappedHost.end() ||
                       std::any_of(t.numaNodes.begin(), t.numaNodes.end(),
                                   [](int node) { return node < 0; })))
    return "mapped host placement unavailable";
  auto const& modeled = candidate.workload;
  if (candidate.topologyIdentity.empty() ||
      candidate.topologyIdentity != t.identity)
    return "execution model topology mismatch";
  if (modeled.collective != w.collective || modeled.dtype != w.dtype ||
      modeled.bytes != w.bytes || modeled.inPlace != w.inPlace ||
      modeled.alignment != w.alignment)
    return "execution model workload mismatch";
  auto order = candidate.plan.rankOrder;
  std::sort(order.begin(), order.end());
  if (order.size() != n) return "rank schedule incomplete";
  for (size_t i = 0; i < n; ++i)
    if (order[i] != int(i)) return "rank schedule is not a permutation";
  if (!candidate.plan.groups.empty()) {
    std::vector<int> grouped;
    for (auto const& group : candidate.plan.groups) {
      if (group.empty()) return "empty rank group";
      grouped.insert(grouped.end(), group.begin(), group.end());
    }
    std::sort(grouped.begin(), grouped.end());
    if (grouped != order) return "rank groups are not a partition";
  }
  if (candidate.plan.id.empty() || !candidate.plan.channels)
    return "invalid plan identity or channels";
  return {};
}
PlanEstimate estimate(Resources const& capacity, ExecutionPlan const& plan,
                      bool allowAdvertised) {
  if (!plan.tasks.empty())
    return estimateTaskGraph(capacity, plan, allowAdvertised);
  PlanEstimate result;
  result.eligible = true;
  double total = 0;
  bool hasModel = false;
  if (plan.stages.empty()) {
    result.reason = "execution model missing";
    return result;
  }
  for (auto const& stage : plan.stages) {
    if (!std::isfinite(stage.overheadUs) || stage.overheadUs < 0 ||
        !stage.repetitions || stage.demands.empty()) {
      result.reason = "invalid stage";
      return result;
    }
    std::map<std::string, size_t> bytes;
    std::map<std::string, bool> onlyBound;
    for (auto const& demand : stage.demands) {
      auto [bound, inserted] =
          onlyBound.emplace(demand.resource, demand.physicalCapacityBound);
      if (!inserted) bound->second &= demand.physicalCapacityBound;
      auto& sum = bytes[demand.resource];
      if (!demand.bytes ||
          demand.bytes > std::numeric_limits<size_t>::max() - sum) {
        result.reason = "invalid resource demand";
        return result;
      }
      sum += demand.bytes;
    }
    double bottleneck = 0;
    for (auto const& [resource, load] : bytes) {
      auto it = capacity.find(resource);
      bool bound = onlyBound.at(resource);
      bool available =
          it != capacity.end() &&
          (bound ? ((it->second.evidence == Evidence::Measured ||
                     it->second.evidence == Evidence::Advertised) &&
                    std::isfinite(it->second.bandwidthGBs) &&
                    it->second.bandwidthGBs > 0)
                 : valid(it->second, allowAdvertised));
      if (!available) {
        result.reason = "performance unavailable: " + resource;
        return result;
      }
      auto const& p = it->second;
      hasModel |= !bound;
      bottleneck =
          std::max(bottleneck, (bound ? 0 : p.latencyUs) +
                                   double(load) / p.bandwidthGBs / 1000);
    }
    total += (stage.overheadUs + bottleneck) * stage.repetitions;
  }
  if (!hasModel) {
    result.reason = "physical bounds alone are not a performance model";
    return result;
  }
  if (!std::isfinite(total)) {
    result.reason = "cost overflow";
    return result;
  }
  result.estimated = true;
  result.microseconds = total;
  result.reason = "resource bottleneck model";
  return result;
}
}  // namespace

PlanChoice choosePlan(Topology const& t, Workload const& w, bool capture,
                      std::vector<PlanCandidate> const& candidates,
                      bool allowAdvertised) {
  auto capacity = resources(t);
  PlanChoice result;
  std::set<std::string> ids;
  double best = std::numeric_limits<double>::infinity();
  for (auto const& candidate : candidates) {
    if (!ids.insert(candidate.plan.id).second)
      throw std::invalid_argument("duplicate plan identity");
    auto reason = supported(t, w, capture, candidate);
    PlanEstimate cost;
    if (reason.empty())
      cost = estimate(capacity, candidate.plan, allowAdvertised);
    else
      cost.reason = reason;
    if (cost.estimated && cost.microseconds < best) {
      best = cost.microseconds;
      result.index = result.candidates.size();
    }
    result.candidates.push_back(std::move(cost));
  }
  return result;
}

void applyHardwareCalibration(Topology& t, HardwareCalibration const& c) {
  auto const& hardwareIdentity =
      t.hardwareIdentity.empty() ? t.identity : t.hardwareIdentity;
  if (hardwareIdentity.empty() || c.topologyIdentity != hardwareIdentity ||
      c.identity.empty() || c.resources.empty())
    throw std::invalid_argument(
        "missing or stale hardware calibration identity");
  auto known = resources(t);
  std::map<std::string, Performance> updates;
  for (auto const& r : c.resources) {
    if (!known.count(r.resource) || !valid(r.performance, false) ||
        r.repetitions < 3 || !updates.emplace(r.resource, r.performance).second)
      throw std::invalid_argument("invalid hardware calibration resource");
  }
  auto next = t;
  for (auto& link : next.links)
    if (updates.count(link.resource))
      link.performance = updates.at(link.resource);
  for (auto& path : next.paths)
    if (updates.count(path.resource))
      path.performance = updates.at(path.resource);
  for (auto& node : next.hardware) {
    auto key = "memory:" + node.id;
    if (updates.count(key)) node.memory = updates.at(key);
  }
  for (auto& port : next.networkPorts) {
    auto key = "nic:" + port.name + ":" + std::to_string(port.port);
    if (updates.count(key)) {
      if (!port.active)
        throw std::invalid_argument("cannot calibrate inactive NIC");
      port.transmit = updates.at(key);
    }
  }
  for (auto const& r : c.resources)
    next.resourceCalibrationIdentities[r.resource] = c.identity;
  auto calibrated = resources(next);
  std::ostringstream identity;
  identity << "uccl-lite-resource-calibration-v1" << std::setprecision(17);
  for (auto const& [key, source] : next.resourceCalibrationIdentities) {
    auto const& value = calibrated.at(key);
    identity << '\n'
             << std::quoted(key) << ' ' << std::quoted(source) << ' '
             << value.bandwidthGBs << ' ' << value.latencyUs;
  }
  next.calibrationIdentity = identity.str();
  t = std::move(next);
}
}  // namespace uccl_lite
