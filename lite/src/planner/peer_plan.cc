#include "uccl_lite/peer_plan.hpp"
#include "uccl_lite/local_plan.hpp"
#include "uccl_lite/peer_tree.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <tuple>

namespace uccl_lite {
namespace {
class Model {
 public:
  Topology const& t;
  PlanStage stage;
  void memory(int rank, size_t bytes) {
    if (!bytes) return;
    std::string resource = "missing:gpu-memory";
    if (size_t(rank) < t.gpuNodes.size() &&
        t.gpuNodes[rank] < t.hardware.size())
      resource = "memory:" + t.hardware[t.gpuNodes[rank]].id;
    stage.demands.push_back({resource, bytes});
  }
  void wire(int source, int destination, size_t bytes,
            PathKind kind = PathKind::PeerRead) {
    if (!bytes || source == destination) return;
    if (t.gpuNodes.size() != t.devices.size()) {
      stage.demands.push_back({"missing:peer-path", bytes});
      return;
    }
    size_t start = t.gpuNodes[source], end = t.gpuNodes[destination];
    auto path = std::find_if(
        t.paths.begin(), t.paths.end(), [&](HardwarePath const& p) {
          return p.source == start && p.destination == end && p.kind == kind;
        });
    bool measuredPath = path != t.paths.end();
    if (measuredPath) stage.demands.push_back({path->resource, bytes});
    std::vector<size_t> prev(t.hardware.size(), t.links.size());
    if (start >= prev.size() || end >= prev.size())
      throw std::invalid_argument("invalid GPU hardware mapping");
    std::vector<bool> seen(prev.size(), false);
    std::queue<size_t> visit;
    visit.push(start);
    seen[start] = true;
    while (!visit.empty() && !seen[end]) {
      auto node = visit.front();
      visit.pop();
      for (size_t i = 0; i < t.links.size(); ++i) {
        auto const& link = t.links[i];
        if (link.destination >= seen.size() || link.source >= seen.size())
          throw std::invalid_argument("invalid hardware link");
        if (link.source != node || seen[link.destination]) continue;
        // Physical PCIe/NUMA paths may traverse bridges and CPUs, not another
        // GPU or NIC endpoint. Transport capability is checked independently.
        auto kind = t.hardware[link.destination].kind;
        if (link.destination != end &&
            (kind == HardwareKind::Gpu || kind == HardwareKind::Nic))
          continue;
        seen[link.destination] = true;
        prev[link.destination] = i;
        visit.push(link.destination);
      }
    }
    if (!seen[end]) {
      if (!measuredPath)
        stage.demands.push_back({"missing:peer-path", bytes});
      else {
        // A measured end-to-end path may cross NUMA domains whose CPU link
        // capacity is unknown. Still account for each known PCIe ingress and
        // egress cut; do not fabricate a measured CPU/interconnect capacity.
        for (auto [node, reverse] :
             {std::pair<size_t, bool>{start, false}, {end, true}}) {
          for (size_t step = 0; step < t.hardware.size(); ++step) {
            auto it = std::find_if(
                t.links.begin(), t.links.end(), [&](HardwareLink const& link) {
                  return link.source == node &&
                         link.destination < t.hardware.size() &&
                         link.resource == t.hardware[node].id + "->" +
                                              t.hardware[link.destination].id;
                });
            if (it == t.links.end()) break;
            stage.demands.push_back(
                {it->resource + (reverse ? ":reverse" : ""), bytes, true});
            node = it->destination;
            if (t.hardware[node].kind == HardwareKind::Numa) break;
          }
        }
      }
      return;
    }
    for (auto node = end; node != start;) {
      auto const& link = t.links[prev[node]];
      if (!measuredPath || link.performance.evidence != Evidence::Unknown)
        stage.demands.push_back({link.resource, bytes, measuredPath});
      node = link.source;
    }
  }
  void copy(int source, int dest, size_t bytes) {
    memory(source, bytes);
    memory(dest, bytes);
    wire(source, dest, bytes, PathKind::PeerCopy);
  }
  void flush(ExecutionPlan& plan) {
    stage.overheadUs = 0;
    plan.stages.push_back(std::move(stage));
    stage = {};
  }
};
}  // namespace

std::vector<int> peerTreeParents(Topology const& t, size_t basisBytes) {
  size_t n = t.devices.size();
  if (n < 2 || n > 32 || !basisBytes || t.peerAccess.size() != n * n) return {};
  auto cost = [&](int a, int b) -> double {
    double total = 0;
    if (t.gpuNodes.size() != n) return INFINITY;
    for (auto [source, destination] : {std::pair<int, int>{a, b}, {b, a}}) {
      size_t s = t.gpuNodes[source], d = t.gpuNodes[destination];
      if (s >= t.hardware.size() || d >= t.hardware.size()) return INFINITY;
      auto path =
          std::find_if(t.paths.begin(), t.paths.end(), [&](auto const& p) {
            return p.source == s && p.destination == d &&
                   p.kind == PathKind::PeerRead;
          });
      if (path == t.paths.end()) return INFINITY;
      auto const& rate = path->performance;
      auto valid = [](Performance const& p) {
        return p.evidence == Evidence::Measured &&
               std::isfinite(p.bandwidthGBs) && p.bandwidthGBs > 0;
      };
      if (!valid(rate) || !std::isfinite(rate.latencyUs) ||
          rate.latencyUs < 0 || !valid(t.hardware[s].memory) ||
          !valid(t.hardware[d].memory))
        return INFINITY;
      double bound = std::max(
          double(basisBytes) / t.hardware[s].memory.bandwidthGBs / 1000,
          double(basisBytes) / t.hardware[d].memory.bandwidthGBs / 1000);
      total += std::max(
          rate.latencyUs + double(basisBytes) / rate.bandwidthGBs / 1000,
          bound);
    }
    return total;
  };
  std::vector<int> parents(n, -1);
  std::vector<bool> included(n);
  included[0] = true;
  for (size_t step = 1; step < n; ++step) {
    std::tuple<double, int, int, int> best{INFINITY, 2, int(n), int(n)};
    int from = -1, to = -1;
    for (size_t a = 0; a < n; ++a)
      if (included[a])
        for (size_t b = 0; b < n; ++b)
          if (!included[b] && t.peerAccess[a * n + b] &&
              t.peerAccess[b * n + a]) {
            int placement = t.numaNodes.size() == n && t.numaNodes[a] >= 0 &&
                                    t.numaNodes[a] == t.numaNodes[b]
                                ? 0
                                : 1;
            auto candidate =
                std::make_tuple(double(cost(a, b)), placement, int(a), int(b));
            if (candidate < best) {
              best = candidate;
              from = a;
              to = b;
            }
          }
    if (to < 0) return {};
    parents[to] = from;
    included[to] = true;
  }
  return parents;
}
PlanCandidate treePeerCandidate(Topology const& t, Workload const& w,
                                std::vector<int> const& parents) {
  PlanCandidate c;
  c.workload = w;
  c.topologyIdentity = t.identity;
  c.capability.collectives = {Collective::AllReduce, Collective::AllGather,
                              Collective::ReduceScatter};
  c.capability.dtypes = {ScalarType::BFloat16, ScalarType::Float32};
  c.capability.minRanks = 2;
  c.capability.maxRanks = 32;
  c.capability.maxBytes = 1UL << 30;
  c.capability.alignment = scalarBytes(w.dtype);
  c.capability.inPlace = true;
  c.plan.id = "peer-tree-source-ordered-v1/dag-v1";
  c.plan.algorithm = Algorithm::Tree;
  c.plan.transport = Transport::Peer;
  c.plan.protocol = Protocol::LoadStore;
  size_t n = t.devices.size();
  if (parents.size() != n || n < 2 || n > 32 || !w.bytes ||
      w.bytes > (1UL << 30))
    return c;
  try {
    (void)peerTreeRoute(parents, 0);
  } catch (std::invalid_argument const&) {
    return c;
  }
  c.plan.rankOrder.resize(n);
  std::iota(c.plan.rankOrder.begin(), c.plan.rankOrder.end(), 0);
  for (size_t r = 1; r < n; ++r) {
    c.plan.peerEdges.emplace_back(r, parents[r]);
    c.plan.peerEdges.emplace_back(parents[r], r);
    c.plan.id += ":" + std::to_string(parents[r]);
  }
  size_t scalar = scalarBytes(w.dtype);
  if (!scalar) return c;
  size_t input = w.collective == Collective::AllGather ? w.bytes / n : w.bytes;
  size_t output =
      w.collective == Collective::ReduceScatter ? w.bytes / n : w.bytes;
  size_t accumulator = output / scalar * sizeof(float);
  std::vector<std::vector<int>> neighbors(n);
  for (size_t r = 1; r < n; ++r) {
    neighbors[r].push_back(parents[r]);
    neighbors[parents[r]].push_back(int(r));
  }
  std::vector<size_t> previous;
  auto phase = [&](std::string const& name, auto populate, bool neighborWait) {
    size_t first = c.plan.tasks.size();
    for (size_t rank = 0; rank < n; ++rank) {
      Model model{t, {}};
      populate(rank, model);
      PlanTask task;
      task.id = name + ':' + std::to_string(rank);
      task.demands = std::move(model.stage.demands);
      task.overheadUs = 0;  // movement-only prior; launch/polling is qualified
      if (!previous.empty()) {
        task.dependencies.push_back(previous[rank]);
        if (neighborWait)
          for (int peer : neighbors[rank])
            task.dependencies.push_back(previous[peer]);
      }
      c.plan.tasks.push_back(std::move(task));
    }
    previous.resize(n);
    for (size_t rank = 0; rank < n; ++rank) previous[rank] = first + rank;
  };
  phase(
      "copy-input", [&](size_t r, Model& model) { model.memory(r, 2 * input); },
      false);
  if (w.collective != Collective::AllGather)
    phase(
        "zero-accumulator",
        [&](size_t r, Model& model) { model.memory(r, accumulator); }, false);
  for (size_t source = 0; source < n; ++source) {
    auto route = peerTreeRoute(parents, source);
    for (int depth = 1; depth <= route.height; ++depth)
      phase(
          "forward-" + std::to_string(source) + '-' + std::to_string(depth),
          [&](size_t r, Model& model) {
            if (route.depth[r] != depth) return;
            model.memory(route.parent[r], input);
            model.memory(r, input);
            model.wire(route.parent[r], r, input);
          },
          true);
    phase(
        "consume-" + std::to_string(source),
        [&](size_t r, Model& model) {
          model.memory(r, w.collective == Collective::AllGather
                              ? 2 * input
                              : output + 2 * accumulator);
        },
        true);
  }
  if (w.collective != Collective::AllGather)
    phase(
        "finish",
        [&](size_t r, Model& model) { model.memory(r, accumulator + output); },
        true);
  else
    phase(
        "retire", [](size_t, Model&) {}, true);
  return c;
}

std::vector<int> selectPeerTreeParents(Topology const& t, Workload const& w) {
  auto seed = peerTreeParents(t, w.bytes);
  if (seed.empty()) return {};
  size_t n = seed.size();
  std::vector<std::vector<int>> trees{seed};
  std::set<std::vector<int>> unique{seed};
  for (size_t root = 0; root < n; ++root) {
    std::vector<std::vector<int>> edges(n);
    std::vector<bool> seen(n);
    std::queue<size_t> queue;
    queue.push(root);
    seen[root] = true;
    while (!queue.empty()) {
      size_t rank = queue.front();
      queue.pop();
      for (size_t peer = 0; peer < n; ++peer)
        if (!seen[peer] && t.peerAccess[rank * n + peer] &&
            t.peerAccess[peer * n + rank]) {
          seen[peer] = true;
          queue.push(peer);
          edges[rank].push_back(peer);
          edges[peer].push_back(rank);
        }
    }
    std::fill(seen.begin(), seen.end(), false);
    queue.push(0);
    seen[0] = true;
    std::vector<int> parents(n, -1);
    while (!queue.empty()) {
      size_t rank = queue.front();
      queue.pop();
      for (int peer : edges[rank])
        if (!seen[peer]) {
          seen[peer] = true;
          parents[peer] = rank;
          queue.push(peer);
        }
    }
    if (unique.insert(parents).second) trees.push_back(std::move(parents));
  }
  std::vector<PlanCandidate> candidates;
  for (auto const& tree : trees)
    candidates.push_back(treePeerCandidate(t, w, tree));
  auto choice = choosePlan(t, w, false, candidates);
  return choice.index ? trees[*choice.index] : seed;
}

PlanCandidate stagedPeerCandidate(Topology const& t, Workload const& w) {
  PlanCandidate c;
  c.workload = w;
  c.topologyIdentity = t.identity;
  c.capability.collectives = {Collective::AllReduce, Collective::AllGather,
                              Collective::ReduceScatter};
  c.capability.dtypes = {ScalarType::BFloat16, ScalarType::Float32};
  c.capability.minRanks = 2;
  c.capability.maxRanks = 32;
  c.capability.maxBytes = 1UL << 30;
  c.capability.alignment = scalarBytes(w.dtype);
  c.capability.inPlace = true;
  c.capability.allPairsPeer = true;
  c.plan.id = "peer-staged-v1";
  c.plan.algorithm = Algorithm::Direct;
  c.plan.transport = Transport::Peer;
  c.plan.protocol = Protocol::LoadStore;
  size_t n = t.devices.size();
  c.plan.rankOrder.resize(n);
  std::iota(c.plan.rankOrder.begin(), c.plan.rankOrder.end(), 0);
  if (!n || n > 32 || !w.bytes || w.bytes > (1UL << 30)) return c;
  Model model{t, {}};
  size_t shard = w.bytes / n;
  size_t input = w.collective == Collective::AllGather ? shard : w.bytes;
  for (size_t rank = 0; rank < n; ++rank) model.memory(rank, 2 * input);
  model.flush(c.plan);
  size_t read = w.collective == Collective::AllReduce ? w.bytes : shard;
  for (size_t dest = 0; dest < n; ++dest) {
    for (size_t source = 0; source < n; ++source) {
      model.memory(source, read);
      model.wire(source, dest, read);
    }
    model.memory(dest,
                 w.collective == Collective::ReduceScatter ? shard : w.bytes);
  }
  model.flush(c.plan);
  return c;
}

std::vector<PlanCandidate> peerCandidates(Topology const& t, Workload const& w,
                                          PeerTuning const& tuning) {
  size_t n = t.devices.size();
  if (!n || n > kPeerRankCapacity || !w.bytes) return {};
  size_t shard = w.bytes / n;
  std::vector<PlanCandidate> result;
  for (int family = 0; family < 3; ++family) {
    PlanCandidate c;
    c.workload = w;
    c.topologyIdentity = t.identity;
    c.capability.collectives = {Collective::AllReduce, Collective::AllGather,
                                Collective::ReduceScatter};
    c.capability.dtypes = {ScalarType::BFloat16};
    c.capability.minRanks = family == 0 ? 1 : 2;
    c.capability.maxRanks = kPeerRankCapacity;
    c.capability.alignment = 2;
    c.capability.allPairsPeer = true;
    if (family == 2 && w.collective == Collective::AllReduce)
      c.capability.extentMultiple = 2 * n;
    c.plan.id = std::array<char const*, 3>{
        {"peer-direct-v1", "peer-coordinator-v1", "peer-bulk-v1"}}[family];
    c.plan.algorithm = family == 1 ? Algorithm::Coordinator : Algorithm::Direct;
    c.plan.protocol = family == 2 ? Protocol::BulkCopy : Protocol::LoadStore;
    c.plan.rankOrder.resize(n);
    std::iota(c.plan.rankOrder.begin(), c.plan.rankOrder.end(), 0);
    Model model{t, {}};
    if (family == 0) {
      size_t read = w.collective == Collective::AllReduce ? w.bytes : shard;
      for (size_t dest = 0; dest < n; ++dest) {
        for (size_t source = 0; source < n; ++source) {
          model.memory(source, read);
          model.wire(source, dest, read);
        }
        model.memory(
            dest, w.collective == Collective::ReduceScatter ? shard : w.bytes);
      }
      model.flush(c.plan);
    } else if (family == 1) {
      size_t read = w.collective == Collective::AllGather ? shard : w.bytes;
      for (size_t source = 0; source < n; ++source) {
        model.memory(source, read);
        model.wire(source, 0, read);
      }
      for (size_t dest = 0; dest < n; ++dest) {
        size_t write =
            w.collective == Collective::ReduceScatter ? shard : w.bytes;
        model.memory(dest, write);
        model.wire(0, dest, write, PathKind::PeerWrite);
      }
      model.flush(c.plan);
    } else {
      bool gather = w.collective == Collective::AllGather;
      for (size_t phase = 0; phase < n; ++phase) {
        if (!gather && n == 2 && phase == 0) continue;
        for (size_t source = 0; source < n; ++source)
          model.copy(source, (source + phase) % n, shard);
        model.flush(c.plan);
      }
      if (!gather) {
        for (size_t dest = 0; dest < n; ++dest) {
          // All source shards are read in order from destination-local scratch
          // (two-rank self reads come directly from its local input).
          model.memory(dest, w.bytes);
          model.memory(dest, shard);
        }
        model.flush(c.plan);
        if (w.collective == Collective::AllReduce) {
          for (size_t phase = 1; phase < n; ++phase) {
            for (size_t source = 0; source < n; ++source)
              model.copy(source, (source + phase) % n, shard);
            model.flush(c.plan);
          }
        }
      }
    }
    if (!c.plan.stages.empty())
      c.plan.stages.front().overheadUs = tuning.overheadUs[family];
    result.push_back(std::move(c));
  }
  return result;
}

PeerSelection selectPeerPlan(Topology const& t, Workload const& w,
                             PeerTuning const& tuning) {
  auto candidates = peerCandidates(t, w, tuning);
  auto choices = choosePlan(t, w, false, candidates);
  size_t index = 0;
  if (tuning.validationFamily) {
    index = static_cast<size_t>(*tuning.validationFamily);
    if (index >= candidates.size() || !choices.candidates[index].eligible)
      throw std::invalid_argument(
          "diagnostic peer family does not support this call");
  } else if (choices.index)
    index = *choices.index;
  else if (candidates.empty() || !choices.candidates[0].eligible)
    throw std::invalid_argument("no supported peer execution plan");
  return {
      static_cast<PeerFamily>(index), std::move(candidates[index].plan),
      !tuning.validationFamily && bool(choices.index),
      tuning.validationFamily ? "explicit correctness/calibration selection"
      : choices.index
          ? choices.candidates[index].reason
          : "direct baseline: incomplete local resource/overhead calibration"};
}
}  // namespace uccl_lite
