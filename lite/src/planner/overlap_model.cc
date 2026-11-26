#include "overlap_model.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
#include <map>
#include <set>

namespace uccl_lite {
namespace {
constexpr double epsilon = 1e-12;
struct Task {
  std::vector<std::pair<size_t, double>> service;
  double startup = 0, solo = 0, remaining = 1, readyAt = 0;
  int state = 0;  // 0 blocked, 1 starting, 2 moving data, 3 complete
};
}  // namespace
PlanEstimate estimateTaskGraph(
    std::map<std::string, Performance> const& capacity,
    ExecutionPlan const& plan, bool allowAdvertised) {
  PlanEstimate result;
  result.eligible = true;
  auto unavailable = [&](std::string reason) {
    result.reason = std::move(reason);
    result.taskTimings.clear();
    result.serialUs = result.dependencyOnlyUs = result.resourceLoadUs = -1;
    return result;
  };
  size_t n = plan.tasks.size();
  if (!plan.stages.empty() || !n || n > 65536)
    return unavailable("invalid or mixed task-graph model");
  std::vector<Task> tasks(n);
  std::vector<std::vector<size_t>> successors(n);
  std::vector<size_t> pending(n);
  std::map<std::string, size_t> resources;
  std::set<std::string> ids;
  std::vector<double> loads;
  size_t dependencies = 0, demands = 0;
  bool measured = false;
  result.serialUs = 0;
  result.dependencyOnlyUs = 0;
  result.resourceLoadUs = 0;
  for (size_t i = 0; i < n; ++i) {
    auto const& node = plan.tasks[i];
    auto& task = tasks[i];
    if (node.id.empty() || !ids.insert(node.id).second ||
        !std::isfinite(node.overheadUs) || node.overheadUs < 0)
      return unavailable("invalid task identity or overhead");
    dependencies += node.dependencies.size();
    demands += node.demands.size();
    if (dependencies > (1UL << 20) || demands > (1UL << 20))
      return unavailable("task graph exceeds model budget");
    std::set<size_t> unique;
    for (size_t predecessor : node.dependencies) {
      if (predecessor >= n || predecessor == i ||
          !unique.insert(predecessor).second)
        return unavailable("invalid task dependency");
      successors[predecessor].push_back(i);
    }
    pending[i] = unique.size();
    std::map<std::string, std::pair<size_t, bool>> traffic;
    for (auto const& d : node.demands) {
      auto [it, inserted] =
          traffic.emplace(d.resource, std::make_pair(size_t(0), true));
      auto& aggregate = it->second;
      if (d.resource.empty() || !d.bytes ||
          d.bytes > std::numeric_limits<size_t>::max() - aggregate.first)
        return unavailable("invalid task resource demand");
      aggregate.first += d.bytes;
      aggregate.second &= d.physicalCapacityBound;
    }
    bool operational = false;
    double latency = 0;
    for (auto const& [name, load] : traffic) {
      auto found = capacity.find(name);
      bool bound = load.second;
      if (found == capacity.end())
        return unavailable("performance unavailable: " + name);
      auto const& p = found->second;
      bool valid =
          p.evidence == Evidence::Measured ||
          ((bound || allowAdvertised) && p.evidence == Evidence::Advertised);
      if (!valid || !std::isfinite(p.bandwidthGBs) || p.bandwidthGBs <= 0 ||
          (!bound && (!std::isfinite(p.latencyUs) || p.latencyUs < 0)))
        return unavailable("performance unavailable: " + name);
      double service = double(load.first) / p.bandwidthGBs / 1000;
      if (!std::isfinite(service) || service <= 0)
        return unavailable("task service overflow or underflow");
      auto [entry, added] = resources.emplace(name, resources.size());
      if (added) loads.push_back(0);
      task.service.emplace_back(entry->second, service);
      loads[entry->second] += service;
      task.solo = std::max(task.solo, service);
      if (!bound) {
        operational = true;
        latency = std::max(latency, p.latencyUs);
      }
    }
    if (!traffic.empty() && !operational)
      return unavailable(
          "task physical bounds alone are not a performance model");
    measured |= operational;
    task.startup = node.overheadUs + latency;
    if (!std::isfinite(task.startup) ||
        !std::isfinite(task.solo + task.startup) ||
        (task.solo && !std::isfinite(1 / task.solo)))
      return unavailable("task duration overflow");
    result.serialUs += task.startup + task.solo;
  }
  if (!measured)
    return unavailable("task graph has no operational resource model");
  // Independently compute the dependency-only critical path and detect cycles.
  auto incoming = pending;
  std::deque<size_t> queue;
  std::vector<double> ends(n);
  for (size_t i = 0; i < n; ++i)
    if (!incoming[i]) queue.push_back(i);
  size_t visited = 0;
  while (!queue.empty()) {
    size_t i = queue.front();
    queue.pop_front();
    ++visited;
    ends[i] += tasks[i].startup + tasks[i].solo;
    result.dependencyOnlyUs = std::max(result.dependencyOnlyUs, ends[i]);
    for (size_t child : successors[i]) {
      ends[child] = std::max(ends[child], ends[i]);
      if (!--incoming[child]) queue.push_back(child);
    }
  }
  if (visited != n) return unavailable("cyclic task dependencies");
  for (double load : loads)
    result.resourceLoadUs = std::max(result.resourceLoadUs, load);
  if (!std::isfinite(result.serialUs) ||
      !std::isfinite(result.dependencyOnlyUs) ||
      !std::isfinite(result.resourceLoadUs))
    return unavailable("task graph cost overflow");
  result.taskTimings.resize(n);
  std::deque<size_t> ready;
  for (size_t i = 0; i < n; ++i)
    if (!pending[i]) ready.push_back(i);
  std::vector<size_t> live;
  size_t finished = 0, work = 0;
  double now = 0;
  auto complete = [&](size_t i) {
    tasks[i].state = 3;
    ++finished;
    result.taskTimings[i].finishUs = now;
    for (size_t child : successors[i])
      if (!--pending[child]) ready.push_back(child);
  };
  while (finished < n) {
    while (!ready.empty()) {
      size_t i = ready.front();
      ready.pop_front();
      auto& task = tasks[i];
      result.taskTimings[i].startUs = now;
      task.readyAt = now + task.startup;
      if (!std::isfinite(task.readyAt) ||
          (task.startup > 0 && task.readyAt <= now))
        return unavailable("task startup precision exhausted");
      if (!task.startup && task.service.empty())
        complete(i);
      else {
        task.state = task.startup ? 1 : 2;
        live.push_back(i);
      }
    }
    if (finished == n) break;
    std::vector<size_t> active;
    double next = INFINITY;
    for (size_t i : live) {
      if (tasks[i].state == 1)
        next = std::min(next, tasks[i].readyAt);
      else if (tasks[i].state == 2)
        active.push_back(i);
    }
    // Progressive filling: max-min fair fractions of each task's isolated
    // progress rate, bounded by every shared bandwidth resource.
    std::vector<double> fractions(active.size()), used(resources.size());
    std::vector<bool> free(active.size(), true);
    size_t unfrozen = active.size();
    while (unfrozen) {
      std::vector<double> slope(resources.size());
      double delta = INFINITY;
      for (size_t a = 0; a < active.size(); ++a)
        if (free[a]) {
          auto const& t = tasks[active[a]];
          delta = std::min(delta, 1 - fractions[a]);
          for (auto [r, service] : t.service) {
            slope[r] += service / t.solo;
            if (++work > 50000000)
              return unavailable("task sharing model budget exceeded");
          }
        }
      for (size_t r = 0; r < slope.size(); ++r)
        if (slope[r] > 0)
          delta = std::min(delta, std::max(0.0, 1 - used[r]) / slope[r]);
      if (!std::isfinite(delta) || delta < 0)
        return unavailable("invalid sharing increment");
      for (size_t a = 0; a < active.size(); ++a)
        if (free[a]) fractions[a] += delta;
      for (size_t r = 0; r < slope.size(); ++r) used[r] += delta * slope[r];
      size_t previous = unfrozen;
      for (size_t a = 0; a < active.size(); ++a)
        if (free[a]) {
          bool saturated = fractions[a] >= 1 - epsilon;
          for (auto [r, service] : tasks[active[a]].service)
            saturated |= used[r] >= 1 - epsilon;
          if (saturated) {
            free[a] = false;
            --unfrozen;
          }
        }
      if (previous == unfrozen)
        return unavailable("sharing model made no progress");
    }
    std::vector<double> rates(active.size()), deadlines(active.size());
    for (size_t a = 0; a < active.size(); ++a) {
      auto const& t = tasks[active[a]];
      rates[a] = fractions[a] / t.solo;
      if (!std::isfinite(rates[a]) || rates[a] <= 0)
        return unavailable("invalid task progress rate");
      deadlines[a] = now + t.remaining / rates[a];
      next = std::min(next, deadlines[a]);
    }
    if (!std::isfinite(next) || next <= now)
      return unavailable("task schedule precision exhausted");
    double step = next - now;
    now = next;
    for (size_t a = 0; a < active.size(); ++a) {
      size_t i = active[a];
      tasks[i].remaining = std::max(0.0, tasks[i].remaining - step * rates[a]);
      if (deadlines[a] <= now || tasks[i].remaining <= epsilon) complete(i);
    }
    for (size_t i : live)
      if (tasks[i].state == 1 && tasks[i].readyAt <= now) {
        if (tasks[i].service.empty())
          complete(i);
        else
          tasks[i].state = 2;
      }
    live.erase(std::remove_if(live.begin(), live.end(),
                              [&](size_t i) { return tasks[i].state == 3; }),
               live.end());
  }
  result.estimated = true;
  result.microseconds = now;
  result.reason = "dependency graph with normalized fair bandwidth sharing";
  return result;
}
}  // namespace uccl_lite
