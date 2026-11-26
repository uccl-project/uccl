#include "uccl_lite/planner.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
using namespace uccl_lite;
void require(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
void near(double actual, double expected, char const* message) {
  require(std::abs(actual - expected) < 1e-8, message);
}
Topology topology() {
  Topology t;
  t.identity = "overlap-fixture";
  t.devices = {0, 1};
  t.hardware = {{"gpu0", HardwareKind::Gpu, 0, 0, {10, 0, Evidence::Measured}},
                {"gpu1", HardwareKind::Gpu, 1, 1, {10, 0, Evidence::Measured}},
                {"bridge", HardwareKind::PciBridge},
                {"nic", HardwareKind::Nic}};
  t.links = {{0, 2, "up", {10, 0, Evidence::Measured}},
             {2, 1, "down", {10, 0, Evidence::Measured}}};
  t.networkPorts = {{"ib", "pci", 1, 0, 3, true, {10, 0, Evidence::Measured}}};
  return t;
}
PlanCandidate candidate(Topology const& t) {
  PlanCandidate c;
  c.topologyIdentity = t.identity;
  c.workload = {Collective::AllReduce, ScalarType::Float32, 1000000, false, 4};
  c.capability.collectives = {Collective::AllReduce};
  c.capability.dtypes = {ScalarType::Float32};
  c.capability.minRanks = c.capability.maxRanks = 2;
  c.capability.alignment = 4;
  c.plan.id = "task-graph";
  c.plan.rankOrder = {0, 1};
  return c;
}
PlanEstimate estimate(Topology const& t, PlanCandidate const& c) {
  return choosePlan(t, c.workload, false, {c}).candidates[0];
}
PlanCandidate pipeline(Topology const& t, bool sharedMemory, bool oneResource) {
  auto c = candidate(t);
  for (size_t chunk = 0; chunk < 3; ++chunk)
    for (size_t stage = 0; stage < 3; ++stage) {
      PlanTask task;
      task.id = std::to_string(chunk) + ":" + std::to_string(stage);
      task.overheadUs = 0;
      task.demands = {{oneResource  ? "up"
                       : stage == 0 ? "memory:gpu0"
                       : stage == 1 ? "up"
                                    : "nic:ib:1",
                       1000000}};
      if (sharedMemory && stage == 1)
        task.demands.push_back({"memory:gpu0", 1000000});
      if (stage) task.dependencies.push_back(chunk * 3 + stage - 1);
      if (chunk) task.dependencies.push_back((chunk - 1) * 3 + stage);
      c.plan.tasks.push_back(std::move(task));
    }
  return c;
}
int main() {
  try {
    auto t = topology();
    auto c = candidate(t);
    c.plan.tasks = {{"a", {{"up", 1000000}}, {}, 0},
                    {"b", {{"down", 1000000}}, {}, 0}};
    auto e = estimate(t, c);
    require(e.estimated, "independent graph unavailable");
    near(e.microseconds, 100, "independent resources serialized");
    near(e.serialUs, 200, "serial comparison");
    c.plan.tasks[1].demands[0].resource = "up";
    e = estimate(t, c);
    near(e.microseconds, 200, "shared uplink double counted as independent");
    near(e.resourceLoadUs, 200, "aggregate resource load");
    c.plan.tasks[1].demands[0].bytes = 2000000;
    e = estimate(t, c);
    near(e.taskTimings[0].finishUs, 200, "unequal flow fairness");
    near(e.taskTimings[1].finishUs, 300,
         "remaining bandwidth not redistributed");
    c.plan.tasks[1].demands[0].bytes = 1000000;
    c.plan.tasks[1].overheadUs = 50;
    e = estimate(t, c);
    near(e.taskTimings[0].finishUs, 150, "startup held bandwidth");
    near(e.microseconds, 200, "delayed contender schedule");
    c.plan.tasks[1].overheadUs = 0;
    c.plan.tasks[1].dependencies = {0};
    e = estimate(t, c);
    near(e.microseconds, 200, "dependency ignored");
    near(e.dependencyOnlyUs, 200, "critical path incorrect");
    auto p = pipeline(t, false, false);
    e = estimate(t, p);
    near(e.microseconds, 500, "pipeline fill/steady/drain incorrect");
    near(e.serialUs, 900, "pipeline serial work");
    near(e.resourceLoadUs, 300, "pipeline resource load");
    near(e.dependencyOnlyUs, 500, "pipeline dependency path");
    auto contended = pipeline(t, true, false);
    near(estimate(t, contended).microseconds, 700,
         "shared GPU memory did not reduce overlap");
    auto serialized = pipeline(t, false, true);
    near(estimate(t, serialized).microseconds, 900,
         "fully shared pipeline exceeded bandwidth");
    // Node storage order must not become an artificial scheduling priority.
    auto reordered = p;
    size_t n = p.plan.tasks.size();
    std::reverse(reordered.plan.tasks.begin(), reordered.plan.tasks.end());
    for (auto& task : reordered.plan.tasks)
      for (auto& dependency : task.dependencies)
        dependency = n - 1 - dependency;
    near(estimate(t, reordered).microseconds, 500,
         "task indexing changed overlap");
    for (auto const& task : e.taskTimings)
      require(task.startUs >= 0 && task.finishUs >= task.startUs,
              "invalid task timeline");
    auto zero = p;
    zero.plan.tasks.push_back({"retire", {}, {2, 5, 8}, 0});
    near(estimate(t, zero).microseconds, 500,
         "dependency-only node changed cost");
    zero.plan.tasks.back().overheadUs = 7;
    near(estimate(t, zero).microseconds, 507, "known control delay lost");
    c.plan.tasks = {{"a", {{"up", 1000000}, {"down", 100000}}, {}, 0},
                    {"b", {{"up", 100000}, {"down", 1000000}}, {}, 0}};
    near(estimate(t, c).microseconds, 110,
         "coupled resource fairness violated capacity");
    auto bad = p;
    bad.plan.tasks[0].dependencies = {8};
    require(!estimate(t, bad).estimated, "cycle accepted");
    bad = p;
    bad.plan.tasks[0].dependencies = {99};
    require(!estimate(t, bad).estimated, "bad dependency index accepted");
    bad = p;
    bad.plan.tasks[1].dependencies = {0, 0};
    require(!estimate(t, bad).estimated, "duplicate dependency accepted");
    bad = p;
    bad.plan.tasks[1].id = bad.plan.tasks[0].id;
    require(!estimate(t, bad).estimated, "duplicate task id accepted");
    bad = p;
    bad.plan.tasks[0].overheadUs = -1;
    require(!estimate(t, bad).estimated, "unknown launch cost invented");
    bad = p;
    bad.plan.tasks[0].demands[0].resource = "absent";
    require(!estimate(t, bad).estimated, "missing resource predicted");
    auto unknown = estimate(t, bad);
    require(unknown.serialUs == -1 && unknown.dependencyOnlyUs == -1 &&
                unknown.resourceLoadUs == -1 && unknown.taskTimings.empty(),
            "unavailable model exposed partial costs as complete diagnostics");
    bad = p;
    bad.plan.stages = {{{{"up", 1}}, 0, 1}};
    require(!estimate(t, bad).estimated, "mixed models accepted");
    auto inactive = t;
    inactive.networkPorts[0].active = false;
    require(!estimate(inactive, p).estimated, "inactive NIC predicted");
    auto bound = t;
    bound.links[0].performance = {10, -1, Evidence::Advertised};
    c.plan.tasks = {{"bound", {{"up", 1000000, true}}, {}, 0}};
    require(!estimate(bound, c).estimated, "capacity-only task predicted");
    c.plan.tasks[0].demands.push_back({"memory:gpu0", 100000});
    near(estimate(bound, c).microseconds, 100, "physical bound not enforced");
    bound.hardware[0].memory = {};
    require(!estimate(bound, c).estimated, "unknown memory rate became free");
    serialized.plan.id = "serialized";
    p.plan.id = "pipelined";
    auto choice = choosePlan(t, p.workload, false, {serialized, p});
    require(choice.index == 1,
            "overlap model did not affect candidate ranking");
    std::mt19937 random(617);
    std::vector<std::string> names{"up", "down", "memory:gpu0", "memory:gpu1",
                                   "nic:ib:1"};
    for (int trial = 0; trial < 300; ++trial) {
      auto graph = candidate(t);
      size_t count = 1 + random() % 16;
      for (size_t i = 0; i < count; ++i) {
        PlanTask task;
        task.id = std::to_string(i);
        task.overheadUs = random() % 6;
        if (!i || random() % 4)
          for (size_t j = 0, demands = 1 + random() % 3; j < demands; ++j)
            task.demands.push_back(
                {names[random() % names.size()], 10000 * (1 + random() % 10)});
        std::set<size_t> predecessors;
        if (i)
          for (size_t j = 0; j < random() % 4; ++j)
            predecessors.insert(random() % i);
        task.dependencies.assign(predecessors.begin(), predecessors.end());
        graph.plan.tasks.push_back(std::move(task));
      }
      auto result = estimate(t, graph);
      require(result.estimated, "valid random DAG unestimated");
      std::vector<double> critical(count);
      std::map<std::string, double> resourceLoad;
      double serial = 0, criticalEnd = 0;
      for (size_t i = 0; i < count; ++i) {
        auto const& task = graph.plan.tasks[i];
        std::map<std::string, size_t> bytes;
        for (auto const& demand : task.demands) {
          bytes[demand.resource] += demand.bytes;
          resourceLoad[demand.resource] += double(demand.bytes) / 10000;
        }
        double isolated = task.overheadUs;
        size_t maximum = 0;
        for (auto const& entry : bytes)
          maximum = std::max(maximum, entry.second);
        isolated += double(maximum) / 10000;
        serial += isolated;
        for (size_t parent : task.dependencies) {
          critical[i] = std::max(critical[i], critical[parent]);
          require(result.taskTimings[i].startUs + 1e-8 >=
                      result.taskTimings[parent].finishUs,
                  "schedule violated a dependency");
        }
        critical[i] += isolated;
        criticalEnd = std::max(criticalEnd, critical[i]);
        require(result.taskTimings[i].finishUs - result.taskTimings[i].startUs +
                        1e-8 >=
                    isolated,
                "task exceeded isolated throughput");
      }
      double load = 0;
      for (auto const& entry : resourceLoad)
        load = std::max(load, entry.second);
      near(result.serialUs, serial, "random serial comparison");
      near(result.dependencyOnlyUs, criticalEnd,
           "random dependency comparison");
      near(result.resourceLoadUs, load, "random resource load");
      require(result.microseconds + 1e-8 >= std::max(criticalEnd, load) &&
                  result.microseconds <= serial + 1e-8,
              "schedule violates workload limits");
      auto reverse = graph;
      std::reverse(reverse.plan.tasks.begin(), reverse.plan.tasks.end());
      for (auto& task : reverse.plan.tasks)
        for (auto& parent : task.dependencies) parent = count - 1 - parent;
      near(estimate(t, reverse).microseconds, result.microseconds,
           "random task order changed result");
    }
    std::cout
        << "PASS: dependency graphs, pipeline fill/drain, shared "
           "memory/PCIe/NIC, fairness, invalid graphs and unknown rates\n";
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
