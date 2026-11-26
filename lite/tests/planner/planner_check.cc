#include "uccl_lite/planner.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace uccl_lite;
namespace {
int checks = 0;
void require(bool condition, char const* message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}
template <class F>
void rejects(F f, char const* message) {
  bool rejected = false;
  try {
    f();
  } catch (std::invalid_argument const&) {
    rejected = true;
  }
  require(rejected, message);
}
Topology topology() {
  Topology t;
  t.identity = "test-topology-software";
  t.devices = {3, 0};
  t.peerAccess = {false, true, true, false};
  t.hardware = {{"gpu3", HardwareKind::Gpu},
                {"gpu0", HardwareKind::Gpu},
                {"root", HardwareKind::PciBridge},
                {"host", HardwareKind::Numa},
                {"nic", HardwareKind::Nic}};
  t.links = {{0, 2, "uplink", {10, 1, Evidence::Measured}},
             {1, 2, "uplink", {10, 1, Evidence::Measured}},
             {2, 0, "downlink", {10, 1, Evidence::Measured}}};
  t.hardware[0].memory = {100, 0, Evidence::Measured};
  t.hardware[3].memory = {20, 0, Evidence::Measured};
  t.networkPorts = {
      {"ib0", "test", 1, 0, 4, true, {25, 2, Evidence::Measured}}};
  return t;
}
PlanCandidate plan(std::string id, std::vector<PlanStage> stages) {
  PlanCandidate c;
  c.workload = {Collective::AllReduce, ScalarType::BFloat16, 1000000, false,
                16};
  c.topologyIdentity = "test-topology-software";
  c.capability.collectives = {Collective::AllReduce, Collective::AllGather};
  c.capability.dtypes = {ScalarType::BFloat16};
  c.capability.minRanks = c.capability.maxRanks = 2;
  c.capability.alignment = 2;
  c.capability.allPairsPeer = true;
  c.plan.id = id;
  c.plan.rankOrder = {1, 0};
  c.plan.stages = std::move(stages);
  return c;
}
}  // namespace
int main() {
  try {
    auto t = topology();
    Workload w{Collective::AllReduce, ScalarType::BFloat16, 1000000, false, 16};
    auto shared =
        plan("shared", {{{{"uplink", 1000000}, {"uplink", 1000000}}, 3, 1}});
    auto duplex =
        plan("duplex", {{{{"uplink", 1000000}, {"downlink", 1000000}}, 3, 1}});
    auto result = choosePlan(t, w, false, {shared, duplex});
    require(result.index == 1, "full duplex beats shared uplink contention");
    require(result.candidates[0].microseconds == 204 &&
                result.candidates[1].microseconds == 104,
            "aggregate shared demands, overlap independent resources");
    auto boundTopology = t;
    boundTopology.links[0].performance = {10, -1, Evidence::Advertised};
    boundTopology.links[1].performance = boundTopology.links[0].performance;
    auto boundPlan =
        plan("physical-bound",
             {{{{"uplink", 2000000, true}, {"memory:gpu3", 1000000}}, 0, 1}});
    auto bounded = choosePlan(boundTopology, w, false, {boundPlan});
    require(bounded.index == 0 && bounded.candidates[0].microseconds == 200,
            "shared advertised bandwidth cut constrains measured traffic "
            "without inventing link latency");
    boundPlan.plan.stages[0].demands.pop_back();
    require(!choosePlan(boundTopology, w, false, {boundPlan}).index,
            "physical capacity bounds alone cannot select an algorithm");
    auto pipeline = duplex;
    pipeline.plan.id = "pipeline";
    pipeline.plan.stages[0].repetitions = 3;
    pipeline.plan.stages.push_back({{{"memory:host", 1000000}}, 2, 1});
    require(
        choosePlan(t, w, false, {pipeline}).candidates[0].microseconds == 364,
        "serial pipeline groups and drain are added");
    auto memory = plan("memory", {{{{"memory:gpu3", 1000000}}, 1, 1}});
    auto network = plan("network", {{{{"nic:ib0:1", 1000000}}, 1, 1}});
    result = choosePlan(t, w, false, {network, memory});
    require(result.index == 1, "GPU-memory and NIC bandwidth affect ranking");
    t.hardware[0].memory.bandwidthGBs = 1;
    require(choosePlan(t, w, false, {network, memory}).index == 0,
            "memory bandwidth change reverses choice without GPU-count change");
    t.networkPorts[0].active = false;
    require(!choosePlan(t, w, false, {network}).index,
            "inactive NIC cannot win");
    t.networkPorts[0].active = true;
    t.networkPorts[0].transmit.evidence = Evidence::Advertised;
    require(!choosePlan(t, w, false, {network}).index,
            "advertised rates require explicit model opt-in");
    require(choosePlan(t, w, false, {network}, true).index == 0,
            "explicit advertised modeling");
    t.networkPorts[0].transmit.latencyUs = -1;
    require(!choosePlan(t, w, false, {network}, true).index,
            "unknown latency is not free");
    t = topology();
    require(!choosePlan(t, w, true, {duplex}).candidates[0].eligible,
            "capture capability filter");
    w.inPlace = true;
    require(!choosePlan(t, w, false, {duplex}).candidates[0].eligible,
            "in-place capability filter");
    w.inPlace = false;
    w.dtype = ScalarType::Float32;
    require(!choosePlan(t, w, false, {duplex}).index, "dtype filter");
    w.dtype = ScalarType::BFloat16;
    w.alignment = 1;
    require(!choosePlan(t, w, false, {duplex}).index, "alignment filter");
    w.alignment = 16;
    t.peerAccess[1] = false;
    require(!choosePlan(t, w, false, {duplex}).index,
            "same GPU count with unavailable P2P rejected");
    t = topology();
    auto otherSize = w;
    otherSize.bytes *= 2;
    require(!choosePlan(t, otherSize, false, {duplex}).index,
            "resource volumes cannot be reused for a different message size");
    auto otherTopology = t;
    otherTopology.identity = "different-physical-order";
    require(!choosePlan(otherTopology, w, false, {duplex}).index,
            "schedule cannot be reused for another topology identity");
    auto invalid = duplex;
    invalid.plan.rankOrder = {0, 0};
    require(!choosePlan(t, w, false, {invalid}).index,
            "duplicate rank mapping rejected");
    invalid = duplex;
    invalid.plan.groups = {{0}, {0}};
    require(!choosePlan(t, w, false, {invalid}).index,
            "invalid group partition rejected");
    invalid = duplex;
    invalid.plan.stages[0].demands[0].resource = "missing";
    require(!choosePlan(t, w, false, {invalid}).index,
            "missing resource rejected");
    invalid.plan.stages[0].overheadUs = std::nan("");
    require(!choosePlan(t, w, false, {invalid}).index,
            "non-finite stage rejected");
    auto inconsistent = t;
    inconsistent.links[1].performance.bandwidthGBs = 20;
    rejects([&] { choosePlan(inconsistent, w, false, {duplex}); },
            "shared resource capacities must agree");
    rejects(
        [&] {
          choosePlan(t, w, false, {duplex, duplex});
        },
        "plan IDs must be unique");
    HardwareCalibration cal{t.identity,
                            "probe-build-and-workload",
                            {{"uplink", {5, 2, Evidence::Measured}, 3}}};
    applyHardwareCalibration(t, cal);
    require(t.links[0].performance.bandwidthGBs == 5 &&
                t.links[1].performance.bandwidthGBs == 5,
            "calibration updates every view of a shared resource");
    require(choosePlan(t, w, false, {shared}).candidates[0].microseconds == 405,
            "calibrated properties drive costs");
    auto firstIdentity = t.calibrationIdentity;
    applyHardwareCalibration(t, cal);
    require(t.calibrationIdentity == firstIdentity,
            "identical calibration keeps stable cache identity");
    cal.resources[0].performance.latencyUs = 3;
    applyHardwareCalibration(t, cal);
    require(
        t.calibrationIdentity != firstIdentity,
        "changed values invalidate cached plans even under same source label");
    auto previousSource = t.resourceCalibrationIdentities.at("uplink");
    HardwareCalibration memoryCal{
        t.identity,
        "memory-probe",
        {{"memory:gpu3", {200, 0, Evidence::Measured}, 3}}};
    applyHardwareCalibration(t, memoryCal);
    require(
        t.resourceCalibrationIdentities.at("uplink") == previousSource &&
            t.resourceCalibrationIdentities.at("memory:gpu3") == "memory-probe",
        "partial calibration preserves other resource provenance");
    cal.topologyIdentity = "stale";
    rejects([&] { applyHardwareCalibration(t, cal); },
            "stale topology calibration rejected");
    cal.topologyIdentity = t.identity;
    cal.resources.push_back({"nonexistent", {3, 0, Evidence::Measured}, 3});
    cal.resources[0].performance.bandwidthGBs = 100;
    rejects([&] { applyHardwareCalibration(t, cal); },
            "malformed resource rejected");
    require(t.links[0].performance.bandwidthGBs == 5,
            "failed calibration is atomic");
    std::cout
        << "Planner: " << checks
        << " capability, contention, bandwidth and calibration checks passed\n";
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
