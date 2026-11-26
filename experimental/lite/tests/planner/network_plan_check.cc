#include "uccl_lite/network_plan.hpp"
#include <iostream>
#include <stdexcept>
using namespace uccl_lite;
namespace {
void require(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
template <class F>
void rejects(F f, char const* message) {
  bool failed = false;
  try {
    f();
  } catch (std::invalid_argument const&) {
    failed = true;
  }
  require(failed, message);
}
void link(Topology& t, size_t a, size_t b, std::string id, double bw) {
  t.links.push_back({a, b, id, {bw, -1, Evidence::Advertised}});
  t.links.push_back({b, a, id + ":reverse", {bw, -1, Evidence::Advertised}});
}
Topology fixture() {
  Topology t;
  t.devices = {2, 0, 3, 1};
  t.gpuNodes = {0, 1, 2, 3};
  for (size_t i = 0; i < 4; ++i)
    t.hardware.push_back({"gpu" + std::to_string(t.devices[i]),
                          HardwareKind::Gpu, t.devices[i], t.devices[i] / 2});
  t.hardware.push_back({"numa0", HardwareKind::Numa, -1, 0});
  t.hardware.push_back({"numa1", HardwareKind::Numa, -1, 1});
  t.hardware.push_back({"nic0", HardwareKind::Nic, -1, 0});
  t.hardware.push_back({"nic1", HardwareKind::Nic, -1, 1});
  for (size_t i = 0; i < 4; ++i)
    link(t, i, 4 + t.devices[i] / 2, "gpu-up" + std::to_string(i), 32);
  link(t, 4, 6, "nic-up0", 64);
  link(t, 5, 7, "nic-up1", 64);
  t.networkPorts = {
      {"nic0", "pci0", 1, 0, 6, true, {50, -1, Evidence::Advertised}},
      {"nic1", "pci1", 1, 1, 7, true, {50, -1, Evidence::Advertised}}};
  return t;
}
}  // namespace
int main() {
  try {
    auto t = fixture();
    std::vector<RailDevice> rails{{"nic0", 4}, {"nic1", 1}};
    auto plan = planNetworkRails(t, rails);
    auto candidates = networkRailCandidates(t, rails);
    require(candidates.size() == 4 && candidates[0].size() == 1 &&
                candidates[0][0].transmit.count("nic:nic1:1:tx") &&
                candidates[0][0].receive.count("nic:nic1:1:rx"),
            "directed physical rail candidates missing");
    auto noReverse = t;
    for (auto& edge : noReverse.links)
      if (edge.resource == "nic-up1:reverse") edge.performance = {};
    require(networkRailCandidates(noReverse, rails)[0].empty(),
            "missing receive capacity became a zero-cost path");
    for (size_t i = 0; i < 4; ++i)
      require(plan[i].device == t.devices[i] &&
                  plan[i].transportIndex == (t.devices[i] < 2 ? 4 : 1),
              "GPU/rank order must not be used as the transport rail index");
    auto reordered = planNetworkRails(t, {{"nic1", 0}, {"nic0", 1}});
    require(
        reordered[0].transportIndex == 0 && reordered[1].transportIndex == 1,
        "transport device-name mapping must honor HCA reorder");
    t.links.push_back({4, 5, "numa:0->numa:1", {}, 21});
    t.links.push_back({5, 4, "numa:1->numa:0", {}, 21});
    auto missing = t;
    missing.networkPorts[1].active = false;
    rejects([&] { planNetworkRails(missing, rails); },
            "down NIC selected across an unknown NUMA path");
    auto measurementOnly = networkRailCandidates(missing, rails, true);
    require(measurementOnly[0].size() == 1 &&
                measurementOnly[0][0].assignment.nic == "nic0" &&
                measurementOnly[0][0].requiresMeasurement &&
                measurementOnly[0][0].assignment.capacityGBs < 0 &&
                measurementOnly[0][0].transmit.size() == 1 &&
                !measurementOnly[1][0].requiresMeasurement,
            "unknown cross-NUMA capacity was fabricated or down NIC admitted");
    auto reverseOnly = networkRailCandidates(noReverse, rails, true);
    require(reverseOnly[0].back().requiresMeasurement,
            "missing receive path did not require complete measurements");
    auto unknown = t;
    unknown.networkPorts[0].transmit.evidence = Evidence::Unknown;
    rejects([&] { planNetworkRails(unknown, rails); },
            "unknown NIC bandwidth admitted");
    auto multi = t;
    auto port = multi.networkPorts[1];
    port.port = 2;
    multi.networkPorts.push_back(port);
    rejects([&] { planNetworkRails(multi, rails); },
            "ambiguous active multi-port HCA admitted");
    rejects(
        [&] {
          planNetworkRails(t, {{"nic0", 0}, {"nic1", 0}});
        },
        "duplicate transport index admitted");
    // Both NICs now share NUMA0. Equal capacities spread traffic; different
    // capacities change the allocation without changing GPU count or rank IDs.
    Topology shared;
    shared.devices = {0, 1};
    shared.gpuNodes = {0, 1};
    shared.hardware = {{"g0", HardwareKind::Gpu, 0, 0},
                       {"g1", HardwareKind::Gpu, 1, 0},
                       {"cpu", HardwareKind::Numa, -1, 0},
                       {"a", HardwareKind::Nic, -1, 0},
                       {"b", HardwareKind::Nic, -1, 0}};
    link(shared, 0, 2, "g0up", 80);
    link(shared, 1, 2, "g1up", 80);
    link(shared, 2, 3, "aup", 100);
    link(shared, 2, 4, "bup", 100);
    shared.networkPorts = {
        {"a", "pa", 1, 0, 3, true, {40, -1, Evidence::Advertised}},
        {"b", "pb", 1, 0, 4, true, {40, -1, Evidence::Advertised}}};
    auto balanced = planNetworkRails(shared, {{"a", 0}, {"b", 1}});
    require(balanced[0].transportIndex != balanced[1].transportIndex,
            "shared NIC contention was ignored");
    shared.networkPorts[0].transmit.bandwidthGBs = 10;
    shared.networkPorts[1].transmit.bandwidthGBs = 100;
    auto faster = planNetworkRails(shared, {{"a", 0}, {"b", 1}});
    require(faster[0].transportIndex == 1 && faster[1].transportIndex == 1,
            "NIC bandwidth did not change assignment");
    std::cout << "PASS: topology-derived rails, HCA reordering, shared NIC "
                 "load, bandwidth changes and unavailable paths\n";
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
