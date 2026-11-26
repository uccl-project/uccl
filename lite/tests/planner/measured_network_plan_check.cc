#include "uccl_lite/network_plan.hpp"
#include <iostream>
#include <stdexcept>
using namespace uccl_lite;
void require(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
int main() {
  try {
    NetworkProfile p;
    p.probeIdentity = "synthetic-only";
    p.hosts = {{{"h0", "t0", "software0"}, {"h1", "t1", "software1"}}};
    std::vector<NetworkEndpoint> endpoints;
    for (int host = 0; host < 2; ++host)
      for (int gpu = 0; gpu < 2; ++gpu) {
        NetworkEndpoint e{
            p.hosts[host], "gpu" + std::to_string(host * 2 + gpu), gpu, {}};
        for (int nic = 0; nic < 2; ++nic) {
          std::string name = nic ? "b" : "a";
          double rate = nic ? 20 : 10;
          NetworkRailCandidate candidate;
          candidate.assignment = {gpu, nic ^ host, name, 1, rate, {}};
          candidate.transmit = {{name + ":tx", rate}, {e.gpu + ":tx", 100}};
          candidate.receive = {{name + ":rx", rate}, {e.gpu + ":rx", 100}};
          e.candidates.push_back(candidate);
        }
        endpoints.push_back(e);
      }
    std::vector<std::pair<size_t, size_t>> peers{{0, 2}, {1, 3}};
    for (auto [a, b] : peers)
      for (int an = 0; an < 2; ++an)
        for (int bn = 0; bn < 2; ++bn) {
          for (bool reverse : {false, true}) {
            NetworkPathObservation o;
            auto source = reverse ? b : a, destination = reverse ? a : b;
            o.key = {reverse ? 1 : 0,
                     reverse ? 0 : 1,
                     endpoints[source].gpu,
                     endpoints[destination].gpu,
                     (reverse ? bn : an) ? "b" : "a",
                     (reverse ? an : bn) ? "b" : "a",
                     1,
                     1};
            o.iterations = 1;
            double time = an == bn ? (an ? 150 : 100) : 300;
            o.sizes = {{16, {2, 2, 2}}, {1UL << 20, {time, time, time}}};
            p.paths.push_back(o);
          }
        }
    auto plan = planMeasuredNetworkRails(p, endpoints, peers);
    require(plan.calibrated && plan.basisBytes == (1UL << 20),
            "complete live path evidence rejected");
    require(plan.assignments[0].nic == "a" && plan.assignments[2].nic == "a" &&
                plan.assignments[1].nic == "b" &&
                plan.assignments[3].nic == "b",
            "shared capacity did not split coupled pairs");
    require(plan.assignments[0].transportIndex == 0 &&
                plan.assignments[2].transportIndex == 1,
            "remote HCA enumeration confused with physical NIC name");
    auto changed = p;
    for (auto& path : changed.paths)
      if (path.key.sourceNic == "a" && path.key.destinationNic == "a")
        path.sizes[1].microseconds = {600, 600, 600};
    plan = planMeasuredNetworkRails(changed, endpoints, peers);
    for (auto const& assignment : plan.assignments)
      require(assignment.nic == "b",
              "measured path bandwidth did not change rails");
    changed = p;
    for (auto& path : changed.paths)
      if (path.key.sourceHost == 1 && path.key.sourceNic == "a" &&
          path.key.destinationNic == "a")
        path.sizes[1].microseconds = {900, 900, 900};
    plan = planMeasuredNetworkRails(changed, endpoints, peers);
    for (auto const& assignment : plan.assignments)
      require(assignment.nic == "b", "reverse-path evidence ignored");
    auto stale = endpoints;
    stale[3].host.transportIdentity += "-new";
    require(!planMeasuredNetworkRails(p, stale, peers).calibrated,
            "stale remote identity admitted");
    changed = p;
    changed.paths.pop_back();
    require(!planMeasuredNetworkRails(changed, endpoints, peers).calibrated,
            "partial evidence changed one side's route");
    changed = p;
    changed.paths[0].rejection = "registration unsupported";
    changed.paths[0].sizes.clear();
    require(!planMeasuredNetworkRails(changed, endpoints, peers).calibrated,
            "rejected path admitted");
    bool rejected = false;
    try {
      planMeasuredNetworkRails(p, endpoints, {{0, 2}, {0, 3}});
    } catch (std::invalid_argument const&) {
      rejected = true;
    }
    require(rejected, "duplicate peer assignment accepted");
    auto inconsistent = endpoints;
    inconsistent[1].candidates[0].transmit["a:tx"] = 9;
    rejected = false;
    try {
      planMeasuredNetworkRails(p, inconsistent, peers);
    } catch (std::invalid_argument const&) {
      rejected = true;
    }
    require(rejected, "inconsistent shared NIC capacity accepted");
    auto grouped = p;
    grouped.groupControlIdentities = {"control0", "control1"};
    auto addGroup = [&](std::string label, std::vector<int> nics, double time) {
      NetworkGroupObservation group;
      group.label = std::move(label);
      group.iterations = 1;
      for (auto [a, b] : peers)
        for (bool reverse : {false, true}) {
          auto source = reverse ? b : a, destination = reverse ? a : b;
          group.flows.push_back(
              {reverse ? 1 : 0, reverse ? 0 : 1, endpoints[source].gpu,
               endpoints[destination].gpu, nics[source] ? "b" : "a",
               nics[destination] ? "b" : "a", 1, 1});
        }
      for (size_t bytes : {16UL, 1UL << 20}) {
        NetworkGroupObservation::Size size;
        size.bytes = bytes;
        for (int r = 0; r < 3; ++r) {
          size.senderMicroseconds.push_back(std::vector<double>(4, time / 2));
          size.cohortMicroseconds.push_back(
              std::vector<double>(4, time + r - 1));
        }
        group.sizes.push_back(std::move(size));
      }
      grouped.groups.push_back(std::move(group));
    };
    addGroup("baseline", {0, 1, 0, 1}, 500);
    addGroup("all-b", {1, 1, 1, 1}, 300);
    auto concurrent = planMeasuredNetworkRails(grouped, endpoints, peers);
    require(concurrent.concurrentMeasured &&
                concurrent.concurrentGroup == "all-b" &&
                concurrent.concurrentUs == 300,
            "concurrent evidence did not replace optimistic isolated choice");
    for (auto const& assignment : concurrent.assignments)
      require(assignment.nic == "b", "concurrent winner was not executed");
    require(concurrent.isolatedPathUs == 150,
            "winner retained stale isolated diagnostics");
    auto partial = grouped;
    partial.groups.erase(partial.groups.begin());
    require(
        !planMeasuredNetworkRails(partial, endpoints, peers).concurrentMeasured,
        "unmeasured baseline compared with measured candidate");
    auto overlap = grouped;
    overlap.groups[1].sizes[1].cohortMicroseconds[0][0] = 501;
    require(
        planMeasuredNetworkRails(overlap, endpoints, peers).concurrentGroup ==
            "baseline",
        "overlapping concurrent samples admitted");
    auto wrongExtent = grouped;
    for (auto& group : wrongExtent.groups) group.sizes[1].bytes *= 2;
    require(!planMeasuredNetworkRails(wrongExtent, endpoints, peers)
                 .concurrentMeasured,
            "different concurrent byte extent extrapolated");
    auto incomplete = endpoints;
    for (auto& endpoint : incomplete) {
      endpoint.candidates = {endpoint.candidates[1]};
      auto& c = endpoint.candidates[0];
      c.requiresMeasurement = true;
      c.assignment.capacityGBs = -1;
      c.transmit = {{"b:tx", 20}};
      c.receive = {{"b:rx", 20}};
    }
    require(!planMeasuredNetworkRails(p, incomplete, peers).calibrated,
            "isolated paths bypassed unknown shared physical capacities");
    auto covered = planMeasuredNetworkRails(grouped, incomplete, peers);
    require(covered.calibrated && covered.concurrentMeasured &&
                covered.concurrentGroup == "all-b",
            "complete path/cohort measurements did not cover unknown physical "
            "path");
    require(
        !planMeasuredNetworkRails(wrongExtent, incomplete, peers).calibrated,
        "wrong cohort extent covered unknown path");
    auto missingReverse = grouped;
    missingReverse.paths.pop_back();
    require(
        !planMeasuredNetworkRails(missingReverse, incomplete, peers).calibrated,
        "missing reverse evidence covered unknown path");
    auto staleIncomplete = incomplete;
    staleIncomplete[0].host.transportIdentity += "-stale";
    require(
        !planMeasuredNetworkRails(grouped, staleIncomplete, peers).calibrated,
        "measurement-only path bypassed live identity");
    auto boundedPrior = endpoints;
    for (auto& endpoint : boundedPrior) {
      endpoint.candidates[1].requiresMeasurement = true;
      endpoint.candidates[1].assignment.capacityGBs = -1;
    }
    auto prior = planMeasuredNetworkRails(p, boundedPrior, peers);
    require(prior.calibrated && !prior.concurrentMeasured,
            "supported physical prior was discarded");
    for (auto const& a : prior.assignments)
      require(a.nic == "a",
              "unknown physical bounds entered the isolated prior");
    addGroup("all-a", {0, 0, 0, 0}, 500);
    auto promotion = planMeasuredNetworkRails(grouped, boundedPrior, peers);
    require(
        promotion.concurrentMeasured && promotion.concurrentGroup == "all-b",
        "complete concurrent evidence did not qualify an unknown-bound "
        "alternative");
    std::cout << "PASS: coupled measured rails, shared capacity, reversed "
                 "enumeration, live identities and complete evidence\n";
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
