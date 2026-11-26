#include "uccl_lite/hardware_calibration.hpp"
#include "uccl_lite/router_plan.hpp"
#include <iostream>
#include <stdexcept>
using namespace uccl_lite;
void require(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
int main() {
  try {
    Topology t;
    t.identity = "synthetic-worker-plans";
    t.avx2 = true;
    t.devices = {0, 1, 2, 3};
    t.numaNodes = {0, 0, 1, 1};
    t.gpuNames.resize(4, "test GPU");
    t.mappedHost.resize(4, true);
    t.peerAccess.resize(16, false);
    t.permittedCpuIds = {{0, {0, 2, 4, 6}}, {1, {1, 3, 5, 7}}};
    t.permittedCpusPerNode = {{0, 4}, {1, 4}};
    t.hardware = {{"numa:0", HardwareKind::Numa, -1, 0},
                  {"numa:1", HardwareKind::Numa, -1, 1}};
    for (int count : {1, 2}) {
      HostProbeOptions options;
      options.workersPerNode = count;
      auto paths = registerHostCopyPaths(t, options);
      for (auto const& path : paths)
        for (auto& p : t.paths)
          if (p.resource == path.resource)
            p.performance = {count == 1 ? 100.0 : 10.0, 0, Evidence::Measured};
    }
    auto configurations = routerWorkerConfigurations(t, {}, {1, 2, 8});
    require(configurations.size() == 2,
            "unavailable CPU budget became a candidate");
    require(
        configurations[1].options.workerCpus == std::vector<int>({0, 2, 1, 3}),
        "group-major worker mapping");
    CollectiveCall call;
    call.count = 65536;
    for (int r = 0; r < 4; ++r)
      call.ports.push_back(
          {reinterpret_cast<void*>(0x100000UL + r * 0x100000UL),
           reinterpret_cast<void*>(0x1000000UL + r * 0x100000UL), nullptr, r});
    int fallbacks = 0;
    auto fallback = [&](CollectiveCall const&) { ++fallbacks; };
    BackendPolicy policy;
    policy.fallbackIdentity = "same-fallback-build";
    RouterPlanDispatcher missing(t, policy, configurations, fallback);
    require(
        !missing.prepare(call).configuration && !missing.routerInitialized(),
        "unmeasured worker plan initialized router");
    missing.enqueue(call);
    require(fallbacks == 1, "fallback callback not used");
    for (size_t i = 0; i < configurations.size(); ++i)
      policy.measurements.push_back(
          {missing.candidateTopology(i).identity,
           policy.fallbackIdentity,
           call.workload(),
           i ? TimingRange{4, 5, 6} : TimingRange{8, 9, 10},
           {20, 21, 22},
           3,
           true});
    RouterPlanDispatcher measured(t, policy, configurations, fallback);
    auto selected = measured.decision(call);
    require(selected.configuration == 1 && selected.memoryPriorUs &&
                !measured.routerInitialized(),
            "exact measurements must beat an opposing copy-based prior without "
            "initialization");
    require(selected.admission.routerMicroseconds == 5,
            "selected measured latency missing");
    bool needsPrepare = false;
    try {
      measured.enqueue(call);
    } catch (std::logic_error const&) {
      needsPrepare = true;
    }
    require(needsPrepare && !measured.routerInitialized(),
            "enqueue initialized an admitted plan");
    ++call.count;
    require(
        !measured.prepare(call).configuration && !measured.routerInitialized(),
        "unmeasured size inherited a worker plan");
    measured.enqueue(call);
    require(fallbacks == 2, "unmeasured extent did not fall back");
    --call.count;
    policy.measurements[1].router = {30, 31, 32};
    RouterPlanDispatcher regressed(t, policy, configurations, fallback);
    require(regressed.decision(call).configuration == 0,
            "regressed configuration remained selected");
    auto gatherCall = call;
    gatherCall.collective = Collective::AllGather;
    auto gatherPolicy = policy;
    gatherPolicy.measurements.push_back({missing.candidateTopology(0).identity,
                                         policy.fallbackIdentity,
                                         gatherCall.workload(),
                                         {1, 2, 3},
                                         {20, 21, 22},
                                         3,
                                         true});
    RouterPlanDispatcher gather(t, gatherPolicy, configurations, fallback);
    require(gather.decision(gatherCall).configuration == 0 &&
                !gather.decision(gatherCall).memoryPriorUs,
            "AllGather flag publication was modeled as CPU payload copy");
    auto changed = t;
    changed.identity += ";changed-affinity";
    RouterPlanDispatcher stale(changed, policy, configurations, fallback);
    require(!stale.decision(call).configuration,
            "stale placement calibration selected workers");
    auto unknown = t;
    unknown.numaNodes[0] = -1;
    auto unavailable = routerWorkerConfigurations(unknown, {}, {1, 2});
    require(unavailable.empty(),
            "unknown NUMA placement generated worker candidates");
    RouterPlanDispatcher unsupported(unknown, policy, unavailable, fallback);
    require(!unsupported.prepare(call).configuration &&
                !unsupported.routerInitialized(),
            "unknown placement initialized routing resources");
    unsupported.enqueue(call);
    std::cout << "PASS: worker plans require exact evidence, consume scoped "
                 "memory priors, and preserve initialization-free fallback\n";
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
