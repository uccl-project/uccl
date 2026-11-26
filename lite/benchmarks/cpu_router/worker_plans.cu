// Reuse the independent guarded oracle, not the old calibration lifecycle.
#define main uccl_lite_unused_joint_benchmark_main
#include "../joint/joint_bench.cu"
#undef main
#include "uccl_lite/collectives/peer_collectives.hpp"
#include "uccl_lite/hardware_calibration.hpp"
#include "uccl_lite/router_plan.hpp"

int main(int argc, char** argv) {
  try {
    if (argc < 2)
      throw std::invalid_argument(
          "usage: worker_plans OUTPUT_PROFILE [HOST_PROFILE]");
    if (std::filesystem::exists(argv[1]))
      throw std::invalid_argument("output profile already exists");
    constexpr size_t bytes = 1UL << 20;
    Buffers buffers(4, bytes);
    auto t = inspectTopology(buffers.devices);
    if (argc > 2) {
      HostProbeOptions options;
      registerHostCopyPaths(t, options);
      applyHardwareCalibration(t, loadHardwareProfile(argv[2]).calibration);
    }
    auto call = buffers.call(0, bytes);
    PeerCollectives peer(t);
    peer.prepare(call);
    auto fallback = [&](CollectiveCall const& c) { peer.enqueue(c); };
    BackendPolicy policy;
    policy.fallbackIdentity = JointCollectives::peerIdentity() +
                              ";selected=" + peer.plan(call).plan.id;
    cpu_router::CpuRouterOptions base;
    base.maxTensorBytes = bytes;
    auto configs = routerWorkerConfigurations(t, base, {1, 2, 4});
    std::ofstream raw(std::string(argv[1]) + ".csv");
    raw << "workers,round,backend,sample_us\n";
    unsigned generation = 1;
    auto block = [&](Enqueue const& enqueue, int workers, int round,
                     char const* backend) {
      std::vector<double> samples;
      for (int i = 0; i < 3; ++i) {
        double us = invocation(buffers, 0, bytes, ++generation, enqueue);
        samples.push_back(us);
        raw << workers << ',' << round << ',' << backend << ',' << us << '\n';
      }
      return samples;
    };
    std::cout << "4 L4, BF16 AR, 1 MiB, out-of-place; baseline prepared peer "
              << peer.plan(call).plan.id
              << "; maximum-GPU event per-call timing; 3 ABBA rounds, 2 "
                 "warmups and 3 calls per block; checks outside timing; "
                 "baseline blocks have no router workers\n";
    for (auto const& config : configs) {
      std::vector<double> routers, baselines;
      std::string identity;
      for (int round = 0; round < 3; ++round) {
        auto baseline = [&] { peer.enqueue(call); };
        for (int i = 0; i < 2; ++i)
          invocation(buffers, 0, bytes, ++generation, baseline);
        auto a = block(baseline, config.options.workersPerNode, round, "peer");
        std::vector<double> b;
        {
          auto force = policy;
          force.mode = RouterMode::Force;
          CollectiveDispatcher router(t, force, fallback, config.options);
          identity = router.topology().identity;
          router.prepare(call);
          auto enqueue = [&] { router.enqueue(call); };
          for (int i = 0; i < 2; ++i)
            invocation(buffers, 0, bytes, ++generation, enqueue);
          b = block(enqueue, config.options.workersPerNode, round, "router");
          auto c =
              block(enqueue, config.options.workersPerNode, round, "router");
          b.insert(b.end(), c.begin(), c.end());
        }
        // Router destruction precedes the second baseline block.
        for (int i = 0; i < 2; ++i)
          invocation(buffers, 0, bytes, ++generation, baseline);
        auto d = block(baseline, config.options.workersPerNode, round, "peer");
        a.insert(a.end(), d.begin(), d.end());
        routers.push_back(median(b));
        baselines.push_back(median(a));
      }
      auto r = summarize(routers), b = summarize(baselines);
      policy.measurements.push_back({identity,
                                     policy.fallbackIdentity,
                                     call.workload(),
                                     {r.minimum, r.median, r.maximum},
                                     {b.minimum, b.median, b.maximum},
                                     3,
                                     true});
      std::cout << config.name << " router_us=" << r.median
                << " peer_us=" << b.median << '\n';
    }
    saveRouterProfile(argv[1], policy.measurements);
    policy.measurements = loadRouterProfile(argv[1]);
    RouterPlanDispatcher runtime(t, policy, configs, fallback);
    auto choice = runtime.prepare(call);
    invocation(
        buffers, 0, bytes, ++generation, [&] { runtime.enqueue(call); }, true);
    if (!choice.configuration && runtime.routerInitialized())
      throw std::runtime_error("fallback initialized router");
    std::cout << "selected="
              << (choice.configuration ? configs[*choice.configuration].name
                                       : "peer")
              << " reason=" << choice.admission.reason << " memory_prior_us="
              << (choice.memoryPriorUs ? std::to_string(*choice.memoryPriorUs)
                                       : "unknown")
              << '\n';
    std::cout << "PASS: real paired profiles reload and execute the admitted "
                 "plan; all samples and negative outcomes retained\n";
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
