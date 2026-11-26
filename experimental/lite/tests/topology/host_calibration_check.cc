#include "uccl_lite/hardware_calibration.hpp"
#include <iostream>
#include <stdexcept>
#include <sched.h>

using namespace uccl_lite;
int main() {
  try {
    cpu_set_t before, after;
    if (sched_getaffinity(0, sizeof(before), &before))
      throw std::runtime_error("affinity query failed");
    auto t = inspectTopology({0, 1, 2, 3});
    HostProbeOptions options;
    auto paths = registerHostCopyPaths(t, options);
    auto registered = t.paths.size();
    registerHostCopyPaths(t, options);
    if (t.paths.size() != registered)
      throw std::runtime_error("host resource registration is not idempotent");
    auto different = t;
    options.workersPerNode = 1;
    auto fewer = registerHostCopyPaths(different, options);
    if (fewer.front().resource == paths.front().resource)
      throw std::runtime_error(
          "worker placement is absent from resource identity");
    options.workersPerNode = 4;
    auto profile = probeHostResources(t, options);
    saveHardwareProfile("build/host-calibration-check.profile", profile);
    auto restored = loadHardwareProfile("build/host-calibration-check.profile");
    auto fresh = inspectTopology({0, 1, 2, 3});
    registerHostCopyPaths(fresh, options);
    applyHardwareCalibration(fresh, restored.calibration);
    if (restored.observations.size() != 2 * paths.size())
      throw std::runtime_error("missing raw samples");
    for (auto const& link : fresh.links)
      if (link.resource.rfind("numa:", 0) == 0 &&
          link.performance.evidence != Evidence::Unknown)
        throw std::runtime_error(
            "CPU memcpy incorrectly qualified physical interconnect bandwidth");
    for (auto const& node : fresh.hardware)
      if (node.kind == HardwareKind::Numa &&
          node.memory.evidence != Evidence::Unknown)
        throw std::runtime_error(
            "worker-specific copy incorrectly became generic memory bandwidth");
    // A worker-side placement failure must cancel barriers and join threads.
    Topology invalid;
    invalid.identity = "synthetic-invalid-numa";
    invalid.hardware = {{"numa:1023", HardwareKind::Numa, -1, 1023, {}}};
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
      if (CPU_ISSET(cpu, &before)) {
        invalid.permittedCpuIds[1023] = {cpu};
        break;
      }
    options.workersPerNode = 1;
    options.bytesPerWorker = 1UL << 20;
    bool rejected = false;
    try {
      probeHostResources(invalid, options);
    } catch (std::exception const&) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("invalid NUMA placement was measured");
    if (sched_getaffinity(0, sizeof(after), &after) ||
        !CPU_EQUAL(&before, &after))
      throw std::runtime_error("host probe changed caller affinity");
    std::cout << "PASS: NUMA placement, full CPU-copy verification, worker "
                 "identity, persistence and failure cleanup\n";
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
