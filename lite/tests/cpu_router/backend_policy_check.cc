#include "uccl_lite/backend_policy.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace uccl_lite;
namespace {
int checks = 0;
void require(bool condition, char const* message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}
template <class F>
void rejects(F run, char const* message) {
  bool rejected = false;
  try {
    run();
  } catch (std::invalid_argument const&) {
    rejected = true;
  }
  require(rejected, message);
}
Topology topology(int ranks) {
  Topology t;
  t.identity = "local-test-" + std::to_string(ranks);
  t.avx2 = true;
  t.cudaRuntimeVersion = 12090;
  t.driverVersion = "580.178.04";
  for (int r = 0; r < ranks; ++r) {
    t.devices.push_back(r);
    t.numaNodes.push_back(r < ranks / 2 ? 0 : 1);
    t.gpuNames.push_back("NVIDIA L4");
    t.mappedHost.push_back(true);
  }
  t.peerAccess.resize(ranks * ranks, false);
  t.permittedCpusPerNode = {{0, 24}, {1, 24}};
  return t;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    BackendPolicy p;
    p.fallbackIdentity = "nccl:22705";
    p.measurements = loadL4ReferenceProfile(
        argc > 1 ? argv[1] : "benchmarks/data/primitives_8l4.csv");
    require(p.measurements.size() == 144, "reference dimensions");
    auto eight = topology(8), four = topology(4);
    Workload w{Collective::AllGather, ScalarType::BFloat16, 1UL << 30, true};
    require(!p.select(eight, w).cpuRouter,
            "historical AG result cannot enable a different host");
    require(!p.select(four, w).cpuRouter,
            "four GPUs must not inherit eight-GPU timings");
    w.collective = Collective::ReduceScatter;
    require(!p.select(eight, w).cpuRouter,
            "historical RS result cannot enable a different host");
    w.collective = Collective::AllReduce;
    require(!p.select(eight, w).cpuRouter,
            "historical AR result cannot enable a different host");
    Measurement localEight{eight.identity, p.fallbackIdentity, w,
                           {8, 9, 10},   {19, 20, 21},       3,
                           true};
    p.measurements.push_back(localEight);
    require(p.select(eight, w).cpuRouter,
            "exact local eight-GPU calibration may enable router");
    auto changed = eight;
    changed.peerAccess[1] = true;
    changed.identity += ";peer-access-changed";
    require(!p.select(changed, w).cpuRouter, "P2P topology mismatch");
    changed = eight;
    changed.gpuNames[0] = "NVIDIA A100";
    changed.identity += ";gpu-model-changed";
    require(!p.select(changed, w).cpuRouter, "GPU model mismatch");
    changed = eight;
    changed.driverVersion = "595.58.03";
    changed.identity += ";driver-changed";
    require(!p.select(changed, w).cpuRouter,
            "driver mismatch needs local calibration");
    changed = eight;
    changed.cudaRuntimeVersion = 13000;
    changed.identity += ";cuda-changed";
    require(!p.select(changed, w).cpuRouter, "toolkit mismatch");
    changed = eight;
    changed.permittedCpusPerNode[0] = 8;
    changed.identity += ";affinity-changed";
    require(!p.select(changed, w).cpuRouter, "CPU budget mismatch");
    changed = eight;
    std::swap(changed.devices[0], changed.devices[1]);
    changed.identity += ";device-order-changed";
    require(!p.select(changed, w).cpuRouter, "device ordering mismatch");
    p.fallbackIdentity = "nccl:22807";
    require(!p.select(eight, w).cpuRouter, "baseline mismatch");
    p.fallbackIdentity = "nccl:22705";
    w.bytes = 100000;
    require(!p.select(eight, w).cpuRouter,
            "unmeasured size must not interpolate");
    w.bytes = 1 << 20;
    w.dtype = ScalarType::Float32;
    require(!p.select(eight, w).cpuRouter,
            "unsupported dtype must retain fallback");
    p.mode = RouterMode::Force;
    rejects([&] { p.select(eight, w); }, "force cannot bypass dtype support");
    w.dtype = ScalarType::BFloat16;
    require(p.select(four, w).cpuRouter,
            "explicit force permits four-GPU validation");
    changed = four;
    changed.mappedHost[0] = false;
    rejects([&] { p.select(changed, w); },
            "force cannot bypass host mapping support");
    p.mode = RouterMode::Off;
    require(!p.select(eight, w).cpuRouter, "explicit off takes precedence");
    p.mode = RouterMode::Auto;
    Measurement m{four.identity, p.fallbackIdentity, w,
                  {8, 9, 10},    {19, 20, 21},       3,
                  true};
    p.measurements.push_back(m);
    require(p.select(four, w).cpuRouter,
            "local evidence may opt in on any supported topology");
    auto outOfPlace = w;
    outOfPlace.inPlace = !w.inPlace;
    require(!p.select(four, outOfPlace).cpuRouter, "placement must match");
    m.router = {22, 23, 24};
    p.measurements.push_back(m);
    require(!p.select(four, w).cpuRouter,
            "newest local regression disables router");
    m.router = {18, 19, 20};
    p.measurements.push_back(m);
    require(!p.select(four, w).cpuRouter,
            "overlapping timing ranges are not a stable win");
    m.router = {18.7, 18.8, 18.9};
    m.fallback = {19, 19.1, 19.2};
    p.measurements.push_back(m);
    require(!p.select(four, w).cpuRouter, "gain below safety margin");
    m.correct = false;
    p.measurements.push_back(m);
    rejects([&] { p.select(four, w); },
            "uncorrected timings must not select a backend");
    p.measurements.pop_back();
    m.correct = true;
    m.repetitions = 1;
    p.measurements.push_back(m);
    rejects([&] { p.select(four, w); }, "one run is insufficient");
    p.measurements.pop_back();
    p.minimumSpeedup = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { p.select(four, w); }, "NaN margin rejected");
    p.minimumSpeedup = 1.03;
    m.repetitions = 3;
    m.router = {8, 9, 10};
    m.fallback = {19, 20, 21};
    m.workload.alignment = w.alignment = 256;
    saveRouterProfile("build/policy-check.profile", {m});
    auto loaded = loadRouterProfile("build/policy-check.profile");
    require(loaded.size() == 1 && loaded[0].topologyIdentity == four.identity,
            "profile round trip");
    require(loaded[0].workload.alignment == 256,
            "profile preserves measured buffer alignment");
    p.measurements = loaded;
    require(p.select(four, w).cpuRouter,
            "persisted calibration selects measured winner");
    auto offset = w;
    offset.alignment = 16;
    require(!p.select(four, offset).cpuRouter,
            "aligned calibration cannot select a slower alignment path");
    {
      std::ofstream legacy("build/policy-check-v1.profile");
      legacy << "uccl-lite-router-profile-v1\n"
             << std::quoted(m.topologyIdentity) << ' '
             << std::quoted(m.fallbackIdentity) << ' '
             << collectiveName(w.collective) << ' ' << w.bytes << ' '
             << w.inPlace << " 3 1 8 9 10 19 20 21\n";
    }
    p.measurements = loadRouterProfile("build/policy-check-v1.profile");
    require(
        p.measurements.size() == 1 && p.measurements[0].workload.alignment == 0,
        "legacy profiles retain unknown alignment for offline reading");
    require(!p.select(four, w).cpuRouter,
            "legacy unknown alignment cannot enable a live aligned call");
    require(parseRouterMode("off") == RouterMode::Off, "off parser");
    rejects([] { parseRouterMode("typo"); }, "invalid mode rejected");
    std::cout << "Backend policy: " << checks
              << " checks passed; historical data cannot enable router\n";
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
