#include "uccl_lite/collective_dispatch.hpp"
#include <array>
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
  } catch (std::logic_error const&) {
    rejected = true;
  }
  require(rejected, message);
}
}  // namespace

int main() {
  try {
    // Host-only test: Off and unmatched Auto must not call CUDA or create
    // workers.
    Topology t;
    t.devices = {0, 1};
    t.numaNodes = {0, 1};
    t.gpuNames = {"test", "test"};
    t.mappedHost = {true, true};
    t.peerAccess = {false, false, false, false};
    t.avx2 = true;
    t.identity = "host-only-dispatch-test";
    t.permittedCpusPerNode = {{0, 2}, {1, 2}};
    t.permittedCpuIds = {{0, {0, 1}}, {1, {2, 3}}};
    alignas(16) std::array<std::array<uint16_t, 16>, 2> inputs{}, outputs{};
    CollectiveCall call;
    call.count = 4;
    for (int r = 0; r < 2; ++r)
      call.ports.push_back({inputs[r].data(), outputs[r].data(), nullptr, r});
    int fallbackCalls = 0;
    BackendPolicy policy;
    auto callback = [&](CollectiveCall const& c) {
      require(c.count == 4, "fallback receives original call");
      ++fallbackCalls;
    };
    CollectiveDispatcher automatic(t, policy, callback);
    require(!automatic.prepare(call).cpuRouter,
            "unmatched Auto keeps fallback");
    require(!automatic.routerInitialized(), "no speculative router allocation");
    automatic.enqueue(call);
    require(fallbackCalls == 1 && !automatic.routerInitialized(),
            "fallback invoked without router");
    call.dtype = ScalarType::Float32;
    automatic.enqueue(call);
    require(fallbackCalls == 2,
            "unsupported router dtype reaches caller backend");
    policy.mode = RouterMode::Force;
    CollectiveDispatcher forced(t, policy, callback);
    rejects([&] { forced.prepare(call); },
            "force cannot reinterpret float32 as BF16");
    call.dtype = ScalarType::BFloat16;
    rejects([&] { forced.enqueue(call); },
            "force requires explicit preparation");
    require(!forced.routerInitialized(), "enqueue must not allocate router");
    call.ports[0].output = inputs[0].data();
    rejects([&] { automatic.enqueue(call); },
            "out-of-place overlap is rejected");
    call.ports[0].output = outputs[0].data();
    call.inPlace = true;
    rejects([&] { automatic.enqueue(call); },
            "incorrect in-place layout rejected");
    call.collective = Collective::AllGather;
    for (int r = 0; r < 2; ++r)
      call.ports[r].input = outputs[r].data() + r * call.count;
    automatic.enqueue(call);
    require(fallbackCalls == 3, "valid in-place AllGather fallback");
    call.collective = Collective::ReduceScatter;
    for (int r = 0; r < 2; ++r) {
      call.ports[r].input = inputs[r].data();
      call.ports[r].output = inputs[r].data() + r * call.count;
    }
    automatic.enqueue(call);
    require(fallbackCalls == 4, "valid in-place ReduceScatter fallback");
    call.ports[1].device = 0;
    rejects([&] { automatic.enqueue(call); }, "changed device order rejected");
    call.ports[1].device = 1;
    call.count = std::numeric_limits<size_t>::max();
    rejects([&] { automatic.enqueue(call); }, "extent overflow rejected");
    require(!automatic.routerInitialized(),
            "all fallback tests avoid router initialization");
    call.count = 4;
    cpu_router::CpuRouterOptions small;
    small.maxTensorBytes = 4;
    BackendPolicy autoPolicy;
    CollectiveDispatcher limited(t, autoPolicy, callback, small);
    require(!limited.prepare(call).cpuRouter && !limited.routerInitialized(),
            "capacity miss retains fallback without allocation");
    CollectiveDispatcher forcedLimited(t, policy, callback, small);
    rejects([&] { forcedLimited.prepare(call); },
            "force rejects capacity overflow");
    BackendPolicy off;
    off.mode = RouterMode::Off;
    CollectiveDispatcher disabled(t, off, callback);
    require(!disabled.prepare(call).cpuRouter && !disabled.routerInitialized(),
            "explicit off remains uninitialized");
    cpu_router::CpuRouterOptions gpu;
    gpu.reduction = cpu_router::ReductionPlacement::Gpu;
    CollectiveDispatcher gpuAutomatic(t, autoPolicy, callback, gpu);
    require(!gpuAutomatic.prepare(call).cpuRouter &&
                !gpuAutomatic.routerInitialized(),
            "unmeasured GPU reduction retains fallback without allocation");
    require(gpuAutomatic.topology().identity != automatic.topology().identity,
            "GPU reduction requires its own local calibration");
    gpu.interleaveScatter = true;
    gpu.reduction = cpu_router::ReductionPlacement::Gpu;
    CollectiveDispatcher striped(t, autoPolicy, callback, gpu);
    require(striped.topology().identity != gpuAutomatic.topology().identity &&
                !striped.prepare(call).cpuRouter &&
                !striped.routerInitialized(),
            "striped placement requires its own measurements and stays "
            "uninitialized");
    gpu.gpuProgress = true;
    gpu.reduction = cpu_router::ReductionPlacement::Gpu;
    CollectiveDispatcher direct(t, autoPolicy, callback, gpu);
    require(direct.topology().identity != striped.topology().identity &&
                !direct.prepare(call).cpuRouter && !direct.routerInitialized(),
            "GPU progress needs separate evidence and remains uninitialized on "
            "fallback");
    gpu.interleaveScatter = false;
    gpu.reduction = cpu_router::ReductionPlacement::Cpu;
    rejects([&] { CollectiveDispatcher invalid(t, autoPolicy, callback, gpu); },
            "GPU progress cannot bypass CPU arithmetic");
    gpu.gpuProgress = false;
    gpu.interleaveScatter = true;
    CollectiveDispatcher cpuStriped(t, autoPolicy, callback, gpu);
    require(cpuStriped.topology().identity != striped.topology().identity &&
                !cpuStriped.prepare(call).cpuRouter &&
                !cpuStriped.routerInitialized(),
            "CPU stripes require separate evidence and preserve fallback");
    gpu.concurrentProducer = true;
    rejects([&] { CollectiveDispatcher invalid(t, autoPolicy, callback, gpu); },
            "striped packets reject flat concurrent-producer epochs");
    gpu.reduction = cpu_router::ReductionPlacement::Gpu;
    CollectiveDispatcher concurrentGpu(t, autoPolicy, callback, gpu);
    require(!concurrentGpu.prepare(call).cpuRouter &&
                !concurrentGpu.routerInitialized(),
            "existing GPU concurrent transport retains unmeasured fallback");
    gpu.concurrentProducer = false;
    gpu.reduction = static_cast<cpu_router::ReductionPlacement>(99);
    rejects(
        [&] { CollectiveDispatcher invalid(t, autoPolicy, callback, gpu); },
        "invalid reduction placement is rejected before CUDA initialization");
    BackendPolicy calibrated;
    calibrated.fallbackIdentity = "host-only-fallback-v1";
    calibrated.measurements.push_back({automatic.topology().identity,
                                       calibrated.fallbackIdentity,
                                       call.workload(),
                                       {8, 9, 10},
                                       {19, 20, 21},
                                       3,
                                       true});
    require(calibrated.select(automatic.topology(), call.workload()).cpuRouter,
            "matching calibration is eligible before changing geometry");
    require(automatic.topology().identity.find(";router-build-id=") !=
                std::string::npos,
            "software identity records the containing CUDA module build");
    auto offsetCall = call;
    for (auto& port : offsetCall.ports) {
      port.input = static_cast<uint16_t const*>(port.input) + 1;
      port.output = static_cast<uint16_t*>(port.output) + 1;
    }
    CollectiveDispatcher offset(t, calibrated, callback);
    require(
        !offset.prepare(offsetCall).cpuRouter && !offset.routerInitialized(),
        "different buffer alignment retains fallback without allocation");
    for (int choice = 0; choice < 12; ++choice) {
      cpu_router::CpuRouterOptions tuned;
      if (choice == 0) tuned.transportVectorBytes = 32;
      if (choice == 1) tuned.singleCtaBytes = 65536;
      if (choice == 2) tuned.singleNumaReduction = true;
      if (choice == 3) tuned.transportBlocks = 512;
      if (choice == 4) tuned.smallReduceScatterWorkersPerNode = 4;
      if (choice == 5) tuned.batchCpuJobs = true;
      if (choice == 6) tuned.directScatterResults = true;
      if (choice == 7) tuned.gpuLaunchWorkersPerNode = 1;
      if (choice == 8) tuned.smallReduceScatterGpuProgressMaxBytes = 65536;
      if (choice == 9) {
        tuned.gpuLaunchWorkersPerNode = 1;
        tuned.gpuLaunchAllowedCpus = {1, 3};
      }
      if (choice == 10) tuned.smallAllReduceGpuProgressMaxBytes = 8192;
      if (choice == 11) tuned.smallAllGatherGpuProgressMaxBytes = 65536;
      CollectiveDispatcher changed(t, calibrated, callback, tuned);
      require(!changed.prepare(call).cpuRouter && !changed.routerInitialized(),
              "new transport options cannot reuse another configuration's "
              "calibration");
    }
    cpu_router::CpuRouterOptions invalidWorkers;
    invalidWorkers.smallReduceScatterWorkersPerNode = 64;
    rejects(
        [&] {
          CollectiveDispatcher invalid(t, autoPolicy, callback, invalidWorkers);
        },
        "small RS worker limit is validated before CUDA initialization");
    invalidWorkers = {};
    invalidWorkers.gpuLaunchWorkersPerNode = 33;
    rejects(
        [&] {
          CollectiveDispatcher invalid(t, autoPolicy, callback, invalidWorkers);
        },
        "GPU launch worker limit is validated before CUDA initialization");
    invalidWorkers = {};
    invalidWorkers.smallReduceScatterGpuProgressMaxBytes = 131072;
    rejects(
        [&] {
          CollectiveDispatcher invalid(t, autoPolicy, callback, invalidWorkers);
        },
        "small RS progress threshold is validated before CUDA initialization");
    for (size_t limit : {size_t(1), size_t(17), size_t(1048592)}) {
      invalidWorkers = {};
      invalidWorkers.smallAllReduceGpuProgressMaxBytes = limit;
      rejects(
          [&] {
            CollectiveDispatcher invalid(t, autoPolicy, callback,
                                         invalidWorkers);
          },
          "small AR progress limit is validated before CUDA initialization");
      invalidWorkers = {};
      invalidWorkers.smallAllGatherGpuProgressMaxBytes = limit;
      rejects(
          [&] {
            CollectiveDispatcher invalid(t, autoPolicy, callback,
                                         invalidWorkers);
          },
          "small AG progress limit is validated before CUDA initialization");
    }
    {
      cpu_router::CpuRouterOptions ring;
      ring.hostRingReduceScatter = true;
      rejects(
          [&] { CollectiveDispatcher invalid(t, autoPolicy, callback, ring); },
          "host ring requires GPU arithmetic and progress before init");
      ring.reduction = cpu_router::ReductionPlacement::Gpu;
      ring.gpuProgress = true;
      ring.smallReduceScatterGpuProgress = false;
      CollectiveCall rs;
      rs.collective = Collective::ReduceScatter;
      rs.count = 8;
      for (int r = 0; r < 2; ++r)
        rs.ports.push_back({inputs[r].data(), outputs[r].data(), nullptr, r});
      int ringFallback = 0;
      auto fallback = [&](CollectiveCall const&) { ++ringFallback; };
      CollectiveDispatcher ringAuto(t, autoPolicy, fallback, ring);
      require(!ringAuto.prepare(rs).cpuRouter && !ringAuto.routerInitialized(),
              "unmeasured host ring must preserve allocation-free fallback");
      ringAuto.enqueue(rs);
      require(ringFallback == 1,
              "unmeasured host ring invokes caller fallback");
      auto flatOptions = ring;
      flatOptions.hostRingReduceScatter = false;
      CollectiveDispatcher flat(t, autoPolicy, fallback, flatOptions);
      require(ringAuto.topology().identity != flat.topology().identity,
              "ring reduction order and schedule require separate calibration");
      BackendPolicy oldEvidence;
      oldEvidence.fallbackIdentity = "test-fallback";
      oldEvidence.measurements.push_back({flat.topology().identity,
                                          oldEvidence.fallbackIdentity,
                                          rs.workload(),
                                          {8, 9, 10},
                                          {19, 20, 21},
                                          3,
                                          true});
      CollectiveDispatcher ringOld(t, oldEvidence, fallback, ring);
      require(!ringOld.prepare(rs).cpuRouter && !ringOld.routerInitialized(),
              "ring cannot reuse flat-GPU calibration");
      CollectiveDispatcher ringForce(t, policy, fallback, ring);
      for (int unsupported = 0; unsupported < 5; ++unsupported) {
        auto c = rs;
        if (unsupported == 0) c.collective = Collective::AllReduce;
        if (unsupported == 1) c.count = 7;
        if (unsupported == 2) c.dtype = ScalarType::Float16;
        if (unsupported == 3) c.ports[0].output = outputs[0].data() + 1;
        if (unsupported == 4) {
          c.inPlace = true;
          for (int r = 0; r < 2; ++r)
            c.ports[r].output = inputs[r].data() + r * c.count;
        }
        require(!ringAuto.prepare(c).cpuRouter && !ringAuto.routerInitialized(),
                "unsupported ring workload falls back before allocation");
        rejects([&] { ringForce.prepare(c); },
                "forced unsupported ring workload rejects before allocation");
        require(!ringForce.routerInitialized(),
                "rejected ring remains uninitialized");
      }
    }
    std::cout << "Dispatcher: " << checks
              << " checks passed; no GPU initialization needed\n";
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
