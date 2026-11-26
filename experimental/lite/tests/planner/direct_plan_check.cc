#include "uccl_lite/direct_plan.hpp"
#include "uccl_lite/peer_plan.hpp"
#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
using namespace uccl_lite;
void require(bool ok, char const* message) {
  if (!ok) throw std::runtime_error(message);
}
template <class F>
void reject(F fn) {
  bool bad = false;
  try {
    fn();
  } catch (std::exception const&) {
    bad = true;
  }
  require(bad, "invalid direct model accepted");
}
Topology topology(int n) {
  Topology t;
  t.identity = "synthetic-global-topology";
  t.peerAccess.assign(n * n, true);
  for (int i = 0; i < n; ++i) {
    t.devices.push_back(i);
    t.gpuNodes.push_back(i);
    t.hardware.push_back({"gpu" + std::to_string(i),
                          HardwareKind::Gpu,
                          i,
                          0,
                          {10000, 0, Evidence::Measured}});
  }
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      if (i != j) {
        auto suffix = std::to_string(i) + ":" + std::to_string(j);
        t.links.push_back({size_t(i),
                           size_t(j),
                           "wire:" + suffix,
                           {1000, 0, Evidence::Advertised}});
        t.paths.push_back({size_t(i),
                           size_t(j),
                           PathKind::PeerCopy,
                           "copy:" + suffix,
                           {100, 0, Evidence::Measured}});
        t.paths.push_back({size_t(i),
                           size_t(j),
                           PathKind::PeerRead,
                           "read:" + suffix,
                           {10, 0, Evidence::Measured}});
      }
  return t;
}
std::vector<DirectPathModel> routes(int n) {
  std::vector<DirectPathModel> result(n * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      if (i != j) {
        auto suffix = std::to_string(i) + ":" + std::to_string(j);
        result[i * n + j] = {
            true, {{"copy:" + suffix, false}, {"wire:" + suffix, true}}};
      }
  return result;
}
int main() {
  DirectModelOverheads zero{0, 0, 0, 0, 0, 0, 0, 0};
  for (int n : {2, 3, 4, 8, 32})
    for (auto type : {ScalarType::Float32, ScalarType::BFloat16})
      for (auto op : {Collective::AllReduce, Collective::AllGather,
                      Collective::ReduceScatter}) {
        size_t scalar = scalarBytes(type), count = 257, bytes = count * scalar;
        Workload w{op, type, bytes * (op == Collective::AllReduce ? 1 : n),
                   false, unsigned(scalar)};
        std::vector<bool> ipc(n * n, false);
        // Independent enumeration of a two-host placement, with unequal sizes.
        size_t peerEdges = 0;
        for (int r = 0; r < n; ++r)
          for (int p = 0; p < n; ++p)
            if (r != p && (r < n / 2) == (p < n / 2)) {
              ipc[r * n + p] = true;
              ++peerEdges;
            }
        auto traffic = directExchangeTraffic(w, n, 256, ipc);
        require(traffic.chunks == (bytes + 255) / 256, "tail chunk count");
        size_t loads = 0, stores = 0, inputCopies = 0, flushes = 0;
        for (int r = 0; r < n; ++r) {
          // Input staging, source-owned receive slot, then all remote writes.
          size_t stages = op == Collective::ReduceScatter ? n : 1;
          loads += stages * bytes + bytes + (n - 1) * bytes;
          stores += stages * bytes + bytes + (n - 1) * bytes;
          inputCopies += stages * traffic.chunks;
          flushes += (n - 1) * traffic.chunks;
          loads += n * bytes;
          stores += (op == Collective::AllGather ? n : 1) * bytes;
          auto steps = directExchangeSteps(op, n, r);
          int transfers = 0, staging = 0, lastInput = -1;
          for (auto step : steps) {
            if (step.kind == DirectStepKind::StageInput) {
              ++staging;
              lastInput = step.target;
            } else {
              require(step.target == transfers++, "noncanonical send order");
              require(lastInput ==
                          (op == Collective::ReduceScatter ? step.target : r),
                      "wrong staged input source");
            }
          }
          require(transfers == n && staging == int(stages),
                  "schedule step count");
        }
        require(traffic.gpuMemoryBytes == loads + stores,
                "GPU memory accounting");
        require(traffic.inputCopies == inputCopies &&
                    traffic.transportFlushes == flushes,
                "operation counts");
        require(traffic.peerPayloadBytes == peerEdges * bytes &&
                    traffic.networkPayloadBytes ==
                        (n * (n - 1) - peerEdges) * bytes,
                "route traffic accounting");
        auto t = topology(n);
        auto paths = routes(n);
        auto candidate = directExchangeModel(t, w, 256, paths, zero);
        std::map<std::string, size_t> ids;
        for (size_t i = 0; i < candidate.plan.tasks.size(); ++i)
          require(ids.emplace(candidate.plan.tasks[i].id, i).second,
                  "duplicate task ID");
        auto deps = [&](std::string const& name, std::vector<size_t> expected) {
          require(candidate.plan.tasks[ids.at(name)].dependencies == expected,
                  "direct dependency differs from execution contract");
        };
        size_t prior = ids.at("entry");
        for (size_t wave = 0; wave < traffic.chunks; ++wave) {
          auto prefix = "chunk:" + std::to_string(wave) + ':';
          std::vector<size_t> sendEnds, consumeEnds;
          for (int rank = 0; rank < n; ++rank) {
            size_t last = prior;
            size_t steps =
                size_t(op == Collective::ReduceScatter ? 2 * n : n + 1);
            for (size_t step = 0; step < steps; ++step) {
              auto id = prefix + "rank:" + std::to_string(rank) +
                        ":send:" + std::to_string(step);
              deps(id, {last});
              last = ids.at(id);
            }
            sendEnds.push_back(last);
          }
          deps(prefix + "publication", sendEnds);
          for (int rank = 0; rank < n; ++rank) {
            auto visibility = prefix + "visibility:" + std::to_string(rank);
            deps(visibility, {ids.at(prefix + "publication")});
            size_t last = ids.at(visibility);
            if (op == Collective::AllGather)
              for (int source = 0; source < n; ++source) {
                auto id = prefix + "copy-output:" + std::to_string(rank) + ':' +
                          std::to_string(source);
                deps(id, {last});
                last = ids.at(id);
              }
            else {
              auto id = prefix + "reduce:" + std::to_string(rank);
              deps(id, {last});
              last = ids.at(id);
            }
            auto complete = prefix + "complete:" + std::to_string(rank);
            deps(complete, {last});
            consumeEnds.push_back(ids.at(complete));
          }
          deps(prefix + "retirement", consumeEnds);
          prior = ids.at(prefix + "retirement");
        }
        size_t memory = 0, pathBytes = 0;
        for (auto const& task : candidate.plan.tasks)
          for (auto const& d : task.demands) {
            if (d.resource.rfind("memory:", 0) == 0) memory += d.bytes;
            if (d.resource.rfind("copy:", 0) == 0) pathBytes += d.bytes;
          }
        require(memory == loads + stores &&
                    pathBytes == size_t(n) * (n - 1) * bytes,
                "graph and independent byte oracle differ");
      }
  auto t = topology(4);
  auto paths = routes(4);
  Workload w{Collective::AllReduce, ScalarType::BFloat16, 65536, false, 16};
  auto direct = directExchangeModel(t, w, 65536, paths, zero);
  PeerTuning tuning;
  tuning.overheadUs = {0, 0, 0};
  auto peer = peerCandidates(t, w, tuning).front();
  auto fast = choosePlan(t, w, false, {peer, direct});
  require(fast.index && *fast.index == 1,
          "fast memory/copy paths did not favor staged direct model");
  for (auto& node : t.hardware) node.memory.bandwidthGBs = 1;
  auto slow = choosePlan(t, w, false, {peer, direct});
  require(slow.index && *slow.index == 0,
          "memory pressure did not change model choice");
  t = topology(4);
  auto unknown = directExchangeModel(t, w, 65536, paths);
  require(!choosePlan(t, w, false, {unknown}).index,
          "unknown control cost became zero");
  t.hardware[0].memory = {};
  require(!choosePlan(t, w, false, {direct}).index,
          "unknown GPU memory became usable");
  t = topology(4);
  paths[1].resources.clear();
  require(
      !choosePlan(t, w, false, {directExchangeModel(t, w, 65536, paths, zero)})
           .index,
      "missing path became free");
  paths = routes(4);
  for (auto& path : paths)
    for (auto& r : path.resources)
      if (!r.physicalCapacityBound) {
        r.resource = "shared-nic";
        path.cudaIpc = false;
      }
  t.links.push_back({0, 1, "shared-nic", {1, 0, Evidence::Measured}});
  auto shared =
      choosePlan(t, w, false, {directExchangeModel(t, w, 65536, paths, zero)});
  require(
      shared.index && shared.candidates[0].resourceLoadUs >= 12. * 65536 / 1000,
      "NIC shared load omitted");
  auto overhead = zero;
  overhead.publicationUs = 10;
  overhead.retirementUs = 20;
  auto a = choosePlan(t, w, false,
                      {directExchangeModel(t, w, 65536, paths, overhead)});
  auto b = choosePlan(t, w, false,
                      {directExchangeModel(t, w, 32768, paths, overhead)});
  require(a.index && b.index &&
              b.candidates[0].microseconds > a.candidates[0].microseconds,
          "chunk control overhead missing");
  reject([&] { directExchangeModel(t, w, 1, paths, zero); });
  reject([&] {
    auto p = paths;
    p[1].resources.push_back({"memory:gpu0", false});
    directExchangeModel(t, w, 65536, p, zero);
  });
  reject([&] {
    auto p = paths;
    p[1].cudaIpc = true;
    directExchangeModel(t, w, 65536, p, zero);
  });
  reject([&] {
    auto huge = w;
    huge.bytes = 1UL << 30;
    directExchangeModel(t, huge, 2, paths, zero);
  });
  DirectModelCalibration calibration;
  calibration.topology = topology(2);
  calibration.identity = calibration.topology.identity;
  calibration.dtype = ScalarType::Float32;
  calibration.chunkBytes = 1024;
  calibration.paths = routes(2);
  calibration.overheads = zero;
  auto service = [&](std::string name, size_t factor) {
    calibration.observations.push_back(
        {name, "synthetic-nonmonotonic-service", {}, 4, factor, {10, 10, 10}});
    calibration.observations.push_back(
        {name, "synthetic-nonmonotonic-service", {}, 1024, factor, {5, 5, 5}});
  };
  for (int rank = 0; rank < 2; ++rank) {
    auto suffix = std::to_string(rank);
    calibration.operations.inputCopy.push_back("input" + suffix);
    service("input" + suffix, 2);
    calibration.operations.selfCopy.push_back("self" + suffix);
    service("self" + suffix, 2);
    calibration.operations.gather.push_back("gather" + suffix);
    service("gather" + suffix, 4);
    calibration.operations.reduction.push_back("reduce" + suffix);
    service("reduce" + suffix, 3);
  }
  service("copy:0:1", 1);
  service("copy:1:0", 1);
  Workload sampled{Collective::AllGather, ScalarType::Float32, 2 * 512, false,
                   4};
  require(estimateDirectModel(calibration, sampled).estimated,
          "nonmonotonic positive service samples were discarded");
  auto noData = calibration;
  noData.observations.clear();
  require(!estimateDirectModel(noData, sampled).estimated,
          "missing raw operation samples were ignored");
  auto missing = calibration;
  missing.observations.erase(missing.observations.begin(),
                             missing.observations.begin() + 2);
  require(!estimateDirectModel(missing, sampled).estimated,
          "partial operation coverage accepted");
  auto outside = calibration;
  outside.chunkBytes = 2048;
  sampled.bytes = 4096;
  require(!estimateDirectModel(outside, sampled).estimated,
          "operation service was extrapolated");
  auto mixedScope = calibration;
  mixedScope.observations[1].scope = "different-timing-boundary";
  require(!estimateDirectModel(mixedScope, sampled).estimated,
          "different timing scopes were combined");
  auto wrong = calibration;
  wrong.topology.identity = "different";
  sampled.bytes = 1024;
  require(!estimateDirectModel(wrong, sampled).estimated,
          "calibration identity mismatch ignored");
  std::cout
      << "PASS direct model: independent traffic oracle, source ordering, "
         "memory-driven choice, shared NIC, controls, tails and bounds\n";
}
