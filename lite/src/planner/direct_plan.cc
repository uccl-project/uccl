#include "uccl_lite/direct_plan.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>

namespace uccl_lite {
namespace {
size_t elements(Workload const& w, size_t ranks, size_t chunk) {
  size_t scalar = scalarBytes(w.dtype);
  if (ranks < 2 || ranks > 32 ||
      (w.dtype != ScalarType::Float32 && w.dtype != ScalarType::BFloat16) ||
      int(w.collective) < 0 || int(w.collective) > 2 || !w.bytes ||
      w.bytes > (1UL << 30) || chunk < scalar || chunk > (1UL << 20) ||
      chunk % scalar ||
      w.bytes % (scalar * (w.collective == Collective::AllReduce ? 1 : ranks)))
    throw std::invalid_argument("invalid direct model workload/extent");
  return w.bytes / (w.collective == Collective::AllReduce ? 1 : ranks);
}
size_t multiply(size_t a, size_t b) {
  if (a && b > std::numeric_limits<size_t>::max() / a)
    throw std::overflow_error("direct model traffic overflow");
  return a * b;
}
}  // namespace
std::vector<DirectStep> directExchangeSteps(Collective op, int ranks,
                                            int rank) {
  if (ranks < 2 || ranks > 32 || rank < 0 || rank >= ranks || int(op) < 0 ||
      int(op) > 2)
    throw std::invalid_argument("invalid direct schedule");
  std::vector<DirectStep> steps;
  if (op != Collective::ReduceScatter)
    steps.push_back({DirectStepKind::StageInput, rank});
  for (int target = 0; target < ranks; ++target) {
    if (op == Collective::ReduceScatter)
      steps.push_back({DirectStepKind::StageInput, target});
    steps.push_back({DirectStepKind::Transfer, target});
  }
  return steps;
}
DirectTraffic directExchangeTraffic(Workload const& w, size_t ranks,
                                    size_t chunk,
                                    std::vector<bool> const& ipc) {
  size_t bytes = elements(w, ranks, chunk);
  if (ipc.size() != ranks * ranks)
    throw std::invalid_argument("direct route matrix extent mismatch");
  size_t peers = 0;
  for (size_t r = 0; r < ranks; ++r) {
    if (ipc[r * ranks + r])
      throw std::invalid_argument("direct self route is not IPC");
    for (size_t p = 0; p < ranks; ++p)
      if (r != p) {
        if (ipc[r * ranks + p] != ipc[p * ranks + r])
          throw std::invalid_argument("asymmetric direct transport choice");
        peers += ipc[r * ranks + p];
      }
  }
  DirectTraffic result;
  result.chunks = (bytes + chunk - 1) / chunk;
  result.inputCopies = multiply(
      result.chunks,
      w.collective == Collective::ReduceScatter ? ranks * ranks : ranks);
  result.selfCopies = multiply(ranks, result.chunks);
  result.outputCopies = w.collective == Collective::AllGather
                            ? multiply(ranks * ranks, result.chunks)
                            : 0;
  result.reductionLaunches = w.collective == Collective::AllGather
                                 ? 0
                                 : multiply(ranks, result.chunks);
  result.callerStreamSynchronizations =
      ranks + result.inputCopies + result.selfCopies + ranks * result.chunks;
  result.transportFlushes = multiply(ranks * (ranks - 1), result.chunks);
  result.agreements = 3 + 2 * result.chunks;
  result.controlPayloadBytes =
      multiply(ranks * (ranks - 1),
               2 * sizeof(DirectStatus) + sizeof(DirectCallSignature) +
                   2 * result.chunks * sizeof(DirectStatus));
  result.peerPayloadBytes = multiply(peers, bytes);
  result.networkPayloadBytes = multiply(ranks * (ranks - 1) - peers, bytes);
  // Staging + self copy + each remote read/write + final consumption.
  size_t coefficient =
      w.collective == Collective::AllGather   ? 4 * ranks * ranks + 2 * ranks
      : w.collective == Collective::AllReduce ? 3 * ranks * ranks + 3 * ranks
                                              : 5 * ranks * ranks + ranks;
  result.gpuMemoryBytes = multiply(coefficient, bytes);
  return result;
}
PlanCandidate directExchangeModel(Topology const& t, Workload const& w,
                                  size_t chunk,
                                  std::vector<DirectPathModel> const& paths,
                                  DirectModelOverheads const& overhead,
                                  DirectOperationResources const* operations) {
  size_t ranks = t.devices.size(), bytes = elements(w, ranks, chunk);
  if (t.gpuNodes.size() != ranks || paths.size() != ranks * ranks ||
      std::set<size_t>(t.gpuNodes.begin(), t.gpuNodes.end()).size() != ranks)
    throw std::invalid_argument("invalid direct model topology");
  for (auto gpu : t.gpuNodes)
    if (gpu >= t.hardware.size() || t.hardware[gpu].kind != HardwareKind::Gpu)
      throw std::invalid_argument("invalid direct model GPU node");
  if (operations)
    for (auto const* names : {&operations->inputCopy, &operations->selfCopy,
                              &operations->gather, &operations->reduction}) {
      if (names->size() != ranks ||
          std::any_of(names->begin(), names->end(),
                      [](auto const& name) { return name.empty(); }))
        throw std::invalid_argument("incomplete direct operation resources");
    }
  size_t chunks = (bytes + chunk - 1) / chunk;
  size_t perChunk =
      ranks *
          (w.collective == Collective::ReduceScatter ? 2 * ranks : ranks + 1) +
      ranks + (w.collective == Collective::AllGather ? ranks * ranks : ranks) +
      ranks + 2;
  if (chunks > (65536 - 1) / perChunk)
    throw std::length_error("direct task graph exceeds model budget");
  PlanCandidate candidate;
  candidate.workload = w;
  candidate.topologyIdentity = t.identity;
  candidate.capability.collectives = {
      Collective::AllReduce, Collective::AllGather, Collective::ReduceScatter};
  candidate.capability.dtypes = {ScalarType::Float32, ScalarType::BFloat16};
  candidate.capability.minRanks = 2;
  candidate.capability.maxRanks = 32;
  candidate.capability.maxBytes = 1UL << 30;
  candidate.capability.alignment = scalarBytes(w.dtype);
  candidate.capability.inPlace = true;
  auto& plan = candidate.plan;
  plan.id = "direct-exchange-v1;bulk-synchronous-model-v1";
  plan.algorithm = Algorithm::Direct;
  plan.transport = Transport::Network;
  plan.protocol = Protocol::BulkCopy;
  plan.chunkBytes = chunk;
  plan.rankOrder.resize(ranks);
  std::iota(plan.rankOrder.begin(), plan.rankOrder.end(), 0);
  for (size_t r = 0; r < ranks; ++r)
    for (size_t p = 0; p < ranks; ++p) {
      auto const& path = paths[r * ranks + p];
      if (r == p && (!path.resources.empty() || path.cudaIpc))
        throw std::invalid_argument("direct self path must be empty");
      std::set<std::string> unique;
      for (auto const& resource : path.resources)
        if (resource.resource.empty() ||
            resource.resource.rfind("memory:", 0) == 0 ||
            !unique.insert(resource.resource).second)
          throw std::invalid_argument(
              "duplicate or invalid direct path resource");
      if (path.cudaIpc != paths[p * ranks + r].cudaIpc)
        throw std::invalid_argument("asymmetric direct transport choice");
      if (path.cudaIpc) plan.peerEdges.emplace_back(int(r), int(p));
    }
  auto memory = [&](size_t rank, size_t traffic) {
    return ResourceDemand{"memory:" + t.hardware[t.gpuNodes[rank]].id, traffic};
  };
  auto add = [&](std::string name, std::vector<ResourceDemand> demands,
                 std::vector<size_t> deps, double us) {
    size_t index = plan.tasks.size();
    plan.tasks.push_back(
        {std::move(name), std::move(demands), std::move(deps), us});
    return index;
  };
  size_t previous = add("entry", {}, {}, overhead.entryUs);
  for (size_t offset = 0, wave = 0; offset < bytes; offset += chunk, ++wave) {
    size_t payload = std::min(chunk, bytes - offset);
    std::string prefix = "chunk:" + std::to_string(wave) + ':';
    std::vector<size_t> sent, consumed;
    for (size_t rank = 0; rank < ranks; ++rank) {
      size_t last = previous, index = 0;
      for (auto step :
           directExchangeSteps(w.collective, int(ranks), int(rank))) {
        std::vector<ResourceDemand> demands;
        double cost;
        if (step.kind == DirectStepKind::StageInput ||
            step.target == int(rank)) {
          demands.push_back(memory(rank, 2 * payload));
          if (operations)
            demands.push_back({step.kind == DirectStepKind::StageInput
                                   ? operations->inputCopy[rank]
                                   : operations->selfCopy[rank],
                               2 * payload});
          cost = overhead.copyUs < 0 || overhead.streamSyncUs < 0
                     ? -1
                     : overhead.copyUs + overhead.streamSyncUs;
        } else {
          demands = {memory(rank, payload), memory(step.target, payload)};
          auto const& path = paths[rank * ranks + step.target];
          if (path.resources.empty())
            demands.push_back({"missing:direct-path:" + std::to_string(rank) +
                                   ':' + std::to_string(step.target),
                               payload});
          for (auto const& resource : path.resources)
            demands.push_back(
                {resource.resource, payload, resource.physicalCapacityBound});
          cost = overhead.transferFlushUs;
        }
        last = add(prefix + "rank:" + std::to_string(rank) +
                       ":send:" + std::to_string(index++),
                   std::move(demands), {last}, cost);
      }
      sent.push_back(last);
    }
    auto publication =
        add(prefix + "publication", {}, sent, overhead.publicationUs);
    for (size_t rank = 0; rank < ranks; ++rank) {
      size_t last = add(prefix + "visibility:" + std::to_string(rank), {},
                        {publication}, overhead.visibilityUs);
      if (w.collective == Collective::AllGather) {
        if (operations)
          last = add(prefix + "gather-output:" + std::to_string(rank),
                     {memory(rank, 2 * ranks * payload),
                      {operations->gather[rank], 2 * ranks * payload}},
                     {last}, overhead.copyUs);
        else
          for (size_t source = 0; source < ranks; ++source)
            last = add(prefix + "copy-output:" + std::to_string(rank) + ':' +
                           std::to_string(source),
                       {memory(rank, 2 * payload)}, {last}, overhead.copyUs);
      } else {
        std::vector<ResourceDemand> demands{
            memory(rank, (ranks + 1) * payload)};
        if (operations)
          demands.push_back(
              {operations->reduction[rank], (ranks + 1) * payload});
        last = add(prefix + "reduce:" + std::to_string(rank),
                   std::move(demands), {last}, overhead.reductionUs);
      }
      consumed.push_back(add(prefix + "complete:" + std::to_string(rank), {},
                             {last}, overhead.streamSyncUs));
    }
    previous = add(prefix + "retirement", {}, consumed, overhead.retirementUs);
  }
  return candidate;
}
PlanEstimate estimateDirectModel(DirectModelCalibration const& calibration,
                                 Workload const& workload) {
  PlanEstimate unavailable;
  unavailable.reason = "direct model calibration/workload mismatch";
  if (calibration.identity.empty() ||
      calibration.topology.identity != calibration.identity ||
      workload.dtype != calibration.dtype)
    return unavailable;
  try {
    auto candidate = directExchangeModel(
        calibration.topology, workload, calibration.chunkBytes,
        calibration.paths, calibration.overheads, &calibration.operations);
    auto topology = calibration.topology;
    // Convert operation observations to exact-size service resources. Positive
    // durations need not increase with size; forcing a linear bandwidth slope
    // would discard real dispatch/cache behavior or fabricate a positive rate.
    std::map<std::string, std::map<size_t, std::vector<double>>> tables;
    std::map<std::string, size_t> factors;
    std::map<std::string, std::string> scopes;
    for (auto const& observation : calibration.observations) {
      if (!observation.trafficFactor) continue;
      if (!observation.rejection.empty() || !observation.bytes ||
          observation.microseconds.empty())
        continue;
      if (observation.scope.empty() || observation.microseconds.size() < 3 ||
          observation.bytes % scalarBytes(calibration.dtype))
        throw std::invalid_argument(
            "invalid direct operation measurement scope");
      auto [scope, addedScope] =
          scopes.emplace(observation.resource, observation.scope);
      if (!addedScope && scope->second != observation.scope)
        throw std::invalid_argument(
            "incompatible direct operation measurement scopes");
      auto [it, inserted] =
          factors.emplace(observation.resource, observation.trafficFactor);
      if (!inserted && it->second != observation.trafficFactor)
        throw std::invalid_argument("inconsistent operation traffic scope");
      auto& samples = tables[observation.resource][observation.bytes];
      for (double sample : observation.microseconds) {
        if (!std::isfinite(sample) || sample <= 0)
          throw std::invalid_argument("invalid operation service observation");
        samples.push_back(sample);
      }
    }
    auto median = [](std::vector<double> samples) {
      std::sort(samples.begin(), samples.end());
      size_t n = samples.size();
      return n % 2 ? samples[n / 2]
                   : samples[n / 2 - 1] / 2 + samples[n / 2] / 2;
    };
    std::set<std::string> required;
    for (auto const* names :
         {&calibration.operations.inputCopy, &calibration.operations.selfCopy,
          &calibration.operations.gather, &calibration.operations.reduction})
      required.insert(names->begin(), names->end());
    for (auto const& path : calibration.paths)
      for (auto const& resource : path.resources)
        if (!resource.physicalCapacityBound) required.insert(resource.resource);
    std::set<std::string> inserted;
    for (auto& task : candidate.plan.tasks)
      for (auto& demand : task.demands) {
        auto table = tables.find(demand.resource);
        if (table == tables.end()) {
          if (required.count(demand.resource))
            throw std::invalid_argument("direct operation observation missing");
          continue;
        }
        size_t factor = factors.at(demand.resource);
        if (demand.bytes % factor)
          throw std::invalid_argument("operation traffic factor mismatch");
        size_t bytes = demand.bytes / factor;
        auto upper = table->second.lower_bound(bytes);
        if (upper == table->second.end() ||
            (upper == table->second.begin() && upper->first != bytes))
          throw std::invalid_argument(
              "direct service observation does not cover payload");
        double duration = median(upper->second);
        if (upper->first != bytes) {
          auto lower = std::prev(upper);
          double fraction = double(bytes - lower->first) /
                            double(upper->first - lower->first);
          duration =
              median(lower->second) * (1 - fraction) + duration * fraction;
        }
        std::string scoped =
            demand.resource + ";payload=" + std::to_string(bytes);
        if (inserted.insert(scoped).second)
          topology.links.push_back({0,
                                    0,
                                    scoped,
                                    {double(demand.bytes) / duration / 1000, 0,
                                     Evidence::Measured}});
        demand.resource = std::move(scoped);
      }
    return choosePlan(topology, workload, false, {candidate})
        .candidates.front();
  } catch (std::exception const& error) {
    unavailable.reason = error.what();
    return unavailable;
  }
}
}  // namespace uccl_lite
