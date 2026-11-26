#include "uccl_lite/router_plan.hpp"
#include "uccl_lite/hardware_calibration.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace uccl_lite {
namespace {
std::vector<int> groups(Topology const& t) {
  std::vector<int> result;
  for (int node : t.numaNodes) {
    if (node < 0)
      throw std::invalid_argument(
          "router worker planning needs known NUMA placement");
    if (std::find(result.begin(), result.end(), node) == result.end())
      result.push_back(node);
  }
  return result;
}
void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
struct DeviceGuard {
  int device;
  DeviceGuard() { check(cudaGetDevice(&device)); }
  ~DeviceGuard() { cudaSetDevice(device); }
};
}  // namespace
std::vector<RouterConfiguration> routerWorkerConfigurations(
    Topology const& t, cpu_router::CpuRouterOptions const& base,
    std::vector<int> const& counts) {
  if (!base.workerCpus.empty() || !base.numaNodes.empty() ||
      base.replayCpu >= 0 || base.smallAllReduceWorkersPerNode ||
      base.smallReduceScatterWorkersPerNode)
    throw std::invalid_argument(
        "automatic worker candidates need an unbound, constant-worker base "
        "configuration");
  if (t.numaNodes.size() != t.devices.size() ||
      std::any_of(t.numaNodes.begin(), t.numaNodes.end(),
                  [](int node) { return node < 0; }))
    return {};  // Unknown placement keeps the caller's existing backend.
  auto nodes = groups(t);
  std::set<int> seen;
  std::vector<RouterConfiguration> result;
  for (int count : counts) {
    if (count < 1 || count > 63 || !seen.insert(count).second)
      throw std::invalid_argument("invalid worker candidate count");
    auto options = base;
    options.workersPerNode = count;
    options.numaNodes = t.numaNodes;
    bool available = !nodes.empty();
    for (int node : nodes) {
      auto cpus = t.permittedCpuIds.find(node);
      if (cpus == t.permittedCpuIds.end() ||
          count > std::max(1, int(cpus->second.size()) - 1) ||
          cpus->second.size() < size_t(count)) {
        available = false;
        break;
      }
      options.workerCpus.insert(options.workerCpus.end(), cpus->second.begin(),
                                cpus->second.begin() + count);
    }
    if (available)
      result.push_back(
          {"workers-per-node=" + std::to_string(count), std::move(options)});
  }
  return result;
}
RouterPlanDispatcher::RouterPlanDispatcher(
    Topology t, BackendPolicy policy,
    std::vector<RouterConfiguration> configurations, Fallback fallback)
    : topology_(std::move(t)),
      policy_(std::move(policy)),
      configurations_(std::move(configurations)),
      fallback_(std::move(fallback)) {
  if (!fallback_) throw std::invalid_argument("caller fallback is required");
  if (policy_.mode == RouterMode::Force)
    throw std::invalid_argument(
        "multi-configuration selection requires measured admission; force "
        "individual configurations for calibration");
  auto off = policy_;
  off.mode = RouterMode::Off;
  validator_ =
      std::make_unique<CollectiveDispatcher>(topology_, off, fallback_);
  std::set<std::string> identities;
  for (auto const& config : configurations_) {
    auto candidate = std::make_unique<CollectiveDispatcher>(
        topology_, policy_, fallback_, config.options);
    if (!identities.insert(candidate->topology().identity).second)
      throw std::invalid_argument("duplicate router configuration");
    candidates_.push_back(std::move(candidate));
  }
}
Topology const& RouterPlanDispatcher::candidateTopology(size_t i) const {
  return candidates_.at(i)->topology();
}
std::optional<double> RouterPlanDispatcher::memoryPrior(
    CollectiveCall const& call, size_t index) const {
  auto const& options = configurations_[index].options;
  // Copy/AG in the packet pipeline publishes readiness; it does not execute
  // the memcpy payload schedule measured by the host-copy probe. GPU reduction
  // also needs a different model. Do not invent a CPU-compute estimate for
  // them.
  if (call.collective == Collective::AllGather ||
      options.reduction != cpu_router::ReductionPlacement::Cpu ||
      options.gpuProgress || options.smallAllReduceWorkersPerNode ||
      options.smallReduceScatterWorkersPerNode ||
      (!options.numaNodes.empty() && options.numaNodes != topology_.numaNodes))
    return std::nullopt;
  if (call.count < size_t(options.workersPerNode) * options.grainElements)
    return std::nullopt;
  auto t = topology_;
  HostProbeOptions probe;
  probe.workersPerNode = options.workersPerNode;
  auto paths = registerHostCopyPaths(t, probe);
  auto nodes = groups(t);
  double worst = 0;
  for (size_t group = 0; group < nodes.size(); ++group) {
    int node = nodes[group];
    auto path =
        std::find_if(paths.begin(), paths.end(), [&](HostCopyPath const& p) {
          return p.sourceNode == node && p.destinationNode == node;
        });
    if (path == paths.end() ||
        options.workerCpus.size() !=
            nodes.size() * size_t(options.workersPerNode))
      return std::nullopt;
    auto begin = options.workerCpus.begin() + group * options.workersPerNode;
    if (!std::equal(path->cpus.begin(), path->cpus.end(), begin,
                    begin + options.workersPerNode))
      return std::nullopt;
    auto measured = std::find_if(
        topology_.paths.begin(), topology_.paths.end(),
        [&](HardwarePath const& p) { return p.resource == path->resource; });
    if (measured == topology_.paths.end() ||
        measured->performance.evidence != Evidence::Measured ||
        !std::isfinite(measured->performance.bandwidthGBs) ||
        measured->performance.bandwidthGBs <= 0)
      return std::nullopt;
    size_t sources = std::count(topology_.numaNodes.begin(),
                                topology_.numaNodes.end(), node);
    // Advisory local partial-reduction traffic: source reads plus one result
    // write. A copy payload implies two memory accesses. Arithmetic, cross-NUMA
    // merge, GPU transport and synchronization are deliberately not estimated.
    double copyEquivalent = double(call.workload().bytes) * (sources + 1) / 2;
    worst = std::max(
        worst, copyEquivalent / measured->performance.bandwidthGBs / 1000);
  }
  return worst;
}
RouterPlanChoice RouterPlanDispatcher::decision(
    CollectiveCall const& call) const {
  validator_->decision(call);
  RouterPlanChoice result;
  result.admission = {
      false, "no exact admitted router configuration; keep caller backend"};
  double best = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < candidates_.size(); ++i) {
    auto admitted = candidates_[i]->decision(call);
    if (!admitted.cpuRouter) continue;
    if (!std::isfinite(admitted.routerMicroseconds) ||
        admitted.routerMicroseconds <= 0)
      throw std::logic_error(
          "admitted router configuration has no measured latency");
    auto prior = memoryPrior(call, i);
    if (admitted.routerMicroseconds < best ||
        (admitted.routerMicroseconds == best && prior &&
         (!result.memoryPriorUs || *prior < *result.memoryPriorUs))) {
      best = admitted.routerMicroseconds;
      result = {i, admitted, prior};
    }
  }
  return result;
}
RouterPlanDispatcher::Key RouterPlanDispatcher::key(
    CollectiveCall const& call) {
  auto w = call.workload();
  return {w.collective, w.dtype, w.bytes, w.inPlace, w.alignment};
}
RouterPlanChoice RouterPlanDispatcher::prepare(CollectiveCall const& call) {
  auto selected = decision(call);
  if (!active_ && !selected.configuration) {
    prepared_[key(call)] = selected;
    return selected;
  }
  DeviceGuard restore;
  // Reject capture before tearing down the preceding active implementation.
  for (auto const& port : call.ports) {
    check(cudaSetDevice(port.device));
    cudaStreamCaptureStatus capture;
    check(cudaStreamIsCapturing(port.stream, &capture));
    if (capture != cudaStreamCaptureStatusNone)
      throw std::invalid_argument(
          "router configuration preparation is outside capture");
    check(cudaStreamSynchronize(port.stream));
  }
  if (activeIndex_ != selected.configuration) {
    active_.reset();
    activeIndex_.reset();
  }
  if (selected.configuration) {
    if (!active_) {
      auto candidate = std::make_unique<CollectiveDispatcher>(
          topology_, policy_, fallback_,
          configurations_[*selected.configuration].options);
      candidate->prepare(call);
      active_ = std::move(candidate);
      activeIndex_ = selected.configuration;
    } else
      active_->prepare(call);
  }
  prepared_[key(call)] = selected;
  return selected;
}
RouterPlanChoice RouterPlanDispatcher::enqueue(CollectiveCall const& call) {
  validator_->decision(call);
  auto found = prepared_.find(key(call));
  if (found == prepared_.end()) {
    // Preserve allocation-free fallback for unprepared calls. An admitted
    // router must be explicitly prepared; no model is evaluated here.
    for (auto const& candidate : candidates_)
      if (candidate->decision(call).cpuRouter)
        throw std::logic_error("prepare the router plan before enqueue");
    fallback_(call);
    return {std::nullopt,
            {false, "unprepared fallback; no admitted router"},
            std::nullopt};
  }
  auto selected = found->second;
  if (!selected.configuration) {
    fallback_(call);
    return selected;
  }
  if (!active_ || activeIndex_ != selected.configuration)
    throw std::logic_error(
        "prepare the selected router configuration before enqueue");
  active_->enqueue(call);
  return selected;
}
bool RouterPlanDispatcher::routerInitialized() const {
  return active_ && active_->routerInitialized();
}
}  // namespace uccl_lite
