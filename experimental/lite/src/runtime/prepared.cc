#include "uccl_lite/runtime/prepared.hpp"
#include "uccl_lite/runtime/prepared_local.hpp"
#include "uccl_lite/runtime/prepared_network.hpp"
#include <mscclpp/core.hpp>
#include <algorithm>
#include <array>
#include <cstring>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

namespace uccl_lite::runtime {
namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
uint64_t hash(std::string const& text) {
  uint64_t value = 14695981039346656037ULL;
  for (unsigned char c : text) value = (value ^ c) * 1099511628211ULL;
  return value;
}
std::string key(CollectiveRequest const& r) {
  std::ostringstream s;
  s << int(r.operation) << ':' << r.count << ':' << r.fallbackOnly;
  for (auto l : r.layouts) s << ';' << l.inPlace << ':' << l.alignment;
  return s.str();
}
unsigned alignment(void const* input, void const* output) {
  auto bits =
      reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(output);
  unsigned result = 1;
  while (result < 256 && !(bits & result)) result *= 2;
  return result;
}
RuntimeBackend backend(LocalBackend kind) {
  switch (kind) {
    case LocalBackend::Peer:
      return RuntimeBackend::Peer;
    case LocalBackend::Host:
      return RuntimeBackend::Host;
    case LocalBackend::PeerPacket:
      return RuntimeBackend::PeerPacket;
    case LocalBackend::PeerTree:
      return RuntimeBackend::PeerTree;
    default:
      return RuntimeBackend::Caller;
  }
}
PreparedSelection report(LocalSelection const& s) {
  PreparedSelection result;
  result.request = s.request;
  result.backend = backend(s.backend);
  result.graph = s.graph;
  result.topologyIdentity = s.topologyIdentity;
  result.executorIdentity = s.executorIdentity;
  result.baselineIdentity = s.fallbackIdentity;
  result.reason = s.reason;
  for (size_t i = 0; i < kLocalAlternatives.size(); ++i) {
    CandidateEvidence c;
    c.backend = backend(kLocalAlternatives[i]);
    c.eligible = i == 0   ? s.peerEligible
                 : i == 1 ? s.hostEligible
                 : i == 2 ? s.peerPacketEligible
                          : s.peerTreeEligible;
    c.correct = s.candidateCorrect[i];
    c.samples = s.samples[size_t(kLocalAlternatives[i])];
    c.baselineSamples = s.fallbackSamples[i];
    if (i == 0) c.resourceEstimateUs = s.peerResourceEstimateUs;
    if (i == 2) c.executionRankOrder = s.peerPacketRankOrder;
    if (i == 3) {
      c.peerTreeParents = s.peerTreeParents;
      c.resourceEstimateUs = s.peerTreeResourceEstimateUs;
      c.resourceSerialUs = s.peerTreeSerialUs;
      c.resourceDependencyUs = s.peerTreeDependencyUs;
      c.resourceLoadUs = s.peerTreeResourceLoadUs;
    }
    result.candidates.push_back(std::move(c));
  }
  return result;
}
PreparedSelection report(NetworkSelection const& s) {
  PreparedSelection result;
  result.request = s.request;
  result.fromCache = (s.fromCache || s.direct.fromCache) &&
                     (!s.candidateEligible || s.fromCache) &&
                     (!s.direct.eligible || s.direct.fromCache) &&
                     s.poolCandidateSamples.empty();
  result.cacheStatus =
      "hierarchy: " + s.cacheStatus + "; direct: " + s.direct.cacheStatus;
  result.backend = s.backend == NetworkBackend::Hierarchy
                       ? RuntimeBackend::NetworkHierarchy
                   : s.backend == NetworkBackend::DirectExchange
                       ? RuntimeBackend::DirectExchange
                       : RuntimeBackend::Caller;
  result.poolCandidateSamples = s.poolCandidateSamples;
  result.poolBaselineSamples = s.poolBaselineSamples;
  result.topologyIdentity = s.topologyIdentity;
  result.executorIdentity = s.executorIdentity;
  result.baselineIdentity = s.baselineIdentity;
  result.reason = s.reason;
  CandidateEvidence c;
  c.backend = RuntimeBackend::NetworkHierarchy;
  c.fromCache = s.fromCache;
  c.cacheStatus = s.cacheStatus;
  c.eligible = s.candidateEligible;
  c.correct = s.candidateCorrect;
  c.samples = s.candidateSamples;
  c.baselineSamples = s.baselineSamples;
  c.executionRankOrder = s.executionRankOrder;
  c.rails = s.rails;
  c.networkCalibration = s.calibration;
  c.validationSamples = s.validationCandidateSamples;
  c.validationBaselineSamples = s.validationBaselineSamples;
  result.candidates.push_back(std::move(c));
  CandidateEvidence d;
  d.backend = RuntimeBackend::DirectExchange;
  d.eligible = s.direct.eligible;
  d.correct = s.direct.correct;
  d.samples = s.direct.samples;
  d.baselineSamples = s.direct.baselineSamples;
  d.configurationIdentities = s.direct.configurationIdentities;
  d.traffic = s.direct.traffic;
  d.resourceModelIdentity = s.direct.modelIdentity;
  d.resourceModelReason = s.direct.model.reason;
  d.resourceModelObservations = s.direct.modelObservations;
  d.unmodeledResources = s.direct.unmodeledResources;
  if (s.direct.model.estimated) {
    d.resourceEstimateUs = s.direct.model.microseconds;
    d.resourceSerialUs = s.direct.model.serialUs;
    d.resourceDependencyUs = s.direct.model.dependencyOnlyUs;
    d.resourceLoadUs = s.direct.model.resourceLoadUs;
  }
  d.fromCache = s.direct.fromCache;
  d.cacheStatus = s.direct.cacheStatus;
  d.validationSamples = s.direct.validationSamples;
  d.validationBaselineSamples = s.direct.validationBaselineSamples;
  result.candidates.push_back(std::move(d));
  return result;
}
}  // namespace
class PreparedRuntime::Impl {
 public:
  RuntimeContext context;
  PreparationOptions options;
  std::shared_ptr<mscclpp::TcpBootstrap> control;
  std::unique_ptr<PreparedLocalRuntime> local;
  std::unique_ptr<PreparedNetworkRuntime> distributed;
  RuntimePlacement placement;
  std::string fingerprint;
  uint64_t sequence = 0;
  bool valid = false, recorded = false;
  cudaEvent_t done = nullptr;
  cudaStream_t last = nullptr;
  struct Plan {
    size_t child;
    PreparedSelection selection;
  };
  struct Graph {
    size_t child;
    bool local, active = true;
    PreparedSelection selection;
  };
  std::vector<Plan> plans;
  std::vector<Graph> graphs;
  Impl(RuntimeContext c, PreparationOptions o)
      : context(c), options(std::move(o)) {
    if (c.ranks < 1 || c.ranks > 32 || c.rank < 0 || c.rank >= c.ranks ||
        c.device < 0 || !c.capacity || c.capacity > (1UL << 30))
      throw std::invalid_argument("invalid runtime participant context");
  }
  ~Impl() {
    int previous = -1;
    cudaGetDevice(&previous);
    cudaSetDevice(context.device);
    if (recorded) cudaEventSynchronize(done);
    try {
      synchronize();
    } catch (...) {
    }
    for (auto const& g : graphs)
      if (g.active && !g.local) {
        try {
          options.fallback.backend->releaseGraph(g.child);
        } catch (...) {
        }
      }
    local.reset();
    distributed.reset();
    if (done) cudaEventDestroy(done);
    if (previous >= 0 && previous != context.device) cudaSetDevice(previous);
  }
  template <class T>
  std::vector<T> gather(T const& item) {
    std::vector<T> result(context.ranks);
    result[context.rank] = item;
    control->allGather(result.data(), sizeof(T));
    return result;
  }
  void agreed(bool ready, char const* message) {
    auto votes = gather(int(ready));
    if (std::find(votes.begin(), votes.end(), 0) != votes.end())
      throw std::runtime_error(message);
  }
  bool idle() const {
    return (!recorded || cudaEventQuery(done) == cudaSuccess) &&
           (!local || local->idle()) && (!distributed || distributed->idle());
  }
  void synchronize() {
    if (recorded) check(cudaEventSynchronize(done));
    if (local) local->synchronize();
    if (distributed) distributed->synchronize();
  }
  bool requestValid(CollectiveRequest const& r) const {
    size_t scalar = scalarBytes(options.dtype);
    int op = int(r.operation);
    if (!scalar || op < 1 || op > 3 || !r.count ||
        r.count > context.capacity / (scalar * (op == 3 ? 1 : context.ranks)) ||
        r.layouts.size() != size_t(context.ranks))
      return false;
    for (int rank = 0; rank < context.ranks; ++rank) {
      auto l = r.layouts[rank];
      if (l.alignment < scalar || l.alignment > 256 ||
          (l.alignment & (l.alignment - 1)) ||
          (l.inPlace && op != 3 &&
           (size_t(rank) * r.count * scalar) % l.alignment))
        return false;
    }
    return true;
  }
  bool bindings(CollectiveRequest const& r, void const* input,
                void* output) const {
    if (!requestValid(r)) return false;
    size_t scalar = scalarBytes(options.dtype), count = r.count;
    int op = int(r.operation);
    auto l = r.layouts[context.rank];
    auto a = reinterpret_cast<uintptr_t>(input),
         b = reinterpret_cast<uintptr_t>(output);
    size_t in = count * scalar * (op == 2 ? context.ranks : 1),
           out = count * scalar * (op == 1 ? context.ranks : 1);
    if (!a || !b || a > UINTPTR_MAX - in || b > UINTPTR_MAX - out) return false;
    bool overlap = a < b + out && b < a + in;
    bool legal = op == 1   ? a == b + size_t(context.rank) * count * scalar
                 : op == 2 ? b == a + size_t(context.rank) * count * scalar
                           : a == b;
    return alignment(input, output) == l.alignment && overlap == l.inPlace &&
           (!l.inPlace || legal);
  }
  void begin(int operation, std::string const& request, bool ready) {
    if (!control) {
      control =
          std::make_shared<mscclpp::TcpBootstrap>(context.rank, context.ranks);
      control->initialize(context.id, 120);
    }
    std::ostringstream policy, state;
    policy << int(options.dtype) << ':' << options.fallbackOnly << ':'
           << options.allowPeer << ':' << options.allowHost << ':'
           << options.allowPeerPacket << ':' << options.allowNetwork << ':'
           << options.allowPeerDirect << ':' << options.allowPeerTree << ':'
           << options.probeResources << ':' << options.hardwareProfile << ':'
           << options.networkPlanCache << ':' << options.fallback.identity;
    state << plans.size() << ':' << graphs.size() << ':' << bool(local) << ':'
          << bool(distributed);
    for (auto const& g : graphs) state << ':' << g.active;
    int device = -1;
    ready &= cudaGetDevice(&device) == cudaSuccess &&
             device == context.device && idle() && options.fallback.backend &&
             options.fallback.dtype == options.dtype &&
             !options.fallback.identity.empty();
    struct Envelope {
      uint64_t sequence, request, policy, state;
      size_t capacity;
      int operation, ready;
    } item{sequence++,        hash(request),    hash(policy.str()),
           hash(state.str()), context.capacity, operation,
           int(ready)};
    for (auto v : gather(item))
      if (!v.ready || !item.ready || v.sequence != item.sequence ||
          v.request != item.request || v.policy != item.policy ||
          v.state != item.state || v.capacity != item.capacity ||
          v.operation != operation)
        throw std::invalid_argument(
            "runtime operation/request/policy/idle consensus failed");
  }
  void discover() {
    struct Participant {
      int ready = 1, device, pid;
      uint64_t baseline;
      char host[256]{}, gpu[16]{};
    } item{};
    item.ready = 1;
    item.device = context.device;
    item.pid = getpid();
    item.baseline = hash(options.fallback.localIdentity);
    try {
      check(cudaDeviceSynchronize());
      if (gethostname(item.host, sizeof(item.host) - 1))
        throw std::runtime_error("host identity unavailable");
      cudaDeviceProp property{};
      check(cudaGetDeviceProperties(&property, context.device));
      std::memcpy(item.gpu, property.uuid.bytes, 16);
    } catch (...) {
      item.ready = 0;
    }
    auto participants = gather(item);
    std::map<std::string, std::vector<int>> hosts;
    for (int r = 0; r < context.ranks; ++r) {
      auto const& p = participants[r];
      if (!p.ready)
        throw std::invalid_argument("runtime participant discovery failed");
      for (int prior : hosts[p.host])
        if (participants[prior].pid == p.pid ||
            !std::memcmp(participants[prior].gpu, p.gpu, 16))
          throw std::invalid_argument(
              "runtime requires distinct rank processes and GPUs");
      hosts[p.host].push_back(r);
    }
    struct View {
      int ready = 1, complete = 1;
      std::array<int, 32> devices;
    } view;
    view.devices.fill(-1);
    try {
      int count;
      check(cudaGetDeviceCount(&count));
      for (int device = 0; device < count; ++device) {
        cudaDeviceProp property{};
        check(cudaGetDeviceProperties(&property, device));
        for (int r : hosts[item.host])
          if (!std::memcmp(property.uuid.bytes, participants[r].gpu, 16))
            view.devices[r] = device;
      }
      for (int r : hosts[item.host]) view.complete &= view.devices[r] >= 0;
      view.complete &= view.devices[context.rank] == context.device;
    } catch (...) {
      view.ready = 0;
    }
    auto views = gather(view);
    bool complete = true;
    for (auto const& host : hosts)
      for (int r : host.second) {
        if (!views[r].ready)
          throw std::invalid_argument("runtime CUDA view discovery failed");
        complete &= views[r].complete &&
                    views[r].devices == views[host.second.front()].devices;
      }
    RuntimePlacement found;
    found.domain = hosts.size() > 1 ? RuntimeDomain::Distributed
                   : complete       ? RuntimeDomain::Local
                                    : RuntimeDomain::CallerOnly;
    std::ostringstream identity;
    identity << int(found.domain) << ':' << int(options.dtype) << ':'
             << context.capacity;
    for (int r = 0; r < context.ranks; ++r) {
      auto const& p = participants[r];
      std::ostringstream uuid;
      for (unsigned char byte : p.gpu)
        uuid << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
      found.ranks.push_back({p.host, uuid.str(), p.device});
      identity << ';' << p.host << ':' << uuid.str() << ':' << p.device << ':'
               << p.baseline;
      for (int d : views[r].devices) identity << ',' << d;
    }
    if (!fingerprint.empty() && fingerprint != identity.str())
      throw std::invalid_argument("runtime participant/device view changed");
    if (local || distributed) return;
    struct Identifier {
      int ready = 1;
      NetworkId id{};
    } id;
    if (!context.rank) {
      try {
        id.id = PreparedNetworkRuntime::createUniqueId();
      } catch (...) {
        id.ready = 0;
      }
    }
    id = gather(id).front();
    if (!id.ready) throw std::runtime_error("runtime selector ID unavailable");
    auto c = context.backendContext();
    c.id = id.id;
    if (found.domain == RuntimeDomain::Local) {
      for (int r = 0; r < context.ranks; ++r)
        c.localDevices.push_back(view.devices[r]);
    } else if (found.domain == RuntimeDomain::Distributed) {
      for (int r : hosts[item.host])
        if (view.devices[r] >= 0) c.localDevices.push_back(view.devices[r]);
      std::sort(c.localDevices.begin(), c.localDevices.end());
    } else
      c.localDevices = {context.device};
    std::unique_ptr<PreparedLocalRuntime> localCandidate;
    std::unique_ptr<PreparedNetworkRuntime> distributedCandidate;
    int ready = 1;
    cudaEvent_t event = nullptr;
    try {
      if (found.domain == RuntimeDomain::Local) {
        LocalPreparationOptions o;
        o.dtype = options.dtype;
        o.fallback = options.fallback;
        o.fallbackOnly = options.fallbackOnly;
        o.allowPeer = options.allowPeer;
        o.allowHost = options.allowHost;
        o.allowPeerPacket = options.allowPeerPacket;
        o.allowPeerDirect = options.allowPeerDirect;
        o.allowPeerTree = options.allowPeerTree;
        o.probeResources = options.probeResources;
        o.hardwareProfile = options.hardwareProfile;
        localCandidate = std::make_unique<PreparedLocalRuntime>(c, o);
      } else {
        NetworkPreparationOptions o;
        o.dtype = options.dtype;
        o.fallback = options.fallback;
        o.fallbackOnly = options.fallbackOnly;
        o.allowNetwork =
            options.allowNetwork && found.domain == RuntimeDomain::Distributed;
        o.probeResources = options.probeResources;
        o.planCache = options.networkPlanCache;
        distributedCandidate = std::make_unique<PreparedNetworkRuntime>(c, o);
      }
      check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    } catch (...) {
      ready = 0;
    }
    try {
      agreed(ready, "runtime selector construction failed collectively");
    } catch (...) {
      if (event) cudaEventDestroy(event);
      throw;
    }
    done = event;
    local = std::move(localCandidate);
    distributed = std::move(distributedCandidate);
    placement = std::move(found);
    fingerprint = identity.str();
  }
  void chain(cudaStream_t stream) {
    if (!valid)
      throw std::logic_error(
          "runtime execution requires successful collective preparation");
    int device;
    check(cudaGetDevice(&device));
    if (device != context.device)
      throw std::invalid_argument("runtime current device mismatch");
    cudaStreamCaptureStatus capture;
    check(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone)
      throw std::invalid_argument("runtime external capture unsupported");
    if (recorded && stream != last) check(cudaStreamWaitEvent(stream, done, 0));
  }
  void record(cudaStream_t stream) {
    check(cudaEventRecord(done, stream));
    recorded = true;
    last = stream;
  }
};
NetworkId PreparedRuntime::createUniqueId() {
  return PreparedNetworkRuntime::createUniqueId();
}
PreparedRuntime::PreparedRuntime(RuntimeContext c, PreparationOptions o)
    : impl_(std::make_unique<Impl>(c, std::move(o))) {}
PreparedRuntime::~PreparedRuntime() = default;
size_t PreparedRuntime::prepare(CollectiveRequest const& request) {
  auto& s = *impl_;
  s.valid = false;
  s.begin(0, key(request), s.requestValid(request));
  s.discover();
  auto child =
      s.local ? s.local->prepare(request) : s.distributed->prepare(request);
  if (s.distributed) {
    int ready = 1;
    try {
      for (auto& p : s.plans)
        p.selection = report(s.distributed->selection(p.child));
    } catch (...) {
      ready = 0;
    }
    s.agreed(ready, "distributed pool report refresh failed collectively");
  }
  for (size_t i = 0; i < s.plans.size(); ++i)
    if (s.plans[i].child == child) {
      s.valid = true;
      return i;
    }
  PreparedSelection selection;
  int ready = 1;
  try {
    s.plans.reserve(s.plans.size() + 1);
    selection = s.local ? report(s.local->selection(child))
                        : report(s.distributed->selection(child));
  } catch (...) {
    ready = 0;
  }
  s.agreed(ready, "runtime plan recording failed collectively");
  s.plans.push_back({child, std::move(selection)});
  s.valid = true;
  return s.plans.size() - 1;
}
void PreparedRuntime::enqueue(size_t plan, void const* input, void* output,
                              cudaStream_t stream) {
  auto& s = *impl_;
  auto const& p = s.plans.at(plan);
  if (!s.bindings(p.selection.request, input, output))
    throw std::invalid_argument("runtime buffers do not match prepared layout");
  s.chain(stream);
  if (s.local)
    s.local->enqueue(p.child, input, output, stream);
  else
    s.distributed->enqueue(p.child, input, output, stream);
  s.record(stream);
}
size_t PreparedRuntime::prepareGraph(CollectiveRequest const& request,
                                     void const* input, void* output) {
  auto& s = *impl_;
  s.valid = false;
  s.begin(1, key(request), s.bindings(request, input, output));
  s.discover();
  size_t child = 0, plan = 0;
  if (s.local)
    child = s.local->prepareGraph(request, input, output);
  else {
    s.agreed(s.options.fallback.backend->supportsGraphs(),
             "caller has no distributed graph implementation");
    auto fallback = request;
    fallback.fallbackOnly = true;
    plan = s.distributed->prepare(fallback);
    int ready = 1;
    bool made = false;
    try {
      child = s.options.fallback.backend->prepareGraph(request.operation, input,
                                                       output, request.count);
      made = true;
    } catch (...) {
      ready = 0;
    }
    try {
      s.agreed(ready, "caller graph construction failed collectively");
    } catch (...) {
      if (made) s.options.fallback.backend->releaseGraph(child);
      throw;
    }
  }
  PreparedSelection selection;
  int ready = 1;
  try {
    s.graphs.reserve(s.graphs.size() + 1);
    if (s.local)
      selection = report(s.local->graphSelection(child));
    else {
      selection = report(s.distributed->selection(plan));
      selection.request = request;
      selection.graph = true;
      selection.reason =
          "caller graph: no accelerated distributed graph candidate";
    }
  } catch (...) {
    ready = 0;
  }
  try {
    s.agreed(ready, "runtime graph recording failed collectively");
  } catch (...) {
    if (s.local)
      s.local->releaseGraph(child);
    else
      s.options.fallback.backend->releaseGraph(child);
    throw;
  }
  s.graphs.push_back({child, bool(s.local), true, std::move(selection)});
  s.valid = true;
  return s.graphs.size() - 1;
}
void PreparedRuntime::launchGraph(size_t graph, cudaStream_t stream) {
  auto& s = *impl_;
  auto const& g = s.graphs.at(graph);
  if (!g.active) throw std::invalid_argument("runtime graph released");
  s.chain(stream);
  if (g.local)
    s.local->launchGraph(g.child, stream);
  else
    s.options.fallback.backend->launchGraph(g.child, stream);
  s.record(stream);
}
void PreparedRuntime::releaseGraph(size_t graph) {
  auto& s = *impl_;
  s.begin(2, std::to_string(graph),
          graph < s.graphs.size() && s.graphs[graph].active);
  auto& g = s.graphs[graph];
  if (g.local)
    s.local->releaseGraph(g.child);
  else
    s.options.fallback.backend->releaseGraph(g.child);
  g.active = false;
}
void PreparedRuntime::synchronize() { impl_->synchronize(); }
bool PreparedRuntime::idle() const { return impl_->idle(); }
bool PreparedRuntime::acceleratedResourcesInitialized() const {
  auto& s = *impl_;
  return s.local ? (s.local->peerInitialized() || s.local->hostInitialized() ||
                    s.local->peerPacketInitialized() ||
                    s.local->peerTreeInitialized())
                 : s.distributed && s.distributed->networkInitialized();
}
RuntimePlacement const& PreparedRuntime::placement() const {
  return impl_->placement;
}
PreparedSelection const& PreparedRuntime::selection(size_t plan) const {
  return impl_->plans.at(plan).selection;
}
PreparedSelection const& PreparedRuntime::graphSelection(size_t graph) const {
  return impl_->graphs.at(graph).selection;
}
}  // namespace uccl_lite::runtime
