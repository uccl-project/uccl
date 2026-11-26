#include "qualification.cuh"
#include "uccl_lite/bootstrap/control.hpp"
#include "uccl_lite/hardware_calibration.hpp"
#include "uccl_lite/runtime/build_identity.hpp"
#include "uccl_lite/runtime/prepared_local.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <unistd.h>

extern char** environ;
namespace uccl_lite::runtime {
namespace {
void check(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
uint64_t hash(std::string const& text) {
  uint64_t value = 14695981039346656037ULL;
  for (unsigned char c : text) value = (value ^ c) * 1099511628211ULL;
  return value;
}
unsigned alignment(void const* a, void const* b) {
  auto bits = reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b);
  unsigned result = 1;
  while (result < 256 && !(bits & result)) result *= 2;
  return result;
}
Collective collective(NetworkOperation op) {
  if (op == NetworkOperation::AllGather) return Collective::AllGather;
  if (op == NetworkOperation::ReduceScatter) return Collective::ReduceScatter;
  return Collective::AllReduce;
}
using qualification::Probe;
using qualification::value;

}  // namespace

class PreparedLocalRuntime::Impl {
 public:
  BackendContext context;
  LocalPreparationOptions options;
  std::unique_ptr<bootstrap::HostControl> control;
  Topology topology;
  std::string topologyIdentity, executorIdentity, fallbackIdentity;
  std::unique_ptr<CollectiveBackend> peer, host, packet, tree;
  std::vector<int> treeParents;
  std::vector<LocalSelection> plans;
  bool probed = false, recorded = false;
  bool preparationValid = false;
  cudaEvent_t done = nullptr;
  cudaStream_t lastStream = nullptr;
  static constexpr size_t absent = std::numeric_limits<size_t>::max();
  struct GraphBinding {
    size_t plan, executable = absent;
    void const* input;
    void* output;
    bool active = true;
  };
  std::vector<GraphBinding> graphs;

  Impl(BackendContext c, LocalPreparationOptions o)
      : context(std::move(c)), options(std::move(o)) {
    if (context.ranks < (options.fallback.backend ? 1 : 2) ||
        context.ranks > 32 || context.rank < 0 ||
        context.rank >= context.ranks || !context.capacity ||
        context.capacity > (1UL << 30) ||
        context.localDevices.size() != size_t(context.ranks) ||
        std::set<int>(context.localDevices.begin(), context.localDevices.end())
                .size() != size_t(context.ranks) ||
        context.localDevices[context.rank] != context.device)
      throw std::invalid_argument("invalid prepared local communicator");
  }
  ~Impl() {
    int previous = -1;
    cudaGetDevice(&previous);
    cudaSetDevice(context.device);
    if (recorded) cudaEventSynchronize(done);
    for (auto const& graph : graphs)
      if (graph.executable != absent) {
        try {
          backend(plans[graph.plan].backend)->releaseGraph(graph.executable);
        } catch (...) {
        }
      }
    host.reset();
    packet.reset();
    peer.reset();
    tree.reset();
    if (done) cudaEventDestroy(done);
    if (previous >= 0 && previous != context.device) cudaSetDevice(previous);
  }
  template <class T>
  std::vector<T> gather(T const& local) {
    std::vector<T> all(context.ranks);
    control->allGather(&local, all.data(), sizeof(T));
    return all;
  }
  void barrier() { (void)gather(1); }
  CollectiveBackend* backend(LocalBackend kind) {
    return kind == LocalBackend::PeerTree     ? tree.get()
           : kind == LocalBackend::Fallback   ? options.fallback.backend.get()
           : kind == LocalBackend::PeerPacket ? packet.get()
           : kind == LocalBackend::Host       ? host.get()
                                              : peer.get();
  }
  bool bindingsMatch(LocalRequest const& request, void const* input,
                     void* output) const {
    if (request.layouts.size() != size_t(context.ranks)) return false;
    auto l = request.layouts[context.rank];
    size_t scalar = scalarBytes(options.dtype), count = request.count;
    int op = int(request.operation);
    if (!scalar || op < 1 || op > 3 || !count ||
        count > context.capacity / (scalar * (op == 3 ? 1 : context.ranks)))
      return false;
    auto a = reinterpret_cast<uintptr_t>(input),
         b = reinterpret_cast<uintptr_t>(output);
    size_t inputBytes = count * scalar * (op == 2 ? context.ranks : 1);
    size_t outputBytes = count * scalar * (op == 1 ? context.ranks : 1);
    if (!a || !b || a > UINTPTR_MAX - inputBytes ||
        b > UINTPTR_MAX - outputBytes)
      return false;
    bool overlap = a < b + outputBytes && b < a + inputBytes;
    bool legal = op == 1   ? a == b + size_t(context.rank) * count * scalar
                 : op == 2 ? b == a + size_t(context.rank) * count * scalar
                           : a == b;
    return alignment(input, output) == l.alignment && overlap == l.inPlace &&
           (!l.inPlace || legal);
  }
  size_t makeGraph(CollectiveBackend& executor, LocalRequest const& request,
                   void const* input, void* output) {
    size_t handle = absent;
    try {
      handle = executor.prepareGraph(request.operation, input, output,
                                     request.count);
    } catch (std::exception const& error) {
      std::fprintf(stderr, "local graph prepare rank=%d: %s\n", context.rank,
                   error.what());
    }
    auto votes = gather(int(handle != absent));
    if (std::find(votes.begin(), votes.end(), 0) != votes.end()) {
      if (handle != absent) executor.releaseGraph(handle);
      throw std::runtime_error("local graph construction failed collectively");
    }
    return handle;
  }
  void suspendGraphs() {
    for (auto& graph : graphs) {
      int released = 1;
      try {
        if (graph.executable != absent) {
          backend(plans[graph.plan].backend)->releaseGraph(graph.executable);
          graph.executable = absent;
        }
      } catch (...) {
        released = 0;
      }
      auto votes = gather(released);
      if (std::find(votes.begin(), votes.end(), 0) != votes.end())
        throw std::runtime_error(
            "prepared graph suspension failed collectively");
    }
  }
  size_t finish(size_t plan) {
    // Recreate every retained executor after isolated qualification, including
    // recovery through a cache hit following a failed preparation.
    for (auto const& p : plans) {
      if (p.backend == LocalBackend::Peer && !peer)
        peer = create(peerStagedBackend(options.dtype));
      if (p.backend == LocalBackend::Host && !host)
        host = create(hostPacketBackend());
      if (p.backend == LocalBackend::PeerPacket && !packet)
        packet = create(peerPacketBackend());
      if (p.backend == LocalBackend::PeerTree && !tree)
        tree = create(peerTreeBackend(options.dtype, treeParents));
    }
    if (!done) check(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    for (auto& graph : graphs)
      if (graph.active && graph.executable == absent)
        graph.executable =
            makeGraph(*backend(plans[graph.plan].backend),
                      plans[graph.plan].request, graph.input, graph.output);
    preparationValid = true;
    return plan;
  }
  void consensus(LocalRequest const& request, bool graph, bool bindingsValid) {
    if (!control)
      control = std::make_unique<bootstrap::HostControl>(
          "127.0.0.1", context.id.data(), context.id.size(), context.rank,
          context.ranks, 120000);
    struct Wire {
      uint64_t count, capacity, artifact, hardware, gpus, options,
          fallbackLocal;
      int valid, op, ranks, device, pid;
      std::array<int, 32> devices{}, layouts{};
    } wire{};
    wire.count = request.count;
    wire.capacity = context.capacity;
    wire.op = int(request.operation);
    wire.ranks = context.ranks;
    wire.device = context.device;
    wire.pid = getpid();
    wire.fallbackLocal = hash(options.fallback.localIdentity);
    wire.options = hash(
        options.hardwareProfile +
        (options.probeResources ? ";probe" : ";no-probe") +
        (options.fallbackOnly ? ";global-fallback-only" : ";global-auto") +
        (request.fallbackOnly ? ";request-fallback" : ";request-auto") +
        (options.allowPeer ? ";peer" : ";no-peer") +
        (options.allowHost ? ";host" : ";no-host") +
        (options.allowPeerPacket ? ";packet" : ";no-packet") +
        (options.allowPeerDirect ? ";direct" : ";no-direct") +
        (options.allowPeerTree ? ";tree" : ";no-tree") +
        (graph ? ";owned-graph" : ";eager") +
        ";dtype=" + std::to_string(int(options.dtype)) +
        (options.fallback.backend ? ";fallback:" + options.fallback.identity
                                  : ";no-fallback"));
    size_t scalar = scalarBytes(options.dtype);
    wire.valid =
        bindingsValid && scalar && wire.op >= 1 && wire.op <= 3 &&
        request.count &&
        request.count <=
            context.capacity / (scalar * (wire.op == 3 ? 1 : context.ranks)) &&
        request.layouts.size() == size_t(context.ranks);
    if (options.fallback.backend)
      wire.valid &= !options.fallback.identity.empty() &&
                    options.fallback.dtype == options.dtype;
    else
      wire.valid &= !options.fallbackOnly && !request.fallbackOnly;
    if (graph && options.fallback.backend)
      wire.valid &= options.fallback.backend->supportsGraphs();
    std::string graphState;
    for (auto const& binding : graphs)
      graphState += std::to_string(binding.plan) +
                    (binding.active ? ":active;" : ":released;");
    wire.options ^= hash(graphState);
    std::copy(context.localDevices.begin(), context.localDevices.end(),
              wire.devices.begin());
    if (wire.valid)
      for (int rank = 0; rank < context.ranks; ++rank) {
        auto l = request.layouts[rank];
        wire.valid &= l.alignment >= scalar && l.alignment <= 256 &&
                      !(l.alignment & (l.alignment - 1));
        if (l.inPlace && wire.op != 3 && l.alignment)
          wire.valid &=
              (size_t(rank) * request.count * scalar) % l.alignment == 0;
        wire.layouts[rank] =
            l.alignment <= 256 ? int(l.alignment) * 2 + l.inPlace : -1;
      }
    try {
      int device;
      check(cudaGetDevice(&device));
      if (device != context.device ||
          (recorded && cudaEventQuery(done) != cudaSuccess))
        wire.valid = false;
      // Fails before any resource mutation if a capture is active.
      check(cudaDeviceSynchronize());
      auto current = inspectTopology(context.localDevices);
      if (!topology.hardwareIdentity.empty() &&
          current.hardwareIdentity != topology.hardwareIdentity)
        wire.valid = false;
      if (topology.hardwareIdentity.empty()) topology = std::move(current);
      std::string gpus;
      for (int ordinal : context.localDevices) {
        cudaDeviceProp property{};
        check(cudaGetDeviceProperties(&property, ordinal));
        gpus.append(property.uuid.bytes, sizeof(property.uuid.bytes));
      }
      wire.gpus = hash(gpus);
      executorIdentity =
          artifactIdentity(reinterpret_cast<uintptr_t>(&value<uint16_t>));
      wire.artifact = hash(executorIdentity);
      std::set<std::string> transportEnvironment;
      for (char** item = environ; *item; ++item)
        if (!std::strncmp(*item, "UCCL_LITE_MP_", 13) ||
            !std::strncmp(*item, "MSCCLPP_", 8))
          transportEnvironment.insert(*item);
      std::string identity = topology.hardwareIdentity;
      for (auto const& setting : transportEnvironment)
        identity += ';' + setting;
      wire.hardware = hash(identity);
    } catch (std::exception const&) {
      wire.valid = false;
    }
    auto votes = gather(wire);
    for (int rank = 0; rank < context.ranks; ++rank) {
      auto const& v = votes[rank];
      if (!v.valid || !wire.valid || v.count != wire.count || v.op != wire.op ||
          v.capacity != wire.capacity || v.ranks != wire.ranks ||
          v.artifact != wire.artifact || v.gpus != wire.gpus ||
          v.options != wire.options || v.devices != wire.devices ||
          v.layouts != wire.layouts || v.device != context.localDevices[rank])
        throw std::invalid_argument(
            "prepared local workload/identity/idle consensus failed");
      for (int previous = 0; previous < rank; ++previous)
        if (votes[previous].pid == v.pid)
          throw std::invalid_argument(
              "prepared local runtime requires one process per rank");
    }
    std::ostringstream identity;
    for (auto const& v : votes) identity << v.hardware << ';';
    if (!topologyIdentity.empty() && topologyIdentity != identity.str())
      throw std::invalid_argument("prepared local topology changed");
    topologyIdentity = identity.str();
    std::ostringstream baseline;
    baseline << options.fallback.identity;
    for (auto const& v : votes) baseline << ';' << v.fallbackLocal;
    if (!fallbackIdentity.empty() && fallbackIdentity != baseline.str())
      throw std::invalid_argument("caller fallback identity changed");
    fallbackIdentity = baseline.str();
  }
  std::unique_ptr<CollectiveBackend> create(BackendRegistration registration) {
    auto c = context;
    int ready = 1;
    if (context.rank == 0) {
      try {
        c.id = registration.uniqueId();
      } catch (std::exception const&) {
        ready = 0;
      }
    }
    control->broadcast(&ready, sizeof(ready));
    if (!ready) throw std::runtime_error("prepared backend ID unavailable");
    control->broadcast(c.id.data(), c.id.size());
    auto result = registration.create(c);
    result->prepare();
    barrier();
    return result;
  }
  Workload workload(LocalRequest const& request) const {
    // Transport traffic is independent of pointer layout. Layout validation
    // and qualification still use the full per-rank descriptor.
    return {collective(request.operation), options.dtype,
            request.count * scalarBytes(options.dtype) *
                (int(request.operation) == 3 ? 1 : context.ranks),
            false, unsigned(scalarBytes(options.dtype))};
  }
  void resources() {
    if (probed) return;
    int success = 1;
    if (context.rank == 0) {
      try {
        if (!options.hardwareProfile.empty())
          applyHardwareCalibration(
              topology,
              loadHardwareProfile(options.hardwareProfile).calibration);
        else if (options.probeResources)
          applyHardwareCalibration(topology,
                                   probeGpuResources(topology).calibration);
      } catch (std::exception const&) {
        success = 0;
      }
    }
    control->broadcast(&success, sizeof(success));
    if (!success) throw std::runtime_error("local resource calibration failed");
    probed = true;
  }
  bool hostEligible() {
    bool eligible =
        options.dtype == ScalarType::BFloat16 && options.allowHost &&
        context.ranks >= 2 && topology.avx2 &&
        topology.mappedHost.size() == size_t(context.ranks) &&
        topology.numaNodes.size() == size_t(context.ranks) &&
        std::all_of(topology.mappedHost.begin(), topology.mappedHost.end(),
                    [](bool x) { return x; }) &&
        std::all_of(topology.numaNodes.begin(), topology.numaNodes.end(),
                    [](int n) { return n >= 0; });
    // The automatic path currently calibrates the generic host algorithm.
    // Retained opt-in specializations change worker placement/protocol and
    // require their own explicit configuration identity and qualification.
    for (char** item = environ; *item; ++item)
      if (!std::strncmp(*item, "UCCL_LITE_MP_", 13)) eligible = false;
    auto votes = gather(int(eligible));
    return std::all_of(votes.begin(), votes.end(),
                       [](int x) { return x != 0; });
  }
  bool block(CollectiveBackend& backend, Probe& probe,
             LocalRequest const& request, int& seed,
             std::vector<double>& samples, bool graph = false) {
    struct GraphScope {
      CollectiveBackend& backend;
      size_t handle;
      ~GraphScope() {
        if (handle != absent) {
          try {
            backend.synchronize();
            backend.releaseGraph(handle);
          } catch (...) {
          }
        }
      }
    } scope{backend,
            graph ? makeGraph(backend, request, probe.input(), probe.output())
                  : absent};
    bool correct = true;
    for (int iteration = 0; iteration < 5; ++iteration, ++seed) {
      probe.initialize(seed);
      check(cudaStreamSynchronize(probe.stream));
      barrier();
      auto start = std::chrono::steady_clock::now();
      if (graph)
        backend.launchGraph(scope.handle, probe.stream);
      else
        backend.enqueue(request.operation, probe.input(), probe.output(),
                        request.count, probe.stream);
      check(cudaStreamSynchronize(probe.stream));
      double us = std::chrono::duration<double, std::micro>(
                      std::chrono::steady_clock::now() - start)
                      .count();
      auto times = gather(us);
      if (iteration >= 2)
        samples.push_back(*std::max_element(times.begin(), times.end()));
      check(cudaMemsetAsync(probe.wrong, 0, sizeof(unsigned), probe.stream));
      probe.verify(seed);
      unsigned wrong;
      check(cudaMemcpyAsync(&wrong, probe.wrong, sizeof(wrong),
                            cudaMemcpyDeviceToHost, probe.stream));
      check(cudaStreamSynchronize(probe.stream));
      auto errors = gather(wrong);
      correct &= std::all_of(errors.begin(), errors.end(),
                             [](unsigned e) { return !e; });
    }
    if (scope.handle != absent) {
      backend.releaseGraph(scope.handle);
      scope.handle = absent;
    }
    return correct;
  }
  void prepareCallerFallback() {
    int ready = 1;
    try {
      options.fallback.backend->prepare();
    } catch (std::exception const& error) {
      ready = 0;
      std::fprintf(stderr, "fallback prepare rank=%d: %s\n", context.rank,
                   error.what());
    }
    auto preparations = gather(ready);
    if (std::find(preparations.begin(), preparations.end(), 0) !=
        preparations.end())
      throw std::runtime_error(
          "caller fallback preparation failed collectively");
    if (!done) check(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
  }
  bool packetEligible(LocalRequest const& request) {
    bool eligible = options.allowPeer && options.allowPeerPacket &&
                    uccl_lite::peerPacketEligible(topology, workload(request));
    auto votes = gather(int(eligible));
    return std::find(votes.begin(), votes.end(), 0) == votes.end();
  }
  BackendRegistration alternative(size_t index) {
    if (index == 0) return peerStagedBackend(options.dtype);
    if (index == 1) return hostPacketBackend();
    if (index == 2) return peerPacketBackend();
    if (index == 3) return peerTreeBackend(options.dtype, treeParents);
    throw std::invalid_argument("invalid local alternative");
  }
  LocalSelection prepareFallback(LocalRequest const& request, bool peerEligible,
                                 bool graph) {
    LocalSelection result;
    result.request = request;
    result.graph = graph;
    result.backend = LocalBackend::Fallback;
    result.topologyIdentity = topologyIdentity;
    result.executorIdentity = executorIdentity;
    result.fallbackIdentity = fallbackIdentity;
    bool fallbackOnly = options.fallbackOnly || request.fallbackOnly;
    result.peerEligible = peerEligible && !fallbackOnly;
    result.hostEligible = !graph && !fallbackOnly && hostEligible();
    result.peerPacketEligible =
        !graph && !fallbackOnly && packetEligible(request);
    bool treeEligible =
        !fallbackOnly && options.allowPeer && options.allowPeerTree &&
        choosePlan(
            topology, workload(request), false,
            {treePeerCandidate(topology, workload(request),
                               peerTreeParents(topology, context.capacity))})
            .candidates[0]
            .eligible;
    auto treeVotes = gather(int(treeEligible));
    result.peerTreeEligible =
        !fallbackOnly &&
        std::find(treeVotes.begin(), treeVotes.end(), 0) == treeVotes.end();
    if (result.peerPacketEligible)
      result.peerPacketRankOrder = uccl_lite::pairedNumaRankOrder(topology);
    if (!result.peerEligible && !result.hostEligible &&
        !result.peerPacketEligible && !result.peerTreeEligible) {
      result.reason = "caller fallback: no alternative requested or supported";
      return result;
    }
    if (result.peerEligible || result.peerTreeEligible) {
      resources();
      if (context.rank == 0) {
        auto w = workload(request);
        auto estimate =
            choosePlan(topology, w, false, {stagedPeerCandidate(topology, w)})
                .candidates[0];
        if (estimate.estimated)
          result.peerResourceEstimateUs = estimate.microseconds;
      }
      control->broadcast(&result.peerResourceEstimateUs, sizeof(double));
      if (result.peerTreeEligible) {
        std::array<int, 32> parents;
        parents.fill(-1);
        if (context.rank == 0) {
          if (treeParents.empty()) {
            size_t scalar = scalarBytes(options.dtype);
            Workload basis{Collective::AllReduce, options.dtype,
                           context.capacity / scalar * scalar, false,
                           unsigned(scalar)};
            treeParents = selectPeerTreeParents(topology, basis);
          }
          std::copy(treeParents.begin(), treeParents.end(), parents.begin());
          auto estimate =
              choosePlan(
                  topology, workload(request), false,
                  {treePeerCandidate(topology, workload(request), treeParents)})
                  .candidates[0];
          if (estimate.estimated) {
            result.peerTreeResourceEstimateUs = estimate.microseconds;
            result.peerTreeSerialUs = estimate.serialUs;
            result.peerTreeDependencyUs = estimate.dependencyOnlyUs;
            result.peerTreeResourceLoadUs = estimate.resourceLoadUs;
          }
        }
        control->broadcast(parents.data(), sizeof(parents));
        treeParents.assign(parents.begin(), parents.begin() + context.ranks);
        result.peerTreeParents = treeParents;
        control->broadcast(&result.peerTreeResourceEstimateUs, sizeof(double));
        control->broadcast(&result.peerTreeSerialUs, sizeof(double));
        control->broadcast(&result.peerTreeDependencyUs, sizeof(double));
        control->broadcast(&result.peerTreeResourceLoadUs, sizeof(double));
      }
    }
    // Baseline blocks contain no UCCL peer/host instance. Resources needed by
    // earlier handles are restored only after the new qualification finishes.
    peer.reset();
    host.reset();
    packet.reset();
    tree.reset();
    barrier();
    Probe probe;
    int allocated = 1;
    try {
      probe.allocate(request, context.rank, context.ranks, options.dtype);
    } catch (std::exception const&) {
      allocated = 0;
    }
    auto allocations = gather(allocated);
    if (std::find(allocations.begin(), allocations.end(), 0) !=
        allocations.end())
      throw std::runtime_error("fallback comparison probe allocation failed");
    for (size_t candidate = 0; candidate < kLocalAlternatives.size();
         ++candidate) {
      bool eligible = candidate == 0   ? result.peerEligible
                      : candidate == 1 ? result.hostEligible
                      : candidate == 2 ? result.peerPacketEligible
                                       : result.peerTreeEligible;
      if (!eligible) continue;
      result.candidateCorrect[candidate] = true;
      int seed = 0;
      for (int round = 0; round < 3; ++round) {
        if (!block(*options.fallback.backend, probe, request, seed,
                   result.fallbackSamples[candidate], graph))
          throw std::runtime_error(
              "caller fallback failed complete-buffer qualification");
        auto executor = create(alternative(candidate));
        for (int repeat = 0; repeat < 2; ++repeat)
          result.candidateCorrect[candidate] &= block(
              *executor, probe, request, seed,
              result.samples[size_t(kLocalAlternatives[candidate])], graph);
        executor.reset();
        barrier();
        if (!block(*options.fallback.backend, probe, request, seed,
                   result.fallbackSamples[candidate], graph))
          throw std::runtime_error(
              "caller fallback failed after alternative teardown");
      }
      result.samples[2].insert(result.samples[2].end(),
                               result.fallbackSamples[candidate].begin(),
                               result.fallbackSamples[candidate].end());
    }
    result.hostCorrect = result.candidateCorrect[1];
    std::array<std::vector<double>, 4> alternatives{
        {result.samples[0], result.samples[1], result.samples[3],
         result.samples[4]}};
    int choice = int(LocalBackend::Fallback);
    if (context.rank == 0)
      choice = int(qualifiedFallbackAlternative(
          result.fallbackSamples, alternatives, result.candidateCorrect));
    control->broadcast(&choice, sizeof(choice));
    result.backend = LocalBackend(choice);
    auto needed = [&](LocalBackend kind) {
      return result.backend == kind ||
             std::any_of(plans.begin(), plans.end(),
                         [&](auto const& p) { return p.backend == kind; });
    };
    if (needed(LocalBackend::Peer))
      peer = create(peerStagedBackend(options.dtype));
    if (needed(LocalBackend::Host)) host = create(hostPacketBackend());
    if (needed(LocalBackend::PeerPacket)) packet = create(peerPacketBackend());
    if (needed(LocalBackend::PeerTree))
      tree = create(peerTreeBackend(options.dtype, treeParents));
    result.reason =
        result.backend == LocalBackend::Fallback
            ? "caller fallback retained: no robust measured alternative"
            : "alternative qualified against caller fallback";
    return result;
  }
};

PreparedLocalRuntime::PreparedLocalRuntime(BackendContext context,
                                           LocalPreparationOptions options)
    : impl_(std::make_unique<Impl>(std::move(context), std::move(options))) {}
PreparedLocalRuntime::~PreparedLocalRuntime() = default;
size_t PreparedLocalRuntime::prepare(LocalRequest const& request) {
  return preparePlan(request, false);
}
size_t PreparedLocalRuntime::preparePlan(LocalRequest const& request,
                                         bool graph, void const* input,
                                         void* output) {
  auto& s = *impl_;
  s.preparationValid = false;
  s.consensus(request, graph,
              !graph || s.bindingsMatch(request, input, output));
  if (s.options.fallback.backend) s.prepareCallerFallback();
  s.suspendGraphs();
  size_t cached = s.plans.size();
  for (size_t i = 0; i < s.plans.size(); ++i) {
    auto const& old = s.plans[i].request;
    if (s.plans[i].graph != graph || old.operation != request.operation ||
        old.count != request.count ||
        old.fallbackOnly != request.fallbackOnly ||
        old.layouts.size() != request.layouts.size())
      continue;
    bool same = true;
    for (size_t rank = 0; rank < old.layouts.size(); ++rank)
      same &= old.layouts[rank].inPlace == request.layouts[rank].inPlace &&
              old.layouts[rank].alignment == request.layouts[rank].alignment;
    if (same) {
      cached = i;
      break;
    }
  }
  struct CacheVote {
    size_t index, size;
  } cacheVote{cached, s.plans.size()};
  auto cachedPlans = s.gather(cacheVote);
  if (!std::all_of(cachedPlans.begin(), cachedPlans.end(), [&](CacheVote vote) {
        return vote.index == cached && vote.size == s.plans.size();
      }))
    throw std::logic_error("prepared plan cache consensus failed");
  if (cached < s.plans.size()) {
    return s.finish(cached);
  }
  auto w = s.workload(request);
  auto candidate = stagedPeerCandidate(s.topology, w);
  auto eligible =
      choosePlan(s.topology, w, false, {candidate}).candidates[0].eligible;
  auto votes = s.gather(int(eligible));
  bool peerEligible = s.options.allowPeer && s.options.allowPeerDirect &&
                      std::find(votes.begin(), votes.end(), 0) == votes.end();
  if (s.options.fallback.backend) {
    s.plans.push_back(s.prepareFallback(request, peerEligible, graph));
    return s.finish(s.plans.size() - 1);
  }
  if (!peerEligible)
    throw std::invalid_argument(
        "no supported peer baseline for requested dtype/topology");
  if (!s.peer) {
    s.peer = s.create(peerStagedBackend(s.options.dtype));
    if (!s.done)
      check(cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming));
  }
  s.resources();
  LocalSelection selected;
  selected.request = request;
  selected.graph = graph;
  selected.peerEligible = true;
  selected.topologyIdentity = s.topologyIdentity;
  selected.executorIdentity = s.executorIdentity;
  if (s.context.rank == 0) {
    auto estimates =
        choosePlan(s.topology, w, false, {stagedPeerCandidate(s.topology, w)});
    if (estimates.candidates[0].estimated)
      selected.peerResourceEstimateUs = estimates.candidates[0].microseconds;
  }
  s.control->broadcast(&selected.peerResourceEstimateUs, sizeof(double));
  selected.hostEligible = !graph && s.hostEligible();
  selected.hostCorrect = selected.hostEligible;
  selected.peerPacketEligible = !graph && s.packetEligible(request);
  if (selected.peerPacketEligible)
    selected.peerPacketRankOrder = uccl_lite::pairedNumaRankOrder(s.topology);
  bool alternativeEligible =
      selected.hostEligible || selected.peerPacketEligible;
  LocalBackend alternative =
      selected.hostEligible ? LocalBackend::Host : LocalBackend::PeerPacket;
  bool alternativeCorrect = alternativeEligible;
  // Every baseline block is measured without host buffers/workers. A host
  // provider needed by an earlier plan is restored only after qualification.
  s.host.reset();
  s.packet.reset();
  s.barrier();
  Probe probe;
  int allocated = 1;
  try {
    probe.allocate(request, s.context.rank, s.context.ranks, s.options.dtype);
  } catch (std::exception const&) {
    allocated = 0;
  }
  auto allocations = s.gather(allocated);
  if (std::find(allocations.begin(), allocations.end(), 0) != allocations.end())
    throw std::runtime_error("collective probe allocation failed");
  int seed = 0;
  for (int round = 0; round < 3; ++round) {
    if (!s.block(*s.peer, probe, request, seed, selected.samples[0], graph))
      throw std::runtime_error(
          "peer baseline failed full-buffer qualification");
    if (alternativeEligible) {
      auto host = s.create(selected.hostEligible ? hostPacketBackend()
                                                 : peerPacketBackend());
      for (int block = 0; block < 2; ++block)
        alternativeCorrect &= s.block(*host, probe, request, seed,
                                      selected.samples[size_t(alternative)]);
      host.reset();
      s.barrier();
      if (!s.block(*s.peer, probe, request, seed, selected.samples[0]))
        throw std::runtime_error("peer baseline failed after host teardown");
    }
  }
  int choice = 0;
  selected.hostCorrect = selected.hostEligible && alternativeCorrect;
  selected.candidateCorrect = {
      {true, selected.hostCorrect,
       selected.peerPacketEligible && alternativeCorrect}};
  if (s.context.rank == 0 && alternativeEligible &&
      qualifiedLocalBackend(selected.samples[0],
                            selected.samples[size_t(alternative)],
                            alternativeCorrect) == LocalBackend::Host)
    choice = int(alternative);
  s.control->broadcast(&choice, sizeof(choice));
  selected.backend = LocalBackend(choice);
  selected.reason = choice
                        ? "alternative qualified against isolated peer baseline"
                        : "peer baseline: alternative unavailable, incorrect "
                          "or no robust gain";
  bool needHost =
      selected.backend == LocalBackend::Host ||
      std::any_of(s.plans.begin(), s.plans.end(), [](LocalSelection const& p) {
        return p.backend == LocalBackend::Host;
      });
  if (needHost) s.host = s.create(hostPacketBackend());
  bool needPacket =
      selected.backend == LocalBackend::PeerPacket ||
      std::any_of(s.plans.begin(), s.plans.end(), [](auto const& p) {
        return p.backend == LocalBackend::PeerPacket;
      });
  if (needPacket) s.packet = s.create(peerPacketBackend());
  s.plans.push_back(std::move(selected));
  return s.finish(s.plans.size() - 1);
}
void PreparedLocalRuntime::enqueue(size_t plan, void const* input, void* output,
                                   cudaStream_t stream) {
  auto& s = *impl_;
  if (!s.preparationValid)
    throw std::logic_error(
        "enqueue requires successful collective preparation");
  auto const& p = s.plans.at(plan);
  if (p.graph)
    throw std::invalid_argument("graph-qualified plans require launchGraph");
  if (!s.bindingsMatch(p.request, input, output))
    throw std::invalid_argument(
        "buffers do not match the collective prepared layout");
  int device;
  check(cudaGetDevice(&device));
  if (device != s.context.device)
    throw std::invalid_argument("prepared local current device mismatch");
  cudaStreamCaptureStatus capture;
  check(cudaStreamIsCapturing(stream, &capture));
  if (capture != cudaStreamCaptureStatusNone)
    throw std::invalid_argument("prepared local capture is not qualified");
  if (s.recorded && stream != s.lastStream)
    check(cudaStreamWaitEvent(stream, s.done, 0));
  auto* backend = s.backend(p.backend);
  if (!backend) throw std::logic_error("prepared backend is absent");
  backend->enqueue(p.request.operation, input, output, p.request.count, stream);
  check(cudaEventRecord(s.done, stream));
  s.recorded = true;
  s.lastStream = stream;
}
size_t PreparedLocalRuntime::prepareGraph(LocalRequest const& request,
                                          void const* input, void* output) {
  size_t plan = preparePlan(request, true, input, output);
  auto& s = *impl_;
  s.preparationValid = false;
  int allocated = 1;
  try {
    s.graphs.reserve(s.graphs.size() + 1);
  } catch (...) {
    allocated = 0;
  }
  auto votes = s.gather(allocated);
  if (std::find(votes.begin(), votes.end(), 0) != votes.end())
    throw std::runtime_error("prepared graph storage allocation failed");
  size_t executable =
      s.makeGraph(*s.backend(s.plans[plan].backend), request, input, output);
  s.graphs.push_back({plan, executable, input, output, true});
  s.preparationValid = true;
  return s.graphs.size() - 1;
}
void PreparedLocalRuntime::launchGraph(size_t graph, cudaStream_t stream) {
  auto& s = *impl_;
  if (!s.preparationValid)
    throw std::logic_error("graph launch requires successful preparation");
  auto const& binding = s.graphs.at(graph);
  if (!binding.active || binding.executable == Impl::absent)
    throw std::invalid_argument("prepared graph is absent or released");
  int device;
  check(cudaGetDevice(&device));
  if (device != s.context.device)
    throw std::invalid_argument("prepared graph current device mismatch");
  cudaStreamCaptureStatus capture;
  check(cudaStreamIsCapturing(stream, &capture));
  if (capture != cudaStreamCaptureStatusNone)
    throw std::invalid_argument("owned graph launch rejects external capture");
  if (s.recorded && stream != s.lastStream)
    check(cudaStreamWaitEvent(stream, s.done, 0));
  s.backend(s.plans[binding.plan].backend)
      ->launchGraph(binding.executable, stream);
  check(cudaEventRecord(s.done, stream));
  s.recorded = true;
  s.lastStream = stream;
}
void PreparedLocalRuntime::releaseGraph(size_t graph) {
  auto& s = *impl_;
  if (!s.control) throw std::logic_error("no prepared graph session");
  int device;
  check(cudaGetDevice(&device));
  struct Vote {
    size_t graph;
    int valid;
  } local{graph, int(graph < s.graphs.size() && s.graphs[graph].active &&
                     idle() && device == s.context.device)};
  auto votes = s.gather(local);
  for (auto vote : votes)
    if (!vote.valid || vote.graph != graph)
      throw std::invalid_argument(
          "graph release requires collective idle handle");
  auto& binding = s.graphs[graph];
  if (binding.executable != Impl::absent)
    s.backend(s.plans[binding.plan].backend)->releaseGraph(binding.executable);
  binding.executable = Impl::absent;
  binding.active = false;
}
LocalSelection const& PreparedLocalRuntime::graphSelection(size_t graph) const {
  return impl_->plans.at(impl_->graphs.at(graph).plan);
}
void PreparedLocalRuntime::synchronize() {
  if (impl_->recorded) check(cudaEventSynchronize(impl_->done));
}
LocalSelection const& PreparedLocalRuntime::selection(size_t plan) const {
  return impl_->plans.at(plan);
}
bool PreparedLocalRuntime::hostInitialized() const { return bool(impl_->host); }
bool PreparedLocalRuntime::peerInitialized() const { return bool(impl_->peer); }
bool PreparedLocalRuntime::peerPacketInitialized() const {
  return bool(impl_->packet);
}
bool PreparedLocalRuntime::peerTreeInitialized() const {
  return bool(impl_->tree);
}
bool PreparedLocalRuntime::idle() const {
  return !impl_->recorded || cudaEventQuery(impl_->done) == cudaSuccess;
}
}  // namespace uccl_lite::runtime
