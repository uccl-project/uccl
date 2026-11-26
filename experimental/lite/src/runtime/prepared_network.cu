#include "network_preflight.hpp"
#include "network_probe.hpp"
#include "qualification.cuh"
#include "uccl_lite/collectives/direct_exchange.hpp"
#include "uccl_lite/local_plan.hpp"
#include "uccl_lite/network_cache.hpp"
#include "uccl_lite/network_calibration.hpp"
#include "uccl_lite/runtime/build_identity.hpp"
#include "uccl_lite/runtime/network_identity.hpp"
#include "uccl_lite/runtime/prepared_network.hpp"
#include <mscclpp/core.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <unistd.h>

extern char** environ;
namespace uccl_lite::runtime {
namespace {
using qualification::alignment;
using qualification::check;
uint64_t hash(std::string const& text) {
  uint64_t value = 14695981039346656037ULL;
  for (unsigned char c : text) value = (value ^ c) * 1099511628211ULL;
  return value;
}
std::string key(CollectiveRequest const& r) {
  std::ostringstream s;
  s << int(r.operation) << ':' << r.count << ':' << r.fallbackOnly;
  for (auto const& l : r.layouts) s << ';' << l.inPlace << ':' << l.alignment;
  return s.str();
}
}  // namespace
class PreparedNetworkRuntime::Impl {
 public:
  BackendContext context;
  NetworkPreparationOptions options;
  std::shared_ptr<mscclpp::TcpBootstrap> control;
  std::unique_ptr<CollectiveBackend> network, direct;
  std::vector<std::string> directIdentities;
  std::shared_ptr<DirectModelCalibration const> directModel;
  bool directCapability = false;
  std::shared_ptr<NetworkProfile const> calibration;
  NetworkPlanCache cache;
  bool cacheLoaded = false;
  std::string cacheNotice;
  std::vector<NetworkSelection> plans;
  std::vector<int> hostCounts;
  std::string identity, baselineIdentity, artifact;
  bool valid = false, recorded = false, capability = false;
  cudaEvent_t done = nullptr;
  cudaStream_t last = nullptr;
  Impl(BackendContext c, NetworkPreparationOptions o)
      : context(std::move(c)), options(std::move(o)) {
    if (options.planCache.empty())
      if (auto path = std::getenv("UCCL_LITE_NETWORK_PLAN_CACHE"))
        options.planCache = path;
    if (context.ranks < 1 || context.ranks > 32 || context.rank < 0 ||
        context.rank >= context.ranks || context.device < 0 ||
        !context.capacity || context.capacity > (1UL << 30))
      throw std::invalid_argument("invalid distributed preparation context");
  }
  ~Impl() {
    int previous = -1;
    cudaGetDevice(&previous);
    cudaSetDevice(context.device);
    if (recorded) cudaEventSynchronize(done);
    if (direct) {
      try {
        direct->synchronize();
      } catch (...) {
      }
      direct.reset();
    }
    if (network) {
      try {
        network->synchronize();
      } catch (...) {
      }
      network.reset();
    }
    if (done) cudaEventDestroy(done);
    if (previous >= 0 && previous != context.device) cudaSetDevice(previous);
  }
  template <class T>
  std::vector<T> gather(T const& local) {
    std::vector<T> result(context.ranks);
    result[context.rank] = local;
    control->allGather(result.data(), sizeof(T));
    return result;
  }
  void barrier() { (void)gather(1); }
  std::string cacheFingerprint() {
    struct Part {
      int ready = 1;
      uint64_t transport = 0;
    } part;
    try {
      std::set<std::string> execution;
      for (char** item = environ; *item; ++item) {
        bool uccl = !std::strncmp(*item, "UCCL_LITE_", 10) &&
                    std::strncmp(*item, "UCCL_LITE_NETWORK_PLAN_CACHE=", 29);
        if (uccl || !std::strncmp(*item, "CUDA_", 5) ||
            !std::strncmp(*item, "NV_COMPUTE_SANITIZER", 20))
          execution.insert(*item);
      }
      std::ostringstream configuration;
      configuration << networkTransportIdentity();
      for (auto const& entry : execution)
        configuration << ';' << entry.size() << ':' << entry;
      part.transport = hash(configuration.str());
    } catch (...) {
      part.ready = 0;
    }
    auto parts = gather(part);
    if (std::any_of(parts.begin(), parts.end(),
                    [](auto p) { return !p.ready; }))
      return {};
    std::ostringstream key;
    key << "network-plan-context-v1;" << identity << ';' << artifact
        << ";capacity=" << context.capacity;
    for (auto p : parts) key << ';' << p.transport;
    return key.str();
  }
  void validateCacheContext() {
    if (!cache.identity.empty() && cacheFingerprint() != cache.identity)
      throw std::invalid_argument(
          "persistent network cache live identity changed");
  }
  NetworkPlanRecord descriptor(CollectiveRequest const& r,
                               NetworkCacheCandidate candidate =
                                   NetworkCacheCandidate::Hierarchy) const {
    NetworkPlanRecord result;
    result.candidate = candidate;
    result.workload = {
        r.operation == NetworkOperation::AllReduce ? Collective::AllReduce
        : r.operation == NetworkOperation::AllGather
            ? Collective::AllGather
            : Collective::ReduceScatter,
        options.dtype,
        r.count * scalarBytes(options.dtype) *
            (r.operation == NetworkOperation::AllReduce ? 1 : context.ranks),
        false, unsigned(scalarBytes(options.dtype))};
    for (auto l : r.layouts) result.layouts.push_back({l.inPlace, l.alignment});
    return result;
  }
  void loadCache() {
    if (cacheLoaded || options.planCache.empty()) return;
    auto current = cacheFingerprint();
    if (current.empty()) {
      cacheNotice = "transport identity unavailable";
      return;
    }
    struct Status {
      int ready = 1, hit = 0, stale = 0;
      size_t bytes = 0;
      char error[512]{};
    } status;
    std::string encoded;
    if (!context.rank) {
      try {
        auto saved = loadNetworkPlanCache(options.planCache);
        if (saved && saved->identity == current) {
          encoded = encodeNetworkPlanCache(*saved);
          status.hit = 1;
          status.bytes = encoded.size();
        } else
          status.stale = bool(saved);
      } catch (std::exception const& error) {
        status.ready = 0;
        std::snprintf(status.error, sizeof(status.error), "%s", error.what());
      }
    }
    status = gather(status).front();
    if (!status.ready)
      throw std::runtime_error(std::string("network cache read failed: ") +
                               status.error);
    NetworkPlanCache found;
    std::shared_ptr<NetworkProfile const> resources;
    int ready = 1;
    if (status.hit) {
      try {
        encoded.resize(status.bytes);
      } catch (...) {
        ready = 0;
      }
      auto votes = gather(ready);
      if (std::find(votes.begin(), votes.end(), 0) != votes.end())
        throw std::runtime_error(
            "network cache distribution allocation failed");
      control->broadcast(encoded.data(), int(encoded.size()), 0);
      try {
        found = decodeNetworkPlanCache(encoded);
        if (found.identity != current)
          throw std::invalid_argument("distributed cache identity mismatch");
        if (found.calibration && !std::getenv("UCCL_LITE_NETWORK_PROFILE")) {
          char host[256]{};
          if (gethostname(host, sizeof(host) - 1))
            throw std::runtime_error("cache host identity unavailable");
          auto const& hosts = found.calibration->hosts;
          auto h = std::find_if(hosts.begin(), hosts.end(), [&](auto const& h) {
            return h.hostname == host;
          });
          if (h == hosts.end() ||
              h->topologyIdentity !=
                  inspectTopology(context.localDevices).hardwareIdentity ||
              h->transportIdentity != networkTransportIdentity())
            throw std::invalid_argument(
                "embedded network resource identity mismatch");
          resources =
              std::make_shared<NetworkProfile const>(*found.calibration);
        }
      } catch (...) {
        ready = 0;
      }
      votes = gather(ready);
      if (std::find(votes.begin(), votes.end(), 0) != votes.end())
        throw std::runtime_error(
            "network cache/resource validation failed collectively");
      cache = std::move(found);
      if (resources) calibration = std::move(resources);
      cacheNotice = "matched snapshot";
    } else {
      cache.identity = std::move(current);
      cacheNotice = status.stale ? "identity miss" : "snapshot missing";
    }
    cacheLoaded = true;
  }
  std::optional<NetworkPlanRecord> cachedRecord(
      CollectiveRequest const& request,
      NetworkCacheCandidate candidate =
          NetworkCacheCandidate::Hierarchy) const {
    auto query = descriptor(request, candidate);
    for (auto const& record : cache.records)
      if (sameNetworkRequest(record, query)) return record;
    return {};
  }
  void persist(
      NetworkSelection const& selection, bool erase = false,
      NetworkCacheCandidate candidate = NetworkCacheCandidate::Hierarchy) {
    if (options.planCache.empty() || cache.identity.empty()) return;
    struct Result {
      int ready = 1;
      char error[512]{};
    } result;
    if (!context.rank) {
      try {
        auto record = descriptor(selection.request, candidate);
        if (erase)
          eraseNetworkPlanCacheRecord(options.planCache, cache.identity,
                                      record);
        else {
          if (candidate == NetworkCacheCandidate::Hierarchy) {
            record.candidateCorrect = selection.candidateCorrect;
            record.useHierarchy =
                selection.backend == NetworkBackend::Hierarchy;
            record.rankOrder = selection.executionRankOrder;
            record.rails = selection.rails;
            record.candidateSamples = selection.candidateSamples;
            record.baselineSamples = selection.baselineSamples;
          } else {
            record.candidateCorrect = selection.direct.correct;
            record.candidateSamples = selection.direct.samples;
            record.baselineSamples = selection.direct.baselineSamples;
            record.configurationIdentities =
                selection.direct.configurationIdentities;
            record.useDirect = qualifiedLocalBackend(record.baselineSamples,
                                                     record.candidateSamples,
                                                     record.candidateCorrect) ==
                               LocalBackend::Host;
          }
          auto snapshot = cache;
          if (calibration) snapshot.calibration = *calibration;
          updateNetworkPlanCache(options.planCache, snapshot, record);
        }
      } catch (std::exception const& error) {
        result.ready = 0;
        std::snprintf(result.error, sizeof(result.error), "%s", error.what());
      }
    }
    result = gather(result).front();
    if (!result.ready)
      throw std::runtime_error(std::string("network cache update failed: ") +
                               result.error);
  }
  void consensus(CollectiveRequest const& request) {
    if (!control) {
      control =
          std::make_shared<mscclpp::TcpBootstrap>(context.rank, context.ranks);
      control->initialize(context.id, 120);
    }
    struct Wire {
      uint64_t request, options, artifact, localIdentity, baselineLocal,
          profile;
      size_t capacity;
      int valid, rank, device, pid, candidate, directCandidate;
      char host[256]{}, gpu[16]{};
    } wire{};
    wire.rank = context.rank;
    wire.device = context.device;
    wire.pid = getpid();
    wire.capacity = context.capacity;
    wire.request = hash(key(request));
    wire.options = hash(std::to_string(int(options.dtype)) + ';' +
                        std::to_string(options.allowNetwork) + ';' +
                        std::to_string(options.fallbackOnly) + ';' +
                        std::to_string(options.probeResources) + ';' +
                        std::to_string(options.allowHierarchy) + ';' +
                        std::to_string(options.allowDirect) + ';' +
                        options.planCache + ';' + options.fallback.identity);
    wire.baselineLocal = hash(options.fallback.localIdentity);
    size_t scalar = scalarBytes(options.dtype);
    int op = int(request.operation);
    wire.valid =
        scalar && op >= 1 && op <= 3 && request.count &&
        request.count <=
            context.capacity / (scalar * (op == 3 ? 1 : context.ranks)) &&
        request.layouts.size() == size_t(context.ranks) &&
        options.fallback.backend && !options.fallback.identity.empty() &&
        options.fallback.dtype == options.dtype;
    if (wire.valid)
      for (int r = 0; r < context.ranks; ++r) {
        auto l = request.layouts[r];
        wire.valid &= l.alignment >= scalar && l.alignment <= 256 &&
                      !(l.alignment & (l.alignment - 1));
        if (l.inPlace && op != 3 && l.alignment)
          wire.valid &= (size_t(r) * request.count * scalar) % l.alignment == 0;
      }
    try {
      int device;
      check(cudaGetDevice(&device));
      if (device != context.device ||
          (recorded && cudaEventQuery(done) != cudaSuccess) ||
          (network && !network->idle()) || (direct && !direct->idle()))
        throw std::invalid_argument(
            "distributed preparation requires idle current device");
      check(cudaDeviceSynchronize());
      if (gethostname(wire.host, sizeof(wire.host) - 1))
        throw std::runtime_error("host identity unavailable");
      cudaDeviceProp property{};
      check(cudaGetDeviceProperties(&property, device));
      std::memcpy(wire.gpu, property.uuid.bytes, 16);
      auto devices = context.localDevices.empty() ? std::vector<int>{device}
                                                  : context.localDevices;
      if (std::find(devices.begin(), devices.end(), device) == devices.end() ||
          std::set<int>(devices.begin(), devices.end()).size() !=
              devices.size())
        throw std::invalid_argument("invalid participating local inventory");
      auto topology = inspectTopology(devices);
      if (calibration) {
        auto found = std::find_if(
            calibration->hosts.begin(), calibration->hosts.end(),
            [&](auto const& h) { return h.hostname == wire.host; });
        if (found == calibration->hosts.end() ||
            found->topologyIdentity != topology.hardwareIdentity ||
            found->transportIdentity != networkTransportIdentity())
          throw std::invalid_argument(
              "automatic network calibration identity changed");
      }
      std::set<std::string> environment;
      for (char** item = environ; *item; ++item)
        if (!std::strncmp(*item, "MSCCLPP_", 8)) environment.insert(*item);
      std::string local = topology.hardwareIdentity;
      for (auto const& entry : environment) local += ';' + entry;
      if (options.allowNetwork && options.allowDirect &&
          !options.fallbackOnly && context.ranks >= 2 &&
          (options.dtype == ScalarType::Float32 ||
           options.dtype == ScalarType::BFloat16)) {
        wire.directCandidate =
            DirectExchangeCollectives::supportsCurrentDevice(options.dtype);
        if (wire.directCandidate) local += ';' + networkTransportIdentity();
      }
      wire.localIdentity = hash(local);
      // Resource configuration is immutable for a selector that can use it.
      // Global native-only/dtype-ineligible instances do not read this file.
      if (options.allowNetwork && options.allowHierarchy &&
          !options.fallbackOnly && options.dtype == ScalarType::Float32 &&
          context.ranks == 8) {
        if (auto file = std::getenv("UCCL_LITE_NETWORK_PROFILE")) {
          (void)loadNetworkProfile(file);
          std::ifstream input(file);
          std::string bytes((std::istreambuf_iterator<char>(input)), {});
          if (!input || bytes.empty())
            throw std::runtime_error("network profile read failed");
          wire.profile = hash(bytes);
        }
      }
      artifact = artifactIdentity(
          reinterpret_cast<uintptr_t>(&qualification::value<uint32_t>));
      wire.artifact = hash(artifact);
      Workload w{Collective::AllReduce, ScalarType::Float32, 128, false, 4};
      wire.candidate =
          property.cooperativeLaunch && peerPacketEligible(topology, w);
    } catch (std::exception const& error) {
      wire.valid = 0;
      std::fprintf(stderr, "distributed preparation rank=%d: %s\n",
                   context.rank, error.what());
    }
    auto votes = gather(wire);
    std::map<std::string, std::vector<int>> hosts;
    std::ostringstream topologyKey, baseline;
    capability = true;
    directCapability = true;
    for (int r = 0; r < context.ranks; ++r) {
      auto const& v = votes[r];
      if (!v.valid || !wire.valid || v.rank != r || v.request != wire.request ||
          v.options != wire.options || v.artifact != wire.artifact ||
          v.capacity != wire.capacity || v.profile != wire.profile)
        throw std::invalid_argument(
            "distributed workload/configuration consensus failed");
      for (int earlier : hosts[v.host])
        if (votes[earlier].pid == v.pid ||
            !std::memcmp(votes[earlier].gpu, v.gpu, 16))
          throw std::invalid_argument(
              "distributed selector requires distinct rank processes and GPUs");
      hosts[v.host].push_back(r);
      capability &= v.candidate != 0;
      directCapability &= v.directCandidate != 0;
      topologyKey << v.host << ':' << v.device << ':' << v.localIdentity << ':'
                  << v.profile << ';';
      baseline << v.baselineLocal << ';';
    }
    hostCounts.clear();
    for (auto const& h : hosts) hostCounts.push_back(int(h.second.size()));
    baselineIdentity = options.fallback.identity + ';' + baseline.str();
    auto current = topologyKey.str() + "|" + baselineIdentity;
    if (!identity.empty() && identity != current)
      throw std::invalid_argument(
          "distributed topology/software/baseline identity changed");
    identity = std::move(current);
  }
  void prepareBaseline() {
    int ready = 1;
    try {
      options.fallback.backend->prepare();
    } catch (std::exception const& error) {
      ready = 0;
      std::fprintf(stderr, "distributed baseline rank=%d: %s\n", context.rank,
                   error.what());
    }
    auto votes = gather(ready);
    if (std::find(votes.begin(), votes.end(), 0) != votes.end())
      throw std::runtime_error(
          "distributed caller baseline preparation failed");
    if (!done) check(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
  }
  std::unique_ptr<CollectiveBackend> createNetwork(
      NetworkBackend kind = NetworkBackend::Hierarchy) {
    auto registration = kind == NetworkBackend::Hierarchy
                            ? networkHierarchyBackend()
                            : directExchangeBackend(options.dtype);
    struct Id {
      int ready = 1;
      NetworkId value{};
    } id;
    if (!context.rank) {
      try {
        id.value = registration.uniqueId();
      } catch (...) {
        id.ready = 0;
      }
    }
    id = gather(id).front();
    if (!id.ready) throw std::runtime_error("network candidate ID unavailable");
    auto c = context;
    c.id = id.value;
    auto result = kind == NetworkBackend::Hierarchy
                      ? preparedNetworkBackend(c, calibration)
                      : registration.create(c);
    result->prepare();
    if (kind == NetworkBackend::DirectExchange) {
      auto text = result->configurationIdentity();
      auto lengths = gather(text.size());
      size_t width = *std::max_element(lengths.begin(), lengths.end());
      if (!width || width > (1UL << 20))
        throw std::runtime_error("invalid direct configuration identity");
      std::vector<char> data(width * context.ranks);
      std::memcpy(data.data() + width * context.rank, text.data(), text.size());
      control->allGather(data.data(), int(width));
      std::vector<std::string> identities;
      for (int r = 0; r < context.ranks; ++r)
        identities.emplace_back(data.data() + r * width, lengths[r]);
      if (!directIdentities.empty() && directIdentities != identities)
        throw std::runtime_error(
            "direct executor placement/transport identity changed");
      directIdentities = std::move(identities);
    }
    barrier();
    return result;
  }
  void describeDirectModel(CollectiveBackend& backend,
                           CollectiveRequest const& request,
                           DirectNetworkEvidence& evidence) {
    if (!options.probeResources) {
      evidence.model.reason = "direct model probing disabled";
      return;
    }
    if (!directModel) directModel = calibrateDirectExchange(backend);
    if (directModel->configurationIdentities != directIdentities)
      throw std::runtime_error(
          "direct calibration does not match current execution identity");
    auto op = request.operation == NetworkOperation::AllReduce
                  ? Collective::AllReduce
              : request.operation == NetworkOperation::AllGather
                  ? Collective::AllGather
                  : Collective::ReduceScatter;
    unsigned alignment = 256;
    bool inplace = false;
    for (auto layout : request.layouts) {
      alignment = std::min(alignment, layout.alignment);
      inplace |= layout.inPlace;
    }
    Workload work{op, options.dtype,
                  request.count * scalarBytes(options.dtype) *
                      (op == Collective::AllReduce ? 1 : context.ranks),
                  inplace, alignment};
    evidence.model = estimateDirectModel(*directModel, work);
    evidence.modelIdentity = directModel->identity;
    evidence.modelObservations = directModel->observations.size();
    evidence.unmodeledResources = directModel->unmodeledResources;
  }
  bool block(CollectiveBackend& backend, qualification::Probe& probe,
             CollectiveRequest const& request, int& seed,
             std::vector<double>& samples) {
    bool correct = true;
    for (int iteration = 0; iteration < 5; ++iteration, ++seed) {
      probe.initialize(seed);
      check(cudaStreamSynchronize(probe.stream));
      barrier();
      auto start = std::chrono::steady_clock::now();
      backend.enqueue(request.operation, probe.input(), probe.output(),
                      request.count, probe.stream);
      backend.synchronize();
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
    return correct;
  }
  void resetAlternatives() {
    if (direct) direct->synchronize();
    if (network) network->synchronize();
    direct.reset();
    network.reset();
    barrier();
  }
  void restore(bool hierarchy, bool exchange) {
    if (hierarchy && !network) network = createNetwork();
    if (exchange && !direct)
      direct = createNetwork(NetworkBackend::DirectExchange);
  }
  CollectiveBackend& executor(NetworkBackend kind) {
    if (kind == NetworkBackend::Hierarchy && network) return *network;
    if (kind == NetworkBackend::DirectExchange && direct) return *direct;
    throw std::logic_error("prepared network executor absent");
  }
  qualification::Probe* allocate(qualification::Probe& probe,
                                 CollectiveRequest const& request) {
    int allocated = 1;
    try {
      probe.allocate(request, context.rank, context.ranks, options.dtype);
    } catch (...) {
      allocated = 0;
    }
    for (auto vote : gather(allocated))
      if (!vote)
        throw std::runtime_error("network qualification allocation failed");
    return &probe;
  }
  size_t finish(size_t plan) {
    auto needs = [&] {
      std::pair<bool, bool> result{};
      for (auto const& p : plans) {
        result.first |= p.backend == NetworkBackend::Hierarchy;
        result.second |= p.backend == NetworkBackend::DirectExchange;
      }
      return result;
    };
    auto required = needs();
    bool pending = std::any_of(plans.begin(), plans.end(), [](auto const& p) {
      return p.backend != NetworkBackend::Caller &&
             p.poolCandidateSamples.empty();
    });
    if (required.first && required.second && pending) {
      // Validate the actual simultaneous resource pool. Native blocks remain
      // free of BOTH candidate executors, including resources for older plans.
      resetAlternatives();
      for (auto& p : plans) {
        if (p.backend == NetworkBackend::Caller ||
            !p.poolCandidateSamples.empty())
          continue;
        p.poolCandidateSamples.clear();
        p.poolBaselineSamples.clear();
        qualification::Probe probe;
        allocate(probe, p.request);
        int seed = 0;
        bool correct = true;
        for (int round = 0; round < 3; ++round) {
          if (!block(*options.fallback.backend, probe, p.request, seed,
                     p.poolBaselineSamples))
            throw std::runtime_error(
                "caller baseline failed pool qualification");
          restore(true, true);
          for (int repeat = 0; repeat < 2; ++repeat)
            correct &= block(executor(p.backend), probe, p.request, seed,
                             p.poolCandidateSamples);
          resetAlternatives();
          if (!block(*options.fallback.backend, probe, p.request, seed,
                     p.poolBaselineSamples))
            throw std::runtime_error(
                "caller baseline failed after pool teardown");
        }
        if (qualifiedLocalBackend(p.poolBaselineSamples, p.poolCandidateSamples,
                                  correct) != LocalBackend::Host) {
          p.backend = NetworkBackend::Caller;
          p.reason =
              "caller retained: simultaneous executor pool failed "
              "correctness/gain qualification";
        }
      }
      consensus(plans[plan].request);
      prepareBaseline();
      required = needs();
      if (!required.first || !required.second) {
        for (auto& p : plans)
          if (p.backend != NetworkBackend::Caller) {
            p.poolCandidateSamples.clear();
            p.poolBaselineSamples.clear();
          }
      }
    }
    if (!required.first && network) network.reset();
    if (!required.second && direct) direct.reset();
    restore(required.first, required.second);
    valid = true;
    return plan;
  }
};
NetworkId PreparedNetworkRuntime::createUniqueId() {
  return mscclpp::TcpBootstrap::createUniqueId();
}
PreparedNetworkRuntime::PreparedNetworkRuntime(BackendContext c,
                                               NetworkPreparationOptions o)
    : impl_(std::make_unique<Impl>(std::move(c), std::move(o))) {}
PreparedNetworkRuntime::~PreparedNetworkRuntime() = default;
size_t PreparedNetworkRuntime::prepare(CollectiveRequest const& request) {
  auto& s = *impl_;
  s.valid = false;
  s.consensus(request);
  s.prepareBaseline();
  if (s.cacheLoaded) s.validateCacheContext();
  size_t cached = s.plans.size();
  for (size_t i = 0; i < s.plans.size(); ++i)
    if (key(s.plans[i].request) == key(request)) {
      cached = i;
      break;
    }
  struct Cache {
    size_t size, index;
  } cache{s.plans.size(), cached};
  for (auto v : s.gather(cache))
    if (v.size != cache.size || v.index != cached)
      throw std::logic_error("distributed plan cache consensus failed");
  if (cached < s.plans.size()) return s.finish(cached);
  s.resetAlternatives();
  NetworkSelection selected;
  selected.request = request;
  selected.topologyIdentity = s.identity;
  selected.executorIdentity = s.artifact;
  selected.baselineIdentity = s.baselineIdentity;
  size_t bytes = request.count * scalarBytes(s.options.dtype) *
                 (int(request.operation) == 3 ? 1 : s.context.ranks);
  selected.candidateEligible =
      s.options.allowNetwork && s.options.allowHierarchy &&
      !s.options.fallbackOnly && !request.fallbackOnly &&
      s.options.dtype == ScalarType::Float32 &&
      s.hostCounts == std::vector<int>({4, 4}) && s.capability &&
      !(bytes % 128);
  selected.reason =
      "caller baseline: network layout/dtype/extent/capability unavailable or "
      "disabled";
  if (selected.candidateEligible) {
    s.loadCache();
    selected.cacheStatus =
        s.options.planCache.empty() ? "disabled" : s.cacheNotice;
    int available = 1;
    NetworkPreflight proposal;
    try {
      proposal = preflightNetwork(s.context, *s.control, s.calibration,
                                  s.options.probeResources);
      s.calibration = proposal.calibration;
    } catch (NetworkCalibrationFailure const&) {
      throw;
    } catch (std::exception const& error) {
      available = 0;
      selected.reason =
          std::string("caller baseline: network preflight unavailable: ") +
          error.what();
    }
    auto votes = s.gather(available);
    selected.candidateEligible =
        std::find(votes.begin(), votes.end(), 0) == votes.end();
    if (selected.candidateEligible) {
      selected.executionRankOrder.assign(proposal.rankOrder.begin(),
                                         proposal.rankOrder.end());
      selected.rails = s.gather(proposal.rail);
    }
  }
  selected.calibration = s.calibration;
  if (selected.candidateEligible) {
    auto saved = s.cachedRecord(request);
    if (saved && (saved->rankOrder != selected.executionRankOrder ||
                  saved->rails != selected.rails)) {
      saved.reset();
      selected.cacheStatus = "execution-map miss";
    }
    if (s.network) {
      s.network->synchronize();
      s.network.reset();
    }
    s.barrier();
    qualification::Probe probe;
    int allocated = 1;
    try {
      probe.allocate(request, s.context.rank, s.context.ranks, s.options.dtype);
    } catch (...) {
      allocated = 0;
    }
    auto votes = s.gather(allocated);
    if (std::find(votes.begin(), votes.end(), 0) != votes.end())
      throw std::runtime_error("distributed qualification allocation failed");
    int seed = 0;
    selected.candidateCorrect = true;
    bool fresh = true, persistFresh = true;
    if (saved) {
      if (!s.block(*s.options.fallback.backend, probe, request, seed,
                   selected.validationBaselineSamples))
        throw std::runtime_error(
            "cached caller baseline failed full-buffer revalidation");
      bool correct = true;
      if (saved->useHierarchy) {
        auto candidate = s.createNetwork();
        for (int repeat = 0; repeat < 2; ++repeat)
          correct &= s.block(*candidate, probe, request, seed,
                             selected.validationCandidateSamples);
        candidate.reset();
        s.barrier();
        if (!s.block(*s.options.fallback.backend, probe, request, seed,
                     selected.validationBaselineSamples))
          throw std::runtime_error(
              "cached caller baseline failed after candidate teardown");
      }
      if (!saved->useHierarchy ||
          (correct && qualifiedLocalBackend(selected.validationBaselineSamples,
                                            selected.validationCandidateSamples,
                                            true) == LocalBackend::Host)) {
        selected.fromCache = true;
        selected.candidateCorrect = saved->candidateCorrect;
        selected.candidateSamples = saved->candidateSamples;
        selected.baselineSamples = saved->baselineSamples;
        selected.backend = saved->useHierarchy ? NetworkBackend::Hierarchy
                                               : NetworkBackend::Caller;
        selected.cacheStatus =
            saved->useHierarchy
                ? "hit: hierarchy ABBA revalidated"
                : "hit: caller revalidated without candidate initialization";
        selected.reason = selected.cacheStatus;
        fresh = false;
      } else if (!correct) {
        selected.candidateCorrect = false;
        selected.candidateSamples = selected.validationCandidateSamples;
        selected.baselineSamples = selected.validationBaselineSamples;
        selected.cacheStatus =
            "evicted: candidate failed correctness revalidation";
        selected.reason = selected.cacheStatus;
        s.persist(selected, true);
        fresh = false;
        persistFresh = false;
      } else
        selected.cacheStatus =
            "performance drift: full qualification refreshed";
    }
    for (int round = 0; fresh && round < 3; ++round) {
      if (!s.block(*s.options.fallback.backend, probe, request, seed,
                   selected.baselineSamples))
        throw std::runtime_error(
            "caller baseline failed full-buffer qualification");
      auto candidate = s.createNetwork();
      for (int repeat = 0; repeat < 2; ++repeat)
        selected.candidateCorrect &= s.block(*candidate, probe, request, seed,
                                             selected.candidateSamples);
      candidate.reset();
      s.barrier();
      if (!s.block(*s.options.fallback.backend, probe, request, seed,
                   selected.baselineSamples))
        throw std::runtime_error(
            "caller baseline failed after candidate teardown");
    }
    if (fresh) {
      int chosen = s.context.rank == 0 &&
                   qualifiedLocalBackend(
                       selected.baselineSamples, selected.candidateSamples,
                       selected.candidateCorrect) == LocalBackend::Host;
      chosen = s.gather(chosen).front();
      selected.backend =
          chosen ? NetworkBackend::Hierarchy : NetworkBackend::Caller;
      selected.reason =
          chosen ? "network qualified against paired caller baseline"
                 : "caller baseline retained: no robust correct network gain";
    }
    // Revalidate live identities after both cache checkpoints and cold trials.
    s.consensus(request);
    if (s.cacheLoaded) s.validateCacheContext();
    s.prepareBaseline();
    if (fresh && persistFresh) s.persist(selected);
  }
  auto& direct = selected.direct;
  direct.eligible = s.options.allowNetwork && s.options.allowDirect &&
                    !s.options.fallbackOnly && !request.fallbackOnly &&
                    s.hostCounts.size() > 1 && s.context.ranks >= 2 &&
                    s.directCapability &&
                    (s.options.dtype == ScalarType::Float32 ||
                     s.options.dtype == ScalarType::BFloat16);
  direct.reason = "direct capability unavailable or disabled";
  bool directQualified = false;
  if (direct.eligible) {
    s.loadCache();
    direct.cacheStatus =
        s.options.planCache.empty() ? "disabled" : s.cacheNotice;
    auto saved = s.cachedRecord(request, NetworkCacheCandidate::DirectExchange);
    s.resetAlternatives();
    qualification::Probe probe;
    s.allocate(probe, request);
    int seed = 0;
    bool fresh = true, persistFresh = true;
    direct.correct = true;
    if (saved) {
      if (!s.block(*s.options.fallback.backend, probe, request, seed,
                   direct.validationBaselineSamples))
        throw std::runtime_error(
            "cached direct caller baseline failed full-buffer revalidation");
      bool correct = true, matching = true;
      if (saved->useDirect) {
        auto candidate = s.createNetwork(NetworkBackend::DirectExchange);
        direct.traffic = candidate->traffic(request.operation, request.count);
        matching = saved->configurationIdentities == s.directIdentities;
        if (matching) s.describeDirectModel(*candidate, request, direct);
        if (matching)
          for (int repeat = 0; repeat < 2; ++repeat)
            correct &= s.block(*candidate, probe, request, seed,
                               direct.validationSamples);
        candidate.reset();
        s.barrier();
        if (!s.block(*s.options.fallback.backend, probe, request, seed,
                     direct.validationBaselineSamples))
          throw std::runtime_error(
              "cached direct caller baseline failed after candidate teardown");
      }
      if (!matching) {
        direct.cacheStatus =
            "execution identity miss: full direct qualification refreshed";
      } else if (!saved->useDirect ||
                 (correct &&
                  qualifiedLocalBackend(direct.validationBaselineSamples,
                                        direct.validationSamples,
                                        true) == LocalBackend::Host)) {
        direct.fromCache = true;
        direct.correct = saved->candidateCorrect;
        direct.samples = saved->candidateSamples;
        direct.baselineSamples = saved->baselineSamples;
        direct.configurationIdentities = saved->configurationIdentities;
        directQualified = saved->useDirect;
        direct.cacheStatus =
            saved->useDirect
                ? "hit: direct ABBA revalidated"
                : "hit: caller revalidated without direct initialization";
        fresh = false;
      } else if (!correct) {
        direct.correct = false;
        direct.samples = direct.validationSamples;
        direct.baselineSamples = direct.validationBaselineSamples;
        direct.configurationIdentities = s.directIdentities;
        direct.cacheStatus = "evicted: direct failed correctness revalidation";
        s.persist(selected, true, NetworkCacheCandidate::DirectExchange);
        fresh = false;
        persistFresh = false;
      } else
        direct.cacheStatus =
            "performance drift: full direct qualification refreshed";
    }
    for (int round = 0; fresh && round < 3; ++round) {
      if (!s.block(*s.options.fallback.backend, probe, request, seed,
                   direct.baselineSamples))
        throw std::runtime_error("caller baseline failed direct qualification");
      auto candidate = s.createNetwork(NetworkBackend::DirectExchange);
      direct.traffic = candidate->traffic(request.operation, request.count);
      s.describeDirectModel(*candidate, request, direct);
      for (int repeat = 0; repeat < 2; ++repeat)
        direct.correct &=
            s.block(*candidate, probe, request, seed, direct.samples);
      candidate.reset();
      s.barrier();
      if (!s.block(*s.options.fallback.backend, probe, request, seed,
                   direct.baselineSamples))
        throw std::runtime_error(
            "caller baseline failed after direct teardown");
    }
    if (fresh) {
      direct.configurationIdentities = s.directIdentities;
      directQualified =
          qualifiedLocalBackend(direct.baselineSamples, direct.samples,
                                direct.correct) == LocalBackend::Host;
    }
    direct.reason = direct.fromCache ? direct.cacheStatus
                    : directQualified
                        ? "direct qualified against paired caller baseline"
                        : "caller retained: no robust correct direct gain";
    s.consensus(request);
    if (s.cacheLoaded) s.validateCacheContext();
    s.prepareBaseline();
    if (fresh && persistFresh)
      s.persist(selected, false, NetworkCacheCandidate::DirectExchange);
  }
  auto winner = chooseNetworkCandidate(
      selected.backend == NetworkBackend::Hierarchy,
      selected.fromCache && !selected.validationCandidateSamples.empty()
          ? selected.validationCandidateSamples
          : selected.candidateSamples,
      directQualified,
      direct.fromCache && !direct.validationSamples.empty()
          ? direct.validationSamples
          : direct.samples);
  if (winner == NetworkBackend::DirectExchange)
    selected.reason = direct.reason;
  else if (winner == NetworkBackend::Caller && direct.eligible)
    selected.reason =
        "caller retained: no qualified network candidate; " + direct.reason;
  selected.backend = winner;
  // Catch configuration changes during trials before admitting application
  // work.
  s.consensus(request);
  s.prepareBaseline();
  s.plans.push_back(std::move(selected));
  return s.finish(s.plans.size() - 1);
}
void PreparedNetworkRuntime::enqueue(size_t plan, void const* input,
                                     void* output, cudaStream_t stream) {
  auto& s = *impl_;
  if (!s.valid)
    throw std::logic_error(
        "distributed enqueue requires successful collective preparation");
  auto const& p = s.plans.at(plan);
  auto l = p.request.layouts[s.context.rank];
  int op = int(p.request.operation), device;
  size_t scalar = scalarBytes(s.options.dtype), count = p.request.count;
  auto a = reinterpret_cast<uintptr_t>(input),
       b = reinterpret_cast<uintptr_t>(output);
  size_t in = count * scalar * (op == 2 ? s.context.ranks : 1),
         out = count * scalar * (op == 1 ? s.context.ranks : 1);
  if (!a || !b || a > UINTPTR_MAX - in || b > UINTPTR_MAX - out)
    throw std::invalid_argument("invalid distributed buffers");
  bool overlap = a < b + out && b < a + in;
  bool legal = op == 1   ? a == b + size_t(s.context.rank) * count * scalar
               : op == 2 ? b == a + size_t(s.context.rank) * count * scalar
                         : a == b;
  if (alignment(input, output) != l.alignment || overlap != l.inPlace ||
      (l.inPlace && !legal))
    throw std::invalid_argument(
        "distributed buffers do not match prepared layout");
  check(cudaGetDevice(&device));
  if (device != s.context.device)
    throw std::invalid_argument("distributed current device mismatch");
  cudaStreamCaptureStatus capture;
  check(cudaStreamIsCapturing(stream, &capture));
  if (capture != cudaStreamCaptureStatusNone)
    throw std::invalid_argument("distributed capture is not qualified");
  if (s.recorded && stream != s.last)
    check(cudaStreamWaitEvent(stream, s.done, 0));
  auto* backend = p.backend == NetworkBackend::Caller
                      ? s.options.fallback.backend.get()
                      : &s.executor(p.backend);
  if (!backend) throw std::logic_error("prepared network executor absent");
  backend->enqueue(p.request.operation, input, output, count, stream);
  check(cudaEventRecord(s.done, stream));
  s.recorded = true;
  s.last = stream;
}
void PreparedNetworkRuntime::synchronize() {
  auto& s = *impl_;
  if (s.recorded) check(cudaEventSynchronize(s.done));
  if (s.network) s.network->synchronize();
  if (s.direct) s.direct->synchronize();
  if (s.options.fallback.backend) s.options.fallback.backend->synchronize();
}
bool PreparedNetworkRuntime::idle() const {
  return (!impl_->recorded || cudaEventQuery(impl_->done) == cudaSuccess) &&
         (!impl_->network || impl_->network->idle()) &&
         (!impl_->direct || impl_->direct->idle());
}
bool PreparedNetworkRuntime::networkInitialized() const {
  return bool(impl_->network) || bool(impl_->direct);
}
NetworkSelection const& PreparedNetworkRuntime::selection(size_t plan) const {
  return impl_->plans.at(plan);
}
}  // namespace uccl_lite::runtime
