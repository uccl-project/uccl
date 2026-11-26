#include "network_preflight.hpp"
#include "network_probe.hpp"
#include "rank_permutation.hpp"
#include "uccl_lite/collectives.h"
#include "uccl_lite/local_plan.hpp"
#include "uccl_lite/network_plan.hpp"
#include "uccl_lite/runtime/collective_backend.hpp"
#include "uccl_lite/runtime/network_identity.hpp"
#include <mscclpp/core.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

namespace uccl_lite::runtime {
namespace {
std::vector<RailDevice> rails() {
  auto const& configured = mscclpp::env()->hcaDevices;
  int count =
      configured.empty()
          ? mscclpp::getIBDeviceCount()
          : int(std::count(configured.begin(), configured.end(), ',')) + 1;
  std::vector<RailDevice> result;
  for (int i = 0; i < std::min(8, count); ++i) {
    auto transport =
        static_cast<mscclpp::Transport>(int(mscclpp::Transport::IB0) + i);
    auto name = mscclpp::getIBDeviceName(transport);
    if (!name.empty()) result.push_back({name, i});
  }
  return result;
}
uint64_t digest(NetworkProfile const& p) {
  std::ostringstream data;
  data << std::setprecision(17) << std::quoted(p.probeIdentity) << ' '
       << std::quoted(p.pathScope) << ' ' << std::quoted(p.groupScope);
  for (auto const& h : p.hosts)
    data << ' ' << std::quoted(h.hostname) << ' '
         << std::quoted(h.topologyIdentity) << ' '
         << std::quoted(h.transportIdentity);
  for (auto const& path : p.paths) {
    auto const& k = path.key;
    data << ' ' << k.sourceHost << ' ' << k.destinationHost << ' '
         << std::quoted(k.sourceGpu) << ' ' << std::quoted(k.destinationGpu)
         << ' ' << std::quoted(k.sourceNic) << ' '
         << std::quoted(k.destinationNic) << ' ' << k.sourcePort << ' '
         << k.destinationPort << ' ' << path.warmups << ' ' << path.iterations
         << ' ' << path.repetitions << ' ' << std::quoted(path.rejection) << ' '
         << path.sizes.size();
    for (auto const& o : path.sizes) {
      data << ' ' << o.bytes << ' ' << o.microseconds.size();
      for (double value : o.microseconds) data << ' ' << value;
    }
  }
  data << ' ' << std::quoted(p.groupControlIdentities[0]) << ' '
       << std::quoted(p.groupControlIdentities[1]) << ' ' << p.groups.size();
  for (auto const& group : p.groups) {
    data << ' ' << std::quoted(group.label) << ' ' << group.warmups << ' '
         << group.iterations << ' ' << group.repetitions << ' '
         << std::quoted(group.rejection) << ' ' << group.flows.size() << ' '
         << group.sizes.size();
    for (auto const& k : group.flows)
      data << ' ' << k.sourceHost << ' ' << k.destinationHost << ' '
           << std::quoted(k.sourceGpu) << ' ' << std::quoted(k.destinationGpu)
           << ' ' << std::quoted(k.sourceNic) << ' '
           << std::quoted(k.destinationNic) << ' ' << k.sourcePort << ' '
           << k.destinationPort;
    for (auto const& o : group.sizes) {
      data << ' ' << o.bytes << ' ' << o.senderMicroseconds.size();
      for (size_t s = 0; s < o.senderMicroseconds.size(); ++s)
        for (size_t r = 0; r < group.flows.size(); ++r)
          data << ' ' << o.senderMicroseconds[s][r] << ' '
               << o.cohortMicroseconds[s][r];
    }
  }
  uint64_t hash = 14695981039346656037ULL;
  for (unsigned char c : data.str()) hash = (hash ^ c) * 1099511628211ULL;
  return hash;
}
template <class T>
std::vector<T> gather(mscclpp::Bootstrap& bootstrap, T const& local) {
  std::vector<T> values(bootstrap.getNranks());
  values[bootstrap.getRank()] = local;
  bootstrap.allGather(values.data(), sizeof(T));
  return values;
}
std::vector<std::string> strings(mscclpp::Bootstrap& bootstrap,
                                 std::string const& local) {
  auto lengths = gather(bootstrap, local.size());
  size_t width = *std::max_element(lengths.begin(), lengths.end());
  if (!width || width > (1 << 20))
    throw std::invalid_argument("network metadata extent invalid");
  std::vector<char> data(width * lengths.size());
  std::memcpy(data.data() + width * bootstrap.getRank(), local.data(),
              local.size());
  bootstrap.allGather(data.data(), int(width));
  std::vector<std::string> result;
  for (size_t r = 0; r < lengths.size(); ++r)
    result.emplace_back(data.data() + r * width, lengths[r]);
  return result;
}
struct EndpointRecord {
  NetworkEndpoint endpoint;
  int fallback = -1;
};
std::string encode(EndpointRecord const& record) {
  auto const& e = record.endpoint;
  std::ostringstream out;
  out << std::setprecision(17) << std::quoted(e.host.hostname) << ' '
      << std::quoted(e.host.topologyIdentity) << ' '
      << std::quoted(e.host.transportIdentity) << ' ' << std::quoted(e.gpu)
      << ' ' << e.device << ' ' << record.fallback << ' '
      << e.candidates.size();
  for (auto const& candidate : e.candidates) {
    auto const& a = candidate.assignment;
    out << ' ' << a.device << ' ' << a.transportIndex << ' '
        << std::quoted(a.nic) << ' ' << a.port << ' ' << a.capacityGBs << ' '
        << candidate.requiresMeasurement;
    for (auto const* map : {&candidate.transmit, &candidate.receive}) {
      out << ' ' << map->size();
      for (auto const& [name, rate] : *map)
        out << ' ' << std::quoted(name) << ' ' << rate;
    }
  }
  return out.str();
}
EndpointRecord decode(std::string const& text) {
  EndpointRecord record;
  auto& e = record.endpoint;
  std::istringstream in(text);
  size_t count;
  if (!(in >> std::quoted(e.host.hostname) >>
        std::quoted(e.host.topologyIdentity) >>
        std::quoted(e.host.transportIdentity) >> std::quoted(e.gpu) >>
        e.device >> record.fallback >> count) ||
      count > 8)
    throw std::invalid_argument("invalid network endpoint metadata");
  e.candidates.resize(count);
  for (auto& candidate : e.candidates) {
    auto& a = candidate.assignment;
    if (!(in >> a.device >> a.transportIndex >> std::quoted(a.nic) >> a.port >>
          a.capacityGBs >> candidate.requiresMeasurement))
      throw std::invalid_argument("invalid network candidate metadata");
    for (auto* map : {&candidate.transmit, &candidate.receive}) {
      size_t resources;
      if (!(in >> resources) || resources > 256)
        throw std::invalid_argument("invalid network resource extent");
      for (size_t i = 0; i < resources; ++i) {
        std::string name;
        double rate;
        if (!(in >> std::quoted(name) >> rate) ||
            !map->emplace(name, rate).second)
          throw std::invalid_argument("invalid duplicate network resource");
      }
    }
  }
  in >> std::ws;
  if (!in.eof()) throw std::invalid_argument("trailing network endpoint data");
  return record;
}
struct RankMapping {
  std::array<int, 8> order{};
  std::array<int, 2> nodes{};
  int executionRank = -1;
  bool remap = false;
};
RankMapping planRanks(BackendContext const& context,
                      mscclpp::Bootstrap& bootstrap) {
  struct Wire {
    int ready = 1, device = -1;
    size_t capacity = 0;
    std::array<int, 4> devices{}, order{};
    std::array<int, 2> nodes{};
    std::array<std::array<char, 16>, 4> uuids{};
    char host[256]{};
  } local;
  try {
    if (context.localDevices.size() != 4 ||
        std::find(context.localDevices.begin(), context.localDevices.end(),
                  context.device) == context.localDevices.end() ||
        !context.capacity || context.capacity > (1UL << 30) ||
        context.capacity % 128)
      throw std::invalid_argument("network participant extent/device mismatch");
    if (cudaGetDevice(&local.device) != cudaSuccess ||
        local.device != context.device ||
        cudaDeviceSynchronize() != cudaSuccess)
      throw std::invalid_argument(
          "network preparation requires idle current device");
    if (!NetworkCollectives::supportsCurrentDevice())
      throw std::invalid_argument(
          "network compiled image/device capabilities unsupported");
    auto topology = inspectTopology(context.localDevices);
    auto order = pairedNumaRankOrder(topology);
    Workload workload{Collective::AllReduce, ScalarType::Float32, 128, false,
                      4};
    if (order.empty() || !peerPacketEligible(topology, workload))
      throw std::invalid_argument(
          "network local stage requires two peer-accessible NUMA pairs");
    std::copy(order.begin(), order.end(), local.order.begin());
    std::copy(context.localDevices.begin(), context.localDevices.end(),
              local.devices.begin());
    local.nodes = {topology.numaNodes[order[0]], topology.numaNodes[order[2]]};
    local.capacity = context.capacity;
    for (int i = 0; i < 4; ++i) {
      cudaDeviceProp property{};
      if (cudaGetDeviceProperties(&property, local.devices[i]) != cudaSuccess)
        throw std::runtime_error("network GPU identity unavailable");
      std::memcpy(local.uuids[i].data(), property.uuid.bytes, 16);
    }
    if (gethostname(local.host, sizeof(local.host) - 1))
      throw std::runtime_error("network host identity unavailable");
  } catch (std::exception const& error) {
    local.ready = 0;
    std::fprintf(stderr, "network rank mapping rank=%d: %s\n", context.rank,
                 error.what());
  }
  auto metadata = gather(bootstrap, local);
  // Host groups follow first occurrence, keeping execution rank zero on the
  // original root's host. Global logical ranks need not be host-contiguous.
  std::vector<std::vector<int>> hosts;
  for (int rank = 0; rank < 8; ++rank) {
    auto const& v = metadata[rank];
    if (!v.ready || !v.host[0] || v.capacity != local.capacity)
      throw std::invalid_argument(
          "network participant metadata consensus failed");
    auto group =
        std::find_if(hosts.begin(), hosts.end(), [&](auto const& ranks) {
          return !std::strcmp(v.host, metadata[ranks.front()].host);
        });
    if (group == hosts.end())
      hosts.push_back({rank});
    else
      group->push_back(rank);
  }
  if (hosts.size() != 2 || hosts[0].size() != 4 || hosts[1].size() != 4)
    throw std::invalid_argument(
        "network hierarchy requires two actual four-GPU hosts");
  RankMapping result;
  for (size_t host = 0; host < hosts.size(); ++host) {
    auto const& ranks = hosts[host];
    auto const& first = metadata[ranks.front()];
    std::set<int> devices;
    std::set<std::array<char, 16>> uuids(first.uuids.begin(),
                                         first.uuids.end());
    if (uuids.size() != 4)
      throw std::invalid_argument(
          "network host contains duplicate GPU identities");
    for (int rank : ranks) {
      auto const& v = metadata[rank];
      if (v.devices != first.devices || v.order != first.order ||
          v.nodes != first.nodes || v.uuids != first.uuids ||
          !devices.insert(v.device).second ||
          std::find(first.devices.begin(), first.devices.end(), v.device) ==
              first.devices.end())
        throw std::invalid_argument(
            "network rank/device/NUMA mapping consensus failed");
    }
    for (int slot = 0; slot < 4; ++slot) {
      int device = first.devices.at(first.order[slot]);
      auto owner = std::find_if(ranks.begin(), ranks.end(), [&](int rank) {
        return metadata[rank].device == device;
      });
      if (owner == ranks.end())
        throw std::invalid_argument("network execution device has no owner");
      size_t execution = host * 4 + slot;
      result.order[execution] = *owner;
      if (*owner == context.rank) result.executionRank = int(execution);
      result.remap |= *owner != int(execution);
    }
  }
  result.nodes = local.nodes;
  return result;
}
int selectRail(BackendContext const& context, mscclpp::Bootstrap& bootstrap,
               RankMapping const& mapping,
               std::shared_ptr<NetworkProfile const>& calibration,
               bool probeResources = false) {
  struct Vote {
    int ready = 1, profile = 0, explicitRail = 0;
    uint64_t digest = 0;
  } vote;
  NetworkProfile profile;
  EndpointRecord record;
  std::string encoded;
  try {
    auto configuration = mscclpp::env();
    // This is a requirement of MSCCL++'s CUDA host-no-atomic signal forwarding,
    // not a general statement that GPU RDMA needs GDRCopy.
    if (configuration->ibvMode == "host-no-atomic" &&
        configuration->forceDisableGdr)
      throw std::invalid_argument(
          "MSCCL++ host-no-atomic requires GDRCopy, but GDRCopy is explicitly "
          "disabled");
    if (context.localDevices.empty())
      throw std::invalid_argument("network planning requires local devices");
    auto topology = inspectTopology(context.localDevices);
    auto available = rails();
    if (calibration) {
      vote.profile = 1;
      profile = *calibration;
      vote.digest = digest(profile);
    } else if (auto file = std::getenv("UCCL_LITE_NETWORK_PROFILE")) {
      vote.profile = 1;
      profile = loadNetworkProfile(file);
      vote.digest = digest(profile);
    } else if (probeResources && context.rail < 0) {
      int rdma = 0, ordering = 0, flush = 0;
      if (cudaDeviceGetAttribute(&rdma, cudaDevAttrGPUDirectRDMASupported,
                                 context.device) != cudaSuccess ||
          cudaDeviceGetAttribute(&ordering,
                                 cudaDevAttrGPUDirectRDMAWritesOrdering,
                                 context.device) != cudaSuccess ||
          cudaDeviceGetAttribute(&flush,
                                 cudaDevAttrGPUDirectRDMAFlushWritesOptions,
                                 context.device) != cudaSuccess ||
          !rdma ||
          (ordering < cudaGPUDirectRDMAWritesOrderingOwner &&
           !(flush & cudaFlushGPUDirectRDMAWritesOptionHost)))
        throw std::invalid_argument(
            "GPU RDMA registration/owner visibility unavailable for automatic "
            "probing");
      vote.profile =
          2;  // Live metadata first; private calibration after consensus.
    }
    auto found = std::find(context.localDevices.begin(),
                           context.localDevices.end(), context.device);
    if (found == context.localDevices.end())
      throw std::invalid_argument("rank device absent from network topology");
    size_t index = found - context.localDevices.begin();
    vote.explicitRail = context.rail >= 0;
    RailAssignment physical;
    if (vote.explicitRail) {
      auto rail = std::find_if(available.begin(), available.end(),
                               [&](auto const& entry) {
                                 return entry.transportIndex == context.rail;
                               });
      if (rail == available.end())
        throw std::invalid_argument("explicit network rail is unavailable");
      std::vector<NetworkPort> ports;
      for (auto const& port : topology.networkPorts)
        if (port.name == rail->name && port.active) ports.push_back(port);
      if (ports.size() != 1)
        throw std::invalid_argument(
            "explicit rail requires one unambiguous active HCA port");
      // An explicit experiment does not require a complete capacity model.
      physical = {context.device, context.rail, rail->name,
                  ports[0].port,  -1,           {}};
    } else {
      try {
        physical = planNetworkRails(topology, available)[index];
      } catch (std::invalid_argument const&) {
        if (!vote.profile) throw;
        // Validate topology/ports without inventing unknown bandwidth. A
        // missing fallback must later be replaced by complete measurements.
        (void)networkRailCandidates(topology, available, true);
        physical.device = context.device;
        physical.capacityGBs = -1;
      }
    }
    record.fallback = physical.transportIndex;
    if (vote.profile) {
      char host[256]{};
      if (gethostname(host, sizeof(host) - 1))
        throw std::runtime_error("network hostname unavailable");
      record.endpoint.host = {host, topology.hardwareIdentity,
                              networkTransportIdentity()};
      record.endpoint.device = context.device;
      cudaDeviceProp property{};
      if (cudaGetDeviceProperties(&property, context.device) != cudaSuccess)
        throw std::runtime_error("network GPU identity unavailable");
      char uuid[33]{};
      for (int i = 0; i < 16; ++i)
        std::snprintf(uuid + 2 * i, 3, "%02x",
                      static_cast<unsigned char>(property.uuid.bytes[i]));
      record.endpoint.gpu = uuid;
      if (!vote.explicitRail)
        record.endpoint.candidates =
            networkRailCandidates(topology, available, true)[index];
      encoded = encode(record);
    }
    std::fprintf(stderr,
                 "UCCL Lite network capacity path: rank=%d device=%d nic=%s "
                 "port=%d rail=%d capacity_bound_GBs=%.3f explicit=%d\n",
                 context.rank, context.device, physical.nic.c_str(),
                 physical.port, record.fallback, physical.capacityGBs,
                 vote.explicitRail);
  } catch (std::exception const& error) {
    vote.ready = 0;
    std::fprintf(stderr, "UCCL Lite network planning rank=%d: %s\n",
                 context.rank, error.what());
  }
  auto votes = gather(bootstrap, vote);
  bool explicitRail = false;
  for (auto const& v : votes) {
    if (!v.ready || v.profile != vote.profile || v.digest != vote.digest)
      throw std::invalid_argument(
          "network profile/topology preparation consensus failed");
    explicitRail |= v.explicitRail;
  }
  if (!vote.profile || explicitRail) {
    auto physical = gather(bootstrap, int(record.fallback >= 0));
    if (std::find(physical.begin(), physical.end(), 0) != physical.end())
      throw std::invalid_argument(
          "explicit rail policy leaves an unverified rank path");
    return record.fallback;
  }
  auto metadata = strings(bootstrap, encoded);
  std::vector<NetworkEndpoint> endpoints;
  std::vector<int> fallbacks;
  for (auto const& text : metadata) {
    auto item = decode(text);
    fallbacks.push_back(item.fallback);
    endpoints.push_back(std::move(item.endpoint));
  }
  std::vector<std::pair<size_t, size_t>> peers;
  for (size_t rank = 0; rank < 4; ++rank)
    peers.emplace_back(mapping.order[rank], mapping.order[rank + 4]);
  if (vote.profile == 2) {
    profile = probeNetworkResources(
        bootstrap, endpoints, peers, fallbacks,
        std::clamp(context.capacity, size_t(2) << 20, size_t(64) << 20));
    int ready = 1;
    try {
      calibration = std::make_shared<NetworkProfile const>(profile);
    } catch (...) {
      ready = 0;
    }
    auto allocated = gather(bootstrap, ready);
    if (std::find(allocated.begin(), allocated.end(), 0) != allocated.end())
      throw NetworkCalibrationFailure(
          "network calibration retention failed collectively");
    auto identities = gather(bootstrap, digest(profile));
    if (std::any_of(identities.begin(), identities.end(),
                    [&](auto x) { return x != identities.front(); }))
      throw NetworkCalibrationFailure(
          "network calibration observations disagree");
    std::fprintf(stderr,
                 "UCCL Lite automatic network calibration: rank=%d paths=%zu "
                 "cohorts=%zu identity=%llu\n",
                 context.rank, profile.paths.size(), profile.groups.size(),
                 static_cast<unsigned long long>(identities.front()));
  }
  struct Decision {
    int ready = 0, calibrated = 0;
    int groupMeasured = 0;
    std::array<int, 64> rails{};
    size_t bytes = 0;
    double pathUs = 0, capacityUs = 0;
    double groupUs = -1;
    char group[128]{};
    char reason[256]{};
  } decision;
  if (context.rank == 0) {
    try {
      for (size_t r = 0; r < metadata.size(); ++r) {
        decision.rails[r] = fallbacks[r];
      }
      // This retained provider declares {4,4}. Its pairing is part of the
      // executable algorithm, not inferred as a general topology rule.
      auto plan = planMeasuredNetworkRails(profile, endpoints, peers);
      if (calibration && !plan.calibrated)
        throw std::invalid_argument("automatic network evidence unavailable: " +
                                    plan.reason);
      decision.calibrated = plan.calibrated;
      if (plan.calibrated)
        for (size_t r = 0; r < endpoints.size(); ++r)
          decision.rails[r] = plan.assignments[r].transportIndex;
      if (std::any_of(decision.rails.begin(),
                      decision.rails.begin() + endpoints.size(),
                      [](int rail) { return rail < 0; }))
        throw std::invalid_argument(
            "no physical fallback or complete measured network path: " +
            plan.reason);
      decision.bytes = plan.basisBytes;
      decision.pathUs = plan.isolatedPathUs;
      decision.capacityUs = plan.capacityBoundUs;
      decision.groupMeasured = plan.concurrentMeasured;
      decision.groupUs = plan.concurrentUs;
      std::snprintf(decision.group, sizeof(decision.group), "%s",
                    plan.concurrentGroup.c_str());
      std::snprintf(decision.reason, sizeof(decision.reason), "%s",
                    plan.reason.c_str());
      decision.ready = 1;
    } catch (std::exception const& error) {
      std::snprintf(decision.reason, sizeof(decision.reason), "%s",
                    error.what());
    }
  }
  // Only rank zero's coupled choice is executed; every rank sees the same
  // selected physical pair even with different local HCA enumeration.
  decision = gather(bootstrap, decision).front();
  if (!decision.ready) throw std::runtime_error(decision.reason);
  int selected = decision.rails.at(context.rank);
  auto supported = std::find_if(
      record.endpoint.candidates.begin(), record.endpoint.candidates.end(),
      [&](auto const& candidate) {
        return candidate.assignment.transportIndex == selected;
      });
  int valid =
      !decision.calibrated || supported != record.endpoint.candidates.end();
  auto accepted = gather(bootstrap, valid);
  if (std::find(accepted.begin(), accepted.end(), 0) != accepted.end())
    throw std::runtime_error(
        "network plan no longer matches local capabilities");
  std::fprintf(
      stderr,
      "UCCL Lite measured network plan: rank=%d matched=%d rail=%d nic=%s "
      "port=%d "
      "basis_bytes=%zu isolated_path_us=%.9g shared_bound_us=%.9g "
      "group_matched=%d group_us=%.9g group=%s reason=%s\n",
      context.rank, decision.calibrated, selected,
      supported == record.endpoint.candidates.end()
          ? "physical-default"
          : supported->assignment.nic.c_str(),
      supported == record.endpoint.candidates.end()
          ? -1
          : supported->assignment.port,
      decision.bytes, decision.pathUs, decision.capacityUs,
      decision.groupMeasured, decision.groupUs, decision.group,
      decision.reason);
  return selected;
}
class Adapter : public CollectiveBackend {
  BackendContext context_;
  std::shared_ptr<NetworkProfile const> calibration_;
  std::unique_ptr<NetworkCollectives> executor_;
  RankMapping mapping_;
  void *input_ = nullptr, *output_ = nullptr;
  cudaEvent_t done_ = nullptr;
  cudaStream_t last_ = nullptr;
  bool recorded_ = false;
  static void check(cudaError_t status) {
    if (status != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(status));
  }

 public:
  explicit Adapter(BackendContext const& context,
                   std::shared_ptr<NetworkProfile const> calibration = {})
      : context_(context), calibration_(std::move(calibration)) {}
  ~Adapter() {
    int previous = -1;
    cudaGetDevice(&previous);
    cudaSetDevice(context_.device);
    if (recorded_) cudaEventSynchronize(done_);
    executor_.reset();
    if (input_) cudaFree(input_);
    if (output_) cudaFree(output_);
    if (done_) cudaEventDestroy(done_);
    if (previous >= 0 && previous != context_.device) cudaSetDevice(previous);
  }
  void prepare() override {
    if (executor_) return;
    if (context_.ranks != 8 || context_.localDevices.size() != 4)
      throw std::invalid_argument(
          "network hierarchy requires its registered {4,4} layout");
    auto bootstrap =
        std::make_shared<mscclpp::TcpBootstrap>(context_.rank, context_.ranks);
    bootstrap->initialize(context_.id, 120);
    mapping_ = planRanks(context_, *bootstrap);
    int rail = selectRail(context_, *bootstrap, mapping_, calibration_);
    NetworkId executionId = context_.id;
    if (mapping_.remap) {
      struct Identifier {
        int ready = 1;
        NetworkId id{};
      } local;
      try {
        check(cudaMalloc(&input_, context_.capacity));
        check(cudaMalloc(&output_, context_.capacity));
        check(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming));
        if (!context_.rank) local.id = NetworkCollectives::createUniqueId();
      } catch (...) {
        local.ready = 0;
      }
      auto votes = gather(*bootstrap, local);
      for (auto const& v : votes)
        if (!v.ready)
          throw std::runtime_error(
              "network permutation preparation failed collectively");
      executionId = votes.front().id;
      auto executionBootstrap =
          std::make_shared<mscclpp::TcpBootstrap>(mapping_.executionRank, 8);
      executionBootstrap->initialize(executionId, 120);
      bootstrap = std::move(executionBootstrap);
    }
    auto executor = std::make_unique<NetworkCollectives>(
        executionId, mapping_.executionRank, context_.device, rail,
        context_.capacity, mapping_.nodes);
    executor->prepare(bootstrap);
    executor_ = std::move(executor);
    std::ostringstream order;
    for (int rank : mapping_.order) order << rank << ',';
    std::fprintf(stderr,
                 "UCCL Lite network rank mapping: rank=%d execution_rank=%d "
                 "device=%d numa=%d,%d rail=%d remap=%d order=%s\n",
                 context_.rank, mapping_.executionRank, context_.device,
                 mapping_.nodes[0], mapping_.nodes[1], rail,
                 int(mapping_.remap), order.str().c_str());
  }
  void enqueue(NetworkOperation op, void const* input, void* output,
               size_t count, cudaStream_t stream) override {
    if (!executor_) throw std::logic_error("network backend not prepared");
    int operation = int(op), device;
    size_t scale = operation == 3 ? 4 : 32;
    if (operation < 1 || operation > 3 || !input || !output || !count ||
        count > context_.capacity / scale || (count * scale) % 128 ||
        (reinterpret_cast<uintptr_t>(input) % 4) ||
        (reinterpret_cast<uintptr_t>(output) % 4))
      throw std::invalid_argument("network extent/alignment unsupported");
    check(cudaGetDevice(&device));
    if (device != context_.device)
      throw std::invalid_argument("network current device mismatch");
    cudaStreamCaptureStatus capture;
    check(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone)
      throw std::invalid_argument("network external capture unsupported");
    if (!mapping_.remap) {
      executor_->enqueue(op, input, output, count, stream);
      return;
    }
    if (recorded_ && stream != last_)
      check(cudaStreamWaitEvent(stream, done_, 0));
    if (op == NetworkOperation::ReduceScatter) {
      permuteRankShards(input, input_, count, mapping_.order.data(), 8, true,
                        stream);
      input = input_;
    }
    executor_->enqueue(op, input,
                       op == NetworkOperation::AllGather ? output_ : output,
                       count, stream);
    if (op == NetworkOperation::AllGather)
      permuteRankShards(output_, output, count, mapping_.order.data(), 8, false,
                        stream);
    check(cudaEventRecord(done_, stream));
    recorded_ = true;
    last_ = stream;
  }
  void synchronize() override {
    if (recorded_) check(cudaEventSynchronize(done_));
    if (executor_) executor_->synchronize();
  }
  bool initialized() const override {
    return executor_ && executor_->initialized();
  }
  bool idle() const override {
    return (!recorded_ || cudaEventQuery(done_) == cudaSuccess) &&
           (!executor_ || executor_->idle());
  }
};
}  // namespace
NetworkPreflight preflightNetwork(
    BackendContext const& context, mscclpp::Bootstrap& control,
    std::shared_ptr<NetworkProfile const>& calibration, bool probeResources) {
  if (context.ranks != 8 || control.getNranks() != 8 ||
      control.getRank() != context.rank)
    throw std::invalid_argument("network preflight participant mismatch");
  auto mapping = planRanks(context, control);
  int rail = selectRail(context, control, mapping, calibration, probeResources);
  return {mapping.order, rail, calibration};
}
std::unique_ptr<CollectiveBackend> preparedNetworkBackend(
    BackendContext const& context,
    std::shared_ptr<NetworkProfile const> calibration) {
  return std::make_unique<Adapter>(context, std::move(calibration));
}
BackendRegistration networkHierarchyBackend() {
  return {{"network-hierarchy",
           "UCCL_LITE_MSCCLPP_STRICT_V1",
           {4, 4},
           ScalarType::Float32,
           true,
           true},
          [] { return NetworkCollectives::createUniqueId(); },
          [](BackendContext const& c) { return std::make_unique<Adapter>(c); }};
}
}  // namespace uccl_lite::runtime
