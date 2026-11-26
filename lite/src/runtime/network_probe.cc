#include "network_probe.hpp"
#include "uccl_lite/runtime/network_identity.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <set>
#include <sched.h>

namespace uccl_lite::runtime {
namespace {
constexpr size_t guard = 64;
void checked(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
struct Control {
  mscclpp::Bootstrap& bootstrap;
  int rank, ranks;
  explicit Control(mscclpp::Bootstrap& b)
      : bootstrap(b), rank(b.getRank()), ranks(b.getNranks()) {}
  template <class T>
  std::vector<T> gather(T const& value) {
    std::vector<T> result(ranks);
    result[rank] = value;
    bootstrap.allGather(result.data(), sizeof(T));
    return result;
  }
  std::string errors(std::string const& local) {
    auto sizes = gather(local.size());
    size_t width =
        std::max(size_t(1), *std::max_element(sizes.begin(), sizes.end()));
    if (width > (1 << 20))
      throw std::runtime_error("probe error extent invalid");
    std::vector<char> bytes(width * ranks);
    std::memcpy(bytes.data() + rank * width, local.data(), local.size());
    bootstrap.allGather(bytes.data(), int(width));
    std::string result;
    for (int r = 0; r < ranks; ++r)
      if (sizes[r])
        result += "rank " + std::to_string(r) + ": " +
                  std::string(bytes.data() + r * width, sizes[r]) + "; ";
    return result;
  }
  template <class F>
  std::string attempt(F f) {
    std::string error;
    try {
      f();
    } catch (std::exception const& e) {
      error = e.what();
    } catch (...) {
      error = "unknown network probe error";
    }
    return errors(error);
  }
  template <class F>
  void run(F f) {
    auto error = attempt(f);
    if (!error.empty()) throw NetworkCalibrationFailure(error);
  }
  void barrier() { bootstrap.barrier(); }
};
struct Placement {
  cpu_set_t cpus{};
  int mode = 0;
  std::array<unsigned long, 16> nodes{};
  Placement() {
    if (sched_getaffinity(0, sizeof(cpus), &cpus) ||
        syscall(SYS_get_mempolicy, &mode, nodes.data(), sizeof(nodes) * 8,
                nullptr, 0))
      throw std::runtime_error("cannot capture probe CPU/NUMA placement");
  }
  void restore() const {
    int base = mode & ~MPOL_MODE_FLAGS;
    if (syscall(
            SYS_set_mempolicy, mode,
            base == MPOL_DEFAULT || base == MPOL_LOCAL ? nullptr : nodes.data(),
            sizeof(nodes) * 8) ||
        sched_setaffinity(0, sizeof(cpus), &cpus))
      throw std::runtime_error("cannot restore probe CPU/NUMA placement");
  }
  ~Placement() {
    try {
      restore();
    } catch (...) {
    }
  }
};
struct Storage {
  uint32_t *send = nullptr, *receive = nullptr;
  std::vector<uint32_t> input, actual;
  ~Storage() {
    if (send) cudaFree(send);
    if (receive) cudaFree(receive);
  }
  void allocate(size_t maximum) {
    size_t words = maximum / 4 + 2 * guard;
    input.resize(words);
    actual.resize(words);
    checked(cudaMalloc(reinterpret_cast<void**>(&send), words * 4));
    checked(cudaMalloc(reinterpret_cast<void**>(&receive), words * 4));
  }
  static uint32_t value(size_t i, int source, int generation) {
    return uint32_t(i) * 2654435761u + uint32_t(source + 1) * 7919u +
           uint32_t(generation) * 104729u;
  }
  void initialize(size_t bytes, int rank, int generation) {
    std::fill(input.begin(), input.end(), 0x5a5a5a5a);
    checked(cudaMemcpy(receive, input.data(), input.size() * 4,
                       cudaMemcpyHostToDevice));
    for (size_t i = 0; i < bytes / 4; ++i)
      input[guard + i] = value(i, rank, generation);
    checked(cudaMemcpy(send, input.data(), input.size() * 4,
                       cudaMemcpyHostToDevice));
  }
  void verify(size_t bytes, int source, int generation, bool receiver) {
    checked(cudaMemcpy(actual.data(), send, actual.size() * 4,
                       cudaMemcpyDeviceToHost));
    if (actual != input)
      throw std::runtime_error("network probe input/guard corruption");
    checked(cudaMemcpy(actual.data(), receive, actual.size() * 4,
                       cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < actual.size(); ++i) {
      bool payload = receiver && i >= guard && i - guard < bytes / 4;
      uint32_t expected =
          payload ? value(i - guard, source, generation) : 0x5a5a5a5a;
      if (actual[i] != expected)
        throw std::runtime_error("network probe full output/guard mismatch");
    }
  }
};
struct Link {
  mscclpp::RegisteredMemory send, receive, remote;
  mscclpp::Connection connection;
};
double elapsed(std::chrono::steady_clock::time_point begin) {
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now() - begin)
      .count();
}
}  // namespace
NetworkProfile probeNetworkResources(
    mscclpp::Bootstrap& bootstrap,
    std::vector<NetworkEndpoint> const& endpoints,
    std::vector<std::pair<size_t, size_t>> const& pairs,
    std::vector<int> const& physicalRails, size_t maximum) {
  Control control(bootstrap);
  NetworkProfile profile;
  std::vector<int> host, peer(control.ranks, -1), base;
  int ordering = 0;
  std::unique_ptr<Placement> placement;
  Storage storage;
  control.run([&] {
    if (endpoints.size() != size_t(control.ranks) ||
        physicalRails.size() != endpoints.size() ||
        pairs.size() * 2 != endpoints.size() || maximum < (2UL << 20) ||
        maximum > (64UL << 20) || maximum % 4)
      throw std::invalid_argument("network probe metadata/extent invalid");
    for (auto [a, b] : pairs) {
      if (a >= endpoints.size() || b >= endpoints.size() || a == b ||
          peer[a] >= 0 || peer[b] >= 0)
        throw std::invalid_argument("network probe pair layout invalid");
      peer[a] = int(b);
      peer[b] = int(a);
    }
    profile.hosts[0] = endpoints.front().host;
    for (auto const& e : endpoints)
      if (e.host.hostname != profile.hosts[0].hostname) {
        profile.hosts[1] = e.host;
        break;
      }
    for (size_t r = 0; r < endpoints.size(); ++r) {
      auto const& e = endpoints[r];
      int h = e.host.hostname == profile.hosts[0].hostname   ? 0
              : e.host.hostname == profile.hosts[1].hostname ? 1
                                                             : -1;
      if (h < 0 ||
          e.host.topologyIdentity != profile.hosts[h].topologyIdentity ||
          e.host.transportIdentity != profile.hosts[h].transportIdentity ||
          e.candidates.empty() || e.candidates.size() > 8 ||
          e.host.hostname == endpoints[peer[r]].host.hostname)
        throw std::invalid_argument("network probe endpoint identity invalid");
      host.push_back(h);
      int choice = 0;
      for (size_t c = 0; c < e.candidates.size(); ++c)
        if (e.candidates[c].assignment.transportIndex == physicalRails[r])
          choice = int(c);
      base.push_back(choice);
    }
    int device, rdma, flush;
    checked(cudaGetDevice(&device));
    if (device != endpoints[control.rank].device)
      throw std::invalid_argument("probe device mismatch");
    checked(cudaDeviceGetAttribute(&rdma, cudaDevAttrGPUDirectRDMASupported,
                                   device));
    checked(cudaDeviceGetAttribute(
        &ordering, cudaDevAttrGPUDirectRDMAWritesOrdering, device));
    checked(cudaDeviceGetAttribute(
        &flush, cudaDevAttrGPUDirectRDMAFlushWritesOptions, device));
    if (!rdma || (ordering < cudaGPUDirectRDMAWritesOrderingOwner &&
                  !(flush & cudaFlushGPUDirectRDMAWritesOptionHost)))
      throw std::invalid_argument(
          "GPU RDMA registration/owner visibility unavailable");
    placement = std::make_unique<Placement>();
    storage.allocate(maximum);
    profile.probeIdentity =
        artifactIdentity(reinterpret_cast<uintptr_t>(&probeNetworkResources));
    profile.groupScope = kNetworkBootstrapGroupProbeScope;
    auto configuration = mscclpp::env();
    std::string identity =
        "mscclpp-tcp-bootstrap;probe=" + profile.probeIdentity +
        ";transport=" + endpoints[control.rank].host.transportIdentity;
    if (auto iface = std::getenv("MSCCLPP_SOCKET_IFNAME"))
      identity += ";socket-ifname=" + std::string(iface);
    profile.groupControlIdentities[host[control.rank]] = identity;
  });
  // Retain every participating rank's control provenance, grouped by host.
  for (int h = 0; h < 2; ++h) {
    auto all = control.errors(
        host[control.rank] == h ? profile.groupControlIdentities[h] : "");
    profile.groupControlIdentities[h] = all;
  }
  auto flow = [&](int source, int a, int b) {
    int destination = peer[source];
    auto const& x = endpoints[source].candidates[a].assignment;
    auto const& y = endpoints[destination].candidates[b].assignment;
    return NetworkPathKey{host[source],
                          host[destination],
                          endpoints[source].gpu,
                          endpoints[destination].gpu,
                          x.nic,
                          y.nic,
                          x.port,
                          y.port};
  };
  {
    auto borrowed =
        std::shared_ptr<mscclpp::Bootstrap>(&bootstrap, [](auto*) {});
    mscclpp::Communicator comm(borrowed);
    auto connect = [&](int other, int candidate, Link& link) {
      bool active = other >= 0;
      mscclpp::Transport transport = mscclpp::Transport::IB0;
      auto failure = control.attempt([&] {
        if (!active) return;
        transport = static_cast<mscclpp::Transport>(
            int(mscclpp::Transport::IB0) + endpoints[control.rank]
                                               .candidates[candidate]
                                               .assignment.transportIndex);
        link.send = comm.registerMemory(storage.send, storage.input.size() * 4,
                                        transport);
        link.receive = comm.registerMemory(storage.receive,
                                           storage.input.size() * 4, transport);
      });
      if (failure.empty())
        control.run([&] {
          if (!active) return;
          auto connection = comm.connect(transport, other);
          comm.sendMemory(link.receive, other);
          auto incoming = comm.recvMemory(other);
          link.connection = connection.get();
          link.remote = incoming.get();
        });
      control.run([&] { placement->restore(); });
      control.barrier();
      return failure;
    };
    auto sample = [&](Link& link, bool active, bool sender, bool receiver,
                      int source, size_t bytes, int generation, bool cohort) {
      control.run([&] {
        if (active) storage.initialize(bytes, control.rank, generation);
      });
      control.barrier();
      auto cohortBegin = std::chrono::steady_clock::now();
      if (cohort) control.barrier();
      double us = 0;
      std::string error;
      try {
        if (sender) {
          auto start = std::chrono::steady_clock::now();
          link.connection.write(link.remote, guard * 4, link.send, guard * 4,
                                bytes);
          link.connection.flush(30000000);
          us = elapsed(start);
        }
      } catch (std::exception const& e) {
        error = e.what();
      }
      control.barrier();
      double cohortUs = elapsed(cohortBegin);
      auto failure = control.errors(error);
      if (!failure.empty()) throw NetworkCalibrationFailure(failure);
      control.run([&] {
        if (!active) return;
        if (receiver && ordering < cudaGPUDirectRDMAWritesOrderingOwner)
          checked(cudaDeviceFlushGPUDirectRDMAWrites(
              cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
              cudaFlushGPUDirectRDMAWritesToOwner));
        storage.verify(bytes, source, generation, receiver);
      });
      return std::make_pair(control.gather(us), control.gather(cohortUs));
    };
    for (auto [left, right] : pairs)
      for (size_t a = 0; a < endpoints[left].candidates.size(); ++a)
        for (size_t b = 0; b < endpoints[right].candidates.size(); ++b) {
          bool active = control.rank == int(left) || control.rank == int(right);
          Link link;
          auto rejection = connect(active ? peer[control.rank] : -1,
                                   control.rank == int(left) ? a : b, link);
          for (int direction = 0; direction < 2; ++direction) {
            int source = direction ? right : left,
                receiver = direction ? left : right;
            NetworkPathObservation observation;
            observation.key =
                flow(source, direction ? b : a, direction ? a : b);
            observation.rejection = rejection;
            if (rejection.empty())
              for (size_t bytes : {size_t(16), size_t(1) << 20, maximum}) {
                NetworkObservation size;
                size.bytes = bytes;
                for (int repetition = 0; repetition < observation.repetitions;
                     ++repetition)
                  for (int iteration = 0;
                       iteration < observation.warmups + observation.iterations;
                       ++iteration) {
                    auto result =
                        sample(link, active, control.rank == source,
                               control.rank == receiver, source, bytes,
                               1 + repetition * 11 + iteration + direction * 71,
                               false);
                    if (iteration >= observation.warmups)
                      size.microseconds.push_back(result.first[source]);
                  }
                observation.sizes.push_back(std::move(size));
              }
            profile.paths.push_back(std::move(observation));
          }
        }
    std::vector<std::vector<int>> configurations;
    std::set<std::vector<int>> unique;
    auto add = [&](std::vector<int> config, std::string label) {
      if (!unique.insert(config).second) return;
      NetworkGroupObservation group;
      group.label = std::move(label);
      for (int r = 0; r < control.ranks; ++r)
        group.flows.push_back(flow(r, config[r], config[peer[r]]));
      group.sizes = {{size_t(1) << 20, {}, {}}, {maximum, {}, {}}};
      configurations.push_back(std::move(config));
      profile.groups.push_back(std::move(group));
    };
    std::array<size_t, 2> counts{};
    for (int r = 0; r < control.ranks; ++r)
      counts[host[r]] =
          std::max(counts[host[r]], endpoints[r].candidates.size());
    for (size_t a = 0; a < counts[0]; ++a)
      for (size_t b = 0; b < counts[1]; ++b)
        for (int rotate = 0; rotate < 2; ++rotate) {
          auto config = base;
          for (int r = 0; r < control.ranks; ++r)
            config[r] = ((rotate ? base[r] : 0) + (host[r] ? b : a)) %
                        endpoints[r].candidates.size();
          add(std::move(config), std::string(rotate ? "rotation-" : "index-") +
                                     std::to_string(a) + '-' +
                                     std::to_string(b));
        }
    for (int repetition = 0; repetition < 3; ++repetition)
      for (size_t position = 0; position < configurations.size(); ++position) {
        size_t selected = (position + repetition) % configurations.size();
        auto& observation = profile.groups[selected];
        if (!observation.rejection.empty()) continue;
        Link link;
        observation.rejection = connect(
            peer[control.rank], configurations[selected][control.rank], link);
        if (!observation.rejection.empty()) continue;
        for (auto& size : observation.sizes)
          for (int iteration = 0;
               iteration < observation.warmups + observation.iterations;
               ++iteration) {
            auto result = sample(
                link, true, true, true, peer[control.rank], size.bytes,
                101 + repetition * 11 + iteration + int(selected) * 71, true);
            if (iteration >= observation.warmups) {
              size.senderMicroseconds.push_back(std::move(result.first));
              size.cohortMicroseconds.push_back(std::move(result.second));
            }
          }
      }
  }
  control.run([&] {
    placement->restore();
    if (networkTransportIdentity() !=
        endpoints[control.rank].host.transportIdentity)
      throw std::runtime_error("probe changed transport/placement identity");
  });
  return profile;
}
}  // namespace uccl_lite::runtime
