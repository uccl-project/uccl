#include "uccl_lite/network_calibration.hpp"
#include "uccl_lite/network_plan.hpp"
#include "uccl_lite/runtime/network_identity.hpp"
#include <linux/mempolicy.h>
#include <mscclpp/core.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <cuda_runtime.h>
#include <mpi.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

using namespace uccl_lite;
namespace {
constexpr size_t kGuard = 64;
void check(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
std::vector<std::string> strings(std::string const& local, int world) {
  if (local.size() > (1UL << 20))
    throw std::runtime_error("network metadata too large");
  int bytes = int(local.size());
  std::vector<int> counts(world), offsets(world);
  MPI_Allgather(&bytes, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
  int total = 0;
  for (int i = 0; i < world; ++i) {
    if (counts[i] < 0 || counts[i] > (1 << 20))
      throw std::runtime_error("invalid network metadata length");
    offsets[i] = total;
    total += counts[i];
  }
  std::vector<char> data(size_t(total) + 1);
  MPI_Allgatherv(local.data(), bytes, MPI_CHAR, data.data(), counts.data(),
                 offsets.data(), MPI_CHAR, MPI_COMM_WORLD);
  std::vector<std::string> result;
  for (int i = 0; i < world; ++i)
    result.emplace_back(data.data() + offsets[i], counts[i]);
  return result;
}
struct Nic {
  std::string name;
  int port, transport;
};
struct Rank {
  NetworkHostIdentity host;
  std::string gpu;
  int local = 0;
  std::vector<Nic> nics;
};
__host__ __device__ uint32_t pattern(size_t i, int source, int generation) {
  return uint32_t(i) * 2654435761u + uint32_t(source + 1) * 7919u +
         uint32_t(generation) * 104729u;
}
__global__ void initialize(uint32_t* send, uint32_t* receive, size_t extent,
                           size_t words, int rank, int generation) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < extent;
       i += size_t(gridDim.x) * blockDim.x) {
    send[i] = i >= kGuard && i - kGuard < words
                  ? pattern(i - kGuard, rank, generation)
                  : 0x5a5a5a5a;
    receive[i] = 0x5a5a5a5a;
  }
}
__global__ void verify(uint32_t const* send, uint32_t const* receive,
                       size_t extent, size_t words, int rank, int source,
                       int receiver, int generation, unsigned* errors) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < extent;
       i += size_t(gridDim.x) * blockDim.x) {
    bool payload = i >= kGuard && i - kGuard < words;
    uint32_t expectedSend =
        payload ? pattern(i - kGuard, rank, generation) : 0x5a5a5a5a;
    uint32_t expectedReceive = payload && rank == receiver
                                   ? pattern(i - kGuard, source, generation)
                                   : 0x5a5a5a5a;
    if (send[i] != expectedSend || receive[i] != expectedReceive)
      atomicExch(errors, 1);
  }
}
struct Storage {
  uint32_t *send = nullptr, *receive = nullptr;
  unsigned* errors = nullptr;
  ~Storage() {
    if (send) cudaFree(send);
    if (receive) cudaFree(receive);
    if (errors) cudaFree(errors);
  }
};
struct HostPlacement {
  cpu_set_t cpus{};
  int mode = 0;
  std::array<unsigned long, 16> nodes{};
  HostPlacement() {
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
  ~HostPlacement() {
    try {
      restore();
    } catch (...) {
    }
  }
};
}  // namespace
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, world, localRank, localRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm local;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                      &local);
  MPI_Comm_rank(local, &localRank);
  MPI_Comm_size(local, &localRanks);
  try {
    if ((argc != 2 && argc != 3) || world > 64)
      throw std::invalid_argument(
          "calibrate_network PROFILE [MAX_BYTES]; two equal-sized hosts "
          "required");
    size_t maximum = argc == 3 ? std::stoull(argv[2]) : 64UL << 20;
    if (maximum < (2UL << 20) || maximum > (1UL << 30) || maximum % 4)
      throw std::invalid_argument(
          "maximum must be an aligned 2 MiB to 1 GiB extent");
    int devices;
    check(cudaGetDeviceCount(&devices));
    if (localRanks > devices)
      throw std::invalid_argument(
          "one distinct visible GPU per local rank required");
    check(cudaSetDevice(localRank));
    int rdma, ordering, flush;
    check(cudaDeviceGetAttribute(&rdma, cudaDevAttrGPUDirectRDMASupported,
                                 localRank));
    check(cudaDeviceGetAttribute(
        &ordering, cudaDevAttrGPUDirectRDMAWritesOrdering, localRank));
    check(cudaDeviceGetAttribute(
        &flush, cudaDevAttrGPUDirectRDMAFlushWritesOptions, localRank));
    if (!rdma || (ordering < cudaGPUDirectRDMAWritesOrderingOwner &&
                  !(flush & cudaFlushGPUDirectRDMAWritesOptionHost)))
      throw std::runtime_error(
          "GPU RDMA registration/owner visibility capability unavailable");
    std::vector<int> participating(localRanks);
    std::iota(participating.begin(), participating.end(), 0);
    auto topology = inspectTopology(participating);
    HostPlacement placement;
    Rank here;
    here.local = localRank;
    char hostname[256]{};
    if (gethostname(hostname, sizeof(hostname) - 1))
      throw std::runtime_error("hostname unavailable");
    here.host = {hostname, topology.hardwareIdentity,
                 runtime::networkTransportIdentity()};
    cudaDeviceProp property{};
    check(cudaGetDeviceProperties(&property, localRank));
    char uuid[33]{};
    for (int i = 0; i < 16; ++i)
      std::snprintf(uuid + 2 * i, 3, "%02x",
                    static_cast<unsigned char>(property.uuid.bytes[i]));
    here.gpu = uuid;
    auto configuration = mscclpp::env()->hcaDevices;
    int count =
        configuration.empty()
            ? mscclpp::getIBDeviceCount()
            : std::count(configuration.begin(), configuration.end(), ',') + 1;
    for (int rail = 0; rail < std::min(count, 8); ++rail) {
      auto transport =
          static_cast<mscclpp::Transport>(int(mscclpp::Transport::IB0) + rail);
      auto name = mscclpp::getIBDeviceName(transport);
      std::vector<NetworkPort> ports;
      for (auto const& port : topology.networkPorts)
        if (port.name == name && port.active) ports.push_back(port);
      if (ports.size() == 1) here.nics.push_back({name, ports[0].port, rail});
    }
    std::sort(here.nics.begin(), here.nics.end(),
              [](auto const& a, auto const& b) { return a.name < b.name; });
    if (here.nics.empty())
      throw std::runtime_error("no unambiguous active HCA port");
    std::ostringstream encoded;
    encoded << std::quoted(here.host.hostname) << ' '
            << std::quoted(here.host.topologyIdentity) << ' '
            << std::quoted(here.host.transportIdentity) << ' ' << here.gpu
            << ' ' << localRank << ' ' << here.nics.size();
    for (auto const& nic : here.nics)
      encoded << ' ' << std::quoted(nic.name) << ' ' << nic.port << ' '
              << nic.transport;
    std::vector<Rank> ranks(world);
    auto metadata = strings(encoded.str(), world);
    std::vector<std::string> hosts;
    std::vector<std::vector<int>> groups;
    for (int r = 0; r < world; ++r) {
      auto& entry = ranks[r];
      std::istringstream input(metadata[r]);
      size_t nics;
      if (!(input >> std::quoted(entry.host.hostname) >>
            std::quoted(entry.host.topologyIdentity) >>
            std::quoted(entry.host.transportIdentity) >> entry.gpu >>
            entry.local >> nics) ||
          nics > 8)
        throw std::runtime_error("invalid rank network metadata");
      entry.nics.resize(nics);
      for (auto& nic : entry.nics)
        if (!(input >> std::quoted(nic.name) >> nic.port >> nic.transport))
          throw std::runtime_error("invalid NIC metadata");
      auto it = std::find(hosts.begin(), hosts.end(), entry.host.hostname);
      if (it == hosts.end()) {
        hosts.push_back(entry.host.hostname);
        groups.emplace_back();
      }
      size_t h = std::find(hosts.begin(), hosts.end(), entry.host.hostname) -
                 hosts.begin();
      if (entry.local != int(groups[h].size()))
        throw std::runtime_error("noncanonical local rank metadata");
      groups[h].push_back(r);
    }
    if (groups.size() != 2 || groups[0].size() != groups[1].size() ||
        groups[0].size() != size_t(localRanks))
      throw std::runtime_error("two equal-sized real hosts required");
    NetworkProfile profile;
    profile.probeIdentity =
        runtime::artifactIdentity(reinterpret_cast<uintptr_t>(&pattern));
    for (int h = 0; h < 2; ++h) {
      auto const& first = ranks[groups[h][0]];
      profile.hosts[h] = first.host;
      for (int r : groups[h]) {
        auto const& current = ranks[r];
        if (current.host.topologyIdentity != first.host.topologyIdentity ||
            current.host.transportIdentity != first.host.transportIdentity ||
            current.nics.size() != first.nics.size())
          throw std::runtime_error(
              "host topology/affinity/transport disagreement; use identical "
              "unbound ranks");
        for (size_t i = 0; i < first.nics.size(); ++i)
          if (current.nics[i].name != first.nics[i].name ||
              current.nics[i].port != first.nics[i].port)
            throw std::runtime_error("inconsistent physical HCA inventory");
      }
    }
    Storage storage;
    size_t extent = maximum / 4 + 2 * kGuard;
    check(cudaMalloc(&storage.send, extent * 4));
    check(cudaMalloc(&storage.receive, extent * 4));
    check(cudaMalloc(&storage.errors, sizeof(unsigned)));
    {
      auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, world);
      mscclpp::UniqueId id{};
      if (!rank) id = mscclpp::TcpBootstrap::createUniqueId();
      MPI_Bcast(id.data(), id.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
      bootstrap->initialize(id, 120);
      mscclpp::Communicator comm(bootstrap);
      for (int pair = 0; pair < localRanks; ++pair)
        for (size_t ni = 0; ni < ranks[groups[0][pair]].nics.size(); ++ni)
          for (size_t nj = 0; nj < ranks[groups[1][pair]].nics.size(); ++nj) {
            int endpoints[2] = {groups[0][pair], groups[1][pair]};
            int side = rank == endpoints[0] ? 0 : rank == endpoints[1] ? 1 : -1;
            int remote = side >= 0 ? endpoints[1 - side] : -1;
            mscclpp::RegisteredMemory send, receive, remoteReceive;
            mscclpp::Connection connection;
            int valid = 1;
            std::string failure;
            if (side >= 0) {
              try {
                auto nic = here.nics[side ? nj : ni];
                auto transport = static_cast<mscclpp::Transport>(
                    int(mscclpp::Transport::IB0) + nic.transport);
                send = comm.registerMemory(storage.send, extent * 4, transport);
                receive =
                    comm.registerMemory(storage.receive, extent * 4, transport);
              } catch (std::exception const& error) {
                valid = 0;
                failure = error.what();
              }
            }
            int admitted;
            MPI_Allreduce(&valid, &admitted, 1, MPI_INT, MPI_MIN,
                          MPI_COMM_WORLD);
            auto failures = strings(failure, world);
            std::string rejection;
            if (!admitted)
              for (int r = 0; r < world; ++r)
                if (!failures[r].empty())
                  rejection +=
                      "rank " + std::to_string(r) + ": " + failures[r] + "; ";
            if (admitted && side >= 0) {
              auto nic = here.nics[side ? nj : ni];
              auto transport = static_cast<mscclpp::Transport>(
                  int(mscclpp::Transport::IB0) + nic.transport);
              auto future = comm.connect(transport, remote);
              comm.sendMemory(receive, remote);
              auto incoming = comm.recvMemory(remote);
              connection = future.get();
              remoteReceive = incoming.get();
            }
            placement.restore();
            MPI_Barrier(MPI_COMM_WORLD);
            for (int sourceSide = 0; sourceSide < 2; ++sourceSide) {
              int source = endpoints[sourceSide],
                  receiver = endpoints[1 - sourceSide];
              auto const& sourceNic = ranks[source].nics[sourceSide ? nj : ni];
              auto const& targetNic =
                  ranks[receiver].nics[sourceSide ? ni : nj];
              NetworkPathObservation observation;
              observation.key = {sourceSide,        1 - sourceSide,
                                 ranks[source].gpu, ranks[receiver].gpu,
                                 sourceNic.name,    targetNic.name,
                                 sourceNic.port,    targetNic.port};
              observation.rejection = rejection;
              if (admitted)
                for (size_t bytes : {size_t(16), size_t(1) << 20, maximum}) {
                  NetworkObservation samples;
                  samples.bytes = bytes;
                  for (int repetition = 0; repetition < observation.repetitions;
                       ++repetition)
                    for (int iteration = 0;
                         iteration <
                         observation.warmups + observation.iterations;
                         ++iteration) {
                      int generation =
                          1 + repetition * 11 + iteration + sourceSide * 71;
                      if (side >= 0) {
                        initialize<<<128, 256>>>(storage.send, storage.receive,
                                                 extent, bytes / 4, rank,
                                                 generation);
                        check(cudaGetLastError());
                        check(cudaDeviceSynchronize());
                      }
                      MPI_Barrier(MPI_COMM_WORLD);
                      double us = 0;
                      if (rank == source) {
                        auto start = std::chrono::steady_clock::now();
                        connection.write(remoteReceive, kGuard * 4, send,
                                         kGuard * 4, bytes);
                        connection.flush(30000000);
                        us = std::chrono::duration<double, std::micro>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();
                      }
                      MPI_Barrier(MPI_COMM_WORLD);
                      if (rank == receiver &&
                          ordering < cudaGPUDirectRDMAWritesOrderingOwner)
                        check(cudaDeviceFlushGPUDirectRDMAWrites(
                            cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                            cudaFlushGPUDirectRDMAWritesToOwner));
                      unsigned wrong = 0, anyWrong;
                      if (side >= 0) {
                        check(cudaMemset(storage.errors, 0, sizeof(unsigned)));
                        verify<<<128, 256>>>(
                            storage.send, storage.receive, extent, bytes / 4,
                            rank, source, receiver, generation, storage.errors);
                        check(cudaGetLastError());
                        check(cudaMemcpy(&wrong, storage.errors, sizeof(wrong),
                                         cudaMemcpyDeviceToHost));
                      }
                      MPI_Allreduce(&wrong, &anyWrong, 1, MPI_UNSIGNED, MPI_MAX,
                                    MPI_COMM_WORLD);
                      if (anyWrong)
                        throw std::runtime_error(
                            "network full-buffer/input/guard verification "
                            "failed");
                      MPI_Bcast(&us, 1, MPI_DOUBLE, source, MPI_COMM_WORLD);
                      if (iteration >= observation.warmups)
                        samples.microseconds.push_back(us);
                    }
                  observation.sizes.push_back(std::move(samples));
                }
              profile.paths.push_back(std::move(observation));
              if (!rank) {
                auto const& result = profile.paths.back();
                auto estimate =
                    admitted ? estimateNetworkPath(profile, profile.hosts,
                                                   result.key, maximum)
                             : std::nullopt;
                std::cout << "source=" << source << ':' << sourceNic.name
                          << " destination=" << receiver << ':'
                          << targetNic.name;
                if (estimate)
                  std::cout << " bytes=" << maximum
                            << " payload_GBs=" << estimate->payloadGBs
                            << " sender_us=" << estimate->microseconds;
                else
                  std::cout << " rejected=" << result.rejection;
                std::cout << std::endl;
              }
            }
          }
      // Concurrent cohorts: topology-derived placement, rotations of that
      // placement, and uniform HCA choices on each host. These are probe
      // candidates; they do not enable a user workload's routing backend.
      int host = here.host.hostname == hosts[0] ? 0 : 1;
      placement.restore();
      char mpiVersion[MPI_MAX_LIBRARY_VERSION_STRING]{};
      int mpiLength = 0;
      MPI_Get_library_version(mpiVersion, &mpiLength);
      std::ostringstream controlIdentity;
      controlIdentity << runtime::artifactIdentity(
                             reinterpret_cast<uintptr_t>(&MPI_Barrier))
                      << ';' << std::string(mpiVersion, mpiLength);
      for (auto name :
           {"OMPI_MCA_pml", "OMPI_MCA_btl", "OMPI_MCA_btl_tcp_if_include",
            "OMPI_MCA_oob_tcp_if_include"})
        if (auto value = std::getenv(name))
          controlIdentity << ';' << name << '=' << value;
      auto controls = strings(controlIdentity.str(), world);
      for (int h = 0; h < 2; ++h) {
        profile.groupControlIdentities[h] = controls[groups[h][0]];
        for (int r : groups[h])
          if (controls[r] != profile.groupControlIdentities[h])
            throw std::runtime_error("inconsistent cohort control identity");
      }
      int physical = 0, modeled = 1;
      try {
        std::vector<RailDevice> available;
        for (auto const& nic : here.nics)
          available.push_back({nic.name, nic.transport});
        auto assignment = planNetworkRails(topology, available)[localRank];
        physical = std::find_if(here.nics.begin(), here.nics.end(),
                                [&](auto const& nic) {
                                  return nic.name == assignment.nic;
                                }) -
                   here.nics.begin();
        if (physical >= int(here.nics.size()))
          throw std::runtime_error("probe rail missing");
      } catch (std::exception const&) {
        physical = 0;
        modeled = 0;
      }
      int allModeled;
      MPI_Allreduce(&modeled, &allModeled, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      std::vector<int> base(world);
      MPI_Allgather(&physical, 1, MPI_INT, base.data(), 1, MPI_INT,
                    MPI_COMM_WORLD);
      std::vector<std::vector<int>> configurations;
      std::set<std::vector<int>> unique;
      auto add = [&](std::vector<int> configuration, std::string label) {
        if (!unique.insert(configuration).second) return;
        NetworkGroupObservation group;
        group.label = std::move(label);
        for (int r = 0; r < world; ++r) {
          int side = ranks[r].host.hostname == hosts[0] ? 0 : 1;
          int peer = groups[1 - side][ranks[r].local];
          auto const& source = ranks[r].nics[configuration[r]];
          auto const& destination = ranks[peer].nics[configuration[peer]];
          group.flows.push_back({side, 1 - side, ranks[r].gpu, ranks[peer].gpu,
                                 source.name, destination.name, source.port,
                                 destination.port});
        }
        group.sizes = {{size_t(1) << 20, {}}, {maximum, {}}};
        configurations.push_back(std::move(configuration));
        profile.groups.push_back(std::move(group));
      };
      size_t n0 = ranks[groups[0][0]].nics.size(),
             n1 = ranks[groups[1][0]].nics.size();
      for (size_t a = 0; a < n0; ++a)
        for (size_t b = 0; b < n1; ++b) {
          auto placement = base;
          for (int r = 0; r < world; ++r) {
            bool side = ranks[r].host.hostname != hosts[0];
            placement[r] = (base[r] + (side ? b : a)) % ranks[r].nics.size();
          }
          add(std::move(placement),
              std::string(allModeled ? "topology-rotation-"
                                     : "active-hca-rotation-") +
                  std::to_string(a) + '-' + std::to_string(b));
        }
      for (size_t a = 0; a < n0; ++a)
        for (size_t b = 0; b < n1; ++b) {
          std::vector<int> placement;
          for (auto const& entry : ranks)
            placement.push_back(entry.host.hostname == hosts[0] ? a : b);
          add(std::move(placement),
              "uniform-" + std::to_string(a) + '-' + std::to_string(b));
        }
      // Rotate candidate order between repetitions. Setup/teardown never enter
      // sender timings. Failed setup retains any earlier samples as negative
      // evidence.
      for (int repetition = 0; repetition < 3; ++repetition)
        for (size_t position = 0; position < configurations.size();
             ++position) {
          size_t selected = (position + repetition) % configurations.size();
          auto& observation = profile.groups[selected];
          if (!observation.rejection.empty()) continue;
          auto nic = here.nics[configurations[selected][rank]];
          auto transport = static_cast<mscclpp::Transport>(
              int(mscclpp::Transport::IB0) + nic.transport);
          int peer = groups[1 - host][localRank];
          mscclpp::RegisteredMemory send, receive, remoteReceive;
          mscclpp::Connection connection;
          std::string failure;
          int valid = 1, admitted;
          try {
            send = comm.registerMemory(storage.send, extent * 4, transport);
            receive =
                comm.registerMemory(storage.receive, extent * 4, transport);
          } catch (std::exception const& error) {
            valid = 0;
            failure = error.what();
          }
          MPI_Allreduce(&valid, &admitted, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
          auto failures = strings(failure, world);
          if (!admitted) {
            for (int r = 0; r < world; ++r)
              if (!failures[r].empty())
                observation.rejection +=
                    "rank " + std::to_string(r) + ": " + failures[r] + "; ";
            placement.restore();
            continue;
          }
          auto future = comm.connect(transport, peer);
          comm.sendMemory(receive, peer);
          auto incoming = comm.recvMemory(peer);
          connection = future.get();
          remoteReceive = incoming.get();
          placement.restore();
          MPI_Barrier(MPI_COMM_WORLD);
          for (auto& size : observation.sizes) {
            for (int iteration = 0;
                 iteration < observation.warmups + observation.iterations;
                 ++iteration) {
              int generation =
                  101 + repetition * 11 + iteration + int(selected) * 71;
              initialize<<<128, 256>>>(storage.send, storage.receive, extent,
                                       size.bytes / 4, rank, generation);
              check(cudaGetLastError());
              check(cudaDeviceSynchronize());
              MPI_Barrier(MPI_COMM_WORLD);
              auto cohortBegin = std::chrono::steady_clock::now();
              MPI_Barrier(MPI_COMM_WORLD);
              auto begin = std::chrono::steady_clock::now();
              connection.write(remoteReceive, kGuard * 4, send, kGuard * 4,
                               size.bytes);
              connection.flush(30000000);
              double us = std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - begin)
                              .count();
              MPI_Barrier(MPI_COMM_WORLD);
              double cohortUs =
                  std::chrono::duration<double, std::micro>(
                      std::chrono::steady_clock::now() - cohortBegin)
                      .count();
              if (ordering < cudaGPUDirectRDMAWritesOrderingOwner)
                check(cudaDeviceFlushGPUDirectRDMAWrites(
                    cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                    cudaFlushGPUDirectRDMAWritesToOwner));
              check(cudaMemset(storage.errors, 0, sizeof(unsigned)));
              verify<<<128, 256>>>(storage.send, storage.receive, extent,
                                   size.bytes / 4, rank, peer, rank, generation,
                                   storage.errors);
              check(cudaGetLastError());
              unsigned wrong, anyWrong;
              check(cudaMemcpy(&wrong, storage.errors, sizeof(wrong),
                               cudaMemcpyDeviceToHost));
              MPI_Allreduce(&wrong, &anyWrong, 1, MPI_UNSIGNED, MPI_MAX,
                            MPI_COMM_WORLD);
              if (anyWrong)
                throw std::runtime_error(
                    "concurrent network full-buffer/input/guard verification "
                    "failed");
              std::vector<double> durations(world);
              std::vector<double> cohorts(world);
              MPI_Allgather(&us, 1, MPI_DOUBLE, durations.data(), 1, MPI_DOUBLE,
                            MPI_COMM_WORLD);
              MPI_Allgather(&cohortUs, 1, MPI_DOUBLE, cohorts.data(), 1,
                            MPI_DOUBLE, MPI_COMM_WORLD);
              if (iteration >= observation.warmups) {
                size.senderMicroseconds.push_back(std::move(durations));
                size.cohortMicroseconds.push_back(std::move(cohorts));
              }
            }
          }
          if (!rank)
            std::cout << "concurrent group=" << observation.label
                      << " repetition=" << repetition << " verified\n"
                      << std::flush;
        }
    }
    if (!rank) {
      saveNetworkProfile(argv[1], profile);
      auto loaded = loadNetworkProfile(argv[1]);
      if (loaded.paths.size() != profile.paths.size())
        throw std::runtime_error("network profile round-trip failed");
      if (loaded.groups.size() != profile.groups.size())
        throw std::runtime_error("network group profile round-trip failed");
      for (auto const& group : profile.groups) {
        auto estimate =
            estimateNetworkGroup(profile, profile.hosts, group.flows, maximum);
        std::cout << "group=" << group.label << " flows=" << group.flows.size();
        if (estimate)
          std::cout << " bytes_per_flow=" << maximum
                    << " max_cohort_us=" << estimate->microseconds
                    << " aggregate_payload_GBs=" << estimate->payloadGBs;
        else
          std::cout << " rejected=" << group.rejection;
        std::cout << '\n';
      }
      size_t measured = std::count_if(
          profile.paths.begin(), profile.paths.end(),
          [](auto const& path) { return path.rejection.empty(); });
      std::cout << "PASS: network profile entries measured=" << measured
                << " rejected=" << profile.paths.size() - measured
                << " concurrent_groups=" << profile.groups.size()
                << "; uint32 payload, sender "
                   "post+CQ flush; 2 warmups, 5 calls x 3 repetitions per "
                   "size; receiver visibility/input/guard checks outside "
                   "timing; no baseline or best-variant envelope\n";
    }
  } catch (std::exception const& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  MPI_Comm_free(&local);
  MPI_Finalize();
}
