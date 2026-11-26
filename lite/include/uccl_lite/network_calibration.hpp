#pragma once
#include "topology.hpp"
#include <array>
#include <optional>
#include <tuple>

namespace uccl_lite {
struct NetworkHostIdentity {
  std::string hostname, topologyIdentity, transportIdentity;
};
struct NetworkPathKey {
  int sourceHost = 0, destinationHost = 1;
  std::string sourceGpu, destinationGpu, sourceNic, destinationNic;
  int sourcePort = 0, destinationPort = 0;
  auto tuple() const {
    return std::tie(sourceHost, destinationHost, sourceGpu, destinationGpu,
                    sourceNic, destinationNic, sourcePort, destinationPort);
  }
  bool operator<(NetworkPathKey const& other) const {
    return tuple() < other.tuple();
  }
  bool operator==(NetworkPathKey const& other) const {
    return tuple() == other.tuple();
  }
};
struct NetworkObservation {
  size_t bytes = 0;
  // Every timed call in repetition order, not best-size or best-repetition
  // data.
  std::vector<double> microseconds;
};
struct NetworkPathObservation {
  NetworkPathKey key;
  int warmups = 2, iterations = 5, repetitions = 3;
  // Unsupported registration/transport paths remain explicit negative evidence.
  std::string rejection;
  std::vector<NetworkObservation> sizes;
};
// CPU-posted MSCCL++ GPU-memory RDMA write + sender CQ flush. One active flow
// globally. Receiver owner-visibility flush and full verification are outside
// timing. These measurements are not pure NIC/PCIe capacity or GPU-ready
// latency.
inline constexpr char kLegacyNetworkProbeScope[] =
    "isolated-gpu-write-host-post-cq-flush-v1";
inline constexpr char kNetworkProbeScope[] =
    "isolated-gpu-write-host-post-cq-flush-affinity-restored-v2";
inline constexpr char kNetworkGroupProbeScope[] =
    "concurrent-gpu-writes-max-cohort-with-mpi-barriers-v1";
inline constexpr char kNetworkBootstrapGroupProbeScope[] =
    "concurrent-gpu-writes-max-cohort-with-mscclpp-bootstrap-barriers-v1";
struct NetworkGroupObservation {
  std::string label;
  // Canonical identity is the complete directed flow set, not this label.
  std::vector<NetworkPathKey> flows;
  int warmups = 2, iterations = 5, repetitions = 3;
  std::string rejection;
  struct Size {
    size_t bytes = 0;  // Payload per directed flow.
    // [timed call][flow index]: retain every sender duration. The group
    // Both individual posting/CQ duration and whole-cohort duration are kept.
    std::vector<std::vector<double>> senderMicroseconds;
    // Includes timed start/end barriers of the profile's declared control
    // scope, so staggered posting cannot be mistaken for simultaneous
    // throughput. Receiver visibility is outside.
    std::vector<std::vector<double>> cohortMicroseconds;
  };
  std::vector<Size> sizes;
};
struct NetworkProfile {
  std::array<NetworkHostIdentity, 2> hosts;
  std::string probeIdentity;
  std::string pathScope = kNetworkProbeScope;
  std::string groupScope = kNetworkGroupProbeScope;
  std::vector<NetworkPathObservation> paths;
  std::vector<NetworkGroupObservation> groups;
  // Provenance of timed cohort barriers. MPI and bootstrap measurements keep
  // distinct scopes; the consumer itself need not link either control API.
  std::array<std::string, 2> groupControlIdentities;
};
struct NetworkPathEstimate {
  double microseconds = 0;
  double payloadGBs = 0;
  size_t lowerBytes = 0, upperBytes = 0;
};
void saveNetworkProfile(std::string const& file, NetworkProfile const& profile);
NetworkProfile loadNetworkProfile(std::string const& file);
// Same bounded/validated format as files, for collective distribution and
// embedding immutable resource evidence in a plan-cache snapshot.
std::string encodeNetworkProfile(NetworkProfile const& profile);
NetworkProfile decodeNetworkProfile(std::string const& data);
// Exact endpoint/identity match. Interpolate medians only within the observed
// byte interval. Never extrapolate, collapse endpoints, or overwrite NIC rates.
std::optional<NetworkPathEstimate> estimateNetworkPath(
    NetworkProfile const& profile,
    std::array<NetworkHostIdentity, 2> const& currentHosts,
    NetworkPathKey const& key, size_t bytes);
// Requires the exact complete flow set, both current host identities and an
// observed per-flow extent. Concurrent measurements are never inferred from
// isolated observations, subsets, or a differently shared NIC assignment.
std::optional<NetworkPathEstimate> estimateNetworkGroup(
    NetworkProfile const& profile,
    std::array<NetworkHostIdentity, 2> const& currentHosts,
    std::vector<NetworkPathKey> flows, size_t bytes);
std::vector<double> networkGroupSamples(NetworkGroupObservation const& group,
                                        size_t bytes);
}  // namespace uccl_lite
