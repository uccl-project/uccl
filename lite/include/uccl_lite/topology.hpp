#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace uccl_lite {

// Rates are decimal GB/s; latency is microseconds. Unknown is never zero-cost.
enum class Evidence { Unknown, Advertised, Measured };
struct Performance {
  double bandwidthGBs = 0;
  double latencyUs = -1;  // Negative means unknown.
  Evidence evidence = Evidence::Unknown;
};
enum class HardwareKind { Gpu, PciBridge, Numa, Nic };
struct HardwareNode {
  std::string id;
  HardwareKind kind = HardwareKind::PciBridge;
  int device = -1;
  int numa = -1;
  // DRAM or GPU memory bandwidth. Discovery does not invent a measured rate.
  Performance memory;
};
struct HardwareLink {
  size_t source = 0, destination = 0;
  // Transfers using the same resource compete for its bandwidth. Opposite
  // PCIe directions have different resources because the link is full duplex.
  std::string resource;
  Performance performance;
  int numaDistance =
      0;  // Relative OS distance, never bandwidth or microseconds.
};
enum class PathKind { PeerCopy, PeerRead, PeerWrite, HostCopy };
struct HardwarePath {
  size_t source = 0, destination = 0;
  PathKind kind = PathKind::PeerCopy;
  std::string resource;
  Performance performance;
};
struct NetworkPort {
  std::string name, pciId;
  int port = 0, numa = -1;
  size_t node = 0;
  bool active = false;
  Performance transmit;
  // GDR support must be established by the transport for each GPU/port pair.
};
struct Topology {
  std::vector<int> devices, numaNodes;
  std::vector<std::string> gpuNames;
  std::vector<bool> mappedHost, peerAccess;  // Peer matrix is row-major.
  std::map<int, int> permittedCpusPerNode;
  std::map<int, std::vector<int>> permittedCpuIds;
  bool avx2 = false;
  int cudaRuntimeVersion = 0;
  std::string driverVersion;
  // Exact local identity: UUIDs, PCI IDs, CPU/affinity, driver/toolkit, router
  // knobs.
  std::string identity;
  // Discovery snapshot, before backend-specific policy/build knobs are
  // appended.
  std::string hardwareIdentity;
  std::string calibrationIdentity;
  std::map<std::string, std::string> resourceCalibrationIdentities;
  std::vector<HardwareNode> hardware;
  std::vector<HardwareLink> links;
  std::vector<HardwarePath> paths;
  std::vector<NetworkPort> networkPorts;
  // Entries follow devices/rank order; their values index hardware.
  std::vector<size_t> gpuNodes;
  std::vector<std::string> gpuPciIds;
};

// Passive discovery only: no worker creation, bandwidth probes, allocations,
// peer enablement, or changes to the current CUDA device.
Topology inspectTopology(std::vector<int> const& devices);

}  // namespace uccl_lite
