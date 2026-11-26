#pragma once
#include "planner.hpp"

namespace uccl_lite {
struct ProbeObservation {
  std::string resource, operation, timingScope;
  size_t bytes = 0;
  int warmups = 0, iterations = 0;
  // Per-call microseconds for each complete repetition, not a best envelope.
  std::vector<double> microseconds;
};
struct HardwareProfile {
  HardwareCalibration calibration;
  std::vector<ProbeObservation> observations;
};
// Versioned, lossless profile. Parsing never mutates or admits a live topology.
void saveHardwareProfile(std::string const& path,
                         HardwareProfile const& profile);
HardwareProfile loadHardwareProfile(std::string const& path);
struct GpuProbeOptions {
  size_t bytes = 64UL << 20;
  int warmups = 2, iterations = 5, repetitions = 3;
};
// Explicit idle initialization/calibration boundary, never enqueue/capture.
// Allocates private probe buffers, verifies every output, and restores current
// device and peer-access state. Does not start CPU-router workers or touch the
// caller's buffers. Measurements are GPU memory-copy and end-to-end P2P paths;
// they do not claim to identify individual PCIe bridge capacities.
HardwareProfile probeGpuResources(Topology const& topology,
                                  GpuProbeOptions const& options = {});
struct HostProbeOptions {
  size_t bytesPerWorker = 64UL << 20;
  int workersPerNode = 4, warmups = 2, iterations = 5, repetitions = 3;
};
struct HostCopyPath {
  std::string resource;
  int sourceNode = -1, destinationNode = -1;
  std::vector<int> cpus;
};
// Register explicitly scoped CPU-copy resources; worker CPU IDs are part of
// their names. Rates count one-way copied payload, not read+write traffic.
// Register the same configuration before loading/applying a saved
// host profile. Generic node memory and interconnect rates stay unknown.
std::vector<HostCopyPath> registerHostCopyPaths(
    Topology& topology, HostProbeOptions const& options = {});
HardwareProfile probeHostResources(Topology& topology,
                                   HostProbeOptions const& options = {});
}  // namespace uccl_lite
