#pragma once

#include "uccl_lite/topology.hpp"
#include <filesystem>

namespace uccl_lite {
// Separate from CUDA enumeration so sysfs discovery can be tested on synthetic
// PCI trees, including inaccessible links and non-contiguous NUMA IDs.
void discoverHardware(Topology& topology,
                      std::filesystem::path const& sysfs = "/sys");
std::string hardwareIdentity(Topology const& topology);
}  // namespace uccl_lite
