// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu_utils.hpp"
#include <fstream>
#include <numa.h>

// Convert a logical cudaDev index to the NVML device minor number
static const std::string getBusId(int cudaDev) {
  // On most systems, the PCI bus ID comes back as in the 0000:00:00.0
  // format. Still need to allocate proper space in case PCI domain goes
  // higher.
  char busIdChar[] = "00000000:00:00.0";
  MSCCLPP_CUDATHROW(
      cudaDeviceGetPCIBusId(busIdChar, sizeof(busIdChar), cudaDev));
  // we need the hex in lower case format
  for (size_t i = 0; i < sizeof(busIdChar); i++) {
    busIdChar[i] = std::tolower(busIdChar[i]);
  }
  return std::string(busIdChar);
}

namespace mscclpp {

int getDeviceNumaNode(int cudaDev) {
  std::string busId = getBusId(cudaDev);
  std::string file_str = "/sys/bus/pci/devices/" + busId + "/numa_node";
  std::ifstream file(file_str);
  int numaNode;
  if (file.is_open()) {
    if (!(file >> numaNode)) {
      throw Error("Failed to read NUMA node from file: " + file_str,
                  ErrorCode::SystemError);
    }
  } else {
    throw Error("Failed to open file: " + file_str, ErrorCode::SystemError);
  }
  return numaNode;
}

void numaBind(int node) {
  int totalNumNumaNodes = numa_num_configured_nodes();
  if (node < 0 || node >= totalNumNumaNodes) {
    throw Error("Invalid NUMA node " + std::to_string(node) +
                    ", must be between 0 and " +
                    std::to_string(totalNumNumaNodes),
                ErrorCode::InvalidUsage);
  }
  nodemask_t mask;
  nodemask_zero(&mask);
  nodemask_set_compat(&mask, node);
  numa_bind_compat(&mask);
}

}  // namespace mscclpp
