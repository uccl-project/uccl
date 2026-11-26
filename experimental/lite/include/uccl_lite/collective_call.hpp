#pragma once
#include "collective_types.hpp"
#include "device_port.hpp"
#include <vector>

namespace uccl_lite {
struct CollectiveCall {
  Collective collective = Collective::AllReduce;
  ScalarType dtype = ScalarType::BFloat16;
  // AR tensor elements, AG contribution elements, or RS output-shard elements.
  size_t count = 0;
  bool inPlace = false;
  std::vector<DevicePort> ports;
  Workload workload() const;
};

}  // namespace uccl_lite
