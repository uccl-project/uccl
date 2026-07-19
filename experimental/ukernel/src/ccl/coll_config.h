#pragma once

#include "coll_types.h"
#include <cstddef>
#include <vector>

namespace UKernel {
namespace CCL {

struct CollectiveConfig {
  CollKind kind = CollKind::AllReduceRing;
  int nranks = 1;
  int rank = 0;
  size_t input_bytes = 0;
  size_t output_bytes = 0;
  size_t tile_bytes = 0;
  std::vector<size_t> input_split_bytes;
  std::vector<size_t> output_split_bytes;
  ScalarType dtype = ScalarType::Float32;
  ReductionKind reduction = ReductionKind::Sum;
  // Signal aggregation: one Signal/WaitSignal pair per this many tiles
  // (per chunk pair) instead of per tile. 1 = per-tile (finest pipeline),
  // larger values cut signal-op counts at the cost of coarser pipelining
  // at group boundaries. Both sides derive identical group tags.
  uint32_t signal_group_tiles = 1;
};

}  // namespace CCL
}  // namespace UKernel
