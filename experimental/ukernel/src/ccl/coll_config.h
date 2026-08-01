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
  // Explicit in-place declaration. The executor cannot infer it from
  // buffer pointers for every collective: NCCL AllGather/ReduceScatter
  // in-place layouts use DISTINCT pointers that physically overlap
  // (sendbuff = recvbuff + rank*sendcount), so `input == output` is
  // false. The shim sets this when it detects the overlapping layout;
  // for AllReduce/AllToAll the executor also accepts input==output as
  // the implicit form.
  bool inplace = false;
};

}  // namespace CCL
}  // namespace UKernel
