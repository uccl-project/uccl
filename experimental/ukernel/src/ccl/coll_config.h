#pragma once

#include "coll_types.h"
#include <cstdlib>
#include <cstddef>
#include <string>
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

// Tile sizing rule, shared by the NCCL shim and the spray benchmarks:
// messages at or below the sweet spot move as ONE tile (tiling tiny
// messages only adds per-tile fixed overhead), and larger messages are
// tiled to at most kMaxTilesPerMessage tiles so per-tile overhead stays
// bounded at the large end. Measured on the A40/L40S pair, 64KB is the
// best sweet spot for small messages (256KB-4MB: 64KB tiles win by
// 6-29% over 256KB-1MB floors — more tiles pipeline the ring better
// than the fixed per-tile cost hurts); 16MB prefers a 256KB-1MB floor
// (-16%), and >=64MB is transport/device-bound regardless.
// UK_CCL_TILE_MIN_BYTES overrides the sweet spot for tuning.
inline size_t adaptive_tile_bytes(size_t bytes) {
  constexpr size_t kMaxTilesPerMessage = 256;
  static size_t const kTileSweetSpotBytes = [] {
    char const* v = std::getenv("UK_CCL_TILE_MIN_BYTES");
    return v ? static_cast<size_t>(std::stoull(v)) : (size_t{1} << 16);
  }();
  size_t tile = (bytes + kMaxTilesPerMessage - 1) / kMaxTilesPerMessage;
  tile = ((tile + 31) / 32) * 32;  // 32B-aligned tile boundaries
  return std::max(kTileSweetSpotBytes, tile);
}

}  // namespace CCL
}  // namespace UKernel
