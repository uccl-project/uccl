#pragma once

#include "algo/chunk_graph.h"
#include "coll_config.h"
#include "coll_types.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace UKernel {
namespace CCL {

// Tile the algorithm DAG and insert Signal/WaitSignal ops. Signals are
// aggregated: one Signal/WaitSignal pair per signal_group_tiles tiles of
// a chunk pair (1 = per-tile). reduce_snap_hs: for in-place AllReduce
// phase-1, gate each sender Put on the receiver's shard-snapshot Signal.
TiledResult lower_algo(CollAlgo const& algo, size_t tile_bytes,
                       bool inplace = false, bool stage_puts = false,
                       uint32_t signal_group_tiles = 1,
                       bool reduce_snap_hs = false);

// Plan + lower: build CollAlgo from config, then lower to TiledResult.
TiledResult build_tiled(CollectiveConfig const& config, bool inplace);

}  // namespace CCL
}  // namespace UKernel
