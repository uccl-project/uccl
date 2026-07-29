#pragma once

#include "coll_algo.h"
#include "coll_config.h"
#include "coll_types.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace UKernel {
namespace CCL {

// Tile the algorithm DAG and insert Signal/WaitSignal ops. Signals are
// aggregated: one Signal/WaitSignal pair per signal_group_tiles tiles of
// a chunk pair (1 = per-tile). Coordination needs (staging, snapshots)
// come from MacroOp declarations, not from config switches.
TiledResult lower_algo(CollAlgo const& algo, size_t tile_bytes,
                       uint32_t signal_group_tiles = 1);

// Plan + lower: build CollAlgo from config, then lower to TiledResult.
TiledResult build_tiled(CollectiveConfig const& config, bool inplace);

}  // namespace CCL
}  // namespace UKernel
