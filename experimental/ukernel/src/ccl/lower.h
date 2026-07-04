#pragma once

#include "algo/chunk_graph.h"
#include "coll_config.h"
#include "coll_types.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace UKernel {
namespace CCL {

// Tile the algorithm DAG and insert per-tile Signal/WaitSignal ops.
TiledResult lower_algo(CollAlgo const& algo, size_t tile_bytes,
                       bool inplace = false);

// Plan + lower: build CollAlgo from config, then lower to TiledResult.
TiledResult build_tiled(CollectiveConfig const& config, bool inplace);

}  // namespace CCL
}  // namespace UKernel
