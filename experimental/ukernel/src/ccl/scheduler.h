#pragma once

#include "algo/chunk_graph.h"
#include "coll_config.h"
#include "coll_types.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace UKernel {
namespace CCL {

struct Op {
  OpKind kind = OpKind::Copy;
  size_t bytes = 0;
  size_t src_off = 0;
  size_t dst_off = 0;
  uint32_t src_peer = 0;
  uint32_t dst_peer = 0;
  bool copy_from_staging = false;
  std::vector<uint32_t> deps;
};

struct Schedule {
  std::vector<std::vector<uint32_t>> layers;
};

struct TiledResult {
  std::vector<Op> ops;
  std::vector<uint32_t> chunk_of;  // chunk_of[tile_idx] = chunk index
  size_t staging_bytes_required = 0;
  size_t input_bytes = 0;
  size_t output_bytes = 0;
  int rank = 0;
  int nranks = 1;
  ReductionKind reduction = ReductionKind::None;
  Schedule schedule;
};

TiledResult tile_and_schedule(CollAlgo const& algo, size_t tile_bytes);

Schedule schedule_ops(std::vector<Op> const& ops);

TiledResult build_tiled(CollectiveConfig const& config, bool inplace);

}  // namespace CCL
}  // namespace UKernel
