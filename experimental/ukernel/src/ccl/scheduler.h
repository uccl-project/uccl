#pragma once

#include "coll_algo.h"
#include "coll_config.h"
#include "coll_types.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace UKernel {
namespace CCL {

struct Op {
  OpKind kind = OpKind::DeviceCopy;
  size_t bytes = 0;
  size_t src_off = 0;
  size_t dst_off = 0;
  uint32_t src_peer = 0;
  uint32_t dst_peer = 0;
  bool copy_from_staging = false;
  std::vector<uint32_t> deps;
};

struct Schedule {
  uint32_t num_streams = 1;
  std::vector<std::vector<uint32_t>> stream_ops;
};

struct TiledResult {
  std::vector<Op> ops;
  size_t staging_bytes_required = 0;
  size_t input_bytes = 0;
  size_t output_bytes = 0;
  Schedule schedule;
};

TiledResult tile_and_schedule(CollAlgo const& algo, size_t tile_bytes);

Schedule schedule_ops(std::vector<Op> const& ops);

TiledResult build_plan(CollectiveConfig const& config, bool inplace);

}  // namespace CCL
}  // namespace UKernel
