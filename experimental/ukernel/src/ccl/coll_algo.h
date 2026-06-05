#pragma once

#include "coll_config.h"
#include "coll_types.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace UKernel {
namespace CCL {

enum class TileOrder : uint8_t {
  Independent = 0,  // tiles within this op are independent (ring, sm-ipc)
  Sequential = 1,   // tiles must be ordered sequentially (dma staging)
};

struct AlgoOp {
  OpKind kind = OpKind::DeviceCopy;
  size_t bytes = 0;
  size_t src_off = 0;
  size_t dst_off = 0;
  uint32_t src_peer = 0;
  uint32_t dst_peer = 0;
  bool copy_from_staging = false;
  TileOrder tile_order = TileOrder::Independent;
  std::vector<uint32_t> deps;
};

struct CollAlgo {
  CollectiveKind collective = CollectiveKind::AllReduce;
  int nranks = 1;
  int rank = 0;
  size_t input_bytes = 0;
  size_t output_bytes = 0;
  ReductionKind reduction = ReductionKind::None;
  std::vector<AlgoOp> ops;
};

CollAlgo build_coll_algo(CollectiveConfig const& config, bool inplace);

}  // namespace CCL
}  // namespace UKernel
