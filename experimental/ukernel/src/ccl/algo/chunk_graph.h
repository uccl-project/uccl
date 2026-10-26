#pragma once

#include "coll_config.h"
#include "coll_types.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace UKernel {
namespace CCL {

struct Chunk {
  AlgoOpKind op = AlgoOpKind::Put;
  size_t bytes = 0;
  size_t src_off = 0;
  size_t dst_off = 0;
  int src_rank = -1;  // -1 = local buffer, >=0 = remote rank
  int dst_rank = -1;
  uint32_t pair_id = 0;  // matched Send/Recv pairs share the same id
  std::vector<uint32_t> deps;
};

struct CollAlgo {
  CollKind kind = CollKind::AllReduceRing;
  int nranks = 1;
  int rank = 0;
  size_t input_bytes = 0;
  size_t output_bytes = 0;
  ReductionKind reduction = ReductionKind::None;
  std::vector<Chunk> chunks;
};

CollAlgo build_coll_algo(CollectiveConfig const& config, bool inplace);

}  // namespace CCL
}  // namespace UKernel
