#pragma once

#include "../../coll_config.h"

namespace UKernel {
namespace CCL {
namespace Testing {

inline CollectiveConfig make_test_config(int nranks, int rank,
                                         size_t tensor_bytes,
                                         size_t tile_bytes) {
  CollectiveConfig config{};
  config.nranks = nranks;
  config.rank = rank;
  config.input_bytes = tensor_bytes;
  config.output_bytes = tensor_bytes;
  config.tile_bytes = tile_bytes;
  config.kind = CollKind::AllReduceRing;
  return config;
}

}  // namespace Testing
}  // namespace CCL
}  // namespace UKernel
