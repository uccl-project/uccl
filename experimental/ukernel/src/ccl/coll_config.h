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
  bool use_sm_ipc = true;
};

}  // namespace CCL
}  // namespace UKernel
