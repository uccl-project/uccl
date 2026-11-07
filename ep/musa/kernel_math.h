#pragma once

#include <cstddef>

#if defined(__CUDACC__) || defined(__MUSACC__)
#define EP_DEVICE __host__ __device__
#else
#define EP_DEVICE
#endif

namespace uccl::musa_ep {
EP_DEVICE inline size_t gather_index(size_t element, size_t hidden,
                                     size_t const* rows) {
  return rows[element / hidden] * hidden + element % hidden;
}

// One output element owns its lane-ordered reduction; no atomic ordering.
EP_DEVICE inline float combine_element(float const* returned,
                                       size_t const* lane_to_route,
                                       float const* weights, size_t element,
                                       size_t hidden, size_t topk) {
  size_t const token = element / hidden, h = element % hidden;
  float result = 0;
  for (size_t lane = 0; lane < topk; ++lane) {
    size_t const i = token * topk + lane, row = lane_to_route[i];
    if (row != ~size_t{0}) {
      float const product = weights[i] * returned[row * hidden + h];
      result = result + product;
    }
  }
  return result;
}
}  // namespace uccl::musa_ep
#undef EP_DEVICE
