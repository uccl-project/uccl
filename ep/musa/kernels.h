#pragma once

#include <cstddef>
#include <musa_runtime.h>

namespace uccl::musa_ep {
// Maps are device size_t arrays. Scatter requires a validated permutation.
musaError_t gather_rows(float const* input, float* output, size_t const* rows,
                        size_t count, size_t hidden, musaStream_t stream);
musaError_t scatter_rows(float const* input, float* output, size_t const* rows,
                         size_t count, size_t hidden, musaStream_t stream);
musaError_t weighted_combine(float const* returned, float* output,
                             size_t const* lane_to_route, float const* weights,
                             size_t tokens, size_t hidden, size_t topk,
                             musaStream_t stream);
}  // namespace uccl::musa_ep
