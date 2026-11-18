#include "kernel_math.h"
#include "kernels.h"
#include <algorithm>
#include <cstdint>
#include <limits>

namespace uccl::musa_ep {
namespace {
constexpr unsigned kThreads = 256;

// Flattened row copies coalesce even for H=1. Vector lanes are used only when
// every source/destination row is 16-byte aligned; tails use the scalar path.
template <class T, bool Scatter>
__global__ void copy_rows(T const* input, T* output, size_t const* rows,
                          size_t elements, size_t width) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < elements;
       i += size_t(gridDim.x) * blockDim.x) {
    size_t const mapped = gather_index(i, width, rows);
    if constexpr (Scatter)
      output[mapped] = input[i];
    else
      output[i] = input[mapped];
  }
}

__global__ void combine(float const* returned, float* output,
                        size_t const* lanes, float const* weights,
                        size_t elements, size_t hidden, size_t topk) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < elements;
       i += size_t(gridDim.x) * blockDim.x)
    output[i] = combine_element(returned, lanes, weights, i, hidden, topk);
}

__global__ void combine4(float4 const* returned, float4* output,
                         size_t const* lanes, float const* weights,
                         size_t elements, size_t width, size_t topk) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < elements;
       i += size_t(gridDim.x) * blockDim.x) {
    float4 result = {0, 0, 0, 0};
    size_t const token = i / width, column = i % width;
    for (size_t lane = 0; lane < topk; ++lane) {
      size_t const at = token * topk + lane, row = lanes[at];
      if (row == ~size_t{0}) continue;
      float const w = weights[at];
      float4 const value = returned[row * width + column];
      result.x += w * value.x;
      result.y += w * value.y;
      result.z += w * value.z;
      result.w += w * value.w;
    }
    output[i] = result;
  }
}

unsigned blocks(size_t elements) {
  return unsigned(std::min(size_t{4096}, (elements - 1) / kThreads + 1));
}

bool valid_shape(size_t rows, size_t hidden) {
  return hidden && hidden <= 8192 &&
         rows <= std::numeric_limits<size_t>::max() / hidden / sizeof(float);
}

template <bool Scatter>
musaError_t copy(float const* input, float* output, size_t const* rows,
                 size_t count, size_t hidden, musaStream_t stream) {
  if (!valid_shape(count, hidden)) return musaErrorInvalidValue;
  if (!count) return musaSuccess;
  if (!input || !output || !rows || input == output ||
      (reinterpret_cast<uintptr_t>(input) |
       reinterpret_cast<uintptr_t>(output)) %
              alignof(float) !=
          0)
    return musaErrorInvalidValue;
  if (hidden % 4 == 0 && (reinterpret_cast<uintptr_t>(input) |
                          reinterpret_cast<uintptr_t>(output)) %
                                 alignof(float4) ==
                             0) {
    size_t const n = count * (hidden / 4);
    copy_rows<float4, Scatter><<<blocks(n), kThreads, 0, stream>>>(
        reinterpret_cast<float4 const*>(input),
        reinterpret_cast<float4*>(output), rows, n, hidden / 4);
  } else {
    size_t const n = count * hidden;
    copy_rows<float, Scatter>
        <<<blocks(n), kThreads, 0, stream>>>(input, output, rows, n, hidden);
  }
  return musaGetLastError();
}
}  // namespace

musaError_t gather_rows(float const* input, float* output, size_t const* rows,
                        size_t count, size_t hidden, musaStream_t stream) {
  return copy<false>(input, output, rows, count, hidden, stream);
}
musaError_t scatter_rows(float const* input, float* output, size_t const* rows,
                         size_t count, size_t hidden, musaStream_t stream) {
  return copy<true>(input, output, rows, count, hidden, stream);
}
musaError_t weighted_combine(float const* returned, float* output,
                             size_t const* lanes, float const* weights,
                             size_t tokens, size_t hidden, size_t topk,
                             musaStream_t stream) {
  if (!valid_shape(tokens, hidden) || !topk || topk > 8 ||
      tokens > std::numeric_limits<size_t>::max() / topk)
    return musaErrorInvalidValue;
  if (!tokens) return musaSuccess;
  if (!output || !lanes || !weights || returned == output ||
      (reinterpret_cast<uintptr_t>(returned) |
       reinterpret_cast<uintptr_t>(output)) %
              alignof(float) !=
          0)
    return musaErrorInvalidValue;
  size_t const n = tokens * hidden;
  if (hidden % 4 == 0 && (reinterpret_cast<uintptr_t>(returned) |
                          reinterpret_cast<uintptr_t>(output)) %
                                 alignof(float4) ==
                             0)
    combine4<<<blocks(n / 4), kThreads, 0, stream>>>(
        reinterpret_cast<float4 const*>(returned),
        reinterpret_cast<float4*>(output), lanes, weights, n / 4, hidden / 4,
        topk);
  else
    combine<<<blocks(n), kThreads, 0, stream>>>(returned, output, lanes,
                                                weights, n, hidden, topk);
  return musaGetLastError();
}
}  // namespace uccl::musa_ep
