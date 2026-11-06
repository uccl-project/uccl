#include <cstddef>
#include <musa_runtime.h>

namespace {
__global__ void delay(unsigned steps) {
  unsigned volatile value = 1;
  for (unsigned i = 0; i < steps; ++i) value = value * 1664525u + 1013904223u;
}
__global__ void affine(float const* x, float* y, size_t const* experts,
                       size_t elements, size_t hidden) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < elements;
       i += size_t(blockDim.x) * gridDim.x) {
    float const expert = float(experts[i / hidden]);
    y[i] = (expert + 1) * x[i] - expert;
  }
}
}  // namespace
musaError_t delay_work(unsigned steps, musaStream_t stream) {
  if (steps) delay<<<1, 1, 0, stream>>>(steps);
  return musaGetLastError();
}
musaError_t expert_affine(float const* x, float* y, size_t const* experts,
                          size_t rows, size_t hidden, musaStream_t stream) {
  if (!rows) return musaSuccess;
  size_t const n = rows * hidden;
  unsigned const blocks = unsigned((n - 1) / 256 + 1);
  unsigned const grid = blocks < 4096 ? blocks : 4096;
  affine<<<grid, 256, 0, stream>>>(x, y, experts, n, hidden);
  return musaGetLastError();
}
