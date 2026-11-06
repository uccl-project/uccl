#include "musa_visibility_kernels.h"

__device__ uint8_t pattern(size_t index, unsigned epoch) {
  uint32_t x = uint32_t(index) ^ (epoch * 0x9e3779b9u);
  x ^= x >> 16;
  x *= 0x85ebca6bu;
  x ^= x >> 13;
  return uint8_t(x);
}

__global__ void fill_kernel(uint8_t* data, size_t bytes, unsigned epoch) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < bytes) data[i] = pattern(i, epoch);
}

__global__ void check_kernel(uint8_t const* data, size_t bytes, unsigned epoch,
                             unsigned* errors) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= bytes + 2 * kVisibilityGuard) return;
  bool const payload = i >= kVisibilityGuard && i < bytes + kVisibilityGuard;
  uint8_t const expected =
      payload ? pattern(i - kVisibilityGuard, epoch) : 0xa5;
  if (data[i] != expected) atomicAdd(errors, 1u);
}

musaError_t fill_payload(uint8_t* data, size_t bytes, unsigned epoch,
                         musaStream_t stream) {
  fill_kernel<<<(bytes + 255) / 256, 256, 0, stream>>>(data, bytes, epoch);
  return musaGetLastError();
}

musaError_t check_payload(uint8_t* allocation, size_t bytes, unsigned epoch,
                          unsigned* errors, musaStream_t stream) {
  check_kernel<<<(bytes + 2 * kVisibilityGuard + 255) / 256, 256, 0, stream>>>(
      allocation, bytes, epoch, errors);
  return musaGetLastError();
}
