#pragma once

#include <cstddef>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace uccl_lite {
namespace collective {
// Plain peer addresses: the owning runtime (CUDA peer access or a registered
// memory channel) establishes addressability and producer/consumer ordering.
// These primitives neither signal nor synchronize and allocate no resources.
template <typename T, int Capacity>
struct PeerInputs {
  T const* values[Capacity];
};

template <typename T>
__device__ __forceinline__ void copyElement(T const* input, T* output,
                                            size_t index) {
  output[index] = input[index];
}

// Caller supplies 16-byte-aligned addresses and a whole-vector range. This is
// a bitwise copy for both FP32 and BF16, including NaN payloads and signed
// zero.
__device__ __forceinline__ void copyVector(void const* input, void* output,
                                           size_t index) {
  reinterpret_cast<uint4*>(output)[index] =
      reinterpret_cast<uint4 const*>(input)[index];
}

template <typename T>
struct FloatAccumulator;
template <>
struct FloatAccumulator<float> {
  __device__ static __forceinline__ float load(float value) { return value; }
  __device__ static __forceinline__ float store(float value) { return value; }
};
template <>
struct FloatAccumulator<__nv_bfloat16> {
  __device__ static __forceinline__ float load(__nv_bfloat16 value) {
    return __bfloat162float(value);
  }
  __device__ static __forceinline__ __nv_bfloat16 store(float value) {
    return __float2bfloat16_rn(value);
  }
};

// Ascending source order, seeded by source zero (not +0), FP32 accumulation,
// one final conversion. Caller guarantees 1 <= ranks <= Capacity. A constant
// rank count lets the compiler specialize the existing four-L4 kernels.
template <typename T, int Capacity>
__device__ __forceinline__ T orderedSum(PeerInputs<T, Capacity> const& inputs,
                                        size_t index, int ranks = Capacity) {
  float sum = FloatAccumulator<T>::load(inputs.values[0][index]);
  for (int source = 1; source < ranks; ++source)
    sum += FloatAccumulator<T>::load(inputs.values[source][index]);
  return FloatAccumulator<T>::store(sum);
}

// Eight BF16 lanes per transaction; identical accumulation order and rounding
// to orderedSum. Offset is in BF16 elements; index is in 128-bit vectors.
// Caller guarantees each input + offset is 16-byte aligned.
template <int Capacity>
__device__ __forceinline__ uint4
orderedSumBf16Vector(PeerInputs<__nv_bfloat16, Capacity> const& inputs,
                     size_t index, size_t offset, int ranks = Capacity) {
  float sums[8];
  for (int source = 0; source < ranks; ++source) {
    uint4 packet =
        reinterpret_cast<uint4 const*>(inputs.values[source] + offset)[index];
    unsigned words[4] = {packet.x, packet.y, packet.z, packet.w};
#pragma unroll
    for (int word = 0; word < 4; ++word) {
      float low = __bfloat162float(__ushort_as_bfloat16(words[word] & 65535));
      float high = __bfloat162float(__ushort_as_bfloat16(words[word] >> 16));
      if (source == 0) {
        sums[2 * word] = low;
        sums[2 * word + 1] = high;
      } else {
        sums[2 * word] += low;
        sums[2 * word + 1] += high;
      }
    }
  }
  unsigned words[4];
#pragma unroll
  for (int word = 0; word < 4; ++word)
    words[word] =
        unsigned(__bfloat16_as_ushort(__float2bfloat16_rn(sums[2 * word]))) |
        (unsigned(__bfloat16_as_ushort(__float2bfloat16_rn(sums[2 * word + 1])))
         << 16);
  return make_uint4(words[0], words[1], words[2], words[3]);
}
}  // namespace collective
}  // namespace uccl_lite
