#pragma once
// Private exact-request qualification shared by prepared runtime selectors.
#include "uccl_lite/runtime/collective_backend.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace uccl_lite::runtime::qualification {
namespace {
inline void check(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
inline unsigned alignment(void const* a, void const* b) {
  auto bits = reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b);
  unsigned result = 1;
  while (result < 256 && !(bits & result)) result *= 2;
  return result;
}
template <class Bits>
__host__ __device__ Bits encodeValue(float x) {
  if constexpr (sizeof(Bits) == 4) {
#ifdef __CUDA_ARCH__
    return __float_as_uint(x);
#else
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return bits;
#endif
  } else
    return __bfloat16_as_ushort(__float2bfloat16_rn(x));
}
template <class Bits>
__host__ __device__ float decodeValue(Bits x) {
  if constexpr (sizeof(Bits) == 4) {
#ifdef __CUDA_ARCH__
    return __uint_as_float(x);
#else
    float result;
    std::memcpy(&result, &x, sizeof(result));
    return result;
#endif
  } else
    return __bfloat162float(__ushort_as_bfloat16(x));
}
template <class Bits>
__host__ __device__ Bits value(size_t index, int rank, int seed, bool gather) {
  if (gather) {
    if constexpr (sizeof(Bits) == 2)
      return static_cast<Bits>(index + rank * 191 + seed);
    else {
      switch (index % 32) {
        case 0:
          return 0x80000000u;
        case 1:
          return 0x7f800000u;
        case 2:
          return 0xff800000u;
        case 3:
          return 0x7fc00000u | uint32_t(rank + 1);
        case 4:
          return 0x7f800001u;
        default:
          return uint32_t(index) * 2654435761u + uint32_t(rank) * 7919u +
                 uint32_t(seed);
      }
    }
  }
  // Every partial sum is integral within [-128,128] for 32 ranks.
  return encodeValue<Bits>(
      float((int(index % 7) - 3) * (rank % 3 - 1) + seed % 3 - 1));
}
struct ProbeLayout {
  size_t extent, inputOffset, outputOffset, inputCount, outputCount, count;
  int rank, ranks, op;
  bool inputB, outputA;
};
template <class Bits>
__global__ void initializeProbe(Bits* a, Bits* b, ProbeLayout l, int seed) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < l.extent;
       i += size_t(gridDim.x) * blockDim.x) {
    a[i] = b[i] = sizeof(Bits) == 4 ? Bits(0x5a5a5a5a) : Bits(0x5a5a);
    if (i >= l.inputOffset && i - l.inputOffset < l.inputCount)
      (l.inputB ? b : a)[i] =
          value<Bits>(i - l.inputOffset, l.rank, seed, l.op == 1);
  }
}
// Independent full-buffer oracle: double accumulation of the exact inputs.
template <class Bits>
__global__ void verifyProbe(Bits const* a, Bits const* b, ProbeLayout l,
                            int seed, unsigned* wrong) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < l.extent;
       i += size_t(gridDim.x) * blockDim.x) {
    Bits ea = sizeof(Bits) == 4 ? Bits(0x5a5a5a5a) : Bits(0x5a5a), eb = ea;
    if (i >= l.inputOffset && i - l.inputOffset < l.inputCount)
      (l.inputB ? eb : ea) =
          value<Bits>(i - l.inputOffset, l.rank, seed, l.op == 1);
    if (i >= l.outputOffset && i - l.outputOffset < l.outputCount) {
      size_t index = i - l.outputOffset;
      Bits expected;
      if (l.op == 1)
        expected = value<Bits>(index % l.count, index / l.count, seed, true);
      else {
        double sum = 0;
        for (int source = 0; source < l.ranks; ++source)
          sum += decodeValue(
              value<Bits>(l.op == 2 ? size_t(l.rank) * l.count + index : index,
                          source, seed, false));
        expected = encodeValue<Bits>(float(sum));
      }
      (l.outputA ? ea : eb) = expected;
    }
    if (a[i] != ea || b[i] != eb) atomicExch(wrong, 1);
  }
}
struct Probe {
  void *a = nullptr, *b = nullptr;
  unsigned* wrong = nullptr;
  cudaStream_t stream = nullptr;
  ProbeLayout layout{};
  ScalarType dtype = ScalarType::BFloat16;
  size_t scalar = 2;
  void const* input() const {
    return static_cast<char const*>(layout.inputB ? b : a) +
           layout.inputOffset * scalar;
  }
  void* output() const {
    return static_cast<char*>(layout.outputA ? a : b) +
           layout.outputOffset * scalar;
  }
  ~Probe() {
    if (stream) cudaStreamSynchronize(stream);
    if (a) cudaFree(a);
    if (b) cudaFree(b);
    if (wrong) cudaFree(wrong);
    if (stream) cudaStreamDestroy(stream);
  }
  void allocate(LocalRequest const& request, int rank, int ranks,
                ScalarType type) {
    dtype = type;
    scalar = scalarBytes(type);
    if (type != ScalarType::BFloat16 && type != ScalarType::Float32)
      throw std::invalid_argument("unsupported qualification dtype");
    int op = int(request.operation);
    auto placement = request.layouts[rank];
    size_t inputs = request.count * (op == 2 ? ranks : 1),
           outputs = request.count * (op == 1 ? ranks : 1),
           offset = placement.alignment / scalar;
    layout = {
        std::max(inputs, outputs) + 512 / scalar,
        offset + (placement.inPlace && op == 1 ? rank * request.count : 0),
        offset + (placement.inPlace && op == 2 ? rank * request.count : 0),
        inputs,
        outputs,
        request.count,
        rank,
        ranks,
        op,
        placement.inPlace && op == 1,
        placement.inPlace && op != 1};
    check(cudaMalloc(&a, layout.extent * scalar));
    check(cudaMalloc(&b, layout.extent * scalar));
    check(cudaMalloc(&wrong, sizeof(unsigned)));
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (alignment(input(), output()) != placement.alignment)
      throw std::invalid_argument(
          "in-place shard offset violates requested alignment");
  }
  void initialize(int seed) {
    if (dtype == ScalarType::Float32)
      initializeProbe<<<128, 256, 0, stream>>>(
          static_cast<uint32_t*>(a), static_cast<uint32_t*>(b), layout, seed);
    else
      initializeProbe<<<128, 256, 0, stream>>>(
          static_cast<uint16_t*>(a), static_cast<uint16_t*>(b), layout, seed);
    check(cudaGetLastError());
  }
  void verify(int seed) {
    if (dtype == ScalarType::Float32)
      verifyProbe<<<128, 256, 0, stream>>>(static_cast<uint32_t*>(a),
                                           static_cast<uint32_t*>(b), layout,
                                           seed, wrong);
    else
      verifyProbe<<<128, 256, 0, stream>>>(static_cast<uint16_t*>(a),
                                           static_cast<uint16_t*>(b), layout,
                                           seed, wrong);
    check(cudaGetLastError());
  }
};
}  // namespace
}  // namespace uccl_lite::runtime::qualification
