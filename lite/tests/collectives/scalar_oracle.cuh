#pragma once
// Independent CPU oracle shared by typed core and public NCCL ABI clients.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
namespace scalar_test {
inline void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
template <class Bits>
Bits encode(float x) {
  if constexpr (sizeof(Bits) == 4) {
    Bits result;
    std::memcpy(&result, &x, 4);
    return result;
  } else
    return __half_as_ushort(__float2half_rn(x));
}
template <class Bits>
float decode(Bits x) {
  if constexpr (sizeof(Bits) == 4) {
    float result;
    std::memcpy(&result, &x, 4);
    return result;
  } else
    return __half2float(__ushort_as_half(x));
}
template <class Bits>
Bits input(size_t i, int rank, int generation, bool gather) {
  if (gather) {
    if constexpr (sizeof(Bits) == 2)
      return Bits(i * 313 + rank * 43 + generation);
    else {
      switch (i % 16) {
        case 0:
          return 0x80000000u;
        case 1:
          return 0x7f800000u;
        case 2:
          return 0xff800000u;
        case 3:
          return 0x7fc00000u + rank;
        case 4:
          return 0x7f800001u;
        default:
          return uint32_t(i) * 2246822519u + rank * 7919u + generation;
      }
    }
  }
  float v = float((int(i % 17) - 8) * (rank + 1));
  v = sizeof(Bits) == 4 ? v / float(rank + 3) : v / 4;
  return encode<Bits>(v + generation * 0.125f);
}
template <class Bits, class Invoke>
void exercise(int op, size_t count, bool inplace, unsigned alignment, int rank,
              int ranks, std::array<cudaStream_t, 2> streams, Invoke invoke) {
  size_t in = count * (op == 2 ? ranks : 1),
         out = count * (op == 1 ? ranks : 1),
         extent = std::max(in, out) + 512 / sizeof(Bits);
  size_t start = alignment / sizeof(Bits),
         aOffset = start + (inplace && op == 1 ? rank * count : 0),
         bOffset = start + (inplace && op == 2 ? rank * count : 0);
  bool inputB = inplace && op == 1, outputA = inplace && op != 1;
  Bits *a, *b;
  check(cudaMalloc(&a, extent * sizeof(Bits)));
  check(cudaMalloc(&b, extent * sizeof(Bits)));
  Bits guard = sizeof(Bits) == 4 ? Bits(0x5a5a5a5a) : Bits(0x5a5a);
  for (int generation = 0; generation < 3; ++generation) {
    std::vector<Bits> ha(extent, guard), hb(extent, guard);
    auto& source = inputB ? hb : ha;
    for (size_t i = 0; i < in; ++i)
      source[aOffset + i] = input<Bits>(i, rank, generation, op == 1);
    // Keep host input storage stable until its transfer completes, and order
    // production on the actual nonblocking consumer stream.
    auto expectedA = ha, expectedB = hb;
    auto stream = streams[generation % 2];
    check(cudaMemcpyAsync(a, ha.data(), extent * sizeof(Bits),
                          cudaMemcpyHostToDevice, stream));
    check(cudaMemcpyAsync(b, hb.data(), extent * sizeof(Bits),
                          cudaMemcpyHostToDevice, stream));
    auto& expected = outputA ? expectedA : expectedB;
    std::vector<double> tolerance(extent, 0);
    for (size_t i = 0; i < out; ++i) {
      if (op == 1)
        expected[bOffset + i] =
            input<Bits>(i % count, i / count, generation, true);
      else {
        double sum = 0, magnitude = 0;
        for (int sourceRank = 0; sourceRank < ranks; ++sourceRank) {
          double x = decode(input<Bits>(op == 2 ? size_t(rank) * count + i : i,
                                        sourceRank, generation, false));
          sum += x;
          magnitude += std::abs(x);
        }
        expected[bOffset + i] = encode<Bits>(float(sum));
        tolerance[bOffset + i] =
            sizeof(Bits) == 4 ? 2e-6 * std::max(1.0, magnitude) : 0;
      }
    }
    invoke(op, (inputB ? b : a) + aOffset, (outputA ? a : b) + bOffset, count,
           stream);
    check(cudaStreamSynchronize(stream));
    std::vector<Bits> actualA(extent), actualB(extent);
    check(cudaMemcpy(actualA.data(), a, extent * sizeof(Bits),
                     cudaMemcpyDeviceToHost));
    check(cudaMemcpy(actualB.data(), b, extent * sizeof(Bits),
                     cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < extent; ++i)
      for (int buffer = 0; buffer < 2; ++buffer) {
        Bits actual = (buffer ? actualB : actualA)[i],
             wanted = (buffer ? expectedB : expectedA)[i];
        bool outputRegion = (outputA ? buffer == 0 : buffer == 1) &&
                            i >= bOffset && i - bOffset < out && op != 1;
        if (outputRegion && tolerance[i] > 0) {
          float x = decode(actual), y = decode(wanted);
          if (!std::isfinite(x) || std::abs(double(x) - y) > tolerance[i])
            throw std::runtime_error("FP32 reduction tolerance failed");
        } else if (actual != wanted)
          throw std::runtime_error("scalar bitwise/input/guard oracle failed");
      }
  }
  check(cudaFree(a));
  check(cudaFree(b));
}
}  // namespace scalar_test
