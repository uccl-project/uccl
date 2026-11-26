#include "rank_permutation.hpp"
#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>
using uccl_lite::runtime::permuteRankShards;
void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
int main() {
  try {
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (int ranks : {1, 2, 4, 8, 32})
      for (size_t count : {1UL, 257UL, 4096UL, 1048576UL}) {
        constexpr size_t guard = 17;
        size_t extent = count * ranks + 2 * guard;
        std::vector<uint32_t> input(extent, 0x5a5a5a5a),
            expected(extent, 0x5a5a5a5a), actual(extent);
        for (size_t i = 0; i < count * ranks; ++i)
          input[guard + i] =
              i % 16 == 0 ? 0x7f800001u : uint32_t(i) * 2654435761u;
        uint32_t *a, *b, *c;
        check(cudaMalloc(&a, extent * 4));
        check(cudaMalloc(&b, extent * 4));
        check(cudaMalloc(&c, extent * 4));
        check(cudaMemcpy(a, input.data(), extent * 4, cudaMemcpyHostToDevice));
        for (int rotation : {0, 1, 3}) {
          std::vector<int> order(ranks);
          for (int i = 0; i < ranks; ++i)
            order[i] = (ranks - 1 - i + rotation) % ranks;
          std::fill(expected.begin(), expected.end(), 0x5a5a5a5a);
          check(cudaMemcpy(b, expected.data(), extent * 4,
                           cudaMemcpyHostToDevice));
          check(cudaMemcpy(c, expected.data(), extent * 4,
                           cudaMemcpyHostToDevice));
          permuteRankShards(a + guard, b + guard, count, order.data(), ranks,
                            true, stream);
          permuteRankShards(b + guard, c + guard, count, order.data(), ranks,
                            false, stream);
          check(cudaStreamSynchronize(stream));
          for (int execution = 0; execution < ranks; ++execution)
            for (size_t i = 0; i < count; ++i)
              expected[guard + execution * count + i] =
                  input[guard + order[execution] * count + i];
          check(
              cudaMemcpy(actual.data(), b, extent * 4, cudaMemcpyDeviceToHost));
          if (actual != expected)
            throw std::runtime_error("permutation/shard/guard failure");
          check(
              cudaMemcpy(actual.data(), c, extent * 4, cudaMemcpyDeviceToHost));
          if (actual != input)
            throw std::runtime_error("inverse/guard failure");
          check(
              cudaMemcpy(actual.data(), a, extent * 4, cudaMemcpyDeviceToHost));
          if (actual != input) throw std::runtime_error("input mutation");
        }
        if (ranks > 1) {
          std::vector<int> invalid(ranks, 0);
          bool rejected = false;
          try {
            permuteRankShards(a, b, count, invalid.data(), ranks, true, stream);
          } catch (std::invalid_argument const&) {
            rejected = true;
          }
          if (!rejected)
            throw std::runtime_error("nonbijective permutation accepted");
        }
        check(cudaFree(a));
        check(cudaFree(b));
        check(cudaFree(c));
      }
    check(cudaStreamDestroy(stream));
    std::cout << "PASS: bitwise rank permutation, full buffers/guards, input "
                 "preservation, inverse and invalid-map rejection; 1/2/4/8/32 "
                 "shards\n";
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
