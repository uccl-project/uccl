// SPDX-License-Identifier: Apache-2.0
// Backward probability slots must follow the original top-k, including holes.
#include <cuda_runtime.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace nccl_ep::hybridep {
void dense_to_sparse_prob_combine(const float*, const int64_t*, float*, int64_t*, int, int, int, cudaStream_t);
}
#define CUDA_OK(call) do { auto status = (call); if (status != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); std::exit(2); \
} } while (0)

int main() {
  constexpr int experts = 8, topk = 2;
  constexpr std::array<std::array<int64_t, topk>, 4> routes{{{7, 0}, {0, 7}, {-1, 5}, {2, -1}}};
  int checked = 0;
  for (int tokens : {1, 255, 256, 257}) {
    std::vector<int64_t> indices(tokens * topk), actual_indices(tokens * topk);
    std::vector<float> dense(tokens * experts, 0), expected(tokens * topk), actual(tokens * topk);
    for (int token = 0; token < tokens; ++token) for (int k = 0; k < topk; ++k) {
      int slot = token * topk + k;
      indices[slot] = routes[token % routes.size()][k];
      expected[slot] = indices[slot] < 0 ? 0.0f : float((token % 7) + 1 + k * 8) / 16.0f;
      if (indices[slot] >= 0) dense[token * experts + indices[slot]] = expected[slot];
    }
    float *device_dense, *device_weights;
    int64_t *device_indices, *device_output_indices;
    CUDA_OK(cudaMalloc(&device_dense, dense.size() * sizeof(float)));
    CUDA_OK(cudaMalloc(&device_weights, actual.size() * sizeof(float)));
    CUDA_OK(cudaMalloc(&device_indices, indices.size() * sizeof(int64_t)));
    CUDA_OK(cudaMalloc(&device_output_indices, actual_indices.size() * sizeof(int64_t)));
    CUDA_OK(cudaMemcpy(device_dense, dense.data(), dense.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(device_indices, indices.data(), indices.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    for (bool output_indices : {false, true}) {
      CUDA_OK(cudaMemset(device_weights, 0xa5, actual.size() * sizeof(float)));
      CUDA_OK(cudaMemset(device_output_indices, 0xa5, actual_indices.size() * sizeof(int64_t)));
      nccl_ep::hybridep::dense_to_sparse_prob_combine(device_dense, device_indices, device_weights,
          output_indices ? device_output_indices : nullptr, tokens, topk, experts, nullptr);
      CUDA_OK(cudaGetLastError());
      CUDA_OK(cudaMemcpy(actual.data(), device_weights, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
      if (output_indices) CUDA_OK(cudaMemcpy(actual_indices.data(), device_output_indices,
          actual_indices.size() * sizeof(int64_t), cudaMemcpyDeviceToHost));
      for (int slot = 0; slot < tokens * topk; ++slot) {
        if (actual[slot] != expected[slot] || (output_indices && actual_indices[slot] != indices[slot])) {
          std::fprintf(stderr, "probability slot mismatch: tokens=%d slot=%d\n", tokens, slot);
          return 1;
        }
        ++checked;
      }
    }
    CUDA_OK(cudaFree(device_output_indices)); CUDA_OK(cudaFree(device_indices));
    CUDA_OK(cudaFree(device_weights)); CUDA_OK(cudaFree(device_dense));
  }
  std::printf("Original top-k probability order and masked slots PASS: %d entries\n", checked);
}
