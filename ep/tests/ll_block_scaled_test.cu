#include "common.hpp"
#include "ep_config.hpp"
#include "internode_ll.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define CHECK(c)                                                       \
  do {                                                                 \
    if (auto error = (c); error != cudaSuccess) {                      \
      std::fprintf(stderr, "%s: %s\n", #c, cudaGetErrorString(error)); \
      std::exit(2);                                                    \
    }                                                                  \
  } while (false)

constexpr int kRanks = 2;
constexpr int kExperts = 2;
constexpr int kTokens = 4;
constexpr int kHidden = 4096;
constexpr int kTopK = 2;
constexpr int kRows = kTokens * kRanks;
using bf16 = __nv_bfloat16;

template <typename T>
T* allocate(size_t count) {
  T* ptr = nullptr;
  CHECK(cudaMalloc(&ptr, sizeof(T) * count));
  CHECK(cudaMemset(ptr, 0, sizeof(T) * count));
  return ptr;
}

template <typename T>
std::vector<T> read(const T* ptr, size_t count) {
  std::vector<T> output(count);
  CHECK(cudaMemcpy(output.data(), ptr, sizeof(T) * count,
                   cudaMemcpyDeviceToHost));
  return output;
}

float token_value(int source, int token) {
  return std::ldexp(token % 2 == 0 ? 1.0f : -1.0f, token - 1 + source);
}

struct Rank {
  cudaStream_t stream{};
  uint8_t *rdma{}, *atomic{}, *workspace{};
  void** peer_rdma{};
  bf16 *input{}, *expert_out{}, *combined{};
  int64_t *routes{}, *ranges{};
  uint8_t *received{}, *scales{};
  float *token_scale{}, *weights{};
  int *recv_count{}, *src_info{};
};

int main() {
  int devices = 0;
  CHECK(cudaGetDeviceCount(&devices));
  if (devices != kRanks) return 3;

  const uccl::LowLatencyLayout layout_size(nullptr, kTokens, kHidden, kRanks,
                                           kExperts);
  std::array<Rank, kRanks> ranks;
  for (int r = 0; r < kRanks; ++r) {
    CHECK(cudaSetDevice(r));
    int access = 0;
    CHECK(cudaDeviceCanAccessPeer(&access, r, 1 - r));
    if (!access) return 3;
    CHECK(cudaDeviceEnablePeerAccess(1 - r, 0));
    auto& x = ranks[r];
    CHECK(cudaStreamCreateWithFlags(&x.stream, cudaStreamNonBlocking));
    x.rdma = allocate<uint8_t>(layout_size.total_bytes);
    x.atomic = allocate<uint8_t>(kAtomicBufferSize);
    x.workspace = allocate<uint8_t>(NUM_WORKSPACE_BYTES);
    x.peer_rdma = allocate<void*>(kRanks);
    x.input = allocate<bf16>(kTokens * kHidden);
    x.received = allocate<uint8_t>(kRows * kHidden);
    x.scales = allocate<uint8_t>(kRows * kHidden / 16);
    x.token_scale = allocate<float>(kRows);
    x.recv_count = allocate<int>(1);
    x.src_info = allocate<int>(kRows);
    x.ranges = allocate<int64_t>(kRanks);
    x.routes = allocate<int64_t>(kTokens * kTopK);
    x.weights = allocate<float>(kTokens * kTopK);
    x.expert_out = allocate<bf16>(kRows * kHidden);
    x.combined = allocate<bf16>(kTokens * kHidden);
    std::vector<bf16> input(kTokens * kHidden);
    std::vector<int64_t> routes(kTokens * kTopK);
    std::vector<float> weights(kTokens * kTopK, 0.5f);
    for (int i = 0; i < kTokens; ++i) {
      std::fill_n(input.begin() + i * kHidden, kHidden,
                  __float2bfloat16(token_value(r, i)));
      for (int k = 0; k < kTopK; ++k) routes[i * kTopK + k] = k;
    }
    CHECK(cudaMemcpy(x.input, input.data(), input.size() * sizeof(bf16),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(x.routes, routes.data(), routes.size() * sizeof(int64_t),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(x.weights, weights.data(), weights.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
  }
  const std::array<void*, kRanks> peer_rdma{ranks[0].rdma, ranks[1].rdma};
  for (int r = 0; r < kRanks; ++r) {
    CHECK(cudaSetDevice(r));
    CHECK(cudaMemcpy(ranks[r].peer_rdma, peer_rdma.data(), sizeof(peer_rdma),
                     cudaMemcpyHostToDevice));
  }

  bool pass = true;
  for (int mode = 0; mode < 2; ++mode) {
    bool mx = mode == 0;
    int fp4_mode = mx ? 0 : 1;
    for (int r = 0; r < kRanks; ++r) {
      CHECK(cudaSetDevice(r));
      auto& x = ranks[r];
      CHECK(cudaMemset(x.rdma, 0, layout_size.total_bytes));
      CHECK(cudaMemset(x.atomic, 0, kAtomicBufferSize));
      CHECK(cudaMemset(x.workspace, 0, NUM_WORKSPACE_BYTES));
      CHECK(cudaDeviceSynchronize());
    }
    for (int r = 0; r < kRanks; ++r) {
      CHECK(cudaSetDevice(r));
      auto& x = ranks[r];
      uccl::LowLatencyLayout layout(x.rdma, kTokens, kHidden, kRanks, kExperts,
                                    x.atomic);
      auto& current = layout.buffers[0];
      auto [clean, clean_atomic, clean_n] = layout.buffers[1].clean_meta();
      uccl::internode_ll::dispatch(
          x.received, x.scales, x.src_info, x.ranges, x.recv_count, nullptr,
          nullptr, current.dispatch_rdma_recv_data_buffer,
          current.dispatch_rdma_recv_count_buffer,
          current.dispatch_rdma_send_buffer, x.input, x.routes, clean,
          clean_atomic, clean_n, kTokens, kHidden, kTokens, kTopK, kExperts, r,
          kRanks, mx, false, false, x.workspace, 2, x.stream,
          LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE, nullptr, 0, kRanks,
          0, x.peer_rdma, x.rdma, x.atomic,
          current.dispatch_rdma_recv_count_buffer_internode, fp4_mode,
          x.token_scale, mx);
    }
    for (int r = 0; r < kRanks; ++r) {
      CHECK(cudaSetDevice(r));
      auto& x = ranks[r];
      CHECK(cudaStreamSynchronize(x.stream));
      auto recv_count = read(x.recv_count, 1);
      auto metadata = read(x.ranges, kRanks);
      auto tokens = read(x.src_info, kRows);
      auto packed = read(x.received, kRows * kHidden / (mx ? 1 : 2));
      auto scales = read(x.scales, kRows * kHidden / (mx ? 32 : 16));
      auto token_scales = read(x.token_scale, kRows);
      std::vector<bf16> expert(kRows * kHidden);
      size_t checks = 0, bad = recv_count[0] != kRows;
      for (int src = 0; src < kRanks; ++src) {
        int count = static_cast<uint32_t>(metadata[src]);
        int start = static_cast<uint64_t>(metadata[src]) >> 32;
        if (count != kTokens || start < 0 || start > kRows - count) {
          ++bad;
          continue;
        }
        for (int row = start; row < start + count; ++row) {
          if (tokens[row] < 0 || tokens[row] >= kTokens) {
            ++bad;
            continue;
          }
          float expected = token_value(src, tokens[row]);
          if (mx) {
            auto code = expected > 0 ? 0x78 : 0xf8;
            for (int i = 0; i < kHidden; ++i)
              bad += packed[row * kHidden + i] != code;
            for (int i = 0; i < kHidden / 32; ++i)
              bad += scales[row * kHidden / 32 + i] !=
                     std::ilogb(std::abs(expected)) + 119;
          } else {
            auto code = expected > 0 ? 0x77 : 0xff;
            for (int i = 0; i < kHidden / 2; ++i)
              bad += packed[row * kHidden / 2 + i] != code;
            for (int i = 0; i < kHidden / 16; ++i)
              bad += scales[row * kHidden / 16 + i] != 0x7e;
            bad += std::abs(2688.0f * token_scales[row] - std::abs(expected)) >
                   1e-5f;
          }
          checks += kHidden / (mx ? 1 : 2) + kHidden / (mx ? 32 : 16) + !mx;
          std::fill_n(expert.begin() + row * kHidden, kHidden,
                      __float2bfloat16(expected));
        }
      }
      CHECK(cudaMemcpy(x.expert_out, expert.data(),
                       expert.size() * sizeof(bf16), cudaMemcpyHostToDevice));
      std::printf("%s dispatch rank=%d rows=%d checked=%zu mismatches=%zu\n",
                  mx ? "mxfp8" : "nvfp4", r, recv_count[0], checks, bad);
      pass &= bad == 0;
    }
    for (int r = 0; r < kRanks; ++r) {
      CHECK(cudaSetDevice(r));
      auto& x = ranks[r];
      uccl::LowLatencyLayout layout(x.rdma, kTokens, kHidden, kRanks, kExperts,
                                    x.atomic);
      auto& current = layout.buffers[1];
      auto [clean, clean_atomic, clean_n] = layout.buffers[0].clean_meta();
      uccl::internode_ll::combine(
          x.combined, current.combine_rdma_recv_data_buffer,
          current.combine_rdma_recv_flag_buffer,
          current.combine_rdma_send_buffer, x.expert_out, x.routes, x.weights,
          x.src_info, x.ranges, nullptr, clean, clean_atomic, clean_n, kTokens,
          kHidden, kTokens, kTopK, kExperts, r, kRanks, false, x.workspace, 2,
          x.stream, LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE, false,
          nullptr, 0, kRanks, 1, x.peer_rdma, x.rdma, x.atomic,
          current.combine_rdma_recv_flag_buffer_internode, fp4_mode);
    }
    for (int r = 0; r < kRanks; ++r) {
      CHECK(cudaSetDevice(r));
      CHECK(cudaStreamSynchronize(ranks[r].stream));
      auto combined = read(ranks[r].combined, kTokens * kHidden);
      size_t bad = 0;
      for (int i = 0; i < kTokens * kHidden; ++i)
        bad += __bfloat162float(combined[i]) != token_value(r, i / kHidden);
      std::printf("%s combine rank=%d checked=%zu mismatches=%zu\n",
                  mx ? "bf16" : "nvfp4", r, combined.size(), bad);
      pass &= bad == 0;
    }
  }
  return pass ? 0 : 1;
}
