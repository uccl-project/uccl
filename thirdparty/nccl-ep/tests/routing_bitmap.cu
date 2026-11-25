// SPDX-License-Identifier: Apache-2.0
// HT metadata preprocessing must decode the globally packed routing bitmap
// that convert_topk_to_routing_map produces: ceil(total_experts / 8) bytes per
// token, with node sections that may start inside a byte. Every scan output is
// compared with a CPU oracle for each node and local rank on one GPU.
#include "device/hybridep_adapter.cuh"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_OK(call)                                                      \
  do {                                                                     \
    auto status = (call);                                                  \
    if (status != cudaSuccess) {                                           \
      std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); \
      std::exit(2);                                                        \
    }                                                                      \
  } while (0)

namespace {

namespace ht = nccl_ep::hybridep;
constexpr int kTopK = 4;
constexpr uint8_t kUnwritten = 0xa5;

template <typename T>
struct Device {
  T* data = nullptr;
  size_t size;
  explicit Device(size_t size) : size(size) {
    CUDA_OK(cudaMalloc(&data, size * sizeof(T) + 1));
  }
  ~Device() { cudaFree(data); }
  void fill(int byte) { CUDA_OK(cudaMemset(data, byte, size * sizeof(T))); }
  void upload(std::vector<T> const& values) {
    CUDA_OK(cudaMemcpy(data, values.data(), size * sizeof(T),
                       cudaMemcpyHostToDevice));
  }
  std::vector<T> read() const {
    std::vector<T> values(size);
    CUDA_OK(cudaMemcpy(values.data(), data, size * sizeof(T),
                       cudaMemcpyDeviceToHost));
    return values;
  }
};

long checked = 0;

struct Case {
  int nodes, ranks, experts, tokens, node, rank;
  bool counts;
};

template <typename T>
void expect(std::vector<T> const& actual, std::vector<T> const& expected,
            char const* name, Case const& c) {
  for (size_t i = 0; i < expected.size(); ++i, ++checked) {
    if (actual[i] == expected[i]) continue;
    std::fprintf(stderr,
                 "FAIL %s[%zu]: got %lld, want %lld (nodes=%d ranks=%d "
                 "experts_per_rank=%d tokens=%d node_rank=%d local_rank=%d "
                 "counts=%d)\n",
                 name, i, static_cast<long long>(actual[i]),
                 static_cast<long long>(expected[i]), c.nodes, c.ranks,
                 c.experts, c.tokens, c.node, c.rank, c.counts);
    std::exit(1);
  }
}

void check(int nodes, int ranks, int experts, int tokens) {
  int const rows = tokens * ranks * nodes;
  int const per_node = experts * ranks, total = per_node * nodes;
  int const row_bytes = (total + 7) / 8, padded = (tokens + 15) / 16 * 16;

  // Masked slots and fully dropped tokens are part of the routing input.
  std::vector<int64_t> topk(static_cast<size_t>(rows) * kTopK);
  for (int row = 0; row < rows; ++row)
    for (int k = 0; k < kTopK; ++k) {
      uint32_t hash = (row * 2654435761u) ^ (k * 40503u + 977u);
      hash ^= hash >> 13;
      hash *= 0x5bd1e995u;
      hash ^= hash >> 15;
      bool const masked = row % 11 == 3 || hash % 5 == 0;
      topk[row * kTopK + k] = masked ? int64_t{-1} : int64_t{hash % total};
    }
  std::vector<uint8_t> bitmap(static_cast<size_t>(rows) * row_bytes);
  for (int row = 0; row < rows; ++row)
    for (int k = 0; k < kTopK; ++k)
      if (int64_t const e = topk[row * kTopK + k]; e >= 0)
        bitmap[row * row_bytes + e / 8] |= 1u << (e % 8);
  auto bit = [&](int row, int e) {
    return (bitmap[row * row_bytes + e / 8] >> (e % 8)) & 1;
  };
  auto any = [&](int row, int begin, int count) {
    for (int e = begin; e < begin + count; ++e)
      if (bit(row, e)) return true;
    return false;
  };

  // Production converts and all-gathers rank-major rows of this exact width.
  Device<int64_t> device_topk(topk.size());
  Device<uint8_t> device_bitmap(bitmap.size());
  device_topk.upload(topk);
  device_bitmap.fill(0);
  ht::convert_topk_to_routing_map(device_topk.data, device_bitmap.data, rows,
                                  kTopK, row_bytes, nullptr);
  CUDA_OK(cudaGetLastError());
  expect(device_bitmap.read(), bitmap, "producer_bitmap",
         Case{nodes, ranks, experts, tokens, -1, -1, false});

  Device<uint8_t> scan_tmp(ht::get_preprocessing_scan_tmp_size(ranks));
  Device<int32_t> s2d(static_cast<size_t>(nodes) * tokens * ranks);
  Device<uint8_t> rdma_to_attn(static_cast<size_t>(nodes) * padded);
  Device<uint8_t> attn_to_rdma(static_cast<size_t>(tokens) * (nodes - 1));
  Device<uint8_t> rank_mask(rows);
  Device<int32_t> received(1), expert_counts(experts);
  Device<uint8_t> local_routing(static_cast<size_t>(rows) * experts);

  for (int node = 0; node < nodes; ++node)
    for (int rank = 0; rank < ranks; ++rank) {
      std::vector<uint8_t> want_mask(rows),
          want_rdma(rdma_to_attn.size, kUnwritten);
      std::vector<int32_t> want_s2d(s2d.size, -1), want_counts(experts, 0);
      std::vector<uint8_t> want_local(local_routing.size, kUnwritten);
      std::vector<uint8_t> want_attn(attn_to_rdma.size);
      std::vector<int32_t> position(ranks, 0);
      for (int row = 0; row < rows; ++row) {
        int const source_node = row / (tokens * ranks);
        int const source_rank = row / tokens % ranks, token = row % tokens;
        for (int r = 0; r < ranks; ++r)
          want_mask[row] |= any(row, node * per_node + r * experts, experts)
                            << r;
        if (want_mask[row] >> rank & 1) {
          for (int k = 0; k < experts; ++k) {
            int const routed = bit(row, node * per_node + rank * experts + k);
            want_local[position[rank] * experts + k] = routed;
            want_counts[k] += routed;
          }
        }
        for (int r = 0; r < ranks; ++r) {
          bool const needed = want_mask[row] >> r & 1;
          if (source_rank == rank)
            want_s2d[(source_node * tokens + token) * ranks + r] =
                needed ? position[r] : -1;
          position[r] += needed;
        }
        if (source_rank == rank)
          want_rdma[source_node * padded + token] = want_mask[row] != 0;
      }
      int const own = (node * ranks + rank) * tokens;
      for (int token = 0; token < tokens; ++token)
        for (int slot = 0; slot < nodes - 1; ++slot) {
          int const remote = slot < node ? slot : slot + 1;
          want_attn[token * (nodes - 1) + slot] =
              any(own + token, remote * per_node, per_node);
        }

      for (bool counts : {true, false}) {
        s2d.fill(0xff);
        rdma_to_attn.fill(kUnwritten);
        attn_to_rdma.fill(kUnwritten);
        rank_mask.fill(kUnwritten);
        received.fill(kUnwritten);
        local_routing.fill(kUnwritten);
        expert_counts.fill(kUnwritten);
        ht::call_metadata_preprocessing(
            device_bitmap.data, s2d.data,
            reinterpret_cast<bool*>(rdma_to_attn.data),
            reinterpret_cast<bool*>(attn_to_rdma.data), rank_mask.data,
            received.data, reinterpret_cast<bool*>(local_routing.data),
            counts ? expert_counts.data : nullptr, scan_tmp.data, node, rank,
            tokens, /*hidden_dim=*/0, nodes, ranks, experts, nullptr);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
        Case const c{nodes, ranks, experts, tokens, node, rank, counts};
        expect(rank_mask.read(), want_mask, "token_rank_mask", c);
        expect(s2d.read(), want_s2d, "sparse_to_dense_map", c);
        expect(rdma_to_attn.read(), want_rdma, "rdma_to_attn_map", c);
        expect(attn_to_rdma.read(), want_attn, "attn_to_rdma_map", c);
        expect(received.read(), std::vector<int32_t>{position[rank]},
               "num_tokens_for_experts", c);
        expect(local_routing.read(), want_local, "local_expert_routing_map", c);
        if (counts)
          expect(expert_counts.read(), want_counts, "per_expert_token_counts",
                 c);
      }
    }
  std::printf(
      "PASS nodes=%d ranks=%d experts_per_rank=%d tokens_per_rank=%d "
      "row_bytes=%d\n",
      nodes, ranks, experts, tokens, row_bytes);
}

}  // namespace

int main() {
  // Node sections start inside a byte: 2 x 1 x 4 experts, 2 x 4 x 1, 2 x 1 x
  // 12 and 3 x 1 x 4. 2 x 1 x 8 is byte aligned. 37-token tails exercise the
  // 16-token rdma_to_attn padding; the larger counts span several CTAs.
  check(2, 1, 4, 37);
  check(2, 1, 4, 4100);
  check(2, 4, 1, 37);
  check(2, 4, 1, 1031);
  check(2, 1, 12, 300);
  check(3, 1, 4, 37);
  check(3, 1, 4, 2900);
  check(2, 1, 8, 37);
  std::printf("Globally packed routing bitmap decode PASS: %ld entries\n",
              checked);
}
