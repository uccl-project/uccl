#pragma once

#include "route_plan.h"
#include <array>
#include <utility>

namespace uccl::musa_ep {

static_assert(sizeof(size_t) == sizeof(uint64_t),
              "MUSA EP device maps require a 64-bit host");

size_t checked_product(size_t a, size_t b);
size_t checked_sum(size_t a, size_t b);

// Byte offsets in one reusable, aligned GPU allocation.
enum BufferIndex {
  kSend,
  kRecv,
  kExpert,
  kTokens,
  kPermutation,
  kLanes,
  kWeights,
  kBufferCount
};
struct ExchangePlan {
  std::vector<size_t> recv_counts, recv_offsets;
  std::vector<int> send_meta_counts, send_meta_offsets;
  std::vector<int> recv_meta_counts, recv_meta_offsets;
  std::array<size_t, kBufferCount> buffers;
  size_t arena_bytes, output_bytes, required_bytes;
};

struct ExpertGrouping {
  // Gather recv rows into expert order; scatter with the same map to invert.
  std::vector<size_t> expert_to_recv, expert_offsets;
};

ExchangePlan make_exchange_plan(RoutePlan const& plan,
                                std::vector<size_t> const& recv_counts,
                                size_t experts, size_t max_routes,
                                size_t byte_budget);
std::vector<int64_t> route_metadata(RoutePlan const& plan);
ExpertGrouping group_metadata(
    std::vector<int64_t> const& meta, ExchangePlan const& layout, size_t rank,
    size_t experts,
    std::vector<std::pair<size_t, size_t>> const& source_shapes);

}  // namespace uccl::musa_ep
