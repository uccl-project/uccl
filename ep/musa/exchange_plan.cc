#include "exchange_plan.h"
#include <algorithm>
#include <climits>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace uccl::musa_ep {
size_t checked_product(size_t a, size_t b) {
  if (a && b > std::numeric_limits<size_t>::max() / a)
    throw std::overflow_error("EP product overflow");
  return a * b;
}

size_t checked_sum(size_t a, size_t b) {
  if (b > std::numeric_limits<size_t>::max() - a)
    throw std::overflow_error("EP sum overflow");
  return a + b;
}

ExchangePlan make_exchange_plan(RoutePlan const& plan,
                                std::vector<size_t> const& recv_counts,
                                size_t experts, size_t max_routes,
                                size_t byte_budget) {
  size_t const ranks = plan.send_counts.size();
  if (recv_counts.size() != ranks || !ranks || !experts || experts % ranks)
    throw std::invalid_argument("invalid EP count topology");
  ExchangePlan layout{};
  layout.recv_counts = recv_counts;
  layout.recv_offsets.push_back(0);
  for (size_t peer = 0; peer < ranks; ++peer) {
    layout.recv_offsets.push_back(
        checked_sum(layout.recv_offsets.back(), recv_counts[peer]));
    for (size_t count : {plan.send_counts[peer], recv_counts[peer]})
      if (checked_product(checked_product(count, plan.hidden), sizeof(float)) >
          INT_MAX)
        throw std::length_error("EP peer payload exceeds plugin limit");
    for (auto const& item :
         {std::make_pair(plan.send_counts[peer], plan.send_offsets[peer]),
          std::make_pair(recv_counts[peer], layout.recv_offsets[peer])})
      if (checked_product(checked_sum(item.first, item.second), 4) > INT_MAX)
        throw std::length_error("EP metadata exceeds MPI count limit");
    layout.send_meta_counts.push_back(int(plan.send_counts[peer] * 4));
    layout.send_meta_offsets.push_back(int(plan.send_offsets[peer] * 4));
    layout.recv_meta_counts.push_back(int(recv_counts[peer] * 4));
    layout.recv_meta_offsets.push_back(int(layout.recv_offsets[peer] * 4));
  }
  size_t const sent = plan.routes.size(), received = layout.recv_offsets.back();
  if (received > max_routes)
    throw std::length_error("EP receive route capacity exceeded");
  size_t const lanes = checked_product(plan.tokens, plan.topk);
  std::array<size_t, kBufferCount> const sizes = {
      checked_product(checked_product(sent, plan.hidden), sizeof(float)),
      checked_product(checked_product(received, plan.hidden), sizeof(float)),
      checked_product(checked_product(received, plan.hidden), sizeof(float)),
      checked_product(sent, sizeof(size_t)),
      checked_product(received, sizeof(size_t)),
      checked_product(lanes, sizeof(size_t)),
      checked_product(lanes, sizeof(float))};
  for (size_t i = 0; i < sizes.size(); ++i) {
    layout.arena_bytes = checked_sum(layout.arena_bytes, 255) & ~size_t{255};
    layout.buffers[i] = layout.arena_bytes;
    layout.arena_bytes = checked_sum(layout.arena_bytes, sizes[i]);
  }
  layout.output_bytes =
      checked_product(checked_product(plan.tokens, plan.hidden), sizeof(float));
  // Own GPU storage plus known host arrays and grouping scratch. Caller
  // tensors, allocator overhead and MPI/MCCL runtime storage are not included.
  size_t required = checked_sum(layout.arena_bytes, layout.output_bytes);
  required = checked_sum(required,
                         checked_product(sent, sizeof(Route) + sizeof(size_t) +
                                                   4 * sizeof(int64_t)));
  required = checked_sum(
      required,
      checked_product(received, sizeof(size_t) + 4 * sizeof(int64_t)));
  required = checked_sum(
      required, checked_product(lanes, sizeof(size_t) + sizeof(float)));
  // Plan/layout counts+offsets, MPI counts+offsets, wire counts/shapes,
  // consensus, temporary source shapes and receive-count conversion.
  required = checked_sum(
      required, checked_product(ranks, 7 * sizeof(size_t) + 4 * sizeof(int) +
                                           10 * sizeof(uint64_t)));
  required = checked_sum(required, 2 * sizeof(size_t));
  size_t const largest_peer =
      *std::max_element(recv_counts.begin(), recv_counts.end());
  required = checked_sum(
      required,
      checked_product(largest_peer, sizeof(std::pair<int64_t, int64_t>)));
  required = checked_sum(
      required,
      checked_product(checked_sum(experts / ranks, 1), 2 * sizeof(size_t)));
  if (required > byte_budget)
    throw std::length_error("EP round byte budget exceeded");
  layout.required_bytes = required;
  return layout;
}

std::vector<int64_t> route_metadata(RoutePlan const& plan) {
  std::vector<int64_t> meta;
  meta.reserve(checked_product(plan.routes.size(), 4));
  for (auto const& route : plan.routes)
    for (size_t value :
         {route.source_rank, route.token, route.lane, size_t(route.expert)}) {
      if (value > size_t(std::numeric_limits<int64_t>::max()))
        throw std::overflow_error("EP metadata not representable");
      meta.push_back(int64_t(value));
    }
  return meta;
}

ExpertGrouping group_metadata(
    std::vector<int64_t> const& meta, ExchangePlan const& layout, size_t rank,
    size_t experts,
    std::vector<std::pair<size_t, size_t>> const& source_shapes) {
  size_t const ranks = layout.recv_counts.size();
  if (!ranks || rank >= ranks || !experts || experts % ranks ||
      source_shapes.size() != ranks ||
      meta.size() != checked_product(layout.recv_offsets.back(), 4))
    throw std::invalid_argument("EP metadata shape mismatch");
  size_t const local_experts = experts / ranks;
  ExpertGrouping grouping;
  grouping.expert_offsets.resize(local_experts + 1);
  grouping.expert_to_recv.resize(meta.size() / 4);
  for (size_t peer = 0; peer < ranks; ++peer) {
    std::vector<std::pair<int64_t, int64_t>> seen;
    seen.reserve(layout.recv_counts[peer]);
    for (size_t row = layout.recv_offsets[peer];
         row < layout.recv_offsets[peer + 1]; ++row) {
      auto const* m = meta.data() + row * 4;
      if (m[0] != int64_t(peer) || m[1] < 0 ||
          size_t(m[1]) >= source_shapes[peer].first || m[2] < 0 ||
          size_t(m[2]) >= source_shapes[peer].second || m[3] < 0 ||
          size_t(m[3]) >= experts || size_t(m[3]) / local_experts != rank)
        throw std::invalid_argument("invalid EP route metadata");
      seen.emplace_back(m[1], m[2]);
      ++grouping.expert_offsets[size_t(m[3]) % local_experts + 1];
    }
    std::sort(seen.begin(), seen.end());
    if (std::adjacent_find(seen.begin(), seen.end()) != seen.end())
      throw std::invalid_argument("duplicate EP source token/lane");
  }
  std::partial_sum(grouping.expert_offsets.begin(),
                   grouping.expert_offsets.end(),
                   grouping.expert_offsets.begin());
  auto next = grouping.expert_offsets;
  // Stable counting sort: preserve source segment order within each expert.
  for (size_t row = 0; row < meta.size() / 4; ++row)
    grouping.expert_to_recv[next[size_t(meta[row * 4 + 3]) % local_experts]++] =
        row;
  return grouping;
}
}  // namespace uccl::musa_ep
