#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace uccl::musa_ep {

struct Route {
  size_t source_rank;
  size_t token;
  size_t lane;
  int64_t expert;
  float weight;
};

// Immutable CPU layout. No GPU, MPI or MCCL dependency is needed to test it.
struct RoutePlan {
  size_t const tokens;
  size_t const hidden;
  size_t const topk;
  std::vector<Route> const routes;
  std::vector<size_t> const send_counts;
  std::vector<size_t> const send_offsets;
  std::vector<size_t> const lane_to_route;
};

RoutePlan make_plan(size_t rank, size_t ranks, size_t experts, size_t tokens,
                    size_t hidden, size_t topk,
                    std::vector<int64_t> const& indices,
                    std::vector<float> const& weights,
                    size_t max_routes = 65536,
                    size_t byte_budget = size_t{1} << 30);

// CPU reference operations, not a claim of GPU/network EP support.
std::vector<float> pack_reference(RoutePlan const& plan,
                                  std::vector<float> const& input);
std::vector<float> combine_reference(RoutePlan const& plan,
                                     std::vector<float> const& expert_output);

}  // namespace uccl::musa_ep
