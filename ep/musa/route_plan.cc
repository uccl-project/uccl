#include "route_plan.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace uccl::musa_ep {
namespace {
constexpr size_t kMissing = std::numeric_limits<size_t>::max();

size_t product(size_t a, size_t b) {
  if (a && b > std::numeric_limits<size_t>::max() / a)
    throw std::overflow_error("EP shape overflow");
  return a * b;
}
}  // namespace

RoutePlan make_plan(size_t rank, size_t ranks, size_t experts, size_t tokens,
                    size_t hidden, size_t topk,
                    std::vector<int64_t> const& indices,
                    std::vector<float> const& weights, size_t max_routes,
                    size_t byte_budget) {
  if (!ranks || rank >= ranks || !experts || experts % ranks ||
      experts > size_t(std::numeric_limits<int64_t>::max()) || !hidden ||
      hidden > 8192 || !topk || topk > 8)
    throw std::invalid_argument("invalid EP topology or shape");
  size_t const lanes = product(tokens, topk);
  if (indices.size() != lanes || weights.size() != lanes)
    throw std::invalid_argument("route/weight shape mismatch");
  size_t count = 0;
  for (size_t i = 0; i < lanes; ++i) {
    if (indices[i] == -1) continue;
    if (indices[i] < 0 || size_t(indices[i]) >= experts ||
        !std::isfinite(weights[i]))
      throw std::invalid_argument("invalid expert index or weight");
    ++count;
  }
  if (count > max_routes) throw std::length_error("EP route capacity exceeded");
  // Reserve source pack, returned rows, staging and output, plus plan metadata.
  // Receive-side capacity must be checked again after the global count
  // exchange.
  size_t remaining = byte_budget;
  for (size_t bytes :
       {product(product(product(count, hidden), sizeof(float)), 3),
        product(product(tokens, hidden), sizeof(float)),
        product(count, sizeof(Route)), product(lanes, sizeof(size_t)),
        product(ranks, 2 * sizeof(size_t))}) {
    if (bytes > remaining) throw std::length_error("EP byte budget exceeded");
    remaining -= bytes;
  }
  std::vector<Route> routes;
  routes.reserve(count);
  for (size_t i = 0; i < lanes; ++i)
    if (indices[i] != -1)
      routes.push_back({rank, i / topk, i % topk, indices[i], weights[i]});
  // Global expert order implies destination rank then local expert order.
  std::stable_sort(routes.begin(), routes.end(),
                   [](auto const& a, auto const& b) {
                     return std::tie(a.expert, a.token, a.lane) <
                            std::tie(b.expert, b.token, b.lane);
                   });
  std::vector<size_t> counts(ranks), offsets(ranks + 1);
  std::vector<size_t> lane_to_route(lanes, kMissing);
  for (size_t i = 0; i < routes.size(); ++i) {
    auto const& route = routes[i];
    ++counts[size_t(route.expert) / (experts / ranks)];
    lane_to_route[route.token * topk + route.lane] = i;
  }
  for (size_t i = 0; i < ranks; ++i) offsets[i + 1] = offsets[i] + counts[i];
  return {tokens,
          hidden,
          topk,
          std::move(routes),
          std::move(counts),
          std::move(offsets),
          std::move(lane_to_route)};
}

std::vector<float> pack_reference(RoutePlan const& plan,
                                  std::vector<float> const& input) {
  if (input.size() != product(plan.tokens, plan.hidden))
    throw std::invalid_argument("input shape mismatch");
  std::vector<float> packed(product(plan.routes.size(), plan.hidden));
  for (size_t row = 0; row < plan.routes.size(); ++row)
    std::copy_n(input.begin() + plan.routes[row].token * plan.hidden,
                plan.hidden, packed.begin() + row * plan.hidden);
  return packed;
}

std::vector<float> combine_reference(RoutePlan const& plan,
                                     std::vector<float> const& expert_output) {
  if (expert_output.size() != product(plan.routes.size(), plan.hidden))
    throw std::invalid_argument("expert output shape mismatch");
  std::vector<float> output(product(plan.tokens, plan.hidden), 0);
  for (size_t token = 0; token < plan.tokens; ++token)
    for (size_t h = 0; h < plan.hidden; ++h)
      for (size_t lane = 0; lane < plan.topk; ++lane) {
        size_t row = plan.lane_to_route[token * plan.topk + lane];
        if (row != kMissing)
          output[token * plan.hidden + h] +=
              plan.routes[row].weight * expert_output[row * plan.hidden + h];
      }
  return output;
}

}  // namespace uccl::musa_ep
