#include "exchange_plan.h"
#include "kernel_math.h"
#include "numeric.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace uccl::musa_ep;

template <class F>
void rejects(F f) {
  bool caught = false;
  try {
    f();
  } catch (std::exception const&) {
    caught = true;
  }
  assert(caught);
}

int main() {
  assert(!test::matches(std::numeric_limits<float>::quiet_NaN(), 1));
  assert(!test::matches(std::numeric_limits<float>::infinity(), 1));
  assert(!test::matches(1, std::numeric_limits<double>::quiet_NaN()));
  assert(!test::matches(1.1f, 1));
  assert(test::matches(1.00001f, 1));
  assert(!test::matches(1.00001f, 1, true));
  assert(test::matches(1, 1, true));
  auto plan =
      make_plan(0, 2, 4, 2, 3, 3, {3, 0, 3, -1, 1, 2}, {0.5f, 1, -1, 0, 2, 0});
  auto layout = make_exchange_plan(plan, {2, 3}, 4, 65536, 1 << 20);
  make_exchange_plan(plan, {2, 3}, 4, 65536, layout.required_bytes);
  rejects([&] {
    make_exchange_plan(plan, {2, 3}, 4, 65536, layout.required_bytes - 1);
  });
  assert((layout.recv_offsets == std::vector<size_t>{0, 2, 5}));
  std::vector<int64_t> meta = {0, 1, 1, 1, 0, 0, 1, 0, 1, 1,
                               2, 1, 1, 0, 0, 0, 1, 1, 0, 0};
  auto grouping = group_metadata(meta, layout, 0, 4, {{2, 3}, {2, 3}});
  assert((grouping.expert_to_recv == std::vector<size_t>{1, 3, 4, 0, 2}));
  assert((grouping.expert_offsets == std::vector<size_t>{0, 3, 5}));
  auto bad = meta;
  bad[0] = 1;  // Source must match the segment, not merely be a valid rank.
  rejects([&] { group_metadata(bad, layout, 0, 4, {{2, 3}, {2, 3}}); });
  bad = meta;
  bad[3] = 2;  // Expert belongs to the wrong rank.
  rejects([&] { group_metadata(bad, layout, 0, 4, {{2, 3}, {2, 3}}); });
  bad = meta;
  bad[1] = 2;
  rejects([&] { group_metadata(bad, layout, 0, 4, {{2, 3}, {2, 3}}); });
  bad = meta;
  std::copy_n(bad.begin(), 4, bad.begin() + 4);
  rejects([&] { group_metadata(bad, layout, 0, 4, {{2, 3}, {2, 3}}); });
  rejects([&] { make_exchange_plan(plan, {2, 3}, 4, 4, 1 << 20); });
  rejects([&] { make_exchange_plan(plan, {2, 3}, 4, 65536, 1); });
  rejects([&] {
    make_exchange_plan(plan, {std::numeric_limits<size_t>::max(), 1}, 4,
                       std::numeric_limits<size_t>::max(), 1 << 20);
  });
  auto invalid_lanes =
      make_plan(0, 2, 4, 1000, 1, 8, std::vector<int64_t>(8000, -1),
                std::vector<float>(8000), 65536, 1 << 20);
  rejects([&] {
    make_exchange_plan(invalid_lanes, {0, 0}, 4, 65536, 128 << 10);
  });
  std::vector<float> x = {1, 2, 3, 4, 5, 6};
  auto packed = pack_reference(plan, x);
  std::vector<size_t> tokens;
  std::vector<float> weights(plan.tokens * plan.topk);
  for (auto const& route : plan.routes) {
    tokens.push_back(route.token);
    weights[route.token * plan.topk + route.lane] = route.weight;
  }
  for (size_t i = 0; i < packed.size(); ++i)
    assert(x[gather_index(i, plan.hidden, tokens.data())] == packed[i]);
  auto combined = combine_reference(plan, packed);
  for (size_t i = 0; i < combined.size(); ++i)
    assert(combine_element(packed.data(), plan.lane_to_route.data(),
                           weights.data(), i, plan.hidden,
                           plan.topk) == combined[i]);
  std::cout << "EP layout/metadata/kernel indexing: PASS\n";
}
