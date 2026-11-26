#include "route_plan.h"
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

using namespace uccl::musa_ep;

void rejects(std::function<void()> action) {
  bool rejected = false;
  try {
    action();
  } catch (std::exception const&) {
    rejected = true;
  }
  assert(rejected);
}

int main() {
  auto plan =
      make_plan(0, 2, 4, 2, 2, 3, {3, 0, 3, -1, 1, -1}, {1, 2, -1, 0, .5, 0});
  assert((plan.send_counts == std::vector<size_t>{2, 2}));
  assert((plan.send_offsets == std::vector<size_t>{0, 2, 4}));
  auto packed = pack_reference(plan, {2, 4, 6, 8});
  // Distinct expert functions catch a correct roundtrip through wrong experts.
  for (size_t row = 0; row < plan.routes.size(); ++row)
    for (size_t h = 0; h < plan.hidden; ++h)
      packed[row * plan.hidden + h] += plan.routes[row].expert + 1;
  auto combined = combine_reference(plan, packed);
  assert((combined == std::vector<float>{6, 10, 4, 5}));
  assert(combine_reference(make_plan(0, 2, 4, 0, 1, 1, {}, {}), {}).empty());
  auto empty = make_plan(0, 2, 4, 2, 1, 1, {-1, -1}, {NAN, NAN});
  assert((combine_reference(empty, {}) == std::vector<float>{0, 0}));
  rejects([] { make_plan(0, 2, 3, 1, 1, 1, {0}, {1}); });
  rejects([] { make_plan(0, 2, 4, 1, 1, 1, {4}, {1}); });
  rejects([] { make_plan(0, 2, 4, 1, 1, 1, {-2}, {1}); });
  rejects([] { make_plan(0, 2, 4, 1, 1, 1, {0}, {NAN}); });
  rejects([] { make_plan(0, 2, 4, 1, 1, 1, {0}, {1}, 0); });
  rejects([] { make_plan(0, 2, 4, 1, 1, 1, {0}, {1}, 10, 1); });
  rejects([&] { pack_reference(plan, {}); });
  rejects([&] { combine_reference(plan, {}); });
  rejects([] {
    make_plan(0, 2, 4, std::numeric_limits<size_t>::max(), 8192, 8, {}, {});
  });

  std::mt19937 random(17);
  for (int seed = 0; seed < 100; ++seed) {
    std::vector<int64_t> idx(17 * 8);
    std::vector<float> weights(idx.size()), x(17 * 3);
    for (size_t i = 0; i < idx.size(); ++i) {
      idx[i] = int(random() % 5) - 1;
      weights[i] = float(int(random() % 5) - 2) / 2;
    }
    for (auto& value : x) value = int(random() % 17) - 8;
    auto p = make_plan(seed % 2, 2, 4, 17, 3, 8, idx, weights);
    auto expert = pack_reference(p, x);
    for (size_t row = 0; row < p.routes.size(); ++row)
      for (size_t h = 0; h < 3; ++h)
        expert[row * 3 + h] =
            expert[row * 3 + h] * (p.routes[row].expert + 1) + .5f;
    auto y = combine_reference(p, expert);
    for (size_t t = 0; t < 17; ++t)
      for (size_t h = 0; h < 3; ++h) {
        double reference = 0;
        // Oracle uses original lanes, never the production route permutation.
        for (size_t k = 0; k < 8; ++k)
          if (idx[t * 8 + k] != -1)
            reference += weights[t * 8 + k] *
                         (double(x[t * 3 + h]) * (idx[t * 8 + k] + 1) + .5);
        assert(std::abs(y[t * 3 + h] - reference) <=
               1e-5 + 1e-4 * std::abs(reference));
      }
  }
  std::cout << "EP CPU routing/oracle: PASS (100 seeds)\n";
}
