#pragma once

#include <cmath>

namespace uccl::musa_ep::test {
inline bool matches(float actual, double expected, bool exact = false) {
  if (!std::isfinite(actual) || !std::isfinite(expected)) return false;
  double const error = std::abs(double(actual) - expected);
  return std::isfinite(error) &&
         (exact ? error == 0 : error <= 1e-5 + 1e-4 * std::abs(expected));
}
}  // namespace uccl::musa_ep::test
