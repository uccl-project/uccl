#include "uccl_lite/runtime/prepared_network.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace uccl_lite::runtime {
NetworkBackend chooseNetworkCandidate(
    bool hierarchyQualified, std::vector<double> const& hierarchyCurrent,
    bool directQualified, std::vector<double> const& directCurrent) {
  auto median = [](std::vector<double> values) {
    if (values.size() < 3 ||
        std::any_of(values.begin(), values.end(),
                    [](double x) { return !std::isfinite(x) || x <= 0; }))
      throw std::invalid_argument(
          "qualified network candidate has invalid current samples");
    std::sort(values.begin(), values.end());
    auto n = values.size();
    return n % 2 ? values[n / 2] : values[n / 2 - 1] / 2 + values[n / 2] / 2;
  };
  double hierarchy = hierarchyQualified ? median(hierarchyCurrent) : INFINITY;
  double direct = directQualified ? median(directCurrent) : INFINITY;
  if (directQualified && direct < hierarchy)
    return NetworkBackend::DirectExchange;
  return hierarchyQualified ? NetworkBackend::Hierarchy
                            : NetworkBackend::Caller;
}
}  // namespace uccl_lite::runtime
