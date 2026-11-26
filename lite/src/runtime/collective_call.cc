#include "uccl_lite/collective_call.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace uccl_lite {
Workload CollectiveCall::workload() const {
  size_t scalar = dtype == ScalarType::Float32 ? 4 : 2;
  size_t expansion = collective == Collective::AllReduce ? 1 : ports.size();
  if (!count || !expansion ||
      count > std::numeric_limits<size_t>::max() / scalar / expansion)
    throw std::invalid_argument("empty or overflowing collective extent");
  uintptr_t addresses = 0;
  bool complete = !ports.empty();
  for (auto const& port : ports) {
    addresses |= reinterpret_cast<uintptr_t>(port.input) |
                 reinterpret_cast<uintptr_t>(port.output);
    complete &= port.input && port.output;
  }
  unsigned alignment = !complete ? 0
                       : !addresses
                           ? 256
                           : 1U << std::min(8, __builtin_ctzll(addresses));
  return {collective, dtype, count * scalar * expansion, inPlace, alignment};
}

}  // namespace uccl_lite
