#include "uccl_lite/runtime/collective_backend.hpp"
#include <iostream>
#include <stdexcept>
using namespace uccl_lite;
int main() {
  try {
    runtime::BackendSpec spec{"test", "test", {2, 3}, ScalarType::BFloat16};
    if (spec.ranks() != 5 || spec.elementBytes() != 2)
      throw std::runtime_error("descriptor extents");
    for (int rank = 0; rank < 5; ++rank)
      if (spec.hostForRank(rank) != (rank < 2 ? 0 : 1) ||
          spec.localRank(rank) != (rank < 2 ? rank : rank - 2))
        throw std::runtime_error("heterogeneous host layout mapping");
    if (spec.hostForRank(-1) != -1 || spec.localRank(5) != -1)
      throw std::runtime_error("invalid rank accepted");
    runtime::BackendSpec host{"host", "legacy", {8}, ScalarType::BFloat16,
                              false,  true,     2,   32};
    if (!host.acceptsLayout({3}) || !host.acceptsLayout({32}) ||
        host.acceptsLayout({1}) || host.acceptsLayout({33}) ||
        host.acceptsLayout({4, 4}))
      throw std::runtime_error("variable local rank capability mismatch");
    runtime::BackendSpec general;
    general.minHosts = 1;
    general.maxHosts = 32;
    general.minLocalRanks = 1;
    general.maxLocalRanks = 32;
    general.minTotalRanks = 2;
    general.maxTotalRanks = 32;
    for (auto layout :
         {std::vector<int>{2}, {32}, {1, 1}, {3, 1}, {1, 2, 3}, {8, 8, 8, 8}})
      if (!general.acceptsLayout(layout))
        throw std::runtime_error("general host layout rejected");
    for (auto layout : {std::vector<int>{},
                        {1},
                        {0, 2},
                        {-1, 3},
                        {16, 17},
                        {33},
                        std::vector<int>(33, 1)})
      if (general.acceptsLayout(layout))
        throw std::runtime_error("invalid general extent accepted");
    general.minHosts = 2;
    if (general.acceptsLayout({4}))
      throw std::runtime_error("minimum host bound ignored");
    general.maxTotalRanks = 33;
    if (general.acceptsLayout({2, 2}))
      throw std::runtime_error("invalid capacity descriptor accepted");
    if (!runtime::availableBackends().empty())
      throw std::runtime_error("unlinked provider appeared in registry");
    std::cout << "PASS: runtime layout descriptors are independent of "
                 "four/eight-rank ABI builds\n";
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
