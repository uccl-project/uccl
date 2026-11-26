#include "uccl_lite/runtime/prepared_network.hpp"
#include <iostream>
#include <stdexcept>
using namespace uccl_lite::runtime;
int main() {
  auto check = [](bool h, std::vector<double> a, bool d, std::vector<double> b,
                  NetworkBackend expected) {
    if (chooseNetworkCandidate(h, a, d, b) != expected)
      throw std::runtime_error("network portfolio choice mismatch");
  };
  check(false, {}, false, {}, NetworkBackend::Caller);
  check(true, {4, 5, 6}, false, {}, NetworkBackend::Hierarchy);
  check(false, {}, true, {4, 5, 6}, NetworkBackend::DirectExchange);
  check(true, {4, 5, 6}, true, {2, 3, 4}, NetworkBackend::DirectExchange);
  check(true, {2, 3, 4}, true, {4, 5, 6}, NetworkBackend::Hierarchy);
  check(true, {2, 3, 4}, true, {2, 3, 4}, NetworkBackend::Hierarchy);
  // Historical hierarchy=1 would win incorrectly; fresh checkpoint=10 loses.
  check(true, {9, 10, 11, 9, 10, 11}, true, {4, 5, 6},
        NetworkBackend::DirectExchange);
  bool rejected = false;
  try {
    chooseNetworkCandidate(true, {1, -1, 2}, false, {});
  } catch (std::invalid_argument const&) {
    rejected = true;
  }
  if (!rejected) throw std::runtime_error("invalid admitted evidence accepted");
  std::cout << "PASS network portfolio: qualified-only, current samples, "
               "deterministic ties\n";
}
