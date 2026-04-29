#pragma once

#include <cstdint>

namespace UKernel {
namespace CCL {

struct RingTopology {
  int nranks = 1;

  int next(int rank) const;
  int prev(int rank) const;
  int wrap(int rank) const;
};

}  // namespace CCL
}  // namespace UKernel
