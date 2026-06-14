#include "topology.h"

namespace UKernel {
namespace CCL {

int RingTopology::wrap(int rank) const {
  if (nranks <= 0) return 0;
  int value = rank % nranks;
  return value < 0 ? value + nranks : value;
}

int RingTopology::next(int rank) const { return wrap(rank + 1); }

int RingTopology::prev(int rank) const { return wrap(rank - 1); }

}  // namespace CCL
}  // namespace UKernel
