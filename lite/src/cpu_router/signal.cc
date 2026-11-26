#include "signal.hpp"
#include <immintrin.h>
namespace uccl_lite::cpu_router {
void SignalRef::wait(Generation g) const {
  while (acquire() < g) _mm_pause();
}
}  // namespace uccl_lite::cpu_router
