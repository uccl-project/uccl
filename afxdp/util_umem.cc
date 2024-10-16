#include "util_umem.h"

namespace uccl {

thread_local CircularBuffer<uint64_t, false, kNumCachedItemsPerCPU> th_cache_;

}  // namespace uccl