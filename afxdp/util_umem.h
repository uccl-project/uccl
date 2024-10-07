#pragma once

#include <glog/logging.h>

#include <functional>

#include "util.h"
#include "util_cb.h"

namespace uccl {

class FramePool {
   private:
    constexpr static uint32_t kNumCachedItemsPerCPU = 8;

    static thread_local CircularBuffer<uint64_t, false, kNumCachedItemsPerCPU>
        th_cache_;
    CircularBuffer<uint64_t, /* sync = */ false> global_pool_;
    Spin global_spin_;

   public:
    FramePool(uint32_t capacity);
    void push(uint64_t item);
    uint64_t pop();
    uint32_t size();
};

}  // namespace uccl