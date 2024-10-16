#pragma once

#include <glog/logging.h>

#include <functional>

#include "util.h"
#include "util_cb.h"

namespace uccl {

constexpr static uint32_t kNumCachedItemsPerCPU = 8;
extern thread_local CircularBuffer<uint64_t, false, kNumCachedItemsPerCPU>
    th_cache_;

template <bool Sync = false>
class FramePool {
   private:
    CircularBuffer<uint64_t, /* sync = */ false> global_pool_;
    Spin global_spin_;

   public:
    FramePool(uint32_t capacity) : global_pool_(capacity) {}
    void push(uint64_t item) {
        if constexpr (Sync) {
            auto &cache = th_cache_;
            if (unlikely(cache.size() == kNumCachedItemsPerCPU)) {
                global_spin_.Lock();
                auto spin_guard = finally([&]() { global_spin_.Unlock(); });
                for (uint32_t i = 0; i < kNumCachedItemsPerCPU; i++) {
                    uint64_t migrated;
                    CHECK(cache.pop_front(&migrated));
                    CHECK(global_pool_.push_front(migrated));
                }
            }
            CHECK(cache.push_front(item));
        } else {
            CHECK(global_pool_.push_front(item));
        }
    }
    uint64_t pop() {
        if constexpr (Sync) {
            auto &cache = th_cache_;
            if (unlikely(!cache.size())) {
                global_spin_.Lock();
                auto spin_guard = finally([&]() { global_spin_.Unlock(); });
                for (uint32_t i = 0; i < kNumCachedItemsPerCPU; i++) {
                    uint64_t migrated;
                    CHECK(global_pool_.pop_front(&migrated));
                    CHECK(cache.push_front(migrated));
                }
            }
            uint64_t item;
            CHECK(cache.pop_front(&item));
            return item;
        } else {
            uint64_t item;
            CHECK(global_pool_.pop_front(&item));
            return item;
        }
    }
    uint32_t size() {
        if constexpr (Sync) {
            return th_cache_.size() + global_pool_.size();
        } else {
            return global_pool_.size();
        }
    }
};

}  // namespace uccl