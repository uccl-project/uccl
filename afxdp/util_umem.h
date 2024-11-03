#pragma once

#include <glog/logging.h>

#include <deque>
#include <functional>

#include "util.h"
#include "util_cb.h"

namespace uccl {

constexpr static uint32_t kNumCachedItemsPerCPU = 8;
extern thread_local CircularBuffer<uint64_t, false, kNumCachedItemsPerCPU>
    th_cache_;

// #define USING_DEQUE

template <bool Sync = false>
class FramePool {
   private:
    CircularBuffer<uint64_t, /* sync = */ false> global_pool_;
    Spin global_spin_;

    std::deque<uint64_t> global_pool2_;

   public:
    FramePool(uint32_t capacity) : global_pool_(capacity) {}
    void push(uint64_t item) {
#ifdef USING_DEQUE
        static_assert(Sync == false);
        global_pool2_.push_back(item);
#else
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
#endif
    }
    uint64_t pop() {
#ifdef USING_DEQUE
        CHECK(!global_pool2_.empty());
        uint64_t item = global_pool2_.front();
        global_pool2_.pop_front();
        return item;
#else
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
#endif
    }
    uint32_t size() {
#ifdef USING_DEQUE
        return global_pool2_.size();
#else
        if constexpr (Sync) {
            return th_cache_.size() + global_pool_.size();
        } else {
            return global_pool_.size();
        }
#endif
    }
};

}  // namespace uccl