#pragma once

#include <glog/logging.h>

#include <deque>
#include <functional>

#include "util.h"
#include "util_cb.h"

namespace uccl {

constexpr static uint32_t kNumCachedItemsPerCPU = 8;

template <typename T, bool Sync = false>
class SharedPool {
    static thread_local inline CircularBuffer<T, false, kNumCachedItemsPerCPU>
        th_cache_;
    CircularBuffer<T, /* sync = */ false> global_pool_;
    Spin global_spin_;

   public:
    SharedPool(uint32_t capacity) : global_pool_(capacity) {}
    void push(T item) {
        if constexpr (Sync) {
            auto &cache = th_cache_;
            if (unlikely(cache.size() == kNumCachedItemsPerCPU)) {
                global_spin_.Lock();
                auto spin_guard = finally([&]() { global_spin_.Unlock(); });
                for (uint32_t i = 0; i < kNumCachedItemsPerCPU; i++) {
                    T migrated;
                    DCHECK(cache.pop_front(&migrated));
                    DCHECK(global_pool_.push_front(migrated));
                }
            }
            DCHECK(cache.push_front(item));
        } else {
            DCHECK(global_pool_.push_front(item));
        }
    }
    T pop() {
        if constexpr (Sync) {
            auto &cache = th_cache_;
            if (unlikely(!cache.size())) {
                global_spin_.Lock();
                auto spin_guard = finally([&]() { global_spin_.Unlock(); });
                for (uint32_t i = 0; i < kNumCachedItemsPerCPU; i++) {
                    T migrated;
                    DCHECK(global_pool_.pop_front(&migrated));
                    DCHECK(cache.push_front(migrated));
                }
            }
            T item;
            DCHECK(cache.pop_front(&item));
            return item;
        } else {
            T item;
            DCHECK(global_pool_.pop_front(&item));
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