#pragma once

#include <glog/logging.h>

#include <deque>
#include <functional>

#include "util.h"
#include "util_cb.h"

namespace uccl {

// Be caseful that only SharedPool with the same T and Sync will share the same
// thread_local cache.
template <typename T, bool Sync = false>
class SharedPool {
    constexpr static uint32_t kNumCachedItemsPerCPU = 8;
    using global_pool_t = CircularBuffer<T, /* sync = */ false>;
    using th_cache_t = CircularBuffer<T, false, kNumCachedItemsPerCPU>;

    // Adding another class to release the thread cache on destruction.
    class ThreadCache {
        th_cache_t cache_;
        global_pool_t *global_pool_ptr_ = nullptr;

       public:
        ThreadCache() {}
        ~ThreadCache() {
            if (!global_pool_ptr_) return;
            T item;
            while (cache_.pop_front(&item)) {
                global_pool_ptr_->push_front(item);
            }
        }
        inline void set_global_pool_ptr(global_pool_t &global_pool) {
            global_pool_ptr_ = &global_pool;
        }
        inline bool push_front(T item) { return cache_.push_front(item); }
        inline bool pop_front(T *item) { return cache_.pop_front(item); }
        inline uint32_t size() { return cache_.size(); }
    };

    Spin global_spin_;
    global_pool_t global_pool_;
    static thread_local inline ThreadCache th_cache_;

   public:
    SharedPool(uint32_t capacity) : global_pool_(capacity) {}
    void push(T item) {
        if constexpr (Sync) {
            auto &cache = th_cache_;
            if (unlikely(cache.size() == kNumCachedItemsPerCPU)) {
                global_spin_.Lock();
                auto spin_guard = finally([&]() { global_spin_.Unlock(); });
                cache.set_global_pool_ptr(global_pool_);
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
                cache.set_global_pool_ptr(global_pool_);
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