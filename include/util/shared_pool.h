#pragma once

#include "cb.h"
#include "util.h"
#include <atomic>
#include <deque>
#include <functional>

namespace uccl {

// Be caseful that only SharedPool with the same T and Sync will share the same
// thread_local cache.
template <typename T, bool Sync = false>
class SharedPool {
  constexpr static uint32_t kNumCachedItemsPerCPU = 64;
  using global_pool_t = CircularBuffer<T, /* sync = */ false>;
  using th_cache_t = CircularBuffer<T, false, kNumCachedItemsPerCPU>;
  static inline std::atomic<bool> shutdown_{false};

  Spin global_spin_;
  global_pool_t global_pool_;
  std::function<void(T)> cleanup_;

  // Adding another class to release the thread cache on destruction.
  class ThreadCache {
    th_cache_t cache_;
    SharedPool* shared_pool_ptr_ = nullptr;

   public:
    ThreadCache() {}
    ~ThreadCache() {
      if (!shared_pool_ptr_) return;

      // global_pool_ has been deleted.
      if (shutdown_) return;

      // Return all items to the global pool when thread exits normally.
      shared_pool_ptr_->global_spin_.Lock();
      auto _ = finally([&]() { shared_pool_ptr_->global_spin_.Unlock(); });
      T item;
      while (cache_.pop_front(&item)) {
        shared_pool_ptr_->global_pool_.push_front(item);
      }
    }
    inline void set_shared_pool_ptr(SharedPool* shared_pool) {
      shared_pool_ptr_ = shared_pool;
    }
    inline bool push_front(T item) { return cache_.push_front(item); }
    inline bool pop_front(T* item) { return cache_.pop_front(item); }
    inline uint32_t size() { return cache_.size(); }
  };
  static thread_local inline ThreadCache th_cache_;

 public:
  SharedPool(uint32_t capacity, std::function<void(T)> cleanup)
      : global_pool_(capacity), cleanup_(cleanup) {}
  ~SharedPool() { shutdown_ = true; }

  void push(T item) {
    cleanup_(item);
    if constexpr (Sync) {
      auto& cache = th_cache_;
      if (unlikely(cache.size() == kNumCachedItemsPerCPU)) {
        cache.set_shared_pool_ptr(this);
        global_spin_.Lock();
        auto _ = finally([&]() { global_spin_.Unlock(); });
        for (uint32_t i = 0; i < kNumCachedItemsPerCPU; i++) {
          T migrated;
          CHECK(cache.pop_front(&migrated));
          CHECK(global_pool_.push_front(migrated));
        }
      }
      CHECK(cache.push_front(item));
    } else {
      CHECK(global_pool_.push_front(item));
    }
  }

  T pop() {
    if constexpr (Sync) {
      auto& cache = th_cache_;
      if (unlikely(!cache.size())) {
        cache.set_shared_pool_ptr(this);
        global_spin_.Lock();
        auto _ = finally([&]() { global_spin_.Unlock(); });
        for (uint32_t i = 0; i < kNumCachedItemsPerCPU; i++) {
          T migrated;
          CHECK(global_pool_.pop_front(&migrated));
          CHECK(cache.push_front(migrated));
        }
      }
      T item;
      CHECK(cache.pop_front(&item));
      return item;
    } else {
      T item;
      CHECK(global_pool_.pop_front(&item));
      return item;
    }
  }

  void flush_th_cache() {
    if constexpr (Sync) {
      auto& cache = th_cache_;
      cache.set_shared_pool_ptr(this);
      global_spin_.Lock();
      auto _ = finally([&]() { global_spin_.Unlock(); });
      T item;
      while (cache.pop_front(&item)) {
        global_pool_.push_front(item);
      }
    }
  }

  uint32_t size() {
    if constexpr (Sync) {
      global_spin_.Lock();
      auto _ = finally([&]() { global_spin_.Unlock(); });
      return th_cache_.size() + global_pool_.size();
    } else {
      return global_pool_.size();
    }
  }
};

}  // namespace uccl
