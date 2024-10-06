#include "transport_umem.h"

namespace uccl {

thread_local CircularBuffer<uint64_t, false, FramePool::kNumCachedItemsPerCPU>
    FramePool::th_cache_;

FramePool::FramePool(uint32_t capacity) : global_pool_(capacity) {}

void FramePool::push(uint64_t item) {
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
}

uint64_t FramePool::pop() {
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
}

uint32_t FramePool::size() { return th_cache_.size() + global_pool_.size(); }

}  // namespace uccl