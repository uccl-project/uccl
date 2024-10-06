#pragma once

#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>

#include "util.h"

namespace uccl {

template <typename T, bool Sync, uint64_t Capacity = 0>
class CircularBuffer {
   private:
    using FixedArray = T[Capacity + 1];
    constexpr static bool kIsDynamic = (Capacity == 0);

    typename std::conditional<kIsDynamic, std::unique_ptr<T[]>,
                              FixedArray>::type items_;
    uint32_t head_ = 0;
    uint32_t tail_ = 0;
    uint32_t capacity_ = Capacity + 1;
    Spin spin_;

   public:
    CircularBuffer();
    template <bool B = kIsDynamic,
              typename = typename std::enable_if_t<B, void>>
    CircularBuffer(uint32_t size);
    CircularBuffer<T, Sync, Capacity> &operator=(
        CircularBuffer<T, Sync, Capacity> &&other);
    CircularBuffer(CircularBuffer<T, Sync, Capacity> &&other);
    uint32_t capacity() const;
    uint32_t size() const;
    template <typename D>
    bool push_front(D &&d);
    template <typename D>
    bool push_back(D &&d);
    template <typename D>
    std::optional<T> push_back_override(D &&d);
    template <typename D>
    bool pop_front(D *d);
    bool work_steal(CircularBuffer<T, Sync, Capacity> *cb);
    void clear();
    void for_each(const std::function<void(T)> &f);
};

template <typename T, bool Sync, uint64_t Capacity>
CircularBuffer<T, Sync, Capacity>::CircularBuffer() {}

template <typename T, bool Sync, uint64_t Capacity>
template <bool B, typename U>
CircularBuffer<T, Sync, Capacity>::CircularBuffer(uint32_t size) {
    capacity_ = size + 1;
    items_ = std::unique_ptr<T[]>(new T[capacity_]);
}

template <typename T, bool Sync, uint64_t Capacity>
CircularBuffer<T, Sync, Capacity> &CircularBuffer<T, Sync, Capacity>::operator=(
    CircularBuffer<T, Sync, Capacity> &&other) {
    if constexpr (kIsDynamic) {
        items_ = std::move(other.items_);
    } else {
        memcpy(items_, other.items_, sizeof(items_));
    }
    head_ = other.head_;
    tail_ = other.tail_;
    capacity_ = other.capacity_;
    return *this;
}

template <typename T, bool Sync, uint64_t Capacity>
CircularBuffer<T, Sync, Capacity>::CircularBuffer(
    CircularBuffer<T, Sync, Capacity> &&other) {
    *this = std::move(other);
}

template <typename T, bool Sync, uint64_t Capacity>
uint32_t CircularBuffer<T, Sync, Capacity>::capacity() const {
    return capacity_ - 1;
}

template <typename T, bool Sync, uint64_t Capacity>
uint32_t CircularBuffer<T, Sync, Capacity>::size() const {
    uint32_t ret;
    auto tail = load_acquire(&tail_);
    auto head = head_;
    if (tail < head) {
        ret = tail + capacity_ - head;
    } else {
        ret = tail - head;
    }
    return ret;
}

template <typename T, bool Sync, uint64_t Capacity>
template <typename D>
bool CircularBuffer<T, Sync, Capacity>::push_front(D &&d) {
    if constexpr (Sync) {
        spin_.Lock();
    }
    auto head = load_acquire(&head_);
    auto new_head = (head + capacity_ - 1) % capacity_;
    bool success = (new_head != tail_);
    if (likely(success)) {
        items_[new_head] = std::move(d);
        store_release(&head_, new_head);
    }
    if constexpr (Sync) {
        spin_.Unlock();
    }
    return success;
}

template <typename T, bool Sync, uint64_t Capacity>
template <typename D>
bool CircularBuffer<T, Sync, Capacity>::push_back(D &&d) {
    if constexpr (Sync) {
        spin_.Lock();
    }
    auto tail = load_acquire(&tail_);
    auto new_tail = (tail + 1) % capacity_;
    bool success = (new_tail != head_);
    if (likely(success)) {
        items_[tail] = std::move(d);
        store_release(&tail_, new_tail);
    }
    if constexpr (Sync) {
        spin_.Unlock();
    }
    return success;
}

template <typename T, bool Sync, uint64_t Capacity>
template <typename D>
std::optional<T> CircularBuffer<T, Sync, Capacity>::push_back_override(D &&d) {
    std::optional<T> overrided;
    if constexpr (Sync) {
        spin_.Lock();
    }
    auto tail = load_acquire(&tail_);
    auto new_tail = (tail + 1) % capacity_;
    if (unlikely(new_tail == head_)) {
        overrided = std::move(items_[head_]);
        head_ = (head_ + 1) % capacity_;
    }
    items_[tail] = std::move(d);
    store_release(&tail_, new_tail);
    if constexpr (Sync) {
        spin_.Unlock();
    }
    return overrided;
}

template <typename T, bool Sync, uint64_t Capacity>
template <typename D>
bool CircularBuffer<T, Sync, Capacity>::pop_front(D *d) {
    if constexpr (Sync) {
        spin_.Lock();
    }
    auto head = load_acquire(&head_);
    auto tail = tail_;
    bool success = (head != tail);
    if (likely(success)) {
        *d = std::move(items_[head]);
        store_release(&head_, (head + 1) % capacity_);
    }
    if constexpr (Sync) {
        spin_.Unlock();
    }
    return success;
}

template <typename T, bool Sync, uint64_t Capacity>
bool CircularBuffer<T, Sync, Capacity>::work_steal(
    CircularBuffer<T, Sync, Capacity> *cb) {
    static_assert(Sync);
    spin_.Lock();
    auto self_spin_releaser = finally([&]() { spin_.Unlock(); });
    if (!(cb->size() / 2)) {
        return false;
    }
    if (!cb->spin_.TryLock()) {
        return false;
    }
    auto cb_spin_releaser = finally([&]() { cb->spin_.Unlock(); });
    uint32_t cb_size = cb->size();
    if (unlikely(!(cb_size / 2))) {
        return false;
    }
    auto tail = load_acquire(&tail_);
    auto steal_size = std::min(cb_size / 2, capacity() - size());
    for (uint32_t i = 0; i < steal_size; i++) {
        items_[tail] = std::move(cb->items_[cb->head_]);
        cb->head_ = (cb->head_ + 1) % (cb->capacity_);
        tail = (tail + 1) % (capacity_);
        assert(tail < head_ + capacity_);
        assert(cb->head_ != cb->tail_);
    }
    store_release(&tail_, tail);
    return true;
}

template <typename T, bool Sync, uint64_t Capacity>
void CircularBuffer<T, Sync, Capacity>::clear() {
    auto tail = load_acquire(&tail_);
    store_release(&head_, tail);
}

template <typename T, bool Sync, uint64_t Capacity>
void CircularBuffer<T, Sync, Capacity>::for_each(
    const std::function<void(T)> &f) {
    if constexpr (Sync) {
        spin_.Lock();
    }
    auto idx = load_acquire(&head_);
    while (idx != tail_) {
        f(items_[idx]);
        idx = (idx + 1) % capacity_;
    }
    if constexpr (Sync) {
        spin_.Unlock();
    }
}

}  // namespace uccl