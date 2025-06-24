#ifndef UTIL_BUFFPOOL_H
#define UTIL_BUFFPOOL_H

#include <infiniband/verbs.h>
#include <cstdint>
#include <stdexcept>
#include <sys/mman.h>

/**
 * @brief A buffer pool with the following properties:
 * - Constructed with a memory region provided by the caller or mmap by itself.
 * - Not thread-safe, single producer, single consumer.
 * - Fixed size elements.
 * - Size must be power of 2.
 * - Actual size is num_elements - 1.
 */
class BuffPool {
 public:
  BuffPool(uint32_t num_elements, size_t element_size,
           struct ibv_mr* mr = nullptr,
           void (*init_cb)(uint64_t buff) = nullptr)
      : num_elements_(num_elements), element_size_(element_size), mr_(mr) {
    if (mr_) {
      base_addr_ = mr->addr;
    } else {
      base_addr_ =
          mmap(nullptr, num_elements_ * element_size_, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      UCCL_INIT_CHECK(base_addr_ != MAP_FAILED,
                      "Failed to allocate memory for BuffPool");
    }
    buffer_pool_ = new uint64_t[num_elements_];
    head_ = tail_ = 0;
    // Reserve one element for distinguished empty/full state.
    for (uint32_t i = 0; i < num_elements_ - 1; i++) {
      if (init_cb) init_cb((uint64_t)base_addr_ + i * element_size_);
      free_buff((uint64_t)base_addr_ + i * element_size_);
    }
  }

  ~BuffPool() {
    if (!mr_) {
      munmap(base_addr_, num_elements_ * element_size_);
    }
    delete[] buffer_pool_;
  }

  inline bool full(void) {
    return ((tail_ + 1) & (num_elements_ - 1)) == head_;
  }

  inline bool empty(void) { return head_ == tail_; }

  inline uint32_t size(void) { return (tail_ - head_) & (num_elements_ - 1); }

  inline uint32_t get_lkey(void) {
    if (!mr_) return 0;
    return mr_->lkey;
  }

  inline int alloc_buff(uint64_t* buff_addr) {
    if (empty()) return -1;

    *buff_addr = (uint64_t)base_addr_ + buffer_pool_[head_];
    head_ = (head_ + 1) & (num_elements_ - 1);
    return 0;
  }

  inline void free_buff(uint64_t buff_addr) {
    if (full()) return;
    buff_addr -= (uint64_t)base_addr_;
    buffer_pool_[tail_] = buff_addr;
    tail_ = (tail_ + 1) & (num_elements_ - 1);
  }

 protected:
  void* base_addr_;
  uint32_t head_;
  uint32_t tail_;
  uint32_t num_elements_;
  size_t element_size_;
  struct ibv_mr* mr_;
  uint64_t* buffer_pool_;
};

#endif