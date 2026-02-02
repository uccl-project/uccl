#pragma once
#include <atomic>
#include <condition_variable>
#include <mutex>

// Simple flow control for RDMA send queue
// Limits outstanding work requests to prevent ENOMEM errors
class RDMAFlowControl {
 public:
  explicit RDMAFlowControl(uint32_t max_outstanding = 512)
      : max_outstanding_(max_outstanding), outstanding_count_(0) {}

  // Acquire permission to submit a work request
  // Blocks if limit is reached
  void acquireSlot() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]() {
      return outstanding_count_.load(std::memory_order_acquire) < max_outstanding_;
    });
    outstanding_count_.fetch_add(1, std::memory_order_release);
  }

  // Release a slot after work request completion
  void releaseSlot(uint32_t count = 1) {
    outstanding_count_.fetch_sub(count, std::memory_order_release);
    if (count > 0) {
      std::lock_guard<std::mutex> lock(mutex_);
      cv_.notify_all();
    }
  }

  // Get current outstanding count
  uint32_t getOutstandingCount() const {
    return outstanding_count_.load(std::memory_order_acquire);
  }

 private:
  uint32_t max_outstanding_;
  std::atomic<uint32_t> outstanding_count_;
  std::mutex mutex_;
  std::condition_variable cv_;
};