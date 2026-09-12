#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>

// Byte credits belong to posted WRs, not CQEs: WRITE send completions do not
// report a byte length. A record must be published before posting the WR so a
// concurrent poller cannot return credits before they have been reserved.
class CcSendTracker {
 public:
  explicit CcSendTracker(uint32_t next_id = 0) : next_id_(next_id) {}

  uint32_t record(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Do not replace a live record when the 32-bit identifier wraps.
    while (bytes_.find(next_id_) != bytes_.end()) ++next_id_;
    uint32_t id = next_id_++;
    bytes_.emplace(id, bytes);
    inflight_.fetch_add(bytes, std::memory_order_relaxed);
    return id;
  }

  // Also used to roll back a WR which the provider rejected. Missing records
  // must not acknowledge another message or subtract the same credits twice.
  std::optional<size_t> complete(uint32_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = bytes_.find(id);
    if (it == bytes_.end()) return std::nullopt;
    size_t bytes = it->second;
    bytes_.erase(it);
    inflight_.fetch_sub(bytes, std::memory_order_relaxed);
    return bytes;
  }

  size_t inflight_bytes() const {
    return inflight_.load(std::memory_order_relaxed);
  }

 private:
  std::mutex mutex_;
  std::unordered_map<uint32_t, size_t> bytes_;
  uint32_t next_id_;
  std::atomic<size_t> inflight_{0};
};
