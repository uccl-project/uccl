#pragma once
#include <atomic>
#include <bitset>
#include <vector>

class BitmapPacketTracker {
 private:
  static constexpr size_t WINDOW_SIZE = 65536;
  std::vector<bool> ack_bitmap_;
  uint32_t base_seq_num_;
  uint32_t next_seq_num_;

 public:
  BitmapPacketTracker(uint32_t initial_seq = 0)
      : ack_bitmap_(WINDOW_SIZE, false),
        base_seq_num_(initial_seq),
        next_seq_num_(initial_seq) {}

  uint32_t sendPacket() {
    uint32_t seq_num = next_seq_num_++;
    if (seq_num - base_seq_num_ >= WINDOW_SIZE) {
      slideWindow();
    }
    ack_bitmap_[seq_num % WINDOW_SIZE] = false;
    return seq_num;
  }

  void acknowledge(uint32_t seq_num) {
    if (seq_num >= base_seq_num_ && seq_num - base_seq_num_ < WINDOW_SIZE) {
      ack_bitmap_[seq_num % WINDOW_SIZE] = true;
      slideWindow();
    }
  }

  bool isAcknowledged(uint32_t seq_num) const {
    if (seq_num < base_seq_num_ || seq_num - base_seq_num_ >= WINDOW_SIZE) {
      return true;
    }
    return ack_bitmap_[seq_num % WINDOW_SIZE];
  }

  std::vector<uint32_t> getUnacknowledgedPackets() const {
    std::vector<uint32_t> unacked;
    for (uint32_t i = 0; i < WINDOW_SIZE; ++i) {
      uint32_t seq_num = base_seq_num_ + i;
      if (seq_num < next_seq_num_ && !ack_bitmap_[i]) {
        unacked.push_back(seq_num);
      }
    }
    return unacked;
  }

 private:
  void slideWindow() {
    while (base_seq_num_ < next_seq_num_ &&
           ack_bitmap_[base_seq_num_ % WINDOW_SIZE]) {
      ack_bitmap_[base_seq_num_ % WINDOW_SIZE] = false;
      base_seq_num_++;
    }
  }
};

class AtomicBitmapPacketTracker {
 private:
  static constexpr size_t WINDOW_SIZE = 65536;
  std::vector<std::atomic<bool>> ack_bitmap_;
  std::vector<std::atomic<size_t>> packet_sizes_;  // Store packet sizes
  std::atomic<uint32_t> base_seq_num_;
  std::atomic<uint32_t> next_seq_num_;

 public:
  AtomicBitmapPacketTracker(uint32_t initial_seq = 0)
      : ack_bitmap_(WINDOW_SIZE),
        packet_sizes_(WINDOW_SIZE),
        base_seq_num_(initial_seq),
        next_seq_num_(initial_seq) {
    for (size_t i = 0; i < WINDOW_SIZE; ++i) {
      ack_bitmap_[i].store(false, std::memory_order_relaxed);
      packet_sizes_[i].store(0, std::memory_order_relaxed);
    }
  }

  uint32_t sendPacket(size_t packet_size = 0) {
    uint32_t seq_num = next_seq_num_.fetch_add(1, std::memory_order_acq_rel);
    uint32_t base = base_seq_num_.load(std::memory_order_acquire);
    if (seq_num - base >= WINDOW_SIZE) {
      slideWindow();
    }
    ack_bitmap_[seq_num % WINDOW_SIZE].store(false, std::memory_order_release);
    packet_sizes_[seq_num % WINDOW_SIZE].store(packet_size,
                                               std::memory_order_release);
    return seq_num;
  }

  void acknowledge(uint32_t seq_num) {
    uint32_t base = base_seq_num_.load(std::memory_order_acquire);
    if (seq_num >= base && seq_num - base < WINDOW_SIZE) {
      ack_bitmap_[seq_num % WINDOW_SIZE].store(true, std::memory_order_release);
      slideWindow();
    }
  }

  bool isAcknowledged(uint32_t seq_num) const {
    uint32_t base = base_seq_num_.load(std::memory_order_acquire);
    if (seq_num < base || seq_num - base >= WINDOW_SIZE) {
      return true;
    }
    return ack_bitmap_[seq_num % WINDOW_SIZE].load(std::memory_order_acquire);
  }

  std::vector<uint32_t> getUnacknowledgedPackets() const {
    std::vector<uint32_t> unacked;
    uint32_t base = base_seq_num_.load(std::memory_order_acquire);
    uint32_t next = next_seq_num_.load(std::memory_order_acquire);
    for (uint32_t i = 0; i < WINDOW_SIZE; ++i) {
      uint32_t seq_num = base + i;
      if (seq_num < next &&
          !ack_bitmap_[seq_num % WINDOW_SIZE].load(std::memory_order_acquire)) {
        unacked.push_back(seq_num);
      }
    }
    return unacked;
  }

  // Get the number of inflight (unacknowledged) packets
  uint32_t getInflightCount() const {
    uint32_t count = 0;
    uint32_t base = base_seq_num_.load(std::memory_order_acquire);
    uint32_t next = next_seq_num_.load(std::memory_order_acquire);
    for (uint32_t i = 0; i < WINDOW_SIZE; ++i) {
      uint32_t seq_num = base + i;
      if (seq_num < next &&
          !ack_bitmap_[seq_num % WINDOW_SIZE].load(std::memory_order_acquire)) {
        count++;
      }
    }
    return count;
  }

  // Get the size of a specific packet
  size_t getPacketSize(uint32_t seq_num) const {
    uint32_t base = base_seq_num_.load(std::memory_order_acquire);
    if (seq_num < base || seq_num - base >= WINDOW_SIZE) {
      return 0;  // Packet is outside the window
    }
    return packet_sizes_[seq_num % WINDOW_SIZE].load(std::memory_order_acquire);
  }

  // Get the total bytes of inflight (unacknowledged) packets
  size_t getTotalInflightBytes() const {
    size_t total_bytes = 0;
    uint32_t base = base_seq_num_.load(std::memory_order_acquire);
    uint32_t next = next_seq_num_.load(std::memory_order_acquire);
    for (uint32_t i = 0; i < WINDOW_SIZE; ++i) {
      uint32_t seq_num = base + i;
      if (seq_num < next &&
          !ack_bitmap_[seq_num % WINDOW_SIZE].load(std::memory_order_acquire)) {
        total_bytes += packet_sizes_[seq_num % WINDOW_SIZE].load(
            std::memory_order_acquire);
      }
    }
    return total_bytes;
  }

 private:
  void slideWindow() {
    uint32_t base = base_seq_num_.load(std::memory_order_acquire);
    uint32_t next = next_seq_num_.load(std::memory_order_acquire);

    while (base < next &&
           ack_bitmap_[base % WINDOW_SIZE].load(std::memory_order_acquire)) {
      ack_bitmap_[base % WINDOW_SIZE].store(false, std::memory_order_release);
      packet_sizes_[base % WINDOW_SIZE].store(0, std::memory_order_release);
      // Try to advance base_seq_num_ atomically
      if (base_seq_num_.compare_exchange_weak(base, base + 1,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
        base++;
      } else {
        // Another thread advanced it, reload and continue
        base = base_seq_num_.load(std::memory_order_acquire);
      }
      next = next_seq_num_.load(std::memory_order_acquire);
    }
  }
};

class AtomicBitmapPacketTrackerMultiAck {
 private:
  static constexpr size_t WINDOW_SIZE = 65536;

  // Whether fully acknowledged (true means seq finished)
  std::vector<std::atomic<bool>> ack_bitmap_;

  // Expected ack count per seq_num
  std::vector<std::atomic<uint32_t>> expected_ack_count_;

  // Current ack count per seq_num
  std::vector<std::atomic<uint32_t>> current_ack_count_;

  // Packet sizes (for inflight bytes calculation)
  std::vector<std::atomic<size_t>> packet_sizes_;

  std::atomic<uint32_t> base_seq_num_;
  std::atomic<uint32_t> next_seq_num_;

 public:
  AtomicBitmapPacketTrackerMultiAck(uint32_t initial_seq = 0)
      : ack_bitmap_(WINDOW_SIZE),
        expected_ack_count_(WINDOW_SIZE),
        current_ack_count_(WINDOW_SIZE),
        packet_sizes_(WINDOW_SIZE),
        base_seq_num_(initial_seq),
        next_seq_num_(initial_seq) {
    for (size_t i = 0; i < WINDOW_SIZE; ++i) {
      ack_bitmap_[i].store(false, std::memory_order_relaxed);
      expected_ack_count_[i].store(1, std::memory_order_relaxed);
      current_ack_count_[i].store(0, std::memory_order_relaxed);
      packet_sizes_[i].store(0, std::memory_order_relaxed);
    }
  }

  // Send a packet with expected number of ACKs required
  uint32_t sendPacket(size_t packet_size = 0, uint32_t expected_ack = 1) {
    uint32_t seq_num = next_seq_num_.fetch_add(1, std::memory_order_acq_rel);
    uint32_t base = base_seq_num_.load(std::memory_order_acquire);

    if (seq_num - base >= WINDOW_SIZE) {
      slideWindow();
    }

    size_t pos = seq_num % WINDOW_SIZE;

    ack_bitmap_[pos].store(false, std::memory_order_release);
    expected_ack_count_[pos].store(expected_ack, std::memory_order_release);
    current_ack_count_[pos].store(0, std::memory_order_release);
    packet_sizes_[pos].store(packet_size, std::memory_order_release);

    return seq_num;
  }

  // Acknowledge one “sub-ACK”
  void acknowledge(uint32_t seq_num) {
    uint32_t base = base_seq_num_.load(std::memory_order_acquire);
    if (seq_num < base || seq_num - base >= WINDOW_SIZE) return;

    size_t pos = seq_num % WINDOW_SIZE;

    uint32_t cur =
        current_ack_count_[pos].fetch_add(1, std::memory_order_acq_rel) + 1;

    uint32_t need = expected_ack_count_[pos].load(std::memory_order_acquire);

    if (cur >= need) {
      // Fully acknowledged
      ack_bitmap_[pos].store(true, std::memory_order_release);
      slideWindow();
    }
  }

  bool updateExpectedAckCount(uint32_t seq_num, uint32_t new_expected_ack) {
    uint32_t base = base_seq_num_.load(std::memory_order_acquire);
    uint32_t next = next_seq_num_.load(std::memory_order_acquire);

    // If seq outside window → cannot update
    if (seq_num < base || seq_num >= next || seq_num - base >= WINDOW_SIZE) {
      return false;
    }

    size_t pos = seq_num % WINDOW_SIZE;

    // Update expected ack count
    expected_ack_count_[pos].store(new_expected_ack, std::memory_order_release);

    // If new expected count is already satisfied, mark as acked
    uint32_t cur = current_ack_count_[pos].load(std::memory_order_acquire);

    if (cur >= new_expected_ack) {
      ack_bitmap_[pos].store(true, std::memory_order_release);
      slideWindow();
    }

    return true;
  }

  // Query if fully acked
  bool isAcknowledged(uint32_t seq_num) const {
    uint32_t base = base_seq_num_.load(std::memory_order_acquire);

    if (seq_num < base || seq_num - base >= WINDOW_SIZE) {
      return true;
    }

    return ack_bitmap_[seq_num % WINDOW_SIZE].load(std::memory_order_acquire);
  }

  std::vector<uint32_t> getUnacknowledgedPackets() const {
    std::vector<uint32_t> unacked;
    uint32_t base = base_seq_num_.load(std::memory_order_acquire);
    uint32_t next = next_seq_num_.load(std::memory_order_acquire);

    for (uint32_t i = 0; i < WINDOW_SIZE; ++i) {
      uint32_t seq_num = base + i;
      if (seq_num < next &&
          !ack_bitmap_[seq_num % WINDOW_SIZE].load(std::memory_order_acquire)) {
        unacked.push_back(seq_num);
      }
    }
    return unacked;
  }

  uint32_t getInflightCount() const {
    uint32_t count = 0;
    uint32_t base = base_seq_num_.load(std::memory_order_acquire);
    uint32_t next = next_seq_num_.load(std::memory_order_acquire);

    for (uint32_t i = 0; i < WINDOW_SIZE; ++i) {
      uint32_t seq_num = base + i;
      if (seq_num < next &&
          !ack_bitmap_[seq_num % WINDOW_SIZE].load(std::memory_order_acquire)) {
        count++;
      }
    }
    return count;
  }

  size_t getPacketSize(uint32_t seq_num) const {
    uint32_t base = base_seq_num_.load(std::memory_order_acquire);
    if (seq_num < base || seq_num - base >= WINDOW_SIZE) {
      return 0;
    }
    return packet_sizes_[seq_num % WINDOW_SIZE].load(std::memory_order_acquire);
  }

  size_t getTotalInflightBytes() const {
    size_t total = 0;
    uint32_t base = base_seq_num_.load(std::memory_order_acquire);
    uint32_t next = next_seq_num_.load(std::memory_order_acquire);

    for (uint32_t i = 0; i < WINDOW_SIZE; ++i) {
      uint32_t seq_num = base + i;
      if (seq_num < next &&
          !ack_bitmap_[seq_num % WINDOW_SIZE].load(std::memory_order_acquire)) {
        total += packet_sizes_[seq_num % WINDOW_SIZE].load(
            std::memory_order_acquire);
      }
    }
    return total;
  }

 private:
  void slideWindow() {
    uint32_t base = base_seq_num_.load(std::memory_order_acquire);
    uint32_t next = next_seq_num_.load(std::memory_order_acquire);

    while (base < next) {
      size_t pos = base % WINDOW_SIZE;

      if (!ack_bitmap_[pos].load(std::memory_order_acquire)) break;

      // Clear entry
      ack_bitmap_[pos].store(false, std::memory_order_release);
      expected_ack_count_[pos].store(1, std::memory_order_release);
      current_ack_count_[pos].store(0, std::memory_order_release);
      packet_sizes_[pos].store(0, std::memory_order_release);

      // Advance base atomically
      uint32_t new_base = base + 1;
      if (base_seq_num_.compare_exchange_weak(base, new_base,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
        base = new_base;
      } else {
        base = base_seq_num_.load(std::memory_order_acquire);
      }

      next = next_seq_num_.load(std::memory_order_acquire);
    }
  }
};
