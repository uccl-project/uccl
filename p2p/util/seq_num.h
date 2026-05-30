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
  BitmapPacketTracker(uint32_t initial_seq = 0);

  uint32_t send_packet();

  void acknowledge(uint32_t seq_num);

  bool is_acknowledged(uint32_t seq_num) const;

  std::vector<uint32_t> get_unacknowledged_packets() const;

 private:
  void slide_window();
};

class AtomicBitmapPacketTracker {
 private:
  static constexpr size_t WINDOW_SIZE = 65536;
  std::vector<std::atomic<bool>> ack_bitmap_;
  std::vector<std::atomic<size_t>> packet_sizes_;  // Store packet sizes
  std::atomic<uint32_t> base_seq_num_;
  std::atomic<uint32_t> next_seq_num_;

 public:
  AtomicBitmapPacketTracker(uint32_t initial_seq = 0);

  uint32_t send_packet(size_t packet_size = 0);

  void acknowledge(uint32_t seq_num);

  bool is_acknowledged(uint32_t seq_num) const;

  std::vector<uint32_t> get_unacknowledged_packets() const;

  // Get the number of inflight (unacknowledged) packets
  uint32_t get_inflight_count() const;

  // Get the size of a specific packet
  size_t get_packet_size(uint32_t seq_num) const;

  // Get the total bytes of inflight (unacknowledged) packets
  size_t get_total_inflight_bytes() const;

 private:
  void slide_window();
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
  AtomicBitmapPacketTrackerMultiAck(uint32_t initial_seq = 0);

  // Send a packet with expected number of ACKs required
  uint32_t send_packet(size_t packet_size = 0, uint32_t expected_ack = 1);

  // Acknowledge one “sub-ACK”
  void acknowledge(uint32_t seq_num);

  bool update_expected_ack_count(uint32_t seq_num, uint32_t new_expected_ack);

  // Query if fully acked
  bool is_acknowledged(uint32_t seq_num) const;

  std::vector<uint32_t> get_unacknowledged_packets() const;

  uint32_t get_inflight_count() const;

  size_t get_packet_size(uint32_t seq_num) const;

  size_t get_total_inflight_bytes() const;

  // Update packet size for a seq_num that was registered with size 0.
  // Used when the actual size is known later (e.g. after popping from queue).
  void update_packet_size(uint32_t seq_num, size_t packet_size);

 private:
  void slide_window();
};
