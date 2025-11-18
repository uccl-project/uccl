#include "seq_num.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

static constexpr int NUM_SENDERS = 8;
static constexpr int NUM_ACKERS = 10;
static constexpr int NUM_CHECKERS = 10;
static constexpr int PACKETS_PER_SENDER = 6000;

thread_local std::mt19937 rng(std::random_device{}());

void random_delay_us(int max_us = 300) {
  std::uniform_int_distribution<int> dist(0, max_us);
  std::this_thread::sleep_for(std::chrono::microseconds(dist(rng)));
}

int main() {
  AtomicBitmapPacketTracker tracker(0);

  std::vector<uint32_t> sent_seqs;
  sent_seqs.reserve(NUM_SENDERS * PACKETS_PER_SENDER);
  std::mutex seq_lock;

  std::atomic<bool> sending_done{false};
  std::atomic<bool> acking_done{false};

  // ============================
  // 1) 发送线程
  // ============================
  std::vector<std::thread> send_threads;
  for (int t = 0; t < NUM_SENDERS; ++t) {
    send_threads.emplace_back([&]() {
      for (int i = 0; i < PACKETS_PER_SENDER; ++i) {
        random_delay_us();
        uint32_t seq = tracker.sendPacket();
        {
          std::lock_guard<std::mutex> lock(seq_lock);
          sent_seqs.push_back(seq);
        }
      }
    });
  }

  // ============================
  // 2) ACK 线程（乱序 + 延迟）
  // ============================
  std::vector<std::thread> ack_threads;
  for (int t = 0; t < NUM_ACKERS; ++t) {
    ack_threads.emplace_back([&]() {
      while (!sending_done.load(std::memory_order_acquire)) {
        // 抽样 ACK 已发送的序号
        size_t size;
        {
          std::lock_guard<std::mutex> lock(seq_lock);
          size = sent_seqs.size();
        }

        if (size == 0) continue;

        std::uniform_int_distribution<size_t> dist(0, size - 1);

        random_delay_us(400);

        size_t idx;
        {
          std::lock_guard<std::mutex> lock(seq_lock);
          if (sent_seqs.empty()) continue;
          idx = dist(rng);
        }

        uint32_t seq = sent_seqs[idx];
        tracker.acknowledge(seq);
      }

      // 发送完后统一 ACK 所有包一次（包括乱序）
      std::vector<uint32_t> seq_copy;
      {
        std::lock_guard<std::mutex> lock(seq_lock);
        seq_copy = sent_seqs;
      }
      std::shuffle(seq_copy.begin(), seq_copy.end(), rng);

      for (uint32_t seq : seq_copy) {
        random_delay_us(500);
        tracker.acknowledge(seq);
      }
    });
  }

  // ============================
  // 3) isAcknowledged 检查线程
  // ============================
  std::atomic<bool> checker_stop{false};
  std::vector<std::thread> check_threads;

  for (int t = 0; t < NUM_CHECKERS; ++t) {
    check_threads.emplace_back([&]() {
      while (!checker_stop.load(std::memory_order_acquire)) {
        size_t size;
        {
          std::lock_guard<std::mutex> lock(seq_lock);
          size = sent_seqs.size();
        }
        if (size == 0) continue;

        std::uniform_int_distribution<size_t> dist(0, size - 1);

        size_t idx;
        {
          std::lock_guard<std::mutex> lock(seq_lock);
          if (sent_seqs.empty()) continue;
          idx = dist(rng);
        }
        uint32_t seq = sent_seqs[idx];

        // 读取 ack 状态（可能在窗口滑动）
        bool acked = tracker.isAcknowledged(seq);

        random_delay_us(200);
      }
    });
  }

  // ============================
  // 等待发送完毕
  // ============================
  for (auto& th : send_threads) th.join();
  sending_done.store(true);
  std::cout << "Senders finished. Total packets: " << sent_seqs.size()
            << std::endl;

  // ============================
  // 等待 ACK 完成
  // ============================
  for (auto& th : ack_threads) th.join();
  acking_done.store(true);
  std::cout << "Ack threads finished." << std::endl;

  // ============================
  // 停止 checker 线程
  // ============================
  checker_stop.store(true);
  for (auto& th : check_threads) th.join();
  std::cout << "Checker threads finished." << std::endl;

  // ============================
  // 最终一致性检查
  // ============================
  std::vector<uint32_t> seq_copy;
  {
    std::lock_guard<std::mutex> lock(seq_lock);
    seq_copy = sent_seqs;
  }

  for (uint32_t seq : seq_copy) {
    if (!tracker.isAcknowledged(seq)) {
      std::cerr << "ERROR: seq " << seq << " NOT acknowledged!\n";
      return 1;
    }
  }

  auto unacked = tracker.getUnacknowledgedPackets();
  if (!unacked.empty()) {
    std::cerr << "ERROR: remaining unacked packets: " << unacked.size()
              << std::endl;
    return 1;
  }

  std::cout << "Test passed (send + ack + isAck concurrent)." << std::endl;
  return 0;
}
