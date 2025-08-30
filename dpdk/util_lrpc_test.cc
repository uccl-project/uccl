#include "util/lrpc.h"  // Include your LRPC-related headers
#include <glog/logging.h>
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

std::atomic<bool> stop_flag{false};

void lcore_thread(LRPC& lrpc, std::atomic<int>& lcore_msg_count) {
  lrpc_msg msg;
  memset(&msg, 0, sizeof(lrpc_msg));

  while (!stop_flag) {
    msg.cmd = 0;
    *(uint64_t*)msg.data = lcore_msg_count.load();
    if (lrpc.lcore_send(&msg) == 0) {
      lcore_msg_count++;
    }
  }
}

void rcore_thread(LRPC& lrpc, std::atomic<int>& rcore_msg_count) {
  lrpc_msg msg;
  memset(&msg, 0, sizeof(lrpc_msg));

  while (!stop_flag) {
    if (lrpc.rcore_recv(&msg) == 0) {
      auto recv_count = *(uint64_t*)msg.data;
      CHECK(recv_count == rcore_msg_count.load());
      rcore_msg_count++;
    }
  }
}

void throughput_test() {
  LRPC lrpc;
  std::atomic<int> lcore_msg_count(0);
  std::atomic<int> rcore_msg_count(0);

  // Create and launch threads
  std::thread lcore(lcore_thread, std::ref(lrpc), std::ref(lcore_msg_count));
  std::thread rcore(rcore_thread, std::ref(lrpc), std::ref(rcore_msg_count));

  // Run the test for a fixed duration
  constexpr int test_duration_sec = 5;
  auto start_time = std::chrono::steady_clock::now();
  std::this_thread::sleep_for(std::chrono::seconds(test_duration_sec));
  stop_flag = true;

  lcore.join();
  rcore.join();

  auto end_time = std::chrono::steady_clock::now();
  double duration_sec =
      std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time)
          .count();

  std::cout << "Throughput Test Results:\n";
  std::cout << "Test Duration: " << duration_sec << " seconds\n";
  std::cout << "LCore Sent Messages: " << lcore_msg_count.load() << "\n";
  std::cout << "RCore Received Messages: " << rcore_msg_count.load() << "\n";
  std::cout << "Total Throughput: "
            << (lcore_msg_count.load() + rcore_msg_count.load()) / duration_sec
            << " messages/second\n";
}

int main() {
  std::cout << "Starting LRPC Throughput Test...\n";
  throughput_test();
  return 0;
}