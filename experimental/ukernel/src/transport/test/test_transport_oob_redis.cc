#include "oob.h"
#include "test.h"
#include "utils.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using RedisExchanger = UKernel::Transport::RedisExchanger;
using CommunicatorMeta = UKernel::Transport::CommunicatorMeta;
using MR = UKernel::Transport::MR;
using MRInfos = UKernel::Transport::MRInfos;

void publisher() {
  RedisExchanger ex("127.0.0.1", 6379);
  if (!ex.valid()) {
    std::cerr << "[ERROR] Publisher failed to connect to Redis" << std::endl;
    return;
  }

  MRInfos remote{};
  remote.mrs.push_back(MR{1, 0x12345000ULL, 4096, 0, 123});
  remote.mrs.push_back(MR{2, 0x12346000ULL, 8192, 0, 456});

  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  if (!ex.publish("mr:peer:1:0", remote)) {
    std::cerr << "[ERROR] Publisher failed to publish remote MR info"
              << std::endl;
    return;
  }
  std::cout << "[INFO] Publisher thread published remote MR info"
            << std::endl;
}

void test_redis_oob() {
  RedisExchanger ex("127.0.0.1", 6379);
  if (!ex.valid()) {
    std::cerr << "[ERROR] Failed to connect to Redis" << std::endl;
    return;
  }
  std::cout << "[INFO] Connected to Redis" << std::endl;

  MRInfos local{};
  local.mrs.push_back(MR{7, 0xABCDEF00ULL, 16384, 0, 789});

  if (!ex.publish("mr:peer:0:0", local)) {
    std::cerr << "[ERROR] Failed to publish local MR info" << std::endl;
    return;
  }
  std::cout << "[INFO] Published local MR info" << std::endl;

  std::thread pub_thread(publisher);

  MRInfos remote{};
  if (ex.wait_and_fetch("mr:peer:1:0", remote)) {
    std::cout << "[INFO] Got remote MR info:" << std::endl;
    for (size_t i = 0; i < remote.mrs.size(); ++i) {
      auto const& mr = remote.mrs[i];
      std::cout << "  MR[" << i << "]: id=" << mr.id << " addr=0x"
                << std::hex << mr.address << std::dec << " len=" << mr.length
                << " key=" << mr.key << std::endl;
    }
  } else {
    std::cerr << "[WARN] Timeout waiting for remote MR info" << std::endl;
  }

  pub_thread.join();
  std::cout << "[INFO] Redis OOB multithread test completed successfully"
            << std::endl;
}

void rank_thread(int rank, int world_size, std::string const& exchanger_ip,
                 int exchanger_port) {
  auto ex = std::make_shared<RedisExchanger>(exchanger_ip, exchanger_port);
  if (!ex->valid()) {
    std::cerr << "[ERROR] Rank " << rank << " failed to connect to Redis"
              << std::endl;
    return;
  }

  CommunicatorMeta local;
  local.host_id =
      UKernel::Transport::generate_host_id() + "_" + std::to_string(rank);
  local.is_ready = true;

  std::string key = "meta:" + std::to_string(rank);

  if (!ex->publish(key, local)) {
    std::cerr << "[ERROR] Rank " << rank << " failed to publish meta to key "
              << key << std::endl;
    return;
  }
  std::cout << "[INFO] Rank " << rank << " published meta to key " << key
            << std::endl;

  for (int r = 0; r < world_size; ++r) {
    if (r == rank) continue;
    std::string remote_key = "meta:" + std::to_string(r);
    CommunicatorMeta remote;
    if (ex->wait_and_fetch(remote_key, remote, 50, 100)) {
      std::cout << "[INFO] Rank " << rank << " fetched meta for rank " << r
                << ", host_id=" << remote.host_id
                << ", is_ready=" << remote.is_ready << std::endl;
    } else {
      std::cerr << "[WARN] Rank " << rank
                << " timeout waiting for meta of rank " << r << std::endl;
    }
  }

  std::cout << "[INFO] Rank " << rank << " completed meta exchange"
            << std::endl;
}

void test_redis_meta_exchange_multi_threads(int world_size) {
  std::vector<std::thread> threads;
  for (int rank = 0; rank < world_size; ++rank) {
    threads.emplace_back(rank_thread, rank, world_size, "127.0.0.1", 6379);
  }

  for (auto& t : threads) t.join();
  std::cout << "[INFO] CommunicatorMeta multithread Redis test completed"
            << std::endl;
}
