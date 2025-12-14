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

void publisher() {
  RedisExchanger ex("127.0.0.1", 6379);
  if (!ex.valid()) {
    std::cerr << "[ERROR] Publisher failed to connect to Redis" << std::endl;
    return;
  }

  RDMAInfo remote{};
  QpInfo qp{};
  qp.qp_num = 4321;
  qp.psn = 0x123456;
  qp.lid = 8765;
  for (int i = 0; i < 16; i++) qp.gid[i] = 15 - i;
  remote.qps.push_back(qp);

  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  if (!ex.publish("rdma:peer:1:0", remote)) {
    std::cerr << "[ERROR] Publisher failed to publish remote RDMA info"
              << std::endl;
    return;
  }
  std::cout << "[INFO] Publisher thread published remote RDMA info"
            << std::endl;
}

void test_redis_oob() {
  RedisExchanger ex("127.0.0.1", 6379);
  if (!ex.valid()) {
    std::cerr << "[ERROR] Failed to connect to Redis" << std::endl;
    return;
  }
  std::cout << "[INFO] Connected to Redis" << std::endl;

  RDMAInfo local{};
  QpInfo qp{};
  qp.qp_num = 1234;
  qp.psn = 0x654321;
  qp.lid = 5678;
  for (int i = 0; i < 16; i++) qp.gid[i] = i;
  local.qps.push_back(qp);

  if (!ex.publish("rdma:peer:0:0", local)) {
    std::cerr << "[ERROR] Failed to publish local RDMA info" << std::endl;
    return;
  }
  std::cout << "[INFO] Published local RDMA info" << std::endl;

  std::thread pub_thread(publisher);

  RDMAInfo remote{};
  if (ex.wait_and_fetch("rdma:peer:1:0", remote)) {
    std::cout << "[INFO] Got remote RDMA info:" << std::endl;
    for (size_t i = 0; i < remote.qps.size(); ++i) {
      auto const& qp = remote.qps[i];
      std::cout << "  QP[" << i << "]: qp_num=" << qp.qp_num
                << " psn=" << qp.psn << " lid=" << qp.lid << " gid=";
      for (int j = 0; j < 16; j++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0')
                  << static_cast<int>(qp.gid[j]) << " ";
      }
      std::cout << std::dec << std::endl;
    }
  } else {
    std::cerr << "[WARN] Timeout waiting for remote RDMA info" << std::endl;
  }

  pub_thread.join();
  std::cout << "[INFO] Redis OOB multithread test completed successfully"
            << std::endl;
}

void rank_thread(int local_rank, int world_size, std::string const& exchanger_ip,
                 int exchanger_port) {
  auto ex = std::make_shared<RedisExchanger>(exchanger_ip, exchanger_port);
  if (!ex->valid()) {
    std::cerr << "[ERROR] Rank " << local_rank << " failed to connect to Redis"
              << std::endl;
    return;
  }

  CommunicatorMeta local;
  local.host_id = generate_host_id() + "_" + std::to_string(local_rank);
  local.is_ready = true;

  std::string key = "meta:" + std::to_string(local_rank);

  if (!ex->publish(key, local)) {
    std::cerr << "[ERROR] Rank " << local_rank
              << " failed to publish meta to key " << key << std::endl;
    return;
  }
  std::cout << "[INFO] Rank " << local_rank << " published meta to key " << key
            << std::endl;

  for (int r = 0; r < world_size; ++r) {
    if (r == local_rank) continue;
    std::string remote_key = "meta:" + std::to_string(r);
    CommunicatorMeta remote;
    if (ex->wait_and_fetch(remote_key, remote, 50, 100)) {
      std::cout << "[INFO] Rank " << local_rank << " fetched meta for rank "
                << r << ", host_id=" << remote.host_id
                << ", is_ready=" << remote.is_ready << std::endl;
    } else {
      std::cerr << "[WARN] Rank " << local_rank
                << " timeout waiting for meta of rank " << r << std::endl;
    }
  }

  std::cout << "[INFO] Rank " << local_rank << " completed meta exchange"
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
