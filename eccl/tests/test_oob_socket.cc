#include "oob.h"
#include "test.h"
#include "utils.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

void publisher_socket() {
  SockExchanger ex(false, "127.0.0.1", 6379);
  if (!ex.valid()) {
    std::cerr << "[ERROR] Publisher failed to connect to server\n";
    return;
  }

  RDMAInfo remote{};
  QpInfo qp{};
  qp.qp_num = 4321;
  qp.psn = 0x123456;
  qp.lid = 8765;
  for (int i = 0; i < 16; ++i) qp.gid[i] = 15 - i;
  remote.qps.push_back(qp);

  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  if (!ex.publish("rdma:peer:1:0", remote)) {
    std::cerr << "[ERROR] Publisher failed to publish RDMA info\n";
    return;
  }
  std::cout << "[INFO] Publisher published RDMA info\n";
}

void test_socket_oob() {
  SockExchanger ex(true, "127.0.0.1", 6379);
  if (!ex.valid()) {
    std::cerr << "[ERROR] Failed to start SockExchanger server\n";
    return;
  }
  std::cout << "[INFO] SockExchanger server started\n";

  RDMAInfo local{};
  QpInfo qp{};
  qp.qp_num = 1234;
  qp.psn = 0x654321;
  qp.lid = 5678;
  for (int i = 0; i < 16; ++i) qp.gid[i] = i;
  local.qps.push_back(qp);

  if (!ex.publish("rdma:peer:0:0", local)) {
    std::cerr << "[ERROR] Failed to publish local RDMA info\n";
    return;
  }
  std::cout << "[INFO] Published local RDMA info\n";

  std::thread pub_thread(publisher_socket);

  RDMAInfo remote{};
  if (ex.wait_and_fetch("rdma:peer:1:0", remote, 50, 100)) {
    std::cout << "[INFO] Received remote RDMA info:\n";
    for (size_t i = 0; i < remote.qps.size(); ++i) {
      auto const& q = remote.qps[i];
      std::cout << "  QP[" << i << "]: qp_num=" << q.qp_num << " psn=" << q.psn
                << " lid=" << q.lid << " gid=";
      for (int j = 0; j < 16; ++j) {
        std::cout << std::hex << std::setw(2) << std::setfill('0')
                  << static_cast<int>(q.gid[j]) << " ";
      }
      std::cout << std::dec << "\n";
    }
  } else {
    std::cerr << "[WARN] Timeout waiting for remote RDMA info\n";
  }

  pub_thread.join();
  std::cout << "[INFO] RDMA socket OOB test complete\n";
}

void rank_thread_socket(int local_rank, int world_size, std::string const& ip,
                        int port) {
  bool is_server = (local_rank == 0);
  auto ex = std::make_shared<SockExchanger>(is_server, ip, port);
  if (!ex->valid()) {
    std::cerr << "[ERROR] Rank " << local_rank
              << " failed to init SockExchanger\n";
    return;
  }

  CommunicatorMeta local;
  local.host_id = generate_host_id() + "_" + std::to_string(local_rank);
  local.ip = "127.0.0.1";
  local.is_ready = true;

  std::string key = "meta:" + std::to_string(local_rank);
  if (!ex->publish(key, local)) {
    std::cerr << "[ERROR] Rank " << local_rank << " failed to publish meta\n";
    return;
  }
  std::cout << "[INFO] Rank " << local_rank << " published meta ("
            << local.host_id << ")\n";

  for (int r = 0; r < world_size; ++r) {
    if (r == local_rank) continue;
    std::string remote_key = "meta:" + std::to_string(r);
    CommunicatorMeta remote;
    if (ex->wait_and_fetch(remote_key, remote, 50, 100)) {
      std::cout << "[INFO] Rank " << local_rank << " fetched meta for rank "
                << r << " host_id=" << remote.host_id << " ip=" << remote.ip
                << " ready=" << remote.is_ready << "\n";
    } else {
      std::cerr << "[WARN] Rank " << local_rank
                << " timeout waiting for meta of rank " << r << "\n";
    }
  }

  std::cout << "[INFO] Rank " << local_rank << " completed meta exchange\n";
}

void test_socket_meta_exchange_multi_threads(int world_size) {
  std::vector<std::thread> threads;
  for (int rank = 0; rank < world_size; ++rank) {
    threads.emplace_back(rank_thread_socket, rank, world_size, "127.0.0.1",
                         12345);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  for (auto& t : threads) t.join();
  std::cout << "[INFO] Multi-rank socket exchange test complete\n";
}