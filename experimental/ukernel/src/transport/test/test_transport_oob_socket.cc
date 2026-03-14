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
#include <unistd.h>
#include <vector>

using SockExchanger = UKernel::Transport::SockExchanger;
using CommunicatorMeta = UKernel::Transport::CommunicatorMeta;
using MR = UKernel::Transport::MR;
using MRInfos = UKernel::Transport::MRInfos;

namespace {

int socket_test_port() { return 20379 + static_cast<int>(::getpid() % 1000); }

}  // namespace

void publisher_socket(int port) {
  SockExchanger ex(false, "127.0.0.1", port);
  if (!ex.valid()) {
    std::cerr << "[ERROR] Publisher failed to connect to server\n";
    return;
  }

  MRInfos remote{};
  remote.mrs.push_back(MR{1, 0x12345000ULL, 4096, 0, 123});
  remote.mrs.push_back(MR{2, 0x12346000ULL, 8192, 0, 456});

  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  if (!ex.publish("mr:peer:1:0", remote)) {
    std::cerr << "[ERROR] Publisher failed to publish MR info\n";
    return;
  }
  std::cout << "[INFO] Publisher published MR info\n";
}

void test_socket_oob() {
  int port = socket_test_port();
  SockExchanger ex(true, "127.0.0.1", port);
  if (!ex.valid()) {
    std::cerr << "[ERROR] Failed to start SockExchanger server\n";
    return;
  }
  std::cout << "[INFO] SockExchanger server started\n";

  MRInfos local{};
  local.mrs.push_back(MR{7, 0xABCDEF00ULL, 16384, 0, 789});

  if (!ex.publish("mr:peer:0:0", local)) {
    std::cerr << "[ERROR] Failed to publish local MR info\n";
    return;
  }
  std::cout << "[INFO] Published local MR info\n";

  std::thread pub_thread(publisher_socket, port);

  MRInfos remote{};
  if (ex.wait_and_fetch("mr:peer:1:0", remote, 50, 100)) {
    std::cout << "[INFO] Received remote MR info:\n";
    for (size_t i = 0; i < remote.mrs.size(); ++i) {
      auto const& mr = remote.mrs[i];
      std::cout << "  MR[" << i << "]: id=" << mr.id << " addr=0x" << std::hex
                << mr.address << std::dec << " len=" << mr.length
                << " key=" << mr.key << "\n";
    }
  } else {
    std::cerr << "[WARN] Timeout waiting for remote MR info\n";
  }

  pub_thread.join();
  std::cout << "[INFO] socket OOB test complete\n";
}

void rank_thread_socket(int rank, int world_size, std::string const& ip,
                        int port) {
  bool is_server = (rank == 0);
  auto ex = std::make_shared<SockExchanger>(is_server, ip, port);
  if (!ex->valid()) {
    std::cerr << "[ERROR] Rank " << rank << " failed to init SockExchanger\n";
    return;
  }

  CommunicatorMeta local;
  local.host_id =
      UKernel::Transport::generate_host_id() + "_" + std::to_string(rank);
  local.ip = "127.0.0.1";
  local.is_ready = true;

  std::string key = "meta:" + std::to_string(rank);
  if (!ex->publish(key, local)) {
    std::cerr << "[ERROR] Rank " << rank << " failed to publish meta\n";
    return;
  }
  std::cout << "[INFO] Rank " << rank << " published meta (" << local.host_id
            << ")\n";

  for (int r = 0; r < world_size; ++r) {
    if (r == rank) continue;
    std::string remote_key = "meta:" + std::to_string(r);
    CommunicatorMeta remote;
    if (ex->wait_and_fetch(remote_key, remote, 50, 100)) {
      std::cout << "[INFO] Rank " << rank << " fetched meta for rank " << r
                << " host_id=" << remote.host_id << " ip=" << remote.ip
                << " ready=" << remote.is_ready << "\n";
    } else {
      std::cerr << "[WARN] Rank " << rank
                << " timeout waiting for meta of rank " << r << "\n";
    }
  }

  std::cout << "[INFO] Rank " << rank << " completed meta exchange\n";
}

void test_socket_meta_exchange_multi_threads(int world_size) {
  int port = socket_test_port() + 1;
  std::vector<std::thread> threads;
  for (int rank = 0; rank < world_size; ++rank) {
    threads.emplace_back(rank_thread_socket, rank, world_size, "127.0.0.1", port);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  for (auto& t : threads) t.join();
  std::cout << "[INFO] Multi-rank socket exchange test complete\n";
}
