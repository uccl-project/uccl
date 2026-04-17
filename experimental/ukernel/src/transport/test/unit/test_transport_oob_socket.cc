#include "oob/oob.h"
#include "test.h"
#include "test_utils.h"
#include "util/utils.h"
#include <chrono>
#include <atomic>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

using SockExchanger = UKernel::Transport::SockExchanger;
using CommunicatorMeta = UKernel::Transport::CommunicatorMeta;
using MR = UKernel::Transport::MR;
using NamedMR = UKernel::Transport::NamedMR;
using NamedMRInfos = UKernel::Transport::NamedMRInfos;
using StringPayload = UKernel::Transport::StringPayload;

namespace {

using UKernel::Transport::TestUtil::require;
using UKernel::Transport::TestUtil::run_case;

int socket_test_port() { return 20379 + static_cast<int>(::getpid() % 1000); }

void publisher_socket(int port) {
  SockExchanger ex(false, "127.0.0.1", port);
  require(ex.valid(), "publisher failed to connect to socket exchanger");

  NamedMRInfos remote{};
  remote.entries.push_back(NamedMR{5, MR{1, 0x12345000ULL, 4096, 0, 123}});
  remote.entries.push_back(NamedMR{8, MR{2, 0x12346000ULL, 8192, 0, 456}});

  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  require(ex.put("test/named_mr", "named-mr:peer:1:0", remote) != 0,
          "publisher failed to publish MR info");
}

void test_socket_publish_fetch() {
  int port = socket_test_port();
  SockExchanger ex(true, "127.0.0.1", port);
  require(ex.valid(), "failed to start socket exchanger server");

  NamedMRInfos local{};
  local.entries.push_back(NamedMR{9, MR{7, 0xABCDEF00ULL, 16384, 0, 789}});
  require(ex.put("test/named_mr", "named-mr:peer:0:0", local) != 0,
          "failed to publish local MR info");

  std::exception_ptr pub_error;
  std::thread pub_thread([&] {
    try {
      publisher_socket(port);
    } catch (...) {
      pub_error = std::current_exception();
    }
  });

  NamedMRInfos remote{};
  require(ex.get("test/named_mr", "named-mr:peer:1:0", remote, nullptr, 5000),
          "timeout waiting for remote MR info");
  require(remote.entries.size() == 2, "remote MR count mismatch");
  require(remote.entries[0].mr.id == 1 && remote.entries[1].mr.id == 2,
          "remote MR ids mismatch");

  pub_thread.join();
  if (pub_error) std::rethrow_exception(pub_error);
}

void rank_thread_socket(int rank, int world_size, std::string const& ip,
                        int port, std::exception_ptr& error) {
  try {
    bool is_server = (rank == 0);
    auto ex = std::make_shared<SockExchanger>(is_server, ip, port);
    require(ex->valid(), "rank failed to init socket exchanger");

    CommunicatorMeta local;
    local.host_id =
        UKernel::Transport::generate_host_id() + "_" + std::to_string(rank);
    local.ip = "127.0.0.1";

    std::string key = "meta:" + std::to_string(rank);
    require(ex->put("test/meta", key, local) != 0,
            "rank failed to publish local meta");

    for (int r = 0; r < world_size; ++r) {
      if (r == rank) continue;
      std::string remote_key = "meta:" + std::to_string(r);
      CommunicatorMeta remote;
      require(ex->get("test/meta", remote_key, remote, nullptr, 5000),
              "timeout waiting for remote communicator meta");
      require(!remote.host_id.empty(), "remote host id should not be empty");
      require(remote.ip == "127.0.0.1", "remote ip mismatch");
    }
  } catch (...) {
    error = std::current_exception();
  }
}

void test_socket_meta_exchange(int world_size) {
  int port = socket_test_port() + 1;
  std::vector<std::thread> threads;
  std::vector<std::exception_ptr> errors(world_size);
  for (int rank = 0; rank < world_size; ++rank) {
    threads.emplace_back(rank_thread_socket, rank, world_size, "127.0.0.1",
                         port, std::ref(errors[rank]));
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  for (auto& t : threads) t.join();
  for (auto const& error : errors) {
    if (error) std::rethrow_exception(error);
  }
}

void test_socket_subscribe_callback_once_on_client_put() {
  int port = socket_test_port() + 2;
  SockExchanger server(true, "127.0.0.1", port);
  require(server.valid(), "failed to start socket exchanger server");

  SockExchanger client(false, "127.0.0.1", port);
  require(client.valid(), "failed to connect socket exchanger client");

  std::atomic<int> callback_count{0};
  require(client.subscribe(
              "test/sub", [&callback_count](std::string const& ns,
                                            std::string const& key,
                                            uint64_t /*tag*/) {
                if (ns == "test/sub" && key == "k") {
                  callback_count.fetch_add(1, std::memory_order_relaxed);
                }
              }),
          "client subscribe failed");

  StringPayload payload;
  payload.value = "v";
  require(client.put("test/sub", "k", payload) != 0, "client put failed");

  auto const deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(1000);
  while (std::chrono::steady_clock::now() < deadline &&
         callback_count.load(std::memory_order_relaxed) == 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  require(callback_count.load(std::memory_order_relaxed) == 1,
          "client put should trigger exactly one subscribe callback");
}

}  // namespace

void test_socket_oob() {
  run_case("transport unit", "socket oob publish/fetch",
           test_socket_publish_fetch);
  run_case("transport unit", "socket oob subscribe callback once",
           test_socket_subscribe_callback_once_on_client_put);
}

void test_socket_meta_exchange_multi_threads(int world_size) {
  run_case("transport unit", "socket oob meta exchange",
           [&] { test_socket_meta_exchange(world_size); });
}
