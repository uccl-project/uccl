#include "oob/oob.h"
#include "test.h"
#include "test_utils.h"
#include "util/utils.h"
#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <memory>
#include <netinet/in.h>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <vector>
#include <unistd.h>

using SocketExchanger = UKernel::Transport::SocketExchanger;
using HierarchicalExchanger = UKernel::Transport::HierarchicalExchanger;
using Exchanger = UKernel::Transport::Exchanger;
using CommunicatorMeta = UKernel::Transport::CommunicatorMeta;
using MR = UKernel::Transport::MR;
using NamedMR = UKernel::Transport::NamedMR;
using NamedMRInfos = UKernel::Transport::NamedMRInfos;

namespace {

using UKernel::Transport::TestUtil::require;
using UKernel::Transport::TestUtil::run_case;
using UKernel::Transport::TestUtil::ScopedEnvVar;

constexpr int kTestTimeoutMs = 10000;
constexpr int kMetaExchangeRetries = 150;
constexpr int kMetaExchangeDelayMs = 100;

int socket_test_port() {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    return 20379 + static_cast<int>(::getpid() % 1000);
  }
  int opt = 1;
  (void)::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(0);
  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(fd);
    return 20379 + static_cast<int>(::getpid() % 1000);
  }
  socklen_t addr_len = sizeof(addr);
  if (::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &addr_len) != 0) {
    ::close(fd);
    return 20379 + static_cast<int>(::getpid() % 1000);
  }
  int const port = static_cast<int>(ntohs(addr.sin_port));
  ::close(fd);
  return port;
}

bool wait_for_listener(std::string const& ip, int port, int timeout_ms) {
  auto const deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      continue;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (::inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) <= 0) {
      ::close(fd);
      return false;
    }

    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
      ::shutdown(fd, SHUT_RDWR);
      ::close(fd);
      return true;
    }
    ::close(fd);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return false;
}

std::string unique_oob_namespace(char const* prefix) {
  std::ostringstream oss;
  oss << prefix << "-" << static_cast<long long>(::getpid()) << "-"
      << std::chrono::steady_clock::now().time_since_epoch().count();
  return oss.str();
}

void test_socket_exchanger_direct_sync_and_publish() {
  int port = socket_test_port() + 10;
  SocketExchanger root(
      /*is_root=*/true, "127.0.0.1", port, /*timeout_ms=*/kTestTimeoutMs,
      /*max_line_bytes=*/1 * 1024 * 1024,
      [](std::string const&, std::string const&) {});
  require(root.start(), "root socket exchanger start failed");
  require(root.valid(), "root socket exchanger should be valid after start");
  require(wait_for_listener("127.0.0.1", port, 1000),
          "root socket listener not ready in time");

  CommunicatorMeta seed{};
  seed.host_id = "seed-root";
  seed.ip = "127.0.0.1";
  seed.local_id = 0;
  seed.rdma_capable = true;
  require(root.put("seed:key", seed), "root failed to publish seed payload");

  std::atomic<int> client_callback_count{0};
  SocketExchanger client(
      /*is_root=*/false, "127.0.0.1", port, /*timeout_ms=*/kTestTimeoutMs,
      /*max_line_bytes=*/1 * 1024 * 1024,
      [&](std::string const&, std::string const&) {
        client_callback_count.fetch_add(1, std::memory_order_relaxed);
      });
  require(client.start(), "client socket exchanger start failed");
  require(client.valid(), "client socket exchanger should be valid after sync");
  require(client.connection_epoch() > 0,
          "client connection epoch should advance after sync");

  CommunicatorMeta from_snapshot{};
  require(client.get("seed:key", from_snapshot),
          "client should read root snapshot payload");
  require(from_snapshot.host_id == "seed-root", "snapshot payload mismatch");

  CommunicatorMeta from_client{};
  from_client.host_id = "client";
  from_client.ip = "127.0.0.1";
  from_client.local_id = 7;
  from_client.rdma_capable = false;
  require(client.put("client:key", from_client),
          "client failed to publish payload");

  CommunicatorMeta root_seen{};
  require(root.wait("client:key", root_seen, Exchanger::WaitOptions{80, 25}),
          "root should receive client payload");
  require(root_seen.host_id == "client", "root received payload mismatch");

  CommunicatorMeta from_root{};
  from_root.host_id = "root";
  from_root.ip = "127.0.0.1";
  from_root.local_id = 3;
  from_root.rdma_capable = false;
  require(root.put("root:key", from_root), "root failed to publish payload");

  CommunicatorMeta client_seen{};
  require(client.wait("root:key", client_seen, Exchanger::WaitOptions{80, 25}),
          "client should receive root payload");
  require(client_seen.host_id == "root", "client received payload mismatch");
  require(client_callback_count.load(std::memory_order_relaxed) > 0,
          "client callback should be invoked on incoming publish");
}

void publisher_socket(int port) {
  HierarchicalExchanger ex(
      /*is_server=*/false, "127.0.0.1", port, /*timeout_ms=*/kTestTimeoutMs,
      /*max_line_bytes=*/1 * 1024 * 1024, /*local_id=*/0);
  require(ex.valid(), "publisher failed to connect to socket exchanger");

  NamedMRInfos remote{};
  remote.entries.push_back(NamedMR{5, MR{1, 0x12345000ULL, 4096, 0, 123}});
  remote.entries.push_back(NamedMR{8, MR{2, 0x12346000ULL, 8192, 0, 456}});

  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  require(ex.put("named-mr:peer:1:0", remote),
          "publisher failed to publish MR info");
}

void test_socket_publish_fetch() {
  int port = socket_test_port();
  std::string const ns = unique_oob_namespace("oob-socket-publish");
  ScopedEnvVar oob_ns_guard("UHM_OOB_NAMESPACE", ns.c_str());
  HierarchicalExchanger ex(
      /*is_server=*/true, "127.0.0.1", port, /*timeout_ms=*/kTestTimeoutMs);
  require(ex.valid(), "failed to start socket exchanger server");
  require(wait_for_listener("127.0.0.1", port, 1000),
          "socket listener not ready in time");

  NamedMRInfos local{};
  local.entries.push_back(NamedMR{9, MR{7, 0xABCDEF00ULL, 16384, 0, 789}});
  require(ex.put("named-mr:peer:0:0", local),
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
  require(ex.wait("named-mr:peer:1:0", remote, Exchanger::WaitOptions{50, 100}),
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
    auto ex = std::make_shared<HierarchicalExchanger>(
        is_server, ip, port, /*timeout_ms=*/kTestTimeoutMs,
        /*max_line_bytes=*/1 * 1024 * 1024, /*local_id=*/0);
    require(ex->valid(), "rank failed to init socket exchanger");

    CommunicatorMeta local;
    local.host_id =
        UKernel::Transport::generate_host_id() + "_" + std::to_string(rank);
    local.ip = "127.0.0.1";

    std::string key = "meta:" + std::to_string(rank);
    require(ex->put(key, local), "rank failed to publish local meta");

    for (int r = 0; r < world_size; ++r) {
      if (r == rank) continue;
      std::string remote_key = "meta:" + std::to_string(r);
      CommunicatorMeta remote;
      require(ex->wait(remote_key, remote,
                       Exchanger::WaitOptions{kMetaExchangeRetries,
                                              kMetaExchangeDelayMs}),
              "timeout waiting for remote communicator meta");
      require(!remote.host_id.empty(), "remote host id should not be empty");
      require(remote.ip == "127.0.0.1", "remote ip mismatch");
    }
  } catch (...) {
    error = std::current_exception();
  }
}

void test_socket_meta_exchange(int world_size) {
  int port = socket_test_port();
  std::string const ns = unique_oob_namespace("oob-socket-meta");
  ScopedEnvVar oob_ns_guard("UHM_OOB_NAMESPACE", ns.c_str());
  std::vector<std::thread> threads;
  std::vector<std::exception_ptr> errors(world_size);

  // Launch rank0 first and wait for listener readiness to reduce startup races.
  threads.emplace_back(rank_thread_socket, 0, world_size, "127.0.0.1", port,
                       std::ref(errors[0]));
  require(wait_for_listener("127.0.0.1", port, 2000),
          "rank0 socket listener not ready in time");

  for (int rank = 1; rank < world_size; ++rank) {
    threads.emplace_back(rank_thread_socket, rank, world_size, "127.0.0.1",
                         port, std::ref(errors[rank]));
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
  }

  for (auto& t : threads) t.join();
  for (auto const& error : errors) {
    if (error) std::rethrow_exception(error);
  }
}

void test_socket_valid_implies_snapshot_ready() {
  int port = socket_test_port();
  std::string const ns = unique_oob_namespace("oob-socket-snapshot");
  ScopedEnvVar oob_ns_guard("UHM_OOB_NAMESPACE", ns.c_str());
  HierarchicalExchanger server(
      /*is_server=*/true, "127.0.0.1", port, /*timeout_ms=*/kTestTimeoutMs);
  require(server.valid(), "failed to start socket exchanger server");
  require(wait_for_listener("127.0.0.1", port, 1000),
          "socket listener not ready in time");

  NamedMRInfos payload{};
  payload.generation = 7;
  payload.entries.push_back(NamedMR{3, MR{11, 0x11111000ULL, 2048, 0, 11}});
  payload.entries.push_back(NamedMR{4, MR{12, 0x22222000ULL, 4096, 0, 12}});
  require(server.put("snapshot:key", payload),
          "server failed to publish snapshot payload");

  HierarchicalExchanger client(
      /*is_server=*/false, "127.0.0.1", port, /*timeout_ms=*/kTestTimeoutMs,
      /*max_line_bytes=*/1 * 1024 * 1024, /*local_id=*/0);
  require(client.valid(), "client failed to finish initial sync");

  NamedMRInfos fetched{};
  require(client.get("snapshot:key", fetched),
          "client should see snapshot payload immediately after valid");
  require(fetched.generation == 7, "snapshot generation mismatch");
  require(fetched.entries.size() == 2, "snapshot entry count mismatch");
}

void test_socket_reconnect_gets_full_snapshot() {
  int port = socket_test_port();
  std::string const ns = unique_oob_namespace("oob-socket-reconnect");
  ScopedEnvVar oob_ns_guard("UHM_OOB_NAMESPACE", ns.c_str());
  HierarchicalExchanger server(
      /*is_server=*/true, "127.0.0.1", port, /*timeout_ms=*/kTestTimeoutMs);
  require(server.valid(), "failed to start socket exchanger server");
  require(wait_for_listener("127.0.0.1", port, 1000),
          "socket listener not ready in time");

  CommunicatorMeta first{};
  first.host_id = "node-a";
  first.ip = "127.0.0.1";
  require(server.put("meta:10", first), "failed to publish first snapshot key");

  {
    HierarchicalExchanger client(
        /*is_server=*/false, "127.0.0.1", port, /*timeout_ms=*/kTestTimeoutMs,
        /*max_line_bytes=*/1 * 1024 * 1024, /*local_id=*/0);
    require(client.valid(), "first client failed to sync");
    CommunicatorMeta fetched{};
    require(client.get("meta:10", fetched),
            "first client missing first snapshot key");
  }

  CommunicatorMeta second{};
  second.host_id = "node-b";
  second.ip = "127.0.0.1";
  require(server.put("meta:11", second),
          "failed to publish second snapshot key");

  HierarchicalExchanger reconnected(
      /*is_server=*/false, "127.0.0.1", port, /*timeout_ms=*/kTestTimeoutMs,
      /*max_line_bytes=*/1 * 1024 * 1024, /*local_id=*/0);
  require(reconnected.valid(), "reconnected client failed to sync");

  CommunicatorMeta fetched_first{};
  CommunicatorMeta fetched_second{};
  require(reconnected.get("meta:10", fetched_first),
          "reconnected client missing first snapshot key");
  require(reconnected.get("meta:11", fetched_second),
          "reconnected client missing second snapshot key");
  require(fetched_first.host_id == "node-a",
          "first snapshot payload mismatch after reconnect");
  require(fetched_second.host_id == "node-b",
          "second snapshot payload mismatch after reconnect");
}

void test_hierarchical_cross_namespace_relay() {
  int port = socket_test_port();
  std::string const root_ns = unique_oob_namespace("oob-root-ns");
  std::string const client_ns = unique_oob_namespace("oob-client-ns");

  ScopedEnvVar root_guard("UHM_OOB_NAMESPACE", root_ns.c_str());
  HierarchicalExchanger root(
      /*is_server=*/true, "127.0.0.1", port, /*timeout_ms=*/kTestTimeoutMs,
      /*max_line_bytes=*/1 * 1024 * 1024, /*local_id=*/0);
  require(root.valid(), "root hierarchical exchanger should be valid");
  require(wait_for_listener("127.0.0.1", port, 1000),
          "hierarchical root listener not ready in time");

  CommunicatorMeta root_bootstrap{};
  root_bootstrap.host_id = "root-bootstrap";
  root_bootstrap.ip = "127.0.0.1";
  root_bootstrap.local_id = 0;
  root_bootstrap.rdma_capable = true;
  require(root.put("bootstrap:key", root_bootstrap),
          "root failed to publish bootstrap key");

  ScopedEnvVar client_guard("UHM_OOB_NAMESPACE", client_ns.c_str());
  HierarchicalExchanger client_leader(
      /*is_server=*/false, "127.0.0.1", port, /*timeout_ms=*/kTestTimeoutMs,
      /*max_line_bytes=*/1 * 1024 * 1024, /*local_id=*/0);
  require(client_leader.valid(), "client leader should be valid");

  CommunicatorMeta bootstrap_seen{};
  Exchanger::WaitOptions const wait_options{/*max_retries=*/120,
                                            /*delay_ms=*/25};
  require(client_leader.wait("bootstrap:key", bootstrap_seen, wait_options),
          "client leader should receive root bootstrap payload");
  require(bootstrap_seen.host_id == "root-bootstrap",
          "client leader bootstrap payload mismatch");

  HierarchicalExchanger client_worker(
      /*is_server=*/false, "127.0.0.1", port, /*timeout_ms=*/kTestTimeoutMs,
      /*max_line_bytes=*/1 * 1024 * 1024, /*local_id=*/1);
  require(client_worker.valid(), "client worker should be valid via shm only");

  CommunicatorMeta root_broadcast{};
  root_broadcast.host_id = "root-broadcast";
  root_broadcast.ip = "127.0.0.1";
  root_broadcast.local_id = 0;
  root_broadcast.rdma_capable = false;
  require(root.put("broadcast:key", root_broadcast),
          "root failed to publish broadcast key");

  CommunicatorMeta worker_seen{};
  require(client_worker.wait("broadcast:key", worker_seen, wait_options),
          "client worker should receive root broadcast via local shm");
  require(worker_seen.host_id == "root-broadcast",
          "client worker received payload mismatch");

  CommunicatorMeta client_payload{};
  client_payload.host_id = "client-payload";
  client_payload.ip = "127.0.0.1";
  client_payload.local_id = 1;
  client_payload.rdma_capable = false;
  require(client_leader.put("client:key", client_payload),
          "client leader failed to publish payload");

  CommunicatorMeta root_seen{};
  require(root.wait("client:key", root_seen, wait_options),
          "root should receive client payload through socket relay");
  require(root_seen.host_id == "client-payload",
          "root received client payload mismatch");
}

}  // namespace

void test_socket_oob() {
  run_case("transport unit", "socket exchanger direct sync and publish",
           test_socket_exchanger_direct_sync_and_publish);
  run_case("transport unit", "socket oob publish/fetch",
           test_socket_publish_fetch);
  run_case("transport unit", "socket oob valid implies snapshot ready",
           test_socket_valid_implies_snapshot_ready);
  run_case("transport unit", "socket oob reconnect snapshot replay",
           test_socket_reconnect_gets_full_snapshot);
  run_case("transport unit", "hierarchical exchanger cross-namespace relay",
           test_hierarchical_cross_namespace_relay);
}

void test_socket_meta_exchange_multi_threads(int world_size) {
  run_case("transport unit", "socket oob meta exchange",
           [&] { test_socket_meta_exchange(world_size); });
}
