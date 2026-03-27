#include "transport.h"
#include "test_utils.h"
#include "util/util.h"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

using CommunicatorConfig = UKernel::Transport::CommunicatorConfig;
using Communicator = UKernel::Transport::Communicator;
using MR = UKernel::Transport::MR;

static constexpr int kWorldSize = 2;
static constexpr int kClientGpu = 0;
static constexpr int kServerGpu = 0;
static constexpr int kClientRank = 1;
static constexpr int kServerRank = 0;

namespace {

using UKernel::Transport::TestUtil::get_arg;
using UKernel::Transport::TestUtil::get_int_arg;
using UKernel::Transport::TestUtil::require;
using UKernel::Transport::TestUtil::run_case;

constexpr size_t kMessageBytes = 4 * 1024;
constexpr size_t kPadding = 256;
constexpr size_t kSlotStride = kMessageBytes + 2 * kPadding;
constexpr std::chrono::seconds kNotifierTimeout(5);
constexpr std::chrono::seconds kPollTimeout(10);
constexpr std::chrono::seconds kAcceptTimeout(10);
constexpr char kScenarioUsage[] =
    "basic|batch|poll-release|notifier|ipc-buffer-meta";

enum class CompletionMode { WaitEach, WaitAll, PollRelease };

struct Scenario {
  std::string name;
  int message_count = 1;
  CompletionMode completion_mode = CompletionMode::WaitEach;
  bool expect_notifier = false;
};

struct NotifierQueue {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<unsigned> ids;
};

size_t slot_offset(int index) {
  return kPadding + static_cast<size_t>(index) * kSlotStride;
}

size_t buffer_bytes_for(int message_count) {
  return slot_offset(message_count - 1) + kMessageBytes + kPadding;
}

Scenario get_scenario(std::string const& name) {
  if (name == "basic") {
    return Scenario{"basic", 1, CompletionMode::WaitEach, false};
  }
  if (name == "batch") {
    return Scenario{"batch", 4, CompletionMode::WaitAll, false};
  }
  if (name == "poll-release") {
    return Scenario{"poll-release", 4, CompletionMode::PollRelease, false};
  }
  if (name == "notifier") {
    return Scenario{"notifier", 4, CompletionMode::WaitAll, true};
  }
  if (name == "ipc-buffer-meta") {
    return Scenario{"ipc-buffer-meta", 1, CompletionMode::WaitEach, false};
  }
  throw std::invalid_argument("unknown transport communicator test case: " +
                              name);
}

void fill_pattern(std::vector<uint8_t>& buf, uint8_t seed) {
  for (size_t i = 0; i < buf.size(); ++i) {
    buf[i] = static_cast<uint8_t>((seed + i) & 0xFF);
  }
}

bool check_pattern(std::vector<uint8_t> const& buf, uint8_t seed) {
  for (size_t i = 0; i < buf.size(); ++i) {
    if (buf[i] != static_cast<uint8_t>((seed + i) & 0xFF)) {
      std::cerr << "[transport communicator] mismatch at " << i
                << " expect=" << static_cast<int>((seed + i) & 0xFF)
                << " got=" << static_cast<int>(buf[i]) << std::endl;
      return false;
    }
  }
  return true;
}

std::vector<unsigned> collect_notification_ids(NotifierQueue& queue,
                                               size_t expected) {
  std::unique_lock<std::mutex> lk(queue.mu);
  bool ready = queue.cv.wait_for(lk, kNotifierTimeout,
                                 [&] { return queue.ids.size() >= expected; });
  require(ready, "timed out waiting for transport completion notifications");

  std::vector<unsigned> ids;
  ids.reserve(queue.ids.size());
  while (!queue.ids.empty()) {
    ids.push_back(queue.ids.front());
    queue.ids.pop_front();
  }
  return ids;
}

void verify_notification_set(NotifierQueue& queue,
                             std::vector<unsigned> const& requests) {
  auto notified = collect_notification_ids(queue, requests.size());
  require(notified.size() == requests.size(),
          "notification count does not match request count");

  std::vector<unsigned> expected;
  expected.reserve(requests.size());
  for (unsigned req : requests) {
    expected.push_back(req);
  }
  std::sort(expected.begin(), expected.end());
  std::sort(notified.begin(), notified.end());
  require(expected == notified,
          "notified request ids do not match submitted requests");
}

void wait_until_completed_without_release(
    std::shared_ptr<Communicator> const& comm,
    std::vector<unsigned> const& requests) {
  std::vector<unsigned> remaining = requests;
  auto deadline = std::chrono::steady_clock::now() + kPollTimeout;
  while (!remaining.empty()) {
    for (auto it = remaining.begin(); it != remaining.end();) {
      if (comm->poll(*it)) {
        it = remaining.erase(it);
      } else {
        ++it;
      }
    }
    if (remaining.empty()) break;
    require(std::chrono::steady_clock::now() < deadline,
            "timed out polling transport requests");
    std::this_thread::yield();
  }
}

void accept_with_retry(std::shared_ptr<Communicator> const& comm, int peer_rank,
                       char const* what) {
  auto deadline = std::chrono::steady_clock::now() + kAcceptTimeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (comm->accept_from(peer_rank)) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  require(false, what);
}

void complete_requests(std::shared_ptr<Communicator> const& comm,
                       std::vector<unsigned> const& requests,
                       CompletionMode mode) {
  if (mode == CompletionMode::WaitEach) {
    for (unsigned req : requests) {
      require(comm->wait_finish(req), "wait_finish(req) failed");
    }
    return;
  }

  if (mode == CompletionMode::WaitAll) {
    require(comm->wait_finish(std::vector<unsigned>{}),
            "wait_finish({}) failed");
    return;
  }

  wait_until_completed_without_release(comm, requests);
  for (unsigned req : requests) {
    comm->release(req);
  }
}

std::shared_ptr<Communicator> make_communicator(
    int gpu, int rank, int world_size, std::string const& exchanger_ip,
    int exchanger_port,
    UKernel::Transport::PreferredTransport preferred_transport) {
  auto cfg = std::make_shared<CommunicatorConfig>();
  cfg->exchanger_ip = exchanger_ip;
  cfg->exchanger_port = exchanger_port;
  cfg->local_id = rank;
  cfg->preferred_transport = preferred_transport;
  return std::make_shared<Communicator>(gpu, rank, world_size, cfg);
}

int run_sender(Scenario const& scenario, std::string const& exchanger_ip,
               int exchanger_port,
               UKernel::Transport::PreferredTransport preferred_transport) {
  auto comm =
      make_communicator(kClientGpu, kClientRank, kWorldSize, exchanger_ip,
                        exchanger_port, preferred_transport);
  require(comm->connect_to(kServerRank), "client connect_to failed");
  auto transport_kind = comm->peer_transport_kind(kServerRank);

  NotifierQueue queue;
  std::shared_ptr<void> notifier_guard;
  if (scenario.expect_notifier) {
    notifier_guard = comm->register_completion_notifier(
        [&](unsigned id, std::chrono::steady_clock::time_point) {
          std::lock_guard<std::mutex> lk(queue.mu);
          queue.ids.push_back(id);
          queue.cv.notify_all();
        });
  }

  GPU_RT_CHECK(gpuSetDevice(kClientGpu));
  void* sendbuf_d = nullptr;
  GPU_RT_CHECK(gpuMalloc(&sendbuf_d, buffer_bytes_for(scenario.message_count)));
  auto free_buf = uccl::finally([&] { gpuFree(sendbuf_d); });
  GPU_RT_CHECK(
      gpuMemset(sendbuf_d, 0, buffer_bytes_for(scenario.message_count)));

  MR local_mr =
      comm->reg_mr(sendbuf_d, buffer_bytes_for(scenario.message_count));

  MR remote_mr{};
  if (transport_kind == UKernel::Transport::PeerTransportKind::Uccl) {
    require(comm->notify_mr(kServerRank, local_mr), "client notify_mr failed");
    require(comm->wait_mr_notify(kServerRank, remote_mr),
            "client wait_mr_notify failed");
  }

  std::vector<unsigned> requests;
  requests.reserve(scenario.message_count);
  for (int i = 0; i < scenario.message_count; ++i) {
    std::vector<uint8_t> host(kMessageBytes);
    fill_pattern(host, static_cast<uint8_t>(0x10 + i));
    GPU_RT_CHECK(gpuMemcpy(static_cast<char*>(sendbuf_d) + slot_offset(i),
                           host.data(), host.size(), gpuMemcpyHostToDevice));
    unsigned req = comm->isend(kServerRank, sendbuf_d, slot_offset(i),
                               kMessageBytes, local_mr.id, remote_mr.id, true);
    require(req != 0, "client isend failed");
    requests.push_back(req);
  }

  if (scenario.expect_notifier) {
    wait_until_completed_without_release(comm, requests);
    verify_notification_set(queue, requests);
    for (unsigned req : requests) comm->release(req);
  } else {
    complete_requests(comm, requests, scenario.completion_mode);
  }

  std::cout << "[CLIENT][" << scenario.name << "] OK" << std::endl;
  return 0;
}

int run_receiver(Scenario const& scenario, std::string const& exchanger_ip,
                 int exchanger_port,
                 UKernel::Transport::PreferredTransport preferred_transport) {
  auto comm =
      make_communicator(kServerGpu, kServerRank, kWorldSize, exchanger_ip,
                        exchanger_port, preferred_transport);
  accept_with_retry(comm, kClientRank, "server accept_from failed");
  auto transport_kind = comm->peer_transport_kind(kClientRank);

  NotifierQueue queue;
  std::shared_ptr<void> notifier_guard;
  if (scenario.expect_notifier) {
    notifier_guard = comm->register_completion_notifier(
        [&](unsigned id, std::chrono::steady_clock::time_point) {
          std::lock_guard<std::mutex> lk(queue.mu);
          queue.ids.push_back(id);
          queue.cv.notify_all();
        });
  }

  GPU_RT_CHECK(gpuSetDevice(kServerGpu));
  void* recvbuf_d = nullptr;
  GPU_RT_CHECK(gpuMalloc(&recvbuf_d, buffer_bytes_for(scenario.message_count)));
  auto free_buf = uccl::finally([&] { gpuFree(recvbuf_d); });
  GPU_RT_CHECK(
      gpuMemset(recvbuf_d, 0, buffer_bytes_for(scenario.message_count)));

  MR local_mr =
      comm->reg_mr(recvbuf_d, buffer_bytes_for(scenario.message_count));

  MR remote_mr{};
  if (transport_kind == UKernel::Transport::PeerTransportKind::Uccl) {
    require(comm->notify_mr(kClientRank, local_mr), "server notify_mr failed");
    require(comm->wait_mr_notify(kClientRank, remote_mr),
            "server wait_mr_notify failed");
  }

  std::vector<unsigned> requests;
  requests.reserve(scenario.message_count);
  for (int i = 0; i < scenario.message_count; ++i) {
    unsigned req = comm->irecv(kClientRank, recvbuf_d, slot_offset(i),
                               kMessageBytes, true);
    require(req != 0, "server irecv failed");
    requests.push_back(req);
  }

  if (scenario.expect_notifier) {
    wait_until_completed_without_release(comm, requests);
    verify_notification_set(queue, requests);
    for (unsigned req : requests) comm->release(req);
  } else {
    complete_requests(comm, requests, scenario.completion_mode);
  }

  for (int i = 0; i < scenario.message_count; ++i) {
    std::vector<uint8_t> host(kMessageBytes);
    GPU_RT_CHECK(gpuMemcpy(host.data(),
                           static_cast<char*>(recvbuf_d) + slot_offset(i),
                           host.size(), gpuMemcpyDeviceToHost));
    require(check_pattern(host, static_cast<uint8_t>(0x10 + i)),
            "received payload pattern mismatch");
  }

  std::cout << "[SERVER][" << scenario.name << "] OK" << std::endl;
  return 0;
}

int run_ipc_buffer_metadata_client(
    std::string const& exchanger_ip, int exchanger_port,
    UKernel::Transport::PreferredTransport preferred_transport) {
  auto comm =
      make_communicator(kClientGpu, kClientRank, kWorldSize, exchanger_ip,
                        exchanger_port, preferred_transport);
  require(comm->connect_to(kServerRank), "client connect_to failed");
  require(comm->peer_transport_kind(kServerRank) ==
              UKernel::Transport::PeerTransportKind::Ipc,
          "ipc-buffer-meta requires IPC transport");

  GPU_RT_CHECK(gpuSetDevice(kClientGpu));
  void* buf = nullptr;
  GPU_RT_CHECK(gpuMalloc(&buf, 4096));
  auto free_buf = uccl::finally([&] { gpuFree(buf); });
  void* ack_buf = nullptr;
  GPU_RT_CHECK(gpuMalloc(&ack_buf, 1));
  auto free_ack = uccl::finally([&] { gpuFree(ack_buf); });

  constexpr uint32_t kValidInfoMr = 4242;
  require(comm->notify_ipc_buffer(kServerRank, kValidInfoMr, buf, 4096),
          "notify_ipc_buffer should succeed");

  constexpr uint32_t kInvalidInfoMr = 9999;
  require(comm->notify_ipc_buffer(kServerRank, kInvalidInfoMr, nullptr, 0),
          "invalid ipc metadata publish should still succeed");

  unsigned ack_req = comm->irecv(kServerRank, ack_buf, 0, 1, true);
  require(ack_req != 0, "client ack irecv failed");
  require(comm->wait_finish(ack_req), "client ack wait_finish failed");

  std::cout << "[CLIENT][ipc-buffer-meta] OK" << std::endl;
  return 0;
}

int run_ipc_buffer_metadata_server(
    std::string const& exchanger_ip, int exchanger_port,
    UKernel::Transport::PreferredTransport preferred_transport) {
  auto comm =
      make_communicator(kServerGpu, kServerRank, kWorldSize, exchanger_ip,
                        exchanger_port, preferred_transport);
  accept_with_retry(comm, kClientRank, "server accept_from failed");
  require(comm->same_host(kClientRank),
          "ipc-buffer-meta requires same-host peers");
  require(comm->peer_transport_kind(kClientRank) ==
              UKernel::Transport::PeerTransportKind::Ipc,
          "ipc-buffer-meta requires IPC transport");

  GPU_RT_CHECK(gpuSetDevice(kServerGpu));
  void* ack_buf = nullptr;
  GPU_RT_CHECK(gpuMalloc(&ack_buf, 1));
  auto free_ack = uccl::finally([&] { gpuFree(ack_buf); });
  static_cast<void>(free_ack);

  void* resolved = nullptr;
  int device_idx = -1;

  constexpr uint32_t kValidInfoMr = 4242;
  require(comm->wait_ipc_buffer(kClientRank, kValidInfoMr),
          "wait_ipc_buffer should succeed");
  require(comm->resolve_remote_buffer_pointer(kClientRank, kValidInfoMr,
                                              /*offset=*/128, /*bytes=*/256,
                                              &resolved, &device_idx),
          "resolve_remote_buffer_pointer should succeed");
  require(resolved != nullptr, "resolved remote pointer should not be null");
  require(device_idx == kClientGpu,
          "resolved remote pointer should preserve device index");
  require(!comm->resolve_remote_buffer_pointer(kClientRank, kValidInfoMr,
                                               /*offset=*/4096, /*bytes=*/1,
                                               &resolved, &device_idx),
          "out-of-range remote pointer resolution should fail");

  constexpr uint32_t kInvalidInfoMr = 9999;
  require(comm->wait_ipc_buffer(kClientRank, kInvalidInfoMr),
          "invalid ipc metadata fetch should succeed");
  require(!comm->resolve_remote_buffer_pointer(kClientRank, kInvalidInfoMr, 0,
                                               1, &resolved, &device_idx),
          "invalid ipc metadata should not resolve to a pointer");

  unsigned ack_req = comm->isend(kClientRank, ack_buf, 0, 1, 0, 0, true);
  require(ack_req != 0, "server ack isend failed");
  require(comm->wait_finish(ack_req), "server ack wait_finish failed");

  std::cout << "[SERVER][ipc-buffer-meta] OK" << std::endl;
  return 0;
}

int unique_local_port() { return 19000 + static_cast<int>(::getpid() % 1000); }

}  // namespace

void test_transport_communicator_local() {
  run_case("transport integration", "communicator local mr lifecycle", [] {
    auto comm = make_communicator(/*gpu=*/0, /*rank=*/0, /*world_size=*/1,
                                  /*exchanger_ip=*/"0.0.0.0",
                                  /*exchanger_port=*/unique_local_port(),
                                  UKernel::Transport::PreferredTransport::Auto);

    GPU_RT_CHECK(gpuSetDevice(0));
    void* buf = nullptr;
    GPU_RT_CHECK(gpuMalloc(&buf, 4096));
    auto free_buf = uccl::finally([&] { gpuFree(buf); });

    MR mr0 = comm->reg_mr(buf, 4096);
    MR mr1 = comm->reg_mr(buf, 4096);
    require(mr0.id == mr1.id,
            "reg_mr should be idempotent for the same buffer");
    require(comm->get_local_mr(static_cast<char*>(buf) + 128).id == mr0.id,
            "communicator local MR range lookup failed");
    require(
        comm->get_local_mr(mr0.id).address == reinterpret_cast<uint64_t>(buf),
        "communicator local MR lookup by id failed");

    require(comm->dereg_mr(buf), "dereg_mr failed");
    MR mr2 = comm->reg_mr(buf, 4096);
    require(mr2.id != mr0.id,
            "re-registering after dereg_mr should allocate a new MR id");
  });
}

int test_transport_communicator(int argc, char** argv) {
  std::string role = get_arg(argc, argv, "--role", "");
  std::string test_case = get_arg(argc, argv, "--case", "basic");
  std::string exchanger_ip = get_arg(argc, argv, "--exchanger-ip", "");
  int exchanger_port = get_int_arg(argc, argv, "--exchanger-port", 6979);
  std::string transport = get_arg(argc, argv, "--transport", "auto");

  if (role.empty()) role = get_arg(argc, argv, "-r", "");
  if (role.empty()) {
    std::cerr << "transport communicator test requires --role=server|client\n";
    return 1;
  }

  Scenario scenario = get_scenario(test_case);
  if (exchanger_ip.empty()) {
    exchanger_ip = (role == "server") ? "0.0.0.0" : "127.0.0.1";
  }

  try {
    UKernel::Transport::PreferredTransport preferred_transport =
        UKernel::Transport::PreferredTransport::Auto;
    if (transport == "ipc") {
      preferred_transport = UKernel::Transport::PreferredTransport::Ipc;
    } else if (transport == "uccl") {
      preferred_transport = UKernel::Transport::PreferredTransport::Uccl;
    } else if (transport != "auto") {
      throw std::invalid_argument("unknown transport override: " + transport);
    }
    if (scenario.name == "ipc-buffer-meta") {
      if (role == "server") {
        return run_ipc_buffer_metadata_server(exchanger_ip, exchanger_port,
                                              preferred_transport);
      }
      if (role == "client") {
        return run_ipc_buffer_metadata_client(exchanger_ip, exchanger_port,
                                              preferred_transport);
      }
    }
    if (role == "server") {
      return run_receiver(scenario, exchanger_ip, exchanger_port,
                          preferred_transport);
    }
    if (role == "client") {
      return run_sender(scenario, exchanger_ip, exchanger_port,
                        preferred_transport);
    }
  } catch (std::exception const& e) {
    std::cerr << "[transport communicator][" << role << "][" << scenario.name
              << "] failed: " << e.what() << std::endl;
    return 2;
  }

  std::cerr << "Usage:\n"
            << "  --role=server|client --case=" << kScenarioUsage
            << " [--exchanger-ip IP] [--exchanger-port PORT]"
            << " [--transport auto|ipc|uccl]\n";
  return 1;
}
