#include "transport.h"
#include "test_utils.h"
#include "util/util.h"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

using CommunicatorConfig = UKernel::Transport::CommunicatorConfig;
using Communicator = UKernel::Transport::Communicator;
using MR = UKernel::Transport::MR;
using NamedMR = UKernel::Transport::NamedMR;
using NamedMRInfos = UKernel::Transport::NamedMRInfos;

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
constexpr std::chrono::seconds kAcceptTimeout(30);
constexpr uint64_t kNamedMrGeneration = 1;

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

void accept_with_retry(std::shared_ptr<Communicator> const& comm, int peer_rank,
                       char const* what) {
  auto deadline = std::chrono::steady_clock::now() + kAcceptTimeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (comm->accept_from(peer_rank)) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  require(false, what);
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

int run_exchange_client(std::string const& exchanger_ip, int exchanger_port,
                        UKernel::Transport::PreferredTransport preferred) {
  auto comm =
      make_communicator(kClientGpu, kClientRank, kWorldSize, exchanger_ip,
                        exchanger_port, preferred);
  require(comm->connect_to(kServerRank), "client connect_to failed");

  GPU_RT_CHECK(gpuSetDevice(kClientGpu));
  void* sendbuf_d = nullptr;
  GPU_RT_CHECK(gpuMalloc(&sendbuf_d, kMessageBytes));
  auto free_buf = uccl::finally([&] { gpuFree(sendbuf_d); });

  std::vector<uint8_t> send_host(kMessageBytes);
  fill_pattern(send_host, 0x10);
  GPU_RT_CHECK(
      gpuMemcpy(sendbuf_d, send_host.data(), send_host.size(), gpuMemcpyHostToDevice));

  MR send_mr = comm->reg_mr(sendbuf_d, kMessageBytes);
  NamedMRInfos remote_recv_infos{};
  if (comm->peer_transport_kind(kServerRank) ==
      UKernel::Transport::PeerTransportKind::Uccl) {
    require(comm->wait_named_mrs(kServerRank, kNamedMrGeneration, remote_recv_infos),
            "client wait_named_mrs failed");
  }
  uint32_t remote_recv_mr_id =
      remote_recv_infos.entries.empty() ? 0 : remote_recv_infos.entries.front().mr.id;

  unsigned send_req =
      comm->isend(kServerRank, sendbuf_d, 0, kMessageBytes, send_mr.id,
                  remote_recv_mr_id, true);
  require(send_req != 0, "client isend failed");
  require(comm->wait_finish(send_req), "client wait_finish(send) failed");

  std::cout << "[CLIENT][exchange] OK" << std::endl;
  return 0;
}

int run_exchange_server(std::string const& exchanger_ip, int exchanger_port,
                        UKernel::Transport::PreferredTransport preferred) {
  auto comm =
      make_communicator(kServerGpu, kServerRank, kWorldSize, exchanger_ip,
                        exchanger_port, preferred);
  accept_with_retry(comm, kClientRank, "server accept_from failed");

  GPU_RT_CHECK(gpuSetDevice(kServerGpu));
  void* recvbuf_d = nullptr;
  GPU_RT_CHECK(gpuMalloc(&recvbuf_d, kMessageBytes));
  auto free_buf = uccl::finally([&] { gpuFree(recvbuf_d); });
  GPU_RT_CHECK(gpuMemset(recvbuf_d, 0, kMessageBytes));

  MR recv_mr = comm->reg_mr(recvbuf_d, kMessageBytes);
  if (comm->peer_transport_kind(kClientRank) ==
      UKernel::Transport::PeerTransportKind::Uccl) {
    NamedMRInfos infos{};
    infos.generation = kNamedMrGeneration;
    infos.entries.push_back(NamedMR{0, recv_mr});
    require(comm->notify_named_mrs(kClientRank, kNamedMrGeneration, infos),
            "server notify_named_mrs failed");
  }

  unsigned recv_req = comm->irecv(kClientRank, recvbuf_d, 0, kMessageBytes, true);
  require(recv_req != 0, "server irecv failed");
  require(comm->wait_finish(recv_req), "server wait_finish(recv) failed");

  std::vector<uint8_t> host(kMessageBytes);
  GPU_RT_CHECK(
      gpuMemcpy(host.data(), recvbuf_d, host.size(), gpuMemcpyDeviceToHost));
  require(check_pattern(host, 0x10), "received payload pattern mismatch");

  std::cout << "[SERVER][exchange] OK" << std::endl;
  return 0;
}

int run_ipc_buffer_metadata_client(
    std::string const& exchanger_ip, int exchanger_port,
    UKernel::Transport::PreferredTransport preferred) {
  (void)preferred;
  (void)exchanger_port;
  auto comm =
      make_communicator(kClientGpu, kClientRank, kWorldSize, exchanger_ip,
                        exchanger_port, preferred);
  require(comm->same_host(kServerRank),
          "ipc-buffer-meta requires same-host peers");

  GPU_RT_CHECK(gpuSetDevice(kClientGpu));
  void* buf = nullptr;
  GPU_RT_CHECK(gpuMalloc(&buf, 4096));
  auto free_buf = uccl::finally([&] { gpuFree(buf); });

  constexpr uint32_t kValidIpcId = 4242;
  require(comm->notify_ipc_buffer(kServerRank, kValidIpcId, buf, 4096),
          "notify_ipc_buffer should succeed");

  constexpr uint32_t kInvalidIpcId = 9999;
  require(comm->notify_ipc_buffer(kServerRank, kInvalidIpcId, nullptr, 0),
          "invalid ipc metadata publish should still succeed");

  constexpr uint32_t kAckIpcId = 7777;
  require(comm->wait_ipc_buffer(kServerRank, kAckIpcId),
          "client wait_ipc_buffer(ack) should succeed");
  // Publish a completion marker so the server keeps the exchanger process
  // alive until the client has observed the ack.
  constexpr uint32_t kClientDoneIpcId = 8888;
  require(comm->notify_ipc_buffer(kServerRank, kClientDoneIpcId, nullptr, 0),
          "client notify_ipc_buffer(done) should succeed");

  std::cout << "[CLIENT][ipc-buffer-meta] OK" << std::endl;
  return 0;
}

int run_ipc_buffer_metadata_server(
    std::string const& exchanger_ip, int exchanger_port,
    UKernel::Transport::PreferredTransport preferred) {
  (void)preferred;
  (void)exchanger_port;
  auto comm =
      make_communicator(kServerGpu, kServerRank, kWorldSize, exchanger_ip,
                        exchanger_port, preferred);
  require(comm->same_host(kClientRank),
          "ipc-buffer-meta requires same-host peers");

  void* resolved = nullptr;
  int device_idx = -1;

  constexpr uint32_t kValidIpcId = 4242;
  require(comm->wait_ipc_buffer(kClientRank, kValidIpcId),
          "wait_ipc_buffer should succeed");
  require(comm->resolve_ipc_buffer_pointer(kClientRank, kValidIpcId, 128, 256,
                                           &resolved, &device_idx),
          "resolve_ipc_buffer_pointer should succeed");
  require(resolved != nullptr, "resolved remote pointer should not be null");
  require(device_idx == kClientGpu,
          "resolved remote pointer should preserve device index");
  require(!comm->resolve_ipc_buffer_pointer(kClientRank, kValidIpcId, 4096, 1,
                                            &resolved, &device_idx),
          "out-of-range ipc pointer resolution should fail");

  constexpr uint32_t kInvalidIpcId = 9999;
  require(comm->wait_ipc_buffer(kClientRank, kInvalidIpcId),
          "invalid ipc metadata fetch should succeed");
  require(!comm->resolve_ipc_buffer_pointer(kClientRank, kInvalidIpcId, 0, 1,
                                            &resolved, &device_idx),
          "invalid ipc metadata should not resolve to a pointer");

  constexpr uint32_t kAckIpcId = 7777;
  require(comm->notify_ipc_buffer(kClientRank, kAckIpcId, nullptr, 0),
          "server notify_ipc_buffer(ack) should succeed");
  // Wait for client completion marker to avoid tearing down the in-process
  // exchanger before the client has consumed the ack.
  constexpr uint32_t kClientDoneIpcId = 8888;
  require(comm->wait_ipc_buffer(kClientRank, kClientDoneIpcId),
          "server wait_ipc_buffer(done) should succeed");

  std::cout << "[SERVER][ipc-buffer-meta] OK" << std::endl;
  return 0;
}

int unique_local_port() { return 19000 + static_cast<int>(::getpid() % 1000); }

}  // namespace

void test_transport_communicator_local() {
  run_case("transport integration", "communicator local mr lifecycle", [] {
    auto comm = make_communicator(0, 0, 1, "0.0.0.0", unique_local_port(),
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
    require(comm->get_local_mr(mr0.id).address ==
                reinterpret_cast<uint64_t>(buf),
            "communicator local MR lookup by id failed");

    require(comm->dereg_mr(buf), "dereg_mr failed");
    MR mr2 = comm->reg_mr(buf, 4096);
    require(mr2.id != mr0.id,
            "re-registering after dereg_mr should allocate a new MR id");
  });
}

int test_transport_communicator(int argc, char** argv) {
  std::string role = get_arg(argc, argv, "--role", "");
  std::string test_case = get_arg(argc, argv, "--case", "exchange");
  std::string exchanger_ip = get_arg(argc, argv, "--exchanger-ip", "");
  int exchanger_port = get_int_arg(argc, argv, "--exchanger-port", 6979);
  std::string transport = get_arg(argc, argv, "--transport", "auto");

  if (role.empty()) role = get_arg(argc, argv, "-r", "");
  if (role.empty()) {
    std::cerr << "transport communicator test requires --role=server|client\n";
    return 1;
  }
  if (exchanger_ip.empty()) {
    exchanger_ip = (role == "server") ? "0.0.0.0" : "127.0.0.1";
  }

  try {
    UKernel::Transport::PreferredTransport preferred =
        UKernel::Transport::PreferredTransport::Auto;
    if (transport == "ipc") {
      preferred = UKernel::Transport::PreferredTransport::Ipc;
    } else if (transport == "uccl") {
      preferred = UKernel::Transport::PreferredTransport::Uccl;
    } else if (transport != "auto") {
      throw std::invalid_argument("unknown transport override: " + transport);
    }

    if (test_case == "ipc-buffer-meta") {
      if (role == "server") {
        return run_ipc_buffer_metadata_server(exchanger_ip, exchanger_port,
                                              preferred);
      }
      if (role == "client") {
        return run_ipc_buffer_metadata_client(exchanger_ip, exchanger_port,
                                              preferred);
      }
    } else if (test_case == "exchange") {
      if (role == "server") {
        return run_exchange_server(exchanger_ip, exchanger_port, preferred);
      }
      if (role == "client") {
        return run_exchange_client(exchanger_ip, exchanger_port, preferred);
      }
    } else {
      throw std::invalid_argument("unknown transport communicator test case: " +
                                  test_case);
    }
  } catch (std::exception const& e) {
    std::cerr << "[transport communicator][" << role << "][" << test_case
              << "] failed: " << e.what() << std::endl;
    return 2;
  }

  return 1;
}
