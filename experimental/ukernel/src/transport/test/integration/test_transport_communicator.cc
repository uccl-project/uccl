#include "../../util/utils.h"
#include "test_utils.h"
#include "transport.h"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

using CommunicatorConfig = UKernel::Transport::CommunicatorConfig;
using Communicator = UKernel::Transport::Communicator;
using MR = UKernel::Transport::MR;
using LocalSlice = UKernel::Transport::LocalSlice;
using RemoteSlice = UKernel::Transport::RemoteSlice;

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
constexpr uint32_t kClientSendBufferId = 1001;
constexpr uint32_t kServerRecvBufferId = 2001;

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

bool setup_bidirectional_peer(std::shared_ptr<Communicator> const& comm,
                              int rank, int peer_rank) {
  if (rank < peer_rank) {
    return comm->connect(peer_rank) && comm->accept(peer_rank);
  }
  return comm->accept(peer_rank) && comm->connect(peer_rank);
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
  auto comm = make_communicator(kClientGpu, kClientRank, kWorldSize,
                                exchanger_ip, exchanger_port, preferred);
  require(setup_bidirectional_peer(comm, kClientRank, kServerRank),
          "client bidirectional connect/accept failed");

  GPU_RT_CHECK(gpuSetDevice(kClientGpu));
  void* sendbuf_d = nullptr;
  GPU_RT_CHECK(gpuMalloc(&sendbuf_d, kMessageBytes));
  auto free_buf = UKernel::Transport::finally([&] { gpuFree(sendbuf_d); });

  std::vector<uint8_t> send_host(kMessageBytes);
  fill_pattern(send_host, 0x10);
  GPU_RT_CHECK(gpuMemcpy(sendbuf_d, send_host.data(), send_host.size(),
                         gpuMemcpyHostToDevice));

  require(comm->reg_mr(kClientSendBufferId, sendbuf_d, kMessageBytes, false),
          "client reg_mr failed");
  uint32_t remote_recv_buffer_id = 0;
  auto peer_kind = comm->peer_transport_kind(kServerRank);
  if (peer_kind == UKernel::Transport::PeerTransportKind::Uccl) {
    require(comm->wait_mr(kServerRank, kServerRecvBufferId),
            "client wait_mr failed");
    (void)comm->get_mr(kServerRank, kServerRecvBufferId);
    remote_recv_buffer_id = kServerRecvBufferId;
  }
  std::optional<RemoteSlice> dst_hint = std::nullopt;
  if (remote_recv_buffer_id != 0) {
    dst_hint = RemoteSlice{remote_recv_buffer_id, 0};
  }
  unsigned send_req = comm->isend(
      kServerRank, LocalSlice{kClientSendBufferId, 0, kMessageBytes}, dst_hint);
  require(send_req != 0, "client isend failed");
  require(comm->wait_finish(send_req), "client wait_finish(send) failed");

  std::cout << "[CLIENT][exchange] OK" << std::endl;
  return 0;
}

int run_exchange_server(std::string const& exchanger_ip, int exchanger_port,
                        UKernel::Transport::PreferredTransport preferred) {
  auto comm = make_communicator(kServerGpu, kServerRank, kWorldSize,
                                exchanger_ip, exchanger_port, preferred);
  require(setup_bidirectional_peer(comm, kServerRank, kClientRank),
          "server bidirectional connect/accept failed");

  GPU_RT_CHECK(gpuSetDevice(kServerGpu));
  void* recvbuf_d = nullptr;
  GPU_RT_CHECK(gpuMalloc(&recvbuf_d, kMessageBytes));
  auto free_buf = UKernel::Transport::finally([&] { gpuFree(recvbuf_d); });
  GPU_RT_CHECK(gpuMemset(recvbuf_d, 0, kMessageBytes));

  require(comm->reg_mr(kServerRecvBufferId, recvbuf_d, kMessageBytes, true),
          "server reg_mr failed");
  auto peer_kind = comm->peer_transport_kind(kClientRank);
  (void)peer_kind;

  unsigned recv_req =
      comm->irecv(kClientRank, LocalSlice{kServerRecvBufferId, 0, kMessageBytes});
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
  auto comm = make_communicator(kClientGpu, kClientRank, kWorldSize,
                                exchanger_ip, exchanger_port, preferred);
  require(comm->same_host(kServerRank),
          "ipc-buffer-meta requires same-host peers");

  GPU_RT_CHECK(gpuSetDevice(kClientGpu));
  void* buf = nullptr;
  GPU_RT_CHECK(gpuMalloc(&buf, 4096));
  auto free_buf = UKernel::Transport::finally([&] { gpuFree(buf); });

  constexpr uint32_t kValidIpcId = 4242;
  require(comm->reg_ipc(kValidIpcId, buf, 4096, true), "reg_ipc should succeed");

  constexpr uint32_t kInvalidIpcId = 9999;
  require(comm->reg_ipc(kInvalidIpcId, nullptr, 0, true),
          "invalid ipc metadata publish should still succeed");

  constexpr uint32_t kAckIpcId = 7777;
  require(comm->wait_ipc(kServerRank, kAckIpcId),
          "client wait_ipc(ack) should succeed");
  // Publish a completion marker so the server keeps the exchanger process
  // alive until the client has observed the ack.
  constexpr uint32_t kClientDoneIpcId = 8888;
  require(comm->reg_ipc(kClientDoneIpcId, nullptr, 0, true),
          "client reg_ipc(done) should succeed");

  std::cout << "[CLIENT][ipc-buffer-meta] OK" << std::endl;
  return 0;
}

int run_ipc_buffer_metadata_server(
    std::string const& exchanger_ip, int exchanger_port,
    UKernel::Transport::PreferredTransport preferred) {
  (void)preferred;
  (void)exchanger_port;
  auto comm = make_communicator(kServerGpu, kServerRank, kWorldSize,
                                exchanger_ip, exchanger_port, preferred);
  require(comm->same_host(kClientRank),
          "ipc-buffer-meta requires same-host peers");

  constexpr uint32_t kValidIpcId = 4242;
  require(comm->wait_ipc(kClientRank, kValidIpcId), "wait_ipc should succeed");
  auto ipc = comm->get_ipc(kClientRank, kValidIpcId);
  require(ipc.direct_ptr != nullptr,
          "get_ipc should resolve a non-null direct_ptr for valid IPC");
  require(ipc.device_idx == kClientGpu,
          "get_ipc should preserve remote device index");
  bool out_of_range_ok = (4096 > ipc.bytes || 1 > (ipc.bytes - 4096));
  require(out_of_range_ok, "out-of-range span check should fail");

  constexpr uint32_t kInvalidIpcId = 9999;
  require(comm->wait_ipc(kClientRank, kInvalidIpcId),
          "invalid ipc metadata fetch should succeed");
  bool invalid_failed = false;
  try {
    (void)comm->get_ipc(kClientRank, kInvalidIpcId);
  } catch (...) {
    invalid_failed = true;
  }
  require(invalid_failed, "invalid ipc metadata should not resolve to IPC");

  constexpr uint32_t kAckIpcId = 7777;
  require(comm->reg_ipc(kAckIpcId, nullptr, 0, true),
          "server reg_ipc(ack) should succeed");
  // Wait for client completion marker to avoid tearing down the in-process
  // exchanger before the client has consumed the ack.
  constexpr uint32_t kClientDoneIpcId = 8888;
  require(comm->wait_ipc(kClientRank, kClientDoneIpcId),
          "server wait_ipc(done) should succeed");

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
    auto free_buf = UKernel::Transport::finally([&] { gpuFree(buf); });

    constexpr uint32_t kLocalBufferId = 1;
    require(comm->reg_mr(kLocalBufferId, buf, 4096, false), "reg_mr#0 failed");
    MR mr0 = comm->get_mr(kLocalBufferId);
    require(comm->reg_mr(kLocalBufferId, buf, 4096, false), "reg_mr#1 failed");
    MR mr1 = comm->get_mr(kLocalBufferId);
    require(mr0.address == mr1.address && mr0.length == mr1.length,
            "reg_mr should be idempotent for the same buffer");
    require(
        comm->get_mr(kLocalBufferId).address == reinterpret_cast<uint64_t>(buf),
        "communicator local MR lookup by buffer_id failed");

    require(comm->dereg_mr(kLocalBufferId), "dereg_mr failed");
    require(comm->reg_mr(kLocalBufferId, buf, 4096, false), "reg_mr#2 failed");
    MR mr2 = comm->get_mr(kLocalBufferId);
    require(mr2.address == mr0.address && mr2.length == mr0.length,
            "re-registering after dereg_mr should preserve MR metadata");
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
    } else if (transport == "tcp") {
      preferred = UKernel::Transport::PreferredTransport::Tcp;
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
