#include "transport.h"
#include "util/util.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using CommunicatorConfig = UKernel::Transport::CommunicatorConfig;
using Communicator = UKernel::Transport::Communicator;
using MR = UKernel::Transport::MR;

static constexpr int kWorldSize = 2;
static constexpr int client_gpu = 0;
static constexpr int server_gpu = 0;
static constexpr int client_rank = 1;
static constexpr int server_rank = 0;

namespace {
constexpr size_t kBytes = 4 * 1024;

static void fill_pattern(std::vector<uint8_t>& buf) {
  for (size_t i = 0; i < buf.size(); ++i)
    buf[i] = static_cast<uint8_t>(i & 0xFF);
}

static bool check_pattern(std::vector<uint8_t> const& buf) {
  for (size_t i = 0; i < buf.size(); ++i) {
    if (buf[i] != static_cast<uint8_t>(i & 0xFF)) {
      std::cerr << "[SERVER] mismatch at " << i << "\n";
      return false;
    }
  }
  return true;
}

static std::string get_arg(int argc, char** argv, char const* key,
                           char const* def) {
  // --role=server --role server
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], key, std::strlen(key)) == 0) {
      char const* p = argv[i] + std::strlen(key);
      if (*p == '=') return std::string(p + 1);
      if (*p == '\0' && i + 1 < argc) return std::string(argv[i + 1]);
    }
  }
  return std::string(def);
}
}  // namespace

static int run_client() {
  auto cfg = std::make_shared<CommunicatorConfig>();
  auto comm =
      std::make_shared<Communicator>(client_gpu, client_rank, kWorldSize, cfg);

  int peer_rank = server_rank;
  if (!comm->connect_to(peer_rank)) {
    std::cerr << "[CLIENT] connect_to failed\n";
    return 2;
  }

  // host pattern
  std::vector<uint8_t> sendbuf_h(kBytes);
  fill_pattern(sendbuf_h);

  // device buffer
  GPU_RT_CHECK(gpuSetDevice(client_gpu));
  void* sendbuf_d = nullptr;
  GPU_RT_CHECK(gpuMalloc(&sendbuf_d, kBytes));
  auto send_free = uccl::finally([&] {
    if (sendbuf_d) GPU_RT_CHECK(gpuFree(sendbuf_d));
  });

  GPU_RT_CHECK(
      gpuMemcpy(sendbuf_d, sendbuf_h.data(), kBytes, gpuMemcpyHostToDevice));

  // only RDMA
  MR local_mr = comm->reg_mr(sendbuf_d, kBytes);
  if (!comm->notify_mr(peer_rank, local_mr)) {
    return 3;
  }
  MR remote_mr;
  if (!comm->wait_mr_notify(peer_rank, remote_mr)) {
    return 3;
  }

  unsigned sreq =
      comm->isend(peer_rank, sendbuf_d, 0, kBytes, local_mr.id, remote_mr.id,
                  /*on_gpu*/ true);

  if (!comm->wait_finish(sreq)) {
    std::cerr << "[CLIENT] wait_finish(send) failed\n";
    return 4;
  }

  std::cout << "[CLIENT] Send done\n";
  comm.reset();
  return 0;
}

static int run_server() {
  auto cfg = std::make_shared<CommunicatorConfig>();
  auto comm =
      std::make_shared<Communicator>(server_gpu, server_rank, kWorldSize, cfg);

  int peer_rank = client_rank;
  if (!comm->accept_from(peer_rank)) {
    std::cerr << "[SERVER] accept_from failed\n";
    return 2;
  }

  GPU_RT_CHECK(gpuSetDevice(server_gpu));
  void* recvbuf_d = nullptr;
  GPU_RT_CHECK(gpuMalloc(&recvbuf_d, kBytes));
  auto recv_free = uccl::finally([&] {
    if (recvbuf_d) GPU_RT_CHECK(gpuFree(recvbuf_d));
  });

  MR local_mr = comm->reg_mr(recvbuf_d, kBytes);
  if (!comm->notify_mr(peer_rank, local_mr)) {
    return 3;
  }
  MR remote_mr;
  if (!comm->wait_mr_notify(peer_rank, remote_mr)) {
    return 3;
  }

  unsigned rreq = comm->irecv(peer_rank, recvbuf_d, 0, kBytes,
                              /*on_gpu*/ true);

  if (!comm->wait_finish(rreq)) {
    std::cerr << "[SERVER] wait_finish(recv) failed\n";
    return 4;
  }

  // copy back and check
  std::vector<uint8_t> recvbuf_h(kBytes, 0);
  GPU_RT_CHECK(
      gpuMemcpy(recvbuf_h.data(), recvbuf_d, kBytes, gpuMemcpyDeviceToHost));

  bool ok = check_pattern(recvbuf_h);
  std::cout << (ok ? "[SERVER] OK\n" : "[SERVER] FAILED\n");

  comm.reset();
  return ok ? 0 : 5;
}

int test_communicator(int argc, char** argv) {
  std::string role = get_arg(argc, argv, "--role", "");
  if (role.empty()) role = get_arg(argc, argv, "-r", "");

  if (role != "server" && role != "client") {
    std::cerr << "Usage:\n"
              << "  " << argv[0] << " --role=server\n"
              << "  " << argv[0] << " --role=client\n";
    return 1;
  }

  if (role == "server") return run_server();
  return run_client();
}
