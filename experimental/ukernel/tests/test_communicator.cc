#include "transport.h"
#include "util/util.h"
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
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

struct NotifierQueue {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<unsigned> q;
};

static int run_client() {
  auto cfg = std::make_shared<CommunicatorConfig>();
  auto comm =
      std::make_shared<Communicator>(client_gpu, client_rank, kWorldSize, cfg);

  if (!comm->connect_to(server_rank)) return 2;

  std::vector<uint8_t> sendbuf_h(kBytes);
  fill_pattern(sendbuf_h);

  GPU_RT_CHECK(gpuSetDevice(client_gpu));
  void* sendbuf_d = nullptr;
  GPU_RT_CHECK(gpuMalloc(&sendbuf_d, kBytes));
  auto free_buf = uccl::finally([&] { gpuFree(sendbuf_d); });

  GPU_RT_CHECK(
      gpuMemcpy(sendbuf_d, sendbuf_h.data(), kBytes, gpuMemcpyHostToDevice));

  MR local_mr = comm->reg_mr(sendbuf_d, kBytes);
  if (!comm->notify_mr(server_rank, local_mr)) return 3;

  MR remote_mr;
  if (!comm->wait_mr_notify(server_rank, remote_mr)) return 3;

  unsigned sreq = comm->isend(server_rank, sendbuf_d, 0, kBytes, local_mr.id,
                              remote_mr.id, true);

  if (!comm->wait_finish(sreq)) return 4;

  std::cout << "[CLIENT] OK\n";
  return 0;
}

static int run_server() {
  auto cfg = std::make_shared<CommunicatorConfig>();
  auto comm =
      std::make_shared<Communicator>(server_gpu, server_rank, kWorldSize, cfg);

  if (!comm->accept_from(client_rank)) return 2;

  GPU_RT_CHECK(gpuSetDevice(server_gpu));
  void* recvbuf_d = nullptr;
  GPU_RT_CHECK(gpuMalloc(&recvbuf_d, kBytes));
  auto free_buf = uccl::finally([&] { gpuFree(recvbuf_d); });

  MR local_mr = comm->reg_mr(recvbuf_d, kBytes);
  if (!comm->notify_mr(client_rank, local_mr)) return 3;

  MR remote_mr;
  if (!comm->wait_mr_notify(client_rank, remote_mr)) return 3;

  unsigned rreq = comm->irecv(client_rank, recvbuf_d, 0, kBytes, true);

  if (!comm->wait_finish(rreq)) return 4;

  std::vector<uint8_t> recvbuf_h(kBytes);
  GPU_RT_CHECK(
      gpuMemcpy(recvbuf_h.data(), recvbuf_d, kBytes, gpuMemcpyDeviceToHost));

  std::cout << (check_pattern(recvbuf_h) ? "[SERVER] OK\n"
                                         : "[SERVER] FAILED\n");
  return 0;
}

static int run_client_notifier_only() {
  auto cfg = std::make_shared<CommunicatorConfig>();
  auto comm =
      std::make_shared<Communicator>(client_gpu, client_rank, kWorldSize, cfg);

  if (!comm->connect_to(server_rank)) return 2;

  NotifierQueue nq;
  auto h = comm->register_completion_notifier(
      [&](unsigned id, std::chrono::steady_clock::time_point) {
        std::lock_guard<std::mutex> lk(nq.mu);
        nq.q.push_back(id);
        nq.cv.notify_all();
      });

  std::vector<uint8_t> sendbuf_h(kBytes);
  fill_pattern(sendbuf_h);

  GPU_RT_CHECK(gpuSetDevice(client_gpu));
  void* sendbuf_d = nullptr;
  GPU_RT_CHECK(gpuMalloc(&sendbuf_d, kBytes));
  auto free_buf = uccl::finally([&] { gpuFree(sendbuf_d); });

  GPU_RT_CHECK(
      gpuMemcpy(sendbuf_d, sendbuf_h.data(), kBytes, gpuMemcpyHostToDevice));

  MR local_mr = comm->reg_mr(sendbuf_d, kBytes);
  if (!comm->notify_mr(server_rank, local_mr)) return 3;

  MR remote_mr;
  if (!comm->wait_mr_notify(server_rank, remote_mr)) return 3;

  unsigned sreq = comm->isend(server_rank, sendbuf_d, 0, kBytes, local_mr.id,
                              remote_mr.id, true);

  {
    std::unique_lock<std::mutex> lk(nq.mu);
    if (!nq.cv.wait_for(lk, std::chrono::seconds(5),
                        [&] { return !nq.q.empty(); }))
      return 10;

    while (!nq.q.empty()) {
      unsigned id = nq.q.front();
      nq.q.pop_front();
      std::cout << "[CLIENT-N] notify req " << id << " sreq " << sreq << "\n";
    }
  }

  if (sreq == 0) return 11;
  return 0;
}

static int run_server_notifier_only() {
  auto cfg = std::make_shared<CommunicatorConfig>();
  auto comm =
      std::make_shared<Communicator>(server_gpu, server_rank, kWorldSize, cfg);

  if (!comm->accept_from(client_rank)) return 2;

  NotifierQueue nq;
  auto h = comm->register_completion_notifier(
      [&](unsigned id, std::chrono::steady_clock::time_point) {
        std::lock_guard<std::mutex> lk(nq.mu);
        nq.q.push_back(id);
        nq.cv.notify_all();
      });

  GPU_RT_CHECK(gpuSetDevice(server_gpu));
  void* recvbuf_d = nullptr;
  GPU_RT_CHECK(gpuMalloc(&recvbuf_d, kBytes));
  auto free_buf = uccl::finally([&] { gpuFree(recvbuf_d); });

  MR local_mr = comm->reg_mr(recvbuf_d, kBytes);
  if (!comm->notify_mr(client_rank, local_mr)) return 3;

  MR remote_mr;
  if (!comm->wait_mr_notify(client_rank, remote_mr)) return 3;

  unsigned rreq = comm->irecv(client_rank, recvbuf_d, 0, kBytes, true);

  {
    std::unique_lock<std::mutex> lk(nq.mu);
    if (!nq.cv.wait_for(lk, std::chrono::seconds(5),
                        [&] { return !nq.q.empty(); }))
      return 10;

    while (!nq.q.empty()) {
      unsigned id = nq.q.front();
      nq.q.pop_front();
      std::cout << "[SERVER-N] notify req " << id << " rreq " << rreq << "\n";
    }
  }

  std::vector<uint8_t> recvbuf_h(kBytes);
  GPU_RT_CHECK(
      gpuMemcpy(recvbuf_h.data(), recvbuf_d, kBytes, gpuMemcpyDeviceToHost));

  std::cout << (check_pattern(recvbuf_h) ? "[SERVER-N] OK\n"
                                         : "[SERVER-N] FAILED\n");
  return 0;
}

int test_communicator(int argc, char** argv) {
  std::string role = get_arg(argc, argv, "--role", "");
  if (role.empty()) role = get_arg(argc, argv, "-r", "");

  if (role == "server") return run_server();
  if (role == "client") return run_client();
  if (role == "server-notifier") return run_server_notifier_only();
  if (role == "client-notifier") return run_client_notifier_only();

  std::cerr << "Usage:\n"
            << "  --role=server\n"
            << "  --role=client\n"
            << "  --role=server-notifier\n"
            << "  --role=client-notifier\n";
  return 1;
}
