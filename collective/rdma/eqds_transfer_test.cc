#include "transport.h"
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>
#include <sys/socket.h>
#include <sys/wait.h>

using namespace uccl;

// Build with kReceiverCCA = RECEIVER_CCA_EQDS. Both peers use one local NIC;
// no second machine or GPU allocation is needed for the registered host
// buffers.
static void run_peer(int socket, bool receiver) {
  // Also bound bootstrap/peer synchronization if the other process fails.
  alarm(60);
  RDMAEndpoint ep(ucclParamNUM_ENGINES());
  UCCL_CHECK(ep.initialize_engine_by_dev(0, true));
  ConnID conn;
  if (receiver) {
    uint16_t port = ep.get_p2p_listen_port(0);
    UCCL_CHECK(send_message(socket, &port, sizeof(port)) == sizeof(port));
    std::string remote_ip;
    int remote_dev, remote_gpu;
    conn = ep.test_uccl_accept(0, 0, remote_ip, &remote_dev, &remote_gpu);
  } else {
    uint16_t port;
    UCCL_CHECK(receive_message(socket, &port, sizeof(port)) == sizeof(port));
    conn = ep.test_uccl_connect(0, 0, 0, 0, ep.get_p2p_listen_ip(0), port);
  }
  auto* flow = static_cast<UcclFlow*>(conn.context);
  std::vector<uint8_t> buffer(32 << 20);
  Mhandle* mh;
  UCCL_CHECK(ep.uccl_regmr(flow, buffer.data(), buffer.size(), 0, &mh) == 0);
  int const sizes[] = {1, 1024, 65536, 1 << 20, 32 << 20};
  int sequence = 0;
  for (int size : sizes) {
    ++sequence;
    auto expected = [&](int i) {
      return static_cast<uint8_t>(i * 131u + sequence * 17u);
    };
    if (!receiver) {
      for (int i = 0; i < size; ++i) buffer[i] = expected(i);
    }
    ucclRequest req{};
    void* data = buffer.data();
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    auto check_deadline = [&] {
      UCCL_CHECK(std::chrono::steady_clock::now() < deadline)
          << "EQDS transfer stalled, size=" << size;
    };
    char ready = 1;
    if (receiver) {
      int capacity = size;
      UCCL_CHECK(ep.uccl_recv_async(flow, &mh, &data, &capacity, 1, &req) == 0);
      UCCL_CHECK(send_message(socket, &ready, 1) == 1);
    } else {
      UCCL_CHECK(receive_message(socket, &ready, 1) == 1);
      while (ep.uccl_send_async(flow, mh, data, size, &req)) check_deadline();
    }
    while (!ep.uccl_poll_ureq_once(&req)) check_deadline();
    if (receiver) {
      UCCL_CHECK(req.recv.data_len[0] == size);
      for (int i = 0; i < size; ++i) {
        UCCL_CHECK(buffer[i] == expected(i))
            << "Data mismatch, size=" << size << " offset=" << i;
      }
      std::cout << "PASS: " << size << " bytes\n" << std::flush;
      UCCL_CHECK(send_message(socket, &ready, 1) == 1);
    } else {
      UCCL_CHECK(receive_message(socket, &ready, 1) == 1);
    }
  }
  ep.uccl_deregmr(mh);
  close(socket);
}

int main() {
  if constexpr (kReceiverCCA != RECEIVER_CCA_EQDS) {
    std::cerr << "Rebuild with kReceiverCCA = RECEIVER_CCA_EQDS\n";
    return 77;
  }
  int sockets[2];
  UCCL_CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
  // Fork before either peer initializes CUDA or creates transport threads.
  pid_t child = fork();
  UCCL_CHECK(child >= 0);
  if (child == 0) {
    close(sockets[0]);
    run_peer(sockets[1], false);
    return 0;
  }
  close(sockets[1]);
  run_peer(sockets[0], true);
  int status;
  UCCL_CHECK(waitpid(child, &status, 0) == child);
  UCCL_CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  return 0;
}
