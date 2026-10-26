// Exercise connection teardown while peer resets generate EPOLLHUP/EPOLLERR.
// This test uses loopback sockets only and needs no accelerator or RDMA device.
#include "util/epoll_client.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
#include <sys/socket.h>
#include <unistd.h>

// The normal definitions are in common.cc, which links the verbs stack.
int make_socket_non_blocking(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  return flags == -1 ? -1 : fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

ssize_t try_send(int fd, char const* buf, size_t len) {
  ssize_t n = ::send(fd, buf, len, MSG_NOSIGNAL);
  if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return 0;
  return n;
}

static void on_timeout(int) {
  static char const message[] = "FAIL: epoll client teardown timed out\n";
  (void)!write(STDERR_FILENO, message, sizeof(message) - 1);
  _exit(1);
}

int main() {
  std::signal(SIGALRM, on_timeout);
  alarm(60);
  constexpr int kRounds = 40;
  constexpr int kConnections = 32;
  for (int round = 0; round < kRounds; ++round) {
    int listener = socket(AF_INET, SOCK_STREAM, 0);
    if (listener < 0) return 1;
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) <
            0 ||
        listen(listener, kConnections) < 0)
      return 1;
    socklen_t len = sizeof(address);
    if (getsockname(listener, reinterpret_cast<sockaddr*>(&address), &len) < 0)
      return 1;
    int port = ntohs(address.sin_port);
    std::atomic<int> accepted{0};
    std::atomic<bool> release{false};
    std::thread server([&] {
      std::vector<int> peers;
      for (int i = 0; i < kConnections; ++i) {
        int fd = accept(listener, nullptr, nullptr);
        if (fd < 0) std::abort();
        peers.push_back(fd);
        accepted.fetch_add(1);
      }
      while (!release.load()) std::this_thread::yield();
      // Reset each peer to create HUP/ERR while the client is stopping.
      linger reset{1, 0};
      for (int fd : peers) {
        setsockopt(fd, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset));
        close(fd);
      }
    });

    EpollClient client;
    if (!client.start()) std::abort();
    for (int i = 1; i <= kConnections; ++i) {
      // 127/8 addresses share the same loopback listener but produce distinct
      // connection keys in EpollClient.
      std::string ip = "127.0.0." + std::to_string(i);
      if (client.connect_to_server(ip, port).empty()) std::abort();
    }
    while (accepted.load() != kConnections) std::this_thread::yield();
    if (client.get_connection_count() != kConnections) std::abort();
    release.store(true);
    client.stop();
    server.join();
    close(listener);
    if (client.get_connection_count() != 0) std::abort();
  }
  alarm(0);
  std::printf("PASS: %d peer-reset/stop rounds with %d connections\n", kRounds,
              kConnections);
  return 0;
}
