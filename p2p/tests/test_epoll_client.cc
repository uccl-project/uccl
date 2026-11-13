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

std::atomic<bool> hold_stale_event{false}, stale_event_ready{false},
    release_stale_event{false};
extern "C" int __real_epoll_wait(int, epoll_event*, int, int);
extern "C" int __wrap_epoll_wait(int fd, epoll_event* ev, int cap, int ms) {
  int n = __real_epoll_wait(fd, ev, cap, ms);
  for (int i = 0; i < n; ++i)
    if (hold_stale_event && (ev[i].events & (EPOLLERR | EPOLLHUP))) {
      stale_event_ready = true;
      while (!release_stale_event) std::this_thread::yield();
      break;
    }
  return n;
}

static bool preserves_reconnected_peer() {
  int listener = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  socklen_t address_len = sizeof(address);
  if (listener < 0) return false;
  if (bind(listener, (sockaddr*)&address, address_len) || listen(listener, 2) ||
      getsockname(listener, (sockaddr*)&address, &address_len))
    return false;
  std::atomic<bool> reset{false}, connected{false}, finish{false};
  std::thread server([&] {
    int original = accept(listener, nullptr, nullptr);
    if (original < 0) std::abort();
    while (!reset) std::this_thread::yield();
    linger rst{1, 0};
    setsockopt(original, SOL_SOCKET, SO_LINGER, &rst, sizeof rst);
    close(original);
    int replacement = accept(listener, nullptr, nullptr);
    if (replacement < 0) std::abort();
    connected = true;
    char frame[8];
    int n = recv(replacement, frame, 8, MSG_WAITALL);
    if (n == 8) send(replacement, frame, 8, MSG_NOSIGNAL);
    while (!finish) std::this_thread::yield();
    close(replacement);
  });
  EpollClient client;
  if (!client.start()) std::abort();
  auto old_key = client.connect_to_server("127.0.0.1", ntohs(address.sin_port));
  if (old_key.empty()) std::abort();
  hold_stale_event = true;
  reset = true;
  while (!stale_event_ready) std::this_thread::yield();
  bool retired = !client.send_meta(old_key, "original");
  auto new_key = client.connect_to_server("127.0.0.1", ntohs(address.sin_port));
  if (new_key.empty()) std::abort();
  while (!connected) std::this_thread::yield();
  std::atomic<bool> reply{false};
  bool sent = client.send_meta(
      new_key, "live", [&](std::string const& x) { reply = x == "live"; });
  release_stale_event = true;
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(2000);
  while (!reply && client.get_connection_count() != 0 &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  size_t count = client.get_connection_count();
  finish = true;
  client.stop();
  server.join();
  close(listener);
  printf(
      "pending_real_hup=%d retired=%d new_send_accepted=%d live_response=%d "
      "active_connections=%zu\n",
      int(stale_event_ready.load()), retired, sent, int(reply.load()), count);
  hold_stale_event = false;
  return retired && sent && reply && count == 1;
}

static void on_timeout(int) {
  static char const message[] = "FAIL: epoll client teardown timed out\n";
  (void)!write(STDERR_FILENO, message, sizeof(message) - 1);
  _exit(1);
}

int main() {
  std::signal(SIGALRM, on_timeout);
  alarm(60);
  if (!preserves_reconnected_peer()) return 1;
  constexpr int kRounds = 40;
  constexpr int kConnections = 32;
  EpollClient client;
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
