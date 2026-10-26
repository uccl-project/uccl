// Standalone test for the OOB metadata server in p2p/util/epoll_server.cc.
// Needs no GPU or RDMA device. A handler that throws must drop only the
// offending connection; the server must keep serving others and stop().
#include "util/epoll_server.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

// The two socket helpers below normally come from util/common.cc, which pulls
// in the whole verbs stack. Keep this test standalone and inject short writes
// or EAGAIN at the same send boundary.
static std::atomic<int> next_send_limit{-1};

int make_socket_non_blocking(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  if (flags == -1) return -1;
  return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

ssize_t try_send(int fd, char const* buf, size_t len) {
  int limit = next_send_limit.exchange(-1);
  if (limit >= 0) len = std::min(len, static_cast<size_t>(limit));
  if (len == 0) return 0;
  ssize_t n = ::send(fd, buf, len, MSG_NOSIGNAL);
  if (n < 0) return (errno == EAGAIN || errno == EWOULDBLOCK) ? 0 : -1;
  return n;
}

static int connect_to(int port) {
  int s = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(port);
  inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
  if (connect(s, (sockaddr*)&a, sizeof(a)) < 0) {
    perror("connect");
    exit(1);
  }
  timeval tv{3, 0};
  setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  return s;
}

static void send_frame(int s, std::string const& payload) {
  uint32_t len = htonl((uint32_t)payload.size());
  std::string pkt(reinterpret_cast<char const*>(&len), sizeof(len));
  pkt += payload;
  if (send(s, pkt.data(), pkt.size(), MSG_NOSIGNAL) != (ssize_t)pkt.size()) {
    perror("send");
    exit(1);
  }
}

static bool recv_all(int s, char* buf, size_t n) {
  size_t got = 0;
  while (got < n) {
    ssize_t r = recv(s, buf + got, n - got, 0);
    if (r <= 0) return false;
    got += r;
  }
  return true;
}

static bool recv_frame(int s, std::string* out) {
  uint32_t len;
  if (!recv_all(s, reinterpret_cast<char*>(&len), sizeof(len))) return false;
  out->resize(ntohl(len));
  return recv_all(s, out->data(), out->size());
}

static std::atomic<bool> handler_entered{false};
static std::atomic<bool> client_gone{false};

static void on_timeout(int) {
  static char const msg[] = "FAIL: server hung (deadlock in event loop)\n";
  (void)!write(STDOUT_FILENO, msg, sizeof(msg) - 1);
  _exit(1);
}

int main() {
  signal(SIGALRM, on_timeout);
  alarm(20);

  int failures = 0;
  {
    EpollServer server(0, [](std::string const& in, std::string& out,
                             std::string const&, int) {
      if (in == "boom") throw std::runtime_error("bad meta");
      if (in == "boom-unknown") throw 42;
      if (in == "no-reply") return;
      if (in == "wait-for-reset") {
        // Reply only after the client has reset the connection, so the
        // response send fails.
        handler_entered = true;
        while (!client_gone) std::this_thread::yield();
      }
      out = "pong:" + in;
    });
    if (!server.start()) {
      std::printf("FAIL: server did not start\n");
      return 1;
    }

    for (char const* bad : {"boom", "boom-unknown"}) {
      int a = connect_to(server.get_port());
      send_frame(a, bad);
      std::string reply;
      if (recv_frame(a, &reply)) {
        std::printf("FAIL: %s: connection got a reply\n", bad);
        ++failures;
      }
      close(a);

      int b = connect_to(server.get_port());
      send_frame(b, "ping");
      if (!recv_frame(b, &reply) || reply != "pong:ping") {
        std::printf("FAIL: server stopped serving after '%s'\n", bad);
        ++failures;
      }
      close(b);
    }

    // A response send that fails because the peer reset the connection.
    int c = connect_to(server.get_port());
    send_frame(c, "wait-for-reset");
    while (!handler_entered) std::this_thread::yield();
    linger lg{1, 0};
    setsockopt(c, SOL_SOCKET, SO_LINGER, &lg, sizeof(lg));
    close(c);
    client_gone = true;
    int d = connect_to(server.get_port());
    send_frame(d, "ping");
    std::string reply;
    if (!recv_frame(d, &reply) || reply != "pong:ping") {
      std::printf("FAIL: server stopped serving after a failed send\n");
      ++failures;
    }
    close(d);

    for (int first_send_limit : {-1, 2, 0}) {
      int fd = connect_to(server.get_port());
      std::string requests, expected;
      for (std::string const payload : {"first", "no-reply", "second"}) {
        uint32_t len = htonl(static_cast<uint32_t>(payload.size()));
        requests.append(reinterpret_cast<char const*>(&len), sizeof(len));
        requests.append(payload);
        if (payload == "no-reply") continue;
        std::string response = "pong:" + payload;
        len = htonl(static_cast<uint32_t>(response.size()));
        expected.append(reinterpret_cast<char const*>(&len), sizeof(len));
        expected.append(response);
      }
      // Queue both replies before draining a short write or an EAGAIN result.
      next_send_limit = first_send_limit;
      if (send(fd, requests.data(), requests.size(), MSG_NOSIGNAL) !=
          static_cast<ssize_t>(requests.size())) {
        std::printf("FAIL: could not send pipelined requests\n");
        ++failures;
      } else {
        std::string actual(expected.size(), '\0');
        if (!recv_all(fd, actual.data(), actual.size()) || actual != expected) {
          std::printf("FAIL: response order after first send limit %d\n",
                      first_send_limit);
          ++failures;
        }
        send_frame(fd, "ping");
        if (!recv_frame(fd, &reply) || reply != "pong:ping") {
          std::printf("FAIL: connection unusable after pipelined replies\n");
          ++failures;
        }
      }
      close(fd);
    }
    server.stop();
  }
  if (failures == 0) std::printf("OK\n");
  return failures ? 1 : 0;
}
