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
// in the whole verbs stack. Same behaviour, defined here to keep this test
// standalone.
int make_socket_non_blocking(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  if (flags == -1) return -1;
  return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

ssize_t try_send(int fd, char const* buf, size_t len) {
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
    server.stop();
  }
  if (failures == 0) std::printf("OK\n");
  return failures ? 1 : 0;
}
