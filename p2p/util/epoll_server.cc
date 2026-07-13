#include "epoll_server.h"
#include "util/debug.h"

EpollServer::EpollServer(int port, MetaHandler handler, int max_events)
    : port_(port),
      handler_(std::move(handler)),
      max_events_(max_events),
      running_(false) {}

EpollServer::~EpollServer() { stop(); }

int EpollServer::get_listen_fd() const { return listen_fd_; }

bool EpollServer::start() {
  if (running_) return false;

  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) {
    perror("socket");
    return false;
  }

  int opt = 1;
  setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port_);

  if (bind(listen_fd_, (sockaddr*)&addr, sizeof(addr)) < 0) {
    perror("bind");
    ::close(listen_fd_);
    return false;
  }

  // If port was 0, get the actual assigned port
  if (port_ == 0) {
    sockaddr_in assigned_addr{};
    socklen_t addr_len = sizeof(assigned_addr);
    if (getsockname(listen_fd_, (sockaddr*)&assigned_addr, &addr_len) < 0) {
      perror("getsockname");
      ::close(listen_fd_);
      return false;
    }
    port_ = ntohs(assigned_addr.sin_port);
    std::cout << "System assigned port: " << port_ << "\n";
  }

  if (make_socket_non_blocking(listen_fd_) < 0) {
    perror("make nonblock");
    ::close(listen_fd_);
    return false;
  }

  if (listen(listen_fd_, SOMAXCONN) < 0) {
    perror("listen");
    ::close(listen_fd_);
    return false;
  }

  epoll_fd_ = epoll_create1(0);
  if (epoll_fd_ < 0) {
    perror("epoll_create1");
    ::close(listen_fd_);
    return false;
  }

  epoll_event ev{};
  ev.events = EPOLLIN | EPOLLET;  // edge-triggered accept
  ev.data.fd = listen_fd_;
  if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, listen_fd_, &ev) < 0) {
    perror("epoll_ctl add listen");
    ::close(listen_fd_);
    ::close(epoll_fd_);
    return false;
  }

  running_ = true;
  worker_thread_ = std::thread([this] { this->event_loop(); });
  return true;
}

void EpollServer::stop() {
  if (!running_) return;
  running_ = false;
  // wake up epoll_wait by closing listen_fd_ or using eventfd; simplest:
  // shutdown fd
  ::shutdown(listen_fd_, SHUT_RDWR);
  if (worker_thread_.joinable()) worker_thread_.join();
  ::close(listen_fd_);
  ::close(epoll_fd_);

  // close all client fds
  std::lock_guard<std::mutex> lk(conns_mtx_);
  for (auto& kv : conns_) {
    if (kv.second.fd >= 0) ::close(kv.second.fd);
  }
  conns_.clear();
}

void EpollServer::run_forever() {
  if (!start()) {
    std::cerr << "Failed to start server\n";
    return;
  }
  // block calling thread
  if (worker_thread_.joinable()) worker_thread_.join();
}

int EpollServer::get_port() const { return port_; }

void EpollServer::event_loop() {
  std::vector<epoll_event> events(max_events_);
  while (running_) {
    int n = epoll_wait(epoll_fd_, events.data(), max_events_, 1000);
    if (n < 0) {
      if (errno == EINTR) continue;
      perror("epoll_wait");
      break;
    }
    for (int i = 0; i < n; ++i) {
      epoll_event& ev = events[i];
      if ((ev.events & EPOLLERR) || (ev.events & EPOLLHUP)) {
        // error on fd
        int fd = ev.data.fd;
        remove_connection(fd);
        continue;
      }
      if (ev.data.fd == listen_fd_) {
        // Accept all incoming connections (edge-triggered)
        accept_loop();
      } else {
        // Client socket events
        if (ev.events & EPOLLIN) handle_read(ev.data.fd);
        if (ev.events & EPOLLOUT) handle_write(ev.data.fd);
      }
    }
  }
}

void EpollServer::accept_loop() {
  while (true) {
    sockaddr_in in_addr;
    socklen_t in_len = sizeof(in_addr);
    int infd = accept(listen_fd_, (sockaddr*)&in_addr, &in_len);
    if (infd < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        // all accepted
        break;
      } else {
        perror("accept");
        break;
      }
    }
    if (make_socket_non_blocking(infd) < 0) {
      perror("make_socket_non_blocking(infd)");
      ::close(infd);
      continue;
    }
    // add to epoll
    epoll_event event{};
    event.data.fd = infd;
    event.events =
        EPOLLIN | EPOLLET | EPOLLOUT;  // monitor both; EPOLLET for performance
    if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, infd, &event) < 0) {
      perror("epoll_ctl add client");
      ::close(infd);
      continue;
    }
    // create connection state
    Connection conn;
    conn.fd = infd;
    conn.addr = in_addr;
    conn.expected_len = 0;
    {
      std::lock_guard<std::mutex> lk(conns_mtx_);
      conns_.emplace(infd, std::move(conn));
    }
    char ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &in_addr.sin_addr, ip, sizeof(ip));
    std::cout << "Accepted connection fd=" << infd << " from " << ip << ":"
              << ntohs(in_addr.sin_port) << "\n";
  }
}

void EpollServer::handle_read(int fd) {
  // read as much as possible (edge-triggered)
  char buf[4096];
  while (true) {
    ssize_t count = ::recv(fd, buf, sizeof(buf), 0);
    if (count == -1) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) break;  // all data read
      // error
      remove_connection(fd);
      return;
    } else if (count == 0) {
      // peer closed
      remove_connection(fd);
      return;
    } else {
      // got bytes
      std::lock_guard<std::mutex> lk(conns_mtx_);
      auto it = conns_.find(fd);
      if (it == conns_.end()) {
        // connection not found
        continue;
      }
      Connection& conn = it->second;
      conn.in_buf.insert(conn.in_buf.end(), buf, buf + count);
      // try to parse messages (maybe multiple)
      parse_messages(conn);
    }
  }
}

void EpollServer::parse_messages(Connection& conn) {
  // We expect frames of the form: uint32_t len (network byte order) + payload
  while (true) {
    if (conn.expected_len == 0) {
      // need header
      if (conn.in_buf.size() >= sizeof(uint32_t)) {
        uint32_t net_len;
        std::memcpy(&net_len, conn.in_buf.data(), sizeof(uint32_t));
        uint32_t len = ntohl(net_len);
        conn.expected_len = len;
        // erase header
        conn.in_buf.erase(conn.in_buf.begin(),
                          conn.in_buf.begin() + sizeof(uint32_t));
      } else {
        break;  // wait for more
      }
    }
    if (conn.expected_len > 0) {
      if (conn.in_buf.size() >= conn.expected_len) {
        // got full payload
        std::string payload(conn.in_buf.begin(),
                            conn.in_buf.begin() + conn.expected_len);
        // consume payload
        conn.in_buf.erase(conn.in_buf.begin(),
                          conn.in_buf.begin() + conn.expected_len);
        conn.expected_len = 0;

        // Call handler directly (blocking) with payload and get response
        std::string response;
        // Extract IP and port from connection
        char client_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &conn.addr.sin_addr, client_ip, sizeof(client_ip));
        int client_port = ntohs(conn.addr.sin_port);

        try {
          handler_(payload, response, std::string(client_ip), client_port);
        } catch (std::exception const& e) {
          std::cerr << "Handler exception: " << e.what() << "\n";
          remove_connection(conn.fd);
          return;
        } catch (...) {
          std::cerr << "Handler exception (unknown)\n";
          remove_connection(conn.fd);
          return;
        }

        // Send response back to client with length header
        if (!response.empty()) {
          // Prepend length header (same format as client sends)
          uint32_t len = htonl((uint32_t)response.size());
          std::string pkt;
          pkt.reserve(sizeof(len) + response.size());
          pkt.append(reinterpret_cast<char const*>(&len), sizeof(len));
          pkt.append(response);

          ssize_t s = try_send(conn.fd, pkt.data(), pkt.size());
          if (s < 0) {
            // fatal send error -> close connection
            UCCL_LOG(ERROR) << "Error sending response on fd=" << conn.fd;
            remove_connection(conn.fd);
            return;
          } else if ((size_t)s < pkt.size()) {
            // partial send -> buffer remainder and ensure EPOLLOUT monitored
            conn.out_buf.insert(conn.out_buf.end(), pkt.data() + s,
                                pkt.data() + pkt.size());
            modify_epoll_out(conn.fd, true);
          }
        }
        // loop to see if more messages in buffer
      } else {
        break;  // wait for more payload bytes
      }
    }
  }
}

void EpollServer::handle_write(int fd) {
  std::lock_guard<std::mutex> lk(conns_mtx_);
  auto it = conns_.find(fd);
  if (it == conns_.end()) return;
  Connection& conn = it->second;
  while (!conn.out_buf.empty()) {
    ssize_t n = try_send(fd, conn.out_buf.data(), conn.out_buf.size());
    if (n < 0) {
      remove_connection(fd);
      return;
    } else if (n == 0) {
      // would block
      break;
    } else {
      conn.out_buf.erase(conn.out_buf.begin(), conn.out_buf.begin() + n);
    }
  }
  if (conn.out_buf.empty()) {
    // nothing left to send -> stop monitoring EPOLLOUT
    modify_epoll_out(fd, false);
  }
}

void EpollServer::modify_epoll_out(int fd, bool enable) {
  epoll_event ev{};
  ev.data.fd = fd;
  ev.events = EPOLLIN | EPOLLET;
  if (enable) ev.events |= EPOLLOUT;
  if (epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, fd, &ev) < 0) {
    // sometimes mod fails if fd is gone
    if (errno != ENOENT) perror("epoll_ctl mod");
  }
}

void EpollServer::remove_connection(int fd) {
  std::lock_guard<std::mutex> lk(conns_mtx_);
  auto it = conns_.find(fd);
  if (it == conns_.end()) return;
  ::close(it->second.fd);
  conns_.erase(it);
  if (epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr) < 0) {
    // ignore
  }
}
