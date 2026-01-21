// Header file containing epoll-based client implementation with connection
// reuse

#pragma once

#include "define.h"  // reuse serialization, MetaInfo, helper functions

// ---------------------------
// Client connection state
// ---------------------------
struct ClientConnection {
  int fd = -1;
  std::string server_ip;
  int server_port = 0;
  bool connected = false;
  std::vector<char> in_buf;   // accumulate incoming bytes
  std::vector<char> out_buf;  // pending bytes to send
  uint32_t expected_len = 0;  // 0 means expecting header (len)
  std::queue<std::function<void(std::string const&)>> response_callbacks;
  std::chrono::steady_clock::time_point last_activity;
};

// ---------------------------
// Epoll-based client class with connection reuse
// ---------------------------
class EpollClient {
 public:
  using ResponseCallback = std::function<void(std::string const&)>;

  EpollClient(int max_events = 1024)
      : max_events_(max_events), running_(false) {}

  ~EpollClient() { stop(); }

  bool start() {
    if (running_) return false;

    epoll_fd_ = epoll_create1(0);
    if (epoll_fd_ < 0) {
      perror("epoll_create1");
      return false;
    }

    running_ = true;
    worker_thread_ = std::thread([this] { this->event_loop(); });
    return true;
  }

  void stop() {
    if (!running_) return;
    running_ = false;

    // Close all connections to wake up epoll_wait
    {
      std::lock_guard<std::mutex> lk(conns_mtx_);
      for (auto& kv : conns_) {
        if (kv.second.fd >= 0) {
          ::shutdown(kv.second.fd, SHUT_RDWR);
        }
      }
    }

    if (worker_thread_.joinable()) worker_thread_.join();
    ::close(epoll_fd_);

    // Close all client fds
    std::lock_guard<std::mutex> lk(conns_mtx_);
    for (auto& kv : conns_) {
      if (kv.second.fd >= 0) ::close(kv.second.fd);
    }
    conns_.clear();
  }

  // Get or create a connection to server
  // Returns connection key (server_ip:port)
  std::string connect_to_server(std::string const& server_ip, int server_port) {
    std::string conn_key = server_ip + ":" + std::to_string(server_port);

    std::lock_guard<std::mutex> lk(conns_mtx_);
    auto it = conns_.find(conn_key);
    if (it != conns_.end() && it->second.connected) {
      // Connection already exists and is connected
      it->second.last_activity = std::chrono::steady_clock::now();
      return conn_key;
    }

    // Create new connection
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
      perror("socket");
      return "";
    }

    sockaddr_in srv{};
    srv.sin_family = AF_INET;
    srv.sin_port = htons(server_port);
    inet_pton(AF_INET, server_ip.c_str(), &srv.sin_addr);

    if (::connect(sock, (sockaddr*)&srv, sizeof(srv)) < 0) {
      perror("connect");
      ::close(sock);
      return "";
    }

    if (make_socket_non_blocking(sock) < 0) {
      perror("make_socket_non_blocking");
      ::close(sock);
      return "";
    }

    // Add to epoll
    epoll_event ev{};
    ev.data.ptr = new std::string(conn_key);  // store conn_key in ptr
    ev.events = EPOLLIN | EPOLLET | EPOLLOUT;
    if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, sock, &ev) < 0) {
      perror("epoll_ctl add");
      ::close(sock);
      delete static_cast<std::string*>(ev.data.ptr);
      return "";
    }

    // Create connection state
    ClientConnection conn;
    conn.fd = sock;
    conn.server_ip = server_ip;
    conn.server_port = server_port;
    conn.connected = true;
    conn.expected_len = 0;
    conn.last_activity = std::chrono::steady_clock::now();

    conns_[conn_key] = std::move(conn);

    std::cout << "Connected to " << server_ip << ":" << server_port
              << " (fd=" << sock << ")\n";
    return conn_key;
  }

  // Send serialized payload to server with callback for response
  bool send_meta(std::string const& conn_key, std::string const& payload,
                 ResponseCallback callback = nullptr) {
    uint32_t len = htonl((uint32_t)payload.size());
    std::string pkt;
    pkt.reserve(sizeof(len) + payload.size());
    pkt.append(reinterpret_cast<char const*>(&len), sizeof(len));
    pkt.append(payload);

    std::lock_guard<std::mutex> lk(conns_mtx_);
    auto it = conns_.find(conn_key);
    if (it == conns_.end() || !it->second.connected) {
      std::cerr << "Connection not found or not connected: " << conn_key
                << "\n";
      return false;
    }

    ClientConnection& conn = it->second;
    conn.last_activity = std::chrono::steady_clock::now();

    // Try to send immediately
    ssize_t sent = 0;
    if (conn.out_buf.empty()) {
      sent = try_send(conn.fd, pkt.data(), pkt.size());
      if (sent < 0) {
        std::cerr << "Error sending to " << conn_key << "\n";
        close_connection(conn_key);
        return false;
      }
    }

    // Buffer any remaining data
    if (sent < (ssize_t)pkt.size()) {
      conn.out_buf.insert(conn.out_buf.end(), pkt.data() + sent,
                          pkt.data() + pkt.size());
    }

    // Register callback for response
    if (callback) {
      conn.response_callbacks.push(callback);
    }

    return true;
  }

  // Get number of active connections
  size_t get_connection_count() const {
    std::lock_guard<std::mutex> lk(conns_mtx_);
    return conns_.size();
  }

 private:
  void event_loop() {
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
        std::string* conn_key_ptr = static_cast<std::string*>(ev.data.ptr);
        if (!conn_key_ptr) continue;

        std::string conn_key = *conn_key_ptr;

        if ((ev.events & EPOLLERR) || (ev.events & EPOLLHUP)) {
          LOG(WARNING) << "Error/HUP on connection: " << conn_key;
          close_connection(conn_key);
          continue;
        }

        if (ev.events & EPOLLIN) handle_read(conn_key);
        if (ev.events & EPOLLOUT) handle_write(conn_key);
      }
    }
  }

  void handle_read(std::string const& conn_key) {
    char buf[4096];

    std::lock_guard<std::mutex> lk(conns_mtx_);
    auto it = conns_.find(conn_key);
    if (it == conns_.end()) return;
    ClientConnection& conn = it->second;
    // std::cout << "received" << std::endl;
    while (true) {
      ssize_t count = ::recv(conn.fd, buf, sizeof(buf), 0);
      // std::cout << "received count: " << count << std::endl;
      if (count == -1) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) break;
        std::cerr << "recv error on " << conn_key << "\n";
        close_connection(conn_key);
        return;
      } else if (count == 0) {
        std::cout << "Server closed connection: " << conn_key << "\n";
        close_connection(conn_key);
        return;
      } else {
        conn.in_buf.insert(conn.in_buf.end(), buf, buf + count);
        conn.last_activity = std::chrono::steady_clock::now();
        // std::cout << "message received count: " << count << std::endl;
        parse_responses(conn);
      }
    }
  }

  void parse_responses(ClientConnection& conn) {
    // Parse responses with length framing: [uint32_t len][payload bytes]
    while (true) {
      if (conn.expected_len == 0) {
        // Need to read header (length)
        if (conn.in_buf.size() >= sizeof(uint32_t)) {
          uint32_t net_len;
          std::memcpy(&net_len, conn.in_buf.data(), sizeof(uint32_t));
          uint32_t len = ntohl(net_len);
          conn.expected_len = len;
          // Erase header
          conn.in_buf.erase(conn.in_buf.begin(),
                            conn.in_buf.begin() + sizeof(uint32_t));

          LOG(INFO) << "Parsed response header: expected_len=" << len;
        } else {
          break;  // Wait for more data
        }
      }

      if (conn.expected_len > 0) {
        if (conn.in_buf.size() >= conn.expected_len) {
          // Got full payload
          std::string response(conn.in_buf.begin(),
                               conn.in_buf.begin() + conn.expected_len);
          conn.in_buf.erase(conn.in_buf.begin(),
                            conn.in_buf.begin() + conn.expected_len);
          uint32_t got_len = conn.expected_len;
          conn.expected_len = 0;

          LOG(INFO) << "Got full response payload: len=" << got_len;

          // Call the next callback if available
          if (!conn.response_callbacks.empty()) {
            auto callback = conn.response_callbacks.front();
            conn.response_callbacks.pop();
            try {
              callback(response);
            } catch (std::exception const& e) {
              std::cerr << "Callback exception: " << e.what() << "\n";
            }
          } else {
            // No callback registered
            LOG(INFO) << "Received response (no callback): len="
                      << response.size() << "\n";
          }
          // Loop to see if more messages in buffer
        } else {
          break;  // Wait for more payload bytes
        }
      }
    }
  }

  void handle_write(std::string const& conn_key) {
    std::lock_guard<std::mutex> lk(conns_mtx_);
    auto it = conns_.find(conn_key);
    if (it == conns_.end()) return;
    ClientConnection& conn = it->second;

    while (!conn.out_buf.empty()) {
      ssize_t n = try_send(conn.fd, conn.out_buf.data(), conn.out_buf.size());
      if (n < 0) {
        std::cerr << "send error on " << conn_key << "\n";
        close_connection(conn_key);
        return;
      } else if (n == 0) {
        break;  // would block
      } else {
        conn.out_buf.erase(conn.out_buf.begin(), conn.out_buf.begin() + n);
        conn.last_activity = std::chrono::steady_clock::now();
      }
    }
  }

  void close_connection(std::string const& conn_key) {
    // Note: this should be called with conns_mtx_ locked or from event_loop
    auto it = conns_.find(conn_key);
    if (it == conns_.end()) return;

    if (it->second.fd >= 0) {
      // Remove from epoll first
      epoll_event ev{};
      if (epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, it->second.fd, &ev) < 0) {
        // ignore error
      }
      ::close(it->second.fd);
    }
    conns_.erase(it);
    LOG(INFO) << "Closed connection: " << conn_key;
  }

 private:
  int max_events_;
  int epoll_fd_ = -1;
  std::thread worker_thread_;
  std::atomic<bool> running_;
  mutable std::mutex conns_mtx_;
  std::map<std::string, ClientConnection> conns_;
};
