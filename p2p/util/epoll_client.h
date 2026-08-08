// Header file containing epoll-based client implementation with connection
// reuse

#pragma once

#include "common.h"  // reuse serialization, MetaInfo, helper functions

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

  EpollClient(int max_events = 1024);

  ~EpollClient();

  bool start();

  void stop();

  // Get or create a connection to server
  // Returns connection key (server_ip:port)
  std::string connect_to_server(std::string const& server_ip, int server_port);

  // Send serialized payload to server with callback for response
  bool send_meta(std::string const& conn_key, std::string const& payload,
                 ResponseCallback callback = nullptr);

  // Get number of active connections
  size_t get_connection_count() const;

 private:
  void event_loop();

  void handle_read(std::string const& conn_key);

  void parse_responses(ClientConnection& conn);

  void handle_write(std::string const& conn_key);

  void close_connection(std::string const& conn_key);

 private:
  int max_events_;
  int epoll_fd_ = -1;
  std::thread worker_thread_;
  std::atomic<bool> running_;
  mutable std::mutex conns_mtx_;
  std::map<std::string, ClientConnection> conns_;
};
