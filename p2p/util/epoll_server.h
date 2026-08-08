// Header file containing epoll-based server implementation
//
// Features:
//  - Auto port assignment: Set port=0 to let system assign available port
//  - Member function callbacks: Use lambdas or std::bind to pass object methods
//  - Synchronous (blocking) callback processing
//  - Edge-triggered epoll for high performance
//
// Usage:
//  1. With standalone function callback:
//     EpollServer server(9000, my_callback_func);
//
//  2. With member function callback:
//     MyClass obj;
//     EpollServer server(9000,
//         [&obj](const std::string& input, std::string& output) {
//         obj.process(input, output); });
//
//  3. With auto port assignment:
//     EpollServer server(0, my_callback);  // port=0 for auto-assign
//     server.start();
//     int actual_port = server.get_port();  // Get assigned port

#pragma once

#include "common.h"

// Message framing: [uint32_t len][payload bytes]

// ---------------------------
// Example MetaInfo struct
// ---------------------------
// struct MetaInfo {
//     int32_t client_id;
//     double value;
//     char message[64];
// };

// ---------------------------
// Connection state
// ---------------------------
struct Connection {
  int fd = -1;
  std::vector<char> in_buf;   // accumulate incoming bytes
  std::vector<char> out_buf;  // pending bytes to send
  uint32_t expected_len = 0;  // 0 means expecting header (len)
  bool closed = false;
  sockaddr_in addr{};
};

// ---------------------------
// Helper functions
// ---------------------------

// ---------------------------
// Epoll-based server class
// ---------------------------
class EpollServer {
 public:
  using MetaHandler = std::function<void(std::string const&, std::string&,
                                         std::string const&, int)>;

  EpollServer(int port, MetaHandler handler, int max_events = 1024);

  ~EpollServer();

  int get_listen_fd() const;
  bool start();

  void stop();

  // Run server in background (already start() does that). This function blocks
  // if needed.
  void run_forever();

  // Get the actual port being used (useful when port was set to 0 for
  // auto-assignment)
  int get_port() const;

 private:
  void event_loop();

  void accept_loop();

  void handle_read(int fd);

  void parse_messages(Connection& conn);

  void handle_write(int fd);

  void modify_epoll_out(int fd, bool enable);

  void remove_connection(int fd);

 private:
  int port_;
  MetaHandler handler_;
  int max_events_;
  int listen_fd_ = -1;
  int epoll_fd_ = -1;
  std::thread worker_thread_;
  std::atomic<bool> running_;
  std::mutex conns_mtx_;
  std::map<int, Connection> conns_;
};
