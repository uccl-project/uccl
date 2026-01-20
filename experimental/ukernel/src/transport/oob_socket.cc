#include "oob.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

namespace UKernel {
namespace Transport {

/* ----------------- Internal helpers ----------------- */

static ssize_t send_all_robust(int fd, char const* buf, size_t len) {
  size_t total = 0;
  while (total < len) {
    ssize_t n = ::send(fd, buf + total, len - total, 0);
    if (n > 0) {
      total += static_cast<size_t>(n);
      continue;
    }
    if (n == 0) {
      // peer closed
      return 0;
    }
    if (errno == EINTR) continue;
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    return -1;
  }
  return static_cast<ssize_t>(total);
}

// recv until newline, but with a max cap and ability to break on running=false
static bool recv_line_capped(int fd, std::string& out, size_t max_line_bytes,
                             std::atomic<bool>& running) {
  out.clear();
  char c;
  while (running.load(std::memory_order_relaxed)) {
    ssize_t n = ::recv(fd, &c, 1, 0);
    if (n > 0) {
      if (c == '\n') return true;
      out.push_back(c);
      if (out.size() > max_line_bytes) {
        // protocol violation: line too big
        return false;
      }
      continue;
    }
    if (n == 0) {
      // peer closed
      return false;
    }
    // n < 0
    if (errno == EINTR) continue;
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      // timed out; check running flag again
      continue;
    }
    return false;
  }
  return false;
}

static std::vector<std::string> split_tokens(std::string const& s) {
  std::istringstream iss(s);
  std::vector<std::string> toks;
  std::string t;
  while (iss >> t) toks.push_back(t);
  return toks;
}

static void add_conn_thread(std::vector<std::thread>& vec, std::mutex& mtx,
                            std::thread&& t) {
  std::lock_guard<std::mutex> lk(mtx);
  vec.emplace_back(std::move(t));
}

/* ----------------- Forward decl for friend ----------------- */
static void handle_connection_inner(
    int connfd,
    std::unordered_map<std::string, std::map<std::string, std::string>>& store,
    std::mutex& store_mutex, std::atomic<bool>& running,
    size_t max_line_bytes) {
  try {
    std::string line;
    while (running.load(std::memory_order_relaxed)) {
      if (!recv_line_capped(connfd, line, max_line_bytes, running)) break;

      auto toks = split_tokens(line);
      if (toks.empty()) {
        char const* r = "ERR\n";
        send_all_robust(connfd, r, ::strlen(r));
        continue;
      }
      std::string const& cmd = toks[0];

      if (cmd == "HSET") {
        if (toks.size() < 4 || ((toks.size() - 2) % 2 != 0)) {
          char const* r = "ERR\n";
          send_all_robust(connfd, r, ::strlen(r));
          continue;
        }
        std::string const& key = toks[1];
        {
          std::lock_guard<std::mutex> lk(store_mutex);
          auto& mp = store[key];
          for (size_t i = 2; i + 1 < toks.size(); i += 2) {
            mp[toks[i]] = toks[i + 1];
          }
        }
        char const* r = "OK\n";
        send_all_robust(connfd, r, ::strlen(r));
      } else if (cmd == "HGETALL") {
        if (toks.size() != 2) {
          char const* r = "ERR\n";
          send_all_robust(connfd, r, ::strlen(r));
          continue;
        }
        std::string const& key = toks[1];
        std::ostringstream oss;
        {
          std::lock_guard<std::mutex> lk(store_mutex);
          auto it = store.find(key);
          if (it == store.end()) {
            char const* r = "EMPTY\n";
            send_all_robust(connfd, r, ::strlen(r));
            continue;
          } else {
            auto const& mp = it->second;
            oss << "ARRAY " << mp.size();
            for (auto const& p : mp) {
              oss << " " << p.first << " " << p.second;
            }
            oss << "\n";
          }
        }
        const std::string resp = oss.str();
        send_all_robust(connfd, resp.c_str(), resp.size());
      } else {
        char const* r = "ERR\n";
        send_all_robust(connfd, r, ::strlen(r));
      }
    }
  } catch (std::exception const& e) {
    std::cerr << "[EXCEPTION] handle_connection_inner: " << e.what() << "\n";
  } catch (...) {
    std::cerr << "[EXCEPTION] handle_connection_inner: unknown\n";
  }

  ::shutdown(connfd, SHUT_RDWR);
  ::close(connfd);
}

void handle_connection(
    int connfd,
    std::unordered_map<std::string, std::map<std::string, std::string>>& store,
    std::mutex& store_mutex, std::atomic<bool>& running, size_t max_line_bytes,
    std::function<void(std::thread&&)> register_thread) {
  std::thread t(handle_connection_inner, connfd, std::ref(store),
                std::ref(store_mutex), std::ref(running), max_line_bytes);
  register_thread(std::move(t));
}

/* ----------------- SockExchanger implementation ----------------- */

SockExchanger::SockExchanger(bool is_server, std::string const& host, int port,
                             int timeout_ms, size_t max_line_bytes)
    : sock_fd_(-1),
      host_(host),
      port_(port),
      timeout_ms_(timeout_ms),
      is_server_(is_server),
      listen_fd_(-1),
      running_(false),
      max_line_bytes_(max_line_bytes) {
  if (is_server_) {
    if (!start_server()) {
      std::cerr << "SockExchanger: failed to start server on port " << port_
                << "\n";
    }
  } else {
    int count = 0;
    while (!running_.load(std::memory_order_relaxed) && count <= 10) {
      if (connect_client()) {
        std::cout << "SockExchanger: connect to " << host_ << ":" << port_
                  << " success.\n";
        break;
      }
      std::cout << "SockExchanger: connect to " << host_ << ":" << port_
                << " failed (errno=" << errno << "). Retrying...\n";
      std::this_thread::sleep_for(std::chrono::seconds(1));
      ++count;
    }
    if (!running_.load(std::memory_order_relaxed)) {
      std::cerr << "SockExchanger: failed to connect to " << host_ << ":"
                << port_ << " after " << count << " attempts.\n";
    }
  }
}

SockExchanger::~SockExchanger() {
  // orderly shutdown
  try {
    running_.store(false, std::memory_order_relaxed);

    // Break accept()
    if (listen_fd_ != -1) {
      ::shutdown(listen_fd_, SHUT_RDWR);
    }

    // Give the accept loop a brief moment to observe running_=false
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Close listening socket
    if (listen_fd_ != -1) {
      ::close(listen_fd_);
      listen_fd_ = -1;
    }

    // Close client socket if any
    if (sock_fd_ != -1) {
      ::shutdown(sock_fd_, SHUT_RDWR);
      ::close(sock_fd_);
      sock_fd_ = -1;
    }

    // Join accept thread
    if (server_thread_.joinable()) server_thread_.join();

    // Join all connection handler threads
    {
      std::lock_guard<std::mutex> lk(conn_threads_mutex_);
      for (auto& t : conn_threads_) {
        if (t.joinable()) t.join();
      }
      conn_threads_.clear();
    }
  } catch (...) {
    // do not throw in destructor
    std::cerr << "[WARN] SockExchanger::~SockExchanger caught exception during "
                 "shutdown\n";
  }
}

bool SockExchanger::valid() const {
  if (!running_.load(std::memory_order_relaxed)) return false;
  if (is_server_) return listen_fd_ != -1;
  return sock_fd_ != -1;
}

bool SockExchanger::send_cmd_and_recv(std::string const& cmd,
                                      std::string& resp) {
  if (sock_fd_ == -1) return false;
  std::string out = cmd;
  if (out.empty() || out.back() != '\n') out.push_back('\n');
  if (send_all_robust(sock_fd_, out.c_str(), out.size()) <= 0) return false;
  return recv_line_capped(sock_fd_, resp, max_line_bytes_, running_);
}

bool SockExchanger::publish(std::string const& key, Exchangeable const& obj) {
  auto kv = obj.to_map();

  // Build command: HSET key f1 v1 f2 v2 ... (no spaces in fields/values)
  std::ostringstream oss;
  oss << "HSET " << key;
  for (auto const& p : kv) {
    oss << " " << p.first << " " << p.second;
  }
  std::string cmd = oss.str();

  if (is_server_) {
    std::lock_guard<std::mutex> lk(store_mutex_);
    auto& mp = store_[key];
    for (auto const& p : kv) mp[p.first] = p.second;
    return true;
  } else {
    std::string resp;
    if (!send_cmd_and_recv(cmd, resp)) return false;
    return resp.rfind("OK", 0) == 0;
  }
}

bool SockExchanger::fetch(std::string const& key, Exchangeable& obj) {
  if (is_server_) {
    std::lock_guard<std::mutex> lk(store_mutex_);
    auto it = store_.find(key);
    if (it == store_.end()) return false;
    obj.from_map(it->second);
    return true;
  } else {
    std::string cmd = "HGETALL " + key;
    std::string resp;
    if (!send_cmd_and_recv(cmd, resp)) return false;
    if (resp.rfind("EMPTY", 0) == 0) return false;
    if (resp.rfind("ARRAY ", 0) == 0) {
      // format: ARRAY <n> f1 v1 f2 v2 ...
      std::istringstream iss(resp);
      std::string tag;
      size_t n;
      iss >> tag >> n;
      std::map<std::string, std::string> kv;
      for (size_t i = 0; i < n; ++i) {
        std::string f, v;
        if (!(iss >> f >> v)) return false;
        kv[f] = v;
      }
      obj.from_map(kv);
      return true;
    }
    return false;
  }
}

bool SockExchanger::wait_and_fetch(std::string const& key, Exchangeable& obj,
                                   int max_retries, int delay_ms) {
  if (max_retries > 0) {
    for (int i = 0; i < max_retries; ++i) {
      if (!running_.load(std::memory_order_relaxed)) return false;
      if (fetch(key, obj)) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }
  } else {  // -1 => forever but stop if running_ becomes false
    while (running_.load(std::memory_order_relaxed)) {
      if (fetch(key, obj)) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }
    return false;
  }
  return false;
}

bool SockExchanger::connect_client() {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    std::cerr << "SockExchanger: socket() failed: " << std::strerror(errno)
              << "\n";
    return false;
  }

  // set timeouts
  struct timeval tv;
  tv.tv_sec = timeout_ms_ / 1000;
  tv.tv_usec = (timeout_ms_ % 1000) * 1000;
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (char const*)&tv, sizeof tv);
  ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, (char const*)&tv, sizeof tv);

  struct sockaddr_in serv {};
  serv.sin_family = AF_INET;
  serv.sin_port = htons(port_);
  if (::inet_pton(AF_INET, host_.c_str(), &serv.sin_addr) <= 0) {
    ::close(fd);
    return false;
  }

  if (::connect(fd, (struct sockaddr*)&serv, sizeof(serv)) < 0) {
    ::close(fd);
    return false;
  }

  sock_fd_ = fd;
  running_.store(true, std::memory_order_relaxed);
  return true;
}

bool SockExchanger::start_server() {
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) {
    std::cerr << "SockExchanger: create listen socket failed: "
              << std::strerror(errno) << "\n";
    return false;
  }

  int opt = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  struct sockaddr_in addr {};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port_);

  if (::bind(listen_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
    std::cerr << "SockExchanger: bind failed: " << std::strerror(errno) << "\n";
    ::close(listen_fd_);
    listen_fd_ = -1;
    return false;
  }

  if (::listen(listen_fd_, 16) < 0) {
    std::cerr << "SockExchanger: listen failed: " << std::strerror(errno)
              << "\n";
    ::close(listen_fd_);
    listen_fd_ = -1;
    return false;
  }

  running_.store(true, std::memory_order_relaxed);

  // background accept loop
  server_thread_ = std::thread([this]() {
    try {
      // default per-connection timeout so recv() can exit when shutting down
      struct timeval conn_tv;
      conn_tv.tv_sec = 1;  // 1s
      conn_tv.tv_usec = 0;

      while (running_.load(std::memory_order_relaxed)) {
        struct sockaddr_in cli {};
        socklen_t clilen = sizeof(cli);
        int conn = ::accept(listen_fd_, (struct sockaddr*)&cli, &clilen);
        if (conn < 0) {
          if (!running_.load(std::memory_order_relaxed)) break;
          if (errno == EINTR) continue;
          // transient error; small backoff
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
          continue;
        }

        // Set timeouts on accepted socket so that recv() can wake up
        ::setsockopt(conn, SOL_SOCKET, SO_RCVTIMEO, (const char*)&conn_tv,
                     sizeof conn_tv);
        ::setsockopt(conn, SOL_SOCKET, SO_SNDTIMEO, (const char*)&conn_tv,
                     sizeof conn_tv);

        auto registrar = [this](std::thread&& t) {
          add_conn_thread(this->conn_threads_, this->conn_threads_mutex_,
                          std::move(t));
        };

        handle_connection(conn, this->store_, this->store_mutex_,
                          this->running_, this->max_line_bytes_, registrar);
      }
    } catch (const std::exception& e) {
      std::cerr << "[EXCEPTION] accept loop: " << e.what() << "\n";
    } catch (...) {
      std::cerr << "[EXCEPTION] accept loop: unknown\n";
    }
  });

  return true;
}

}  // namespace Transport
}  // namespace UKernel