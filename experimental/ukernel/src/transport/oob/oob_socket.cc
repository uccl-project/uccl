#include "oob.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <atomic>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

namespace UKernel {
namespace Transport {

/* ----------------- Internal helpers ----------------- */
static std::string compose_store_key(std::string const& ns,
                                     std::string const& key) {
  return ns + "::" + key;
}

static ssize_t send_all_robust(int fd, char const* buf, size_t len) {
  size_t total = 0;
  while (total < len) {
    // Avoid process-killing SIGPIPE when peer closes early.
#ifdef MSG_NOSIGNAL
    ssize_t n = ::send(fd, buf + total, len - total, MSG_NOSIGNAL);
#else
    ssize_t n = ::send(fd, buf + total, len - total, 0);
#endif
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
  while (iss >> std::quoted(t)) toks.push_back(t);
  return toks;
}

static uint64_t now_ns_monotonic() {
  auto const now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

static void add_conn_thread(std::vector<std::thread>& vec, std::mutex& mtx,
                            std::thread&& t) {
  std::lock_guard<std::mutex> lk(mtx);
  vec.emplace_back(std::move(t));
}

static void run_callbacks_noexcept(
    std::vector<std::function<void(std::string const&, std::string const&,
                                   uint64_t)>> const& callbacks,
    std::string const& ns, std::string const& key, uint64_t tag) {
  for (auto const& cb : callbacks) {
    try {
      cb(ns, key, tag);
    } catch (std::exception const& e) {
      std::cerr << "[WARN] exchanger callback exception: " << e.what() << "\n";
    } catch (...) {
      std::cerr << "[WARN] exchanger callback exception: unknown\n";
    }
  }
}

static void fanout_push_to_subscribers(
    std::unordered_map<std::string, std::unordered_set<int>>& subscribers,
    std::mutex& subs_mutex, std::string const& ns, std::string const& key,
    uint64_t tag) {
  std::vector<int> targets;
  {
    std::lock_guard<std::mutex> lk(subs_mutex);
    auto it = subscribers.find(ns);
    if (it != subscribers.end()) targets.assign(it->second.begin(), it->second.end());
  }
  if (targets.empty()) return;

  std::ostringstream push;
  push << "PUSH " << std::quoted(ns) << " " << std::quoted(key) << " " << tag
       << "\n";
  std::string payload = push.str();
  std::vector<int> dead_fds;
  for (int fd : targets) {
    if (send_all_robust(fd, payload.c_str(), payload.size()) <= 0) {
      dead_fds.push_back(fd);
    }
  }
  if (dead_fds.empty()) return;
  std::lock_guard<std::mutex> lk(subs_mutex);
  auto it = subscribers.find(ns);
  if (it == subscribers.end()) return;
  for (int fd : dead_fds) it->second.erase(fd);
}

/* ----------------- Forward decl for friend ----------------- */
static void handle_connection_inner(
    int connfd, std::unordered_map<std::string, SockExchanger::KvEntry>& store,
    std::mutex& store_mutex,
    std::unordered_map<std::string, std::unordered_set<int>>& subscribers,
    std::mutex& subs_mutex, std::atomic<bool>& running, size_t max_line_bytes) {
  std::unordered_set<std::string> local_subscribed_ns;
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

      if (cmd == "SET") {
        // SET ns key n field1 value1 ...
        if (toks.size() < 4) {
          char const* r = "ERR\n";
          send_all_robust(connfd, r, ::strlen(r));
          continue;
        }
        std::string const& ns = toks[1];
        std::string const& key = toks[2];
        size_t n = 0;
        try {
          n = static_cast<size_t>(std::stoull(toks[3]));
        } catch (...) {
          char const* r = "ERR\n";
          send_all_robust(connfd, r, ::strlen(r));
          continue;
        }
        if (toks.size() != (4 + 2 * n)) {
          char const* r = "ERR\n";
          send_all_robust(connfd, r, ::strlen(r));
          continue;
        }
        std::string const full_key = compose_store_key(ns, key);
        uint64_t final_tag = 0;
        {
          std::lock_guard<std::mutex> lk(store_mutex);
          auto& ent = store[full_key];
          final_tag = ent.tag + 1;
          ent.tag = final_tag;
          ent.updated_ns = now_ns_monotonic();
          ent.fields.clear();
          for (size_t i = 4; i + 1 < toks.size(); i += 2) {
            ent.fields[toks[i]] = toks[i + 1];
          }
        }
        fanout_push_to_subscribers(subscribers, subs_mutex, ns, key, final_tag);
        std::ostringstream oss;
        oss << "OK " << final_tag << "\n";
        auto const resp = oss.str();
        send_all_robust(connfd, resp.c_str(), resp.size());
      } else if (cmd == "GET") {
        // GET ns key
        if (toks.size() != 3) {
          char const* r = "ERR\n";
          send_all_robust(connfd, r, ::strlen(r));
          continue;
        }
        std::string const full_key = compose_store_key(toks[1], toks[2]);
        bool found = false;
        SockExchanger::KvEntry ent{};
        {
          std::lock_guard<std::mutex> lk(store_mutex);
          auto it = store.find(full_key);
          if (it != store.end()) {
            found = true;
            ent = it->second;
          }
        }
        if (!found) {
          char const* r = "EMPTY\n";
          send_all_robust(connfd, r, ::strlen(r));
          continue;
        }
        std::ostringstream oss;
        oss << "VALUE " << ent.tag << " " << ent.updated_ns << " "
            << ent.fields.size();
        for (auto const& p : ent.fields) {
          oss << " " << std::quoted(p.first) << " " << std::quoted(p.second);
        }
        oss << "\n";
        const std::string resp = oss.str();
        send_all_robust(connfd, resp.c_str(), resp.size());
      } else if (cmd == "SUB") {
        if (toks.size() != 2) {
          char const* r = "ERR\n";
          send_all_robust(connfd, r, ::strlen(r));
          continue;
        }
        std::string const ns = toks[1];
        {
          std::lock_guard<std::mutex> lk(subs_mutex);
          subscribers[ns].insert(connfd);
        }
        local_subscribed_ns.insert(ns);
        char const* r = "OK\n";
        send_all_robust(connfd, r, ::strlen(r));
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

  if (!local_subscribed_ns.empty()) {
    std::lock_guard<std::mutex> lk(subs_mutex);
    for (auto const& ns : local_subscribed_ns) {
      auto it = subscribers.find(ns);
      if (it == subscribers.end()) continue;
      it->second.erase(connfd);
    }
  }
  ::shutdown(connfd, SHUT_RDWR);
  ::close(connfd);
}

void handle_connection(
    int connfd, std::unordered_map<std::string, SockExchanger::KvEntry>& store,
    std::mutex& store_mutex,
    std::unordered_map<std::string, std::unordered_set<int>>& subscribers,
    std::mutex& subs_mutex, std::atomic<bool>& running, size_t max_line_bytes,
    std::function<void(std::thread&&)> register_thread) {
  std::thread t(handle_connection_inner, connfd, std::ref(store),
                std::ref(store_mutex), std::ref(subscribers),
                std::ref(subs_mutex), std::ref(running), max_line_bytes);
  register_thread(std::move(t));
}

/* ----------------- SockExchanger implementation ----------------- */

SockExchanger::SockExchanger(bool is_server, std::string const& host, int port,
                             int timeout_ms, size_t max_line_bytes)
    : sock_fd_(-1),
      sub_fd_(-1),
      host_(host),
      port_(port),
      timeout_ms_(timeout_ms),
      is_server_(is_server),
      listen_fd_(-1),
      running_(false),
      sub_running_(false),
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
    sub_running_.store(false, std::memory_order_relaxed);
    sub_ack_cv_.notify_all();

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
    if (sub_fd_ != -1) {
      ::shutdown(sub_fd_, SHUT_RDWR);
      ::close(sub_fd_);
      sub_fd_ = -1;
    }

    // Join accept thread
    if (server_thread_.joinable()) server_thread_.join();
    if (subscriber_thread_.joinable()) subscriber_thread_.join();

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
  std::lock_guard<std::mutex> lk(cmd_mu_);
  if (sock_fd_ == -1) return false;
  std::string out = cmd;
  if (out.empty() || out.back() != '\n') out.push_back('\n');
  if (send_all_robust(sock_fd_, out.c_str(), out.size()) <= 0) return false;
  return recv_line_capped(sock_fd_, resp, max_line_bytes_, running_);
}

bool SockExchanger::put_with_tag(std::string const& ns, std::string const& key,
                                 Exchangeable const& obj, uint64_t* out_tag) {
  std::string const full_key = compose_store_key(ns, key);
  auto kv = obj.to_map();

  // Build command: SET ns key n f1 v1 f2 v2 ...
  std::ostringstream oss;
  oss << "SET " << std::quoted(ns) << " " << std::quoted(key) << " "
      << kv.size();
  for (auto const& p : kv) {
    oss << " " << std::quoted(p.first) << " " << std::quoted(p.second);
  }
  std::string cmd = oss.str();

  if (is_server_) {
    uint64_t final_tag = 0;
    {
      std::lock_guard<std::mutex> lk(store_mutex_);
      auto& ent = store_[full_key];
      final_tag = ent.tag + 1;
      ent.tag = final_tag;
      ent.updated_ns = now_ns_monotonic();
      ent.fields = std::move(kv);
    }
    if (out_tag) *out_tag = final_tag;
    fanout_push_to_subscribers(subscribers_, subs_mutex_, ns, key, final_tag);
    std::vector<std::function<void(std::string const&, std::string const&,
                                   uint64_t)>>
        local_callbacks;
    {
      std::lock_guard<std::mutex> lk(callback_mu_);
      auto it = callbacks_.find(ns);
      if (it != callbacks_.end()) local_callbacks = it->second;
    }
    run_callbacks_noexcept(local_callbacks, ns, key, final_tag);
    return true;
  } else {
    std::string resp;
    if (!send_cmd_and_recv(cmd, resp)) return false;
    if (resp.rfind("OK ", 0) != 0) return false;
    std::istringstream iss(resp);
    std::string ok;
    uint64_t tag = 0;
    iss >> ok >> tag;
    if (out_tag) *out_tag = tag;
    // Client callbacks are delivered by subscriber PUSH events.
    // Do not invoke callbacks locally here, otherwise a local put()
    // would trigger duplicate callbacks (local + PUSH).
    return true;
  }
}

bool SockExchanger::get_once(std::string const& ns, std::string const& key,
                             Exchangeable& obj, uint64_t* out_tag) {
  std::string const full_key = compose_store_key(ns, key);
  if (is_server_) {
    std::lock_guard<std::mutex> lk(store_mutex_);
    auto it = store_.find(full_key);
    if (it == store_.end()) return false;
    if (out_tag) *out_tag = it->second.tag;
    obj.from_map(it->second.fields);
    return true;
  } else {
    std::ostringstream cmd;
    cmd << "GET " << std::quoted(ns) << " " << std::quoted(key);
    std::string resp;
    if (!send_cmd_and_recv(cmd.str(), resp)) return false;
    if (resp.rfind("EMPTY", 0) == 0) return false;
    if (resp.rfind("VALUE ", 0) == 0) {
      // format: VALUE <tag> <updated_ns> <n> f1 v1 f2 v2 ...
      std::istringstream iss(resp);
      std::string tag;
      uint64_t value_tag = 0;
      uint64_t updated_ns = 0;
      size_t n = 0;
      iss >> tag >> value_tag >> updated_ns >> n;
      (void)updated_ns;
      std::map<std::string, std::string> kv;
      for (size_t i = 0; i < n; ++i) {
        std::string f, v;
        if (!(iss >> std::quoted(f) >> std::quoted(v))) return false;
        kv[f] = v;
      }
      if (out_tag) *out_tag = value_tag;
      obj.from_map(kv);
      return true;
    }
    return false;
  }
}

uint64_t SockExchanger::put(std::string const& ns, std::string const& key,
                            Exchangeable const& obj) {
  uint64_t tag = 0;
  if (!put_with_tag(ns, key, obj, &tag)) return 0;
  return tag;
}

bool SockExchanger::get(std::string const& ns, std::string const& key,
                        Exchangeable& obj, uint64_t* out_tag,
                        int timeout_ms) {
  if (timeout_ms == 0) {
    return get_once(ns, key, obj, out_tag);
  }
  auto const deadline =
      (timeout_ms > 0)
          ? (std::chrono::steady_clock::now() +
             std::chrono::milliseconds(timeout_ms))
          : std::chrono::steady_clock::time_point::max();
  int const poll_delay_ms = 10;
  while (running_.load(std::memory_order_relaxed)) {
    if (get_once(ns, key, obj, out_tag)) return true;
    if (timeout_ms == 0) return false;
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(poll_delay_ms));
  }
  return false;
}

bool SockExchanger::subscribe(
    std::string const& ns,
    std::function<void(std::string const& ns, std::string const& key,
                       uint64_t tag)> cb) {
  {
    std::lock_guard<std::mutex> lk(callback_mu_);
    callbacks_[ns].push_back(std::move(cb));
  }
  if (is_server_) {
    // Server-local callbacks are supported without network roundtrip.
    return true;
  }
  {
    std::lock_guard<std::mutex> lk(callback_mu_);
    if (subscribed_ns_.find(ns) != subscribed_ns_.end()) return true;
  }

  // Serialize SUB command send path. Only subscriber thread reads sub_fd_;
  // subscribe() waits for an ACK signal from that thread.
  uint64_t ack_target = 0;
  uint64_t err_target = 0;
  std::ostringstream sub_cmd;
  sub_cmd << "SUB " << std::quoted(ns) << "\n";
  {
    std::lock_guard<std::mutex> lk(sub_mu_);
    if (sub_fd_ == -1 && !connect_subscriber_client()) return false;
    ensure_subscriber_thread();
    {
      std::lock_guard<std::mutex> ack_lk(sub_ack_mu_);
      ack_target = sub_ack_count_ + 1;
      err_target = sub_err_count_ + 1;
    }
    auto const cmd = sub_cmd.str();
    if (send_all_robust(sub_fd_, cmd.c_str(), cmd.size()) <= 0) {
      return false;
    }
  }

  auto const deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms_);
  {
    std::unique_lock<std::mutex> lk(sub_ack_mu_);
    bool const ok = sub_ack_cv_.wait_until(lk, deadline, [&] {
      return !sub_running_.load(std::memory_order_relaxed) ||
             sub_ack_count_ >= ack_target || sub_err_count_ >= err_target;
    });
    if (!ok || !sub_running_.load(std::memory_order_relaxed) ||
        sub_err_count_ >= err_target) {
      return false;
    }
  }
  {
    std::lock_guard<std::mutex> lk(callback_mu_);
    subscribed_ns_.insert(ns);
  }
  return true;
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

bool SockExchanger::connect_subscriber_client() {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return false;

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
  sub_fd_ = fd;
  sub_running_.store(true, std::memory_order_relaxed);
  return true;
}

void SockExchanger::ensure_subscriber_thread() {
  if (subscriber_thread_.joinable()) return;
  subscriber_thread_ = std::thread([this]() {
    std::string line;
    while (sub_running_.load(std::memory_order_relaxed)) {
      if (!recv_line_capped(sub_fd_, line, max_line_bytes_, sub_running_)) {
        break;
      }
      // SUB command response path (serialized by subscribe()).
      if (line == "OK") {
        {
          std::lock_guard<std::mutex> lk(sub_ack_mu_);
          ++sub_ack_count_;
        }
        sub_ack_cv_.notify_all();
        continue;
      }
      if (line == "ERR") {
        {
          std::lock_guard<std::mutex> lk(sub_ack_mu_);
          ++sub_err_count_;
        }
        sub_ack_cv_.notify_all();
        continue;
      }

      auto toks = split_tokens(line);
      if (toks.size() != 4 || toks[0] != "PUSH") continue;
      std::string const ns = toks[1];
      std::string const key = toks[2];
      uint64_t tag = 0;
      try {
        tag = std::stoull(toks[3]);
      } catch (...) {
        continue;
      }
      std::vector<std::function<void(std::string const&, std::string const&,
                                     uint64_t)>>
          callbacks;
      {
        std::lock_guard<std::mutex> lk(callback_mu_);
        auto it = callbacks_.find(ns);
        if (it != callbacks_.end()) callbacks = it->second;
      }
      run_callbacks_noexcept(callbacks, ns, key, tag);
    }
    sub_running_.store(false, std::memory_order_relaxed);
    sub_ack_cv_.notify_all();
  });
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
                          this->subscribers_, this->subs_mutex_,
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
