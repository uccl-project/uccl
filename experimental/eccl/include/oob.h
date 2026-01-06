#pragma once

#include "util/gpu_rt.h"
#include <map>
#include <string>

// for redis
#ifdef USE_REDIS_OOB
#include <hiredis/hiredis.h>
#endif
#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <thread>

// for RDMAConnectionInfo
#include <iomanip>
#include <sstream>
#include <vector>

// for socket
#include <atomic>
#include <functional>
#include <mutex>
#include <unordered_map>

// --- Group Exchanger ---
struct Exchangeable {
  virtual std::map<std::string, std::string> to_map() const = 0;
  virtual void from_map(std::map<std::string, std::string> const& kv) = 0;
  virtual ~Exchangeable() {}
};

struct CommunicatorMeta : public Exchangeable {
  std::string host_id;
  std::string ip;
  bool is_ready;
  // TODO: connection abality // Support RDMA, PCIe, NVlink, UAlink, etc.

  CommunicatorMeta() = default;

  std::map<std::string, std::string> to_map() const override {
    std::map<std::string, std::string> kv;
    kv["host_id"] = host_id;
    kv["ip"] = ip;
    kv["is_ready"] = is_ready ? "1" : "0";
    return kv;
  }

  void from_map(std::map<std::string, std::string> const& kv) override {
    host_id = kv.at("host_id");
    ip = kv.at("ip");
    std::string const& ready_str = kv.at("is_ready");
    is_ready = (ready_str == "1");
  }
};

struct QpInfo {
  uint32_t qp_num;
  uint32_t psn;
  uint16_t lid;     // Local ID
  uint8_t gid[16];  // Global ID for RoCE (optional)
};

struct RDMAInfo : public Exchangeable {
  std::vector<QpInfo> qps;

  RDMAInfo() = default;

  std::map<std::string, std::string> to_map() const override {
    std::map<std::string, std::string> kv;
    kv["count"] = std::to_string(qps.size());

    for (size_t i = 0; i < qps.size(); ++i) {
      auto const& qp = qps[i];
      kv["qp_" + std::to_string(i) + "_num"] = std::to_string(qp.qp_num);
      kv["qp_" + std::to_string(i) + "_psn"] = std::to_string(qp.psn);
      kv["qp_" + std::to_string(i) + "_lid"] = std::to_string(qp.lid);

      std::ostringstream oss;
      oss << std::hex << std::setfill('0');
      for (int j = 0; j < 16; ++j) {
        oss << std::setw(2) << static_cast<int>(qp.gid[j]);
      }
      kv["qp_" + std::to_string(i) + "_gid"] = oss.str();
    }

    return kv;
  }

  void from_map(std::map<std::string, std::string> const& kv) override {
    size_t count = std::stoul(kv.at("count"));
    qps.resize(count);

    for (size_t i = 0; i < count; ++i) {
      auto& qp = qps[i];
      qp.qp_num = std::stoul(kv.at("qp_" + std::to_string(i) + "_num"));
      qp.psn = std::stoul(kv.at("qp_" + std::to_string(i) + "_psn"));
      qp.lid = std::stoul(kv.at("qp_" + std::to_string(i) + "_lid"));

      std::string const& hex = kv.at("qp_" + std::to_string(i) + "_gid");
      for (int j = 0; j < 16; ++j) {
        unsigned int byte;
        std::stringstream ss;
        ss << std::hex << hex.substr(j * 2, 2);
        ss >> byte;
        qp.gid[j] = static_cast<unsigned char>(byte);
      }
    }
  }
};

struct MR {
  uint32_t id;
  uint64_t address;
  uint32_t length;
  uint32_t lkey;
  uint32_t key;
};

struct MRInfos : public Exchangeable {
  std::vector<MR> mrs;

  MRInfos() = default;

  std::map<std::string, std::string> to_map() const override {
    std::map<std::string, std::string> kv;
    kv["count"] = std::to_string(mrs.size());

    for (size_t i = 0; i < mrs.size(); ++i) {
      auto const& mr = mrs[i];
      kv["mr_" + std::to_string(i) + "_id"] = std::to_string(mr.id);

      {
        std::ostringstream oss;
        oss << std::hex << std::setw(16) << std::setfill('0') << mr.address;
        kv["mr_" + std::to_string(i) + "_addr"] = oss.str();
      }

      kv["mr_" + std::to_string(i) + "_len"] = std::to_string(mr.length);
      kv["mr_" + std::to_string(i) + "_key"] = std::to_string(mr.key);
    }

    return kv;
  }

  void from_map(std::map<std::string, std::string> const& kv) override {
    size_t count = std::stoul(kv.at("count"));
    mrs.resize(count);

    for (size_t i = 0; i < count; ++i) {
      auto& mr = mrs[i];
      mr.id = std::stoul(kv.at("mr_" + std::to_string(i) + "_id"));
      // address (64-bit)
      {
        std::string hexaddr = kv.at("mr_" + std::to_string(i) + "_addr");
        mr.address = std::stoull(hexaddr, nullptr, 16);
      }
      // length and key
      mr.length = std::stoul(kv.at("mr_" + std::to_string(i) + "_len"));
      mr.key = std::stoul(kv.at("mr_" + std::to_string(i) + "_key"));
    }
  }
};

class Exchanger {
 public:
  virtual ~Exchanger() = default;

  virtual bool valid() const = 0;
  virtual bool publish(std::string const& key, Exchangeable const& obj) = 0;
  virtual bool fetch(std::string const& key, Exchangeable& obj) = 0;
  virtual bool wait_and_fetch(std::string const& key, Exchangeable& obj,
                              int max_retries = 50, int delay_ms = 100) = 0;
};

class RedisExchanger : public Exchanger {
 public:
  RedisExchanger(std::string const& host = "127.0.0.1", int port = 6379,
                 int timeout_ms = 2000);
  ~RedisExchanger();

  bool valid() const override;
  bool publish(std::string const& key, Exchangeable const& obj) override;
  bool fetch(std::string const& key, Exchangeable& obj) override;
  bool wait_and_fetch(std::string const& key, Exchangeable& obj,
                      int max_retries = 50, int delay_ms = 100) override;

 private:
#ifdef USE_REDIS_OOB
  redisContext* ctx_;
#endif
};

class SockExchanger : public Exchanger {
 public:
  SockExchanger(bool is_server, std::string const& host, int port,
                int timeout_ms = 3000,
                size_t max_line_bytes = 1 * 1024 * 1024 /* 1MB */);
  ~SockExchanger();

  bool valid() const;

  // client-only
  bool publish(std::string const& key, Exchangeable const& obj);
  bool fetch(std::string const& key, Exchangeable& obj);

  bool wait_and_fetch(std::string const& key, Exchangeable& obj,
                      int max_retries = -1, int delay_ms = 100);

 private:
  bool start_server();
  bool connect_client();
  bool send_cmd_and_recv(std::string const& cmd, std::string& resp);

 private:
  int sock_fd_;
  std::string host_;
  int port_;
  int timeout_ms_;
  bool is_server_;
  int listen_fd_;
  std::atomic<bool> running_;
  size_t max_line_bytes_;

  // server side state
  std::unordered_map<std::string, std::map<std::string, std::string>> store_;
  std::mutex store_mutex_;
  std::thread server_thread_;
  std::mutex conn_threads_mutex_;
  std::vector<std::thread> conn_threads_;

  friend void handle_connection(
      int, std::unordered_map<std::string, std::map<std::string, std::string>>&,
      std::mutex&, std::atomic<bool>&, size_t,
      std::function<void(std::thread&&)>);
};

// --- P2P Exchanger ---

struct IpcCache {
  gpuIpcMemHandle_t handle;
  bool is_send;
  void* direct_ptr;  // ptr of remote, local get it by mapping from handle
  uintptr_t offset;
  size_t size;
};

// Only message type needed for now
static constexpr uint16_t kTypeIpcCache = 1;
static constexpr uint16_t kTypeAck = 2;

// uds payload
#pragma pack(push, 1)
struct IpcCacheWire {
  gpuIpcMemHandle_t handle;
  uint8_t is_send;
  uint64_t offset;
  uint64_t size;
  uint32_t remote_gpu_idx_;
};
#pragma pack(pop)

struct AckWire {
  uint32_t status;    // 0=fail, 1=ok, or extend
  uint32_t reserved;  // keep 8B aligned
};

class UdsExchanger {
 public:
  UdsExchanger(int self_rank);
  ~UdsExchanger();

  // Lazy start local server (bind/listen) once.
  bool ensure_server_started();

  // Client: connect to peer's UDS with retry until timeout. Idempotent.
  bool connect_to(int peer_rank, int timeout_ms = 30000);

  // Server: accept connections until we receive one from peer_rank (or
  // timeout). Idempotent: returns true immediately if already connected.
  bool accept_from(int peer_rank, int timeout_ms = 30000);

  // Generic framed send
  bool send(int peer_rank, uint16_t type, uint64_t seq, void const* payload,
            uint32_t bytes);

  // Convenience: send IPC cache
  bool send_ipc_cache(int peer_rank, uint64_t seq, IpcCacheWire const& cache);
  bool recv_ipc_cache(int peer_rank, IpcCacheWire& out_cache,
                      uint64_t* out_seq = nullptr, int timeout_ms = 30000);
  bool send_ack(int peer_rank, uint64_t seq, uint32_t status = 1);
  bool recv_ack(int peer_rank, uint32_t* out_status = nullptr,
                uint64_t* out_seq = nullptr, int timeout_ms = 30000,
                uint64_t expected_seq = UINT64_MAX);

  int get_fd(int peer_rank) const;
  void close_peer(int peer_rank);

 private:
  struct Hello {
    uint32_t magic;
    int32_t from_rank;
    int32_t to_rank;
    uint32_t version;
  };

  struct MsgHdr {
    uint32_t magic;
    uint16_t version;
    uint16_t type;
    uint32_t bytes;
    int32_t from_rank;
    int32_t to_rank;
    uint64_t seq;
  };

 private:
  std::string path_for_rank(int rank);
  bool connect_once(std::string const& peer_path, int& out_fd);

  bool send_all(int fd, char const* buf, size_t len);
  bool recv_all(int fd, char* buf, size_t len);

  // Accept one connection with a poll-like timeout.
  // Returns: fd>=0 on success, -1 on timeout, -2 on fatal error.
  int accept_with_timeout(int timeout_ms);

 private:
  int self_rank_;

  // server state
  std::atomic<bool> running_{false};
  int listen_fd_{-1};
  std::string local_path_;

  // connections
  mutable std::mutex mu_;
  std::unordered_map<int, int> rank_to_fd_;
  std::unordered_map<int, std::unique_ptr<std::mutex>> rank_send_mu_;
  std::unordered_map<int, std::unique_ptr<std::mutex>> rank_recv_mu_;

  // Ensure only one thread is doing accept() at a time.
  mutable std::mutex accept_mu_;
};