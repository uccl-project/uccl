#pragma once

#include "../device/gpu_rt.h"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef USE_REDIS_OOB
#include <hiredis/hiredis.h>
#endif

namespace UKernel {
namespace Transport {

struct Exchangeable {
  virtual std::map<std::string, std::string> to_map() const = 0;
  virtual void from_map(std::map<std::string, std::string> const& kv) = 0;
  virtual ~Exchangeable() {}
};

struct CommunicatorMeta : public Exchangeable {
  std::string host_id;
  std::string ip;
  bool is_ready;

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
    is_ready = (kv.at("is_ready") == "1");
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

      std::ostringstream oss;
      oss << std::hex << std::setw(16) << std::setfill('0') << mr.address;
      kv["mr_" + std::to_string(i) + "_addr"] = oss.str();

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
      mr.address =
          std::stoull(kv.at("mr_" + std::to_string(i) + "_addr"), nullptr, 16);
      mr.length = std::stoul(kv.at("mr_" + std::to_string(i) + "_len"));
      mr.key = std::stoul(kv.at("mr_" + std::to_string(i) + "_key"));
    }
  }
};

struct UCCLP2PInfo : public Exchangeable {
  std::string ip;
  uint16_t port;
  int dev_idx;
  int gpu_idx;

  UCCLP2PInfo() = default;
  UCCLP2PInfo(std::string ip_, uint16_t port_, int dev_idx_, int gpu_idx_)
      : ip(std::move(ip_)), port(port_), dev_idx(dev_idx_), gpu_idx(gpu_idx_) {}

  std::map<std::string, std::string> to_map() const override {
    std::map<std::string, std::string> kv;
    kv["ip"] = ip;
    kv["port"] = std::to_string(port);
    kv["dev_idx"] = std::to_string(dev_idx);
    kv["gpu_idx"] = std::to_string(gpu_idx);
    return kv;
  }

  void from_map(std::map<std::string, std::string> const& kv) override {
    ip = kv.at("ip");
    port = static_cast<uint16_t>(std::stoul(kv.at("port")));
    dev_idx = std::stoi(kv.at("dev_idx"));
    gpu_idx = std::stoi(kv.at("gpu_idx"));
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
                size_t max_line_bytes = 1 * 1024 * 1024);
  ~SockExchanger();

  bool valid() const override;
  bool publish(std::string const& key, Exchangeable const& obj) override;
  bool fetch(std::string const& key, Exchangeable& obj) override;
  bool wait_and_fetch(std::string const& key, Exchangeable& obj,
                      int max_retries = -1, int delay_ms = 100) override;

 private:
  bool start_server();
  bool connect_client();
  bool send_cmd_and_recv(std::string const& cmd, std::string& resp);

  int sock_fd_;
  std::string host_;
  int port_;
  int timeout_ms_;
  bool is_server_;
  int listen_fd_;
  std::atomic<bool> running_;
  size_t max_line_bytes_;

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

struct IpcCache {
  gpuIpcMemHandle_t handle;
  bool is_send;
  void* direct_ptr;
  uintptr_t offset;
  size_t size;
  int device_idx = -1;
};

static constexpr uint16_t kTypeIpcCache = 1;
static constexpr uint16_t kTypeAck = 2;

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
  uint32_t status;
  uint32_t reserved;
};

class UdsExchanger {
 public:
  UdsExchanger(int self_rank);
  ~UdsExchanger();

  bool ensure_server_started();
  bool connect_to(int peer_rank, int timeout_ms = 30000);
  bool accept_from(int peer_rank, int timeout_ms = 30000);

  bool send(int peer_rank, uint16_t type, uint64_t seq, void const* payload,
            uint32_t bytes);
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

  std::string path_for_rank(int rank);
  bool connect_once(std::string const& peer_path, int& out_fd);
  bool send_all(int fd, char const* buf, size_t len);
  bool recv_all(int fd, char* buf, size_t len);
  int accept_with_timeout(int timeout_ms);

  int self_rank_;
  std::atomic<bool> running_{false};
  int listen_fd_{-1};
  std::string local_path_;

  mutable std::mutex mu_;
  std::unordered_map<int, int> rank_to_fd_;
  std::unordered_map<int, std::unique_ptr<std::mutex>> rank_send_mu_;
  std::unordered_map<int, std::unique_ptr<std::mutex>> rank_recv_mu_;
  mutable std::mutex accept_mu_;
};

}  // namespace Transport
}  // namespace UKernel
