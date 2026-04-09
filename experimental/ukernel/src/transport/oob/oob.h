#pragma once

#include "../../include/config.h"
#include "../../include/gpu_rt.h"
#include "../util/jring.h"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
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

namespace UKernel {
namespace Transport {

enum class PeerTransportKind { Unknown, Uccl, Ipc, Tcp };

struct CommunicatorMeta;

char const* peer_transport_kind_name(PeerTransportKind kind);

PeerTransportKind resolve_peer_transport_kind(
    CommunicatorConfig const& config, CommunicatorMeta const& local_meta,
    CommunicatorMeta const& peer_meta);

struct Exchangeable {
  virtual std::map<std::string, std::string> to_map() const = 0;
  virtual void from_map(std::map<std::string, std::string> const& kv) = 0;
  virtual ~Exchangeable() {}
};

struct CommunicatorMeta : public Exchangeable {
  std::string host_id;
  std::string ip;
  int local_id = -1;
  bool rdma_capable = false;

  CommunicatorMeta() = default;

  std::map<std::string, std::string> to_map() const override {
    std::map<std::string, std::string> kv;
    kv["host_id"] = host_id;
    kv["ip"] = ip;
    kv["local_id"] = std::to_string(local_id);
    kv["rdma_capable"] = rdma_capable ? "1" : "0";
    return kv;
  }

  void from_map(std::map<std::string, std::string> const& kv) override {
    host_id = kv.at("host_id");
    ip = kv.at("ip");
    local_id = std::stoi(kv.at("local_id"));
    auto it = kv.find("rdma_capable");
    rdma_capable = (it != kv.end() && it->second == "1");
  }
};

struct MR {
  uint32_t id;
  uint64_t address;
  uint64_t length;
  uint32_t lkey;
  uint32_t key;
};

struct NamedMR {
  uint32_t buffer_id = 0;
  MR mr{};
};

struct NamedMRInfos : public Exchangeable {
  uint64_t generation = 0;
  std::vector<NamedMR> entries;

  std::map<std::string, std::string> to_map() const override {
    std::map<std::string, std::string> kv;
    kv["generation"] = std::to_string(generation);
    kv["count"] = std::to_string(entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
      auto const& entry = entries[i];
      kv["entry_" + std::to_string(i) + "_buffer_id"] =
          std::to_string(entry.buffer_id);
      kv["entry_" + std::to_string(i) + "_mr_id"] = std::to_string(entry.mr.id);

      std::ostringstream oss;
      oss << std::hex << std::setw(16) << std::setfill('0') << entry.mr.address;
      kv["entry_" + std::to_string(i) + "_mr_addr"] = oss.str();

      kv["entry_" + std::to_string(i) + "_mr_len"] =
          std::to_string(entry.mr.length);
      kv["entry_" + std::to_string(i) + "_mr_key"] =
          std::to_string(entry.mr.key);
    }
    return kv;
  }

  void from_map(std::map<std::string, std::string> const& kv) override {
    auto generation_it = kv.find("generation");
    generation =
        generation_it == kv.end() ? 0 : std::stoull(generation_it->second);
    size_t count = std::stoul(kv.at("count"));
    entries.resize(count);
    for (size_t i = 0; i < count; ++i) {
      auto& entry = entries[i];
      entry.buffer_id = static_cast<uint32_t>(
          std::stoul(kv.at("entry_" + std::to_string(i) + "_buffer_id")));
      entry.mr.id = static_cast<uint32_t>(
          std::stoul(kv.at("entry_" + std::to_string(i) + "_mr_id")));
      entry.mr.address = std::stoull(
          kv.at("entry_" + std::to_string(i) + "_mr_addr"), nullptr, 16);
      entry.mr.length =
          std::stoull(kv.at("entry_" + std::to_string(i) + "_mr_len"));
      entry.mr.key = static_cast<uint32_t>(
          std::stoul(kv.at("entry_" + std::to_string(i) + "_mr_key")));
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

struct TcpP2PInfo : public Exchangeable {
  std::string ip;
  uint16_t port = 0;

  TcpP2PInfo() = default;
  TcpP2PInfo(std::string ip_, uint16_t port_)
      : ip(std::move(ip_)), port(port_) {}

  std::map<std::string, std::string> to_map() const override {
    std::map<std::string, std::string> kv;
    kv["ip"] = ip;
    kv["port"] = std::to_string(port);
    return kv;
  }

  void from_map(std::map<std::string, std::string> const& kv) override {
    ip = kv.at("ip");
    port = static_cast<uint16_t>(std::stoul(kv.at("port")));
  }
};

struct IpcBufferInfo : public Exchangeable {
  uint32_t ipc_id = 0;
  uint64_t binding_version = 0;
  gpuIpcMemHandle_t handle{};
  uint64_t base_offset = 0;
  uint64_t bytes = 0;
  int32_t device_idx = -1;
  bool valid = false;

  std::map<std::string, std::string> to_map() const override {
    std::map<std::string, std::string> kv;
    kv["ipc_id"] = std::to_string(ipc_id);
    kv["binding_version"] = std::to_string(binding_version);
    kv["base_offset"] = std::to_string(base_offset);
    kv["bytes"] = std::to_string(bytes);
    kv["device_idx"] = std::to_string(device_idx);
    kv["valid"] = valid ? "1" : "0";

    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    auto const* data = reinterpret_cast<uint8_t const*>(&handle);
    for (size_t i = 0; i < sizeof(gpuIpcMemHandle_t); ++i) {
      oss << std::setw(2) << static_cast<unsigned>(data[i]);
    }
    kv["handle_hex"] = oss.str();
    return kv;
  }

  void from_map(std::map<std::string, std::string> const& kv) override {
    ipc_id = static_cast<uint32_t>(std::stoul(kv.at("ipc_id")));
    auto it_binding_version = kv.find("binding_version");
    if (it_binding_version == kv.end()) {
      // Backward-compatible fallback for older peers/tests.
      auto it_generation = kv.find("generation");
      binding_version =
          (it_generation == kv.end()) ? 0 : std::stoull(it_generation->second);
    } else {
      binding_version = std::stoull(it_binding_version->second);
    }
    base_offset = std::stoull(kv.at("base_offset"));
    bytes = std::stoull(kv.at("bytes"));
    device_idx = std::stoi(kv.at("device_idx"));
    valid = (kv.at("valid") == "1");
    std::memset(&handle, 0, sizeof(handle));
    auto const& hex = kv.at("handle_hex");
    auto* data = reinterpret_cast<uint8_t*>(&handle);
    size_t nbytes = std::min(sizeof(gpuIpcMemHandle_t), hex.size() / 2);
    for (size_t i = 0; i < nbytes; ++i) {
      data[i] =
          static_cast<uint8_t>(std::stoul(hex.substr(i * 2, 2), nullptr, 16));
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

class SockExchanger : public Exchanger {
 public:
  SockExchanger(bool is_server, std::string const& host, int port,
                int timeout_ms = 3000, size_t max_line_bytes = 1 * 1024 * 1024);
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

static constexpr uint16_t kTypeIpcCache = 1;
static constexpr uint16_t kTypeAck = 2;

#pragma pack(push, 1)
struct IpcCacheWire {
  gpuIpcMemHandle_t handle;
  uint8_t is_send;
  uint64_t offset;
  uint64_t size;
  uint32_t remote_gpu_idx_;
  uint8_t use_bounce_buffer = 0;
  char bounce_shm_name[128] = {};
};
#pragma pack(pop)

struct AckWire {
  uint32_t status;
  uint32_t reserved;
};

enum class ShmCtrlMsgType : uint32_t {
  Connect = 0,
  ConnectAck = 1,
  IpcCache = 2,
  Ack = 3,
  IpcCacheReq = 4,
};

struct ShmCtrlMsg {
  uint32_t from_rank = 0;
  uint32_t to_rank = 0;
  ShmCtrlMsgType type = ShmCtrlMsgType::Connect;
  uint32_t reserved = 0;
  uint64_t seq = 0;
  union {
    IpcCacheWire cache;
    AckWire ack;
  };

  ShmCtrlMsg() : cache{} {}
};

class ShmRingExchanger {
 public:
  ShmRingExchanger(int self_rank, int world_size, std::string ring_namespace,
                   int self_local_id = -1);
  ~ShmRingExchanger();

  void set_peer_local_id(int peer_rank, int local_id);
  bool ensure_server_started();
  bool connect_to(int peer_rank, int timeout_ms = 30000);
  bool accept_from(int peer_rank, int timeout_ms = 30000);

  bool send(int peer_rank, uint16_t type, uint64_t seq, void const* payload,
            uint32_t bytes);
  bool send_ipc_cache(int peer_rank, uint64_t seq, IpcCacheWire const& cache);
  bool send_ipc_cache_req(int peer_rank, uint64_t seq);
  bool recv_ipc_cache(int peer_rank, IpcCacheWire& out_cache,
                      uint64_t* out_seq = nullptr, int timeout_ms = 30000,
                      uint64_t expected_seq = UINT64_MAX);
  bool recv_ipc_cache_req(int peer_rank, uint64_t* out_seq = nullptr,
                          int timeout_ms = 30000,
                          uint64_t expected_seq = UINT64_MAX);
  bool send_ack(int peer_rank, uint64_t seq, uint32_t status = 1);
  bool recv_ack(int peer_rank, uint32_t* out_status = nullptr,
                uint64_t* out_seq = nullptr, int timeout_ms = 30000,
                uint64_t expected_seq = UINT64_MAX);

  void close_peer(int peer_rank);
  bool is_peer_connected(int peer_rank) const;

 private:
  struct ShmRingHandle {
    jring_t* ring = nullptr;
    int shm_fd = -1;
    size_t shm_size = 0;
    std::string shm_name;
    bool attached = false;
    bool creator = false;
  };

  struct PendingIpcCacheMsg {
    uint64_t seq = 0;
    IpcCacheWire cache{};
  };

  struct PendingAckMsg {
    uint64_t seq = 0;
    AckWire ack{};
  };

  struct PeerState {
    std::mutex send_mu;
    std::mutex recv_mu;
    ShmRingHandle local_inbox;
    ShmRingHandle remote_inbox;
    bool connected = false;
  };

  std::string ring_name(int from_rank, int to_rank) const;
  bool ensure_local_ring(int peer_rank);
  bool ensure_remote_ring_attached(int peer_rank, int timeout_ms);
  bool try_recv_one_locked(int peer_rank, ShmCtrlMsg& msg);
  bool try_take_connect_locked(int peer_rank, uint64_t* out_seq = nullptr);
  bool try_take_cached_connect_ack_locked(int peer_rank, uint64_t expected_seq,
                                          uint64_t* out_seq);
  bool try_take_cached_ipc_cache_locked(int peer_rank, uint64_t expected_seq,
                                        IpcCacheWire& out_cache,
                                        uint64_t* out_seq);
  bool try_take_cached_ipc_cache_req_locked(int peer_rank,
                                            uint64_t expected_seq,
                                            uint64_t* out_seq);
  bool try_take_cached_ack_locked(int peer_rank, uint64_t expected_seq,
                                  AckWire& out_ack, uint64_t* out_seq);
  void cache_connect_message(int peer_rank, uint64_t seq);
  void cache_connect_ack_message(int peer_rank, uint64_t seq);
  void cache_ipc_cache_message(int peer_rank, uint64_t seq,
                               IpcCacheWire const& cache);
  void cache_ipc_cache_req_message(int peer_rank, uint64_t seq);
  void cache_ack_message(int peer_rank, uint64_t seq, AckWire const& ack);
  bool send_msg(int peer_rank, ShmCtrlMsg const& msg);

  int self_rank_;
  int world_size_;
  int self_local_id_;
  std::string ring_namespace_;
  std::atomic<bool> running_{false};

  mutable std::mutex mu_;
  std::vector<std::shared_ptr<PeerState>> peers_;
  std::vector<int> peer_local_ids_;
  mutable std::mutex pending_mu_;
  std::unordered_map<int, std::deque<uint64_t>> pending_connect_;
  std::unordered_map<int, std::deque<uint64_t>> pending_connect_acks_;
  std::unordered_map<int, std::deque<PendingIpcCacheMsg>>
      rank_to_pending_ipc_cache_;
  std::unordered_map<int, std::deque<uint64_t>> rank_to_pending_ipc_cache_req_;
  std::unordered_map<int, std::deque<PendingAckMsg>> rank_to_pending_acks_;
  std::vector<uint64_t> next_connect_seq_;
};

}  // namespace Transport
}  // namespace UKernel
