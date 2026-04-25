#pragma once

#include "../../include/gpu_rt.h"
#include "../util/jring.h"
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace UKernel {
namespace Transport {

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
