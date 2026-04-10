#pragma once

#include "../util/jring.h"
#include "transport_adapter.h"
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>

namespace UKernel {
namespace Transport {

class TcpTransportAdapter final : public TransportAdapter {
 public:
  // Lifecycle.
  TcpTransportAdapter(std::string local_ip, int local_rank);
  ~TcpTransportAdapter() override;

  // Endpoint discovery metadata.
  uint16_t get_listen_port() const;
  std::string const& get_listen_ip() const { return local_ip_; }

  // TransportAdapter common capabilities.
  bool ensure_peer(PeerConnectSpec const& spec) override;
  bool has_peer(int peer_rank) const override;

  unsigned send_async(int peer_rank, void* local_ptr, size_t len,
                      uint64_t local_mr_id,
                      std::optional<RemoteSlice> remote_hint,
                      BounceBufferProvider bounce_provider = nullptr) override;
  unsigned recv_async(int peer_rank, void* local_ptr, size_t len,
                      uint64_t local_mr_id,
                      BounceBufferProvider bounce_provider = nullptr) override;

  bool poll_completion(unsigned id) override;
  bool wait_completion(unsigned id) override;
  bool request_failed(unsigned id) override;
  void release_request(unsigned id) override;

 private:
  // Handshake wire structs.
  struct Handshake {
    uint32_t src_rank = 0;
  };

  struct HandshakeAck {
    uint32_t accepted = 1;
  };

  struct PeerContext {
    int send_fd = -1;
    int recv_fd = -1;
    std::mutex send_mu;
    std::mutex recv_mu;
  };

  enum class RequestState : uint8_t {
    Free = 0,
    Queued = 1,
    Running = 2,
    Completed = 3,
    Failed = 4,
  };

  struct RequestSlot {
    std::atomic<RequestState> state{RequestState::Free};
    std::atomic<uint32_t> generation{1};
    int peer_rank = -1;
    void* host_ptr = nullptr;
    size_t len = 0;
    std::atomic<bool> completed{false};
    std::atomic<bool> failed{false};

    void mark_queued() {
      state.store(RequestState::Queued, std::memory_order_release);
      completed.store(false, std::memory_order_release);
      failed.store(false, std::memory_order_release);
    }
    void mark_running() {
      state.store(RequestState::Running, std::memory_order_release);
    }
    void mark_completed(bool ok) {
      state.store(ok ? RequestState::Completed : RequestState::Failed,
                  std::memory_order_release);
      completed.store(true, std::memory_order_release);
      failed.store(!ok, std::memory_order_release);
    }
    bool is_completed() const {
      return completed.load(std::memory_order_acquire);
    }
    bool is_failed() const { return failed.load(std::memory_order_acquire); }
  };

  // Peer establishment helpers.
  bool connect_to_peer(int peer_rank, std::string remote_ip,
                       uint16_t remote_port);
  bool accept_from_peer(int peer_rank, std::string const& expected_remote_ip);

  // Request execution workers.
  void send_worker_loop();
  void recv_worker_loop();

  // Request slot lifecycle.
  static constexpr uint32_t kRequestSlotBits = 13;
  static constexpr uint32_t kRequestSlotCount = (1u << kRequestSlotBits);
  static constexpr uint32_t kRequestSlotMask = kRequestSlotCount - 1u;
  static unsigned make_request_id(uint32_t slot_idx, uint32_t generation) {
    return static_cast<unsigned>((generation << kRequestSlotBits) | slot_idx);
  }
  static uint32_t request_slot_index(unsigned request_id) {
    return static_cast<uint32_t>(request_id) & kRequestSlotMask;
  }
  static uint32_t request_generation(unsigned request_id) {
    return static_cast<uint32_t>(request_id) >> kRequestSlotBits;
  }
  RequestSlot* try_acquire_request_slot(unsigned* out_request_id);
  RequestSlot* resolve_request_slot(unsigned request_id);
  RequestSlot* resolve_request_slot_const(unsigned request_id) const;
  void release_request_slot(unsigned request_id);
  bool enqueue_request(unsigned request_id, bool is_send);

  // Socket helpers.
  static int create_listen_socket(uint16_t& out_port);
  static bool connect_socket(int& out_fd, std::string const& remote_ip,
                             uint16_t remote_port,
                             std::chrono::milliseconds timeout);
  static bool send_handshake(int fd, Handshake const& hs);
  static bool recv_handshake(int fd, Handshake& hs);
  static bool send_handshake_ack(int fd, HandshakeAck const& ack);
  static bool recv_handshake_ack(int fd, HandshakeAck& ack);
  static bool send_all(int fd, void const* buf, size_t len);
  static bool recv_all(int fd, void* buf, size_t len);

  std::string local_ip_;
  int local_rank_ = -1;
  int listen_fd_ = -1;
  uint16_t listen_port_ = 0;

  mutable std::mutex mu_;
  std::unordered_map<int, std::shared_ptr<PeerContext>> peer_contexts_;
  jring_t* send_task_ring_ = nullptr;
  jring_t* recv_task_ring_ = nullptr;
  std::atomic<bool> stop_{false};
  std::thread send_worker_;
  std::thread recv_worker_;
  std::mutex cv_mu_;
  std::condition_variable cv_;
  std::atomic<int> pending_send_{0};
  std::atomic<int> pending_recv_{0};
  std::unique_ptr<RequestSlot[]> request_slots_;
  std::atomic<uint32_t> request_alloc_cursor_{0};
};

}  // namespace Transport
}  // namespace UKernel
