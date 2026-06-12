#pragma once

#include "../util/jring.h"
#include "transport_adapter.h"
#include "../../../include/gpu_rt.h"
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
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
  TcpTransportAdapter(std::string local_ip, int local_rank, int gpu_id);
  ~TcpTransportAdapter() override;

  uint16_t get_listen_port() const;
  std::string const& get_listen_ip() const { return local_ip_; }

  bool ensure_put_path(PeerConnectSpec const& spec) override;
  bool ensure_wait_path(PeerConnectSpec const& spec) override;
  bool has_put_path(int peer_rank) const override;
  bool has_wait_path(int peer_rank) const override;

  unsigned put_async(int peer_rank, void* local_ptr, uint32_t local_buffer_id,
                     void* remote_ptr, uint32_t remote_buffer_id,
                     size_t len, unsigned comm_rid) override;
  unsigned signal_async(int peer_rank, uint64_t tag,
                        unsigned comm_rid) override;
  unsigned wait_async(int peer_rank, uint64_t expected_tag,
                      std::optional<WaitTarget> target,
                      unsigned comm_rid) override;

  void release(unsigned id) override;

 private:
  struct PeerContext {
    int send_fd = -1;
    int recv_fd = -1;
    std::mutex send_mu;
    std::mutex recv_mu;
  };

  enum class RequestState : uint8_t {
    Free = 0, Queued = 1, Completed = 2, Failed = 3
  };

  struct RequestSlot {
    enum class Kind : uint8_t {
      DataPut = 0, DataWait = 1, Signal = 2, SignalWait = 3
    };
    std::atomic<RequestState> state{RequestState::Free};
    std::atomic<uint32_t> generation{1};
    Kind kind = Kind::DataPut;
    unsigned comm_rid = 0;
    int peer_rank = -1;
    void* host_ptr = nullptr;
    size_t len = 0;
    uint64_t expected_tag = 0;
    uint64_t signal_payload = 0;

    void mark_queued() {
      state.store(RequestState::Queued, std::memory_order_release);
    }
    void mark_completed(bool ok) {
      state.store(ok ? RequestState::Completed : RequestState::Failed,
                  std::memory_order_release);
    }
    bool is_done() const {
      auto s = state.load(std::memory_order_acquire);
      return s == RequestState::Completed || s == RequestState::Failed;
    }
  };

  bool connect_to_peer(int peer_rank, std::string remote_ip, uint16_t remote_port);
  bool accept_from_peer(int peer_rank, std::string const& expected_remote_ip);

  void send_worker_loop();
  void recv_worker_loop();

  static constexpr uint32_t kSlotBits = 13;
  static constexpr uint32_t kSlotCount = (1u << kSlotBits);
  static constexpr uint32_t kSlotMask = kSlotCount - 1u;
  static unsigned make_rid(uint32_t idx, uint32_t gen) {
    return static_cast<unsigned>((gen << kSlotBits) | idx);
  }
  static uint32_t slot_index(unsigned rid) { return rid & kSlotMask; }
  static uint32_t slot_gen(unsigned rid) { return rid >> kSlotBits; }

  RequestSlot* acquire_slot(unsigned* out_rid);
  RequestSlot* resolve_slot(unsigned rid);
  void release_slot(unsigned rid);
  bool enqueue_send(unsigned rid);
  bool enqueue_recv(unsigned rid);

  static int create_listen_socket(uint16_t& out_port);
  static bool connect_socket(int& out_fd, std::string const& remote_ip,
                             uint16_t remote_port, std::chrono::milliseconds timeout);
  static bool handshake(int fd, uint32_t rank, bool is_send);
  static bool recv_handshake(int fd, uint32_t& rank);
  static bool send_all(int fd, void const* buf, size_t len);
  static bool recv_all(int fd, void* buf, size_t len);

  std::string local_ip_;
  int local_rank_ = -1;
  int gpu_id_ = -1;
  gpuStream_t gpu_stream_ = nullptr;
  int listen_fd_ = -1;
  uint16_t listen_port_ = 0;

  mutable std::mutex mu_;
  std::unordered_map<int, std::shared_ptr<PeerContext>> peer_contexts_;
  jring_t* send_task_ring_ = nullptr;
  jring_t* recv_task_ring_ = nullptr;
  std::atomic<bool> stop_{false};
  std::thread send_worker_;
  std::thread recv_worker_;
  std::unique_ptr<RequestSlot[]> slots_;
  std::atomic<uint32_t> alloc_cursor_{0};
};
}  // namespace Transport
}  // namespace UKernel
