#pragma once

#include "transport_adapter.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

// Forward declare uccl classes
namespace uccl {
class RDMAEndpoint;
class Mhandle;
class UcclFlow;
struct ucclRequest;
}  // namespace uccl

namespace UKernel {
namespace Transport {

struct UcclTransportConfig {
  int num_engines = 0;
};

class UcclTransportAdapter final : public TransportAdapter {
 public:
  UcclTransportAdapter(int local_gpu_idx, int world_size,
                       UcclTransportConfig config);
  ~UcclTransportAdapter();

  uint16_t get_p2p_listen_port(int dev_idx) const;
  std::string get_p2p_listen_ip(int dev_idx) const;
  int get_best_dev_idx(int gpu_idx) const;

  bool is_memory_registered(uint32_t buffer_id) const;
  bool register_memory(uint32_t buffer_id, void* ptr, size_t len);
  void deregister_memory(uint32_t buffer_id);
  bool is_initialized() const { return endpoint_ != nullptr; }

  bool ensure_put_path(PeerConnectSpec const& spec) override;
  bool ensure_wait_path(PeerConnectSpec const& spec) override;
  bool has_put_path(int peer_rank) const override {
    return has_send_peer(peer_rank);
  }
  bool has_wait_path(int peer_rank) const override {
    return has_recv_peer(peer_rank);
  }

  unsigned put_async(int peer_rank, void* local_ptr,
                     uint32_t local_buffer_id, void* remote_ptr,
                     uint32_t remote_buffer_id, size_t len) override;
  unsigned signal_async(int peer_rank, uint64_t tag) override;
  unsigned wait_async(int peer_rank, uint64_t expected_tag,
                      std::optional<WaitTarget> target = std::nullopt) override;

  bool poll_completion(unsigned id) override;
  bool wait_completion(unsigned id) override;
  bool request_failed(unsigned id) override;
  void release_request(unsigned id) override;

 private:
  enum class RequestState : uint8_t {
    Free = 0,
    Reserved = 1,
    InFlight = 2,
    Completed = 3,
    Failed = 4,
  };

  struct PendingRequestSlot {
    enum class Kind : uint8_t {
      DataPut = 0,
      DataWait = 1,
      SignalSend = 2,
      SignalWait = 3,
    };
    std::unique_ptr<::uccl::ucclRequest> request;
    Kind kind = Kind::DataPut;
    RequestState state = RequestState::Free;
    uint32_t generation = 1;
    uint64_t control_tag = 0;
    uint64_t expected_signal_tag = 0;
    ::uccl::Mhandle* control_mhandle = nullptr;
    bool failed = false;
  };

  struct PeerContext {
    ::uccl::UcclFlow* send_flow = nullptr;
    ::uccl::UcclFlow* recv_flow = nullptr;
  };

  int local_gpu_idx_;

  std::unique_ptr<::uccl::RDMAEndpoint> endpoint_;
  std::unordered_map<int, PeerContext> peer_contexts_;
  std::unordered_map<uint32_t, ::uccl::Mhandle*> buffer_id_to_mhandle_;

  bool connect_to_peer(int peer_rank, std::string remote_ip,
                       uint16_t remote_port, int local_dev_idx,
                       int local_gpu_idx, int remote_dev_idx,
                       int remote_gpu_idx);
  bool accept_from_peer(int peer_rank, std::string const& expected_remote_ip,
                        int expected_remote_dev_idx,
                        int expected_remote_gpu_idx,
                        uint16_t expected_remote_port = 0);

  int send_async_uccl(int peer_rank, void* local_ptr, size_t len,
                      uint32_t local_buffer_id, uint32_t remote_buffer_id,
                      uint64_t request_id,
                      ::uccl::Mhandle* local_mh_override = nullptr);
  int recv_async_uccl(int peer_rank, void* local_ptr, size_t len,
                      uint32_t local_buffer_id, uint64_t request_id,
                      ::uccl::Mhandle* local_mh_override = nullptr);

  bool has_send_peer(int peer_rank) const;
  bool has_recv_peer(int peer_rank) const;

  static constexpr uint32_t kRequestSlotBits = 13;
  static constexpr uint32_t kRequestSlotCount = (1u << kRequestSlotBits);
  static constexpr uint32_t kRequestSlotMask = kRequestSlotCount - 1u;
  std::unique_ptr<PendingRequestSlot[]> request_slots_;
  std::atomic<uint32_t> request_alloc_cursor_{0};

  static unsigned make_request_id(uint32_t slot_idx, uint32_t generation) {
    return static_cast<unsigned>((generation << kRequestSlotBits) | slot_idx);
  }
  static uint32_t request_slot_index(unsigned request_id) {
    return static_cast<uint32_t>(request_id) & kRequestSlotMask;
  }
  static uint32_t request_generation(unsigned request_id) {
    return static_cast<uint32_t>(request_id) >> kRequestSlotBits;
  }
  PendingRequestSlot* try_acquire_request_slot(unsigned* out_request_id);
  PendingRequestSlot* resolve_request_slot_locked(unsigned request_id);
  void release_request_slot_locked(unsigned request_id);

  mutable std::mutex mu_;
};

}  // namespace Transport
}  // namespace UKernel
