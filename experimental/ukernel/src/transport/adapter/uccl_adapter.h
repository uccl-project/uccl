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
  // Must stay aligned with ucclParamNUM_ENGINES() inside collective/rdma.
  int num_engines = 0;
};

class UcclTransportAdapter final : public TransportAdapter {
 public:
  // Lifecycle.
  UcclTransportAdapter(int local_gpu_idx, int world_size,
                       UcclTransportConfig config);
  ~UcclTransportAdapter();

  // Endpoint discovery.
  uint16_t get_p2p_listen_port(int dev_idx) const;
  std::string get_p2p_listen_ip(int dev_idx) const;
  int get_best_dev_idx(int gpu_idx) const;

  // Memory registration.
  bool is_memory_registered(uint64_t mr_id) const;
  bool register_memory(uint64_t mr_id, void* ptr, size_t len);
  void deregister_memory(uint64_t mr_id);
  bool is_initialized() const { return endpoint_ != nullptr; }

  // TransportAdapter path-state overrides.
  bool ensure_peer(PeerConnectSpec const& spec) override;
  bool has_peer(int peer_rank) const override {
    return has_send_peer(peer_rank) && has_recv_peer(peer_rank);
  }

  // TransportAdapter async data path overrides.
  unsigned send_async(int peer_rank, void* local_ptr, size_t len,
                      uint64_t local_mr_id,
                      std::optional<RemoteSlice> remote_hint,
                      BounceBufferProvider bounce_provider = nullptr) override;
  unsigned recv_async(int peer_rank, void* local_ptr, size_t len,
                      uint64_t local_mr_id,
                      BounceBufferProvider bounce_provider = nullptr) override;

  // TransportAdapter completion overrides.
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
    std::unique_ptr<::uccl::ucclRequest> request;
    RequestState state = RequestState::Free;
    uint32_t generation = 1;
    bool failed = false;
  };

  struct PeerContext {
    ::uccl::UcclFlow* send_flow = nullptr;
    ::uccl::UcclFlow* recv_flow = nullptr;
    int peer_rank = -1;
    std::string remote_ip;
    int remote_dev_idx = -1;
    int remote_gpu_idx = -1;
  };

  int local_gpu_idx_;

  std::unique_ptr<::uccl::RDMAEndpoint> endpoint_;
  std::unordered_map<int, PeerContext> peer_contexts_;
  std::unordered_map<uint64_t, ::uccl::Mhandle*> mr_id_to_mhandle_;

  // Internal peer-connection helpers.
  bool connect_to_peer(int peer_rank, std::string remote_ip,
                       uint16_t remote_port, int local_dev_idx,
                       int local_gpu_idx, int remote_dev_idx,
                       int remote_gpu_idx);
  bool accept_from_peer(int peer_rank, std::string const& expected_remote_ip,
                        int expected_remote_dev_idx,
                        int expected_remote_gpu_idx,
                        uint16_t expected_remote_port = 0);

  // Internal async submit helpers.
  int send_async_uccl(int peer_rank, void* local_ptr, size_t len,
                      uint64_t local_mr_id, uint64_t remote_mr_id,
                      uint64_t request_id,
                      RemoteSlice const* remote_slice = nullptr);
  int recv_async_uccl(int peer_rank, void* local_ptr, size_t len,
                      uint64_t local_mr_id, uint64_t request_id);

  // Internal peer-state helpers.
  bool has_send_peer(int peer_rank) const;
  bool has_recv_peer(int peer_rank) const;

  // Internal request-slot management.
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
