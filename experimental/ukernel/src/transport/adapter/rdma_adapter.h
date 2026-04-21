#pragma once

#include "p2p/rdma/define.h"
#include "transport_adapter.h"
#include <atomic>
#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

class NICEndpoint;

namespace UKernel {
namespace Transport {

struct RdmaTransportConfig {
  std::string local_ip;
  int connect_timeout_ms = 10000;
};

class RdmaTransportAdapter final : public TransportAdapter {
 public:
  RdmaTransportAdapter(int local_gpu_idx, int local_rank, int world_size,
                       RdmaTransportConfig config);
  ~RdmaTransportAdapter();

  bool build_send_bootstrap(int peer_rank, std::string* out_payload);
  bool build_recv_bootstrap(int peer_rank,
                            std::string const& remote_send_payload,
                            std::string* out_payload);
  bool finalize_send_bootstrap(int peer_rank,
                               std::string const& remote_recv_payload);
  void rollback_peer_bootstrap(int peer_rank);

  bool is_memory_registered(uint32_t buffer_id) const;
  bool register_memory(uint32_t buffer_id, void* ptr, size_t len);
  void deregister_memory(uint32_t buffer_id);
  bool query_memory_rkey(uint32_t buffer_id, uint32_t* out_rkey) const;
  bool query_memory_rkeys(
      uint32_t buffer_id,
      std::array<uint32_t, kNICContextNumber>* out_rkeys) const;
  bool is_initialized() const { return initialized_; }

  bool ensure_peer(PeerConnectSpec const& spec) override;
  bool has_peer(int peer_rank) const override;

  unsigned send_async(int peer_rank, void* local_ptr, size_t len,
                      uint32_t local_buffer_id,
                      std::optional<RemoteSlice> remote_hint,
                      BounceBufferProvider bounce_provider = nullptr) override;
  unsigned recv_async(int peer_rank, void* local_ptr, size_t len,
                      uint32_t local_buffer_id,
                      BounceBufferProvider bounce_provider = nullptr) override;

  bool poll_completion(unsigned id) override;
  bool wait_completion(unsigned id) override;
  bool request_failed(unsigned id) override;
  void release_request(unsigned id) override;

 private:
  enum class RequestState : uint8_t {
    Init = 0,
    Posted = 1,
    Completed = 2,
    Failed = 3,
  };

  enum class RequestKind : uint8_t { Send = 0, Recv = 1, Write = 2 };

  struct AdapterRequest {
    RequestKind kind = RequestKind::Send;
    int peer_rank = -1;
    int64_t token = -1;
  };

  struct PendingRequestSlot {
    AdapterRequest request{};
    bool in_use = false;
    uint32_t generation = 1;
    RequestState state = RequestState::Init;
    bool failed = false;
  };

  struct LocalMr {
    std::shared_ptr<RegMemBlock> block;
  };

  struct BootstrapPayload;

  bool initialize_endpoint();
  static MemoryType detect_memory_type(void* ptr);
  static std::string serialize_bootstrap(BootstrapPayload const& payload);
  static bool deserialize_bootstrap(std::string const& payload,
                                    BootstrapPayload* out_payload);
  bool is_backend_request_done(AdapterRequest const& request, bool* ok);

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
  PendingRequestSlot* try_acquire_request_slot(unsigned* out_request_id);
  PendingRequestSlot* resolve_request_slot_locked(unsigned request_id);
  void release_request_slot_locked(unsigned request_id);

  int local_gpu_idx_;
  int local_rank_;
  int world_size_;
  bool initialized_ = false;
  RdmaTransportConfig config_;
  std::unique_ptr<NICEndpoint> endpoint_;
  std::unordered_set<int> ready_peers_;
  std::unordered_map<int, std::shared_ptr<OOBMetaData>> pending_peer_oob_;
  std::unordered_map<uint32_t, LocalMr> buffer_id_to_local_mr_;
  std::unordered_map<uint32_t, bool> buffer_registering_;

  std::unique_ptr<PendingRequestSlot[]> request_slots_;
  std::atomic<uint32_t> request_alloc_cursor_{0};
  mutable std::condition_variable mr_cv_;
  mutable std::mutex mu_;
};

}  // namespace Transport
}  // namespace UKernel
