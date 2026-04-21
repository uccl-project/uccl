#pragma once

#include "p2p/rdma/define.h"
#include "transport_adapter.h"
#include <atomic>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <condition_variable>
#include <unordered_map>
#include <vector>

class SendConnection;
class RecvConnection;
class RdmaContext;
class MemoryAllocator;
class SendControlChannel;
class RDMADataChannel;
struct RegMemBlock;
struct MrCacheEntry;
enum class MemoryType;
struct ChannelMetaData;
struct RemoteMemInfo;

namespace UKernel {
namespace Transport {

struct RdmaTransportConfig {
  // Reserved for optional runtime knobs in the lightweight adapter.
  int num_engines = 0;
};

class RdmaTransportAdapter final : public TransportAdapter {
 public:
  RdmaTransportAdapter(int local_gpu_idx, int local_rank, int world_size,
                       RdmaTransportConfig config);
  ~RdmaTransportAdapter();

  // Bootstrap lifecycle.
  bool build_send_bootstrap(int peer_rank, std::string* out_payload);
  bool build_recv_bootstrap(int peer_rank, std::string const& remote_send_payload,
                            std::string* out_payload);
  bool finalize_send_bootstrap(int peer_rank,
                               std::string const& remote_recv_payload);
  void rollback_peer_bootstrap(int peer_rank);

  // Registration lifecycle.
  bool is_memory_registered(uint32_t buffer_id) const;
  bool register_memory(uint32_t buffer_id, void* ptr, size_t len);
  void deregister_memory(uint32_t buffer_id);
  // Query one valid local rkey for a registered buffer id.
  bool query_memory_rkey(uint32_t buffer_id, uint32_t* out_rkey) const;
  bool query_memory_rkeys(
      uint32_t buffer_id,
      std::array<uint32_t, kNICContextNumber>* out_rkeys) const;
  bool is_initialized() const { return initialized_; }

  // TransportAdapter compatibility.
  bool ensure_peer(PeerConnectSpec const& spec) override;
  bool has_peer(int peer_rank) const override {
    return has_send_peer(peer_rank) && has_recv_peer(peer_rank);
  }

  // Async request path.
  unsigned send_async(int peer_rank, void* local_ptr, size_t len,
                      uint32_t local_buffer_id,
                      std::optional<RemoteSlice> remote_hint,
                      BounceBufferProvider bounce_provider = nullptr) override;
  unsigned recv_async(int peer_rank, void* local_ptr, size_t len,
                      uint32_t local_buffer_id,
                      BounceBufferProvider bounce_provider = nullptr) override;

  // Completion / request state.
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

  enum class RequestKind : uint8_t { Send = 0, Recv = 1, DirectSend = 2 };

  struct AdapterRequest {
    RequestKind kind = RequestKind::Send;
    int peer_rank = -1;
    int64_t token = -1;
    uint32_t expected_completions = 0;
    bool send_mode_tracked = false;
  };

  struct PendingRequestSlot {
    AdapterRequest request{};
    bool in_use = false;
    uint32_t generation = 1;
    RequestState state = RequestState::Init;
    bool failed = false;
  };

  struct PeerContext {
    enum class SendMode : uint8_t { Idle = 0, Queued = 1, Direct = 2 };
    std::shared_ptr<SendConnection> send_conn;
    std::shared_ptr<RecvConnection> recv_conn;
    std::shared_ptr<SendControlChannel> send_ctrl;
    std::array<std::shared_ptr<RDMADataChannel>, kQpNumPerChannel> send_channels{};
    SendMode send_mode = SendMode::Idle;
    uint32_t queued_inflight = 0;
    uint32_t direct_inflight = 0;
    bool send_polling_paused = false;
  };

  struct LocalMr {
    std::shared_ptr<RegMemBlock> block;
    struct CacheRef {
      std::shared_ptr<RdmaContext> context;
      MrCacheEntry* entry = nullptr;
    };
    std::vector<CacheRef> cache_refs;
  };

  struct HandshakePacket;
  struct SendBuildState;

  // Initialization helpers.
  bool initialize_contexts();
  std::shared_ptr<RdmaContext> get_context_by_channel_id(uint32_t channel_id) const;
  static std::string serialize_handshake(HandshakePacket const& packet);
  static bool deserialize_handshake(std::string const& payload,
                                    HandshakePacket* out_packet);
  bool validate_remote_packet(HandshakePacket const& remote,
                              int expected_src_rank) const;

  // Peer setup helpers.
  bool create_recv_from_remote_send(int peer_rank,
                                    HandshakePacket const& remote_send,
                                    HandshakePacket* out_local_recv);
  bool finalize_send_from_remote_recv(int peer_rank,
                                      HandshakePacket const& remote_recv);
  static MemoryType detect_memory_type(void* ptr);
  bool has_send_peer(int peer_rank) const;
  bool has_recv_peer(int peer_rank) const;
  bool wait_direct_send_meta(int peer_rank, SendReqMeta* out_meta, int* out_index);
  bool poll_direct_send_completions(int peer_rank);
  void maybe_pause_send_polling_locked(PeerContext& ctx);
  void maybe_resume_send_polling_locked(PeerContext& ctx);
  bool try_enter_send_mode_locked(int peer_rank, RequestKind kind);
  void leave_send_mode_locked(int peer_rank, RequestKind kind);
  void finalize_request_accounting_locked(PendingRequestSlot& slot);
  bool prepare_send_submission(int peer_rank, uint32_t local_buffer_id,
                               uint64_t request_id,
                               std::shared_ptr<SendConnection>* out_send_conn,
                               LocalMr* out_local_mr);
  int64_t submit_queued_send(std::shared_ptr<SendConnection> const& send_conn,
                             std::shared_ptr<RegMemBlock> const& send_mem) const;
  int64_t submit_direct_send(
      int peer_rank, std::shared_ptr<RegMemBlock> const& send_mem,
      RemoteSlice const& remote_slice, uint32_t* out_expected_completions);

  // Async submit helpers.
  int send_async_rdma(int peer_rank, void* local_ptr, size_t len,
                      uint32_t local_buffer_id, uint32_t remote_buffer_id,
                      uint64_t request_id,
                      RemoteSlice const* remote_slice = nullptr);
  int recv_async_rdma(int peer_rank, void* local_ptr, size_t len,
                      uint32_t local_buffer_id, uint64_t request_id);

  // Completion helpers.
  bool is_backend_request_done(AdapterRequest const& request, bool* ok);

  // Request slot helpers.
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
  bool initialized_ = false;
  int numa_node_ = 0;

  std::vector<std::shared_ptr<RdmaContext>> contexts_;
  // Adapter-level active context slot set used by channel routing and MR
  // registration. Per-MR state only keeps release handles, not topology.
  std::vector<uint32_t> active_context_slots_;
  std::shared_ptr<MemoryAllocator> allocator_;
  std::unordered_map<int, PeerContext> peer_contexts_;
  std::unordered_map<int, std::unique_ptr<SendBuildState>> send_build_states_;
  std::unordered_map<uint32_t, LocalMr> buffer_id_to_local_mr_;
  std::unordered_map<uint32_t, bool> buffer_registering_;

  std::unique_ptr<PendingRequestSlot[]> request_slots_;
  std::atomic<uint32_t> request_alloc_cursor_{0};
  std::atomic<uint64_t> direct_wr_id_cursor_{1ull << 32};
  std::unordered_map<uint64_t, uint32_t> direct_completion_counts_;
  mutable std::condition_variable mr_cv_;
  mutable std::mutex mu_;
};

}  // namespace Transport
}  // namespace UKernel
