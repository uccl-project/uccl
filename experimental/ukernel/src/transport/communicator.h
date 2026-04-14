#pragma once

#include "../../include/config.h"
#include "memory/ipc_manager.h"
#include "memory/mr_manager.h"
#include "memory/shm_manager.h"
#include "oob/oob.h"
#include "request.h"
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace UKernel {
namespace Transport {

class TransportAdapter;
class IpcAdapter;
class TcpTransportAdapter;
class RdmaTransportAdapter;
class UcclTransportAdapter;

class Communicator {
 public:
  // Lifecycle.
  Communicator(
      int gpu_id, int rank, int world_size,
      std::shared_ptr<CommunicatorConfig> config =
          std::make_shared<CommunicatorConfig>(CommunicatorConfig::from_env()));
  ~Communicator();

  // Peer bootstrap / transport path.
  bool connect(int rank);
  bool accept(int rank);
  PeerTransportKind peer_transport_kind(int rank) const;
  bool same_host(int rank) const;

  // Async data path.
  unsigned isend(int rank, LocalSlice src,
                 std::optional<RemoteSlice> dst_hint = std::nullopt);
  unsigned irecv(int rank, LocalSlice dst);
  bool poll(unsigned const req);
  void release(unsigned const req);
  bool wait_finish(unsigned const req);
  bool wait_finish(std::vector<unsigned> const& reqs);

  // Identity.
  int rank() const { return global_rank_; }
  int world_size() const { return world_size_; }

  // Completion notification hook.
  std::shared_ptr<void> register_completion_notifier(
      std::function<void(unsigned, std::chrono::steady_clock::time_point)> cb);

  // MR metadata exchange.
  MR reg_mr(void* local_buf, size_t len);
  bool dereg_mr(void* local_buf);
  bool notify_named_mrs(int remote_rank, uint64_t generation,
                        NamedMRInfos const& infos);
  bool wait_named_mrs(int remote_rank, uint64_t generation,
                      NamedMRInfos& infos);
  MR get_local_mr(void* local_buf);
  MR get_local_mr(uint32_t mr_id);
  MR get_remote_mr(int remote_rank, uint32_t mr_id);

  // IPC metadata exchange.
  bool notify_ipc_buffer(int remote_rank, uint32_t ipc_id, void* local_buf,
                         size_t len, uint64_t binding_version = 0);
  bool wait_ipc_buffer(int remote_rank, uint32_t ipc_id,
                       uint64_t expected_binding_version = 0);
  bool resolve_ipc_buffer_pointer(int remote_rank, uint32_t ipc_id,
                                  size_t offset, size_t bytes, void** out_ptr,
                                  int* out_device_idx);

  bool register_remote_ipc_cache(int remote_rank, gpuIpcMemHandle_t handle,
                                 IPCItem const& ipc);
  IPCItem get_remote_ipc_cache(int remote_rank, gpuIpcMemHandle_t handle);

  // Adapter-facing IPC metadata helpers.
  int local_gpu_idx() const { return local_gpu_idx_; }
  bool ipc_has_fresh_remote_ipc_buffer(int remote_rank, uint32_t ipc_id,
                                       uint64_t expected_binding_version) const;
  bool ipc_fetch_remote_ipc_buffer(int remote_rank, uint32_t ipc_id,
                                   uint64_t expected_binding_version = 0);
  void ipc_invalidate_remote_ipc_buffer(int remote_rank, uint32_t ipc_id);

  // SHM bounce.
  void* get_or_open_bounce_shm(std::string const& shm_name);
  void clear_bounce_shm_cache();

 private:
  struct ResolvedPeer {
    CommunicatorMeta local_meta;
    CommunicatorMeta remote_meta;
    PeerTransportKind kind = PeerTransportKind::Unknown;
  };

  struct PeerState {
    bool has_meta = false;
    CommunicatorMeta meta{};
    PeerTransportKind kind = PeerTransportKind::Unknown;
    bool connected = false;
  };

  struct TrackedRequest {
    enum class SlotState : uint8_t {
      Free = 0,
      Reserved = 1,
      InFlight = 2,
      Completed = 3,
      Failed = 4,
      Releasing = 5
    };
    std::atomic<SlotState> state{SlotState::Free};
    std::atomic<uint32_t> generation{0};
    unsigned request_id = 0;
    unsigned adapter_request_id = 0;
    int peer_rank = -1;
    PeerTransportKind kind = PeerTransportKind::Unknown;
    bool notified = false;
    bool needs_host_to_device_copy = false;
    bool host_copy_submitted = false;
    void* completion_buffer = nullptr;
    size_t completion_offset = 0;
    size_t completion_bytes = 0;
    gpuEvent_t host_copy_event = nullptr;
    SHMItem bounce;
    std::shared_ptr<SHMItem> bounce_owner;
  };

  // Adapter bootstrap / selection.
  UcclTransportAdapter& ensure_uccl_adapter(CommunicatorMeta const& local_meta);
  RdmaTransportAdapter& ensure_rdma_adapter(CommunicatorMeta const& local_meta);
  void bind_rdma_backend_if_needed();
  bool bootstrap_rdma_peer_oob(int rank, RdmaTransportAdapter& rdma_adapter);
  bool exchange_uccl_peer_info(int rank, UcclTransportAdapter& uccl_adapter,
                               bool as_connector,
                               UCCLP2PInfo* out_remote_p2p_info);
  TcpTransportAdapter& ensure_tcp_adapter(CommunicatorMeta const& local_meta);

  // Peer-state lifecycle.
  PeerTransportKind get_peer_transport_kind(int rank) const;
  bool has_peer_path(int rank) const;
  void mark_peer_path_ready(int rank, PeerTransportKind kind);
  void exchange_peer_metas();
  ResolvedPeer resolve_peer(int rank) const;
  bool try_fallback_tcp_connect(int rank, CommunicatorMeta const& local_meta);
  bool try_fallback_tcp_accept(int rank, CommunicatorMeta const& local_meta);

  // Request tracking / progress.
  bool poll_request_completion(unsigned id, bool blocking);
  TrackedRequest* allocate_request_slot(unsigned* out_req_id);
  TrackedRequest* resolve_request_slot(unsigned req_id) const;
  bool try_release_request_slot(unsigned req_id, TrackedRequest* out_snapshot);
  bool enqueue_active_request(unsigned req_id);
  bool dequeue_active_request(unsigned* out_req_id);
  static unsigned make_request_id(uint32_t slot_idx, uint32_t generation);
  static uint32_t request_slot_index(unsigned req_id);
  static uint32_t request_generation(unsigned req_id);
  void notify_request_completion();

  // RDMA MR lifecycle.
  void register_existing_local_mrs_with_uccl();
  bool ensure_uccl_memory_registered(uint64_t mr_id, void* ptr, size_t len);
  void register_existing_local_mrs_with_rdma();
  bool ensure_rdma_memory_registered(uint64_t mr_id, void* ptr, size_t len);

  // IPC metadata cache lifecycle.
  bool fetch_ipc_buffer(int remote_rank, uint32_t ipc_id,
                        uint64_t expected_binding_version = 0);
  bool has_fresh_remote_ipc_buffer(int remote_rank, uint32_t ipc_id,
                                   uint64_t expected_binding_version) const;
  void invalidate_remote_ipc_buffer(int remote_rank, uint32_t ipc_id);

  // Request cleanup helpers.
  void cleanup_tracked_request(TrackedRequest& tracked);
  bool complete_host_bounce_recv(TrackedRequest& tracked, bool blocking);
  SHMManager& require_shm_manager(char const* caller);

  // Background progress loop: advances completion state for all in-flight
  // requests and emits user completion callbacks for requests that finish.
  void progress_loop();

  // Identity.
  int local_gpu_idx_;
  int global_rank_;
  int world_size_;

  // Memory/resource managers.
  MRManager mr_manager_;
  IPCManager ipc_manager_;
  std::optional<SHMManager> shm_manager_;

  // Transport adapters.
  std::unique_ptr<UcclTransportAdapter> uccl_adapter_;
  std::unique_ptr<RdmaTransportAdapter> rdma_adapter_;
  std::unique_ptr<TcpTransportAdapter> tcp_adapter_;
  std::shared_ptr<IpcAdapter> ipc_adapter_;
  gpuStream_t host_copy_stream_ = nullptr;

  // Peer/bootstrapping state.
  mutable std::mutex peer_mu_;
  std::vector<PeerState> peer_states_;
  std::shared_ptr<CommunicatorConfig> config_;
  std::shared_ptr<Exchanger> exchanger_client_;

  // Adapter dispatch.
  TransportAdapter* get_adapter(PeerTransportKind kind);

  // Request lifecycle (slot/ring/event).
  static constexpr uint32_t kRequestSlotBits = 13;
  static constexpr uint32_t kRequestSlotCount = (1u << kRequestSlotBits);
  static constexpr uint32_t kRequestGenerationMask =
      (1u << (32u - kRequestSlotBits)) - 1u;
  std::unique_ptr<TrackedRequest[]> request_slots_;
  std::atomic<uint32_t> request_alloc_cursor_{0};
  std::atomic<uint32_t> inflight_request_count_{0};
  static constexpr uint32_t kActiveRingSize = (1u << 15);
  std::unique_ptr<std::atomic<unsigned>[]> active_ring_;
  std::atomic<uint32_t> active_head_{0};
  std::atomic<uint32_t> active_tail_{0};
  int completion_event_fd_ = -1;
  std::atomic<uint64_t> completion_seq_{0};
  mutable std::mutex mr_exchange_mu_;

  // Despite the historical name, this thread now serves as the communicator's
  // background progress engine. register_completion_notifier() only attaches
  // user callbacks to this always-on loop.
  std::atomic<bool> progress_running_{false};
  std::atomic<bool> progress_started_{false};
  std::thread progress_thread_;
  std::condition_variable progress_cv_;
  std::mutex progress_mu_;

  struct NotifyTarget {
    std::function<void(unsigned, std::chrono::steady_clock::time_point)> emit;
  };
  std::vector<std::weak_ptr<NotifyTarget>> notify_targets_;

  // RDMA-family registration caches.
  mutable std::mutex uccl_reg_mu_;
  std::unordered_set<uint64_t> uccl_direct_reg_failed_mrs_;
  std::unordered_set<uint64_t> uccl_registered_mrs_;
  
  // RDMA registration cache.
  mutable std::mutex rdma_reg_mu_;
  mutable std::mutex rdma_bootstrap_mu_;
  std::unordered_set<uint64_t> rdma_direct_reg_failed_mrs_;
  std::unordered_set<uint64_t> rdma_registered_mrs_;
  bool rdma_backend_bound_ = false;

  // IPC binding versions.
  mutable std::mutex ipc_gen_mu_;
  std::unordered_map<int, std::unordered_map<uint32_t, uint64_t>>
      local_ipc_binding_versions_;
};

}  // namespace Transport
}  // namespace UKernel
