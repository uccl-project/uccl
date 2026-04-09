#pragma once

#include "../../include/config.h"
#include "adapter/tcp_adapter.h"
#include "adapter/transport_adapter.h"
#include "adapter/uccl_adapter.h"
#include "memory/ipc_manager.h"
#include "memory/mr_manager.h"
#include "memory/shm_manager.h"
#include "oob/oob.h"
#include "request.h"
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace UKernel {
namespace Transport {

class IpcChannel;

class Communicator {
 public:
  Communicator(
      int gpu_id, int rank, int world_size,
      std::shared_ptr<CommunicatorConfig> config =
          std::make_shared<CommunicatorConfig>(CommunicatorConfig::from_env()));
  ~Communicator();

  bool connect(int rank);
  bool accept(int rank);
  unsigned isend(int rank, LocalSlice src,
                 std::optional<RemoteSlice> dst_hint = std::nullopt);
  unsigned irecv(int rank, LocalSlice dst);
  bool poll(unsigned const req);
  void release(unsigned const req);
  bool wait_finish(unsigned const req);
  bool wait_finish(std::vector<unsigned> const& reqs);
  int rank() const { return global_rank_; }
  int world_size() const { return world_size_; }
  PeerTransportKind peer_transport_kind(int rank) const;
  bool same_host(int rank) const;

  std::shared_ptr<void> register_completion_notifier(
      std::function<void(unsigned, std::chrono::steady_clock::time_point)> cb);

  MR reg_mr(void* local_buf, size_t len);
  bool dereg_mr(void* local_buf);
  bool notify_named_mrs(int remote_rank, uint64_t generation,
                        NamedMRInfos const& infos);
  bool wait_named_mrs(int remote_rank, uint64_t generation,
                      NamedMRInfos& infos);
  MR get_local_mr(void* local_buf);
  MR get_local_mr(uint32_t mr_id);
  MR get_remote_mr(int remote_rank, uint32_t mr_id);
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
    unsigned adapter_request_id = 0;
    int peer_rank = -1;
    PeerTransportKind kind = PeerTransportKind::Unknown;
    bool completed = false;
    bool failed = false;
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

  UcclTransportAdapter& ensure_uccl_adapter(CommunicatorMeta const& local_meta,
                                            CommunicatorMeta const& peer_meta);
  TcpTransportAdapter& ensure_tcp_adapter(CommunicatorMeta const& local_meta);
  std::shared_ptr<IpcChannel> get_ipc_channel_by_rank(int rank);
  PeerTransportKind get_peer_transport_kind(int rank) const;
  bool has_peer_path(int rank) const;
  void mark_peer_path_ready(int rank, PeerTransportKind kind);
  bool poll_request_completion(unsigned id, bool blocking);
  void register_existing_local_mrs_with_uccl();
  bool ensure_uccl_memory_registered(uint64_t mr_id, void* ptr, size_t len);
  bool fetch_ipc_buffer(int remote_rank, uint32_t ipc_id,
                        uint64_t expected_binding_version = 0);
  bool has_fresh_remote_ipc_buffer(int remote_rank, uint32_t ipc_id,
                                   uint64_t expected_binding_version) const;
  void invalidate_remote_ipc_buffer(int remote_rank, uint32_t ipc_id);
  void cleanup_tracked_request(TrackedRequest& tracked);
  bool complete_host_bounce_recv(TrackedRequest& tracked, bool blocking);
  void exchange_peer_metas();
  ResolvedPeer resolve_peer(int rank) const;
  bool try_fallback_tcp_connect(int rank, CommunicatorMeta const& local_meta);
  bool try_fallback_tcp_accept(int rank, CommunicatorMeta const& local_meta,
                               CommunicatorMeta const& remote_meta);
  bool do_connect(int rank);
  bool do_accept(int rank);
  // Background progress loop: advances completion state for all in-flight
  // requests and emits user completion callbacks for requests that finish.
  void progress_loop();

  int local_gpu_idx_;
  int global_rank_;
  int world_size_;
  MRManager mr_manager_;
  IPCManager ipc_manager_;
  std::unique_ptr<UcclTransportAdapter> uccl_adapter_;
  std::unique_ptr<TcpTransportAdapter> tcp_adapter_;
  std::shared_ptr<IpcChannel> ipc_channel_;
  gpuStream_t host_copy_stream_ = nullptr;

  std::shared_ptr<ShmRingExchanger> shm_control_;
  mutable std::mutex peer_mu_;
  std::vector<PeerState> peer_states_;

  std::shared_ptr<CommunicatorConfig> config_;
  std::shared_ptr<Exchanger> exchanger_client_;
  std::unique_ptr<SHMManager> shm_manager_;

  TransportAdapter* get_adapter(int rank);
  TransportAdapter* get_adapter(PeerTransportKind kind);

  std::unordered_map<unsigned, TrackedRequest> requests_map_;
  mutable std::mutex req_mu_;
  mutable std::mutex mr_exchange_mu_;
  std::atomic<unsigned> next_request_id_{1};

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
  mutable std::mutex uccl_reg_mu_;
  std::unordered_set<uint64_t> uccl_direct_reg_failed_mrs_;
  std::unordered_set<uint64_t> uccl_registered_mrs_;
  mutable std::mutex ipc_gen_mu_;
  std::unordered_map<int, std::unordered_map<uint32_t, uint64_t>>
      local_ipc_binding_versions_;

  friend class IpcChannel;
};

}  // namespace Transport
}  // namespace UKernel
