#pragma once

#include "../../include/config.h"
#include "host_bounce_pool.h"
#include "ipc_cache.h"
#include "memory_registry.h"
#include "oob.h"
#include "peer_session.h"
#include "request.h"
#include "tcp_transport_adapter.h"
#include "uccl_transport_adapter.h"
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace UKernel {
namespace Transport {

class IpcChannel;

class Communicator {
 public:
  Communicator(int gpu_id, int rank, int world_size,
               std::shared_ptr<CommunicatorConfig> config =
                   std::make_shared<CommunicatorConfig>(
                       CommunicatorConfig::from_env()));
  ~Communicator();

  bool connect_to(int rank);
  bool accept_from(int rank);
  bool connect_bidir(int rank);
  unsigned isend(int rank, void* ptr, size_t offset, size_t len,
                 uint32_t local_mr_id, uint32_t remote_mr_id, bool on_gpu);
  unsigned irecv(int rank, void* ptr, size_t offset, size_t len, bool on_gpu);
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
                         size_t len);
  bool wait_ipc_buffer(int remote_rank, uint32_t ipc_id);
  bool resolve_ipc_buffer_pointer(int remote_rank, uint32_t ipc_id,
                                  size_t offset, size_t bytes,
                                  void** out_ptr, int* out_device_idx);

  bool register_remote_ipc_cache(int remote_rank, gpuIpcMemHandle_t handle,
                                 IpcCacheManager::IpcCache const& cache);
  IpcCacheManager::IpcCache get_remote_ipc_cache(int remote_rank,
                                                 gpuIpcMemHandle_t handle);

 private:
  struct ResolvedPeer {
    CommunicatorMeta local_meta;
    CommunicatorMeta remote_meta;
    PeerTransportKind kind = PeerTransportKind::Unknown;
  };

  struct TrackedRequest {
    int peer_rank = -1;
    PeerTransportKind kind = PeerTransportKind::Unknown;
    std::shared_ptr<Request> ipc_request;
    bool completed = false;
    bool failed = false;
    bool notified = false;
    bool needs_host_to_device_copy = false;
    bool host_copy_submitted = false;
    void* completion_buffer = nullptr;
    size_t completion_offset = 0;
    size_t completion_bytes = 0;
    gpuEvent_t host_copy_event = nullptr;
    HostBouncePool::Lease bounce;
  };

  struct RemoteIpcBufferState {
    gpuIpcMemHandle_t handle{};
    uintptr_t base_offset = 0;
    size_t bytes = 0;
    int device_idx = -1;
    bool valid = false;
    void* direct_ptr = nullptr;
  };

  UcclTransportAdapter& ensure_uccl_adapter(CommunicatorMeta const& local_meta,
                                            CommunicatorMeta const& peer_meta);
  TcpTransportAdapter& ensure_tcp_adapter(CommunicatorMeta const& local_meta);
  std::shared_ptr<IpcChannel> get_ipc_channel_by_rank(int rank);
  PeerTransportKind get_peer_transport_kind(int rank) const;
  bool poll_request_completion(unsigned id, bool blocking);
  void register_existing_local_mrs_with_uccl();
  void discard_uccl_registration(uint64_t mr_id);
  bool ensure_uccl_memory_registered(uint64_t mr_id, void* ptr, size_t len);
  void cleanup_tracked_request(unsigned id, TrackedRequest& tracked);
  bool complete_host_bounce_recv(TrackedRequest& tracked, bool blocking);
  void exchange_peer_metas();
  ResolvedPeer resolve_peer(int rank) const;
  uint64_t next_ipc_match_seq(int rank, RequestType type);
  void cache_peer_session(int rank, PeerTransportKind kind,
                          bool mark_send_ready, bool mark_recv_ready);
  bool try_fallback_tcp_connect(int rank, CommunicatorMeta const& local_meta);
  bool try_fallback_tcp_accept(int rank, CommunicatorMeta const& local_meta,
                               CommunicatorMeta const& remote_meta);
  // Background progress loop for transport requests. It advances completion
  // state for all in-flight requests and emits user completion callbacks for
  // requests that become fully completed.
  void completion_notifier_loop();

  int local_gpu_idx_;
  int global_rank_;
  int world_size_;
  MemoryRegistry memory_registry_;
  std::unique_ptr<UcclTransportAdapter> uccl_adapter_;
  std::unique_ptr<TcpTransportAdapter> tcp_adapter_;
  std::shared_ptr<IpcChannel> ipc_channel_;
  gpuStream_t host_copy_stream_ = nullptr;

  std::shared_ptr<ShmRingExchanger> shm_control_;
  std::shared_ptr<PeerSessionManager> peer_manager_;

  std::shared_ptr<CommunicatorConfig> config_;
  std::shared_ptr<Exchanger> exchanger_client_;
  std::unique_ptr<HostBouncePool> host_bounce_pool_;

  std::unordered_map<unsigned, TrackedRequest> requests_map_;
  mutable std::mutex req_mu_;
  mutable std::mutex mr_exchange_mu_;
  std::atomic<unsigned> next_request_id_{1};
  std::mutex ipc_match_seq_mu_;
  std::vector<uint64_t> next_send_match_seq_;
  std::vector<uint64_t> next_recv_match_seq_;

  // Despite the historical name, this thread now serves as the communicator's
  // background progress engine. register_completion_notifier() only attaches
  // user callbacks to this always-on loop.
  std::atomic<bool> notifier_running_{false};
  std::atomic<bool> notifier_started_{false};
  std::thread notifier_thread_;
  std::condition_variable notifier_cv_;
  std::mutex notifier_mu_;

  struct NotifyTarget {
    std::function<void(unsigned, std::chrono::steady_clock::time_point)> emit;
  };
  std::vector<std::weak_ptr<NotifyTarget>> notify_targets_;
  mutable std::mutex remote_ipc_mu_;
  std::unordered_map<int, std::unordered_map<uint32_t, RemoteIpcBufferState>>
      remote_ipc_buffers_;
  mutable std::mutex uccl_reg_mu_;
  std::unordered_set<uint64_t> uccl_direct_reg_failed_mrs_;

  friend class IpcChannel;
};

}  // namespace Transport
}  // namespace UKernel
