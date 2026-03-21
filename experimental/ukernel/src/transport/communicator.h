#pragma once

#include "../../include/config.h"
#include "ipc_cache.h"
#include "memory_registry.h"
#include "oob.h"
#include "peer_session.h"
#include "request.h"
#include "uccl_transport_adapter.h"
#include <cstddef>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace UKernel {
namespace Transport {

class IpcChannel;

class Communicator {
 public:
  Communicator(int gpu_id, int rank, int world_size,
               std::shared_ptr<CommunicatorConfig> config =
                   std::make_shared<CommunicatorConfig>());
  ~Communicator();

  bool connect_to(int rank);
  bool accept_from(int rank);
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

  std::shared_ptr<void> register_completion_notifier(
      std::function<void(unsigned, std::chrono::steady_clock::time_point)> cb);
  void completion_notifier_loop();

  MR reg_mr(void* local_buf, size_t len);
  bool dereg_mr(void* local_buf);
  bool notify_mr(int remote_rank, MR& mr);
  bool wait_mr_notify(int remote_rank, MR& mr);
  MR get_local_mr(void* local_buf);
  MR get_local_mr(uint32_t mr_id);
  MR get_remote_mr(int remote_rank, uint32_t mr_id);

  bool register_remote_ipc_cache(int remote_rank, gpuIpcMemHandle_t handle,
                                  IpcCacheManager::IpcCache const& cache);
  IpcCacheManager::IpcCache get_remote_ipc_cache(int remote_rank, gpuIpcMemHandle_t handle);

 private:
  struct TrackedRequest {
    int peer_rank = -1;
    PeerTransportKind kind = PeerTransportKind::Ipc;
    std::shared_ptr<Request> ipc_request;
    bool completed = false;
    bool failed = false;
    bool notified = false;
  };

  bool check_ready() const;
  UcclTransportAdapter& ensure_uccl_adapter(
      CommunicatorMeta const& local_meta, CommunicatorMeta const& peer_meta);
  std::shared_ptr<IpcChannel> get_ipc_channel_by_rank(int rank);
  bool has_peer_send_path(int rank) const;
  bool has_peer_recv_path(int rank) const;
  PeerTransportKind get_peer_transport_kind(int rank) const;
  bool poll_request_completion(unsigned id, bool blocking);
  void register_existing_local_mrs_with_uccl();
  void exchange_peer_metas();
  void cache_peer_session(int rank, PeerTransportKind kind, bool mark_send_ready,
                          bool mark_recv_ready);
  void shutdown_ipc_channel();

  int local_gpu_idx_;
  int global_rank_;
  int world_size_;
  MemoryRegistry memory_registry_;
  std::unique_ptr<UcclTransportAdapter> uccl_adapter_;
  std::shared_ptr<IpcChannel> ipc_channel_;

  std::shared_ptr<ShmRingExchanger> shm_control_;
  std::shared_ptr<PeerSessionManager> peer_manager_;

  std::shared_ptr<CommunicatorConfig> config_;
  std::shared_ptr<Exchanger> exchanger_client_;

  std::unordered_map<unsigned, TrackedRequest> requests_map_;
  mutable std::mutex req_mu_;
  std::atomic<unsigned> next_request_id_{1};

  std::atomic<bool> notifier_running_{false};
  std::atomic<bool> notifier_started_{false};
  std::thread notifier_thread_;
  std::condition_variable notifier_cv_;
  std::mutex notifier_mu_;

  struct NotifyTarget {
    std::function<void(unsigned, std::chrono::steady_clock::time_point)> emit;
  };
  std::vector<std::weak_ptr<NotifyTarget>> notify_targets_;

  friend class IpcChannel;
};

}  // namespace Transport
}  // namespace UKernel
