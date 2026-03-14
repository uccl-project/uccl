#pragma once

#include "../../include/config.h"
#include "memory_registry.h"
#include "oob.h"
#include "request.h"
#include "transport_engine.h"
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
class UcclTransportEngine;

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
                                 IpcCache const& cache);
  IpcCache get_remote_ipc_cache(int remote_rank, gpuIpcMemHandle_t handle);

 private:
  struct PeerState {
    bool has_meta = false;
    CommunicatorMeta meta{};
    PeerTransportKind kind = PeerTransportKind::Ipc;
    std::shared_ptr<IpcChannel> ipc_channel;
    bool send_ready = false;
    bool recv_ready = false;
  };

  struct TrackedRequest {
    int peer_rank = -1;
    PeerTransportKind kind = PeerTransportKind::Ipc;
    std::shared_ptr<Request> ipc_request;
    bool completed = false;
    bool failed = false;
    bool notified = false;
  };

  void set_peer_meta(int rank, CommunicatorMeta const& meta);
  bool try_get_peer_meta(int rank, CommunicatorMeta& out) const;
  bool check_ready() const;
  std::shared_ptr<IpcChannel> get_ipc_channel_by_rank(int rank);
  void cache_peer_session(int rank, PeerTransportKind kind,
                          std::shared_ptr<IpcChannel> ipc_channel,
                          bool mark_send_ready, bool mark_recv_ready);
  bool has_peer_send_path(int rank) const;
  bool has_peer_recv_path(int rank) const;
  PeerTransportKind get_peer_transport_kind(int rank) const;
  bool poll_request_completion(unsigned id, bool blocking);
  void register_existing_local_mrs_with_uccl();

  std::vector<PeerState> peers_;
  mutable std::mutex peer_mu_;

  int local_gpu_idx_;
  int global_rank_;
  int world_size_;
  MemoryRegistry memory_registry_;
  std::unique_ptr<UcclTransportEngine> uccl_engine_;

  std::shared_ptr<UdsExchanger> uds_;

  std::shared_ptr<CommunicatorConfig> config_;
  std::shared_ptr<Exchanger> exchanger_client_;

  std::unordered_map<unsigned, TrackedRequest> requests_map_;
  std::mutex req_mu_;
  std::atomic<unsigned> next_request_id_{1};

  std::atomic<bool> notifier_running_{false};
  std::atomic<bool> notifier_started_{false};
  std::thread notifier_thread_;
  std::condition_variable notifier_cv_;
  std::mutex notifier_mu_;

  struct NotifyTarget {
    std::function<void(unsigned, std::chrono::steady_clock::time_point)> emit;
  };
  std::vector<std::shared_ptr<NotifyTarget>> notify_targets_;

  friend class IpcChannel;
};

}  // namespace Transport
}  // namespace UKernel
