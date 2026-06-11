#pragma once

#include "../../include/config.h"
#include "memory/ipc_manager.h"
#include "memory/mr_manager.h"
#include "oob/oob.h"
#include "request_tracker.h"
#include "util/jring.h"
#include <atomic>
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
#include <utility>
#include <vector>

namespace UKernel {
namespace Transport {

class TransportAdapter;
class IpcAdapter;
class TcpTransportAdapter;
class UcclTransportAdapter;
class RdmaTransportAdapter;

class Communicator {
 public:
  Communicator(
      int gpu_id, int rank, int world_size,
      std::shared_ptr<CommunicatorConfig> config =
          std::make_shared<CommunicatorConfig>(CommunicatorConfig::from_env()));
  ~Communicator();

  int rank() const { return global_rank_; }
  int world_size() const { return world_size_; }

  bool connect(int rank);
  bool accept(int rank);
  PeerTransportKind peer_transport_kind(int rank) const;
  bool same_host(int rank) const;

  unsigned isend(int rank, uint32_t src_buf_id, size_t src_off,
                 size_t src_bytes, uint32_t dst_buf_id = 0, size_t dst_off = 0);
  unsigned irecv(int rank, uint32_t dst_buf_id, size_t dst_off,
                 size_t dst_bytes);
  bool poll(unsigned const req) { return tracker_->poll(req); }
  void release(unsigned const req) { return tracker_->release(req); }
  bool wait_finish(unsigned const req) { return tracker_->wait_finish(req); }
  bool wait_finish(std::vector<unsigned> const& reqs) {
    return tracker_->wait_finish(reqs);
  }

  // Caller-driven progress: polls all inflight requests (one non-blocking
  // pass). Returns completed (request_id, failed) pairs.
  std::vector<std::pair<unsigned, bool>> progress() {
    return tracker_->progress_all();
  }

  void set_oob_namespace(std::string ns);
  std::string oob_namespace() const;
  bool barrier(std::string const& barrier_namespace = "default",
               int timeout_ms = -1);

  bool reg_mr(uint32_t buffer_id, void* local_buf, size_t len,
              bool publish = true);
  bool dereg_mr(uint32_t buffer_id);
  bool wait_mr(int owner_rank, uint32_t buffer_id, int timeout_ms = -1);
  MR get_mr(uint32_t buffer_id) const;
  MR get_mr(int owner_rank, uint32_t buffer_id) const;

  bool reg_ipc(uint32_t buffer_id, void* local_buf, size_t len,
               bool publish = true);
  bool dereg_ipc(uint32_t buffer_id);
  bool wait_ipc(int owner_rank, uint32_t buffer_id, int timeout_ms = -1);
  IPCItem get_ipc(uint32_t buffer_id);
  IPCItem get_ipc(int owner_rank, uint32_t buffer_id);
  bool try_resolve_remote_ipc_pointer(int remote_rank,
                                      uint32_t remote_buffer_id, size_t offset,
                                      size_t bytes, void** out_ptr,
                                      int* out_device_idx);

  int peer_gpu_idx(int rank) const;

 private:
  struct ResolvedPeer {
    CommunicatorMeta local_meta;
    CommunicatorMeta remote_meta;
    PeerTransportKind kind = PeerTransportKind::Unknown;
  };

  struct PeerState {
    bool has_meta = false;
    CommunicatorMeta meta{};
    PeerTransportKind put_kind = PeerTransportKind::Unknown;
    PeerTransportKind wait_kind = PeerTransportKind::Unknown;
    bool put_ready = false;
    bool wait_ready = false;
    int gpu_idx = -1;
  };

  UcclTransportAdapter& ensure_uccl_adapter(CommunicatorMeta const& local_meta);
  RdmaTransportAdapter& ensure_rdma_adapter(CommunicatorMeta const& local_meta);
  bool exchange_uccl_peer_info(int rank, UcclTransportAdapter& uccl_adapter,
                               UCCLP2PInfo* out_remote_p2p_info);
  bool exchange_rdma_peer_info(int rank, RdmaTransportAdapter& rdma_adapter,
                               RdmaP2PInfo* out_remote_p2p_info);
  TcpTransportAdapter& ensure_tcp_adapter(CommunicatorMeta const& local_meta);

  bool has_put_path(int rank) const;
  bool has_wait_path(int rank) const;
  void mark_put_path_ready(int rank, PeerTransportKind kind);
  void mark_wait_path_ready(int rank, PeerTransportKind kind);
  bool ensure_path(int rank, bool is_put);
  void exchange_peer_metas();
  ResolvedPeer resolve_peer(int rank) const;
  bool try_fallback_tcp_accept(int rank, CommunicatorMeta const& local_meta);

  TransportAdapter* get_adapter(PeerTransportKind kind);

  bool complete_host_bounce_recv(TrackedRequest& tracked, bool blocking);
  void cleanup_tracked_request(TrackedRequest& tracked);

  void register_existing_local_mrs_with_uccl();
  void register_existing_local_mrs_with_rdma();
  bool ensure_uccl_memory_registered(uint32_t buffer_id, void* ptr, size_t len);
  bool ensure_rdma_memory_registered(uint32_t buffer_id, void* ptr, size_t len);

  gpuEvent_t acquire_event();
  void release_event(gpuEvent_t event);
  std::string ipc_open_error_message(int owner_rank, uint32_t buffer_id,
                                     IPCItem const& item, gpuError_t err) const;

  PeerTransportKind get_put_transport_kind(int rank) const;
  PeerTransportKind get_wait_transport_kind(int rank) const;

  // Async task submission via jring + proxy threads (mirrors p2p Endpoint).
  struct CommSendTask {
    int rank;
    uint32_t src_buf_id;
    size_t src_off;
    size_t src_bytes;
    uint32_t dst_buf_id;
    size_t dst_off;
    unsigned request_id;
  };

  struct CommRecvTask {
    int rank;
    uint32_t dst_buf_id;
    size_t dst_off;
    size_t dst_bytes;
    unsigned request_id;
  };

  void send_proxy_loop();
  void recv_proxy_loop();
  void drain_send_tasks();
  void drain_recv_tasks();
  void mark_slot_failed(unsigned request_id);
  void run_isend_body(CommSendTask const& task);
  void run_irecv_body(CommRecvTask const& task);

  static constexpr size_t kTaskRingSize = 1024;
  jring_t* send_task_ring_ = nullptr;
  jring_t* recv_task_ring_ = nullptr;
  std::atomic<bool> stop_{false};
  std::thread send_proxy_thread_;
  std::thread recv_proxy_thread_;

  int local_gpu_idx_;
  int global_rank_;
  int world_size_;

  MRManager mr_manager_;
  IPCManager ipc_manager_;

  std::unique_ptr<UcclTransportAdapter> uccl_adapter_;
  std::unique_ptr<TcpTransportAdapter> tcp_adapter_;
  std::unique_ptr<RdmaTransportAdapter> rdma_adapter_;
  std::shared_ptr<IpcAdapter> ipc_adapter_;
  gpuStream_t host_copy_stream_ = nullptr;

  std::unique_ptr<RequestTracker> tracker_;

  mutable std::mutex event_pool_mu_;
  std::vector<gpuEvent_t> event_pool_;

  mutable std::mutex peer_mu_;
  std::vector<PeerState> peer_states_;
  std::shared_ptr<CommunicatorConfig> config_;
  mutable std::mutex config_mu_;
  std::shared_ptr<Exchanger> exchanger_client_;
  std::atomic<uint64_t> barrier_seq_{0};

  mutable std::mutex resource_mu_;
  std::unordered_map<uint32_t, MR> local_buffer_to_mr_;
  std::unordered_map<int, std::unordered_map<uint32_t, MR>>
      remote_buffer_to_mr_;
  std::unordered_map<uint32_t, IPCItem> local_buffer_to_ipc_;

  mutable std::mutex uccl_reg_mu_;
  std::unordered_set<uint64_t> uccl_direct_reg_failed_mrs_;
  std::unordered_set<uint64_t> uccl_registered_mrs_;
  mutable std::mutex rdma_reg_mu_;
  std::unordered_set<uint64_t> rdma_direct_reg_failed_mrs_;
  std::unordered_set<uint64_t> rdma_registered_mrs_;
  std::atomic<uint32_t> next_ephemeral_buffer_id_{0x80000000u};
  std::atomic<uint64_t> mr_generation_{1};
  std::atomic<uint64_t> ipc_generation_{1};
  mutable std::mutex mr_gen_mu_;
  std::unordered_map<uint64_t, uint64_t> last_mr_generation_;
  std::unordered_map<uint64_t, uint64_t> last_ipc_generation_;
};

}  // namespace Transport
}  // namespace UKernel
