#pragma once

#include "../../include/config.h"
#include "adapter/transport_adapter.h"
#include "memory/ipc_manager.h"
#include "memory/mr_manager.h"
#include "oob/oob.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace UKernel {
namespace Transport {

class TransportAdapter;
class IpcAdapter;
class TcpTransportAdapter;
class RdmaTransportAdapter;

struct CompletionResult {
  unsigned rid;
  bool failed;
  uint32_t user_ctx;
};

struct SignalCompletion {
  unsigned rid;
  uint64_t tag;
  int peer;
  bool failed;
  uint32_t user_ctx;
};

class Communicator {
 public:
  Communicator(
      int gpu_id, int rank, int world_size,
      std::shared_ptr<CommunicatorConfig> config =
          std::make_shared<CommunicatorConfig>(CommunicatorConfig::from_env()));
  ~Communicator();

  void stop_transports();

  int rank() const { return global_rank_; }
  int world_size() const { return world_size_; }

  // Debug: print signal-matching state per peer (buffered arrivals and
  // posted-but-unmatched waits). Used by the executor's SIGUSR2 dump.
  void dump_signal_state() const;

  bool connect(int rank,
               PeerTransportKind transport = PeerTransportKind::Unknown);
  bool accept(int rank,
              PeerTransportKind transport = PeerTransportKind::Unknown);
  PeerTransportKind peer_transport_kind(int rank) const;
  bool same_host(int rank) const;

  // Async data / signal / wait (thin wrappers over adapter)
  //
  // One-sided transports (IPC, RDMA):
  //   send_put_async() writes directly into remote memory. No matching wait
  //   needed. send_signal_async(peer, tag) notifies the peer; peer calls
  //   wait_signal_async(peer, tag).
  //
  // Two-sided transport (TCP):
  //   Every send_put_async() MUST have a matching wait_signal_async() on the
  //   peer. TCP has no remote address space — data flows as a byte stream.
  unsigned send_put_async(
      int peer, uint32_t src_buf, size_t src_off, uint32_t dst_buf,
      size_t dst_off, size_t bytes,
      PeerTransportKind transport = PeerTransportKind::Unknown);
  unsigned send_signal_async(
      int peer, uint64_t tag,
      PeerTransportKind transport = PeerTransportKind::Unknown);

  // Async signal wait (tag-based matching via Communicator table)
  // wait_signal_async(peer, tag): non-blocking, returns rid immediately.
  // Matching is done in on_signal_received() (called by drain_ipc_signals
  // for IPC and by RdmaTransportAdapter for 64-bit signal-QP arrivals) —
  // except RDMA write-with-imm arrivals, which carry only the tag's low
  // 32 bits and match per-peer FIFO in on_imm_received().
  // Completions are dequeued via try_complete_sig_wait().
  unsigned wait_signal_async(
      int peer, uint64_t tag,
      PeerTransportKind transport = PeerTransportKind::Unknown);
  // Data wait (TCP DataWait): still blocking on recv completion.
  unsigned wait_signal_async(
      int peer, uint64_t tag, uint32_t recv_buf, size_t off, size_t len,
      PeerTransportKind transport = PeerTransportKind::Unknown);

  // Variants accepting pre-allocated rid.
  bool send_put_async_with_rid(int peer, uint32_t src_buf, size_t src_off,
                               uint32_t dst_buf, size_t dst_off, size_t bytes,
                               PeerTransportKind transport, unsigned rid,
                               uint32_t qp_affinity = ~0u);
  bool send_signal_async_with_rid(int peer, uint64_t tag,
                                  PeerTransportKind transport, unsigned rid);
  bool wait_signal_async_with_rid(int peer, uint64_t tag,
                                  PeerTransportKind transport, unsigned rid,
                                  uint32_t count = 1, bool force_imm = false);
  // Device-completion flag wait: polls the local flag slot (written by
  // the peer's device task with a plain store + fence, no atomics) until
  // it equals `tag`. count > 1 waits for a G-tile group: `count`
  // consecutive slots at slot..slot+count-1 must equal tag..tag+count-1.
  bool wait_flag_async_with_rid(int peer, uint32_t slot, uint64_t tag,
                                unsigned rid, uint32_t count = 1);

  // Fused put+signal: once the data lands, the peer observes `tag` as a
  // signal (IPC: peer shm ring; RDMA: write-with-imm). One completion
  // for rid. Returns false when the effective transport cannot fuse —
  // callers then fall back to a separate put + signal.
  // qp_affinity (RDMA only, ~0u = auto): pins the op to
  // (qp_affinity % num_qps); puts of one signal group must share a QP.
  bool send_put_signal_async_with_rid(int peer, uint32_t src_buf,
                                      size_t src_off, uint32_t dst_buf,
                                      size_t dst_off, size_t bytes,
                                      PeerTransportKind transport, uint64_t tag,
                                      unsigned rid, uint32_t qp_affinity = ~0u);
  // Whether the effective transport to `peer` supports fused PutSignal.
  bool can_fuse_put_signal(int peer, PeerTransportKind transport);
  // GPU-visible address of the peer's IPC signal ring (zero-copy host
  // mapping), or nullptr when unavailable. Device kernels write fused
  // PutSignal tags through it.
  void* ipc_signal_ring_device_ptr(int peer) const;
  void* ipc_device_flag_ptr(int peer) const;

  // rid encoding: backend-path rids carry a 2-bit tag in the top bits
  // (bit 30 = SignalBackend, bit 31 = TransportBackend); the low 30 bits are
  // the backend's be_idx, so completion paths decode user_ctx directly
  // without touching rid_to_user_ctx_. Legacy rids from alloc_rid() stay in
  // [1, 2^30) and keep using the map.
  static constexpr unsigned kRidTagSignal = 1u << 30;
  static constexpr unsigned kRidTagTransport = 1u << 31;
  static constexpr unsigned kRidTagMask = 3u << 30;
  static constexpr unsigned kRidBeIdxMask = (1u << 30) - 1;

  unsigned alloc_rid() {
    // Legacy rids must be nonzero (0 means failure) and clear of the
    // backend tag bits.
    unsigned r =
        next_rid_.fetch_add(1, std::memory_order_relaxed) & kRidBeIdxMask;
    if (r == 0)
      r = next_rid_.fetch_add(1, std::memory_order_relaxed) & kRidBeIdxMask;
    return r;
  }
  void record_user_ctx(unsigned rid, uint32_t user_ctx);
  uint32_t consume_user_ctx(unsigned rid);

  // C++ advanced API, not exposed to Python binding.
  size_t try_complete_put(CompletionResult* results, size_t max);
  size_t try_complete_sig_wait(SignalCompletion* events, size_t max);
  size_t try_complete_sig_send(CompletionResult* results, size_t max);

  // Returns number of completed rids from the input array.
  // Writes completed rids back into the first N positions of the array.
  // For each rid: checks both put_completion_ring_ and
  // sig_wait_completion_ring_.
  size_t poll(unsigned* rids, size_t count);

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

  // Convenience: register local buffer (MR + IPC)
  bool register_buffer(uint32_t buffer_id, void* ptr, size_t len);

  // Convenience: resolve remote buffer (wait MR + wait IPC)
  bool resolve_remote_buffer(int peer_rank, uint32_t buffer_id,
                             int timeout_ms = 30000);

  int peer_gpu_idx(int rank) const;

  void re_register_all_mrs() { register_existing_local_mrs_with_rdma(); }

  // True when any signal wait is registered (used by the executor's
  // drain_signal_loop to busy-poll instead of yielding — the signal
  // arrival cadence is on the collective critical path at 8 ranks).
  bool has_pending_signal_waits() const;

 private:
  struct ResolvedPeer {
    CommunicatorMeta local_meta;
    CommunicatorMeta remote_meta;
    PeerTransportKind kind = PeerTransportKind::Unknown;
  };

  struct PeerPathState {
    bool put_ready = false;
    bool wait_ready = false;
  };

  struct PeerState {
    bool has_meta = false;
    CommunicatorMeta meta{};
    int gpu_idx = -1;
    // Auto-resolved default kind for this peer (set once, never changes)
    PeerTransportKind resolved_kind = PeerTransportKind::Unknown;
    // Per-transport readiness (key = PeerTransportKind)
    std::unordered_map<PeerTransportKind, PeerPathState> paths;
  };

  RdmaTransportAdapter& ensure_rdma_adapter(CommunicatorMeta const& local_meta);
  bool exchange_rdma_peer_info(int rank, RdmaTransportAdapter& rdma_adapter,
                               RdmaP2PInfo* out_remote_p2p_info);
  TcpTransportAdapter& ensure_tcp_adapter(CommunicatorMeta const& local_meta);

  bool has_put_path(
      int rank, PeerTransportKind transport = PeerTransportKind::Unknown) const;
  bool has_wait_path(
      int rank, PeerTransportKind transport = PeerTransportKind::Unknown) const;
  void mark_put_path_ready(int rank, PeerTransportKind kind);
  void mark_wait_path_ready(int rank, PeerTransportKind kind);
  bool ensure_path(int rank, bool is_put,
                   PeerTransportKind transport = PeerTransportKind::Unknown);
  void exchange_peer_metas();
  ResolvedPeer resolve_peer(
      int rank, PeerTransportKind transport = PeerTransportKind::Unknown) const;
  bool try_fallback_tcp_accept(int rank, CommunicatorMeta const& local_meta);

  TransportAdapter* get_adapter(PeerTransportKind kind);

  friend class RdmaTransportAdapter;
  void on_signal_received(int peer_rank, uint64_t tag);
  // RDMA write-with-imm arrival (fused PutSignal): the immediate carries
  // the tag's low 32 bits only.
  void on_imm_received(int peer_rank, uint32_t low32);
  void drain_ipc_signals();

  void register_existing_local_mrs_with_rdma();
  bool ensure_rdma_memory_registered(uint32_t buffer_id, void* ptr, size_t len);

  std::string ipc_open_error_message(int owner_rank, uint32_t buffer_id,
                                     IPCItem const& item, gpuError_t err) const;

  PeerTransportKind get_put_transport_kind(
      int rank, PeerTransportKind transport = PeerTransportKind::Unknown) const;
  PeerTransportKind get_wait_transport_kind(
      int rank, PeerTransportKind transport = PeerTransportKind::Unknown) const;

  int local_gpu_idx_;
  int global_rank_;
  int world_size_;

  MRManager mr_manager_;
  IPCManager ipc_manager_;

  std::unique_ptr<TcpTransportAdapter> tcp_adapter_;
  std::unique_ptr<RdmaTransportAdapter> rdma_adapter_;
  std::shared_ptr<IpcAdapter> ipc_adapter_;
  jring_t* put_completion_ring_ = nullptr;
  jring_t* sig_wait_completion_ring_ = nullptr;
  jring_t* sig_send_completion_ring_ = nullptr;

  // Overflow buffer for sig_wait completions when the ring is full.
  std::mutex sig_wait_overflow_mu_;
  std::vector<SignalCompletion> sig_wait_overflow_;
  std::atomic<uint32_t> next_rid_{1};

  // Signal matching: peer → tag → waiters. A waiter carries a remaining
  // arrival count: fused signal groups deliver one tag per tile, so the
  // wait completes only after `remaining` arrivals.
  std::unordered_map<
      int,
      std::unordered_map<uint64_t, std::vector<std::pair<unsigned, uint32_t>>>>
      pending_signal_waits_;
  // Buffered signals that arrived before the matching wait was registered.
  // Peer → deque of tag values. Checked first in wait_signal_async.
  std::unordered_map<int, std::deque<uint64_t>> pending_signals_;
  // RDMA write-with-imm matching (fused PutSignal). Immediates carry only
  // the tag's low 32 bits, which collide across runs (the epoch lives in
  // the high bits), so matching is per-peer FIFO in arrival order. This is
  // sound for the same reason the tag map is: ranks issue fused puts in
  // symmetric order, all fused puts to one peer are pinned to a single QP
  // (so arrival order == issue order), and a ±1 run skew still matches the
  // OLDEST pending wait first — the conservative direction.
  struct ImmWait {
    unsigned rid;
    uint32_t remaining;
    uint64_t tag;  // unsalted tag reported in the completion event
    uint32_t low32;
  };
  std::unordered_map<int, std::deque<ImmWait>> pending_imm_waits_;
  // Immediates that arrived before their wait was registered.
  std::unordered_map<int, std::deque<uint32_t>> buffered_imms_;
  // TCP signal completions go through the data ring; this maps rid → {peer,
  // tag}.
  std::unordered_map<unsigned, std::pair<int, uint64_t>> tcp_signal_rids_;
  // Device-flag waits: {rid, slot pointer, expected tag}. Polled by
  // try_complete_sig_wait (single writer per slot, single consumer).
  struct FlagWait {
    unsigned rid;
    int peer;
    uint64_t* base;
    uint64_t expected;
    uint32_t count;
    uint32_t matched;
  };
  std::vector<FlagWait> pending_flag_waits_;
  mutable std::mutex signal_waits_mu_;

  std::unordered_map<unsigned, uint32_t> rid_to_user_ctx_;
  mutable std::mutex user_ctx_mu_;

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
