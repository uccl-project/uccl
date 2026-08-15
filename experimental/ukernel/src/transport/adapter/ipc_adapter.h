#pragma once

#include "../memory/ipc_manager.h"
#include "gpu_rt.h"
#include "ipc_signal_ring.h"
#include "transport_adapter.h"
#include "util/jring.h"
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace UKernel {
namespace Transport {

class Communicator;

// Signal and DataPut share the same SHM region but use independent
// completion counters to prevent tag reuse or sequence interference
// across ping-pong iterations where user tags (e.g., 1, 2) alternate.
struct IpcDataCompletion {
  std::atomic<uint64_t> last_completed[2];  // [0] = dir 0, [1] = dir 1
};

class IpcAdapter final : public TransportAdapter {
 public:
  IpcAdapter(Communicator* comm, std::string ring_namespace, int gpu_id);
  ~IpcAdapter() override;
  void shutdown();

  // Signal worker loops to exit without joining or releasing resources.
  // Use when external threads may still be calling into the adapter.
  void stop();

  uint64_t next_send_match_seq(int peer);
  uint64_t next_recv_match_seq(int peer);

  bool ensure_put_path(PeerConnectSpec const&) override;
  bool ensure_wait_path(PeerConnectSpec const&) override;
  bool has_put_path(int peer) const override;
  bool has_wait_path(int peer) const override;

  unsigned send_put_async(int peer, void* local_ptr, uint32_t local_buf,
                          void* remote_ptr, uint32_t remote_buf, size_t len,
                          unsigned comm_rid) override;
  unsigned send_signal_async(int peer, uint64_t tag,
                             unsigned comm_rid) override;
  unsigned wait_signal_async(int peer, uint64_t tag, std::optional<WaitTarget>,
                             unsigned comm_rid) override;
  bool supports_put_signal() const override { return true; }
  unsigned send_put_signal_async(int peer, void* local_ptr, uint32_t local_buf,
                                 void* remote_ptr, uint32_t remote_buf,
                                 size_t len, uint64_t tag,
                                 unsigned comm_rid) override;

  // Drain signal tags from the peer's shared-memory signal ring.
  // Called directly by Communicator::drain_ipc_signals().
  size_t drain_signal_tags(int peer_rank, uint64_t* tags, size_t max);
  // O(1) arrival hint: true when the peer's ring has unconsumed tags.
  // Lets drain_ipc_signals skip peers with nothing to read.
  bool has_signal_arrivals(int peer_rank) const;

  // GPU-visible address of the peer's signal ring (zero-copy host
  // mapping registered when the peer path was opened), or nullptr when
  // unavailable. Device kernels write fused PutSignal tags through it.
  void* peer_signal_ring_device_ptr(int peer) const;
  // GPU-visible address of the peer's device-flag area (single-writer
  // per-slot completion flags, plain stores + fence — no atomics).
  void* peer_device_flag_ptr(int peer) const;
  // Host-side pointer to THIS rank's local device-flag area for a peer
  // (the peer's device tasks write into it; this rank's host polls it).
  uint64_t* local_device_flag_slots(int peer) const;

  void close_comp(int peer_rank);

 private:
  enum class ReqType : uint8_t { DataPut, DataWait, PutSignal, Signal };

  struct RingElem {
    unsigned comm_rid;
    int peer;
    ReqType type;
    uint64_t seq;
    void* local_ptr;
    void* remote_ptr;
    size_t bytes;
    uint64_t tag = 0;  // PutSignal: signal tag written after data lands
  };

  struct PeerComp {
    IpcDataCompletion* local = nullptr;
    IpcDataCompletion* remote = nullptr;
    PeerSignalRing* signal_ring = nullptr;  // in same SHM as local
    void* remote_device = nullptr;  // remote mapping, GPU-visible (zero-copy)
    bool remote_registered = false;  // gpuHostRegister'd remote SHM
    int shm_fd = -1;
    size_t shm_size = 0;
    std::string shm_name;
  };

  void send_worker();
  void recv_worker();
  bool launch_one(RingElem* e, size_t stream_idx);
  void complete_one(RingElem const* e, bool ok);
  // Non-blocking DataWait completion check (recv_worker polls all
  // outstanding waits each iteration instead of blocking on one).
  bool recv_one_poll(RingElem const* e);
  // Multi-producer-safe signal ring claim: fetch_add on the shared
  // write_idx (the device kernel's fused PutSignal path claims the same
  // counter), then check the claimed slot. Returns true when the slot
  // can be written now (previous lap consumed); on false the claim is
  // MADE and the caller must complete it later via write_signal_slot
  // (deferred) — the slot index is returned through `out_slot`.
  bool claim_signal_slot(int peer, size_t* out_slot);
  void write_signal_slot(int peer, size_t idx, uint64_t tag);

  struct DeferredSignal {
    uint64_t tag;
    unsigned comm_rid;
    size_t slot;  // claimed ring slot; write when its previous lap clears
    // Plain Signal ops own a sig_send completion; a fused PutSignal's
    // rid already completed on the put ring, so its deferred write is a
    // side effect only (dropped at shutdown, never completed again).
    bool publish_sig;
  };
  // Per-peer deferred signal ring writes (back-pressure on the
  // receiver's drain cadence). Written by the send worker only.
  //
  // Correctness bound: the shared signal ring is 4096 slots, and a
  // deferred write keeps its slot CLAIMED until written. The ring cannot
  // wrap onto a deferred slot while the receiver keeps draining each
  // synchronous collective round (the depth of un-drained slots stays
  // << 4096). If a future non-round-synchronous path lets the receiver
  // fall >4096 slots behind while the device kernel also claims slots,
  // a wrap could double-claim a slot and lose a signal — that path must
  // re-examine this bound before shipping.
  std::vector<std::deque<DeferredSignal>> deferred_sigs_;

  bool connect_to(int rank);
  bool accept_from(int rank);

  std::string comp_shm_name(int peer) const;
  bool ensure_local_comp(int peer);
  bool ensure_remote_comp(int peer);

  // One send ring per peer: a peer's puts never block another peer's
  // launch-ahead, so alltoall-style fan-out stays fully concurrent.
  std::vector<jring_t*> send_rings_;
  jring_t* recv_ring_ = nullptr;
  std::atomic<bool> stop_{false};
  std::thread send_th_;
  std::thread recv_th_;
  std::vector<gpuStream_t> ipc_ctx_;
  size_t send_batch_ = 4;         // in-flight puts PER PEER
  size_t streams_per_peer_ = 4;   // per-peer stream pool (round-robin)

  // Per-peer per-direction sequence counters for DataWait matching.
  // next_send_match_seq (send worker) and next_recv_match_seq (recv
  // worker) can touch the same counter, so they are atomics.
  std::vector<std::array<std::atomic<uint64_t>, 2>> seqs_;

  std::string ns_;
  mutable std::mutex dir_mu_;
  std::vector<std::pair<bool, bool>> dir_state_;  // {put_ready, wait_ready}
  std::vector<PeerComp> comps_;
  Communicator* comm_;
  int gpu_id_ = -1;
};
}  // namespace Transport
}  // namespace UKernel
