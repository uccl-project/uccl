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

  // GPU-visible address of the peer's signal ring (zero-copy host
  // mapping registered when the peer path was opened), or nullptr when
  // unavailable. Device kernels write fused PutSignal tags through it.
  void* peer_signal_ring_device_ptr(int peer) const;

  void close_comp(int peer_rank);

 private:
  enum class ReqType : uint8_t { DataPut, DataWait, PutSignal };

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
    int shm_fd = -1;
    size_t shm_size = 0;
    std::string shm_name;
  };

  void send_worker();
  void recv_worker();
  bool launch_one(RingElem* e, size_t stream_idx);
  void complete_one(RingElem const* e, bool ok);
  bool recv_one(RingElem* e);
  // Write a signal tag into the peer's shm ring. Multi-producer safe:
  // both the executor's enqueue thread (plain signals) and the send
  // worker (fused PutSignal) publish through this.
  bool write_signal_ring(int peer, uint64_t tag);

  bool connect_to(int rank);
  bool accept_from(int rank);

  std::string comp_shm_name(int peer) const;
  bool ensure_local_comp(int peer);
  bool ensure_remote_comp(int peer);

  jring_t* send_ring_ = nullptr;
  jring_t* recv_ring_ = nullptr;
  std::atomic<bool> stop_{false};
  std::thread send_th_;
  std::thread recv_th_;
  std::vector<gpuStream_t> ipc_ctx_;
  size_t send_batch_ = 16;

  std::mutex seq_mu_;
  std::vector<std::array<uint64_t, 2>> seqs_;  // [peer][0]=send, [1]=recv

  std::string ns_;
  mutable std::mutex dir_mu_;
  std::vector<std::pair<bool, bool>> dir_state_;  // {put_ready, wait_ready}
  std::vector<PeerComp> comps_;
  Communicator* comm_;
  int gpu_id_ = -1;
};
}  // namespace Transport
}  // namespace UKernel
