#pragma once

#include "../../include/gpu_rt.h"
#include "../memory/ipc_manager.h"
#include "../util/jring.h"
#include "transport_adapter.h"
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

struct IpcDataCompletion {
  std::atomic<uint64_t> last_completed[2];  // [0] = dir 0, [1] = dir 1
};

class IpcAdapter final : public TransportAdapter {
 public:
  IpcAdapter(Communicator* comm, std::string ring_namespace, int gpu_id);
  ~IpcAdapter() override;
  void shutdown();

  uint64_t next_send_match_seq(int peer);
  uint64_t next_recv_match_seq(int peer);

  bool ensure_put_path(PeerConnectSpec const&) override;
  bool ensure_wait_path(PeerConnectSpec const&) override;
  bool has_put_path(int peer) const override;
  bool has_wait_path(int peer) const override;

  unsigned put_async(int peer, void* local_ptr, uint32_t local_buf,
                     void* remote_ptr, uint32_t remote_buf, size_t len,
                     unsigned comm_rid) override;
  unsigned signal_async(int peer, uint64_t tag, unsigned comm_rid) override;
  unsigned wait_async(int peer, uint64_t tag, std::optional<WaitTarget>,
                      unsigned comm_rid) override;

 private:
  enum class ReqState : uint8_t { Free, Queued, Completed, Failed };
  enum class ReqType : uint8_t { DataPut, DataWait, Signal, SignalWait };

  struct Slot {
    std::atomic<ReqState> state{ReqState::Free};
    std::atomic<uint32_t> gen{1};
    unsigned id = 0;
    unsigned comm_rid = 0;
    int peer = -1;
    uint64_t seq = 0;
    ReqType type = ReqType::DataPut;
    void* local_ptr = nullptr;
    void* remote_ptr = nullptr;
    size_t bytes = 0;

    void mark_queued() { state.store(ReqState::Queued, std::memory_order_release); }
    void mark_completed(bool ok) {
      state.store(ok ? ReqState::Completed : ReqState::Failed,
                  std::memory_order_release);
    }
    bool is_done() const {
      auto s = state.load(std::memory_order_acquire);
      return s == ReqState::Completed || s == ReqState::Failed;
    }
  };

  struct PeerComp {
    IpcDataCompletion* local = nullptr;
    IpcDataCompletion* remote = nullptr;
    int shm_fd = -1;
    size_t shm_size = 0;
    std::string shm_name;
  };

  static constexpr uint32_t kSlotBits = 13;
  static constexpr uint32_t kSlotCnt = (1u << kSlotBits);
  static constexpr uint32_t kSlotMask = kSlotCnt - 1u;
  static unsigned make_rid(uint32_t idx, uint32_t gen) {
    return static_cast<unsigned>((gen << kSlotBits) | idx);
  }
  static uint32_t slot_idx(unsigned rid) { return rid & kSlotMask; }
  static uint32_t slot_gen(unsigned rid) { return rid >> kSlotBits; }

  Slot* acquire_slot(unsigned* out);
  Slot* resolve_slot(unsigned id);
  void release_slot(unsigned id);
  bool enqueue(unsigned id, ReqType type);

  void send_worker();
  void recv_worker();
  void done(Slot* s, bool ok);

  std::string comp_shm_name(int peer) const;
  bool ensure_local_comp(int peer);
  bool ensure_remote_comp(int peer);
  void close_comp(int peer);

  jring_t* send_ring_ = nullptr;
  jring_t* recv_ring_ = nullptr;
  std::atomic<bool> stop_{false};
  std::thread send_th_;
  std::thread recv_th_;
  std::vector<std::pair<gpuStream_t, gpuEvent_t>> ipc_ctx_;

  std::mutex seq_mu_;
  std::vector<std::array<uint64_t, 2>> seqs_;  // [peer][0]=send, [1]=recv
  std::unique_ptr<Slot[]> slots_;
  std::atomic<uint32_t> alloc_cursor_{0};

  std::string ns_;
  mutable std::mutex dir_mu_;
  std::vector<std::pair<bool, bool>> dir_state_;  // {put_ready, wait_ready}
  std::vector<PeerComp> comps_;
  Communicator* comm_;
  int gpu_id_ = -1;
};
}  // namespace Transport
}  // namespace UKernel
