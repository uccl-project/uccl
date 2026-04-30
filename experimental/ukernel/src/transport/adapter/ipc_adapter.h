#pragma once

#include "../memory/ipc_manager.h"
#include "../util/jring.h"
#include "../../include/gpu_rt.h"
#include "shmring_exchanger.h"
#include "transport_adapter.h"
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace UKernel {
namespace Transport {

class Communicator;

class IpcAdapter final : public TransportAdapter {
 public:
  IpcAdapter(Communicator* comm, std::string ring_namespace, int self_local_id,
             int local_gpu_idx);
  ~IpcAdapter() override;
  void shutdown();

  void set_peer_local_id(int peer_rank, int local_id);
  void close_peer(int peer_rank);

  uint64_t next_send_match_seq(int peer_rank);
  uint64_t next_recv_match_seq(int peer_rank);

  bool ensure_put_path(PeerConnectSpec const& spec) override;
  bool ensure_wait_path(PeerConnectSpec const& spec) override;
  bool has_put_path(int peer_rank) const override;
  bool has_wait_path(int peer_rank) const override;

  unsigned put_async(int peer_rank, void* local_ptr,
                     uint32_t local_buffer_id, void* remote_ptr,
                     uint32_t remote_buffer_id, size_t len) override;
  unsigned signal_async(int peer_rank, uint64_t tag) override;
  unsigned wait_async(int peer_rank, uint64_t expected_tag,
                      std::optional<WaitTarget> target = std::nullopt) override;

  bool poll_completion(unsigned id) override;
  bool wait_completion(unsigned id) override;
  bool request_failed(unsigned id) override;
  void release_request(unsigned id) override;
  uint64_t completion_payload(unsigned id) const override;

 private:
  bool connect_to(int rank);
  bool accept_from(int rank);

  enum class RequestState : uint8_t {
    Free = 0, Queued = 1, Running = 2, Completed = 3, Failed = 4,
  };
  enum class IpcReqType : uint8_t {
    DataPut = 0, DataWait = 1, Signal = 2, SignalWait = 3
  };

  struct IpcRequestSlot {
    std::atomic<RequestState> state{RequestState::Free};
    std::atomic<uint32_t> generation{1};
    unsigned id = 0;
    int peer_rank = -1;
    uint64_t match_seq = 0;
    IpcReqType req_type = IpcReqType::DataPut;
    void* local_ptr = nullptr;
    void* remote_ptr = nullptr;
    uint32_t remote_buffer_id = 0;
    size_t size_bytes = 0;
    uint64_t signal_payload = 0;
    std::atomic<uint32_t> remaining{0};
    std::atomic<bool> failed{false};
    std::atomic<bool> finished{false};

    void mark_queued(uint32_t completion_count = 1) {
      state.store(RequestState::Queued, std::memory_order_release);
      remaining.store(completion_count, std::memory_order_release);
      failed.store(false, std::memory_order_release);
      finished.store(false, std::memory_order_release);
    }
    void mark_failed() {
      state.store(RequestState::Failed, std::memory_order_release);
      failed.store(true, std::memory_order_release);
      finished.store(true, std::memory_order_release);
      remaining.store(0, std::memory_order_release);
    }
    void mark_running() {
      state.store(RequestState::Running, std::memory_order_release);
    }
    void complete_one() {
      uint32_t prev = remaining.load(std::memory_order_acquire);
      while (prev != 0 && !remaining.compare_exchange_weak(
                              prev, prev - 1, std::memory_order_acq_rel,
                              std::memory_order_acquire)) {}
      if (prev <= 1) {
        state.store(RequestState::Completed, std::memory_order_release);
        finished.store(true, std::memory_order_release);
      }
    }
    bool is_finished() const { return finished.load(std::memory_order_acquire); }
    bool has_failed() const { return failed.load(std::memory_order_acquire); }
    bool is_direct_gpu() const {
      return remote_ptr != nullptr &&
             remote_ptr == local_ptr;  // placeholder; determined by pointer attr
    }
  };

  static constexpr uint32_t kRequestSlotBits = 13;
  static constexpr uint32_t kRequestSlotCount = (1u << kRequestSlotBits);
  static constexpr uint32_t kRequestSlotMask = kRequestSlotCount - 1u;
  static unsigned make_request_id(uint32_t slot_idx, uint32_t generation) {
    return static_cast<unsigned>((generation << kRequestSlotBits) | slot_idx);
  }
  static uint32_t request_slot_index(unsigned request_id) {
    return static_cast<uint32_t>(request_id) & kRequestSlotMask;
  }
  static uint32_t request_generation(unsigned request_id) {
    return static_cast<uint32_t>(request_id) >> kRequestSlotBits;
  }

  IpcRequestSlot* try_acquire_request_slot(unsigned* out_request_id);
  IpcRequestSlot* resolve_request_slot(unsigned request_id);
  IpcRequestSlot* resolve_request_slot_const(unsigned request_id) const;
  void release_request_slot(unsigned request_id);

  bool enqueue_request(unsigned request_id, IpcReqType type);
  bool send_one(IpcRequestSlot* creq);
  bool recv_one(IpcRequestSlot* creq);
  void send_thread_func();
  void recv_thread_func();
  void complete_task(IpcRequestSlot* req, bool ok);

  jring_t* send_task_ring_;
  jring_t* recv_task_ring_;
  std::atomic<bool> stop_{false};
  std::atomic<bool> shutdown_started_{false};
  std::thread send_thread_;
  std::thread recv_thread_;
  std::mutex cv_mu_;
  std::condition_variable cv_;
  std::atomic<int> pending_send_{0};
  std::atomic<int> pending_recv_{0};
  std::vector<gpuStream_t> ipc_streams_;

  std::mutex match_seq_mu_;
  std::vector<std::array<uint64_t, 2>> next_match_seq_per_peer_;
  std::unique_ptr<IpcRequestSlot[]> request_slots_;
  std::atomic<uint32_t> request_alloc_cursor_{0};

  std::shared_ptr<ShmRingExchanger> shm_control_;
  mutable std::mutex peer_dir_mu_;
  struct DirState { bool put_ready = false; bool wait_ready = false; };
  std::vector<DirState> peer_dir_state_;
  Communicator* comm_;
  int local_gpu_idx_ = -1;
};

}  // namespace Transport
}  // namespace UKernel
