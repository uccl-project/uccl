#pragma once

#include "gpu_rt.h"
#include "oob/oob.h"
#include "util/jring.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace UKernel {
namespace Transport {

class UcclTransportAdapter;
class TcpTransportAdapter;
class IpcAdapter;

struct TrackedRequest {
  enum class SlotState : uint8_t {
    Free = 0,
    Reserved = 1,
    InFlight = 2,
    Completed = 3,
    Failed = 4,
    Releasing = 5
  };
  std::atomic<SlotState> state{SlotState::Free};
  std::atomic<uint32_t> generation{0};
  unsigned request_id = 0;
  unsigned adapter_request_id = 0;
  int peer_rank = -1;
  PeerTransportKind kind = PeerTransportKind::Unknown;
  bool notified = false;
  bool needs_host_to_device_copy = false;
  bool host_copy_submitted = false;
  void* completion_buffer = nullptr;
  size_t completion_offset = 0;
  size_t completion_bytes = 0;
  gpuEvent_t host_copy_event = nullptr;
  void* bounce_ptr = nullptr;
};

class RequestTracker {
 public:
  using CompleteBounceFn =
      std::function<bool(TrackedRequest& tracked, bool blocking)>;
  using CleanupFn = std::function<void(TrackedRequest& tracked)>;
  using NotifyCb =
      std::function<void(unsigned req_id,
                         std::chrono::steady_clock::time_point ts)>;

  RequestTracker(UcclTransportAdapter* uccl, TcpTransportAdapter* tcp,
                 IpcAdapter* ipc, CompleteBounceFn complete_bounce,
                 CleanupFn cleanup);
  ~RequestTracker();

  RequestTracker(RequestTracker const&) = delete;
  RequestTracker& operator=(RequestTracker const&) = delete;

  // Slot lifecycle.
  TrackedRequest* allocate(unsigned* out_req_id);
  TrackedRequest* resolve(unsigned req_id) const;

  // Activate after adapter submit.
  bool activate(unsigned req_id, unsigned adapter_req_id, int peer_rank,
                PeerTransportKind kind);

  // Try to release a Completed/Failed request. Returns a snapshot for the
  // caller to run resource cleanup on.
  bool try_release(unsigned req_id, TrackedRequest* snapshot);

  // User-facing completion API.
  bool poll(unsigned req);
  void release(unsigned req);
  bool wait_finish(std::vector<unsigned> const& reqs);
  bool wait_finish(unsigned req);

  // Progress thread lifecycle.
  void start_progress();

  // Update adapter pointers (uccl/tcp are created lazily by Communicator).
  void set_uccl(UcclTransportAdapter* uccl) { uccl_ = uccl; }
  void set_tcp(TcpTransportAdapter* tcp) { tcp_ = tcp; }

  // Completion notification hook.
  std::shared_ptr<void> register_notifier(NotifyCb cb);

  uint32_t inflight_count() const {
    return inflight_count_.load(std::memory_order_acquire);
  }
  bool progress_started() const {
    return started_.load(std::memory_order_acquire);
  }

 private:
  void progress_loop();
  bool poll_one(unsigned id, bool blocking);
  void notify_completion();
  void finish_release(unsigned req_id);
  static void signal_eventfd(int fd);

  static unsigned make_request_id(uint32_t slot_idx, uint32_t generation);
  static uint32_t request_slot_index(unsigned req_id);
  static uint32_t request_generation(unsigned req_id);

  // Adapter references (not owned).
  UcclTransportAdapter* uccl_;
  TcpTransportAdapter* tcp_;
  IpcAdapter* ipc_;

  // Callbacks from Communicator.
  CompleteBounceFn complete_bounce_;
  CleanupFn cleanup_;

  // jring-based active ring (replaces hand-rolled active_ring_[] + head/tail).
  jring_t* active_jring_;

  // Request slots.
  static constexpr uint32_t kSlotBits = 13;
  static constexpr uint32_t kSlotCount = (1u << kSlotBits);
  static constexpr uint32_t kGenerationMask = (1u << (32u - kSlotBits)) - 1u;
  std::unique_ptr<TrackedRequest[]> slots_;
  std::atomic<uint32_t> alloc_cursor_{0};

  // Completion notification plumbing.
  int event_fd_ = -1;
  std::atomic<uint64_t> completion_seq_{0};
  std::atomic<uint32_t> inflight_count_{0};

  // Progress thread.
  std::thread progress_thread_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::atomic<bool> running_{false};
  std::atomic<bool> started_{false};

  // Registered notifier targets.
  struct NotifyTarget {
    std::function<void(unsigned, std::chrono::steady_clock::time_point)> emit;
  };
  std::vector<std::weak_ptr<NotifyTarget>> notifiers_;
};

}  // namespace Transport
}  // namespace UKernel
