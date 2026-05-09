#pragma once

#include "gpu_rt.h"
#include "oob/oob.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
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

  // Caller-driven progress: polls all inflight slots (one non-blocking pass).
  // Returns completed (request_id, failed) pairs. Does NOT release slots.
  std::vector<std::pair<unsigned, bool>> progress_all();

  // Update adapter pointers (uccl/tcp are created lazily by Communicator).
  void set_uccl(UcclTransportAdapter* uccl) { uccl_ = uccl; }
  void set_tcp(TcpTransportAdapter* tcp) { tcp_ = tcp; }

 private:
  bool poll_one(unsigned id, bool blocking);
  void finish_release(unsigned req_id);

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

  // Request slots.
  static constexpr uint32_t kSlotBits = 13;
  static constexpr uint32_t kSlotCount = (1u << kSlotBits);
  static constexpr uint32_t kGenerationMask = (1u << (32u - kSlotBits)) - 1u;
  std::unique_ptr<TrackedRequest[]> slots_;
  std::atomic<uint32_t> alloc_cursor_{0};
};

}  // namespace Transport
}  // namespace UKernel
