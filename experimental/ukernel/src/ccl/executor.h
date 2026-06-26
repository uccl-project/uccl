#pragma once

#include "backend/backend.h"
#include "coll_config.h"
#include "lower.h"
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace UKernel {
namespace Transport {
enum class PeerTransportKind;
struct CommunicatorConfig;
struct SignalCompletion;
class Communicator;
}  // namespace Transport
namespace CCL {

enum class CollectiveOpStatus : uint32_t {
  Queued,
  Running,
  Completed,
  Failed,
};

using CollectiveOpHandle = uint64_t;
inline constexpr CollectiveOpHandle kInvalidHandle = 0;

// ── Op sprayer with async jring backends ────────────────────────────────

struct SprayRun {
  // ── Hot path: accessed every enqueue/drain cycle ──
  std::atomic<CollectiveOpStatus> status{CollectiveOpStatus::Queued};
  std::atomic<size_t> done_count{0};
  uint32_t next_layer = 0;
  std::mutex mtx;
  std::vector<uint8_t> done;
  std::vector<uint8_t> submitted;
  std::vector<uint32_t> ready;
  std::vector<CmdWithId> dev_cmds;
  std::vector<CmdWithId> tpt_cmds;

  // ── Countdown-latch dependency tracking ──
  std::vector<uint32_t> indegree;                    // remaining unsatisfied deps
  std::vector<std::vector<uint32_t>> successors;     // reverse dep map: op → ops that depend on it

  // ── Read-only after construction ──
  TiledResult tiled;

  // ── Buffer IDs (dedup: same ptr+size = same ID) ──
  uint32_t input_buf_id = 0;
  uint32_t output_buf_id = 0;
  uint32_t scratch_buf_id = 0;

  // ── Cold: rarely accessed ──
  std::string error;
};

struct SprayExecutorConfig {
  int gpu_id;
  int rank;
  int world_size;
  size_t device_task_capacity = 256;
  size_t max_device_fifos = 2;
  int threads_per_block = 64;
  size_t fifo_capacity = 256;
  size_t smem_size = 48 * 1024;
  std::shared_ptr<struct UKernel::Transport::CommunicatorConfig>
      communicator_config;
};

struct CmdRunMapping {
  SprayRun* run;
  uint32_t op_idx;
  uint8_t transport = 0;
  uint32_t caller_id = 0;
};

// Per-peer transport metrics for dynamic load balancing
struct PathMetrics {
  std::atomic<uint32_t> inflight{0};
  std::atomic<uint64_t> latency_ns{100000};  // 100 us default
};

struct PeerMetrics {
  PathMetrics ipc;
  PathMetrics rdma;
};

class SprayExecutor {
 public:
  static std::unique_ptr<SprayExecutor> create(
      SprayExecutorConfig const& config);
  SprayExecutor(BatchBackend* device_be, BatchBackend* tpt_be,
                BatchBackend* signal_be = nullptr, int world_size = 0);
  ~SprayExecutor();

  SprayExecutor(SprayExecutor const&) = delete;
  SprayExecutor& operator=(SprayExecutor const&) = delete;

  CollectiveOpHandle submit_allreduce(CollectiveConfig const& cfg, void* input,
                                      void* output, void* scratch);
  CollectiveOpHandle submit_alltoall(CollectiveConfig const& cfg, void* input,
                                     void* output, void* scratch);

  CollectiveOpStatus status(CollectiveOpHandle h) const;
  bool poll(CollectiveOpHandle h);
  bool wait(CollectiveOpHandle h,
            std::chrono::milliseconds to = std::chrono::milliseconds(0));
  void release(CollectiveOpHandle h);
  std::string error_message(CollectiveOpHandle h) const;

  size_t active_count() const;

 private:
  SprayRun* get(CollectiveOpHandle h);

  void enqueue_loop();
  void drain_loop(BatchBackend* be);

  // ── Phase helpers (under SprayRun::mtx) ──
  void collect_ready(SprayRun& run);
  void enqueue_to_ring(SprayRun& run);

  Transport::PeerTransportKind pick_transport(int peer);
  void drain_tpt_loop();
  void drain_signal_loop();

  template <typename F>
  void drain_batch(uint32_t* caller_buf, size_t n, F&& cb) {
    for (size_t i = 0; i < n; ++i) {
      auto& m = cmd_to_run_[caller_buf[i] & (kMaxCmdIdx - 1)];
      if (!m.run || m.caller_id != caller_buf[i]) continue;
      std::lock_guard rlock(m.run->mtx);
      if (!m.run->done[m.op_idx]) {
        m.run->done[m.op_idx] = 1;
        m.run->done_count.fetch_add(1, std::memory_order_release);
        cb(m, caller_buf[i]);
      }
    }
    // Mark completed runs — held outside rlock to avoid AB/BA with enqueue_loop
    std::lock_guard lock(runs_mutex_);
    for (auto& [h, run] : runs_) {
      if (run->status.load(std::memory_order_acquire) != CollectiveOpStatus::Running)
        continue;
      size_t dc = run->done_count.load(std::memory_order_acquire);
      if (dc >= run->tiled.ops.size())
        run->status.store(CollectiveOpStatus::Completed, std::memory_order_release);
    }
  }

  // ── Tensor → buffer ID mapping (dedup: same ptr = same ID) ──
  std::unordered_map<uintptr_t, uint32_t> tensor_to_buf_id_;
  uint32_t next_buf_id_ = 1;
  uint32_t get_or_register_buf(void* ptr, size_t bytes);

  // ── Buffer registration indirection (set by factory, avoids link deps) ──
  void (*register_buf_fn_)(Transport::Communicator*, uint32_t, void*,
                           size_t) = nullptr;

  // ── Owned resources ──
  BatchBackend* device_be_;
  BatchBackend* tpt_be_;
  BatchBackend* signal_be_ = nullptr;
  std::unique_ptr<BatchBackend> owned_device_;
  std::unique_ptr<BatchBackend> owned_transport_;
  std::unique_ptr<BatchBackend> owned_signal_;
  std::shared_ptr<Transport::Communicator> owned_comm_;

  // ── Threads ──
  std::thread enqueue_th_;
  std::thread drain_th_dev_;
  std::thread drain_th_tpt_;
  std::thread drain_th_signal_;
  std::atomic<bool> stop_{false};

  // ── cmd_idx → (run, op_idx) mapping ──
  static constexpr size_t kMaxCmdIdx = 65536;
  CmdRunMapping cmd_to_run_[kMaxCmdIdx];

  // ── Transport LB state ──
  int world_size_ = 0;
  std::unique_ptr<PeerMetrics[]> tpt_metrics_;

  // ── Global cmd_idx counter + run map ──
  uint32_t next_cmd_idx_ = 0;
  std::unordered_map<CollectiveOpHandle, std::unique_ptr<SprayRun>> runs_;
  mutable std::mutex runs_mutex_;
  uint64_t next_handle_ = 1;
};

}  // namespace CCL
}  // namespace UKernel
