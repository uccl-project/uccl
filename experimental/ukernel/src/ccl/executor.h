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
}
namespace CCL {

enum class CollectiveOpStatus : uint32_t {
  Queued, Running, Completed, Failed,
};

using CollectiveOpHandle = uint64_t;
inline constexpr CollectiveOpHandle kInvalidHandle = 0;

// ── Op sprayer with async jring backends ────────────────────────────────

class AsyncBackend;

struct SprayRun {
  CollectiveOpStatus status = CollectiveOpStatus::Queued;
  TiledResult tiled;
  void* input = nullptr;
  void* output = nullptr;
  void* scratch = nullptr;
  std::string error;

  std::vector<bool> done;
  std::vector<bool> submitted;  // op already enqueued to cmd_ring
  std::vector<uint32_t> ready;
  std::atomic<size_t> done_count{0};
  uint32_t next_layer = 0;

  // Mutex for done[] / map[] concurrent access between enqueue & drain threads
  std::mutex mtx;

  // cmd_ring batch bookkeeping — per-cycle
  std::vector<CmdWithId> dev_cmds;
  std::vector<CmdWithId> tpt_cmds;
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
  std::shared_ptr<struct UKernel::Transport::CommunicatorConfig> communicator_config;
};

struct CmdRunMapping {
  SprayRun* run;
  uint32_t op_idx;
};

// Per-peer transport metrics for dynamic load balancing
struct PathMetrics {
  std::atomic<uint32_t> inflight{0};
  std::atomic<double> latency_us{100.0};
};

struct PeerMetrics {
  PathMetrics ipc;
  PathMetrics rdma;
};

class SprayExecutor {
 public:
  static std::unique_ptr<SprayExecutor> create(SprayExecutorConfig const& config);
  SprayExecutor(BatchBackend* device_be, BatchBackend* tpt_be);
  ~SprayExecutor();

  SprayExecutor(SprayExecutor const&) = delete;
  SprayExecutor& operator=(SprayExecutor const&) = delete;

  CollectiveOpHandle submit_allreduce(CollectiveConfig const& cfg,
                                      void* input, void* output, void* scratch);
  CollectiveOpHandle submit_alltoall(CollectiveConfig const& cfg,
                                     void* input, void* output, void* scratch);

  CollectiveOpStatus status(CollectiveOpHandle h) const;
  bool poll(CollectiveOpHandle h);
  bool wait(CollectiveOpHandle h, std::chrono::milliseconds to = std::chrono::milliseconds(0));
  void release(CollectiveOpHandle h);
  std::string error_message(CollectiveOpHandle h) const;

  size_t active_count() const;

 private:
  SprayRun* get(CollectiveOpHandle h);

  void enqueue_loop();
  void drain_loop(AsyncBackend* async_be);

  // ── Phase helpers (under SprayRun::mtx) ──
  void collect_ready(SprayRun& run);
  void enqueue_to_ring(SprayRun& run, AsyncBackend* async_be);

  Transport::PeerTransportKind pick_transport(int peer);
  void drain_tpt_loop();

  // ── Owned resources ──
  BatchBackend* device_be_;
  BatchBackend* tpt_be_;
  std::unique_ptr<AsyncBackend> async_dev_;
  std::unique_ptr<AsyncBackend> async_tpt_;
  std::unique_ptr<BatchBackend> owned_device_;
  std::unique_ptr<BatchBackend> owned_transport_;
  std::shared_ptr<Transport::Communicator> owned_comm_;

  // ── Threads ──
  std::thread enqueue_th_;
  std::thread drain_th_dev_;
  std::thread drain_th_tpt_;
  std::atomic<bool> stop_{false};

  // ── cmd_idx → (run, op_idx) mapping ──
  static constexpr size_t kMaxCmdIdx = 65536;
  CmdRunMapping cmd_to_run_[kMaxCmdIdx];

  // ── Transport LB state ──
  std::unordered_map<int, PeerMetrics> tpt_metrics_;
  std::unordered_map<uint32_t, uint8_t> cmd_transport_;

  // ── Global cmd_idx counter + run map ──
  uint32_t next_cmd_idx_ = 0;
  std::unordered_map<CollectiveOpHandle, std::unique_ptr<SprayRun>> runs_;
  std::mutex runs_mutex_;
  uint64_t next_handle_ = 1;
};

}  // namespace CCL
}  // namespace UKernel
