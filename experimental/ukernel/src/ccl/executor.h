#pragma once

#include "backend/backend.h"
#include "coll_config.h"
#include "lower.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace UKernel {
namespace Transport {
struct CommunicatorConfig;
class Communicator;
}
namespace CCL {

enum class CollectiveOpStatus : uint32_t {
  Queued, Running, Completed, Failed,
};

using CollectiveOpHandle = uint64_t;
inline constexpr CollectiveOpHandle kInvalidHandle = 0;

// ── Op sprayer (new executor) ───────────────────────────────────────────

// Per-collective mutable state.
struct SprayRun {
  CollectiveOpStatus status = CollectiveOpStatus::Queued;
  TiledResult tiled;
  void* input = nullptr;
  void* output = nullptr;
  void* scratch = nullptr;
  std::string error;

  std::vector<bool> done;
  std::vector<uint32_t> ready;    // op indices ready to submit
  size_t done_count = 0;
  uint32_t next_layer = 0;

  // Command buffers — reused across cycles
  std::vector<uint32_t> dev_map;   // cmd index → op index
  std::vector<uint32_t> tpt_map;
};

struct SprayExecutorConfig {
  int gpu_id;
  int rank;
  int world_size;
  size_t device_task_capacity = 256;
  size_t max_device_fifos = 8;
  int threads_per_block = 64;
  size_t fifo_capacity = 256;
  size_t smem_size = 48 * 1024;
  std::shared_ptr<struct UKernel::Transport::CommunicatorConfig> communicator_config;
};

class SprayExecutor {
 public:
  static std::unique_ptr<SprayExecutor> create(SprayExecutorConfig const& config);
  SprayExecutor(BatchExecutorBackends backends);
  ~SprayExecutor() = default;

  CollectiveOpHandle submit_allreduce(CollectiveConfig const& cfg,
                                      void* input, void* output, void* scratch);
  CollectiveOpHandle submit_alltoall(CollectiveConfig const& cfg,
                                     void* input, void* output, void* scratch);

  CollectiveOpStatus status(CollectiveOpHandle h) const;
  bool poll(CollectiveOpHandle h);
  void progress();
  bool wait(CollectiveOpHandle h, std::chrono::milliseconds to = std::chrono::milliseconds(0));
  void release(CollectiveOpHandle h);
  std::string error_message(CollectiveOpHandle h) const;

  size_t active_count() const;
  void run_tiled(TiledResult const& tiled, void* input, void* output, void* scratch);

 private:
  SprayRun* get(CollectiveOpHandle h);
  void advance(SprayRun& run);
  void collect_ready(SprayRun& run);
  void enqueue_ready(SprayRun& run);
  void drain_done(SprayRun& run);

  BatchExecutorBackends be_;
  std::unique_ptr<BatchBackend> owned_device_;
  std::unique_ptr<BatchBackend> owned_transport_;
  std::unique_ptr<Transport::Communicator> owned_comm_;
  std::unordered_map<CollectiveOpHandle, SprayRun> runs_;
  uint64_t next_handle_ = 1;
};

}  // namespace CCL
}  // namespace UKernel
