#pragma once

#include "backend/backend.h"
#include "coll_config.h"
#include "scheduler.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace UKernel {
namespace Transport {
struct CommunicatorConfig;
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

struct ExecutorConfig {
  int gpu_id = 0;
  int rank = 0;
  int world_size = 1;
  std::shared_ptr<UKernel::Transport::CommunicatorConfig> communicator_config;
  uint32_t device_task_capacity = 4096;
  uint32_t max_device_fifos = 8;
  uint32_t threads_per_block = 256;
  uint32_t fifo_capacity = 64;
  uint32_t smem_size = 0;
};

// Per-invocation mutable execution state.
struct CollectiveRun {
  CollectiveOpStatus status = CollectiveOpStatus::Queued;
  TiledResult tiled;
  CollectiveBinding* binding = nullptr;
  std::string error_message;

  std::vector<bool> completed;
  std::vector<BackendToken> tokens;

  // O(1) drain → op index lookup.  Key combines backend + token value
  // because transport and device backends have independent token spaces.
  struct TokenKey {
    Backend* backend;
    uint64_t value;
    bool operator==(TokenKey const& o) const {
      return backend == o.backend && value == o.value;
    }
  };
  struct TokenKeyHash {
    size_t operator()(TokenKey const& k) const {
      return std::hash<uintptr_t>()(reinterpret_cast<uintptr_t>(k.backend)) ^
             std::hash<uint64_t>()(k.value);
    }
  };
  std::unordered_map<TokenKey, size_t, TokenKeyHash> token_to_op_idx;

  std::vector<std::vector<uint32_t>> stream_ops;
  uint32_t num_streams = 1;
  std::vector<size_t> stream_head;
  std::vector<BackendToken> done_buf;
  std::vector<std::pair<uint32_t, uint32_t>> ready;
  size_t completed_count = 0;
  size_t total_ops = 0;
  uint64_t signal_seq_base = 0;
};

class Executor {
 public:
  explicit Executor(
      ExecutorBackends backends,
      std::function<bool(int, uint32_t, size_t, size_t, void**, int*)>
          resolve_ipc_buffer_pointer = {});

  explicit Executor(ExecutorConfig const& config = {});
  ~Executor();

  Executor(Executor const&) = delete;
  Executor& operator=(Executor const&) = delete;

  CollectiveOpHandle submit_allreduce(CollectiveConfig const& config,
                                      CollectiveBinding& binding);
  CollectiveOpHandle submit_alltoall(CollectiveConfig const& config,
                                     CollectiveBinding& binding);

  CollectiveOpStatus status(CollectiveOpHandle handle) const;
  bool poll(CollectiveOpHandle handle);
  void progress();
  bool wait(CollectiveOpHandle handle,
            std::chrono::milliseconds timeout = std::chrono::milliseconds(0));
  void release(CollectiveOpHandle handle);
  std::string error_message(CollectiveOpHandle handle) const;

  void run_plan(TiledResult const& tiled, CollectiveBinding& binding);

  size_t active_count() const;
  UKernel::Transport::Communicator* communicator();
  UKernel::Transport::Communicator const* communicator() const;

 private:
  CollectiveRun* get_run(CollectiveOpHandle handle);
  CollectiveRun const* get_run(CollectiveOpHandle handle) const;
  CollectiveOpHandle submit_collective(CollectiveKind kind,
                                       CollectiveConfig const& config,
                                       CollectiveBinding& binding);
  void advance_run(CollectiveRun& run);

  ExecutorBackends backends_{};
  std::unique_ptr<Backend> owned_transport_backend_;
  std::unique_ptr<Backend> owned_device_backend_;
  std::function<bool(int, uint32_t, size_t, size_t, void**, int*)>
      resolve_ipc_buffer_pointer_;
  std::unordered_map<CollectiveOpHandle, CollectiveRun> runs_;
  uint64_t next_handle_ = 1;
  uint64_t validated_sig_ = 0;

  std::vector<GpuSignalPeer> gpu_comp_;
};

}  // namespace CCL
}  // namespace UKernel
