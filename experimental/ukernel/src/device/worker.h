#pragma once

#include "c2d_fifo.h"
#include "gpu_rt.h"
#include "task.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace UKernel {
namespace Device {

struct MultiBlockSync;

class WorkerPool {
 public:
  static constexpr uint64_t kInvalidTaskId = ~uint64_t{0};

  struct Config {
    uint32_t numMaxWorkers = 16;
    uint32_t threadsPerBlock = 64;
    uint32_t fifoCapacity = 16;
    uint32_t smemSize = 0;
    // Control stream used for host-driven bookkeeping copies such as stop
    // flags. Persistent worker kernels still run on per-worker streams.
    gpuStream_t controlStream = nullptr;
  };

  struct WorkerSpec {
    uint32_t workerId;
    uint32_t fifoId;
    uint32_t numBlocks;
  };

  explicit WorkerPool(Config const& config);
  ~WorkerPool();

  bool createWorker(uint32_t fifoId, uint32_t numBlocks);
  bool pollWorker(uint32_t fifoId);
  void waitWorker(uint32_t fifoId);
  void destroyWorker(uint32_t fifoId);

  uint64_t enqueue(Task const& task, uint32_t fifoId);
  uint64_t enqueue_batch(std::vector<Task> const& tasks, uint32_t fifoId);
  void retireTask(uint32_t fifoId, uint64_t taskId);

  void shutdown_all();

  bool is_done(uint64_t taskId, uint32_t fifoId);

  gpuStream_t control_stream() const { return control_stream_; }

  uint32_t num_fifos() const { return static_cast<uint32_t>(fifos_.size()); }

  Config const& cfg() const { return cfg_; }

  gpuStream_t getWorkerStream(uint32_t fifoId) const {
    for (size_t i = 0; i < workers_.size(); ++i) {
      if (workers_[i]->fifoId == fifoId && workers_[i]->launched) {
        return workers_[i]->stream;
      }
    }
    return nullptr;
  }

 private:
  struct PendingTask {
    uint32_t argsId;
    TaskType type;
  };

  struct FifoContext {
    mscclpp::CpuToGpuFifo<Task> fifo;
    std::mutex pending_mu;
    std::unordered_map<uint64_t, PendingTask> pending;
    std::atomic<int> bound_workers{0};

    explicit FifoContext(int capacity) : fifo(capacity) {}
  };

  struct WorkerContext {
    uint32_t fifoId;
    uint32_t numBlocks;
    bool launched;
    bool ready;
    // Dedicated execution stream for the worker's persistent kernel.
    gpuStream_t stream = nullptr;
    mscclpp::C2DDeviceHandle<Task>* d_fifo_handle = nullptr;
    MultiBlockSync* d_multi_sync = nullptr;
  };

  void launchWorkerForFifo(size_t workerIndex);
  void reclaimAllPendingTasks(FifoContext& ctx);

  Config cfg_;
  std::vector<std::unique_ptr<FifoContext>> fifos_;
  std::vector<std::unique_ptr<WorkerContext>> workers_;

  // Control stream for host-driven runtime coordination. This is distinct from
  // the per-worker execution streams stored in WorkerContext.
  gpuStream_t control_stream_ = nullptr;
  bool owns_control_stream_ = false;

  std::vector<bool*> d_stop_flags_;
  std::vector<bool*> h_stop_flags_;
};

}  // namespace Device
}  // namespace UKernel
