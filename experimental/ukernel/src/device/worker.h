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
  struct Config {
    uint32_t numMaxWorkers = 16;
    uint32_t threadsPerBlock = 64;
    uint32_t fifoCapacity = 16;
    uint32_t smemSize = 0;
    gpuStream_t stream = nullptr;
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

  void shutdown_all();

  bool is_done(uint64_t taskId, uint32_t fifoId);

  gpuStream_t stream() const { return stream_; }

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
    gpuStream_t stream = nullptr;
    mscclpp::C2DDeviceHandle<Task>* d_fifo_handle = nullptr;
    MultiBlockSync* d_multi_sync = nullptr;
  };

  void launchWorkerForFifo(size_t workerIndex);
  void reclaimFinishedTasks(FifoContext& ctx, uint64_t currentTaskId);
  void reclaimAllPendingTasks(FifoContext& ctx);

  Config cfg_;
  std::vector<std::unique_ptr<FifoContext>> fifos_;
  std::vector<std::unique_ptr<WorkerContext>> workers_;

  gpuStream_t stream_ = nullptr;
  bool owns_stream_ = false;

  std::vector<bool*> d_stop_flags_;
  std::vector<bool*> h_stop_flags_;
};

}  // namespace Device
}  // namespace UKernel
