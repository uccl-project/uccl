#pragma once

#include "c2d_fifo.h"
#include "gpu_rt.h"
#include "task.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
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
    // Grace period (µs) of continuous fifo emptiness after which the
    // persistent kernel exits; the host relaunches it on the next
    // enqueue. Default 500µs: consecutive collectives in a burst stay in
    // ONE persistent instance (inter-op gaps are µs-scale), but a long
    // idle lets the kernel exit so device-wide syncs (cudaDeviceSync-
    // hronize, legacy default stream, D2H/.item()) do not deadlock.
    // 0 = always resident (not recommended for apps that sync).
    uint32_t idleExitAfterUs = 500;
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

  // Relaunch the worker bound to fifoId if its kernel exited on the
  // idle grace timer. Called on enqueue and on drain (so a task that
  // raced the kernel's exit always gets picked up).
  void relaunch_if_exited(uint32_t fifoId);

  uint64_t enqueue(Task const& task, uint32_t fifoId);
  uint64_t enqueue_batch(std::vector<Task> const& tasks, uint32_t fifoId);
  void shutdown_all();

  bool is_done(uint64_t taskId, uint32_t fifoId);
  void sync(uint64_t taskId, uint32_t fifoId);

  // Diagnostic: (head, tail) of a fifo as seen by the host (GDR reads).
  std::pair<uint64_t, uint64_t> fifo_head_tail(uint32_t fifoId) {
    if (fifoId >= fifos_.size()) return {0, 0};
    return {fifos_[fifoId]->fifo.head(), fifos_[fifoId]->fifo.currentId()};
  }

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
  struct FifoContext {
    mscclpp::CpuToGpuFifo<Task> fifo;
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
    // Host-mapped flag set by the kernel when it exits on the idle grace
    // timer; the next enqueue relaunches it. Host reads, kernel writes.
    bool* h_exited = nullptr;
  };

  void launchWorkerForFifo(size_t workerIndex);

  Config cfg_;
  std::vector<std::unique_ptr<FifoContext>> fifos_;
  std::vector<std::unique_ptr<WorkerContext>> workers_;

  // Control stream for host-driven runtime coordination. This is distinct from
  // the per-worker execution streams stored in WorkerContext.
  gpuStream_t control_stream_ = nullptr;
  bool owns_control_stream_ = false;

  // Idle-grace in ~100ns polls, derived from Config::idleExitAfterUs
  // (0 = always resident).
  uint32_t exit_idle_iters_ = 0;

  std::vector<bool*> d_stop_flags_;
  std::vector<bool*> h_stop_flags_;
};

}  // namespace Device
}  // namespace UKernel
