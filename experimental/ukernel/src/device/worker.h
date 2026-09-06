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
    // enqueue. 0 = always resident. Default short grace so device-wide
    // syncs work; bursts stay resident (inter-op gaps are µs-scale).
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
  // Block until the fifo tail passes taskId. timeout_ms > 0 bounds the
  // wait (returns early on timeout); 0 = wait forever. The timeout only
  // adds a deadline check to the existing poll — no cost when idle.
  void sync(uint64_t taskId, uint32_t fifoId, uint64_t timeout_ms = 0);

  // NOTE: enqueue / enqueue_batch assume a SINGLE writer per fifo (the
  // executor's enqueue thread). The head read-modify-write in push is not
  // CAS-protected; concurrent writers would lose tasks. Multi-writer push
  // would need a fetch_add slot claim + per-slot ready flag (extra atomic
  // per task) — deferred until a real multi-writer caller exists.
  // relaunch_if_exited / sync / is_done are safe from any thread.
  // Ask every launched worker to exit at its next FIFO-empty poll
  // (instead of waiting out the full idle grace). Workers with tasks in
  // flight / queued do not exit — the flag only takes effect at a true
  // quiescence point, so bursts (multi-stream run-ahead) keep the worker
  // resident and are "auto recycled" as usual. After an exit the next
  // enqueue relaunches the kernel via relaunch_if_exited(); the relaunch
  // clears the flag, so a fresh grid never exits early. Safe from any
  // thread; idempotent.
  void request_idle_exit_all();
  // Cancel a pending request_idle_exit_all(): clear the host flag and
  // push the clear to the device, so a worker about to receive a new
  // burst does not exit at the burst's internal fifo-empty gaps (a
  // sticky force-exit would churn relaunches at every dependent-task
  // boundary). Safe from any thread; idempotent.
  void cancel_idle_exit_all();

  // Per-fifo TaskArgs pool base pointer. Each worker kernel reads args
  // only from its own pool (never the shared singleton), so concurrent
  // workers on different fifos do not race on one args array. Must be set
  // before createWorker()/relaunch; a worker launched with a null pool
  // faults on its first task.
  void set_fifo_task_args(uint32_t fifoId, TaskArgs* pool) {
    if (fifoId < fifo_task_args_.size()) fifo_task_args_[fifoId] = pool;
  }

  // Diagnostic: (head, tail) of a fifo as seen by the host (GDR reads).
  std::pair<uint64_t, uint64_t> fifo_head_tail(uint32_t fifoId) {
    if (fifoId >= fifos_.size()) return {0, 0};
    return {fifos_[fifoId]->fifo.head(), fifos_[fifoId]->fifo.currentId()};
  }

  // Diagnostic: host-visible exit flag for the worker bound to fifoId.
  bool worker_exited(uint32_t fifoId) const {
    for (auto const& wc : workers_)
      if (wc->fifoId == fifoId && wc->launched && wc->h_exited)
        return *wc->h_exited;
    return false;
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
  // Args pool base per fifo (owned by DeviceBackend; WorkerPool only
  // forwards it into each launch).
  std::vector<TaskArgs*> fifo_task_args_;

  // Control stream for host-driven runtime coordination. This is distinct from
  // the per-worker execution streams stored in WorkerContext.
  gpuStream_t control_stream_ = nullptr;
  bool owns_control_stream_ = false;

  // Idle-exit grace in microseconds (Config::idleExitAfterUs). 0 = always
  // resident. The kernel measures it with the wall clock (globaltimer),
  // not with poll counts — the poll rate varies with block count and
  // memory traffic.
  uint32_t idle_exit_us_ = 0;

  std::vector<bool*> d_stop_flags_;
  std::vector<bool*> h_stop_flags_;
  // Host-driven "exit at next quiescence" flags (d_ = device copy the
  // kernel polls, h_ = host-mapped write source). Mirrors the stop-flag
  // plumbing but keeps the worker bound: exit goes through the normal
  // idle-exit rendezvous (sets h_exited), so relaunch_if_exited() can
  // bring it back cheaply without a full createWorker/destroyWorker
  // cycle. Reset to false on every (re)launch.
  std::vector<bool*> d_exit_now_flags_;
  std::vector<bool*> h_exit_now_flags_;
};

}  // namespace Device
}  // namespace UKernel
