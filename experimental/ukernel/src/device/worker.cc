#include "worker.h"
#include "persistent_kernel_ops.h"
#include <algorithm>

namespace UKernel {
namespace Device {

WorkerPool::WorkerPool(Config const& config) : cfg_(config) {
  fifos_.reserve(cfg_.numMaxWorkers);
  for (uint32_t i = 0; i < cfg_.numMaxWorkers; ++i) {
    fifos_.emplace_back(std::make_unique<FifoContext>(cfg_.fifoCapacity));
  }

  workers_.reserve(cfg_.numMaxWorkers);
  for (uint32_t i = 0; i < cfg_.numMaxWorkers; ++i) {
    // create all workercontexts first
    auto* wc = new WorkerContext();
    wc->fifoId = UINT32_MAX;
    wc->numBlocks = 1;
    wc->launched = false;
    wc->ready = false;
    GPU_RT_CHECK(gpuStreamCreateWithFlags(&wc->stream, gpuStreamNonBlocking));
    GPU_RT_CHECK(
        gpuMalloc(&wc->d_fifo_handle, sizeof(mscclpp::C2DDeviceHandle<Task>)));
    workers_.emplace_back(wc);

    bool* d_stop;
    bool* h_stop;
    GPU_RT_CHECK(gpuMalloc(&d_stop, sizeof(bool)));
    GPU_RT_CHECK(gpuHostAlloc(&h_stop, sizeof(bool), gpuHostAllocMapped));
    *h_stop = false;
    GPU_RT_CHECK(
        gpuMemcpy(d_stop, h_stop, sizeof(bool), gpuMemcpyHostToDevice));
    d_stop_flags_.push_back(d_stop);
    h_stop_flags_.push_back(h_stop);
  }

  if (cfg_.stream) {
    stream_ = cfg_.stream;
    owns_stream_ = false;
  } else {
    GPU_RT_CHECK(gpuStreamCreateWithFlags(&stream_, gpuStreamNonBlocking));
    owns_stream_ = true;
  }
}

WorkerPool::~WorkerPool() {
  shutdown_all();

  for (auto& wc : workers_) {
    if (wc->stream) {
      GPU_RT_CHECK(gpuStreamDestroy(wc->stream));
    }
    if (wc->d_fifo_handle) {
      GPU_RT_CHECK(gpuFree(wc->d_fifo_handle));
    }
    if (wc->d_multi_sync) {
      GPU_RT_CHECK(gpuFree(wc->d_multi_sync));
    }
  }
  workers_.clear();

  for (auto* d : d_stop_flags_) {
    GPU_RT_CHECK(gpuFree(d));
  }
  for (auto* h : h_stop_flags_) {
    GPU_RT_CHECK(gpuFreeHost(h));
  }
  d_stop_flags_.clear();
  h_stop_flags_.clear();

  if (owns_stream_ && stream_) {
    GPU_RT_CHECK(gpuStreamDestroy(stream_));
  }
}

bool WorkerPool::createWorker(uint32_t fifoId, uint32_t numBlocks) {
  if (fifoId >= fifos_.size() || numBlocks == 0) {
    return false;
  }
  if (numBlocks > 1) {
    int device = 0;
    int sm_count = 0;
    GPU_RT_CHECK(gpuGetDevice(&device));
    GPU_RT_CHECK(gpuDeviceGetAttribute(&sm_count, gpuDevAttrMultiProcessorCount,
                                       device));
    if (numBlocks > static_cast<uint32_t>(sm_count)) {
      return false;
    }
  }

  auto& ctx = *fifos_[fifoId];
  int expected = 0;
  if (!ctx.bound_workers.compare_exchange_strong(
          expected, 1, std::memory_order_acq_rel, std::memory_order_relaxed)) {
    return false;
  }

  for (size_t i = 0; i < workers_.size(); ++i) {
    if (!workers_[i]->launched) {
      workers_[i]->fifoId = fifoId;
      workers_[i]->numBlocks = numBlocks;
      workers_[i]->ready = false;
      *h_stop_flags_[i] = false;
      GPU_RT_CHECK(gpuMemcpy(d_stop_flags_[i], h_stop_flags_[i], sizeof(bool),
                             gpuMemcpyHostToDevice));
      if (workers_[i]->d_multi_sync) {
        GPU_RT_CHECK(gpuFree(workers_[i]->d_multi_sync));
        workers_[i]->d_multi_sync = nullptr;
      }
      if (numBlocks > 1) {
        GPU_RT_CHECK(
            gpuMalloc(&workers_[i]->d_multi_sync, sizeof(MultiBlockSync)));
        GPU_RT_CHECK(
            gpuMemset(workers_[i]->d_multi_sync, 0, sizeof(MultiBlockSync)));
      }
      launchWorkerForFifo(i);
      workers_[i]->launched = true;
      return true;
    }
  }

  ctx.bound_workers.store(0, std::memory_order_relaxed);
  return false;
}

bool WorkerPool::pollWorker(uint32_t fifoId) {
  if (fifoId >= fifos_.size()) {
    return false;
  }
  for (size_t i = 0; i < workers_.size(); ++i) {
    if (workers_[i]->fifoId == fifoId && workers_[i]->launched) {
      return workers_[i]->ready;
    }
  }
  return false;
}

void WorkerPool::waitWorker(uint32_t fifoId) {
  while (!pollWorker(fifoId)) {
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
}

void WorkerPool::destroyWorker(uint32_t fifoId) {
  if (fifoId >= fifos_.size()) {
    return;
  }

  auto& ctx = *fifos_[fifoId];
  for (size_t i = 0; i < workers_.size(); ++i) {
    if (workers_[i]->fifoId == fifoId && workers_[i]->launched) {
      *h_stop_flags_[i] = true;
      GPU_RT_CHECK(gpuMemcpy(d_stop_flags_[i], h_stop_flags_[i], sizeof(bool),
                             gpuMemcpyHostToDevice));

      GPU_RT_CHECK(gpuStreamSynchronize(workers_[i]->stream));
      reclaimFinishedTasks(ctx, ctx.fifo.currentId());
      if (workers_[i]->d_multi_sync) {
        GPU_RT_CHECK(gpuFree(workers_[i]->d_multi_sync));
        workers_[i]->d_multi_sync = nullptr;
      }
      workers_[i]->launched = false;
      workers_[i]->ready = false;
      workers_[i]->fifoId = UINT32_MAX;
      ctx.bound_workers.store(0, std::memory_order_relaxed);
      break;
    }
  }
}

uint64_t WorkerPool::enqueue(Task const& task, uint32_t fifoId) {
  if (fifoId >= fifos_.size()) {
    return 0;
  }

  auto& ctx = *fifos_[fifoId];
  int workerId = -1;
  for (size_t i = 0; i < workers_.size(); ++i) {
    if (workers_[i]->fifoId == fifoId && workers_[i]->launched) {
      workerId = static_cast<int>(i);
      break;
    }
  }
  if (workerId < 0) {
    printf(
        "[ERROR] enqueue to fifo %u failed: no worker bound, call createWorker "
        "first\n",
        fifoId);
    return 0;
  }

  // Check if there's space in FIFO without blocking initially
  uint64_t tail = ctx.fifo.currentId();
  uint64_t head = ctx.fifo.head();
  if ((int64_t)(head + 1 - tail) > cfg_.fifoCapacity) {
    // FIFO is full, return 0 to indicate failure - caller can retry
    return 0;
  }

  uint64_t taskId = ctx.fifo.push(task);
  {
    std::lock_guard<std::mutex> g(ctx.pending_mu);
    ctx.pending[taskId] = {task.args_index(), (TaskType)task.type_u8()};
  }

  return taskId;
}

void WorkerPool::shutdown_all() {
  for (size_t i = 0; i < workers_.size(); ++i) {
    if (workers_[i]->launched) {
      destroyWorker(workers_[i]->fifoId);
    }
  }

  for (auto& wc : workers_) {
    wc->launched = false;
  }
  for (auto& ctx : fifos_) {
    ctx->bound_workers.store(0, std::memory_order_relaxed);
    reclaimAllPendingTasks(*ctx);
  }
}

bool WorkerPool::is_done(uint64_t taskId, uint32_t fifoId) {
  if (fifoId >= fifos_.size()) return true;

  auto& ctx = *fifos_[fifoId];
  uint64_t current = ctx.fifo.currentId();

  if ((int64_t)(current - taskId) > 0) {
    reclaimFinishedTasks(ctx, current);
    return true;
  }
  return false;
}

void WorkerPool::launchWorkerForFifo(size_t workerIndex) {
  auto& worker = *workers_[workerIndex];
  auto& fifo = fifos_[worker.fifoId]->fifo;
  mscclpp::C2DDeviceHandle<Device::Task> handle = fifo.deviceHandle();

  GPU_RT_CHECK(gpuMemcpyAsync(worker.d_fifo_handle, &handle,
                              sizeof(mscclpp::C2DDeviceHandle<Device::Task>),
                              gpuMemcpyHostToDevice, worker.stream));
  GPU_RT_CHECK(gpuStreamSynchronize(worker.stream));

  auto* d_task_args = TaskManager::instance().d_task_args();

  dim3 grid(worker.numBlocks);
  dim3 block(cfg_.threadsPerBlock);
  size_t smem_size = cfg_.smemSize > 0 ? cfg_.smemSize : 16 * 1024;

  void* args_single[] = {&worker.d_fifo_handle, &d_task_args,
                         &d_stop_flags_[workerIndex]};

  void* args_multi[] = {&worker.d_fifo_handle, &d_task_args,
                        &d_stop_flags_[workerIndex], &worker.d_multi_sync};

  if (worker.numBlocks == 1) {
    GPU_RT_CHECK(gpuLaunchKernel(UKernel::Device::singlePersistentKernel, grid,
                                 block, args_single, smem_size, worker.stream));
  } else {
    GPU_RT_CHECK(gpuLaunchKernel(UKernel::Device::multiPersistentKernel, grid,
                                 block, args_multi, smem_size, worker.stream));
  }

  worker.ready = true;
}

void WorkerPool::reclaimFinishedTasks(FifoContext& ctx,
                                      uint64_t currentTaskId) {
  std::lock_guard<std::mutex> g(ctx.pending_mu);
  bool const task_manager_ready = TaskManager::instance().inited();
  auto it = ctx.pending.begin();
  while (it != ctx.pending.end()) {
    uint64_t tid = it->first;
    if ((int64_t)(currentTaskId - tid) > 0) {
      PendingTask const& p = it->second;
      if (task_manager_ready &&
          (p.type == TaskType::CollCopy || p.type == TaskType::CollReduce)) {
        TaskManager::instance().free_task_args(p.argsId);
      }
      it = ctx.pending.erase(it);
    } else {
      ++it;
    }
  }
}

void WorkerPool::reclaimAllPendingTasks(FifoContext& ctx) {
  std::lock_guard<std::mutex> g(ctx.pending_mu);
  bool const task_manager_ready = TaskManager::instance().inited();
  for (auto const& [_, pending] : ctx.pending) {
    if (task_manager_ready && (pending.type == TaskType::CollCopy ||
                               pending.type == TaskType::CollReduce)) {
      TaskManager::instance().free_task_args(pending.argsId);
    }
  }
  ctx.pending.clear();
}

}  // namespace Device
}  // namespace UKernel
