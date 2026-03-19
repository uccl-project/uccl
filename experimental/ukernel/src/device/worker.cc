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
    auto* wc = new WorkerContext();
    wc->fifoId = UINT32_MAX;
    wc->numBlocks = 1;
    wc->launched = false;
    wc->ready = false;
    GPU_RT_CHECK(gpuStreamCreateWithFlags(&wc->stream, gpuStreamNonBlocking));

    void* d_fifo_handle;
    GPU_RT_CHECK(gpuMalloc(&d_fifo_handle, sizeof(mscclpp::C2DDeviceHandle<Task>)));
    wc->d_fifo_handle = d_fifo_handle;

    workers_.emplace_back(wc);

    bool* d_stop;
    bool* h_stop;
    GPU_RT_CHECK(gpuMalloc(&d_stop, sizeof(bool)));
    GPU_RT_CHECK(gpuHostAlloc(&h_stop, sizeof(bool), gpuHostAllocMapped));
    *h_stop = false;
    GPU_RT_CHECK(gpuMemcpy(d_stop, h_stop, sizeof(bool), gpuMemcpyHostToDevice));
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

  GPU_RT_CHECK(gpuStreamCreateWithFlags(&copy_stream_, gpuStreamNonBlocking));
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

  if (copy_stream_) {
    GPU_RT_CHECK(gpuStreamDestroy(copy_stream_));
  }
  if (owns_stream_ && stream_) {
    GPU_RT_CHECK(gpuStreamDestroy(stream_));
  }
}

bool WorkerPool::createWorker(uint32_t fifoId, uint32_t numBlocks) {
  if (fifoId >= fifos_.size()) {
    return false;
  }

  auto& ctx = *fifos_[fifoId];
  int expected = 0;
  if (!ctx.bound_workers.compare_exchange_strong(expected, 1,
                                                  std::memory_order_acq_rel,
                                                  std::memory_order_relaxed)) {
    return false;
  }

  for (size_t i = 0; i < workers_.size(); ++i) {
    if (!workers_[i]->launched) {
      workers_[i]->fifoId = fifoId;
      workers_[i]->numBlocks = numBlocks;
      workers_[i]->ready = false;
      launchWorkerForFifo(fifoId, i);
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
      GPU_RT_CHECK(gpuMemcpyAsync(d_stop_flags_[i], h_stop_flags_[i],
                                    sizeof(bool), gpuMemcpyHostToDevice,
                                    copy_stream_));
      GPU_RT_CHECK(gpuStreamSynchronize(workers_[i]->stream));
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

  for (;;) {
    uint64_t tail = ctx.fifo.currentId();
    uint64_t head = ctx.fifo.head();
    if ((int64_t)(head + 1 - tail) <= cfg_.fifoCapacity) {
      break;
    }
    std::this_thread::yield();
  }

  uint64_t taskId = ctx.fifo.push(task);
  {
    std::lock_guard<std::mutex> g(ctx.pending_mu);
    ctx.pending[taskId] = {task.args_index(), (TaskType)task.type_u8()};
  }

  int workerId = -1;

  for (size_t i = 0; i < workers_.size(); ++i) {
    if (workers_[i]->fifoId == fifoId && workers_[i]->launched) {
      workerId = static_cast<int>(i);
      break;
    }
  }

  if (workerId < 0) {
    printf("[ERROR] enqueue to fifo %u failed: no worker bound, call createWorker first\n", fifoId);
    return 0;
  }

  return taskId;
}

void WorkerPool::shutdown_all() {
  for (size_t i = 0; i < workers_.size(); ++i) {
    if (workers_[i]->launched) {
      *h_stop_flags_[i] = true;
      GPU_RT_CHECK(gpuMemcpyAsync(d_stop_flags_[i], h_stop_flags_[i],
                                    sizeof(bool), gpuMemcpyHostToDevice,
                                    copy_stream_));
    }
  }
  GPU_RT_CHECK(gpuStreamSynchronize(copy_stream_));

  for (auto& wc : workers_) {
    wc->launched = false;
  }
  for (auto& ctx : fifos_) {
    ctx->bound_workers.store(0, std::memory_order_relaxed);
  }
}

bool WorkerPool::is_done(uint64_t taskId, uint32_t fifoId) {
  if (fifoId >= fifos_.size()) return true;

  auto& ctx = *fifos_[fifoId];
  uint64_t current = ctx.fifo.currentId();

  if ((int64_t)(current - taskId) > 0) {
    std::lock_guard<std::mutex> g(ctx.pending_mu);
    auto it = ctx.pending.begin();
    while (it != ctx.pending.end()) {
      uint64_t tid = it->first;
      if ((int64_t)(current - tid) > 0) {
        PendingTask const& p = it->second;
        if (p.type == TaskType::CollCopy || p.type == TaskType::CollReduce) {
          TaskManager::instance().free_task_args(p.argsId);
        }
        it = ctx.pending.erase(it);
      } else {
        ++it;
      }
    }
    return true;
  }
  return false;
}

void WorkerPool::launchWorkerForFifo(uint32_t fifoId, int workerIdHint) {
  auto& fifo = fifos_[fifoId]->fifo;
  mscclpp::C2DDeviceHandle<Device::Task> handle = fifo.deviceHandle();

  int workerId = -1;
  if (workerIdHint >= 0 && workers_[workerIdHint]->fifoId == fifoId &&
      !workers_[workerIdHint]->launched) {
    workerId = workerIdHint;
  } else {
    for (size_t i = 0; i < workers_.size(); ++i) {
      if (workers_[i]->fifoId == fifoId && workers_[i]->launched) {
        workerId = static_cast<int>(i);
        break;
      }
    }
    if (workerId < 0) {
      for (size_t i = 0; i < workers_.size(); ++i) {
        if (!workers_[i]->launched) {
          workerId = static_cast<int>(i);
          break;
        }
      }
    }
  }
  if (workerId < 0) return;

  GPU_RT_CHECK(gpuMemcpyAsync(workers_[workerId]->d_fifo_handle, &handle,
                         sizeof(mscclpp::C2DDeviceHandle<Device::Task>),
                         gpuMemcpyHostToDevice, workers_[workerId]->stream));

  auto* d_task_args = TaskManager::instance().d_task_args();

  void* args[] = {
    &workers_[workerId]->d_fifo_handle,
    &d_task_args,
    &d_stop_flags_[workerId]
  };

  dim3 grid(workers_[workerId]->numBlocks);
  dim3 block(cfg_.threadsPerBlock);

  GPU_RT_CHECK(gpuLaunchKernel(basePersistentKernel, grid, block, args,
                               0, workers_[workerId]->stream));
  
  workers_[workerId]->ready = true;
}

}  // namespace Device
}  // namespace UKernel
