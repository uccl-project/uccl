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
    wc->d_readyFlag = nullptr;
    GPU_RT_CHECK(gpuStreamCreateWithFlags(&wc->stream, gpuStreamNonBlocking));
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
    if (wc->d_readyFlag) {
      GPU_RT_CHECK(gpuFree(wc->d_readyFlag));
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
      if (numBlocks > 1) { // only use by multi-sm persistent kernel
        GPU_RT_CHECK(gpuMalloc(&workers_[i]->d_readyFlag, sizeof(uint32_t) * workers_[i]->numBlocks));
        GPU_RT_CHECK(gpuMemset(workers_[i]->d_readyFlag, 0, sizeof(uint32_t) * workers_[i]->numBlocks));
      } 
      void* d_fifo_handle;
      GPU_RT_CHECK(gpuMalloc(&d_fifo_handle, sizeof(mscclpp::C2DDeviceHandle<Task>)));
      workers_[i]->d_fifo_handle = d_fifo_handle;
      launchWorkerForFifo(fifoId);
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
      Task stopTask(TaskType::Stop, DataType::Int8, 0, 0);
      fifos_[fifoId]->fifo.push(stopTask);

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
      destroyWorker(workers_[i]->fifoId);
    }
  }

  for (size_t i = 0; i < workers_.size(); ++i) {
    if (workers_[i]->launched) {
      GPU_RT_CHECK(gpuStreamSynchronize(workers_[i]->stream));
    }
  }

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

void WorkerPool::launchWorkerForFifo(uint32_t fifoId) {
    auto& fifo = fifos_[fifoId]->fifo;
    mscclpp::C2DDeviceHandle<Device::Task> handle = fifo.deviceHandle();
    int workerId = -1;

    for (size_t i = 0; i < workers_.size(); ++i) {
      if (!workers_[i]->launched) { // find first worker for fifo
        workerId = static_cast<int>(i);
        break;
      }
    }
    if (workerId < 0) return;

    // FIFO handle to device
    GPU_RT_CHECK(gpuMemcpyAsync(workers_[workerId]->d_fifo_handle, &handle,
                                sizeof(mscclpp::C2DDeviceHandle<Device::Task>),
                                gpuMemcpyHostToDevice,
                                workers_[workerId]->stream));
    GPU_RT_CHECK(gpuStreamSynchronize(workers_[workerId]->stream));

    auto* d_task_args = TaskManager::instance().d_task_args();

    dim3 grid(workers_[workerId]->numBlocks);
    dim3 block(cfg_.threadsPerBlock);

    size_t smem_size = cfg_.smemSize > 0 ? cfg_.smemSize : 16 * 1024;  // Default 16KB

    // kernel args
    void* args_single[] = {
        &workers_[workerId]->d_fifo_handle,
        &d_task_args,
        &d_stop_flags_[workerId]
    };

    void* args_multi[] = {
        &workers_[workerId]->d_fifo_handle,
        &d_task_args,
        &d_stop_flags_[workerId],
        &workers_[workerId]->d_readyFlag
    };

    // launch kernel
    if (workers_[workerId]->numBlocks == 1) {
        GPU_RT_CHECK(gpuLaunchKernel(
            UKernel::Device::singlePersistentKernel,
            grid, block, args_single, smem_size, workers_[workerId]->stream
        ));
    } else {
        GPU_RT_CHECK(gpuLaunchKernel(
            UKernel::Device::multiPersistentKernel,
            grid, block, args_multi, smem_size, workers_[workerId]->stream
        ));
    }

    workers_[workerId]->ready = true;
}

}  // namespace Device
}  // namespace UKernel
