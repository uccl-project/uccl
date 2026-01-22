#pragma once

#include "c2d_fifo.h"
#include "gpu_rt.h"
#include "operator.h"
#include <iostream>
#include <thread>
#include <unordered_map>
#include <vector>

namespace UKernel {
namespace Compute {

constexpr uint64_t kAbortTailValue = (uint64_t)-2;

struct PersistentKernelConfig {
  uint32_t numBlocks = 1;
  uint32_t threadsPerBlock = 64;  // assume that warpsize is 32
  uint32_t fifoCapacity = 16;
  uint32_t smemSize = 0;

  gpuStream_t stream = nullptr;  // if user manage the stream
};

template <typename T>
class PersistentKernel {
 public:
  explicit PersistentKernel(PersistentKernelConfig const& config)
      : cfg_(config) {
    // multi fifos init
    fifos_.reserve(cfg_.numBlocks);
    for (uint32_t i = 0; i < cfg_.numBlocks; ++i) {
      fifos_.emplace_back(std::make_unique<FifoWithPending>(cfg_.fifoCapacity));
    }

    // Allocate memory for stop flag (host and device)
    GPU_RT_CHECK(gpuMalloc(&d_stopFlag_, sizeof(bool)));
    GPU_RT_CHECK(gpuHostAlloc(&h_stopFlag_, sizeof(bool), gpuHostAllocMapped));

    // Initialize stop flag to false
    *h_stopFlag_ = false;
    GPU_RT_CHECK(gpuMemcpy(d_stopFlag_, h_stopFlag_, sizeof(bool),
                           gpuMemcpyHostToDevice));

    // kernel stream
    if (cfg_.stream) {
      stream_ = cfg_.stream;
      owns_stream_ = false;
    } else {
      GPU_RT_CHECK(gpuStreamCreateWithFlags(&stream_, gpuStreamNonBlocking));
      owns_stream_ = true;
    }

    // copy stream
    GPU_RT_CHECK(gpuStreamCreateWithFlags(&copy_stream_, gpuStreamNonBlocking));
  };

  ~PersistentKernel() noexcept(false) {
    if (launched_) stop();

    GPU_RT_CHECK(gpuFree(d_stopFlag_));
    GPU_RT_CHECK(gpuFreeHost(h_stopFlag_));
    if (d_fifo_handles_) {
      GPU_RT_CHECK(gpuFree(d_fifo_handles_));
    }

    if (copy_stream_) GPU_RT_CHECK(gpuStreamDestroy(copy_stream_));
    if (owns_stream_ && stream_) GPU_RT_CHECK(gpuStreamDestroy(stream_));
  };

  bool launch() {
    if (launched_) return false;

    auto* d_coll = UKernel::Compute::TaskManager::instance().d_coll();
    auto* d_moe = UKernel::Compute::TaskManager::instance().d_moe();

    std::vector<mscclpp::C2DDeviceHandle<T>> h_fifo_handles;
    h_fifo_handles.reserve(cfg_.numBlocks);

    for (auto& f : fifos_) {
      h_fifo_handles.push_back(f->fifo.deviceHandle());
    }

    GPU_RT_CHECK(
        gpuMalloc(&d_fifo_handles_, sizeof(*d_fifo_handles_) * cfg_.numBlocks));
    GPU_RT_CHECK(gpuMemcpyAsync(d_fifo_handles_, h_fifo_handles.data(),
                                sizeof(*d_fifo_handles_) * cfg_.numBlocks,
                                gpuMemcpyHostToDevice, copy_stream_));
    GPU_RT_CHECK(gpuStreamSynchronize(copy_stream_));

    void* args[] = {&d_fifo_handles_, &d_coll, &d_moe, &d_stopFlag_};

    dim3 grid(cfg_.numBlocks);
    dim3 block(cfg_.threadsPerBlock);

    GPU_RT_CHECK(gpuLaunchKernel(basePersistentKernel<T>, grid, block, args,
                                 cfg_.smemSize, stream_));

    launched_ = true;
    return true;
  };

  uint64_t submit(const T& task) {
    // TODO: multi-threads submit
    auto& fq = *fifos_[task.block_index()];
    for (;;) {
      uint64_t tail = fq.fifo.currentId();
      uint64_t head = fq.fifo.head();
      if ((int64_t)(head + 1 - tail) <= cfg_.fifoCapacity) {
        break;
      }
      std::this_thread::yield();
    }

    uint64_t taskId = fq.fifo.push(task);
    {
      std::lock_guard<std::mutex> g(fq.pending_mu_);
      fq.pending[taskId] = {task.args_index(), (TaskType)task.type_u8()};
    }
    return taskId;
  };

  bool is_done(uint64_t blockId, uint64_t taskId, size_t count = 0) {
    uint64_t doneBefore = fifos_[blockId]->fifo.currentId();
    {
      std::lock_guard<std::mutex> g(fifos_[blockId]->pending_mu_);
      auto it = fifos_[blockId]->pending.begin();
      while (it != fifos_[blockId]->pending.end()) {
        uint64_t tid = it->first;
        if ((int64_t)(doneBefore - tid) > 0) {
          Pending const& p = it->second;
          switch (p.type) {
            case TaskType::CollCopy:
            case TaskType::CollReduce:
              UKernel::Compute::TaskManager::instance().free_coll_args(
                  p.argsId);
              break;
            case TaskType::MoePreGemm:
            case TaskType::MoePostGemm:
            case TaskType::MoeCombine:
              UKernel::Compute::TaskManager::instance().free_moe_args(p.argsId);
              break;
            case TaskType::BenchNop:
              break;
            default:
              break;
          }
          it = fifos_[blockId]->pending.erase(it);
        } else {
          ++it;
        }
      }
    }
    // doneBefore > taskId + count   (wrap-safe)
    return (int64_t)(doneBefore - (taskId + count)) > 0;
  }

  void stop() {
    if (!launched_) return;
    *h_stopFlag_ = true;

    GPU_RT_CHECK(gpuMemcpyAsync(d_stopFlag_, h_stopFlag_, sizeof(bool),
                                gpuMemcpyHostToDevice, copy_stream_));
    // after launched a persistent kernel, using cudaDeviceSynchronize will
    // block the stream
    GPU_RT_CHECK(gpuStreamSynchronize(copy_stream_));
  };

  gpuStream_t compute_stream() const { return stream_; }
  gpuStream_t copy_stream() const { return copy_stream_; }

 private:
  struct Pending {
    uint32_t argsId;
    UKernel::Compute::TaskType type;
  };
  struct FifoWithPending {
    mscclpp::CpuToGpuFifo<T> fifo;
    std::mutex pending_mu_;
    std::unordered_map<uint64_t, Pending> pending;

    explicit FifoWithPending(uint32_t fifo_capacity) : fifo(fifo_capacity) {}
  };

  PersistentKernelConfig cfg_;
  std::vector<std::unique_ptr<FifoWithPending>>
      fifos_;  // Multi fifos for multi Blocks

  // Mapped memory for stop flag
  bool* d_stopFlag_ = nullptr;  // GPU side stop flag
  bool* h_stopFlag_ = nullptr;  // Host side stop flag
  mscclpp::C2DDeviceHandle<T>* d_fifo_handles_ = nullptr;

  gpuStream_t stream_ = nullptr;       // compute stream（persistent kernel）
  gpuStream_t copy_stream_ = nullptr;  // copy stream（push/stop）
  bool owns_stream_ = false;
  bool launched_ = false;
};
}  // namespace Compute
}  // namespace UKernel
