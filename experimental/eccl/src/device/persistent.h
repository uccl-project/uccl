#pragma once

#include "c2d_fifo.h"
#include "operator.h"
#include <vector>
#include <unordered_map>
#include "gpu_rt.h"

namespace eccl {

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
      : cfg_(config), fifo_(config.fifoCapacity) {
    // Allocate memory for stop flag (host and device)
    GPU_RT_CHECK(gpuMalloc(&d_stopFlag_, sizeof(bool)));
    GPU_RT_CHECK(
        gpuHostAlloc(&h_stopFlag_, sizeof(bool), gpuHostAllocMapped));

    // Initialize stop flag to false
    *h_stopFlag_ = false;
    GPU_RT_CHECK(gpuMemcpy(d_stopFlag_, h_stopFlag_, sizeof(bool),
                                 gpuMemcpyHostToDevice));

    // kernel stream
    if (cfg_.stream) {
      stream_ = cfg_.stream;
      owns_stream_ = false;
    } else {
      GPU_RT_CHECK(
          gpuStreamCreateWithFlags(&stream_, gpuStreamNonBlocking));
      owns_stream_ = true;
    }

    // copy stream
    GPU_RT_CHECK(
        gpuStreamCreateWithFlags(&copy_stream_, gpuStreamNonBlocking));
  };

  ~PersistentKernel() noexcept(false) {
    if (launched_) stop();

    GPU_RT_CHECK(gpuFree(d_stopFlag_));
    GPU_RT_CHECK(gpuFreeHost(h_stopFlag_));

    if (copy_stream_) GPU_RT_CHECK(gpuStreamDestroy(copy_stream_));
    if (owns_stream_ && stream_) GPU_RT_CHECK(gpuStreamDestroy(stream_));
  };

  bool launch() {
    if (launched_) return false;

    mscclpp::C2DDeviceHandle<T> handle = fifo_.deviceHandle();
    auto* d_coll = eccl::TaskManager::instance().d_coll();
    auto* d_moe  = eccl::TaskManager::instance().d_moe();
    void* args[] = { &handle, &d_coll, &d_moe, &d_stopFlag_};

    dim3 grid(cfg_.numBlocks);
    dim3 block(cfg_.threadsPerBlock);
    
    GPU_RT_CHECK(gpuLaunchKernel(basePersistentKernel<T>, grid, block,
                                       args, cfg_.smemSize, stream_));

    launched_ = true;
    return true;
  };

  uint64_t submit(const T& task) {
    uint64_t taskId = fifo_.push(task);
    {
      std::lock_guard<std::mutex> g(pending_mu_);
      pending_[taskId] = { task.args_index(), (TaskType)task.type_u8() };
    }
    return taskId;
  };

  uint64_t submitBatch(std::vector<T>& tasks) {
    assert(!tasks.empty());
    uint64_t startTaskId = fifo_.push(tasks.begin(), tasks.end());
    // taskId -> argsId / type
    {
      std::lock_guard<std::mutex> g(pending_mu_);
      uint64_t taskId = startTaskId;
      for (const auto& task : tasks) {
        pending_.emplace(
            taskId,
            Pending{
                task.args_index(),
                static_cast<TaskType>(task.type_u8())
            }
        );
        ++taskId;
      }
    }
    return startTaskId;
  }


  bool is_done(uint64_t taskId, size_t count = 0) {
  uint64_t doneBefore = fifo_.currentId();
  {
    std::lock_guard<std::mutex> g(pending_mu_);
    auto it = pending_.begin();
    while (it != pending_.end()) {
      uint64_t tid = it->first;
      if (tid < doneBefore) {
        const Pending& p = it->second;
        switch (p.type) {
          case TaskType::CollCopy:
          case TaskType::CollReduce:
            eccl::TaskManager::instance().free_coll_args(p.argsId);
            break;
          case TaskType::MoePreGemm:
          case TaskType::MoePostGemm:
          case TaskType::MoeCombine:
            eccl::TaskManager::instance().free_moe_args(p.argsId);
            break;
          default:
            break;
        }
        it = pending_.erase(it);
      } else {
        ++it;
      }
    }
  }
  return doneBefore > (taskId + count);
}

  void stop() {
    if (!launched_) return;
    *h_stopFlag_ = true;

    GPU_RT_CHECK(gpuMemcpyAsync(d_stopFlag_, h_stopFlag_, sizeof(bool),
                                      gpuMemcpyHostToDevice, copy_stream_));
    // after launched a persistent kernel, using cudaDeviceSynchronize will block the stream
    GPU_RT_CHECK(gpuStreamSynchronize(copy_stream_));
  };

  gpuStream_t compute_stream() const { return stream_; }
  gpuStream_t copy_stream() const { return copy_stream_; }

 private:
  PersistentKernelConfig cfg_;
  mscclpp::CpuToGpuFifo<T> fifo_; // TODO: multi fifos for multi Thread Blocks

  // Mapped memory for stop flag
  bool* d_stopFlag_ = nullptr;  // GPU side stop flag
  bool* h_stopFlag_ = nullptr;  // Host side stop flag

  struct Pending {
    uint32_t argsId;
    eccl::TaskType type;
  };

  std::mutex pending_mu_;
  std::unordered_map<uint64_t, Pending> pending_; // 

  gpuStream_t stream_ = nullptr;       // compute stream（persistent kernel）
  gpuStream_t copy_stream_ = nullptr;  // copy stream（push/stop）
  bool owns_stream_ = false;
  bool launched_ = false;
};

}  // namespace eccl
