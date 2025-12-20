#pragma once

#include "c2d_fifo.h"
#include "operator.h"
#include <cuda_runtime.h>
#include <vector>

namespace eccl {

constexpr uint64_t kAbortTailValue = (uint64_t)-2;

struct PersistentKernelConfig {
  uint32_t numBlocks = 1;
  uint32_t threadsPerBlock = 64; // assume that warpsize is 32
  uint32_t fifoCapacity = 16;
  uint32_t smemSize = 0;

  cudaStream_t stream = nullptr; // if user want to manage the stream
};

template <typename T>
class PersistentKernel {
 public:
  explicit PersistentKernel(const PersistentKernelConfig& config)
    : cfg_(config), fifo_(config.fifoCapacity) {
    // Allocate memory for stop flag (host and device)
    MSCCLPP_CUDATHROW(cudaMalloc(&d_stopFlag_, sizeof(bool)));
    MSCCLPP_CUDATHROW(cudaHostAlloc(&h_stopFlag_, sizeof(bool), cudaHostAllocMapped));

    // Initialize stop flag to false
    *h_stopFlag_ = false;
    MSCCLPP_CUDATHROW(cudaMemcpy(d_stopFlag_, h_stopFlag_, sizeof(bool), cudaMemcpyHostToDevice));

    // kernel stream
    if (cfg_.stream) {
      stream_ = cfg_.stream;
      owns_stream_ = false;
    } else {
      MSCCLPP_CUDATHROW(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
      owns_stream_ = true;
    }

    // copy stream
    MSCCLPP_CUDATHROW(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));
  };
  
  ~PersistentKernel() noexcept(false) {
    if (launched_) stop();

    MSCCLPP_CUDATHROW(cudaFree(d_stopFlag_));
    MSCCLPP_CUDATHROW(cudaFreeHost(h_stopFlag_));

    if (copy_stream_) MSCCLPP_CUDATHROW(cudaStreamDestroy(copy_stream_));
    if (owns_stream_ && stream_) MSCCLPP_CUDATHROW(cudaStreamDestroy(stream_));
  };

  bool launch(){
    if (launched_) return false;

    mscclpp::C2DDeviceHandle<T> handle = fifo_.deviceHandle();
    void* args[] = {&handle, &d_stopFlag_};

    dim3 grid(cfg_.numBlocks);
    dim3 block(cfg_.threadsPerBlock);

    MSCCLPP_CUDATHROW(cudaLaunchKernel(
        basePersistentKernel<T>,
        grid, block,
        args,
        cfg_.smemSize,
        stream_
    ));

    launched_ = true;
    return true;
  };

  uint64_t submit(const T& task) {
    uint64_t taskId = fifo_.push(task);
    return taskId;
  };

  uint64_t submitBatch(std::vector<T>& tasks) {
    // Push a batch of tasks to FIFO and return the start task ID
    uint64_t startTaskId = fifo_.push(tasks.begin(), tasks.end());
    return startTaskId;
  }; // return start_id

  bool is_done(uint64_t startTaskId, size_t count = 0) const {
    // Check if the tasks from startTaskId to startTaskId + count are completed
    // fifo_.sync(startTaskId + count);
    return fifo_.poll(startTaskId + count);
  };  // true if tail > slotIdx

  void stop() {
    if (!launched_) return;
    *h_stopFlag_ = true;

    MSCCLPP_CUDATHROW(cudaMemcpyAsync(
        d_stopFlag_, h_stopFlag_, sizeof(bool),
        cudaMemcpyHostToDevice, copy_stream_));

    MSCCLPP_CUDATHROW(cudaStreamSynchronize(copy_stream_));
  };

  cudaStream_t compute_stream() const { return stream_; }
  cudaStream_t copy_stream() const { return copy_stream_; }

 private:
  PersistentKernelConfig cfg_;
  mscclpp::CpuToGpuFifo<T> fifo_;

  // Mapped memory for stop flag
  bool* d_stopFlag_ = nullptr;  // GPU side stop flag
  bool* h_stopFlag_ = nullptr;  // Host side stop flag

  cudaStream_t stream_ = nullptr;       // compute stream（persistent kernel）
  cudaStream_t copy_stream_ = nullptr;  // copy stream（push/stop）
  bool owns_stream_ = false;
  bool launched_ = false;
};

} // eccl
