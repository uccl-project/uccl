#pragma once

#include "gpu_rt.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <vector>

namespace UKernel {
namespace Device {

enum class TaskType : uint64_t {
  CollCopy,
  CollReduce,
  BenchNop,
  Stop,
};

enum class DataType : uint64_t { Int8, Int32, Int64, Fp8, Fp16, Fp32, Fp64, Bf16 };
enum class ReduceType : uint64_t { Sum, Prod, Max, Min, BitwiseAnd, None };

constexpr unsigned int TaskTypeSize = 8;  // 256
constexpr unsigned int DataTypeSize = 8;
constexpr unsigned int BlockIdSize = 8;
constexpr unsigned int TaskArgsIndexSize = 32;  // Id to Task Args sturct

/// Pair of 64-bit unsigned integers used as a Task.
/// Used as a work element in the concurrent FIFO.
union alignas(16) Task {
  struct {
    uint64_t fst;
    uint64_t snd;
  };

  Task() = default;

  struct {
    uint64_t type : TaskTypeSize;
    uint64_t dataType : DataTypeSize;
    uint64_t blockId : BlockIdSize;
    uint64_t : (64 - TaskTypeSize - DataTypeSize - BlockIdSize);
    uint64_t argsId : TaskArgsIndexSize;
    uint64_t : (64 - TaskArgsIndexSize);
  } fields;

  /// Constructor.
  /// @param type The type of the Task.
  /// @param dType The type of Data.
  /// @param blockIndex Which block the task will be dispatched to.
  /// @param argsIndex The Args Id of Task (in TaskManager).
  __host__ __device__ Task(TaskType type, DataType dType, uint32_t blockIndex,
                           uint32_t argsIndex) {
    const uint64_t t = static_cast<uint64_t>(type);
    const uint64_t dt = static_cast<uint64_t>(dType);
    const uint64_t bi = static_cast<uint64_t>(blockIndex);
    const uint64_t ai = static_cast<uint64_t>(argsIndex);

    assert(t < (1ULL << TaskTypeSize));
    assert(dt < (1ULL << DataTypeSize));
    assert(bi < (1ULL << BlockIdSize));
    assert(ai < (1ULL << TaskArgsIndexSize));

    constexpr uint64_t maskType = (1ULL << TaskTypeSize) - 1;
    constexpr uint64_t maskDType = (1ULL << DataTypeSize) - 1;
    constexpr uint64_t maskBlockId = (1ULL << BlockIdSize) - 1;
    constexpr uint64_t maskArgs = (1ULL << TaskArgsIndexSize) - 1;

    fst = (t & maskType) | ((dt & maskDType) << TaskTypeSize) |
          ((bi & maskBlockId) << (TaskTypeSize + DataTypeSize));

    snd = (ai & maskArgs);
  }

  __host__ __device__ uint8_t type_u8() const { return uint8_t(fst & 0xFFull); }
  __host__ __device__ uint8_t dtype_u8() const {
    return uint8_t((fst >> 8) & 0xFFull);
  }
  __host__ __device__ uint32_t block_index() const {
    return uint32_t((fst >> (TaskTypeSize + DataTypeSize)) &
                    ((1ULL << BlockIdSize) - 1));
  }
  __host__ __device__ uint32_t args_index() const {
    return uint32_t(snd & 0xFFFFFFFFull);
  }
};
static_assert(sizeof(Task) == 16);

struct alignas(16) TaskArgs {
  void* src;
  void* src2;
  void* dst;
  uint64_t bytes;
  int32_t src_rank;
  int32_t dst_rank;
  int32_t src_device;
  int32_t dst_device;
  ReduceType redType;
  uint32_t flags;
  uint32_t pad;
};
static_assert(sizeof(TaskArgs) % 16 == 0,
              "TaskArgs should be 16B aligned size");

class TaskManager {
 public:
  // Singleton entry
  static TaskManager& instance() {
    static TaskManager inst;
    return inst;
  }
  // forbid copy/move
  TaskManager(TaskManager const&) = delete;
  TaskManager& operator=(TaskManager const&) = delete;
  TaskManager(TaskManager&&) = delete;
  TaskManager& operator=(TaskManager&&) = delete;

  ~TaskManager() { release(); }

  void init(uint32_t Cap) {
    std::lock_guard<std::mutex> gc(task_mu_);
    release_nolock_();

    cap_task_ = Cap;

    GPU_RT_CHECK(gpuMalloc(&d_task_, sizeof(TaskArgs) * cap_task_));

    free_task_.clear();
    free_task_.reserve(cap_task_);
    task_in_use_.assign(cap_task_, 0);
    for (uint32_t i = 0; i < cap_task_; ++i)
      free_task_.push_back(cap_task_ - 1 - i);

    inited_ = true;
  }

  void release() {
    std::lock_guard<std::mutex> gc(task_mu_);
    release_nolock_();
    inited_ = false;
  }

  bool inited() const { return inited_; }

  Task create_task(TaskArgs const& h, TaskType tt, DataType dt,
                        uint32_t blockId) {
    assert(tt == TaskType::CollCopy || tt == TaskType::CollReduce);

    uint32_t idx;
    {
      std::lock_guard<std::mutex> g(task_mu_);
      assert(inited_ && "TaskManager not initialized");
      assert(!free_task_.empty() && "args pool exhausted");
      idx = free_task_.back();
      free_task_.pop_back();
      assert(task_in_use_[idx] == 0 && "Task args slot already in use");
      task_in_use_[idx] = 1;
    }

    GPU_RT_CHECK(gpuMemcpy(d_task_ + idx, &h, sizeof(TaskArgs),
                           gpuMemcpyHostToDevice));

    return Task(tt, dt, blockId, idx);
  }

  void free_task_args(uint32_t idx) {
    std::lock_guard<std::mutex> g(task_mu_);
    assert(inited_ && "TaskManager not initialized");
    assert(idx < cap_task_ && "free_task_args idx out of range");
    assert(task_in_use_[idx] == 1 && "double free on task args slot");
    task_in_use_[idx] = 0;
    free_task_.push_back(idx);
  }

  // GPU: get args pointer by index
  __device__ __forceinline__ TaskArgs* task_args(uint32_t idx) const {
    return d_task_ + idx;
  }

  TaskArgs* d_task_args() const { return d_task_; }

 private:
  TaskManager() = default;

  void release_nolock_() {
    if (d_task_) gpuFree(d_task_);
    d_task_ = nullptr;

    free_task_.clear();
    task_in_use_.clear();

    cap_task_ = 0;
    inited_ = false;
  }

  TaskArgs* d_task_{nullptr};

  uint32_t cap_task_{0};

  std::vector<uint32_t> free_task_;
  std::vector<uint8_t> task_in_use_;
  mutable std::mutex task_mu_;
  bool inited_{false};
};
}  // namespace Device
}  // namespace UKernel
