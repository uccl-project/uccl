#pragma once

#include "gpu_rt.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <type_traits>
#include <vector>
#ifndef __CUDA_ARCH__
#include "fifo/fifo_gdrcopy.hpp"
namespace Gdr = mscclpp::detail;
#endif

namespace UKernel {
namespace Device {

enum class TaskType : uint64_t {
  None = 0,      // sentinel: empty/uninitialized task
  CollCopy = 1,  // pure GPU copy (used by tests/benchmarks)
  CollPut = 2,   // GPU copy + signal ring write (used by CCL fused PutSignal)
  CollReduce,    // 3 — local reduction
  BenchNop,
  Stop,
};

enum class DataType : uint64_t {
  Int8,
  Int32,
  Int64,
  Fp8,
  Fp16,
  Fp32,
  Fp64,
  Bf16
};
enum class ReduceType : uint64_t {
  None,
  Sum,
  Prod,
  Max,
  Min,
  BitwiseAnd,
};

inline bool is_supported_reduce_dtype(DataType dt) {
  return dt == DataType::Int8 || dt == DataType::Int32 ||
         dt == DataType::Int64 || dt == DataType::Fp16 ||
         dt == DataType::Fp32 || dt == DataType::Fp64 || dt == DataType::Bf16;
}

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
  static constexpr uint64_t kPublishedMagic = 0x554b544152475331ull;
  // Fused out-of-place reduce: write dst = src op src2 (fresh) instead
  // of dst = dst op src. src is the peer's buffer, src2 the local Input
  // contribution.
  static constexpr uint64_t kFlagReduce3Way = 1ull << 0;
  // Fused reduce+copy: after the reduce, copy dst -> dst2 (peer's
  // accumulation buffer) and, when kFlagSignalAfter is set, write the
  // signal tag (redTypeRaw) into the peer's ring (src2).
  static constexpr uint64_t kFlagReduceCopy = 1ull << 1;
  static constexpr uint64_t kFlagSignalAfter = 1ull << 2;

  void* src;
  void* src2;
  void* dst;
  void* dst2;
  uint64_t bytes;
  int32_t src_rank;
  int32_t dst_rank;
  int32_t src_device;
  int32_t dst_device;
  uint64_t redTypeRaw = static_cast<uint64_t>(ReduceType::None);
  // Fused-task completion signal tag (device flag write). Separate from
  // redTypeRaw: CollReduce needs redTypeRaw for the reduction, and the
  // tag (slot index) can collide with ReduceType::None in its low byte.
  uint64_t signal_tag = 0;
  uint64_t taskFlags = 0;
  uint64_t reserved0 = 0;

  __host__ __device__ ReduceType red_type() const {
    return static_cast<ReduceType>(redTypeRaw);
  }

  __host__ __device__ void set_red_type(ReduceType type) {
    redTypeRaw = static_cast<uint64_t>(type);
  }

  __host__ __device__ bool reduce_3way() const {
    return (taskFlags & kFlagReduce3Way) != 0;
  }

  __host__ __device__ bool reduce_copy() const {
    return (taskFlags & kFlagReduceCopy) != 0;
  }

  __host__ __device__ bool signal_after() const {
    return (taskFlags & kFlagSignalAfter) != 0;
  }

  __host__ __device__ bool is_published() const {
    return reserved0 == kPublishedMagic;
  }
};
static_assert(sizeof(TaskArgs) % 16 == 0,
              "TaskArgs should be 16B aligned size");
static_assert(std::is_standard_layout<TaskArgs>::value,
              "TaskArgs must remain a standard-layout ABI struct");
static_assert(sizeof(TaskArgs) == 96, "TaskArgs ABI size changed");
static_assert(alignof(TaskArgs) == 16, "TaskArgs ABI alignment changed");
static_assert(offsetof(TaskArgs, src) == 0, "TaskArgs.src offset changed");
static_assert(offsetof(TaskArgs, src2) == 8, "TaskArgs.src2 offset changed");
static_assert(offsetof(TaskArgs, dst) == 16, "TaskArgs.dst offset changed");
static_assert(offsetof(TaskArgs, dst2) == 24, "TaskArgs.dst2 offset changed");
static_assert(offsetof(TaskArgs, bytes) == 32, "TaskArgs.bytes offset changed");
static_assert(offsetof(TaskArgs, src_rank) == 40,
              "TaskArgs.src_rank offset changed");
static_assert(offsetof(TaskArgs, dst_rank) == 44,
              "TaskArgs.dst_rank offset changed");
static_assert(offsetof(TaskArgs, src_device) == 48,
              "TaskArgs.src_device offset changed");
static_assert(offsetof(TaskArgs, dst_device) == 52,
              "TaskArgs.dst_device offset changed");
static_assert(offsetof(TaskArgs, redTypeRaw) == 56,
              "TaskArgs.redTypeRaw offset changed");
static_assert(offsetof(TaskArgs, signal_tag) == 64,
              "TaskArgs.signal_tag offset changed");
static_assert(offsetof(TaskArgs, taskFlags) == 72,
              "TaskArgs.taskFlags offset changed");
static_assert(offsetof(TaskArgs, reserved0) == 80,
              "TaskArgs.reserved0 offset changed");

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

  void init(uint32_t Cap) { init_impl(Cap, false); }
  void init_no_gdr(uint32_t Cap) { init_impl(Cap, true); }

 private:
  void init_impl(uint32_t Cap, bool no_gdr) {
    std::lock_guard<std::mutex> gc(task_mu_);
    release_nolock_();

    cap_task_ = Cap;

#ifndef __CUDA_ARCH__
    if (no_gdr) {
      GPU_RT_CHECK(gpuMalloc(&d_task_, sizeof(TaskArgs) * cap_task_));
    } else {
      gdr_task_ = Gdr::gpuCallocGdrUnique<TaskArgs>(Cap);
      d_task_ = gdr_task_.get();
      host_task_ = Gdr::getGdrHostPtr(gdr_task_);
    }
#else
    GPU_RT_CHECK(gpuMalloc(&d_task_, sizeof(TaskArgs) * cap_task_));
#endif

    free_task_.clear();
    free_task_.reserve(cap_task_);
    task_in_use_.assign(cap_task_, 0);
    for (uint32_t i = 0; i < cap_task_; ++i)
      free_task_.push_back(cap_task_ - 1 - i);

#ifndef __CUDA_ARCH__
    fprintf(stderr, "[TaskManager] init done: cap=%u free=%zu host=%p\n",
            cap_task_, free_task_.size(), (void*)host_task_);
#endif
    inited_ = true;
  }

 public:
  void release() {
    std::lock_guard<std::mutex> gc(task_mu_);
    release_nolock_();
    inited_ = false;
  }

  bool inited() const { return inited_; }

  Task create_task(TaskArgs const& h, TaskType tt, DataType dt,
                   uint32_t blockId) {
    assert(tt == TaskType::CollCopy || tt == TaskType::CollReduce ||
           tt == TaskType::CollPut);
    bool is_reduce = (tt == TaskType::CollReduce);
    assert(!is_reduce || is_supported_reduce_dtype(dt));
    if (is_reduce) {
      uint8_t red = static_cast<uint8_t>(h.redTypeRaw & 0xFF);
      assert(red != static_cast<uint8_t>(ReduceType::None) &&
             "SM IPC reduce requires non-None reduction");
    }

    uint32_t idx;
    {
      std::lock_guard<std::mutex> g(task_mu_);
      assert(inited_ && "TaskManager not initialized");
      if (free_task_.empty()) {
        fprintf(
            stderr,
            "[TaskManager] create_task: POOL EMPTY cap=%u free=%zu inited=%d\n",
            cap_task_, free_task_.size(), (int)inited_);
        return Task();
      }
      idx = free_task_.back();
      free_task_.pop_back();
      assert(task_in_use_[idx] == 0 && "Task args slot already in use");
      task_in_use_[idx] = 1;
    }

    TaskArgs staged = h;
    staged.reserved0 = TaskArgs::kPublishedMagic;
#ifndef __CUDA_ARCH__
    if (host_task_) {
      host_task_[idx] = staged;
    } else {
      GPU_RT_CHECK(gpuMemcpy(d_task_ + idx, &staged, sizeof(TaskArgs),
                             gpuMemcpyHostToDevice));
    }
#else
    GPU_RT_CHECK(gpuMemcpy(d_task_ + idx, &staged, sizeof(TaskArgs),
                           gpuMemcpyHostToDevice));
#endif

    return Task(tt, dt, blockId, idx);
  }

  void free_task_args(uint32_t idx) { free_task_args_batch(&idx, 1); }

  void free_task_args_batch(uint32_t const* idxs, size_t n) {
    if (n == 0) return;
    std::lock_guard<std::mutex> g(task_mu_);
    assert(inited_ && "TaskManager not initialized");
    for (size_t i = 0; i < n; ++i) {
      uint32_t idx = idxs[i];
      assert(idx < cap_task_ && "free_task_args idx out of range");
      if (task_in_use_[idx] == 0) {
        std::fprintf(
            stderr, "[TaskManager] WARNING: double free on task args slot %u\n",
            idx);
        continue;
      }
      task_in_use_[idx] = 0;
      free_task_.push_back(idx);
      // Clear the publish marker on GPU so the slot is not seen as
      // published. On the GDR path this is a plain host store into the
      // mapped TaskArgs array — avoid a synchronous gpuMemcpy per
      // completion (it syncs with the device and stalls the drain path).
#ifndef __CUDA_ARCH__
      if (host_task_) {
        host_task_[idx].reserved0 = 0;
      } else
#endif
      {
        uint64_t zero = 0;
        GPU_RT_CHECK(gpuMemcpy(&d_task_[idx].reserved0, &zero, sizeof(zero),
                               gpuMemcpyHostToDevice));
      }
    }
  }

  // GPU: get args pointer by index
  __device__ __forceinline__ TaskArgs* task_args(uint32_t idx) const {
    return d_task_ + idx;
  }

  TaskArgs* d_task_args() const { return d_task_; }

 private:
  TaskManager() = default;

  void release_nolock_() {
#ifndef __CUDA_ARCH__
    gdr_task_.reset();
    if (d_task_ && !host_task_) gpuFree(d_task_);
    d_task_ = nullptr;
    host_task_ = nullptr;
#else
    if (d_task_) gpuFree(d_task_);
    d_task_ = nullptr;
#endif

    free_task_.clear();
    task_in_use_.clear();

    cap_task_ = 0;
    inited_ = false;
  }

  TaskArgs* d_task_{nullptr};
#ifndef __CUDA_ARCH__
  Gdr::UniqueGdrGpuPtr<TaskArgs> gdr_task_;
  TaskArgs* host_task_{nullptr};
#endif

  uint32_t cap_task_{0};

  std::vector<uint32_t> free_task_;
  std::vector<uint8_t> task_in_use_;
  mutable std::mutex task_mu_;
  bool inited_{false};
};
}  // namespace Device
}  // namespace UKernel
