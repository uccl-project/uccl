#pragma once

#include "gpu_rt.h"
#include <cassert>
#include <cstdint>
#include <mutex>
#include <vector>

namespace UKernel {
namespace Compute {

enum class TaskType : uint64_t {
  // CollTaskType
  CollCopy,
  CollReduce,
  // MoeTaskType
  MoePreGemm,
  MoePostGemm,
  MoeCombine,
  // more Type
  BenchNop,  // for benchmark
};

enum class DataType : uint64_t { Fp8, Fp16, Fp32 };

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

// Coll
enum class ReduceType : uint64_t { Sum, Max, None };

struct alignas(16) CollArgs {
  void* src;
  void* src2;
  void* dst;
  uint32_t bytes;
  ReduceType redType;
  uint8_t _pad0[3];
};
static_assert(sizeof(CollArgs) % 16 == 0,
              "CollArgs should be 16B aligned size");

struct alignas(16) MoeArgs {
  // TODO:
  uint32_t dummy;
  uint32_t _pad[3];
};

class TaskManager {
 public:
  // -------- Singleton entry --------
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

  // init: pre-allocate pools on GPU
  void init(uint32_t collCap, uint32_t moeCap) {
    std::lock_guard<std::mutex> gc(coll_mu_);
    std::lock_guard<std::mutex> gm(moe_mu_);
    release_nolock_();  // re-init

    cap_coll_ = collCap;
    cap_moe_ = moeCap;

    GPU_RT_CHECK(gpuMalloc(&d_coll_, sizeof(CollArgs) * cap_coll_));
    GPU_RT_CHECK(gpuMalloc(&d_moe_, sizeof(MoeArgs) * cap_moe_));

    // host-side freelists
    free_coll_.clear();
    free_moe_.clear();
    free_coll_.reserve(cap_coll_);
    for (uint32_t i = 0; i < cap_coll_; ++i)
      free_coll_.push_back(cap_coll_ - 1 - i);
    free_moe_.reserve(cap_moe_);
    for (uint32_t i = 0; i < cap_moe_; ++i)
      free_moe_.push_back(cap_moe_ - 1 - i);

    inited_ = true;
  }

  // explicit release
  void release() {
    std::lock_guard<std::mutex> gc(coll_mu_);
    std::lock_guard<std::mutex> gm(moe_mu_);
    release_nolock_();
    inited_ = false;
  }

  bool inited() const { return inited_; }

  // CPU: fill coll args (host -> device copy), return idx
  Task create_coll_task(CollArgs const& h, TaskType tt, DataType dt,
                        uint32_t blockId) {
    assert(tt == TaskType::CollCopy || tt == TaskType::CollReduce);

    uint32_t idx;
    {
      std::lock_guard<std::mutex> g(coll_mu_);
      assert(inited_ && "TaskManager not initialized");
      assert(!free_coll_.empty() && "coll args pool exhausted");
      idx = free_coll_.back();
      free_coll_.pop_back();
    }

    GPU_RT_CHECK(
        gpuMemcpy(d_coll_ + idx, &h, sizeof(CollArgs), gpuMemcpyHostToDevice));

    return Task(tt, dt, blockId, idx);
  }

  // CPU: free slot back
  void free_coll_args(uint32_t idx) {
    std::lock_guard<std::mutex> g(coll_mu_);
    assert(inited_ && "TaskManager not initialized");
    assert(idx < cap_coll_ && "free_coll idx out of range");
    free_coll_.push_back(idx);
  }

  // CPU: fill moe args (host -> device copy), return idx
  Task create_moe_task(MoeArgs const& h, TaskType tt, DataType dt,
                       uint32_t blockId) {
    assert(tt == TaskType::MoePreGemm || tt == TaskType::MoePostGemm ||
           tt == TaskType::MoeCombine);

    uint32_t idx;
    {
      std::lock_guard<std::mutex> g(moe_mu_);
      assert(inited_ && "TaskManager not initialized");
      assert(!free_moe_.empty() && "coll args pool exhausted");
      idx = free_moe_.back();
      free_moe_.pop_back();
    }
    GPU_RT_CHECK(
        gpuMemcpy(d_moe_ + idx, &h, sizeof(MoeArgs), gpuMemcpyHostToDevice));
    return Task(tt, dt, blockId, idx);
  }

  // CPU: free slot back
  void free_moe_args(uint32_t idx) {
    std::lock_guard<std::mutex> g(moe_mu_);
    assert(inited_ && "TaskManager not initialized");
    assert(idx < cap_moe_ && "free_moe idx out of range");
    free_moe_.push_back(idx);
  }

  // GPU: get args pointer by index
  __device__ __forceinline__ CollArgs* coll_args(uint32_t idx) const {
    return d_coll_ + idx;
  }

  __device__ __forceinline__ MoeArgs* moe_args(uint32_t idx) const {
    return d_moe_ + idx;
  }

  // Expose device pointers for kernels that need them
  CollArgs* d_coll() const { return d_coll_; }
  MoeArgs* d_moe() const { return d_moe_; }

 private:
  TaskManager() = default;

  void release_nolock_() {
    if (d_coll_) gpuFree(d_coll_);
    if (d_moe_) gpuFree(d_moe_);

    d_coll_ = nullptr;
    d_moe_ = nullptr;

    free_coll_.clear();
    free_moe_.clear();

    cap_coll_ = 0;
    cap_moe_ = 0;
  }

 private:
  // device pools
  CollArgs* d_coll_{nullptr};
  MoeArgs* d_moe_{nullptr};

  uint32_t cap_coll_{0};
  uint32_t cap_moe_{0};

  // host freelists
  std::vector<uint32_t> free_coll_, free_moe_;
  mutable std::mutex coll_mu_;
  mutable std::mutex moe_mu_;
  bool inited_{false};
};
}  // namespace Compute
}  // namespace UKernel