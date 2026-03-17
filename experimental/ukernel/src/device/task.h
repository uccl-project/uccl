#pragma once

#include "gpu_rt.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace UKernel {
namespace Device {

#ifndef UKERNEL_ENABLE_REGISTER_OP
#define UKERNEL_ENABLE_REGISTER_OP 1
#endif

#ifndef UKERNEL_ENABLE_TMA
#define UKERNEL_ENABLE_TMA 0
#endif

enum class TaskType : uint64_t {
  // CollTaskType
  CollCopy,
  CollReduce,
  // TK GEMM
  TkGemm,
  // more Type
  BenchNop,  // for benchmark
};

enum class DataType : uint64_t { Fp8, Fp16, Fp32 };
enum class TransferPath : uint32_t { Auto, RegisterOp, TmaOp };

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

struct TransferCapabilities {
  bool compiled_with_tma = false;
  bool runtime_has_tma = false;
  bool can_use_tma = false;
};

struct PkSelectorConfig {
  bool enable_auto_transport = true;
  uint64_t tma_threshold_bytes = 16 * 1024;
};

inline constexpr bool compiled_with_tma_support() {
#if defined(__HIP_PLATFORM_AMD__)
  return false;
#else
  return UKERNEL_ENABLE_TMA != 0;
#endif
}

inline bool device_supports_tma(int device = -1) {
  if (!compiled_with_tma_support()) return false;
#if defined(__HIP_PLATFORM_AMD__)
  (void)device;
  return false;
#else
  int current_device = 0;
  if (device < 0) {
    if (gpuGetDevice(&current_device) != gpuSuccess) return false;
    device = current_device;
  }

#ifdef CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED
  gpuDrvDevice_t drv_device{};
  if (gpuDrvDeviceGet(&drv_device, device) == gpuDrvSuccess) {
    int supported = 0;
    if (gpuDrvDeviceGetAttribute(
            &supported, CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED,
            drv_device) == gpuDrvSuccess) {
      return supported != 0;
    }
  }
#endif

  gpuDeviceProp prop{};
  if (gpuGetDeviceProperties(&prop, device) != gpuSuccess) return false;
  return prop.major >= 9;
#endif
}

inline TransferCapabilities query_transfer_capabilities(int device = -1) {
  TransferCapabilities caps{};
  caps.compiled_with_tma = compiled_with_tma_support();
  caps.runtime_has_tma = device_supports_tma(device);
  caps.can_use_tma = caps.compiled_with_tma && caps.runtime_has_tma;
  return caps;
}

TransferPath resolve_pk_transfer_path(TransferPath requested, uint64_t bytes,
                                      TransferCapabilities const& caps,
                                      PkSelectorConfig const& cfg);

inline __host__ __device__ bool is_pk_transfer_path(TransferPath path) {
  return path == TransferPath::Auto || path == TransferPath::RegisterOp ||
         path == TransferPath::TmaOp;
}

inline __host__ __device__ TransferPath normalize_pk_transfer_path(
    TransferPath requested) {
  if (requested == TransferPath::Auto) return TransferPath::RegisterOp;
#if UKERNEL_ENABLE_TMA
  return requested;
#else
  return requested == TransferPath::TmaOp ? TransferPath::RegisterOp
                                          : requested;
#endif
}

inline size_t data_type_size(DataType dtype) {
  switch (dtype) {
    case DataType::Fp16:
      return 2;
    case DataType::Fp32:
      return 4;
    case DataType::Fp8:
    default:
      return 1;
  }
}

inline bool supports_native_tma_copy(DataType dtype, uint64_t bytes,
                                     void const* src = nullptr,
                                     void const* dst = nullptr) {
  if (!query_transfer_capabilities().can_use_tma) return false;
  if (dtype != DataType::Fp16 && dtype != DataType::Fp32) return false;
  size_t elem_bytes = data_type_size(dtype);
  if (bytes == 0 || bytes % elem_bytes != 0) return false;
  uint64_t elems = bytes / elem_bytes;
  if (elems < 16 || (elems % 16) != 0) return false;
  if (src != nullptr && (reinterpret_cast<uintptr_t>(src) & 0xF) != 0) {
    return false;
  }
  if (dst != nullptr && (reinterpret_cast<uintptr_t>(dst) & 0xF) != 0) {
    return false;
  }
  return true;
}

inline uint32_t select_tma_chunk_elements(DataType dtype, uint64_t bytes) {
  if (!supports_native_tma_copy(dtype, bytes)) return 0;
  uint64_t elems = bytes / data_type_size(dtype);
  for (uint32_t candidate = static_cast<uint32_t>(elems > 256 ? 256 : elems);
       candidate >= 16; candidate -= 16) {
    if ((elems % candidate) == 0) return candidate;
  }
  return 16;
}

struct alignas(16) CollArgs {
  void* src;
  void* src2;
  void* dst;
  void* src_tma_desc;
  void* dst_tma_desc;
  uint64_t bytes;
  uint64_t op_id;
  uint32_t step_id;
  uint32_t chunk_id;
  uint32_t completion_cookie;
  uint32_t tma_chunk_elements;
  int32_t src_rank;
  int32_t dst_rank;
  int32_t src_device;
  int32_t dst_device;
  uint32_t flags;
  ReduceType redType;
  TransferPath requested_path;
  TransferPath resolved_path;
};
static_assert(sizeof(CollArgs) % 16 == 0,
              "CollArgs should be 16B aligned size");

// TK GEMM args: pointer to pre-constructed TkMatmulGlobals (with TMA
// descriptors).
// [TODO: Yihan] perf tuning for GEMM later, change the args if more info needed
struct alignas(16) GemmArgs {
  void* globals;                          // TkMatmulGlobals* on device
  uint32_t num_tile_rows, num_tile_cols;  // total tiles for multi-block
  uint32_t _pad;
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

  void init(uint32_t collCap, uint32_t gemmCap = 64) {
    std::lock_guard<std::mutex> gc(coll_mu_);
    std::lock_guard<std::mutex> gg(gemm_mu_);
    release_nolock_();

    cap_coll_ = collCap;
    cap_gemm_ = gemmCap;

    GPU_RT_CHECK(gpuMalloc(&d_coll_, sizeof(CollArgs) * cap_coll_));
    GPU_RT_CHECK(gpuMalloc(&d_gemm_, sizeof(GemmArgs) * cap_gemm_));
#if !defined(__HIP_PLATFORM_AMD__)
    if (compiled_with_tma_support()) {
      tma_desc_stride_ = sizeof(CUtensorMap);
      GPU_RT_CHECK(gpuMalloc(&d_tma_src_descs_, tma_desc_stride_ * cap_coll_));
      GPU_RT_CHECK(gpuMalloc(&d_tma_dst_descs_, tma_desc_stride_ * cap_coll_));
    }
#endif

    free_coll_.clear();
    free_gemm_.clear();
    free_coll_.reserve(cap_coll_);
    for (uint32_t i = 0; i < cap_coll_; ++i)
      free_coll_.push_back(cap_coll_ - 1 - i);
    free_gemm_.reserve(cap_gemm_);
    for (uint32_t i = 0; i < cap_gemm_; ++i)
      free_gemm_.push_back(cap_gemm_ - 1 - i);

    inited_ = true;
  }

  void release() {
    std::lock_guard<std::mutex> gc(coll_mu_);
    std::lock_guard<std::mutex> gg(gemm_mu_);
    release_nolock_();
    inited_ = false;
  }

  bool inited() const { return inited_; }

  // CPU: fill coll args (host -> device copy), return idx
  Task create_coll_task(CollArgs const& h, TaskType tt, DataType dt,
                        uint32_t blockId) {
    assert(tt == TaskType::CollCopy || tt == TaskType::CollReduce);
    assert(is_pk_transfer_path(h.requested_path) &&
           "coll task must use a persistent-kernel transfer path");

    uint32_t idx;
    {
      std::lock_guard<std::mutex> g(coll_mu_);
      assert(inited_ && "TaskManager not initialized");
      assert(!free_coll_.empty() && "coll args pool exhausted");
      idx = free_coll_.back();
      free_coll_.pop_back();
    }

    CollArgs normalized = h;
    normalized.src_tma_desc = nullptr;
    normalized.dst_tma_desc = nullptr;
    normalized.tma_chunk_elements = 0;
    normalized.resolved_path = resolve_pk_transfer_path(
        h.requested_path, h.bytes, query_transfer_capabilities(),
        PkSelectorConfig{});

    if (normalized.resolved_path == TransferPath::TmaOp &&
        tt != TaskType::CollCopy) {
      normalized.resolved_path = TransferPath::RegisterOp;
    } else if (normalized.resolved_path == TransferPath::TmaOp &&
               tt == TaskType::CollCopy) {
      if (!populate_coll_tma_descriptors_(idx, normalized, dt)) {
        normalized.resolved_path = TransferPath::RegisterOp;
        normalized.src_tma_desc = nullptr;
        normalized.dst_tma_desc = nullptr;
        normalized.tma_chunk_elements = 0;
      }
    } else {
      normalized.resolved_path =
          normalize_pk_transfer_path(normalized.resolved_path);
    }

    GPU_RT_CHECK(gpuMemcpy(d_coll_ + idx, &normalized, sizeof(CollArgs),
                           gpuMemcpyHostToDevice));

    return Task(tt, dt, blockId, idx);
  }

  // CPU: free slot back
  void free_coll_args(uint32_t idx) {
    std::lock_guard<std::mutex> g(coll_mu_);
    assert(inited_ && "TaskManager not initialized");
    assert(idx < cap_coll_ && "free_coll idx out of range");
    free_coll_.push_back(idx);
  }

  // tk level-8 gemm task creation
  Task create_gemm_task(GemmArgs const& h, DataType dt, uint32_t blockId) {
    uint32_t idx;
    {
      std::lock_guard<std::mutex> g(gemm_mu_);
      assert(inited_ && "TaskManager not initialized");
      assert(!free_gemm_.empty() && "gemm args pool exhausted");
      idx = free_gemm_.back();
      free_gemm_.pop_back();
    }
    GPU_RT_CHECK(
        gpuMemcpy(d_gemm_ + idx, &h, sizeof(GemmArgs), gpuMemcpyHostToDevice));
    return Task(TaskType::TkGemm, dt, blockId, idx);
  }

  void free_gemm_args(uint32_t idx) {
    std::lock_guard<std::mutex> g(gemm_mu_);
    assert(inited_ && "TaskManager not initialized");
    assert(idx < cap_gemm_ && "free_gemm idx out of range");
    free_gemm_.push_back(idx);
  }

  // GPU: get args pointer by index
  __device__ __forceinline__ CollArgs* coll_args(uint32_t idx) const {
    return d_coll_ + idx;
  }

  __device__ __forceinline__ GemmArgs* gemm_args(uint32_t idx) const {
    return d_gemm_ + idx;
  }

  CollArgs* d_coll() const { return d_coll_; }
  GemmArgs* d_gemm() const { return d_gemm_; }

 private:
  TaskManager() = default;

  static CUtensorMapDataType to_cu_tma_dtype_(DataType dt) {
#if defined(__HIP_PLATFORM_AMD__)
    (void)dt;
    throw std::runtime_error("TMA descriptors are unavailable on HIP");
#else
    switch (dt) {
      case DataType::Fp16:
        return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
      case DataType::Fp32:
        return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
      case DataType::Fp8:
      default:
        throw std::runtime_error("unsupported dtype for TMA copy");
    }
#endif
  }

  static bool encode_vector_tma_descriptor_(void* device_slot, void const* ptr,
                                            DataType dt, uint64_t bytes,
                                            uint32_t chunk_elements) {
#if defined(__HIP_PLATFORM_AMD__)
    (void)device_slot;
    (void)ptr;
    (void)dt;
    (void)bytes;
    (void)chunk_elements;
    return false;
#else
    if (device_slot == nullptr || ptr == nullptr || chunk_elements == 0) {
      return false;
    }

    size_t elem_bytes = data_type_size(dt);
    uint64_t elements = bytes / elem_bytes;
    if (elements == 0) return false;

    CUtensorMap desc{};
    uint64_t gmem_shape[4] = {elements, 1, 1, 1};
    uint64_t gmem_stride[3] = {elements * elem_bytes, elements * elem_bytes,
                               elements * elem_bytes};
    uint32_t smem_shape[4] = {chunk_elements, 1, 1, 1};
    uint32_t smem_stride[4] = {1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &desc, to_cu_tma_dtype_(dt), 4, const_cast<void*>(ptr), gmem_shape,
        gmem_stride, smem_shape, smem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result != CUDA_SUCCESS) return false;

    return gpuMemcpy(device_slot, &desc, sizeof(desc), gpuMemcpyHostToDevice) ==
           gpuSuccess;
#endif
  }

  bool populate_coll_tma_descriptors_(uint32_t idx, CollArgs& args, DataType dt) {
#if defined(__HIP_PLATFORM_AMD__)
    (void)idx;
    (void)args;
    (void)dt;
    return false;
#else
    if (d_tma_src_descs_ == nullptr || d_tma_dst_descs_ == nullptr) return false;
    if (!supports_native_tma_copy(dt, args.bytes, args.src, args.dst)) {
      return false;
    }
    uint32_t chunk_elements = select_tma_chunk_elements(dt, args.bytes);
    if (chunk_elements == 0) return false;

    auto* src_slot = static_cast<char*>(d_tma_src_descs_) + idx * tma_desc_stride_;
    auto* dst_slot = static_cast<char*>(d_tma_dst_descs_) + idx * tma_desc_stride_;
    if (!encode_vector_tma_descriptor_(src_slot, args.src, dt, args.bytes,
                                       chunk_elements)) {
      return false;
    }
    if (!encode_vector_tma_descriptor_(dst_slot, args.dst, dt, args.bytes,
                                       chunk_elements)) {
      return false;
    }

    args.src_tma_desc = src_slot;
    args.dst_tma_desc = dst_slot;
    args.tma_chunk_elements = chunk_elements;
    return true;
#endif
  }

  void release_nolock_() {
    if (d_coll_) gpuFree(d_coll_);
    if (d_gemm_) gpuFree(d_gemm_);
#if !defined(__HIP_PLATFORM_AMD__)
    if (d_tma_src_descs_) gpuFree(d_tma_src_descs_);
    if (d_tma_dst_descs_) gpuFree(d_tma_dst_descs_);
#endif

    d_coll_ = nullptr;
    d_gemm_ = nullptr;
#if !defined(__HIP_PLATFORM_AMD__)
    d_tma_src_descs_ = nullptr;
    d_tma_dst_descs_ = nullptr;
    tma_desc_stride_ = 0;
#endif

    free_coll_.clear();
    free_gemm_.clear();

    cap_coll_ = 0;
    cap_gemm_ = 0;
  }

 private:
  CollArgs* d_coll_{nullptr};
  GemmArgs* d_gemm_{nullptr};
#if !defined(__HIP_PLATFORM_AMD__)
  void* d_tma_src_descs_{nullptr};
  void* d_tma_dst_descs_{nullptr};
  size_t tma_desc_stride_{0};
#endif

  uint32_t cap_coll_{0};
  uint32_t cap_gemm_{0};

  std::vector<uint32_t> free_coll_, free_gemm_;
  mutable std::mutex coll_mu_;
  mutable std::mutex gemm_mu_;
  bool inited_{false};
};
}  // namespace Device
}  // namespace UKernel
