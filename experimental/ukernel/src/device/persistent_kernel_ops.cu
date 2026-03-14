#include "fifo/d2c_fifo_device.hpp"
#include "persistent_kernel_ops.h"
// #include "operators/operator.cuh"

namespace UKernel {
namespace Device {

template <typename T>
__device__ __forceinline__ T apply_red(ReduceType op, T a, T b) {
  if (op == ReduceType::Sum) return a + b;
  if (op == ReduceType::Max) return a > b ? a : b;
  return a;  // None or unknown
}

template <>
__device__ __forceinline__ __half apply_red<__half>(ReduceType op, __half a,
                                                    __half b) {
  float af = __half2float(a);
  float bf = __half2float(b);
  float rf;
  if (op == ReduceType::Sum)
    rf = af + bf;
  else if (op == ReduceType::Max)
    rf = (af > bf ? af : bf);
  else
    rf = af;
  return __float2half(rf);
}

__device__ void run_copy_register(CollArgs const& a) {
  auto* dst = reinterpret_cast<char*>(a.dst);
  auto* src = reinterpret_cast<char const*>(a.src);

  const uint64_t total = (uint64_t)a.bytes;

  const uint64_t tid = (uint64_t)threadIdx.x;
  const uint64_t nthread = (uint64_t)blockDim.x;

  const uint64_t chunk_size = (total + nthread - 1) / nthread;

  const uint64_t start = tid * chunk_size;
  const uint64_t end = min(start + chunk_size, total);

  for (uint64_t i = start; i < end; ++i) {
    dst[i] = src[i];
  }
}

__device__ void run_copy_tma(CollArgs const& a) {
  // Keep TmaOp on a dedicated implementation path so selection and tests
  // exercise a distinct bulk-copy kernel even before tensor-map descriptors
  // are plumbed through the task model.
  auto* dst = reinterpret_cast<char*>(a.dst);
  auto* src = reinterpret_cast<char const*>(a.src);
  const uint64_t total = static_cast<uint64_t>(a.bytes);
  constexpr uint64_t kVecBytes = sizeof(uint4);

  uintptr_t alignment_bits =
      reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src) |
      static_cast<uintptr_t>(total);
  if ((alignment_bits & (kVecBytes - 1)) != 0) {
    run_copy_register(a);
    return;
  }

  auto* dst_vec = reinterpret_cast<uint4*>(dst);
  auto* src_vec = reinterpret_cast<uint4 const*>(src);
  uint64_t vec_count = total / kVecBytes;

  for (uint64_t i = static_cast<uint64_t>(threadIdx.x); i < vec_count;
       i += static_cast<uint64_t>(blockDim.x)) {
    dst_vec[i] = src_vec[i];
  }
}

__device__ void run_copy(CollArgs const& a) {
  switch (a.resolved_path) {
    case TransferPath::TmaOp:
      run_copy_tma(a);
      return;
    case TransferPath::Auto:
    case TransferPath::RegisterOp:
    default:
      run_copy_register(a);
      return;
  }
}

template <typename T>
__device__ void run_reduce_inplace(CollArgs const& a) {
  auto* dst = reinterpret_cast<T*>(a.dst);
  auto* src = reinterpret_cast<T const*>(a.src);

  const uint64_t n = (uint64_t)a.bytes / sizeof(T);

  const uint64_t tid = (uint64_t)threadIdx.x;
  const uint64_t nthread = (uint64_t)blockDim.x;

  const uint64_t chunk_size = (n + nthread - 1) / nthread;
  const uint64_t start = tid * chunk_size;
  const uint64_t end = min(start + chunk_size, n);

  const ReduceType rop = a.redType;

  if (rop == ReduceType::None) return;

  for (uint64_t i = start; i < end; ++i) {
    dst[i] = apply_red<T>(rop, dst[i], src[i]);
  }
}

template __device__ void run_reduce_inplace<float>(CollArgs const&);
template __device__ void run_reduce_inplace<__half>(CollArgs const&);
// more
// template __device__ void run_reduce_t<double>(const CollArgs&);
// template __device__ void run_reduce_t<half>(const CollArgs&);

// TODO: using sm id to assign task
__global__ void basePersistentKernel(mscclpp::C2DDeviceHandle<Task>* c2d_fifos,
                                     mscclpp::FifoDeviceHandle* d2c_fifo,
                                     CollArgs* d_coll, GemmArgs* d_gemm,
                                     bool* should_stop) {
  extern __shared__ char smem[];
  (void)d2c_fifo;
  (void)d_gemm;
  (void)smem;

  const uint32_t bid = blockIdx.x;
  auto& fifo = c2d_fifos[bid];  // block => fifo

  while (true) {
    if (should_stop && *should_stop) break;

    Task* task = fifo.poll();
    if (task == nullptr) continue;

    /*
    If we later re-introduce inter-SM task fanout, it should live on top of the
    standalone SmFifo primitive rather than as implicit runtime state here.
    */

    __syncthreads();

    const TaskType ttype = (TaskType)task->type_u8();
    const DataType dtype = (DataType)task->dtype_u8();
    const uint32_t idx = task->args_index();
    const uint32_t block_id = task->block_index();
    const CollArgs a = d_coll[idx];

    // if (threadIdx.x == 0) {
    //   printf("cur block=%u : task block_id=%u args_id=%u type=%d dtype=%d
    //   red=%d bytes=%llu path=%u step=%u chunk=%u\n", bid, block_id, idx,
    //          int(ttype), int(dtype), int(a.redType),
    //          static_cast<unsigned long long>(a.bytes),
    //          static_cast<unsigned>(a.resolved_path), a.step_id, a.chunk_id);
    // }

    switch (ttype) {
      case TaskType::CollCopy: {
        run_copy(a);
        break;
      }
      case TaskType::CollReduce: {
        if (dtype == DataType::Fp32) {
          run_reduce_inplace<float>(a);
        } else if (dtype == DataType::Fp16) {
          run_reduce_inplace<__half>(a);
        } else {
          // Fp8 TODO:
        }
        break;
      }
      case TaskType::TkGemm: {
        // skip TK operator now.
        // GemmArgs const& ga = d_gemm[idx];
        // TkMatmulGlobals* g = (TkMatmulGlobals*)ga.globals;
        // run_tk_gemm(*g, ga.num_tile_rows, ga.num_tile_cols, smem);
        break;
      }
      default:
        break;
    }

    __threadfence();

    if (threadIdx.x == 0) {
      // Push completion trigger to host
      // if (d2c_fifo) {
      //   mscclpp::ProxyTrigger trig(ttype, dtype, block_id, idx);
      //   d2c_fifo->push(trig);
      // }
      // Pop task from C2D fifo
      fifo.pop();
    }

    __syncthreads();
  }
}

}  // namespace Device
}  // namespace UKernel
