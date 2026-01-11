#include "operator.h"

// TODO: ThunderKitten/Tilelang? based operators

namespace eccl {

static __device__ __forceinline__ uint32_t decode_wpt(uint32_t encoded_wpt) {
  return encoded_wpt + 1u;
}

template <typename T>
__device__ __forceinline__ T apply_red(ReduceType op, T a, T b) {
  if (op == ReduceType::Sum) return a + b;
  if (op == ReduceType::Max) return a > b ? a : b;
  return a; // None or unknown
}

template <>
__device__ __forceinline__ __half apply_red<__half>(ReduceType op, __half a, __half b) {
  float af = __half2float(a);
  float bf = __half2float(b);
  float rf;
  if (op == ReduceType::Sum) rf = af + bf;
  else if (op == ReduceType::Max) rf = (af > bf ? af : bf);
  else rf = af;
  return __float2half(rf);
}

__device__ void run_copy(const CollArgs& a) {
  auto* dst = reinterpret_cast<char*>(a.dst);
  auto* src = reinterpret_cast<char const*>(a.src);

  const uint64_t total = (uint64_t)a.bytes;

  constexpr uint32_t wpt = 16;

  const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t nthread = (uint64_t)gridDim.x * blockDim.x;

  const uint64_t base = tid * (uint64_t)wpt;
  const uint64_t step = nthread * (uint64_t)wpt;

  for (uint64_t i = base; i < total; i += step) {
#pragma unroll
    for (uint32_t k = 0; k < wpt; ++k) {
      uint64_t j = i + k;
      if (j < total) dst[j] = src[j];
    }
  }
}

template <typename T>
__device__ void run_reduce_inplace(const CollArgs& a) {
  auto* dst = reinterpret_cast<T*>(a.dst);
  auto* src = reinterpret_cast<T const*>(a.src);

  const uint64_t n = (uint64_t)a.bytes / sizeof(T);
  constexpr uint32_t wpt = 8;

  const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t nthread = (uint64_t)gridDim.x * blockDim.x;

  const uint64_t base = tid * (uint64_t)wpt;
  const uint64_t step = nthread * (uint64_t)wpt;

  const ReduceType rop = a.redType;

  if (rop == ReduceType::None) return;

  for (uint64_t i = base; i < n; i += step) {
#pragma unroll
    for (uint32_t k = 0; k < wpt; ++k) {
      uint64_t j = i + k;
      if (j < n) {
        dst[j] = apply_red<T>(rop, dst[j], src[j]);
      }
    }
  }
}

template __device__ void run_reduce_inplace<float>(const CollArgs&);
template __device__ void run_reduce_inplace<__half>(const CollArgs&);
// more
// template __device__ void run_reduce_t<double>(const CollArgs&);
// template __device__ void run_reduce_t<half>(const CollArgs&);

// TODO: using sm id to assign task
template <typename T>
__global__ void basePersistentKernel(mscclpp::C2DDeviceHandle<T> fifo,
                                     CollArgs* d_coll,
                                     MoeArgs*  d_moe,
                                     bool* should_stop) {
  (void)d_moe;

  if (blockIdx.x != 0) return;

  while (true) {
    if (should_stop && *should_stop) break;

    T* task = fifo.poll();
    if (task == nullptr) continue;

    __syncthreads();

    const TaskType ttype = (TaskType)task->type_u8();
    const DataType dtype = (DataType)task->dtype_u8();

    const uint32_t idx = task->args_index();
    const CollArgs a = d_coll[idx];

    if (threadIdx.x == 0) {
      printf("task %u type=%d dtype=%d red=%d bytes=%u\n",
            idx,
            int(ttype),
            int(dtype),
            int(a.redType),
            a.bytes);
    }

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
      default:
        break;
    }

    __threadfence();
    if (threadIdx.x == 0) {
      fifo.pop();
    }
    __syncthreads(); 
  }
}

template __global__ void basePersistentKernel<Task>(
    mscclpp::C2DDeviceHandle<Task> fifo,
    CollArgs* d_coll,
    MoeArgs*  d_moe,
    bool* should_stop);

}  // namespace eccl
