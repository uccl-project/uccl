#include "operator.h"

namespace eccl {

static __device__ __forceinline__ uint32_t decode_wpt(uint32_t encoded_wpt) {
  return encoded_wpt + 1u;
}

template <typename T>
__device__ __forceinline__ T apply_red(OpRedType op, T a, T b) {
  if (op == OpRedSum) return a + b;
  if (op == OpRedMax) return a > b ? a : b;
  return a;
}

template <>
__device__ __forceinline__ __half apply_red<__half>(OpRedType op, __half a,
                                                    __half b) {
  float af = __half2float(a);
  float bf = __half2float(b);
  float rf;
  if (op == OpRedSum)
    rf = af + bf;
  else if (op == OpRedMax)
    rf = (af > bf ? af : bf);
  else
    rf = af;
  return __float2half(rf);
}

__device__ void run_copy(OpTask const& t) {
  auto* dst = reinterpret_cast<char*>((uintptr_t)t.dst);
  auto* src = reinterpret_cast<char const*>((uintptr_t)t.src);

  const uint64_t total = (uint64_t)t.size;
  const uint32_t wpt = decode_wpt((uint32_t)opWpt(t.meta));

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
__device__ void run_reduce_inplace(OpTask const& t) {
  auto* dst = reinterpret_cast<T*>((uintptr_t)t.dst);
  auto* src = reinterpret_cast<const T*>((uintptr_t)t.src);

  const uint64_t n = (uint64_t)t.size / sizeof(T);
  const uint32_t wpt = decode_wpt((uint32_t)opWpt(t.meta));
  const OpRedType rop = (OpRedType)opRedType(t.meta);

  const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t nthread = (uint64_t)gridDim.x * blockDim.x;

  const uint64_t base = tid * (uint64_t)wpt;
  const uint64_t step = nthread * (uint64_t)wpt;

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

template __device__ void run_reduce_inplace<float>(OpTask const&);
template __device__ void run_reduce_inplace<__half>(OpTask const&);
// more
// template __device__ void run_reduce_t<double>(const OpTask&, int, int);
// template __device__ void run_reduce_t<half>(const OpTask&, int, int);

// TODO: using sm id to assign task
template <typename T>
__global__ void basePersistentKernel(mscclpp::C2DDeviceHandle<T> fifo,
                                     bool* should_stop) {
  if (blockIdx.x != 0) return;

  while (true) {
    if (should_stop && *should_stop) break;

    T* task = fifo.poll();
    if (task == nullptr) continue;

    const uint32_t ttype = opTaskType(task->meta);
    const uint32_t dtype = opDataType(task->meta);

    switch ((OpTaskType)ttype) {
      case OpTaskCopy: {
        run_copy(*task);
        break;
      }

      case OpTaskReduce: {
        if ((OpDataType)dtype == OpDataFp32) {
          run_reduce_inplace<float>(*task);
        } else if ((OpDataType)dtype == OpDataFp16) {
          run_reduce_inplace<__half>(*task);
        }
        break;
      }

      default:
        break;
    }

    __threadfence();
    fifo.pop();
  }
}

template __global__ void basePersistentKernel<OpTask>(
    mscclpp::C2DDeviceHandle<OpTask> fifo, bool* should_stop);

}  // namespace eccl
