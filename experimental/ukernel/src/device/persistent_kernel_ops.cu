#include "persistent_kernel_ops.h"

namespace UKernel {
namespace Device {

__device__ void run_copy_register(TaskArgs const& a, uint32_t block_id,
                                  uint32_t num_blocks) {
  auto* dst = reinterpret_cast<char*>(a.dst);
  auto* src = reinterpret_cast<char const*>(a.src);
  const uint64_t total = static_cast<uint64_t>(a.bytes);

  const uint64_t tid =
      static_cast<uint64_t>(block_id) * blockDim.x + threadIdx.x;
  const uint64_t nthread =
      static_cast<uint64_t>(num_blocks) * blockDim.x;
  const uint64_t chunk_size = (total + nthread - 1) / nthread;
  const uint64_t start = tid * chunk_size;
  const uint64_t end = (start + chunk_size < total) ? start + chunk_size : total;

  for (uint64_t i = start; i < end; ++i) {
    dst[i] = src[i];
  }
}

template <typename T>
__device__ __forceinline__ T apply_red(ReduceType op, T a, T b) {
  if (op == ReduceType::Sum) return a + b;
  if (op == ReduceType::Max) return (a > b) ? a : b;
  if (op == ReduceType::Min) return (a < b) ? a : b;
  return a;
}

template <>
__device__ __forceinline__ __half apply_red<__half>(ReduceType op,
                                                    __half a,
                                                    __half b) {
  float af = __half2float(a);
  float bf = __half2float(b);
  float rf = af;
  if (op == ReduceType::Sum) {
    rf = af + bf;
  } else if (op == ReduceType::Max) {
    rf = (af > bf) ? af : bf;
  } else if (op == ReduceType::Min) {
    rf = (af < bf) ? af : bf;
  }
  return __float2half(rf);
}

template <typename T>
__device__ void run_reduce_inplace(TaskArgs const& a, uint32_t block_id,
                                   uint32_t num_blocks) {
  auto* dst = reinterpret_cast<T*>(a.dst);
  auto* src = reinterpret_cast<T const*>(a.src);
  const uint64_t n = static_cast<uint64_t>(a.bytes) / sizeof(T);

  const uint64_t tid =
      static_cast<uint64_t>(block_id) * blockDim.x + threadIdx.x;
  const uint64_t nthread =
      static_cast<uint64_t>(num_blocks) * blockDim.x;
  const uint64_t chunk_size = (n + nthread - 1) / nthread;
  const uint64_t start = tid * chunk_size;
  const uint64_t end = (start + chunk_size < n) ? start + chunk_size : n;

  const ReduceType rop = a.redType;
  if (rop == ReduceType::None) return;

  for (uint64_t i = start; i < end; ++i) {
    dst[i] = apply_red<T>(rop, dst[i], src[i]);
  }
}

template __device__ void run_reduce_inplace<float>(TaskArgs const&, uint32_t,
                                                    uint32_t);
template __device__ void run_reduce_inplace<__half>(TaskArgs const&, uint32_t,
                                                    uint32_t);

__global__ void basePersistentKernel(mscclpp::C2DDeviceHandle<Task>* c2d_fifos,
                                     TaskArgs* d_task_args,
                                     bool* should_stop) {
  const uint32_t bid = blockIdx.x;
  auto& fifo = c2d_fifos[bid];

  while (true) {
    if (should_stop && *should_stop) break;

    Task* task = fifo.poll();
    if (task == nullptr) continue;

    __syncthreads();

    const TaskType ttype = static_cast<TaskType>(task->type_u8());
    const DataType dtype = static_cast<DataType>(task->dtype_u8());
    const uint32_t idx = task->args_index();

    switch (ttype) {
      case TaskType::CollCopy: {
        run_copy_register(d_task_args[idx], bid, gridDim.x);
        break;
      }
      case TaskType::CollReduce: {
        if (dtype == DataType::Fp32) {
          run_reduce_inplace<float>(d_task_args[idx], bid, gridDim.x);
        } else if (dtype == DataType::Fp16) {
          run_reduce_inplace<__half>(d_task_args[idx], bid, gridDim.x);
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

}  // namespace Device
}  // namespace UKernel
