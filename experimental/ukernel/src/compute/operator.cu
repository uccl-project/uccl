#include "d2c_fifo_device.hpp"
#include "operator.h"

// TODO: ThunderKitten/Tilelang? based operators

namespace UKernel {
namespace Compute {

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

__device__ void run_copy(CollArgs const& a) {
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
template <typename T>
__global__ void basePersistentKernel(mscclpp::C2DDeviceHandle<T>* c2d_fifos,
                                     mscclpp::SmDeviceHandle<T>* sm_fifos,
                                     mscclpp::FifoDeviceHandle* d2c_fifo,
                                     CollArgs* d_coll, MoeArgs* d_moe,
                                     bool* should_stop) {
  (void)d_moe;

  const uint32_t bid = blockIdx.x;
  auto& fifo = c2d_fifos[bid];  // block => fifo

  while (true) {
    if (should_stop && *should_stop) break;

    T* task = fifo.poll();
    if (task == nullptr) continue;

    /*
    sm 2 sm 的任务，thread0
    做任务boradcast，然后依次sync那几个被发布task的fifoHead，
    等待并发的任务完成，其余sm 只 poll task，完成后thread0负责pop
    */

    __syncthreads();

    const TaskType ttype = (TaskType)task->type_u8();
    const DataType dtype = (DataType)task->dtype_u8();
    const uint32_t idx = task->args_index();
    const uint32_t block_id = task->block_index();
    const CollArgs a = d_coll[idx];

    // if (threadIdx.x == 0) {
    //   printf("cur block=%u : task block_id=%u args_id=%u type=%d dtype=%d
    //   red=%d bytes=%u\n", bid, block_id, idx, int(ttype),
    //          int(dtype), int(a.redType), a.bytes);
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

template __global__ void basePersistentKernel<Task>(
    mscclpp::C2DDeviceHandle<Task>* c2d_fifos,
    mscclpp::SmDeviceHandle<Task>* sm_fifos,
    mscclpp::FifoDeviceHandle* d2c_fifo, CollArgs* d_coll, MoeArgs* d_moe,
    bool* should_stop);

}  // namespace Compute
}  // namespace UKernel
