#include "persistent_kernel_ops.h"
#include "ops/ops.h"

namespace UKernel {
namespace Device {

__device__ void run_copy(TaskArgs const& a, uint32_t block_id,
                         uint32_t num_blocks, void* smem_buf) {
  auto* dst = reinterpret_cast<char*>(a.dst);
  auto* src = reinterpret_cast<char const*>(a.src);
  const uint64_t total = static_cast<uint64_t>(a.bytes);

  const uint64_t max_threads_per_block = 1024; // CUDA max threads per block
  if (blockDim.x > max_threads_per_block) return;
  
  const uint64_t tid =
      static_cast<uint64_t>(block_id) * blockDim.x + threadIdx.x;
  const uint64_t nthread =
      static_cast<uint64_t>(num_blocks) * blockDim.x;

  if (tid > INT_MAX || nthread > INT_MAX) return;

  copy<char>(dst, src, total, static_cast<int>(tid), static_cast<int>(nthread), smem_buf);
}

template <typename T, ReduceType op>
__device__ void run_reduce(TaskArgs const& a, uint32_t block_id,
                           uint32_t num_blocks, void* smem_buf) {
  auto* dst = reinterpret_cast<T*>(a.dst);
  auto* src = reinterpret_cast<T const*>(a.src);
  const uint64_t count = static_cast<uint64_t>(a.bytes) / sizeof(T);

  const uint64_t max_threads_per_block = 1024; // CUDA max threads per block
  if (blockDim.x > max_threads_per_block) return;
  
  const uint64_t tid =
      static_cast<uint64_t>(block_id) * blockDim.x + threadIdx.x;
  const uint64_t nthread =
      static_cast<uint64_t>(num_blocks) * blockDim.x;

  if (tid > INT_MAX || nthread > INT_MAX) return;

  read_reduce_store<T, op>(dst, src, count, smem_buf, 
                           static_cast<int>(tid), static_cast<int>(nthread));
}

__device__ __forceinline__ void process_task(Task* task, TaskArgs* d_task_args,
                                              uint32_t block_id, uint32_t num_blocks,
                                              void* smem_buf) {
  const TaskType ttype = static_cast<TaskType>(task->type_u8());
  const DataType dtype = static_cast<DataType>(task->dtype_u8());
  const uint32_t idx = task->args_index();
  
  if (idx >= 1UL << TaskArgsIndexSize) {
    return; // Invalid index, skip processing
  }
  
  TaskArgs& args = d_task_args[idx];

  switch (ttype) {
    case TaskType::CollCopy: {
      run_copy(args, block_id, num_blocks, smem_buf);
      break;
    }
    case TaskType::CollReduce: {
      if (dtype == DataType::Fp32) {
        run_reduce<float, ReduceType::Sum>(args, block_id, num_blocks, smem_buf);
      } else if (dtype == DataType::Fp16) {
        run_reduce<__half, ReduceType::Sum>(args, block_id, num_blocks, smem_buf);
      } else if (dtype == DataType::Int8) {
        run_reduce<int8_t, ReduceType::Sum>(args, block_id, num_blocks, smem_buf);
      } else if (dtype == DataType::Int32) {
        run_reduce<int32_t, ReduceType::Sum>(args, block_id, num_blocks, smem_buf);
      } else if (dtype == DataType::Int64) {
        run_reduce<int64_t, ReduceType::Sum>(args, block_id, num_blocks, smem_buf);
      } else if (dtype == DataType::Fp64) {
        run_reduce<double, ReduceType::Sum>(args, block_id, num_blocks, smem_buf);
      } else if (dtype == DataType::Bf16) {
        run_reduce<nv_bfloat16, ReduceType::Sum>(args, block_id, num_blocks, smem_buf);
      } else if (dtype == DataType::Fp8) {
        run_reduce<__half, ReduceType::Sum>(args, block_id, num_blocks, smem_buf);
      }
      break;
    }
    default:
      break;
  }
}

__global__ void singlePersistentKernel(mscclpp::C2DDeviceHandle<Task>* c2d_fifos,
                                       TaskArgs* d_task_args,
                                       bool* should_stop) {
  extern __shared__ char smem[];
  auto& fifo = c2d_fifos[0];
  void* smem_buf = smem;
  const uint32_t block_id = 0;
  const uint32_t num_blocks = 1;

  while (true) {
    if (should_stop && *should_stop) {
      if (blockIdx.x == 0 && threadIdx.x == 0) {
          // block 0 thread 0 clean FIFO
          while (Task* task = fifo.poll()) {
              fifo.pop();
          }
          __threadfence();
      }
      return;
    }

    Task* task = fifo.poll();
    if (task == nullptr) continue;
    if (task->type_u8() == static_cast<uint8_t>(TaskType::Stop)) {
      if (threadIdx.x == 0) {
        fifo.pop();
      }
      break;
    }

    __syncthreads(); // force same task

    process_task(task, d_task_args, block_id, num_blocks, smem_buf);

    if (threadIdx.x == 0) {
      fifo.pop();
    }

    __syncthreads(); // force sync fifo
  }
}

__global__ void multiPersistentKernel(mscclpp::C2DDeviceHandle<Task>* c2d_fifos,
                                       TaskArgs* d_task_args,
                                       bool* should_stop,
                                       uint32_t* d_readyFlag) {
  extern __shared__ char smem[];
  const uint32_t bid = blockIdx.x;
  auto& fifo = c2d_fifos[0];
  void* smem_buf = smem;

  // this block's readyFlag
  volatile uint32_t* readyFlag = d_readyFlag + bid;
  if (threadIdx.x == 0) {
      *readyFlag = 0;
  }
  __syncthreads();

  while (true) {
      if (should_stop && *should_stop) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            // block 0 thread 0 clean FIFO
            while (Task* task = fifo.poll()) {
                fifo.pop();
            }
            __threadfence();
        }
        return;
      }

      Task* task = fifo.poll();
      if (task == nullptr) continue;
      if (task->type_u8() == static_cast<uint8_t>(TaskType::Stop)) {
          if (threadIdx.x == 0) fifo.pop();
          break;
      }

      // sync all warps in this block
      __syncthreads();

      process_task(task, d_task_args, bid, gridDim.x, smem_buf);

      __syncthreads();

      if (threadIdx.x == 0) {
          // mark this block ok
          *readyFlag = 1;

          // wait all block ok
          __threadfence(); // ensure writes are visible to other blocks
          
          // spin wait with memory fence to ensure visibility
          for (uint32_t i = 0; i < gridDim.x; i++) {
              while (d_readyFlag[i] == 0) {
                  // Add memory fence to ensure we see updates from other blocks
                  __threadfence_block();
              }
          }

          // block 0 thread0 pop
          if (bid == 0) fifo.pop();

          // reset readyFlag
          if (bid == 0) {
            for (uint32_t i = 0; i < gridDim.x; i++) {
                d_readyFlag[i] = 0;
            }
            __threadfence(); // ensure resets are visible
          }
      }

      __syncthreads(); // block
  }
}

}  // namespace Device
}  // namespace UKernel