#include "ops/ops.h"
#include "persistent_kernel_ops.h"

namespace UKernel {
namespace Device {

namespace {

constexpr uint32_t kCommandIdle = 0;
constexpr uint32_t kCommandRun = 1;
constexpr uint32_t kCommandExit = 2;

}  // namespace

__device__ void run_copy(TaskArgs const& a, uint32_t block_id,
                         uint32_t num_blocks, void* smem_buf) {
  char* dst = reinterpret_cast<char*>(a.dst);
  char const* src = reinterpret_cast<char const*>(a.src);
  const uint64_t total_count = static_cast<uint64_t>(a.bytes);

  const uint64_t max_threads_per_block = 1024;
  if (blockDim.x > max_threads_per_block) return;

  const uint64_t count_per_block = total_count / num_blocks;
  const uint64_t block_offset = block_id * count_per_block;

  char* my_dst = dst + block_offset;
  char const* my_src = src + block_offset;
  uint64_t my_count = (block_id + 1 == num_blocks)
                          ? (total_count - block_offset)
                          : count_per_block;

  copy<char>(my_dst, my_src, static_cast<size_t>(my_count), smem_buf);
}

template <typename T>
__device__ void run_typed_copy(TaskArgs const& a, uint32_t block_id,
                               uint32_t num_blocks, void* smem_buf) {
  if ((a.bytes % sizeof(T)) != 0) {
    run_copy(a, block_id, num_blocks, smem_buf);
    return;
  }

  T* dst = reinterpret_cast<T*>(a.dst);
  T const* src = reinterpret_cast<T const*>(a.src);
  const uint64_t total_count = static_cast<uint64_t>(a.bytes) / sizeof(T);

  const uint64_t max_threads_per_block = 1024;
  if (blockDim.x > max_threads_per_block) return;

  const uint64_t count_per_block = total_count / num_blocks;
  const uint64_t block_offset = block_id * count_per_block;
  const uint64_t my_count = (block_id + 1 == num_blocks)
                                ? (total_count - block_offset)
                                : count_per_block;

  copy<T>(dst + block_offset, src + block_offset, static_cast<size_t>(my_count),
          smem_buf);
}

template <typename T>
__device__ void run_reduce(TaskArgs const& a, uint32_t block_id,
                           uint32_t num_blocks, void* smem_buf) {
  T* dst = reinterpret_cast<T*>(a.dst);
  T const* src = reinterpret_cast<T const*>(a.src);
  const uint64_t total_count = static_cast<uint64_t>(a.bytes) / sizeof(T);

  const uint64_t max_threads_per_block = 1024;
  if (blockDim.x > max_threads_per_block) return;

  const uint64_t count_per_block = total_count / num_blocks;
  const uint64_t block_offset = block_id * count_per_block;
  const uint64_t my_count = (block_id + 1 == num_blocks)
                                ? (total_count - block_offset)
                                : count_per_block;

  read_reduce_store<T>(dst + block_offset, src + block_offset,
                       static_cast<size_t>(my_count), a.red_type(), smem_buf);
}

__device__ __forceinline__ void process_task(Task const& task,
                                             TaskArgs* d_task_args,
                                             uint32_t block_id,
                                             uint32_t num_blocks,
                                             void* smem_buf) {
  const TaskType ttype = static_cast<TaskType>(task.type_u8());
  const DataType dtype = static_cast<DataType>(task.dtype_u8());
  const uint32_t idx = task.args_index();

  if (idx >= (1UL << TaskArgsIndexSize)) {
    return;
  }

  TaskArgs& args = d_task_args[idx];

  switch (ttype) {
    case TaskType::CollCopy:
      if (dtype == DataType::Int8) {
        run_typed_copy<int8_t>(args, block_id, num_blocks, smem_buf);
      } else if (dtype == DataType::Int32) {
        run_typed_copy<int32_t>(args, block_id, num_blocks, smem_buf);
      } else if (dtype == DataType::Int64) {
        run_typed_copy<int64_t>(args, block_id, num_blocks, smem_buf);
      } else if (dtype == DataType::Fp16) {
        run_typed_copy<__half>(args, block_id, num_blocks, smem_buf);
      } else if (dtype == DataType::Fp32) {
        run_typed_copy<float>(args, block_id, num_blocks, smem_buf);
      } else if (dtype == DataType::Fp64) {
        run_typed_copy<double>(args, block_id, num_blocks, smem_buf);
      } else if (dtype == DataType::Bf16) {
        run_typed_copy<nv_bfloat16>(args, block_id, num_blocks, smem_buf);
      } else {
        run_copy(args, block_id, num_blocks, smem_buf);
      }
      break;
    case TaskType::CollReduce:
      if (dtype == DataType::Fp32) {
        run_reduce<float>(args, block_id, num_blocks, smem_buf);
      } else if (dtype == DataType::Fp16) {
        run_reduce<__half>(args, block_id, num_blocks, smem_buf);
      } else if (dtype == DataType::Int8) {
        run_reduce<int8_t>(args, block_id, num_blocks, smem_buf);
      } else if (dtype == DataType::Int32) {
        run_reduce<int32_t>(args, block_id, num_blocks, smem_buf);
      } else if (dtype == DataType::Int64) {
        run_reduce<int64_t>(args, block_id, num_blocks, smem_buf);
      } else if (dtype == DataType::Fp64) {
        run_reduce<double>(args, block_id, num_blocks, smem_buf);
      } else if (dtype == DataType::Bf16) {
        run_reduce<nv_bfloat16>(args, block_id, num_blocks, smem_buf);
      }
      break;
    default:
      break;
  }
}

__global__ void singlePersistentKernel(
    mscclpp::C2DDeviceHandle<Task>* c2d_fifos, TaskArgs* d_task_args,
    bool* should_stop) {
  extern __shared__ char smem[];
  auto& fifo = c2d_fifos[0];
  void* smem_buf = smem;
  __shared__ Task current_task;
  __shared__ uint32_t command;

  while (true) {
    if (threadIdx.x == 0) {
      command = kCommandIdle;
      if (should_stop && *should_stop) {
        while (Task* task = fifo.poll()) {
          fifo.pop();
        }
        __threadfence();
        command = kCommandExit;
      } else {
        Task* task = fifo.poll();
        if (task != nullptr) {
          current_task = *task;
          command =
              (current_task.type_u8() == static_cast<uint8_t>(TaskType::Stop))
                  ? kCommandExit
                  : kCommandRun;
          if (command == kCommandExit) {
            fifo.pop();
          }
        }
      }
    }
    __syncthreads();

    if (command == kCommandIdle) {
      continue;
    }
    if (command == kCommandExit) {
      return;
    }

    process_task(current_task, d_task_args, 0, 1, smem_buf);
    __syncthreads();

    if (threadIdx.x == 0) {
      // __threadfence();
      fifo.pop();
    }
    __syncthreads();
  }
}

__global__ void multiPersistentKernel(mscclpp::C2DDeviceHandle<Task>* c2d_fifos,
                                      TaskArgs* d_task_args, bool* should_stop,
                                      MultiBlockSync* d_sync) {
  extern __shared__ char smem[];
  auto& fifo = c2d_fifos[0];
  void* smem_buf = smem;
  const uint32_t bid = blockIdx.x;

  __shared__ Task current_task;
  uint32_t local_phase = 0;

  while (true) {
    if (bid == 0 && threadIdx.x == 0) {
      uint32_t command = kCommandRun;
      Task next_task{};

      while (true) {
        if (should_stop && *should_stop) {
          while (Task* pending = fifo.poll()) {
            fifo.pop();
          }
          __threadfence();
          command = kCommandExit;
          break;
        }

        Task* task = fifo.poll();
        if (task == nullptr) {
          continue;
        }

        next_task = *task;
        if (next_task.type_u8() == static_cast<uint8_t>(TaskType::Stop)) {
          fifo.pop();
          command = kCommandExit;
        }
        break;
      }

      d_sync->completedBlocks = 0;
      d_sync->command = command;
      if (command == kCommandRun) {
        d_sync->currentTask = next_task;
      }
      __threadfence();
      mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(
          &d_sync->publishedPhase, local_phase + 1,
          mscclpp::memoryOrderRelease);
    }

    while (mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(
               &d_sync->publishedPhase, mscclpp::memoryOrderAcquire) !=
           local_phase + 1) {
    }
    __syncthreads();

    if (d_sync->command == kCommandExit) {
      return;
    }

    if (threadIdx.x == 0) {
      current_task = d_sync->currentTask;
    }
    __syncthreads();

    process_task(current_task, d_task_args, bid, gridDim.x, smem_buf);
    __syncthreads();

    if (threadIdx.x == 0) {
      mscclpp::atomicFetchAdd<uint32_t, mscclpp::scopeDevice>(
          &d_sync->completedBlocks, 1, mscclpp::memoryOrderAcqRel);
      if (bid == 0) {
        while (mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(
                   &d_sync->completedBlocks, mscclpp::memoryOrderAcquire) <
               gridDim.x) {
        }
        // __threadfence();
        fifo.pop();
        mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(
            &d_sync->publishedPhase, local_phase + 2,
            mscclpp::memoryOrderRelease);
      }
    }

    while (mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(
               &d_sync->publishedPhase, mscclpp::memoryOrderAcquire) !=
           local_phase + 2) {
    }
    local_phase += 2;
  }
}

}  // namespace Device
}  // namespace UKernel
