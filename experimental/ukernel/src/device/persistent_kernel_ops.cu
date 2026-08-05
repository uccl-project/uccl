#include "../transport/adapter/ipc_signal_ring.h"
#include "ops/ops.h"
#include "persistent_kernel_ops.h"

namespace UKernel {
namespace Device {

namespace {

constexpr uint32_t kCommandIdle = 0;
constexpr uint32_t kCommandRun = 1;
constexpr uint32_t kCommandExit = 2;

__device__ __forceinline__ bool task_uses_args(TaskType ttype) {
  return ttype == TaskType::CollCopy || ttype == TaskType::CollReduce ||
         ttype == TaskType::CollPut;
}

__device__ __forceinline__ void publish_tail_progress(uint64_t* tail,
                                                      uint64_t next_tail) {
  __threadfence_system();
  *reinterpret_cast<uint64_t volatile*>(tail) = next_tail;
}

// Fused PutSignal: write the tag into the peer's shared-memory signal
// ring (zero-copy host mapping). Same producer protocol as the host
// side (IpcAdapter::write_signal_ring): atomic claim, per-slot ready
// flag, so GPU workers and host producers can share the ring. The
// receiver always polls from the CPU — nothing ever waits on the GPU.
__device__ __forceinline__ void signal_ring_write(
    Transport::PeerSignalRingDevice* ring, uint64_t tag) {
  unsigned long long w = atomicAdd_system(
      reinterpret_cast<unsigned long long*>(&ring->write_idx), 1ull);
  size_t idx = static_cast<size_t>(w) & (Transport::kSignalRingSize - 1);
  // Back-pressure: wait for this slot's previous lap to be consumed.
  while (*reinterpret_cast<bool volatile*>(&ring->slots[idx].ready)) {
#ifdef __HIP_PLATFORM_AMD__
    __builtin_amdgcn_s_sleep(2);
#else
    __nanosleep(200);
#endif
  }
  ring->slots[idx].tag = tag;
  __threadfence_system();
  *reinterpret_cast<bool volatile*>(&ring->slots[idx].ready) = true;
  __threadfence_system();
}

// If this task is a fused PutSignal, emit the tag now (copy finished).
__device__ __forceinline__ void maybe_signal_ring_write(Task const& task,
                                                        TaskArgs const* args) {
  if (static_cast<TaskType>(task.type_u8()) != TaskType::CollPut ||
      args == nullptr)
    return;
  signal_ring_write(
      reinterpret_cast<Transport::PeerSignalRingDevice*>(args->src2),
      args->redTypeRaw);
}

}  // namespace

__device__ __forceinline__ void run_copy(TaskArgs const& a, uint32_t block_id,
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
__device__ __forceinline__ void run_typed_copy(TaskArgs const& a,
                                               uint32_t block_id,
                                               uint32_t num_blocks,
                                               void* smem_buf) {
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
__device__ __forceinline__ void run_reduce(TaskArgs const& a, uint32_t block_id,
                                           uint32_t num_blocks,
                                           void* smem_buf) {
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

// Benchmarks

__global__ void benchDispatchNopKernel() {}

__global__ void benchDispatchCopyFp32Kernel(TaskArgs args) {
  run_typed_copy<float>(args, blockIdx.x, gridDim.x, nullptr);
}

__global__ void benchDispatchReduceFp32Kernel(TaskArgs args) {
  run_reduce<float>(args, blockIdx.x, gridDim.x, nullptr);
}

// Dispatch

#define RUN_COPY_BODY(dtype, fn)                           \
  if (dtype == DataType::Int8)                             \
    fn<int8_t>(args, block_id, num_blocks, smem_buf);      \
  else if (dtype == DataType::Int32)                       \
    fn<int32_t>(args, block_id, num_blocks, smem_buf);     \
  else if (dtype == DataType::Int64)                       \
    fn<int64_t>(args, block_id, num_blocks, smem_buf);     \
  else if (dtype == DataType::Fp16)                        \
    fn<__half>(args, block_id, num_blocks, smem_buf);      \
  else if (dtype == DataType::Fp32)                        \
    fn<float>(args, block_id, num_blocks, smem_buf);       \
  else if (dtype == DataType::Fp64)                        \
    fn<double>(args, block_id, num_blocks, smem_buf);      \
  else if (dtype == DataType::Bf16)                        \
    fn<nv_bfloat16>(args, block_id, num_blocks, smem_buf); \
  else                                                     \
    run_copy(args, block_id, num_blocks, smem_buf)

#define RUN_REDUCE_BODY(dtype)                                 \
  if (dtype == DataType::Fp32)                                 \
    run_reduce<float>(args, block_id, num_blocks, smem_buf);   \
  else if (dtype == DataType::Fp16)                            \
    run_reduce<__half>(args, block_id, num_blocks, smem_buf);  \
  else if (dtype == DataType::Int8)                            \
    run_reduce<int8_t>(args, block_id, num_blocks, smem_buf);  \
  else if (dtype == DataType::Int32)                           \
    run_reduce<int32_t>(args, block_id, num_blocks, smem_buf); \
  else if (dtype == DataType::Int64)                           \
    run_reduce<int64_t>(args, block_id, num_blocks, smem_buf); \
  else if (dtype == DataType::Fp64)                            \
    run_reduce<double>(args, block_id, num_blocks, smem_buf);  \
  else if (dtype == DataType::Bf16)                            \
  run_reduce<nv_bfloat16>(args, block_id, num_blocks, smem_buf)

__device__ __forceinline__ void dispatch_task(Task const& task,
                                              TaskArgs const* ready_args,
                                              uint32_t block_id,
                                              uint32_t num_blocks,
                                              void* smem_buf) {
  const TaskType ttype = static_cast<TaskType>(task.type_u8());
  const DataType dtype = static_cast<DataType>(task.dtype_u8());

  if (ready_args == nullptr) return;
  TaskArgs const& args = *ready_args;

  switch (ttype) {
    case TaskType::CollCopy:
    case TaskType::CollPut:
      // CollPut: copy first; the signal ring write happens at task
      // completion time (maybe_signal_ring_write), after all blocks.
      RUN_COPY_BODY(dtype, run_typed_copy);
      break;
    case TaskType::CollReduce:
      RUN_REDUCE_BODY(dtype);
      break;
    default:
      break;
  }
}

#undef RUN_COPY_BODY
#undef RUN_REDUCE_BODY

__device__ __forceinline__ void process_task(Task const& task,
                                             TaskArgs* d_task_args,
                                             uint32_t block_id,
                                             uint32_t num_blocks,
                                             void* smem_buf) {
  TaskArgs* ready_args = nullptr;
  const TaskType ttype = static_cast<TaskType>(task.type_u8());
  if (task_uses_args(ttype)) {
    const uint32_t idx = task.args_index();
    if (idx >= (1UL << TaskArgsIndexSize)) {
      return;
    }
    ready_args = d_task_args + idx;
  }
  dispatch_task(task, ready_args, block_id, num_blocks, smem_buf);
}

__device__ __forceinline__ void idle_sleep() {
#ifdef __HIP_PLATFORM_AMD__
  __builtin_amdgcn_s_sleep(2);
#else
  // __nanosleep's argument is a multiple of 100ns. The previous value
  // (100) slept 10us per poll, so a 500us idle grace (5000 polls) took
  // ~50ms wall time to actually exit — the worker spun 100x longer than
  // configured, showing up as a ~25ms periodic gap in nsys traces and
  // relaunch jitter. 1 = 100ns, matching the poll-count derivation in
  // WorkerPool (idleExitAfterUs * 10 polls, ~300-500ns per poll with the
  // loop + syncthreads overhead).
  __nanosleep(1);
#endif
}

__global__ void singlePersistentKernel(
    mscclpp::C2DDeviceHandle<Task>* c2d_fifos, TaskArgs* d_task_args,
    bool* should_stop, bool* exited_flag, uint32_t exit_idle_iters) {
  extern __shared__ char smem[];
  auto& fifo = c2d_fifos[0];
  void* smem_buf = smem;
  __shared__ Task current_task;
  __shared__ __align__(16) unsigned char current_args_storage[sizeof(TaskArgs)];
  __shared__ bool has_current_args;
  __shared__ uint32_t command;
  uint64_t cached_tail = 0;
  uint64_t cached_head = 0;
  uint32_t idle_ticks = 0;
  TaskArgs* current_args = reinterpret_cast<TaskArgs*>(current_args_storage);

  if (threadIdx.x == 0) {
    cached_tail = mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(
        fifo.tail, mscclpp::memoryOrderRelaxed);
    cached_head = cached_tail;
  }

  while (true) {
    if (threadIdx.x == 0) {
      command = kCommandIdle;
      if (should_stop && *should_stop) {
        cached_head = mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(
            fifo.head, mscclpp::memoryOrderAcquire);
        if (cached_tail != cached_head) {
          cached_tail = cached_head;
          publish_tail_progress(fifo.tail, cached_tail);
        }
        command = kCommandExit;
      } else {
        if (cached_tail >= cached_head) {
          cached_head = mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(
              fifo.head, mscclpp::memoryOrderAcquire);
        }
        if (cached_tail >= cached_head) {
          // Fifo is drained. With an idle grace configured, exit after
          // exit_idle_iters consecutive empty polls so host-side
          // device-wide syncs (torch etc.) can pass; the host relaunches
          // us on the next enqueue.
          if (exit_idle_iters && ++idle_ticks >= exit_idle_iters) {
            if (exited_flag) {
              *exited_flag = true;
              __threadfence_system();
            }
            command = kCommandExit;
          } else {
            idle_sleep();
          }
        } else {
          idle_ticks = 0;
          current_task = fifo.buffer[cached_tail % fifo.size];
          has_current_args = false;
          command = kCommandRun;
        }
      }
    }
    __syncthreads();

    if (command == kCommandExit) break;
    if (command != kCommandRun) continue;

    const TaskType ttype = static_cast<TaskType>(current_task.type_u8());
    if (task_uses_args(ttype)) {
      if (threadIdx.x == 0 && !has_current_args) {
        const uint32_t idx = current_task.args_index();
        if (idx < (1UL << TaskArgsIndexSize)) {
          *current_args = d_task_args[idx];
          has_current_args = true;
        }
      }
      __syncthreads();
      if (!has_current_args) continue;
    }
    __syncthreads();

    dispatch_task(current_task, task_uses_args(ttype) ? current_args : nullptr,
                  blockIdx.x, gridDim.x, smem_buf);
    __syncthreads();

    if (threadIdx.x == 0) {
      maybe_signal_ring_write(current_task, current_args);
      ++cached_tail;
      publish_tail_progress(fifo.tail, cached_tail);
    }
    __syncthreads();
  }
}

__global__ void multiPersistentKernel(mscclpp::C2DDeviceHandle<Task>* c2d_fifos,
                                      TaskArgs* d_task_args, bool* should_stop,
                                      MultiBlockSync* d_sync, bool* exited_flag,
                                      uint32_t exit_idle_iters) {
  extern __shared__ char smem[];
  auto& fifo = c2d_fifos[0];
  void* smem_buf = smem;
  const uint32_t bid = blockIdx.x;

  __shared__ Task current_task;
  __shared__ __align__(16) unsigned char current_args_storage[sizeof(TaskArgs)];
  __shared__ bool has_current_args;
  TaskArgs* current_args = reinterpret_cast<TaskArgs*>(current_args_storage);
  uint32_t local_phase = 0;
  uint64_t cached_tail = 0;
  uint64_t cached_head = 0;
  uint32_t idle_ticks = 0;

  if (bid == 0 && threadIdx.x == 0) {
    cached_tail = mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(
        fifo.tail, mscclpp::memoryOrderRelaxed);
    cached_head = cached_tail;
  }

  while (true) {
    if (bid == 0 && threadIdx.x == 0) {
      uint32_t command = kCommandRun;
      Task next_task{};
      bool next_has_args = false;

      while (true) {
        if (should_stop && *should_stop) {
          cached_head = mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(
              fifo.head, mscclpp::memoryOrderAcquire);
          if (cached_tail != cached_head) {
            cached_tail = cached_head;
            publish_tail_progress(fifo.tail, cached_tail);
          }
          command = kCommandExit;
          break;
        }

        if (cached_tail >= cached_head) {
          cached_head = mscclpp::atomicLoad<uint64_t, mscclpp::scopeDevice>(
              fifo.head, mscclpp::memoryOrderAcquire);
        }
        if (cached_tail >= cached_head) {
          cached_head = mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(
              fifo.head, mscclpp::memoryOrderAcquire);
        }
        if (cached_tail >= cached_head) {
          // Fifo drained: exit after the idle grace so host-side
          // device-wide syncs can pass; host relaunches on next enqueue.
          if (exit_idle_iters && ++idle_ticks >= exit_idle_iters) {
            if (exited_flag) {
              *exited_flag = true;
              __threadfence_system();
            }
            command = kCommandExit;
            break;
          }
          idle_sleep();
          continue;
        }
        idle_ticks = 0;

        next_task = fifo.buffer[cached_tail % fifo.size];
        if (next_task.type_u8() == static_cast<uint8_t>(TaskType::Stop)) {
          ++cached_tail;
          publish_tail_progress(fifo.tail, cached_tail);
          command = kCommandExit;
        } else if (task_uses_args(static_cast<TaskType>(next_task.type_u8()))) {
          const uint32_t idx = next_task.args_index();
          if (idx < (1UL << TaskArgsIndexSize)) {
            TaskArgs* args = d_task_args + idx;
            d_sync->currentArgs = *args;
            next_has_args = true;
          }
        }
        break;
      }

      d_sync->completedBlocks = 0;
      d_sync->command = command;
      d_sync->hasCurrentArgs = next_has_args ? 1u : 0u;
      if (command == kCommandRun) {
        d_sync->currentTask = next_task;
      }
      __threadfence();
      mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(
          &d_sync->publishedPhase, local_phase + 1,
          mscclpp::memoryOrderRelease);
    }

    // Phase values are strictly increasing (bid 0 publishes 2k+1 then
    // 2k+2 per task). Waiting with `<` (instead of `!=`) makes a block
    // that was preempted past the value catch up instead of spinning
    // forever on a phase that never reappears — the `!=` form deadlocked
    // under repeated idle-exit/relaunch cycles (observed: bid 0 waiting
    // on completedBlocks=60/64 while 4 blocks were stuck on an earlier
    // task's already-published phase).
    while (mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(
               &d_sync->publishedPhase, mscclpp::memoryOrderAcquire) <
           local_phase + 1) {
    }
    __syncthreads();

    if (d_sync->command == kCommandExit) {
      return;
    }

    if (threadIdx.x == 0) {
      current_task = d_sync->currentTask;
      has_current_args = d_sync->hasCurrentArgs != 0;
      if (has_current_args) {
        *current_args = d_sync->currentArgs;
      }
    }
    __syncthreads();

    dispatch_task(current_task, has_current_args ? current_args : nullptr, bid,
                  gridDim.x, smem_buf);
    __syncthreads();

    if (threadIdx.x == 0) {
      mscclpp::atomicFetchAdd<uint32_t, mscclpp::scopeDevice>(
          &d_sync->completedBlocks, 1, mscclpp::memoryOrderAcqRel);
      if (bid == 0) {
        while (mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(
                   &d_sync->completedBlocks, mscclpp::memoryOrderAcquire) <
               gridDim.x) {
        }
        // Same ordering requirement as the single-block path: all blocks'
        // writes must be visible before host observes task completion.
        // TMA bulk stores complete via the async proxy; fence it to the
        // generic proxy so host-side memsets/next-task loads see the data,
        // then make it device-visible (fence.proxy.async.global only
        // orders the executing thread's async ops vs generic accesses).
        tma_fence_async_global();
        __threadfence();
        maybe_signal_ring_write(current_task, current_args);
        ++cached_tail;
        publish_tail_progress(fifo.tail, cached_tail);
        mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(
            &d_sync->publishedPhase, local_phase + 2,
            mscclpp::memoryOrderRelease);
      }
    }

    while (mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(
               &d_sync->publishedPhase, mscclpp::memoryOrderAcquire) <
           local_phase + 2) {
    }
    local_phase += 2;
  }
}

}  // namespace Device
}  // namespace UKernel
