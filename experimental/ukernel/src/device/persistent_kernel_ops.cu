#include "../transport/adapter/ipc_signal_ring.h"
#include "ops/ops.h"
#include "ops/reduce_dispatch.h"
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
// ring (zero-copy host mapping). Shares the producer protocol with the
// host IPC send worker (IpcAdapter::claim_signal_slot): both claim the
// same atomic write_idx (device: atomicAdd_system, host: fetch_add),
// then check their claimed slot's per-slot ready flag. The receiver
// always polls from the CPU — nothing ever waits on the GPU.
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

// If this task carries a fused signal (CollPut's PutSignal, or a fused
// reduce+copy task), emit the tag now that the work is finished.
__device__ __forceinline__ void signal_flag_write(uint64_t* slot,
                                                  uint64_t tag) {
  // Order the task's data writes (including the copy into the peer's
  // accumulation buffer) before the flag becomes visible to the
  // receiver's host poll. Plain store + fences — no atomics, so this
  // works where gpuDevAttrHostNativeAtomicSupported=0.
  __threadfence_system();
  *slot = tag;
  __threadfence_system();
}

__device__ __forceinline__ void maybe_signal_ring_write(Task const& task,
                                                        TaskArgs const* args) {
  if (args == nullptr) return;
  TaskType const t = static_cast<TaskType>(task.type_u8());
  bool const want = (t == TaskType::CollPut) ||
                    (t != TaskType::CollPut && args->signal_after());
  if (!want) return;
  if (t == TaskType::CollPut) {
    signal_ring_write(
        reinterpret_cast<Transport::PeerSignalRingDevice*>(args->src2),
        args->redTypeRaw);
  } else {
    signal_flag_write(reinterpret_cast<uint64_t*>(args->src2),
                      args->signal_tag);
  }
}

}  // namespace

__device__ __forceinline__ void run_copy(TaskArgs const& a, uint32_t block_id,
                                         uint32_t num_blocks, void* smem_buf) {
  char* dst = reinterpret_cast<char*>(a.dst);
  char const* src = reinterpret_cast<char const*>(a.src);
  const uint64_t total_count = static_cast<uint64_t>(a.bytes);

  const uint64_t max_threads_per_block = 1024;
  if (blockDim.x > max_threads_per_block) return;

  // Round each block's chunk to the 16B vector width so every block
  // starts at a 16B-aligned offset. The AllToAll hybrid splits can
  // produce total counts where total/num_blocks is not a vector
  // multiple (e.g. pct=70 dev half: 2516584 floats / 32 blocks = 78643
  // floats), and the misaligned Vec access hangs the worker on B300.
  const uint64_t count_per_block =
      (total_count / num_blocks / 16) * 16;  // 16 bytes per Vec
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

  constexpr uint64_t kVecElems = 16 / sizeof(T);  // 4 for float
  const uint64_t count_per_block =
      (total_count / num_blocks / kVecElems) * kVecElems;
  const uint64_t block_offset = block_id * count_per_block;
  const uint64_t my_count = (block_id + 1 == num_blocks)
                                ? (total_count - block_offset)
                                : count_per_block;

  copy<T>(dst + block_offset, src + block_offset, static_cast<size_t>(my_count),
          smem_buf);
}

// Benchmarks

__global__ void benchDispatchNopKernel() {}

__global__ void benchDispatchCopyFp32Kernel(TaskArgs args) {
  run_typed_copy<float>(args, blockIdx.x, gridDim.x, nullptr);
}

__global__ void benchDispatchReduceFp32Kernel(TaskArgs args) {
  dispatch_reduce_fp32(args, blockIdx.x, gridDim.x, nullptr);
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
    dispatch_reduce_fp32(args, block_id, num_blocks, smem_buf); \
  else if (dtype == DataType::Fp16)                            \
    dispatch_reduce_fp16(args, block_id, num_blocks, smem_buf); \
  else if (dtype == DataType::Int8)                            \
    dispatch_reduce_int8(args, block_id, num_blocks, smem_buf); \
  else if (dtype == DataType::Int32)                           \
    dispatch_reduce_int32(args, block_id, num_blocks, smem_buf); \
  else if (dtype == DataType::Int64)                           \
    dispatch_reduce_int64(args, block_id, num_blocks, smem_buf); \
  else if (dtype == DataType::Fp64)                            \
    dispatch_reduce_fp64(args, block_id, num_blocks, smem_buf); \
  else if (dtype == DataType::Bf16)                            \
    dispatch_reduce_bf16(args, block_id, num_blocks, smem_buf)

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

  // Clear the host-visible exit flag at kernel entry. The relaunch path
  // is async (the new grid is stream-ordered behind the exiting one), so
  // this runs strictly after the old grid's final h_exited write and
  // stops the host from seeing a stale "exited" and relaunching a live
  // worker.
  if (threadIdx.x == 0) {
    *exited_flag = false;
    __threadfence_system();
  }

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
  const uint64_t all_blocks_mask =
      (gridDim.x == 64) ? ~0ull : ((1ull << gridDim.x) - 1ull);
  const uint64_t own_bit = 1ull << bid;

  __shared__ Task current_task;
  __shared__ __align__(16) unsigned char current_args_storage[sizeof(TaskArgs)];
  __shared__ bool has_current_args;
  __shared__ bool do_exit;
  __shared__ uint64_t sh_head;
  __shared__ uint64_t sh_tail;
  TaskArgs* current_args = reinterpret_cast<TaskArgs*>(current_args_storage);
  uint32_t idle_ticks = 0;
  bool own_idle_vote = false;

  // Clear the host-visible exit flag at kernel entry (see the single
  // kernel above; the async relaunch queues this grid after the exiting
  // one, so the old grid's final h_exited write is overwritten here).
  if (threadIdx.x == 0) {
    *exited_flag = false;
    __threadfence_system();
  }

  while (true) {
    // Exit rendezvous: a block leaves only when every block has voted to
    // exit (idle grace elapsed) or the host requests a stop. A block that
    // sees work clears its vote and processes it, so the mask can only
    // fill while the FIFO is quiescent and no task is in flight. The
    // decision is broadcast through shared memory so the whole block —
    // not just thread 0 — returns together.
    if (threadIdx.x == 0) {
      do_exit = false;
      if (should_stop && *should_stop) {
        // Host stop: best-effort drain (mark everything consumed) then
        // exit. Correctness at teardown is host-observed tail only.
        uint64_t h = mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(
            fifo.head, mscclpp::memoryOrderAcquire);
        uint64_t t = mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(
            fifo.tail, mscclpp::memoryOrderRelaxed);
        if (t != h) publish_tail_progress(fifo.tail, h);
        do_exit = true;
      } else {
        uint64_t mask = mscclpp::atomicLoad<uint64_t, mscclpp::scopeDevice>(
            &d_sync->exitReadyMask, mscclpp::memoryOrderAcquire);
        if (mask == all_blocks_mask) {
          // The grid is actually leaving now: publish the host-visible
          // flag only here, so the host's relaunch never races a grid
          // that is still busy processing (it would block its stream
          // sync until termination that never comes).
          if (exited_flag) {
            *exited_flag = true;
            __threadfence_system();
          }
          do_exit = true;
        }
      }
    }
    __syncthreads();
    if (do_exit) return;

    // Refresh the shared consumed/enqueued pointers. Only block 0 reads
    // the host-pinned FIFO head (one PCIe round trip per iteration) and
    // publishes it to device memory; the other blocks consume that hint
    // (device-scope load, cheap) plus the device-resident tail. Reading
    // the host-pinned head from every block per iteration is N PCIe
    // round trips that contend with the IPC put engine. A block may see
    // the hint one iteration late; it then simply joins the in-flight
    // task before the completion counter drains (task stays in the FIFO
    // until the last block publishes tail).
    if (threadIdx.x == 0) {
      if (bid == 0) {
        sh_head = mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(
            fifo.head, mscclpp::memoryOrderAcquire);
        mscclpp::atomicStore<uint64_t, mscclpp::scopeDevice>(
            &d_sync->headHint, sh_head, mscclpp::memoryOrderRelease);
      }
      // Device-resident snapshot (block 0's own value is already current;
      // others may lag by one poll, which is safe as noted above).
      if (bid != 0) {
        sh_head = mscclpp::atomicLoad<uint64_t, mscclpp::scopeDevice>(
            &d_sync->headHint, mscclpp::memoryOrderAcquire);
      }
      // Tail is written by the GPU (last-finisher publish with a system
      // fence) and only read back here; the host reads it separately via
      // GDR. Device-scope is enough for the GPU and avoids 64 system-
      // scope atomics per task at high block counts.
      sh_tail = mscclpp::atomicLoad<uint64_t, mscclpp::scopeDevice>(
          fifo.tail, mscclpp::memoryOrderAcquire);
    }
    __syncthreads();

    if (sh_tail >= sh_head) {
      // FIFO empty: idle. Once the grace elapses, register this block's
      // exit vote; the mask reaching all-ones triggers the rendezvous.
      if (threadIdx.x == 0) {
        if (exit_idle_iters && ++idle_ticks >= exit_idle_iters) {
          if (!own_idle_vote) {
            mscclpp::atomicOr<uint64_t, mscclpp::scopeDevice>(
                &d_sync->exitReadyMask, own_bit, mscclpp::memoryOrderRelease);
            own_idle_vote = true;
          }
        }
        idle_sleep();
      }
      __syncthreads();
      continue;
    }

    // Work present: revoke this block's exit vote once for the whole
    // burst (a block that keeps its bit set while another is mid-task
    // could strand the completion counter), then process every task
    // already visible in this snapshot without re-polling the fifo
    // between them. Tasks pushed after the snapshot are picked up by the
    // next outer iteration.
    if (threadIdx.x == 0) {
      if (own_idle_vote) {
        mscclpp::atomicAnd<uint64_t, mscclpp::scopeDevice>(
            &d_sync->exitReadyMask, ~own_bit, mscclpp::memoryOrderRelease);
        own_idle_vote = false;
      }
    }
    __syncthreads();
    idle_ticks = 0;

    while (true) {
      // Read the task + args directly (no leader hand-off). All blocks
      // process the same sh_tail sequence; the completion barrier below
      // keeps them in lockstep across tasks in the burst.
      if (threadIdx.x == 0) {
        current_task = fifo.buffer[sh_tail % fifo.size];
        has_current_args = false;
      }
      __syncthreads();

      const TaskType ttype = static_cast<TaskType>(current_task.type_u8());
      if (task_uses_args(ttype)) {
        if (threadIdx.x == 0) {
          const uint32_t idx = current_task.args_index();
          if (idx < (1UL << TaskArgsIndexSize)) {
            *current_args = d_task_args[idx];
            has_current_args = true;
          }
        }
        __syncthreads();
        if (!has_current_args) {
          // Invalid args index: skip the task (advance tail once).
          if (threadIdx.x == 0) {
            publish_tail_progress(fifo.tail, sh_tail + 1);
            ++sh_tail;
          }
          __syncthreads();
          if (sh_tail >= sh_head) break;
          continue;
        }
      }
      __syncthreads();

      dispatch_task(current_task,
                    task_uses_args(ttype) ? current_args : nullptr, bid,
                    gridDim.x, smem_buf);
      __syncthreads();

      // Completion: every block adds 1. The block that reaches gridDim.x
      // performs the task's fence + signal + tail publish, then resets
      // the counter; the others wait for that reset. This is the only
      // cross-block synchronization — no leader, no phase hand-off.
      if (threadIdx.x == 0) {
        uint32_t done =
            mscclpp::atomicFetchAdd<uint32_t, mscclpp::scopeDevice>(
                &d_sync->completedBlocks, 1u, mscclpp::memoryOrderAcqRel) +
            1u;
        if (done == gridDim.x) {
          // Same ordering requirement as the single-block path: all
          // blocks' writes must be visible before the host observes
          // completion.
          tma_fence_async_global();
          __threadfence();
          maybe_signal_ring_write(current_task, current_args);
          publish_tail_progress(fifo.tail, sh_tail + 1);
          mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(
              &d_sync->completedBlocks, 0u, mscclpp::memoryOrderRelease);
        } else {
          while (mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(
                     &d_sync->completedBlocks, mscclpp::memoryOrderAcquire) !=
                 0u) {
          }
        }
        // Every block advances its OWN shared copy of sh_tail — shared
        // memory does not cross blocks and only the last block published
        // the FIFO tail. Without this, non-last blocks re-process the
        // same task (observed: done=1 forever, host spins on a full FIFO).
        ++sh_tail;
      }
      __syncthreads();

      if (sh_tail >= sh_head) break;
    }
  }
}

}  // namespace Device
}  // namespace UKernel
