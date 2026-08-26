#pragma once

#include "c2d_fifo_device.h"
#include "task.h"
#include <cassert>
#include <cstdint>

namespace UKernel {
namespace Device {

struct alignas(16) MultiBlockSync {
  // Monotonic per-task completion counter, never reset. Every block adds
  // 1 after its slice of a task (and also for skipped invalid-args tasks,
  // keeping the count aligned with the FIFO tail). Task N's barrier is
  // complete once the counter reaches gridDim.x * (N+1); the block whose
  // add crosses that threshold is the "last block" and performs the task's
  // fence + signal + tail publish, and everyone else waits for the
  // threshold before advancing to the next task.
  //
  // Unlike a reset-to-0 counter, a slow block can never leak its +1 into
  // the next task's barrier: task N's threshold cannot be reached until
  // every block has added for N, so a late block is absorbed into the
  // correct task's count (with the old counter, repeated-task worker
  // reduce produced wrong results). Zeroed by the host on every (re)launch.
  uint64_t completionCount;
  // Exit rendezvous mask (bit i belongs to block i). A block sets its bit
  // once its idle grace has elapsed and clears it whenever it sees work.
  // The mask can therefore only become all-ones when the FIFO is
  // quiescent and no task is in flight; at that point every block exits
  // together. Tasks pushed into the exit race window stay in the FIFO
  // (head > tail) and are recovered by the host's relaunch. 64-bit so up
  // to 64 blocks are representable (createWorker rejects anything more).
  uint64_t exitReadyMask;
  // FIFO head snapshot refreshed by block 0 (the only block that reads
  // the host-pinned head pointer). Other blocks poll this device-resident
  // copy instead — reading the host-pinned head from N blocks per
  // iteration turns every poll into N PCIe round trips and contends with
  // the IPC put engine (measured: 16-block allreduce dropped to ~15 GB/s
  // vs ~48 GB/s with a single head reader).
  uint64_t headHint;
  // Set by block 0 at kernel entry once the completion counter has been
  // re-anchored to the FIFO tail (gridDim.x * tail); all other blocks
  // wait for it before processing tasks. The host zeroes this on every
  // (re)launch, so a fresh grid always re-anchors.
  uint64_t anchorReady;
};

__device__ __forceinline__ void run_copy(TaskArgs const& a, uint32_t block_id,
                                         uint32_t num_blocks,
                                         void* smem_buf);

template <typename T>
__device__ __forceinline__ void run_reduce(TaskArgs const& a, uint32_t block_id,
                                           uint32_t num_blocks,
                                           void* smem_buf);

__global__ void multiPersistentKernel(mscclpp::C2DDeviceHandle<Task>* c2d_fifos,
                                      TaskArgs* d_task_args, bool* should_stop,
                                      MultiBlockSync* d_sync, bool* exited_flag,
                                      uint32_t exit_idle_iters);

__global__ void benchDispatchNopKernel();
__global__ void benchDispatchCopyFp32Kernel(TaskArgs args);
__global__ void benchDispatchReduceFp32Kernel(TaskArgs args);

}  // namespace Device
}  // namespace UKernel
