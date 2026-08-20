#pragma once

#include "c2d_fifo_device.h"
#include "task.h"
#include <cassert>
#include <cstdint>

namespace UKernel {
namespace Device {

struct alignas(16) MultiBlockSync {
  // Per-task completion counter. Every block adds 1 after its slice; the
  // block that observes gridDim.x is the "last block" and performs the
  // task's fence + signal + tail publish, then resets this to 0. Other
  // blocks wait for the reset (the only cross-block synchronization on the
  // task path — no leader, no phase hand-off).
  uint32_t completedBlocks;
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
