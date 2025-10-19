#include "common.hpp"
#include "fifo_device.hpp"
#include "gpu_kernel.cuh"
#include "gpu_kernel_fifo.cuh"
#include <stdint.h>
#include <stdio.h>

// FIFO-based GPU kernel - each block uses its own FIFO
// Implements kMaxInflight limiting with proper polling and latency measurement
__global__ void gpu_issue_batched_commands_fifo(
    mscclpp::FifoDeviceHandle* fifos, uint64_t* cycle_start_out,
    uint64_t* cycle_end_out, uint64_t* cycle_accum_out,
    uint32_t* op_count_out) {
  int const bid = blockIdx.x;
  int const tid = threadIdx.x;
  int const num_threads = blockDim.x;

  if (tid == 0) {
    printf("Device Block %d: Scheduled with %d threads (FIFO mode)\n", bid,
           num_threads);
  }

  // Get this block's FIFO (shared by all threads)
  mscclpp::FifoDeviceHandle& fifo = fifos[bid];

  // Track completed operations and latency metrics (per-thread)
  uint32_t completed = 0;

  __shared__ unsigned long long cycle_accum_smem;
  __shared__ unsigned int op_count_smem;
  extern __shared__ unsigned long long start_cycle_smem[];
  if (tid == 0) {
    cycle_accum_smem = 0ull;
    op_count_smem = 0u;
  }
  __syncthreads();

  // Track in-flight requests using circular buffer (per-thread)
  // When we reach kMaxInflight, we poll the oldest to maintain the limit
  uint64_t head_buffer[kMaxInflight];
  uint32_t head_write_idx = 0;
  uint32_t head_read_idx = 0;
  uint32_t inflight_count = 0;
  uint64_t block_cycle_start = 0;

  // Each thread processes kIterations/num_threads operations
  int my_iterations = (kIterations + num_threads - 1) / num_threads;

  for (int local_it = 0; local_it < my_iterations; ++local_it) {
    int it = tid + local_it * num_threads;
    if (it >= kIterations) break;

    unsigned long long t0 = clock64();
    start_cycle_smem[it & kQueueMask] = t0;

    int message_idx = it + 1;

    // Push to FIFO
    mscclpp::ProxyTrigger trigger;
    trigger.fst = (static_cast<uint64_t>(bid + 1) << 32) |
                  (static_cast<uint64_t>(CmdType::WRITE) & 0xFFFFFFFF);
    trigger.snd = message_idx;
    uint64_t head = fifo.push(trigger);

    // Store head in circular buffer for tracking
    head_buffer[head_write_idx] = head;
    head_write_idx = (head_write_idx + 1) % kMaxInflight;
    inflight_count++;

    // Once we reach kMaxInflight, poll the oldest request to keep under limit
    if (inflight_count >= kMaxInflight) {
      uint64_t oldest_head = head_buffer[head_read_idx];
      head_read_idx = (head_read_idx + 1) % kMaxInflight;

      // Wait for the oldest request to be consumed by host proxy
      fifo.sync(oldest_head, -1);

      // Measure latency for completed operation
      int abs_it = tid + completed * num_threads;
      if (abs_it >= kWarmupOps && abs_it < kIterations) {
        unsigned long long t1 = clock64();
        unsigned long long cycles = t1 - start_cycle_smem[abs_it & kQueueMask];
        atomicAdd((unsigned long long*)&cycle_accum_smem, cycles);
        atomicAdd(&op_count_smem, 1u);
        if (block_cycle_start == 0) {
          block_cycle_start = t1;
        }
      }
      completed++;
      inflight_count--;
    }
  }

  // Wait for all remaining in-flight operations to complete
  while (inflight_count > 0) {
    uint64_t oldest_head = head_buffer[head_read_idx];
    head_read_idx = (head_read_idx + 1) % kMaxInflight;

    fifo.sync(oldest_head, -1);
    inflight_count--;

    int abs_it = tid + completed * num_threads;
    if (abs_it < kIterations && abs_it >= kWarmupOps) {
      unsigned long long t1 = clock64();
      unsigned long long cycles = t1 - start_cycle_smem[abs_it & kQueueMask];
      atomicAdd((unsigned long long*)&cycle_accum_smem, cycles);
      atomicAdd(&op_count_smem, 1u);
    }
    completed++;
  }

  __syncthreads();

  if (tid == 0) {
    if (cycle_start_out) cycle_start_out[bid] = block_cycle_start;
    if (cycle_end_out) cycle_end_out[bid] = clock64();
    if (cycle_accum_out) cycle_accum_out[bid] = cycle_accum_smem;
    if (op_count_out) op_count_out[bid] = op_count_smem;

    printf(
        "Device Block %d done (%d threads pushed %d operations, measured %u "
        "ops)\n",
        bid, num_threads, kIterations, op_count_smem);
  }
}

// Launcher function implementation
cudaError_t launch_gpu_issue_batched_commands_fifo(
    int blocks, int threads_per_block, size_t shmem_bytes, cudaStream_t stream,
    mscclpp::FifoDeviceHandle* d_fifos, uint64_t* cycle_start,
    uint64_t* cycle_end, uint64_t* cycle_accum, uint32_t* op_count) {
  gpu_issue_batched_commands_fifo<<<blocks, threads_per_block, shmem_bytes,
                                    stream>>>(d_fifos, cycle_start, cycle_end,
                                              cycle_accum, op_count);

  return cudaGetLastError();
}
