#include "common.hpp"
#include "gpu_kernel.cuh"
#include "ring_buffer.cuh"
#include <stdint.h>
#include <stdio.h>

__global__ void gpu_issue_batched_commands(DeviceToHostCmdBuffer* rbs) {
  int const bid = blockIdx.x;
  int const tid = threadIdx.x;
  int const num_threads = blockDim.x;

  if (tid == 0) {
    printf("Device Block %d: Scheduled with %d threads\n", bid, num_threads);
  }
  __syncthreads();

  auto rb = &rbs[bid];
  uint32_t completed = tid;

  __shared__ unsigned long long cycle_accum_smem;
  __shared__ unsigned int op_count_smem;
  __shared__ uint64_t shared_cycle_start;
  __shared__ unsigned long long start_cycle_smem[kQueueSize];

#define kInflightSlotSize (kMaxInflight * kNumThPerBlock)
#define kInflightSlotMask (kInflightSlotSize - 1)
  uint64_t inflight_slots[kInflightSlotSize];

  if (tid == 0) {
    cycle_accum_smem = 0ull;
    op_count_smem = 0u;
    shared_cycle_start = 0;
    rb->cycle_start = 0;
  }
  __syncthreads();

  // Each thread dispatches its own commands with stride
  for (int it = tid; it < kIterations; it += num_threads) {
    uint64_t cur_tail = rb->volatile_tail();

    // Check if there are any completed commands
    while (completed < cur_tail) {
      if (completed >= kWarmupOps) {
        unsigned long long t1 = clock64();
        uint64_t inflight_slot =
            inflight_slots[(completed / kNumThPerBlock) & kInflightSlotMask];
        unsigned long long t0 = start_cycle_smem[inflight_slot & kQueueMask];
        unsigned long long cycles = t1 - t0;
        atomicAdd((unsigned long long*)&cycle_accum_smem, cycles);
        atomicAdd(&op_count_smem, 1u);
        atomicCAS((unsigned long long*)&shared_cycle_start, 0ULL, t1);
      }
      completed += num_threads;
    }

    // Check global ring buffer state and wait if necessary
    while (true) {
      uint64_t cur_head = rb->head;
      cur_tail = rb->volatile_tail();
      uint64_t inflight = cur_head - cur_tail;

      if (inflight < kInflightSlotSize) {
        // Record start time
        unsigned long long t0 = clock64();
        int message_idx = it + 1;

        uint64_t my_slot = cur_head;
        // Create the command
        TransferCmd cmd{.cmd_type = CmdType::WRITE,
                        .cmd = (static_cast<uint64_t>(bid + 1) << 32) |
                               (message_idx & 0xFFFFFFFF),
                        .dst_rank = 1,
                        .dst_gpu = 0,
                        .src_ptr = 0,
                        .bytes = kObjectSize,
                        .req_rptr = 0,
                        .req_lptr = 0,
                        .warp_id = 0,
                        .lane_id = 0,
                        .message_idx = message_idx,
                        .is_atomic = false,
                        .value = 0,
                        .is_combine = false};

        // Space available, atomically reserve and commit
        rb->atomic_set_and_commit(cmd, &my_slot);
        start_cycle_smem[my_slot & kQueueMask] = t0;
        inflight_slots[(it / kNumThPerBlock) & kInflightSlotMask] = my_slot;
        break;
      } else {
        // Otherwise, it is gonna hang here.
        __nanosleep(64);
      }
    }
  }

  // Polling the remaining requests.
  while (completed < kIterations) {
    uint64_t cur_tail = rb->volatile_tail();
    while (completed < cur_tail) {
      if (completed >= kWarmupOps) {
        unsigned long long t1 = clock64();
        uint64_t inflight_slot =
            inflight_slots[(completed / kNumThPerBlock) & kInflightSlotMask];
        unsigned long long t0 = start_cycle_smem[inflight_slot & kQueueMask];
        unsigned long long cycles = t1 - t0;
        atomicAdd((unsigned long long*)&cycle_accum_smem, cycles);
        atomicAdd(&op_count_smem, 1u);
        atomicCAS((unsigned long long*)&shared_cycle_start, 0ULL, t1);
      }
      completed += num_threads;
    }
  }

  __syncthreads();

  if (tid == 0) {
    rb->cycle_start = shared_cycle_start;
    rb->cycle_accum = cycle_accum_smem;
    rb->op_count = op_count_smem;
    rb->cycle_end = clock64();
    printf("Device Block %d done (%d threads, measured %u ops)\n", bid,
           num_threads, op_count_smem);
  }
}