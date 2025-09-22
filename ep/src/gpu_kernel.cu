#include "common.hpp"
#include "gpu_kernel.cuh"
#include "ring_buffer.cuh"
#include <stdint.h>
#include <stdio.h>

__global__ void gpu_issue_batched_commands(DeviceToHostCmdBuffer* rbs) {
  int const bid = blockIdx.x;
  int const tid = threadIdx.x;
  if (tid != 0) {
    return;
  }
  printf("Device Block %d: Scheduled\n", bid);
  auto rb = &rbs[bid];

#ifdef MEASURE_PER_OP_LATENCY
  uint32_t complete = 0;
  __shared__ unsigned long long cycle_accum_smem;
  __shared__ unsigned int op_count_smem;
  cycle_accum_smem = 0ull;
  op_count_smem = 0u;
#endif

  extern __shared__ unsigned long long start_cycle_smem[];
  bool print_warmup_exit = true;

  rb->cycle_start = 0;
  for (int it = 0; it < kIterations;) {
    uint64_t cur_tail;

#ifdef MEASURE_PER_OP_LATENCY
    cur_tail = rb->volatile_tail();
    if (complete < cur_tail) {
      // __threadfence_system();
      for (int i = complete; i < cur_tail; ++i) {
        if (rb->get_entry(complete).cmd != 0) {
          printf(
              "Device Block %d: Error at complete %u, rb->tail:%lu, expected "
              "0, got %lu\n",
              bid, complete, rb->tail, rb->get_entry(complete).cmd);
          return;
        }
        if (complete >= kWarmupOps) {
          unsigned long long t1 = clock64();
          unsigned long long cycles =
              t1 - start_cycle_smem[complete & kQueueMask];
          cycle_accum_smem += cycles;
          op_count_smem++;
          if (rb->cycle_start == 0) {
            rb->cycle_start = t1;
          }
        }
      }
      complete = cur_tail;
    }
#endif

    unsigned int todo =
        (it + kBatchSize <= kIterations) ? kBatchSize : (kIterations - it);

    // Dynamically send the number of todos to send.
    uint64_t cur_head = rb->head;
    uint64_t my_hdr = cur_head;
    cur_tail = rb->volatile_tail();
    uint64_t free_slots = kMaxInflight - (cur_head - cur_tail);

    if (free_slots >= todo) {
    } else if (free_slots >= 1) {
      todo = free_slots;
    } else {
      continue;
    }

    for (int i = 0; i < todo; ++i) {
      unsigned long long t0 = clock64();
      start_cycle_smem[(my_hdr + i) & kQueueMask] = t0;
      int message_idx = it + i + 1;
      rb->set_buffer(my_hdr + i,
                     TransferCmd{.cmd = (static_cast<uint64_t>(bid + 1) << 32) |
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
                                 .is_combine = false});
    }
    rb->commit_with_head(my_hdr + todo);

    it += todo;
    if (complete > kWarmupOps) {
      if (print_warmup_exit) {
        printf("Device Block %d: Exiting warmup phase\n", bid);
        print_warmup_exit = false;
      }
    }
  }

#ifdef MEASURE_PER_OP_LATENCY
  while (complete < kIterations) {
    if (complete >= kWarmupOps && complete < ld_volatile(&rb->tail)) {
      unsigned long long t1 = clock64();
      cycle_accum_smem += (t1 - start_cycle_smem[complete & kQueueMask]);
      ++op_count_smem;
    } else {
      continue;
    }
    ++complete;
  }

  rb->cycle_accum = cycle_accum_smem;
  rb->op_count = op_count_smem;
#endif
  rb->cycle_end = clock64();
  printf("Device Block %d done\n", bid);
}