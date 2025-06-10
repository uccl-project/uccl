#include "common.hpp"
#include "gpu_kernel.cuh"
#include "ring_buffer.cuh"
#include <cuda_pipeline.h>
#include <stdint.h>
#include <stdio.h>

__device__ __forceinline__ uint64_t ld_volatile(uint64_t* ptr) {
  uint64_t ans;
  asm volatile("ld.volatile.global.u64 %0, [%1];"
               : "=l"(ans)
               : "l"(ptr)
               : "memory");
  return ans;
}

__device__ __forceinline__ void populate_data(void* d_ptr, int bid) {
  printf("d_ptrs[%d] = %p\n", bid, d_ptr);
  unsigned char* base_ptr = reinterpret_cast<unsigned char*>(d_ptr);
  for (int i = 0; i < kBatchSize; ++i) {
    unsigned char* obj_ptr = base_ptr + i * kObjectSize;
    int* int_fields = reinterpret_cast<int*>(obj_ptr);
    int_fields[0] = bid;
    int_fields[1] = i;
    if (kObjectSize > 8) {
      memset(obj_ptr + 8, 0xAB, kObjectSize - 8);
    }
  }
  __threadfence_system();
}

__global__ void gpu_issue_batched_commands(RingBuffer* rbs, void** d_ptrs) {
  int const bid = blockIdx.x;
  int const tid = threadIdx.x;
  RingBuffer* rb = &rbs[bid];
  void* const d_ptr = d_ptrs[bid];
  if (tid != 0) {
    return;
  }

  populate_data(d_ptr, bid);

#ifdef MEASURE_PER_OP_LATENCY
  uint32_t complete = 0;
  __shared__ unsigned long long cycle_accum_smem;
  __shared__ unsigned int op_count_smem;
  cycle_accum_smem = 0ull;
  op_count_smem = 0u;
#endif

  extern __shared__ unsigned long long start_cycle_smem[];

  for (int it = 0; it < kIterations;) {
    uint64_t my_hdr;
    uint64_t cur_tail;

    unsigned int const initial_todo =
        (it + kBatchSize <= kIterations) ? kBatchSize : (kIterations - it);
    unsigned int todo = initial_todo;

    // Dynamically send the number of todos to send.
    while (true) {
      uint64_t cur_head = rb->head;
      cur_tail = ld_volatile(&rb->tail);
      uint64_t free_slots = kQueueSize - (cur_head - cur_tail);

      if (free_slots >= todo) {
        rb->head = cur_head + todo;
        my_hdr = cur_head;
        break;
      }

#ifdef False
      // Doesn't seem to work.
      //  The idea is that if there are fewer slots than batch_size
      // Just send whatever is allowed.
      if (free_slots >= 1) {
        todo = free_slots;
        rb->head = cur_head + todo;
        my_hdr = cur_head;
        break;
      }
#endif
      /* Spin */
    }

#pragma unroll
    for (int i = 0; i < todo; ++i) {
      uint32_t idx = (my_hdr + i) & kQueueMask;
      unsigned long long t0 = clock64();
      rb->buf[idx].cmd = (static_cast<uint64_t>(bid) << 32) | (it + i + 1);
      rb->buf[idx].dst_rank = bid;
      rb->buf[idx].dst_gpu = 0;
      rb->buf[idx].src_ptr =
          reinterpret_cast<void*>(static_cast<uintptr_t>(it + i + 1));
      rb->buf[idx].bytes = 8;

      start_cycle_smem[idx] = t0;
    }
    __threadfence_system();

#ifdef MEASURE_PER_OP_LATENCY
    while (complete < my_hdr + todo) {
      uint32_t cidx = complete & kQueueMask;
      if (complete < ld_volatile(&rb->tail)) {
        unsigned long long t1 = clock64();
        unsigned long long cycles = t1 - start_cycle_smem[cidx];
        cycle_accum_smem += cycles;
        op_count_smem++;
        complete++;
      } else {
        break;
      }
    }
#endif
    it += todo;
  }

#ifdef MEASURE_PER_OP_LATENCY
  while (complete < kIterations) {
    uint32_t cidx = complete & kQueueMask;
    while (complete >= ld_volatile(&rb->tail)) { /* spin */
    }

    unsigned long long t1 = clock64();
    cycle_accum_smem += (t1 - start_cycle_smem[cidx]);
    ++op_count_smem;
    ++complete;
  }

  rb->cycle_accum = cycle_accum_smem;
  rb->op_count = op_count_smem;

#endif

  printf("Device Block %d done\n", bid);
}