#include "common.hpp"
#include "ring_buffer.cuh"
#include "gpu_kernel.cuh"
#include <cuda_pipeline.h>
#include <stdio.h>
#include <stdint.h>

__device__ __forceinline__ uint64_t ld_volatile(uint64_t* ptr) {
    uint64_t ans;
    asm volatile("ld.volatile.global.u64 %0, [%1];"
                 : "=l"(ans)
                 : "l"(ptr)
                 : "memory");
    return ans;
}

__global__ void gpu_issue_batched_commands(RingBuffer* rbs) {
    int const bid = blockIdx.x;
    int const tid = threadIdx.x;
    RingBuffer * rb = &rbs[bid];
    if (tid != 0) {
        return;
    }

#ifdef MEASURE_PER_OP_LATENCY
    uint32_t complete = 0;
    __shared__ unsigned long long cycle_accum_smem;
    __shared__ unsigned int op_count_smem;
    cycle_accum_smem = 0ull;
    op_count_smem = 0u;
#endif

    extern __shared__ unsigned long long start_cycle_smem[];

    for (int it = 0; it < kIterations; it += kBatchSize) {
        uint64_t my_hdr;
        uint64_t cur_tail;

        unsigned int const todo =
            (it + kBatchSize <= kIterations) ? kBatchSize : (kIterations - it);

        while (true) {
            uint64_t cur_head = rb->head;
            cur_tail = ld_volatile(&rb->tail);
            if (cur_head - cur_tail + todo <= kQueueSize) {
                rb->head = cur_head + todo;
                my_hdr = cur_head;
                break;
            }
        }

#pragma unroll
        for (int i = 0; i < todo; ++i) {
            uint32_t idx = (my_hdr + i) & kQueueMask;
            unsigned long long t0 = clock64();
            rb->buf[idx].cmd      = (static_cast<uint64_t>(bid) << 32) | (it + i + 1);
            rb->buf[idx].dst_rank = bid;
            rb->buf[idx].dst_gpu  = 0;
            rb->buf[idx].src_ptr  = reinterpret_cast<void*>(static_cast<uintptr_t>(it + i + 1));
            rb->buf[idx].bytes    = 8;

            start_cycle_smem[idx] = t0;
        }
        __threadfence_system();
#ifdef MEASURE_PER_OP_LATENCY
        while (complete < my_hdr + todo) {
            uint32_t cidx = complete & kQueueMask;
            if (complete < rb->tail) {
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
    }

    __threadfence_system();

#ifdef MEASURE_PER_OP_LATENCY
    while (complete < kIterations) {
        uint32_t cidx = complete & kQueueMask;
        while (complete >= rb->tail) { /* spin */ }

        unsigned long long t1 = clock64();
        cycle_accum_smem += (t1 - start_cycle_smem[cidx]);
        ++op_count_smem;
        ++complete;
    }

    rb->cycle_accum = cycle_accum_smem;
    rb->op_count = op_count_smem;

#endif
}