/* batched_gpu_to_cpu_bench.cu
 * make
 * CUDA_MODULE_LOADING=EAGER ./batched_gpu_to_cpu_bench
 */

#include <atomic>
#include <chrono>
#include <thread>
#include <tuple>
#include <vector>
#include <assert.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include "common.hpp"
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

struct alignas(128) Fifo {
  uint64_t head;             // Next slot to produce
  uint64_t tail;             // Next slot to consume
  uint64_t buf[kQueueSize];  // Payload buffer (8 bytes).
};

// Head reserves slots; each payload word is its own publication flag. They
// need atomic accesses, but carry no other data requiring acquire/release.
// Tail releases cleared slots back to the GPU, so it orders slot reuse.
template <bool Acquire = false>
__host__ __device__ __forceinline__ uint64_t load_word(uint64_t* ptr) {
#if defined(__CUDA_ARCH__)
  uint64_t value;
  if constexpr (Acquire) {
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
                 : "=l"(value)
                 : "l"(ptr)
                 : "memory");
  } else {
    asm volatile("ld.relaxed.sys.global.u64 %0, [%1];"
                 : "=l"(value)
                 : "l"(ptr)
                 : "memory");
  }
  return value;
#elif defined(__HIP_DEVICE_COMPILE__)
  return __hip_atomic_load(ptr, Acquire ? __ATOMIC_ACQUIRE : __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_SYSTEM);
#else
  return __atomic_load_n(ptr, Acquire ? __ATOMIC_ACQUIRE : __ATOMIC_RELAXED);
#endif
}

template <bool Release = false>
__host__ __device__ __forceinline__ void store_word(uint64_t* ptr,
                                                    uint64_t value) {
#if defined(__CUDA_ARCH__)
  if constexpr (Release) {
    asm volatile("st.release.sys.global.u64 [%0], %1;"
                 :
                 : "l"(ptr), "l"(value)
                 : "memory");
  } else {
    asm volatile("st.relaxed.sys.global.u64 [%0], %1;"
                 :
                 : "l"(ptr), "l"(value)
                 : "memory");
  }
#elif defined(__HIP_DEVICE_COMPILE__)
  __hip_atomic_store(ptr, value, Release ? __ATOMIC_RELEASE : __ATOMIC_RELAXED,
                     __HIP_MEMORY_SCOPE_SYSTEM);
#else
  __atomic_store_n(ptr, value, Release ? __ATOMIC_RELEASE : __ATOMIC_RELAXED);
#endif
}

__device__ unsigned long long cycle_accum[kNumProxyThs] = {0};
__device__ unsigned int op_count[kNumProxyThs] = {0};

__global__ void gpu_issue_batched_commands(Fifo* fifos) {
  int const bid = blockIdx.x;
  int const tid = threadIdx.x;
  Fifo* my_fifo = &fifos[bid];
  if (tid != 0) {
    return;
  }

  uint32_t complete = 0;
  __shared__ unsigned long long cycle_accum_smem;
  __shared__ unsigned int op_count_smem;
  cycle_accum_smem = 0ull;
  op_count_smem = 0u;

  extern __shared__ unsigned long long start_cycle_smem[];

  for (int it = 0; it < kIterations; it += kBatchSize) {
    uint64_t my_hdr;
    uint64_t cur_tail;

    unsigned int const todo =
        (it + kBatchSize <= kIterations) ? kBatchSize : (kIterations - it);

    while (true) {
      // This thread is the sole producer and has reserved exactly it slots.
      uint64_t cur_head = it;
      cur_tail = load_word<true>(&my_fifo->tail);
      if (cur_head - cur_tail + todo <= kQueueSize) {
        store_word(&my_fifo->head, cur_head + todo);
        my_hdr = cur_head;
        break;
        // if (atomicAdd_system(&(my_fifo->head), todo) == cur_head) {
        // my_hdr = cur_head;
        // break; // Successfully reserved a slot
        // }
      }
    }

#pragma unroll
    for (int i = 0; i < todo; ++i) {
      unsigned long long idx = (my_hdr + i) & kQueueMask;
      unsigned long long t0 = clock64();
      unsigned long long cmd =
          (static_cast<uint64_t>(bid) << 32) | (it + i + 1);

      start_cycle_smem[idx] = t0;
      store_word(&my_fifo->buf[idx], cmd);
    }
    __threadfence_system();
    // Consume the observed batch of completions without a host read per item.
    uint64_t completed_tail = load_word(&my_fifo->tail);
    while (complete < my_hdr + todo) {
      uint32_t cidx = complete & kQueueMask;
      if (complete < completed_tail) {
        unsigned long long t1 = clock64();
        unsigned long long cycles = t1 - start_cycle_smem[cidx];
        cycle_accum_smem += cycles;
        op_count_smem++;
        complete++;
      } else {
        break;
      }
    }
  }
  uint64_t completed_tail = load_word(&my_fifo->tail);
  while (complete < kIterations) {
    uint32_t cidx = complete & kQueueMask;
    while (complete >= completed_tail) {
      completed_tail = load_word(&my_fifo->tail);
    }

    unsigned long long t1 = clock64();
    cycle_accum_smem += (t1 - start_cycle_smem[cidx]);
    ++op_count_smem;
    ++complete;
  }

  cycle_accum[bid] = cycle_accum_smem;
  op_count[bid] = op_count_smem;
}

void cpu_proxy(Fifo* fifo, int thread_idx) {
  // printf("CPU thread for block %d started\n", thread_idx);
  pin_thread_to_cpu(thread_idx);

  uint64_t my_tail = 0;
  for (int seen = 0; seen < kIterations; ++seen) {
    while (load_word(&fifo->head) == my_tail) {
      cpu_relax();
    }
    uint64_t idx = my_tail & kQueueMask;
    uint64_t cmd;
    do {
      cmd = load_word(&fifo->buf[idx]);
      cpu_relax();  // Avoid hammering the cacheline.
    } while (cmd == 0);
    uint64_t expected_cmd =
        (static_cast<uint64_t>(thread_idx) << 32) | (seen + 1);
    if (cmd != expected_cmd) {
      fprintf(stderr, "Error: block %d, expected cmd %llu, got %llu\n",
              thread_idx, static_cast<unsigned long long>(expected_cmd),
              static_cast<unsigned long long>(cmd));
      exit(1);
    }
    store_word(&fifo->buf[idx], 0);
    my_tail++;
    store_word<true>(&fifo->tail, my_tail);
  }
}

int main() {
  gpuStream_t stream1;
  GPU_RT_CHECK(gpuStreamCreate(&stream1));

  int clock_rate_khz = 0;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
  GPU_RT_CHECK(
      cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0));
#else
  gpuDeviceProp prop;
  GPU_RT_CHECK(gpuGetDeviceProperties(&prop, 0));
  clock_rate_khz = prop.clockRate;
#endif
  printf("clock rate: %d kHz\n", clock_rate_khz);

  Fifo* fifos;
  GPU_RT_CHECK(
      gpuHostAlloc(&fifos, sizeof(Fifo) * kNumProxyThs, gpuHostAllocMapped));

  for (int i = 0; i < kNumProxyThs; ++i) {
    fifos[i].head = 0;
    fifos[i].tail = 0;
    for (uint32_t j = 0; j < kQueueSize; ++j) {
      fifos[i].buf[j] = 0;  // Initialize the buffer
    }
  }

  // Launch one CPU polling thread per block
  std::vector<std::thread> cpu_threads;
  for (int i = 0; i < kNumProxyThs; ++i) {
    cpu_threads.emplace_back(cpu_proxy, &fifos[i], i);
  }
  auto t0 = std::chrono::high_resolution_clock::now();
  size_t shmem_bytes = kQueueSize * sizeof(unsigned long long);
  gpu_issue_batched_commands<<<kNumProxyThs, kTestNumGpuThPerBlock, shmem_bytes,
                               stream1>>>(fifos);
  GPU_RT_CHECK_ERRORS("gpu_issue_command kernel failed");
  GPU_RT_CHECK(gpuStreamSynchronize(stream1));
  auto t1 = std::chrono::high_resolution_clock::now();

  for (auto& t : cpu_threads) {
    t.join();
  }

  unsigned long long h_cycles[kNumProxyThs];
  unsigned int h_ops[kNumProxyThs];
  gpuMemcpyFromSymbol(h_cycles, cycle_accum, sizeof(h_cycles));
  gpuMemcpyFromSymbol(h_ops, op_count, sizeof(h_ops));

  unsigned int tot_ops = 0;
  double total_us = 0;
  unsigned long long tot_cycles = 0;
  printf("\nPer-block avg latency:\n");
  for (int b = 0; b < kNumProxyThs; ++b) {
    double us = (double)h_cycles[b] * 1000.0 / clock_rate_khz / h_ops[b];
    printf("  Block %d : %.3f µs over %u ops\n", b, us, h_ops[b]);
    total_us += us;
    tot_cycles += h_cycles[b];
    tot_ops += h_ops[b];
  }
  double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double throughput = (double)(kNumProxyThs * kIterations) / (wall_ms * 1000.0);

  printf("\nOverall avg GPU-measured latency  : %.3f µs\n",
         (double)tot_cycles * 1000.0 / clock_rate_khz / tot_ops);
  printf("Total cycles                       : %llu\n", tot_cycles);
  printf("Total ops                          : %u\n", tot_ops);
  printf("End-to-end Wall-clock time        : %.3f ms\n", wall_ms);
  printf("Throughput                        : %.2f Mops/s\n", throughput);

  GPU_RT_CHECK(gpuFreeHost(fifos));
  GPU_RT_CHECK(gpuStreamDestroy(stream1));
}
