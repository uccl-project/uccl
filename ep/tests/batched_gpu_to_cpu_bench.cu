/* batched_gpu_to_cpu_bench.cu
 * make
 * CUDA_MODULE_LOADING=EAGER ./batched_gpu_to_cpu_bench
 */

#include <algorithm>
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

constexpr int kBenchmarkIterations = 400000;
constexpr int kWarmupTrials = 2;
constexpr int kMeasuredTrials = 7;

__device__ __forceinline__ unsigned long long timestamp() {
#ifdef __CUDA_ARCH__
  // The nanosecond timer does not need a fixed SM clock for conversion.
  unsigned long long ticks;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ticks) : : "memory");
  return ticks;
#else
  return clock64();
#endif
}

struct alignas(128) Fifo {
  // Using volatile and avoiding atomics.
  volatile uint64_t head;             // Next slot to produce
  volatile uint64_t tail;             // Next slot to consume
  uint64_t volatile buf[kQueueSize];  // Payload buffer (8 bytes).
};

__device__ __forceinline__ uint64_t ld_volatile(uint64_t volatile* ptr) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#ifdef __CUDA_ARCH__
  uint64_t ans;
  asm volatile("ld.volatile.global.u64 %0, [%1];"
               : "=l"(ans)
               : "l"(ptr)
               : "memory");
  return ans;
#elif defined(__HIP_DEVICE_COMPILE__)
  uint64_t ans;
  ans = __builtin_nontemporal_load(ptr);
  return ans;
#else
#error "Not supported"
#endif
#else
  return *((volatile uint64_t const*)ptr);
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

  for (int it = 0; it < kBenchmarkIterations; it += kBatchSize) {
    uint64_t my_hdr;
    uint64_t cur_tail;

    unsigned int const todo = (it + kBatchSize <= kBenchmarkIterations)
                                  ? kBatchSize
                                  : (kBenchmarkIterations - it);

    while (true) {
      // CPU does not modify the head.
      uint64_t cur_head = my_fifo->head;
      cur_tail = ld_volatile(&my_fifo->tail);
      // Snapshot the CPU's completion counter once per batch. Retire timestamps
      // before reusing their ring slots, even if the CPU has raced ahead.
      unsigned long long completed_at = timestamp();
      while (complete < cur_tail) {
        cycle_accum_smem +=
            completed_at - start_cycle_smem[complete & kQueueMask];
        ++op_count_smem;
        ++complete;
      }
      if (cur_head - complete + todo <= kQueueSize) {
        my_fifo->head = cur_head + todo;
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
      unsigned long long t0 = timestamp();
      unsigned long long cmd =
          (static_cast<uint64_t>(bid) << 32) | (it + i + 1);

      start_cycle_smem[idx] = t0;
      my_fifo->buf[idx] = cmd;
    }
    __threadfence_system();
  }
  while (complete < kBenchmarkIterations) {
    uint64_t cur_tail = ld_volatile(&my_fifo->tail);
    unsigned long long completed_at = timestamp();
    while (complete < cur_tail) {
      cycle_accum_smem +=
          completed_at - start_cycle_smem[complete & kQueueMask];
      ++op_count_smem;
      ++complete;
    }
  }

  cycle_accum[bid] = cycle_accum_smem;
  op_count[bid] = op_count_smem;
}

void cpu_proxy(Fifo* fifo, int thread_idx, std::atomic<int>* ready) {
  // printf("CPU thread for block %d started\n", thread_idx);
  if (!pin_thread_to_cpu(thread_idx)) {
    fprintf(stderr, "Unable to pin proxy thread %d\n", thread_idx);
    std::abort();
  }
  ready->fetch_add(1, std::memory_order_release);

  uint64_t my_tail = 0;
  for (int seen = 0; seen < kBenchmarkIterations; ++seen) {
    while (fifo->head == my_tail) {
      cpu_relax();
    }
    uint64_t idx = my_tail & kQueueMask;
    uint64_t cmd;
    do {
      cmd = fifo->buf[idx];
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
    fifo->buf[idx] = 0;
    // Publish a reusable slot only after clearing its payload (also on ARM).
    std::atomic_thread_fence(std::memory_order_release);
    my_tail++;
    fifo->tail = my_tail;
    // _mm_clflush(&(fifo->tail));
  }
}

struct Trial {
  double latency_us;
  double throughput;
  double wall_ms;
};

Trial run_trial(Fifo* fifos, gpuStream_t stream, double ticks_per_us) {
  for (int i = 0; i < kNumProxyThs; ++i) {
    fifos[i].head = 0;
    fifos[i].tail = 0;
    for (uint32_t j = 0; j < kQueueSize; ++j) fifos[i].buf[j] = 0;
  }
  std::atomic<int> ready{0};
  std::vector<std::thread> cpu_threads;
  for (int i = 0; i < kNumProxyThs; ++i)
    cpu_threads.emplace_back(cpu_proxy, &fifos[i], i, &ready);
  // Thread creation and CPU affinity setup are outside the measured interval.
  while (ready.load(std::memory_order_acquire) != kNumProxyThs)
    std::this_thread::yield();
  auto t0 = std::chrono::steady_clock::now();
  size_t shmem_bytes = kQueueSize * sizeof(unsigned long long);
  gpu_issue_batched_commands<<<kNumProxyThs, kTestNumGpuThPerBlock, shmem_bytes,
                               stream>>>(fifos);
  GPU_RT_CHECK_ERRORS("gpu_issue_command kernel failed");
  GPU_RT_CHECK(gpuStreamSynchronize(stream));
  auto t1 = std::chrono::steady_clock::now();
  for (auto& thread : cpu_threads) thread.join();

  unsigned long long cycles[kNumProxyThs];
  unsigned int ops[kNumProxyThs];
  GPU_RT_CHECK(gpuMemcpyFromSymbol(cycles, cycle_accum, sizeof(cycles)));
  GPU_RT_CHECK(gpuMemcpyFromSymbol(ops, op_count, sizeof(ops)));
  unsigned long long total_cycles = 0;
  unsigned int total_ops = 0;
  for (int i = 0; i < kNumProxyThs; ++i) {
    assert(ops[i] == kBenchmarkIterations);
    total_cycles += cycles[i];
    total_ops += ops[i];
  }
  double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  return {double(total_cycles) / ticks_per_us / total_ops,
          double(total_ops) / (wall_ms * 1000.0), wall_ms};
}

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

int main() {
  gpuStream_t stream;
  GPU_RT_CHECK(gpuStreamCreate(&stream));
#if defined(CUDART_VERSION)
  double ticks_per_us = 1000.0;  // CUDA globaltimer is in nanoseconds.
  printf("Timer: CUDA globaltimer (nanoseconds)\n");
#else
  gpuDeviceProp prop;
  GPU_RT_CHECK(gpuGetDeviceProperties(&prop, 0));
  double ticks_per_us = prop.clockRate / 1000.0;
  printf("Timer: clock64 at %d kHz\n", prop.clockRate);
#endif
  Fifo* fifos;
  GPU_RT_CHECK(
      gpuHostAlloc(&fifos, sizeof(Fifo) * kNumProxyThs, gpuHostAllocMapped));
  int warmups = 0;
  auto warmup_start = std::chrono::steady_clock::now();
  while (warmups < kWarmupTrials ||
         std::chrono::steady_clock::now() - warmup_start <
             std::chrono::seconds(1)) {
    run_trial(fifos, stream, ticks_per_us);
    ++warmups;
  }
  std::vector<double> latencies, throughputs, wall_times;
  for (int trial = 0; trial < kMeasuredTrials; ++trial) {
    auto sample = run_trial(fifos, stream, ticks_per_us);
    latencies.push_back(sample.latency_us);
    throughputs.push_back(sample.throughput);
    wall_times.push_back(sample.wall_ms);
    printf("Trial %d: %.3f us, %.2f Mops/s\n", trial + 1, sample.latency_us,
           sample.throughput);
  }
  printf("Medians of %d trials after %d warmups (%d ops/block/trial)\n",
         kMeasuredTrials, warmups, kBenchmarkIterations);
  printf("Overall avg GPU-measured latency  : %.3f µs\n", median(latencies));
  printf("End-to-end Wall-clock time        : %.3f ms\n", median(wall_times));
  printf("Throughput                        : %.2f Mops/s\n",
         median(throughputs));
  GPU_RT_CHECK(gpuFreeHost(fifos));
  GPU_RT_CHECK(gpuStreamDestroy(stream));
}
