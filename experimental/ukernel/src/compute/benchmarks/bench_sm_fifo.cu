#include "sm_fifo.h"
#include "task.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

using Task = UKernel::Compute::Task;

// ------------------------------------------------------------
// Kernel: multi-producer -> single-consumer benchmark
// block0..block(P-1): producers
// block(P): consumer
// ------------------------------------------------------------
__global__ void bench_sm_fifo_kernel(mscclpp::SmDeviceHandle<Task> fifo,
                                     int iters_per_producer, int num_producers,
                                     uint64_t* done_flag) {
  int bid = blockIdx.x;

  // Producer blocks (only thread0 pushes)
  if (bid < num_producers) {
    if (threadIdx.x == 0) {
      for (int i = 0; i < iters_per_producer; i++) {
        Task t(UKernel::Compute::TaskType::BenchNop,
               UKernel::Compute::DataType::Fp32,
               /*blockIndex*/ bid,
               /*argsIndex*/ i);

        fifo.push(t);
      }
    }
    return;
  }

  // Consumer block (only thread0 pops)
  if (bid == num_producers) {
    if (threadIdx.x == 0) {
      int total_tasks = iters_per_producer * num_producers;
      int popped = 0;

      while (popped < total_tasks) {
        Task* ptr = fifo.poll();
        if (ptr) {
          fifo.pop();
          popped++;
        }
      }

      // signal done
      atomicExch((unsigned long long*)done_flag, 1ULL);
    }
  }
}

void run_bench(int num_producers, int iters) {
  constexpr int fifo_cap = 4096;

  printf("\n=============================\n");
  printf("SM FIFO Benchmark\n");
  printf("Producers : %d\n", num_producers);
  printf("Consumer  : 1\n");
  printf("Tasks/prod: %d\n", iters);
  printf("Total     : %d\n", num_producers * iters);
  printf("=============================\n");

  // Allocate FIFO
  mscclpp::SmFifo<Task> fifo(fifo_cap);
  auto handle = fifo.deviceHandle();

  // Done flag
  uint64_t* d_done;
  cudaMalloc(&d_done, sizeof(uint64_t));
  cudaMemset(d_done, 0, sizeof(uint64_t));

  // CUDA timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  // Launch: num_producers + 1 consumer
  int blocks = num_producers + 1;
  int threads = 64;

  bench_sm_fifo_kernel<<<blocks, threads>>>(handle, iters, num_producers,
                                            d_done);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  double sec = ms * 1e-3;
  double total_tasks = double(num_producers) * double(iters);

  printf("Elapsed    : %.3f ms\n", ms);
  printf("Throughput : %.2f K tasks/s\n", (total_tasks / sec) / 1e3);

  cudaFree(d_done);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
  int num_producers = 1;
  int iters = 1000;

  if (argc >= 2) num_producers = atoi(argv[1]);
  if (argc >= 3) iters = atoi(argv[2]);

  printf("Usage: %s [num_producers] [iters_per_producer]\n", argv[0]);

  run_bench(num_producers, iters);

  printf("\nDone.\n");
  return 0;
}
