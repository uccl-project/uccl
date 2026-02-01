#include "operators/operator.cuh"
#include "persistent.h"
#include "task.h"
#include <chrono>
#include <cstdio>
#include <cuda_bf16.h>
#include <random>

using namespace UKernel::Compute;
using bf16 = __nv_bfloat16;

static inline uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void cpu_gemm(bf16* A, bf16* B, bf16* C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += __bfloat162float(A[i * K + k]) * __bfloat162float(B[k * N + j]);
      }
      C[i * N + j] = __float2bfloat16(sum);
    }
  }
}

bool validate(bf16* gpu_result, bf16* cpu_result, int M, int N,
              float tol = 0.1f) {
  int errors = 0;
  for (int i = 0; i < M * N && errors < 10; i++) {
    float gpu_val = __bfloat162float(gpu_result[i]);
    float cpu_val = __bfloat162float(cpu_result[i]);
    float diff = fabsf(gpu_val - cpu_val);
    float rel_err = diff / (fabsf(cpu_val) + 1e-6f);
    if (rel_err > tol && diff > 0.5f) {
      printf("Mismatch at %d: GPU=%.4f CPU=%.4f\n", i, gpu_val, cpu_val);
      errors++;
    }
  }
  return errors == 0;
}

int main() {
  constexpr int M = 256, N = 256, K = 256;
  printf("=== TK GEMM Benchmark: M=%d N=%d K=%d ===\n", M, N, K);

  size_t size_A = M * K * sizeof(bf16);
  size_t size_B = K * N * sizeof(bf16);
  size_t size_C = M * N * sizeof(bf16);

  bf16* h_A = (bf16*)malloc(size_A);
  bf16* h_B = (bf16*)malloc(size_B);
  bf16* h_C = (bf16*)malloc(size_C);
  bf16* h_C_ref = (bf16*)malloc(size_C);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int i = 0; i < M * K; i++) h_A[i] = __float2bfloat16(dist(gen));
  for (int i = 0; i < K * N; i++) h_B[i] = __float2bfloat16(dist(gen));
  memset(h_C, 0, size_C);

  bf16 *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, size_A);
  cudaMalloc(&d_B, size_B);
  cudaMalloc(&d_C, size_C);
  cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
  cudaMemset(d_C, 0, size_C);

  TkMatmulGlobals h_globals(d_A, d_B, d_C, M, N, K);
  TkMatmulGlobals* d_globals;
  cudaMalloc(&d_globals, sizeof(TkMatmulGlobals));
  cudaMemcpy(d_globals, &h_globals, sizeof(TkMatmulGlobals),
             cudaMemcpyHostToDevice);

  TaskManager::instance().init(1, 1, 64);

  PersistentKernelConfig cfg;
  cfg.numBlocks = 1;
  cfg.threadsPerBlock = TK_NUM_THREADS;
  cfg.fifoCapacity = 16;
  cfg.smemSize = 200000;

  cudaFuncSetAttribute(basePersistentKernel<Task>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       cfg.smemSize);

  PersistentKernel<Task> kernel(cfg);
  kernel.launch();

  constexpr int TILE_M = TK_M_BLOCK * TK_BLOCK_SIZE;
  constexpr int TILE_N = TK_N_BLOCK * TK_BLOCK_SIZE;
  int num_tile_rows = (M + TILE_M - 1) / TILE_M;
  int num_tile_cols = (N + TILE_N - 1) / TILE_N;

  uint64_t t0 = now_ns();
  uint64_t last_task_id = 0;
  for (int tr = 0; tr < num_tile_rows; tr++) {
    for (int tc = 0; tc < num_tile_cols; tc++) {
      GemmArgs gemm_args{d_globals, (uint32_t)tr, (uint32_t)tc, 0};
      Task task =
          TaskManager::instance().create_gemm_task(gemm_args, DataType::Fp16, 0);
      last_task_id = kernel.submit(task);
    }
  }
  while (!kernel.is_done(0, last_task_id)) {}
  uint64_t t1 = now_ns();

  printf("GEMM completed: %.2f us (%d tiles)\n", (t1 - t0) / 1e3,
         num_tile_rows * num_tile_cols);

  cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
  cpu_gemm(h_A, h_B, h_C_ref, M, N, K);

  if (validate(h_C, h_C_ref, M, N)) {
    printf("PASSED\n");
  } else {
    printf("FAILED\n");
  }

  kernel.stop();
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_globals);
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);
  return 0;
}
