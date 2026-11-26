// What cuBLAS delivers on the DeepSeek-V2-Lite grouped-MoE shapes, on one L4.
//
// This is the floor the fused kernel's hand-written MMA has to be measured
// against, and the compute half of the fusion budget. Shapes are the per-rank
// grouped GEMM at EP=4: 16 local experts, each [M_e, 2048] x [2048, 2816] bf16,
// where M_e = 6T/16 for T tokens per rank (6 top-k assignments per token,
// spread over 4 ranks, landing on 16 local experts).
//
// Also prices the permute kernel a non-fused path needs -- gathering each
// expert's rows into a contiguous A_e -- because the fused kernel deletes it
// and that saving should be attributed separately from the overlap saving.
//
// Build: nvcc -O3 -std=c++20 -arch=sm_89 -o gemm_probe gemm_probe.cu -lcublas

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define CUDACHECK(cmd)                                                   \
  do {                                                                   \
    cudaError_t e = (cmd);                                               \
    if (e != cudaSuccess) {                                              \
      std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e));                               \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

#define CUBLASCHECK(cmd)                                                  \
  do {                                                                    \
    cublasStatus_t s = (cmd);                                             \
    if (s != CUBLAS_STATUS_SUCCESS) {                                     \
      std::fprintf(stderr, "%s:%d cuBLAS error %d\n", __FILE__, __LINE__, \
                   (int)s);                                               \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

constexpr int kH = 2048;   // hidden
constexpr int kN = 2816;   // moe_intermediate 1408, gate+up fused
constexpr int kELoc = 16;  // 64 routed experts at EP=4
constexpr int kTopK = 6;

// Gather rows[i] of src into dst[i]. This is the permute a non-fused MoE path
// must run between dispatch and GEMM; the fused kernel folds it into the
// A-operand cp.async and never materialises A_e.
__global__ void permute(__nv_bfloat16* __restrict__ dst,
                        __nv_bfloat16 const* __restrict__ src,
                        int const* __restrict__ rows, int m, int ldk) {
  int r = blockIdx.x;
  if (r >= m) return;
  __nv_bfloat16 const* s = src + (size_t)rows[r] * ldk;
  __nv_bfloat16* d = dst + (size_t)r * ldk;
  for (int i = threadIdx.x; i < ldk / 8; i += blockDim.x)
    reinterpret_cast<float4*>(d)[i] = reinterpret_cast<float4 const*>(s)[i];
}

int main(int argc, char** argv) {
  int iters = 50, warmup = 10;
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    if (k == "-n" && i + 1 < argc)
      iters = std::atoi(argv[++i]);
    else if (k == "-w" && i + 1 < argc)
      warmup = std::atoi(argv[++i]);
  }

  CUDACHECK(cudaSetDevice(0));
  cublasHandle_t h;
  CUBLASCHECK(cublasCreate(&h));
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUBLASCHECK(cublasSetStream(h, stream));

  int const kTs[] = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
  int const nT = sizeof(kTs) / sizeof(kTs[0]);
  int const maxT = kTs[nT - 1];
  const size_t maxRows =
      (size_t)maxT * kTopK;  // upper bound on rows*experts at a rank

  // Weights for all 16 local experts, plus A/C big enough for the largest sweep
  // point.
  __nv_bfloat16 *W, *A, *Ap, *C;
  int* rows;
  CUDACHECK(cudaMalloc(&W, (size_t)kELoc * kH * kN * sizeof(__nv_bfloat16)));
  CUDACHECK(cudaMalloc(&A, maxRows * kH * sizeof(__nv_bfloat16)));
  CUDACHECK(cudaMalloc(&Ap, maxRows * kH * sizeof(__nv_bfloat16)));
  CUDACHECK(cudaMalloc(&C, maxRows * kN * sizeof(__nv_bfloat16)));
  CUDACHECK(cudaMalloc(&rows, maxRows * sizeof(int)));
  CUDACHECK(
      cudaMemset(W, 0x3c, (size_t)kELoc * kH * kN * sizeof(__nv_bfloat16)));
  CUDACHECK(cudaMemset(A, 0x3c, maxRows * kH * sizeof(__nv_bfloat16)));
  {
    std::vector<int> hr(maxRows);
    for (size_t i = 0; i < maxRows; i++)
      hr[i] = (int)((i * 2654435761u) % maxRows);
    CUDACHECK(cudaMemcpy(rows, hr.data(), maxRows * sizeof(int),
                         cudaMemcpyHostToDevice));
  }

  float const alpha = 1.0f, beta = 0.0f;
  cudaEvent_t e0, e1;
  CUDACHECK(cudaEventCreate(&e0));
  CUDACHECK(cudaEventCreate(&e1));

  // Row-major [M,K] x [K,N] via column-major cuBLAS: swap the operands so the
  // library computes C^T = W^T A^T with everything left in row-major order.
  auto oneGemm = [&](int m, int e, __nv_bfloat16 const* a) {
    CUBLASCHECK(cublasGemmEx(
        h, CUBLAS_OP_N, CUBLAS_OP_N, kN, m, kH, &alpha, W + (size_t)e * kH * kN,
        CUDA_R_16BF, kN, a, CUDA_R_16BF, kH, &beta, C, CUDA_R_16BF, kN,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  };

  std::printf(
      "# cuBLAS grouped MoE GEMM, 1x L4 sm_89, bf16 in/out fp32 accum\n");
  std::printf(
      "# DeepSeek-V2-Lite at EP=4: H=%d N=%d, %d local experts, top-%d\n", kH,
      kN, kELoc, kTopK);
  std::printf("# M_e = 6T/%d rows per expert; %d GEMMs per step\n#\n", kELoc,
              kELoc);
  std::printf("# %8s %8s %10s %12s %10s %12s %10s\n", "T/rank", "M_e",
              "gemm(us)", "TFLOP/s", "%peak", "permute(us)", "perm%");

  for (int ti = 0; ti < nT; ti++) {
    int const T = kTs[ti];
    int const m = (T * kTopK) / kELoc;  // rows per local expert
    if (m < 1) continue;
    double const flops = 2.0 * (double)m * kELoc * kH * kN;

    for (int i = 0; i < warmup; i++)
      for (int e = 0; e < kELoc; e++) oneGemm(m, e, A);
    CUDACHECK(cudaStreamSynchronize(stream));

    CUDACHECK(cudaEventRecord(e0, stream));
    for (int i = 0; i < iters; i++)
      for (int e = 0; e < kELoc; e++) oneGemm(m, e, A);
    CUDACHECK(cudaEventRecord(e1, stream));
    CUDACHECK(cudaEventSynchronize(e1));
    float ms = 0;
    CUDACHECK(cudaEventElapsedTime(&ms, e0, e1));
    double const gus = (double)ms * 1000.0 / iters;

    // The permute the fused kernel deletes.
    for (int i = 0; i < warmup; i++)
      for (int e = 0; e < kELoc; e++)
        permute<<<m, 256, 0, stream>>>(Ap, A, rows, m, kH);
    CUDACHECK(cudaStreamSynchronize(stream));
    CUDACHECK(cudaEventRecord(e0, stream));
    for (int i = 0; i < iters; i++)
      for (int e = 0; e < kELoc; e++)
        permute<<<m, 256, 0, stream>>>(Ap, A, rows, m, kH);
    CUDACHECK(cudaEventRecord(e1, stream));
    CUDACHECK(cudaEventSynchronize(e1));
    CUDACHECK(cudaEventElapsedTime(&ms, e0, e1));
    double const pus = (double)ms * 1000.0 / iters;

    // L4 bf16 dense tensor-core peak.
    double const peak = 121.0;
    double const tf = flops / (gus * 1e-6) / 1e12;
    std::printf("  %8d %8d %10.2f %12.2f %9.1f%% %12.2f %9.1f%%\n", T, m, gus,
                tf, 100.0 * tf / peak, pus, 100.0 * pus / gus);
    std::fflush(stdout);
  }

  CUDACHECK(cudaFree(rows));
  CUDACHECK(cudaFree(C));
  CUDACHECK(cudaFree(Ap));
  CUDACHECK(cudaFree(A));
  CUDACHECK(cudaFree(W));
  CUBLASCHECK(cublasDestroy(h));
  return 0;
}
