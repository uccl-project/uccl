// Does the MoE GEMM overlap a copy-engine dispatch, or collapse it?
//
// docs/allreduce.md ("Overlap does not exist") and docs/alltoall.md both record
// that mixing SM traffic with copy-engine traffic loses on this fabric: the
// all-to-all falls 74.11 -> 71.24 GB/s at a 10% kernel share. Both of those
// measurements used a *bandwidth-bound* kernel -- a reduce, and a compression
// pass already asking 205 GB/s of a 300 GB/s HBM.
//
// The MoE GEMM is a different animal. It is compute-bound at prefill sizes
// (cuBLAS: 65 TFLOP/s at T=2048, 54% of the L4's bf16 peak) and its HBM traffic
// is the 184 MB of expert weights, read once and reused across all rows. If the
// earlier result was about HBM contention rather than about SMs-vs-DMA per se,
// this should overlap and the fused dispatch+GEMM is worth building. If it
// collapses anyway, fusion is dead on this fabric and that is the finding.
//
// Reports both directions -- whether the copy slows the GEMM and whether the
// GEMM slows the copy -- because the fused design can trade one for the other
// by partitioning blocks, and cuBLAS cannot.
//
//   nvcc -O3 -std=c++20 -arch=sm_89 -o moe_overlap_probe moe_overlap_probe.cu
//   -lcublas

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define CK(c)                                                   \
  do {                                                          \
    cudaError_t e = (c);                                        \
    if (e != cudaSuccess) {                                     \
      std::printf("%d: %s\n", __LINE__, cudaGetErrorString(e)); \
      return 1;                                                 \
    }                                                           \
  } while (0)

#define BK(c)                                           \
  do {                                                  \
    cublasStatus_t s = (c);                             \
    if (s != CUBLAS_STATUS_SUCCESS) {                   \
      std::printf("%d: cuBLAS %d\n", __LINE__, (int)s); \
      return 1;                                         \
    }                                                   \
  } while (0)

constexpr int kH = 2048;   // hidden
constexpr int kN = 2816;   // moe_intermediate 1408, gate+up fused
constexpr int kELoc = 16;  // 64 routed experts at EP=4
constexpr int kTopK = 6;

int main(int argc, char** argv) {
  int iters = 30, warmup = 5;
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    if (k == "-n" && i + 1 < argc)
      iters = std::atoi(argv[++i]);
    else if (k == "-w" && i + 1 < argc)
      warmup = std::atoi(argv[++i]);
  }

  int nDev = 0;
  CK(cudaGetDeviceCount(&nDev));
  if (nDev < 2) {
    std::printf("need 2 GPUs\n");
    return 1;
  }
  CK(cudaSetDevice(0));
  cudaError_t pe = cudaDeviceEnablePeerAccess(1, 0);
  if (pe != cudaSuccess && pe != cudaErrorPeerAccessAlreadyEnabled) {
    std::printf("peer access 0->1 failed: %s\n", cudaGetErrorString(pe));
    return 1;
  }

  cublasHandle_t h;
  BK(cublasCreate(&h));
  cudaStream_t sCopy, sGemm;
  CK(cudaStreamCreateWithFlags(&sCopy, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&sGemm, cudaStreamNonBlocking));
  BK(cublasSetStream(h, sGemm));

  int const kTs[] = {256, 512, 1024, 2048, 4096};
  int const nT = sizeof(kTs) / sizeof(kTs[0]);
  int const maxT = kTs[nT - 1];
  const size_t maxRows = (size_t)maxT * kTopK;

  __nv_bfloat16 *W, *A, *C;
  char *dSrc, *dPeer;
  CK(cudaMalloc(&W, (size_t)kELoc * kH * kN * sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&A, maxRows * kH * sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&C, maxRows * kN * sizeof(__nv_bfloat16)));
  CK(cudaMemset(W, 0x3c, (size_t)kELoc * kH * kN * sizeof(__nv_bfloat16)));
  CK(cudaMemset(A, 0x3c, maxRows * kH * sizeof(__nv_bfloat16)));

  // Dispatch egress for one rank at the largest sweep point: 3 flows, each
  // carrying the ~82% of a peer's tokens that route to at least one of our
  // experts. A GPU's peer copies never overlap (docs/allreduce.md), so issuing
  // this as one serial stream of copies to a single peer is the right model of
  // what the egress link sees.
  const size_t maxEgress =
      (size_t)(3.0 * maxT * 0.822) * kH * sizeof(__nv_bfloat16);
  CK(cudaMalloc(&dSrc, maxEgress));
  CK(cudaMemset(dSrc, 0x3c, maxEgress));
  CK(cudaSetDevice(1));
  CK(cudaMalloc(&dPeer, maxEgress));
  CK(cudaSetDevice(0));

  float const alpha = 1.0f, beta = 0.0f;
  cudaEvent_t evC0, evC1, evG0, evG1;
  CK(cudaEventCreate(&evC0));
  CK(cudaEventCreate(&evC1));
  CK(cudaEventCreate(&evG0));
  CK(cudaEventCreate(&evG1));

  std::printf("# MoE GEMM vs copy-engine dispatch, 2x L4, PCIe\n");
  std::printf(
      "# DeepSeek-V2-Lite at EP=4: H=%d N=%d, %d local experts, top-%d\n", kH,
      kN, kELoc, kTopK);
  std::printf(
      "# copy = one rank's dispatch egress (3 flows x 82%% of a peer's "
      "tokens)\n");
  std::printf("# gemm = the %d-expert grouped GEMM, cuBLAS bf16\n#\n", kELoc);
  std::printf("# %6s %9s %9s %9s %9s %9s %9s %8s %8s %9s %8s %8s\n", "T/rank",
              "copy(us)", "gemm(us)", "both(us)", "max(us)", "sum(us)",
              "copyGB/s", "hidden", "vs.sum", "copy@GB/s", "copy.pen",
              "gemm.pen");

  for (int ti = 0; ti < nT; ti++) {
    int const T = kTs[ti];
    int const m = (T * kTopK) / kELoc;
    const size_t egress =
        (size_t)(3.0 * T * 0.822) * kH * sizeof(__nv_bfloat16);

    auto gemmOnce = [&]() {
      for (int e = 0; e < kELoc; e++)
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, kN, m, kH, &alpha,
                     W + (size_t)e * kH * kN, CUDA_R_16BF, kN, A, CUDA_R_16BF,
                     kH, &beta, C, CUDA_R_16BF, kN, CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    };

    // Events on the default stream do not order against non-blocking streams,
    // so time the wall on the host with a full device sync at the end.
    // The per-stream events are additional:
    // each stream's own pair measures that leg in isolation even while the
    // other leg is running, which is what attributes the loss to one side or
    // the other.
    auto run = [&](bool copy, bool gemm, double* legCopy,
                   double* legGemm) -> double {
      for (int i = 0; i < warmup; i++) {
        if (copy) cudaMemcpyPeerAsync(dPeer, 1, dSrc, 0, egress, sCopy);
        if (gemm) gemmOnce();
      }
      cudaDeviceSynchronize();
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; i++) {
        if (copy) cudaEventRecord(evC0, sCopy);
        if (gemm) cudaEventRecord(evG0, sGemm);
        if (copy) cudaMemcpyPeerAsync(dPeer, 1, dSrc, 0, egress, sCopy);
        if (gemm) gemmOnce();
        if (copy) cudaEventRecord(evC1, sCopy);
        if (gemm) cudaEventRecord(evG1, sGemm);
        cudaDeviceSynchronize();
        float ms = 0;
        if (copy && legCopy) {
          cudaEventElapsedTime(&ms, evC0, evC1);
          *legCopy += ms * 1000.0;
        }
        if (gemm && legGemm) {
          cudaEventElapsedTime(&ms, evG0, evG1);
          *legGemm += ms * 1000.0;
        }
      }
      auto t1 = std::chrono::steady_clock::now();
      return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    };

    double cAlone = 0, gAlone = 0, cBoth = 0, gBoth = 0;
    double const c = run(true, false, &cAlone, nullptr);
    double const z = run(false, true, nullptr, &gAlone);
    double const both = run(true, true, &cBoth, &gBoth);
    cAlone /= iters;
    gAlone /= iters;
    cBoth /= iters;
    gBoth /= iters;
    double const mx = c > z ? c : z;

    // Fraction of the *shorter* leg that disappeared into the longer one.
    double const shorter = c < z ? c : z;
    double hidden = (c + z - both) / shorter * 100.0;
    if (hidden < 0) hidden = 0;
    if (hidden > 100) hidden = 100;

    std::printf(
        "  %6d %9.1f %9.1f %9.1f %9.1f %9.1f %9.2f %7.0f%% %8.2fx %9.2f %8.2fx "
        "%8.2fx\n",
        T, c, z, both, mx, c + z, (double)egress / (c * 1e-6) / 1e9, hidden,
        (c + z) / both, (double)egress / (cBoth * 1e-6) / 1e9, cBoth / cAlone,
        gBoth / gAlone);
    std::fflush(stdout);
  }

  std::printf(
      "\n# hidden = (copy+gemm-both)/min(copy,gemm): 100%% is perfect overlap, "
      "0%% is none\n");
  std::printf(
      "# vs.sum  = speedup of running them together against running them back "
      "to back\n");
  std::printf(
      "# copy@GB/s, copy.pen, gemm.pen come from per-stream events inside the "
      "concurrent run:\n");
  std::printf(
      "#   how much each leg individually slowed down. 1.00x = the other leg "
      "cost it nothing.\n");

  cudaFree(dSrc);
  cudaSetDevice(1);
  cudaFree(dPeer);
  cudaSetDevice(0);
  cudaFree(C);
  cudaFree(A);
  cudaFree(W);
  cublasDestroy(h);
  return 0;
}
