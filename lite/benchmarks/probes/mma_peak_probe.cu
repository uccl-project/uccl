// What mma.m16n8k16.f32.bf16 actually sustains on this L4, with no memory in
// the way at all.
//
// The datasheet number for an L4 is 121 TFLOP/s bf16, but that figure is for
// FP16 accumulate. Ada's consumer-class dies run tensor ops with FP32
// accumulate at half rate, and every GEMM here accumulates in FP32 because bf16
// has 8 mantissa bits and K is 2048. So the bar for
// benchmarks/probes/mma_gemm_probe.cu is this number, not the datasheet one --
// without it, "64% of peak" and "96% of peak" are the same measurement.
//
// Fragments are held in registers and never reloaded; the loop is pure
// back-to-back HMMA. Accumulators are consumed at the end so nothing is dead.
//
//   nvcc -O3 -std=c++20 -arch=sm_89 -o mma_peak_probe mma_peak_probe.cu

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

constexpr int kIters = 4096;
constexpr int kMmaPerIter =
    16;  // one warp's 64x32 tile at k=16, same as the GEMM

__global__ void __launch_bounds__(256) mmaPeak(float* out, int iters) {
  uint32_t a[4], b[2];
  int const t = threadIdx.x;
  // Non-constant so nothing folds, but never reloaded inside the loop.
  a[0] = a[1] = a[2] = a[3] = 0x3f803f80u ^ (uint32_t)t;
  b[0] = b[1] = 0x3f803f80u ^ (uint32_t)(t << 1);

  float acc[kMmaPerIter][4];
#pragma unroll
  for (int i = 0; i < kMmaPerIter; i++)
    acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;

  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int i = 0; i < kMmaPerIter; i++)
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(acc[i][0]), "+f"(acc[i][1]), "+f"(acc[i][2]), "+f"(acc[i][3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }

  float s = 0;
#pragma unroll
  for (int i = 0; i < kMmaPerIter; i++)
    s += acc[i][0] + acc[i][1] + acc[i][2] + acc[i][3];
  if (s == 12345.678f) out[0] = s;  // never true; keeps the work live
}

int main() {
  CK(cudaSetDevice(0));
  int nSM = 0, clk = 0;
  CK(cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, 0));
  CK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0));  // kHz

  float* out;
  CK(cudaMalloc(&out, sizeof(float)));

  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));

  std::printf("# mma.m16n8k16.f32.bf16.bf16.f32 sustained rate, 1x L4 sm_89\n");
  std::printf("# %d SMs, %.2f GHz boost\n#\n", nSM, clk / 1e6);
  std::printf("# %10s %10s %12s %12s %14s\n", "blocks/SM", "warps/SM",
              "time(us)", "TFLOP/s", "cyc/mma/subp");

  for (int bps = 1; bps <= 4; bps++) {
    int const grid = nSM * bps;
    for (int i = 0; i < 3; i++) mmaPeak<<<grid, 256>>>(out, kIters);
    CK(cudaDeviceSynchronize());

    CK(cudaEventRecord(e0));
    for (int i = 0; i < 5; i++) mmaPeak<<<grid, 256>>>(out, kIters);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    double const us = (double)ms * 1000.0 / 5;

    double const nMma =
        (double)grid * 8.0 * kIters * kMmaPerIter;  // 8 warps/block
    double const flops = nMma * 2.0 * 16 * 8 * 16;
    double const tf = flops / (us * 1e-6) / 1e12;
    // Cycles the tensor pipe of one sub-partition spends per mma.
    double const cyc = (us * 1e-6) * (clk * 1000.0) * nSM * 4.0 / nMma;
    std::printf("  %10d %10d %12.2f %12.2f %14.2f\n", bps, bps * 8, us, tf,
                cyc);
  }

  std::printf("\n# cyc/mma/subp near 32 means FP32-accumulate half rate;\n");
  std::printf(
      "# near 16 would mean full rate. The GEMM's bar is the best TFLOP/s "
      "above.\n");
  cudaFree(out);
  return 0;
}
