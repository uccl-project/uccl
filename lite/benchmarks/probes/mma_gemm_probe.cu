// The gather-fused bf16 grouped GEMM, standalone, on one GPU.
//
// This is the compute half of the fused dispatch+GEMM kernel, built and priced
// on its own first: if it cannot get close to cuBLAS there is no point wiring
// it into the 4-rank harness, because the GEMM dominates the fused step at
// prefill sizes and any deficit here comes straight off the fusion win.
//
// The bar is not the L4's 121 TFLOP/s bf16 peak. It is what cuBLAS actually
// delivers on these shapes, which benchmarks/probes/gemm_probe.cu measures
// at 65.4 TFLOP/s (54% of peak) at T=2048 -- skinny grouped GEMMs are not where
// cuBLAS is strong either.
//
// What this kernel does that cuBLAS cannot: the A operand is *gathered*. Row i
// of expert e's tile is row `rowIdx[...]` of the dispatch receive buffer,
// pulled straight into shared memory by the cp.async that would have loaded it
// anyway. A non-fused path has to run a permute kernel first to materialise a
// contiguous A_e (benchmarks/probes/gemm_probe.cu prices that at 4.3% of the
// GEMM). Here it is free.
//
// Shape: 128x128x32 tiles, 8 warps at 2x4, each warp 64x32 via mma.m16n8k16
// with fp32 accumulate. Both smem tiles are stored [outer][BK+8]; the +8 pad is
// 16 bytes, so it keeps cp.async and ldmatrix alignment while making every
// fragment load bank-conflict-free.
//
// Weights are stored [E][N][H] -- "n-major", the layout the kernel wants for a
// coalesced load along K. cuBLAS reads the same buffer with OP_T, so the
// comparison is like for like and no second copy of the weights exists.
//
//   nvcc -O3 -std=c++20 -arch=sm_89 -o mma_gemm_probe mma_gemm_probe.cu
//   -lcublas

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

constexpr int kH = 2048;   // hidden, the GEMM's K
constexpr int kN = 2816;   // moe_intermediate 1408, gate+up fused
constexpr int kELoc = 16;  // 64 routed experts at EP=4
constexpr int kTopK = 6;

// Tile shape. Overridable from the build line so the sweep is a shell loop
// rather than an edit; pass definitions through EXTRA_NVCC_FLAGS in the
// Makefile.
//
// The sweep's answer is that the shape barely matters: every combination from
// 128x128x32 to 256x128x64 lands within 39.5 +/- 3 TF/s, and the two big ones
// are worse because their accumulators cost the second resident block. What
// settles it is BENCH_FIXED_K below -- pinning every step to k=0 collapses the
// working set into cache and buys only 13% (36.4 -> 41.3 TF/s). So this kernel
// is not bandwidth-bound, and the bytes-per-flop argument for bigger tiles does
// not apply. It is latency/issue-bound at ~34% of the 120.8 TF/s that
// benchmarks/probes/mma_peak_probe.cu measures for back-to-back mma.
//
// The obvious next move from there is more resident blocks -- 2 per SM is 16 of
// Ada's 48 warps. That was tried four ways and **it is not the lever**:
//
//   __launch_bounds__(256, 3)   80 regs, 152 B spill  ->  22.8 TF/s  (0.44x
//   cuBLAS)
//   __launch_bounds__(256, 4)   64 regs, 428 B spill  ->  11.4 TF/s  (0.20x)
//   warp tile 32x32, 3 blocks   74 regs, no spill     ->  31.7 TF/s  (0.63x)
//   warp tile 64x64, 1 block   110 regs, no spill     ->  36.9 TF/s  (0.74x)
//   warp tile 64x32, 2 blocks  110 regs, no spill     ->  39.4 TF/s  (0.78x) <-
//   best
//
// Buying occupancy by spilling is catastrophic, because the accumulators are
// what spills and every mma touches them. Buying it with a smaller warp tile
// costs more than it returns: 32x32 reads 61 bytes of shared memory per KFLOP
// against 64x32's 46, since each tile is re-read by more warps. And a bigger
// warp tile has the better ratio but pays the second block. 64x32 at 2 blocks
// is a genuine local optimum, and both directions out of it are downhill.
//
// XOR-swizzling the tiles to drop the +8 pad does not rescue it either. The
// swizzle needs log2(BK/8) bits to decorrelate the 8 rows of an ldmatrix phase:
// at BK=64 that is 3 bits and it is conflict-free, but 2-stage smem is already
// 64 KB so occupancy is 1 block with or without it; at BK=32 there are only 2
// bits and rows 0 and 4 collide. The one config the pad blocks -- 3 stages at 2
// blocks/SM -- is exactly the BK=32 case where the swizzle does not work.
//
// So 0.78-0.82x cuBLAS is where this lands with these primitives at 8 warps.
// Left there deliberately: the remaining ideas (warp specialisation, a
// producer/consumer split with async barriers) are a different kernel, not a
// tuning pass, and docs/fusion.md reports our GEMM's deficit separately so
// the fusion results do not hide behind it.
#ifndef TILE_BM
#define TILE_BM 128
#endif
#ifndef TILE_BN
#define TILE_BN 128
#endif
#ifndef TILE_BK
#define TILE_BK 32
#endif
#ifndef TILE_WARPS_M
#define TILE_WARPS_M 2
#endif
#ifndef TILE_STAGES
#define TILE_STAGES 2
#endif
// Minimum blocks per SM to demand of ptxas. The second __launch_bounds__
// argument changes what it optimises for: instead of minimising spills at
// whatever register count it likes, it must fit the budget that many resident
// blocks implies -- 85 registers for 3 blocks at 256 threads, 64 for 4. Some
// spill can be cheaper than the occupancy it buys, and this kernel is
// latency-bound at 2 blocks/SM (16 of Ada's 48 warps), so it is worth asking.
// 0 leaves the choice to ptxas, which is where this started.
#ifndef TILE_MIN_BPSM
#define TILE_MIN_BPSM 0
#endif

constexpr int BM = TILE_BM, BN = TILE_BN, BK = TILE_BK;
constexpr int NSTAGE = TILE_STAGES;
constexpr int PAD = 8;         // 16 bytes: keeps cp.async/ldmatrix alignment
constexpr int LDA = BK + PAD;  // smem row stride, in bf16
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kWarpsM = TILE_WARPS_M, kWarpsN = kWarps / TILE_WARPS_M;
constexpr int kWarpM = BM / kWarpsM, kWarpN = BN / kWarpsN;
constexpr int kMFrag = kWarpM / 16, kNFrag = kWarpN / 8;

// Staging: one thread owns a fixed row of each smem tile and a fixed span of
// 16-byte chunks within it, so the gathered row index is resolved once per tile
// rather than once per K-step.
constexpr int kChunkPerRow = BK / 8;
constexpr int kThrPerRowA = kThreads / BM;
constexpr int kThrPerRowB = kThreads / BN;
constexpr int kSubA = kChunkPerRow / kThrPerRowA;
constexpr int kSubB = kChunkPerRow / kThrPerRowB;

static_assert(kWarpM % 16 == 0 && kWarpN % 8 == 0, "warp tile must fit m16n8");
static_assert(kNFrag % 2 == 0, "B ldmatrix.x4 covers two n-fragments");
static_assert(BK % 16 == 0, "k-step is 16");
static_assert(kThreads % BM == 0 && kThreads % BN == 0,
              "staging wants whole rows per thread");
static_assert(kChunkPerRow % kThrPerRowA == 0 &&
                  kChunkPerRow % kThrPerRowB == 0,
              "chunk split");

// One unit of GEMM work: a BMxBN output tile of one expert, whose rows come
// from one arrival group. `group` is what the fused kernel waits on; here it is
// carried but unused, so the same descriptor serves both.
struct TileDesc {
  int rowStart;  // index into rowIdx, and the row of Y this tile writes
  int mRows;     // <= BM; ragged experts make this a partial tile
  int expert;
  int group;
  int nTile;
};

__device__ __forceinline__ uint32_t smemAddr(void const* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// 16 bytes global -> shared, with `srcBytes` = 0 meaning "zero-fill instead".
// That is how a partial tile is handled: no branch, no separate cleanup path.
__device__ __forceinline__ void cpAsync16(uint32_t dst, void const* src,
                                          int srcBytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst),
               "l"(src), "r"(srcBytes));
}

__device__ __forceinline__ void ldm4(uint32_t addr, uint32_t& r0, uint32_t& r1,
                                     uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

__device__ __forceinline__ void mma16816(float& d0, float& d1, float& d2,
                                         float& d3, uint32_t a0, uint32_t a1,
                                         uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Grouped GEMM over a work queue of tiles.
//
// `arrived` is the fused kernel's hook: a tile whose rows came from arrival
// group g spins until arrived[g] is published. Passing nullptr (as the
// standalone probe does) compiles the wait out of the hot path entirely.
#if TILE_MIN_BPSM > 0
#define GEMM_BOUNDS __launch_bounds__(kThreads, TILE_MIN_BPSM)
#else
#define GEMM_BOUNDS __launch_bounds__(kThreads)
#endif

__global__ void GEMM_BOUNDS groupedGemmMMA(
    TileDesc const* __restrict__ tiles, int nTiles, int* __restrict__ ctr,
    __nv_bfloat16 const* __restrict__ Aall, int const* __restrict__ rowIdx,
    __nv_bfloat16 const* __restrict__ W, __nv_bfloat16* __restrict__ Y, int K,
    int N, const volatile uint32_t* __restrict__ arrived) {
  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16* sA = smem;                      // [NSTAGE][BM][LDA]
  __nv_bfloat16* sB = smem + NSTAGE * BM * LDA;  // [NSTAGE][BN][LDA]

  int const tid = threadIdx.x;
  int const warp = tid >> 5, lane = tid & 31;
  int const wM = (warp / kWarpsN) * kWarpM;
  int const wN = (warp % kWarpsN) * kWarpN;

  int const rowA = tid / kThrPerRowA, subA = (tid % kThrPerRowA) * kSubA;
  int const rowB = tid / kThrPerRowB, subB = (tid % kThrPerRowB) * kSubB;

  for (;;) {
    __shared__ int sTile;
    if (tid == 0) sTile = atomicAdd(ctr, 1);
    __syncthreads();
    int const t = sTile;
    if (t >= nTiles) return;

    const TileDesc td = tiles[t];
    if (arrived != nullptr) {
      // Frontier wait. Tiles are enqueued in arrival order, so this spins only
      // on the group actually in flight, and group 0 (our own rows) never
      // waits.
      if (tid == 0)
        while (arrived[td.group] == 0) __threadfence_block();
      __syncthreads();
    }

    __nv_bfloat16 const* Wt =
        W + (size_t)td.expert * N * K + (size_t)td.nTile * BN * K;
    float acc[kMFrag][kNFrag][4];
#pragma unroll
    for (int i = 0; i < kMFrag; i++)
#pragma unroll
      for (int j = 0; j < kNFrag; j++)
#pragma unroll
        for (int k = 0; k < 4; k++) acc[i][j][k] = 0.0f;

    // Rows of this tile, gathered. Out-of-range rows load as zero and are
    // dropped at writeback, so a ragged expert costs a predicate, not a branch.
    int const gRowA = (rowA < td.mRows) ? rowIdx[td.rowStart + rowA] : -1;

    // Every address the inner loop touches is an affine function of the lane
    // and a compile-time offset, so all of it is hoisted here. Leaving the
    // __cvta_generic_to_shared and the 64-bit row*LDA multiply inside the k
    // loop costs one per ldmatrix, and that integer work issues in the same
    // slots the mma needs.
    const uint32_t aStage = smemAddr(sA + (size_t)rowA * LDA + subA * 8);
    const uint32_t bStage = smemAddr(sB + (size_t)rowB * LDA + subB * 8);
    __nv_bfloat16 const* paBase =
        (gRowA >= 0) ? (Aall + (size_t)gRowA * K + subA * 8) : Aall;
    __nv_bfloat16 const* pbBase = Wt + (size_t)rowB * K + subB * 8;
    const uint32_t aFrag =
        smemAddr(sA + (size_t)(wM + (lane & 15)) * LDA + (lane >> 4) * 8);
    const uint32_t bFrag =
        smemAddr(sB + (size_t)(wN + (lane & 15)) * LDA + (lane >> 4) * 8);
    constexpr uint32_t kBufA = BM * LDA * sizeof(__nv_bfloat16);
    constexpr uint32_t kBufB = BN * LDA * sizeof(__nv_bfloat16);
    constexpr uint32_t kRow = LDA * sizeof(__nv_bfloat16);

    auto stage = [&](int kStep, int buf) {
    // BENCH_FIXED_K pins every step to the same K offset. The instruction
    // stream is bit-identical and the results are garbage, but the whole
    // working set collapses into L1/L2 -- so it separates "this kernel is
    // memory-bound" from "this kernel's instruction mix is the ceiling".
#ifdef BENCH_FIXED_K
      const int k0 = 0;
      (void)kStep;
#else
      const int k0 = kStep * BK;
#endif
#pragma unroll
      for (int s = 0; s < kSubA; s++)
        cpAsync16(aStage + buf * kBufA + s * 16, paBase + k0 + s * 8,
                  gRowA >= 0 ? 16 : 0);
#pragma unroll
      for (int s = 0; s < kSubB; s++)
        cpAsync16(bStage + buf * kBufB + s * 16, pbBase + k0 + s * 8, 16);
      asm volatile("cp.async.commit_group;\n" ::);
    };

    int const nSteps = K / BK;
    // Buffer for step k is k % NSTAGE. Keep NSTAGE-1 steps in flight ahead of
    // the one being computed: issue first, then wait, so the cp.async has the
    // whole compute of the previous step to land in.
#pragma unroll
    for (int s = 0; s < NSTAGE - 1; s++) stage(s, s);

    for (int ks = 0; ks < nSteps; ks++) {
      int const nxt = ks + NSTAGE - 1;
      if (nxt < nSteps) stage(nxt, nxt % NSTAGE);
      // Groups issued but not yet consumed run ks..min(nxt, nSteps-1), so this
      // many may remain pending once step ks has landed. Getting it wrong is
      // silent -- `wait_group n` is already satisfied when fewer than n are
      // pending, so the tail would compute from an unfilled buffer.
      int const pending = (nxt < nSteps ? nxt : nSteps - 1) - ks;
      switch (pending) {
        case 0:
          asm volatile("cp.async.wait_group 0;\n" ::);
          break;
        case 1:
          asm volatile("cp.async.wait_group 1;\n" ::);
          break;
        case 2:
          asm volatile("cp.async.wait_group 2;\n" ::);
          break;
        default:
          asm volatile("cp.async.wait_group 3;\n" ::);
          break;
      }
      __syncthreads();

      int const buf = ks % NSTAGE;
      const uint32_t pA = aFrag + buf * kBufA;
      const uint32_t pB = bFrag + buf * kBufB;

      // Fragment loads are software-pipelined one 16-step ahead of the mma that
      // consumes them. Without this the ldmatrix latency is fully exposed: the
      // barrier above puts every warp into the same phase, so there is no other
      // warp in a different phase to cover it, and the tensor pipe idles once
      // per 16-step. Two register sets, alternating.
      uint32_t a[2][kMFrag][4], b[2][kNFrag][2];
      auto loadFrag = [&](int kk, int slot) {
#pragma unroll
        for (int mi = 0; mi < kMFrag; mi++)
          ldm4(pA + mi * 16 * kRow + kk * 2, a[slot][mi][0], a[slot][mi][1],
               a[slot][mi][2], a[slot][mi][3]);
          // B is stored n-major, so one ldmatrix.x4 covers two 16x8 fragments:
          // r0,r2 are n-frag j, r1,r3 are n-frag j+1.
#pragma unroll
        for (int nj = 0; nj < kNFrag; nj += 2) {
          uint32_t q0, q1, q2, q3;
          ldm4(pB + nj * 8 * kRow + kk * 2, q0, q1, q2, q3);
          b[slot][nj][0] = q0;
          b[slot][nj][1] = q2;
          b[slot][nj + 1][0] = q1;
          b[slot][nj + 1][1] = q3;
        }
      };

      loadFrag(0, 0);
#pragma unroll
      for (int kk = 0; kk < BK; kk += 16) {
        constexpr int kSlots = 2;
        int const slot = (kk / 16) % kSlots;
        if (kk + 16 < BK) loadFrag(kk + 16, slot ^ 1);
#pragma unroll
        for (int mi = 0; mi < kMFrag; mi++)
#pragma unroll
          for (int nj = 0; nj < kNFrag; nj++)
            mma16816(acc[mi][nj][0], acc[mi][nj][1], acc[mi][nj][2],
                     acc[mi][nj][3], a[slot][mi][0], a[slot][mi][1],
                     a[slot][mi][2], a[slot][mi][3], b[slot][nj][0],
                     b[slot][nj][1]);
      }
      __syncthreads();
    }

    // m16n8 accumulator layout: lane holds rows (lane/4, lane/4+8) and column
    // pairs at (lane%4)*2.
    int const cr = lane >> 2, cc = (lane & 3) << 1;
#pragma unroll
    for (int mi = 0; mi < kMFrag; mi++) {
#pragma unroll
      for (int nj = 0; nj < kNFrag; nj++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
          int const m = wM + mi * 16 + cr + h * 8;
          int const n = wN + nj * 8 + cc;
          if (m < td.mRows) {
            __nv_bfloat16* o =
                Y + (size_t)(td.rowStart + m) * N + td.nTile * BN + n;
            o[0] = __float2bfloat16(acc[mi][nj][h * 2 + 0]);
            o[1] = __float2bfloat16(acc[mi][nj][h * 2 + 1]);
          }
        }
      }
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Warp-specialised variant.
//
// The measured problem with the kernel above is neither bandwidth
// (BENCH_FIXED_K buys 13%) nor occupancy
// (benchmarks/data/mma_gemm_occupancy.txt: four attempts, all downhill). It is
// that the two __syncthreads per K-step put all eight warps in the same phase,
// so the ldmatrix latency right after each barrier is exposed with no warp in a
// different phase to cover it.
//
// So split the block: 4 producer warps do nothing but cp.async, 8 consumer
// warps do nothing but ldmatrix+mma, and they hand stages over through
// mbarriers instead of block-wide barriers. cp.async.mbarrier.arrive.noinc
// makes the barrier tick when that thread's outstanding cp.async retire, which
// is what removes the __syncthreads entirely.
//
// Raw mbarrier PTX rather than cuda::pipeline because the A operand needs the
// predicated form of cp.async (src-size 0 zero-fills an out-of-range row), and
// cuda::memcpy_async has no equivalent.
// Producer warps are pure overhead per SM: they issue cp.async and nothing
// else, so every one of them is warp slots and registers taken from the mma.
// At 4+8 the block is 384 threads and 96 registers, which is 1 block/SM -- half
// the mma-issuing warps the non-specialised kernel gets from 2 blocks of 8.
// Fewer producers is what buys the second block back.
#ifndef WS_PROD_WARPS
#define WS_PROD_WARPS 2
#endif
constexpr int kProdWarps = WS_PROD_WARPS;
constexpr int kConsWarps = 8;
constexpr int kThreadsWS = (kProdWarps + kConsWarps) * 32;
constexpr int kWarpsNWS = 4;
constexpr int kWarpMWS = BM / (kConsWarps / kWarpsNWS);
constexpr int kWarpNWS = BN / kWarpsNWS;
constexpr int kMFragWS = kWarpMWS / 16, kNFragWS = kWarpNWS / 8;

__device__ __forceinline__ void mbarInit(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(smemAddr(bar)),
               "r"(count));
}
__device__ __forceinline__ void mbarArrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" ::"r"(smemAddr(bar)));
}
__device__ __forceinline__ void cpAsyncMbarArrive(uint64_t* bar) {
  asm volatile(
      "cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(smemAddr(bar)));
}
// test_wait, not try_wait: mbarrier.try_wait.parity is sm_90+, and this is Ada.
// test_wait is the non-blocking sm_80 form, so it has to be spun rather than
// letting the hardware park the warp -- one of the several small ways this
// pattern is cheaper on Hopper than here.
// One lane per warp spins; the rest wait at __syncwarp. Having all 32 lanes
// poll the same shared-memory barrier costs 32x the issue slots and the
// traffic, and those are exactly the slots the mma wants.
__device__ __forceinline__ void mbarWait(uint64_t* bar, int parity) {
  if ((threadIdx.x & 31) == 0)
    asm volatile(
        "{\n.reg .pred p;\n"
        "WSW%=:\n"
        "mbarrier.test_wait.parity.shared.b64 p, [%0], %1;\n"
        "@!p bra WSW%=;\n}\n" ::"r"(smemAddr(bar)),
        "r"(parity));
  __syncwarp();
}

__global__ void __launch_bounds__(kThreadsWS)
    groupedGemmWS(TileDesc const* __restrict__ tiles, int nTiles,
                  int* __restrict__ ctr, __nv_bfloat16 const* __restrict__ Aall,
                  int const* __restrict__ rowIdx,
                  __nv_bfloat16 const* __restrict__ W,
                  __nv_bfloat16* __restrict__ Y, int K, int N) {
  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16* sA = smem;
  __nv_bfloat16* sB = smem + NSTAGE * BM * LDA;
  uint64_t* barFull = reinterpret_cast<uint64_t*>(sB + NSTAGE * BN * LDA);
  uint64_t* barEmpty = barFull + NSTAGE;

  int const tid = threadIdx.x;
  int const warp = tid >> 5, lane = tid & 31;
  bool const isProd = warp < kProdWarps;

  if (tid == 0)
#pragma unroll
    for (int s = 0; s < NSTAGE; s++) {
      mbarInit(&barFull[s], kProdWarps * 32);
      mbarInit(&barEmpty[s], kConsWarps);
    }
  __syncthreads();

  int const nSteps = K / BK;
  // Parities return to their initial state after every tile only if each stage
  // completes an even number of times, which lets the barriers be reused across
  // tiles without a re-init.
  bool const parityClean = ((nSteps / NSTAGE) % 2) == 0;

  constexpr uint32_t kBufA = BM * LDA * sizeof(__nv_bfloat16);
  constexpr uint32_t kBufB = BN * LDA * sizeof(__nv_bfloat16);
  constexpr uint32_t kRow = LDA * sizeof(__nv_bfloat16);
  constexpr int kChunks = BK / 8;

  for (;;) {
    __shared__ int sTile;
    if (tid == 0) sTile = atomicAdd(ctr, 1);
    __syncthreads();
    int const t = sTile;
    if (t >= nTiles || !parityClean) return;
    const TileDesc td = tiles[t];
    __nv_bfloat16 const* Wt =
        W + (size_t)td.expert * N * K + (size_t)td.nTile * BN * K;

    if (isProd) {
      // Each producer thread owns kRowsPerProd fixed rows of both tiles, so the
      // gathered index is resolved once per tile rather than once per K-step.
      constexpr int kRowsPerProd = BM / (kProdWarps * 32);
      uint32_t aSt[kRowsPerProd], bSt[kRowsPerProd];
      __nv_bfloat16 const *pa[kRowsPerProd], *pb[kRowsPerProd];
      bool live[kRowsPerProd];
#pragma unroll
      for (int r = 0; r < kRowsPerProd; r++) {
        int const row = tid + r * kProdWarps * 32;
        int const gRow = (row < td.mRows) ? (rowIdx ? rowIdx[td.rowStart + row]
                                                    : td.rowStart + row)
                                          : -1;
        live[r] = gRow >= 0;
        aSt[r] = smemAddr(sA + (size_t)row * LDA);
        bSt[r] = smemAddr(sB + (size_t)row * LDA);
        pa[r] = live[r] ? (Aall + (size_t)gRow * K) : Aall;
        pb[r] = Wt + (size_t)row * K;
      }

      for (int k = 0; k < nSteps; k++) {
        int const st = k % NSTAGE;
        if (k >= NSTAGE) mbarWait(&barEmpty[st], ((k / NSTAGE) - 1) & 1);
        int const k0 = k * BK;
#pragma unroll
        for (int r = 0; r < kRowsPerProd; r++) {
#pragma unroll
          for (int c = 0; c < kChunks; c++)
            cpAsync16(aSt[r] + st * kBufA + c * 16, pa[r] + k0 + c * 8,
                      live[r] ? 16 : 0);
#pragma unroll
          for (int c = 0; c < kChunks; c++)
            cpAsync16(bSt[r] + st * kBufB + c * 16, pb[r] + k0 + c * 8, 16);
        }
        cpAsyncMbarArrive(&barFull[st]);
      }
    } else {
      int const w = warp - kProdWarps;
      int const wM = (w / kWarpsNWS) * kWarpMWS;
      int const wN = (w % kWarpsNWS) * kWarpNWS;
      const uint32_t aFrag =
          smemAddr(sA + (size_t)(wM + (lane & 15)) * LDA + (lane >> 4) * 8);
      const uint32_t bFrag =
          smemAddr(sB + (size_t)(wN + (lane & 15)) * LDA + (lane >> 4) * 8);

      float acc[kMFragWS][kNFragWS][4];
#pragma unroll
      for (int i = 0; i < kMFragWS; i++)
#pragma unroll
        for (int j = 0; j < kNFragWS; j++)
#pragma unroll
          for (int q = 0; q < 4; q++) acc[i][j][q] = 0.0f;

      for (int k = 0; k < nSteps; k++) {
        int const st = k % NSTAGE;
        mbarWait(&barFull[st], (k / NSTAGE) & 1);
        const uint32_t pA = aFrag + st * kBufA;
        const uint32_t pB = bFrag + st * kBufB;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
          uint32_t a[kMFragWS][4], b[kNFragWS][2];
#pragma unroll
          for (int mi = 0; mi < kMFragWS; mi++)
            ldm4(pA + mi * 16 * kRow + kk * 2, a[mi][0], a[mi][1], a[mi][2],
                 a[mi][3]);
#pragma unroll
          for (int nj = 0; nj < kNFragWS; nj += 2) {
            uint32_t q0, q1, q2, q3;
            ldm4(pB + nj * 8 * kRow + kk * 2, q0, q1, q2, q3);
            b[nj][0] = q0;
            b[nj][1] = q2;
            b[nj + 1][0] = q1;
            b[nj + 1][1] = q3;
          }
#pragma unroll
          for (int mi = 0; mi < kMFragWS; mi++)
#pragma unroll
            for (int nj = 0; nj < kNFragWS; nj++)
              mma16816(acc[mi][nj][0], acc[mi][nj][1], acc[mi][nj][2],
                       acc[mi][nj][3], a[mi][0], a[mi][1], a[mi][2], a[mi][3],
                       b[nj][0], b[nj][1]);
        }
        __syncwarp();
        if (lane == 0) mbarArrive(&barEmpty[st]);
      }

      int const cr = lane >> 2, cc = (lane & 3) << 1;
#pragma unroll
      for (int mi = 0; mi < kMFragWS; mi++)
#pragma unroll
        for (int nj = 0; nj < kNFragWS; nj++)
#pragma unroll
          for (int h = 0; h < 2; h++) {
            int const m = wM + mi * 16 + cr + h * 8;
            int const n = wN + nj * 8 + cc;
            if (m < td.mRows) {
              __nv_bfloat16* o =
                  Y + (size_t)(td.rowStart + m) * N + td.nTile * BN + n;
              o[0] = __float2bfloat16(acc[mi][nj][h * 2 + 0]);
              o[1] = __float2bfloat16(acc[mi][nj][h * 2 + 1]);
            }
          }
    }
    __syncthreads();
  }
}

int main(int argc, char** argv) {
  int iters = 30, warmup = 5;
  bool check = true;
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    if (k == "-n" && i + 1 < argc)
      iters = std::atoi(argv[++i]);
    else if (k == "-w" && i + 1 < argc)
      warmup = std::atoi(argv[++i]);
    else if (k == "-c" && i + 1 < argc)
      check = std::atoi(argv[++i]) != 0;
  }

  CUDACHECK(cudaSetDevice(0));
  cublasHandle_t h;
  CUBLASCHECK(cublasCreate(&h));
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUBLASCHECK(cublasSetStream(h, stream));

  int const kTs[] = {64, 128, 256, 512, 1024, 2048, 4096};
  int const nT = sizeof(kTs) / sizeof(kTs[0]);
  int const maxT = kTs[nT - 1];
  const size_t maxRows = (size_t)maxT * kTopK;

  __nv_bfloat16 *W, *A, *Y, *Yref;
  int* rowIdx;
  int* ctr;
  TileDesc* dTiles;
  CUDACHECK(cudaMalloc(&W, (size_t)kELoc * kN * kH * sizeof(__nv_bfloat16)));
  CUDACHECK(cudaMalloc(&A, maxRows * kH * sizeof(__nv_bfloat16)));
  CUDACHECK(cudaMalloc(&Y, maxRows * kN * sizeof(__nv_bfloat16)));
  CUDACHECK(cudaMalloc(&Yref, maxRows * kN * sizeof(__nv_bfloat16)));
  CUDACHECK(cudaMalloc(&rowIdx, maxRows * sizeof(int)));
  CUDACHECK(cudaMalloc(&ctr, sizeof(int)));
  CUDACHECK(cudaMalloc(
      &dTiles, (maxRows / BM + kELoc * 4 + 8) * (kN / BN) * sizeof(TileDesc)));

  // Small values: bf16 has 8 mantissa bits, and a K=2048 reduction of large
  // random values would be dominated by accumulation order rather than by
  // whether the kernel is correct.
  {
    std::vector<__nv_bfloat16> hw((size_t)kELoc * kN * kH), ha(maxRows * kH);
    for (size_t i = 0; i < hw.size(); i++)
      hw[i] =
          __float2bfloat16((float)((i * 2654435761u) % 17) * 0.0625f - 0.5f);
    for (size_t i = 0; i < ha.size(); i++)
      ha[i] = __float2bfloat16((float)((i * 40503u) % 13) * 0.125f - 0.75f);
    CUDACHECK(cudaMemcpy(W, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(A, ha.data(), ha.size() * 2, cudaMemcpyHostToDevice));
  }

  const size_t smemBytes =
      (size_t)NSTAGE * (BM + BN) * LDA * sizeof(__nv_bfloat16);
  CUDACHECK(cudaFuncSetAttribute(groupedGemmMMA,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)smemBytes));

  int nSM = 0;
  CUDACHECK(cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, 0));
  // Ask the driver rather than assuming: registers and smem both bind here, and
  // a persistent grid that under-fills the SMs is the easiest way to lose 2x.
  int bpsm = 0;
  CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm, groupedGemmMMA,
                                                          kThreads, smemBytes));
  if (bpsm < 1) bpsm = 1;

  // Warp-specialised kernel: same smem tiles plus 2*NSTAGE mbarriers.
  const size_t smemWS = smemBytes + 2 * NSTAGE * sizeof(uint64_t);
  CUDACHECK(cudaFuncSetAttribute(
      groupedGemmWS, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemWS));
  int bpsmWS = 0;
  CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &bpsmWS, groupedGemmWS, kThreadsWS, smemWS));
  if (bpsmWS < 1) bpsmWS = 1;

  // Grouped cuBLAS is the real bar: cublasGemmEx called once per expert in a
  // loop is 1.3-2.0x slower than cublasGemmGroupedBatchedEx on these shapes,
  // and comparing against the loop is what made an earlier version of the
  // 4-rank ladder report a decode win that did not exist.
  cublasOperation_t gTA[kELoc], gTB[kELoc];
  int gM[kELoc], gN[kELoc], gK[kELoc], gLdA[kELoc], gLdB[kELoc], gLdC[kELoc],
      gGS[kELoc];
  float gAl[kELoc], gBe[kELoc];
  __nv_bfloat16 const **dGA = nullptr, **dGB = nullptr;
  __nv_bfloat16** dGC = nullptr;
  CUDACHECK(cudaMalloc(&dGA, kELoc * sizeof(void*)));
  CUDACHECK(cudaMalloc(&dGB, kELoc * sizeof(void*)));
  CUDACHECK(cudaMalloc(&dGC, kELoc * sizeof(void*)));

  std::printf("# gather-fused bf16 grouped GEMM vs cuBLAS, 1x L4 sm_89\n");
  std::printf(
      "# DeepSeek-V2-Lite at EP=4: K=H=%d N=%d, %d local experts, top-%d\n", kH,
      kN, kELoc, kTopK);
  std::printf(
      "# tile %dx%dx%d, %d stages, %d warps at %dx%d, %d threads, %zu B smem\n",
      BM, BN, BK, NSTAGE, kWarps, kWarpsM, kWarpsN, kThreads, smemBytes);
  std::printf(
      "# %d SMs, %d blocks/SM resident; warp tile %dx%d, %d mma per 16-step\n",
      nSM, bpsm, kWarpM, kWarpN, kMFrag * kNFrag);
  std::printf(
      "# bytes/flop = %.4f, but see BENCH_FIXED_K: this kernel is "
      "latency-bound, not that\n",
      (double)(BM + BN) * 2.0 / (2.0 * BM * BN));
  std::printf(
      "# cuBLAS reads the same n-major weights with OP_T; ours additionally "
      "gathers A\n#\n");
  std::printf(
      "# warp-spec: %d producer + %d consumer warps, %d threads, %zu B smem, "
      "%d blocks/SM\n",
      kProdWarps, kConsWarps, kThreadsWS, smemWS, bpsmWS);
  std::printf("# %7s %7s %9s %9s %10s %9s %8s %8s %10s\n", "T/rank", "M_e",
              "ours(us)", "ws(us)", "cubGrp(us)", "wsTF/s", "ws/grp",
              "ours/grp", "maxrelerr");

  float const alpha = 1.0f, beta = 0.0f;
  cudaEvent_t e0, e1;
  CUDACHECK(cudaEventCreate(&e0));
  CUDACHECK(cudaEventCreate(&e1));

  for (int ti = 0; ti < nT; ti++) {
    int const T = kTs[ti];
    int const m = (T * kTopK) / kELoc;
    if (m < 1) continue;
    const size_t totalRows = (size_t)m * kELoc;

    // Identity gather here: the point of the standalone probe is the GEMM, and
    // a scrambled rowIdx would confound its cost with L2 behaviour that the
    // real dispatch buffer does not have. The gather path is still exercised --
    // every A load goes through rowIdx.
    {
      std::vector<int> hr(totalRows);
      for (size_t i = 0; i < totalRows; i++) hr[i] = (int)i;
      CUDACHECK(cudaMemcpy(rowIdx, hr.data(), totalRows * sizeof(int),
                           cudaMemcpyHostToDevice));
    }

    // Tile order decides L2 reuse, and it is worth more than anything in the
    // inner loop. Each tile reads a 128-wide n-slab of the expert's weights,
    // BN*K*2 = 512 KB of it, and only the m index changes between tiles that
    // share one. Enumerating n fastest (the obvious order) puts every
    // concurrent block on a different slab; enumerating m fastest keeps a
    // block-window's worth of slabs resident. At T=2048 the window is ~19 slabs
    // plus the expert's A, about 13 MB against this L4's 48 MB L2.
    std::vector<TileDesc> ht;
    for (int e = 0; e < kELoc; e++)
      for (int nt = 0; nt < kN / BN; nt++)
        for (int mt = 0; mt * BM < m; mt++) {
          TileDesc d;
          d.rowStart = e * m + mt * BM;
          d.mRows = (m - mt * BM) < BM ? (m - mt * BM) : BM;
          d.expert = e;
          d.group = 0;
          d.nTile = nt;
          ht.push_back(d);
        }
    CUDACHECK(cudaMemcpy(dTiles, ht.data(), ht.size() * sizeof(TileDesc),
                         cudaMemcpyHostToDevice));
    int const nTiles = (int)ht.size();
    // 104 registers and 40 KB of smem both allow 2 blocks per SM, so a
    // persistent grid should ask for both -- one per SM leaves each scheduler
    // with 2 warps, which is not enough to hide the ldmatrix latency.
    int const want = bpsm * nSM;
    int const grid = want < nTiles ? want : nTiles;

    auto ours = [&]() {
      CUDACHECK(cudaMemsetAsync(ctr, 0, sizeof(int), stream));
      groupedGemmMMA<<<grid, kThreads, smemBytes, stream>>>(
          dTiles, nTiles, ctr, A, rowIdx, W, Y, kH, kN, nullptr);
    };
    int const gridWS = std::min(bpsmWS * nSM, nTiles);
    auto oursWS = [&]() {
      CUDACHECK(cudaMemsetAsync(ctr, 0, sizeof(int), stream));
      groupedGemmWS<<<gridWS, kThreadsWS, smemWS, stream>>>(
          dTiles, nTiles, ctr, A, rowIdx, W, Y, kH, kN);
    };
    {
      __nv_bfloat16 const *hA[kELoc], *hB[kELoc];
      __nv_bfloat16* hC[kELoc];
      for (int e = 0; e < kELoc; e++) {
        gTA[e] = CUBLAS_OP_T;
        gTB[e] = CUBLAS_OP_N;
        gM[e] = kN;
        gN[e] = m;
        gK[e] = kH;
        gLdA[e] = kH;
        gLdB[e] = kH;
        gLdC[e] = kN;
        gGS[e] = 1;
        gAl[e] = 1.0f;
        gBe[e] = 0.0f;
        hA[e] = W + (size_t)e * kN * kH;
        hB[e] = A + (size_t)e * m * kH;
        hC[e] = Yref + (size_t)e * m * kN;
      }
      CUDACHECK(cudaMemcpy(dGA, hA, sizeof(hA), cudaMemcpyHostToDevice));
      CUDACHECK(cudaMemcpy(dGB, hB, sizeof(hB), cudaMemcpyHostToDevice));
      CUDACHECK(cudaMemcpy(dGC, hC, sizeof(hC), cudaMemcpyHostToDevice));
    }
    auto cubGrp = [&]() {
      CUBLASCHECK(cublasGemmGroupedBatchedEx(
          h, gTA, gTB, gM, gN, gK, gAl, (const void* const*)dGA, CUDA_R_16BF,
          gLdA, (const void* const*)dGB, CUDA_R_16BF, gLdB, gBe,
          (void* const*)dGC, CUDA_R_16BF, gLdC, kELoc, gGS,
          CUBLAS_COMPUTE_32F));
    };
    // C[m][n] = A[m][K] . W[n][K]^T, row-major, via one column-major TN call.
    auto cub = [&](__nv_bfloat16* out) {
      for (int e = 0; e < kELoc; e++)
        CUBLASCHECK(cublasGemmEx(
            h, CUBLAS_OP_T, CUBLAS_OP_N, kN, m, kH, &alpha,
            W + (size_t)e * kN * kH, CUDA_R_16BF, kH, A + (size_t)e * m * kH,
            CUDA_R_16BF, kH, &beta, out + (size_t)e * m * kN, CUDA_R_16BF, kN,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };

    double maxRel = -1.0;
    if (check) {
      CUDACHECK(cudaMemsetAsync(Y, 0, totalRows * kN * 2, stream));
      oursWS();
      cub(Yref);
      CUDACHECK(cudaStreamSynchronize(stream));
      std::vector<__nv_bfloat16> a(totalRows * kN), b(totalRows * kN);
      CUDACHECK(cudaMemcpy(a.data(), Y, a.size() * 2, cudaMemcpyDeviceToHost));
      CUDACHECK(
          cudaMemcpy(b.data(), Yref, b.size() * 2, cudaMemcpyDeviceToHost));
      maxRel = 0.0;
      for (size_t i = 0; i < a.size(); i++) {
        double const x = (double)__bfloat162float(a[i]),
                     y = (double)__bfloat162float(b[i]);
        double const den = (y < 0 ? -y : y) + 1e-3;
        double const r = ((x - y) < 0 ? (y - x) : (x - y)) / den;
        if (r > maxRel) maxRel = r;
      }
    }

    auto time = [&](auto&& fn) -> double {
      for (int i = 0; i < warmup; i++) fn();
      CUDACHECK(cudaStreamSynchronize(stream));
      CUDACHECK(cudaEventRecord(e0, stream));
      for (int i = 0; i < iters; i++) fn();
      CUDACHECK(cudaEventRecord(e1, stream));
      CUDACHECK(cudaEventSynchronize(e1));
      float ms = 0;
      CUDACHECK(cudaEventElapsedTime(&ms, e0, e1));
      return (double)ms * 1000.0 / iters;
    };

    double const uus = time(ours);
    double const wus = time(oursWS);
    double const gus = time(cubGrp);
    double const flops = 2.0 * (double)totalRows * kH * kN;
    double const wtf = flops / (wus * 1e-6) / 1e12;

    std::printf("  %7d %7d %9.2f %9.2f %10.2f %9.2f %7.2fx %7.2fx %10.2e\n", T,
                m, uus, wus, gus, wtf, gus / wus, gus / uus, maxRel);
    std::fflush(stdout);
  }

  cudaFree(dTiles);
  cudaFree(ctr);
  cudaFree(rowIdx);
  cudaFree(Yref);
  cudaFree(Y);
  cudaFree(A);
  cudaFree(W);
  cublasDestroy(h);
  return 0;
}
