// Fused MoE dispatch + GEMM for a PCIe-only 4-GPU node, built on MSCCL++.
//
// Topology assumed: 4x NVIDIA L4 (sm_89), PCIe Gen4 x16 each, no NVLink, dual
// Xeon Silver 4410Y.
//     NUMA0: GPU0 GPU1     NUMA1: GPU2 GPU3
//
// The expert-parallel MoE step at EP=4, DeepSeek-V2-Lite shape: each rank holds
// 16 of the 64 routed experts, each token picks 6, and a token crosses to a
// peer once if any of its experts lives there. That dispatch is a ragged
// All-to-All, and the GEMM that follows is grouped over the 16 local experts.
//
// Why fuse. docs/allreduce.md concluded "overlap does not exist" on this fabric
// and docs/alltoall.md priced a 10% kernel share at 74.11 -> 71.24 GB/s. Both
// measured a *bandwidth-bound* kernel. benchmarks/probes/moe_overlap_probe.cu
// re-runs the question with the MoE GEMM, which is compute-bound, and gets the
// opposite answer: 85% of the dispatch hides, and the copy engine loses nothing
// at all (22.60 GB/s under full GEMM load, against 22.6 uncontended). The GEMM
// pays instead, 8-22%. That is what makes this worth building, and it also
// settles the partitioning: the fabric does not need protecting, so one
// dispatcher block and every other block on GEMM.
//
// The ladder. Every row runs in this one binary on the same data with the same
// harness, and each changes exactly one thing from the row above, so "our
// collective is faster", "pipelining helps" and "fusion helps" stay separately
// attributable -- the shape of the 14x ladder in docs/alltoall.md.
//
//   gemm-only          no comm                        compute floor
//   disp-only          no GEMM                        comm floor
//   nccl+cublas        NCCL a2av, then cuBLAS         the baseline
//   nccl+cublas-pipe   + per-source event pipelining
//   ours+cublas        + our ce-shift dispatch
//   ours+cublas-pipe   + per-step spin-kernel pipelining
//   fused-ce           one kernel, copy-engine dispatch, tile-granular
//   fused-sm           one kernel, SM-put dispatch
//
// What fusion buys beyond overlap: the A operand of the GEMM is gathered
// straight out of the dispatch receive buffer by the cp.async that would have
// loaded it anyway. The non-fused rows have to run a permute kernel first to
// materialise a contiguous A_e, which benchmarks/probes/gemm_probe.cu prices
// at 4.3%.
//
// Scope: dispatch + GEMM. The combine (the return All-to-All and the weighted
// reduce) is not here.
//
// Build: see Makefile

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/numeric_types.h"
#include "cutlass_gated_grouped.h"
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/memory_channel_device.hpp>
#include <mscclpp/packet_device.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <mscclpp/semaphore.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <sys/wait.h>
#include <unistd.h>

#define NRANKS 4
#define NPEERS (NRANKS - 1)

#define CUDACHECK(cmd)                                                   \
  do {                                                                   \
    cudaError_t e = (cmd);                                               \
    if (e != cudaSuccess) {                                              \
      std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e));                               \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

#define CUTLASSCHECK(cmd)                                             \
  do {                                                                \
    cutlass::Status s_ = (cmd);                                       \
    if (s_ != cutlass::Status::kSuccess) {                            \
      std::fprintf(stderr, "%s:%d CUTLASS error %d (%s)\n", __FILE__, \
                   __LINE__, (int)s_, cutlassGetStatusString(s_));    \
      std::exit(1);                                                   \
    }                                                                 \
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

#define NCCLCHECK(cmd)                                                   \
  do {                                                                   \
    ncclResult_t r = (cmd);                                              \
    if (r != ncclSuccess) {                                              \
      std::fprintf(stderr, "%s:%d NCCL error: %s\n", __FILE__, __LINE__, \
                   ncclGetErrorString(r));                               \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

using Chan = mscclpp::MemoryChannelDeviceHandle;
using PortChan = mscclpp::PortChannelDeviceHandle;

__device__ mscclpp::DeviceSyncer gSyncer;

struct Chans {
  Chan c[NRANKS];
};
struct PChans {
  PortChan c[NRANKS];
};

// DeepSeek-V2-Lite, EP=4.
constexpr int kH = 2048;  // hidden; the GEMM's K
constexpr int kN = 2816;  // moe_intermediate 1408, gate and up fused
constexpr int kE = 64;    // routed experts
constexpr int kELoc = kE / NRANKS;
constexpr int kTopK = 6;

// GEMM tile. benchmarks/probes/mma_gemm_probe.cu sweeps this on one GPU; the
// shape barely matters there (everything from 128x128x32 to 256x128x64 lands
// within 39.5 +/- 3 TF/s) because that kernel is latency-bound rather than
// bandwidth-bound. Here BK matters for a different reason: the A operand is
// gathered, so each row contributes BK*2 contiguous bytes per K-step and BK
// sets the transaction size of a scattered read. Overridable to sweep it.
#ifndef TILE_BM
#define TILE_BM 128
#endif
#ifndef TILE_BN
#define TILE_BN 128
#endif
#ifndef TILE_BK
// 64, not the 32 the standalone probe prefers. Each gathered row contributes
// BK*2 contiguous bytes per K-step, so BK sets the transaction size of a
// scattered read: at 32 that is a 64-byte half-line per row and the A loads
// run at half efficiency. Measured at T=2048: 4724 us at BK=32 against 4265
// at BK=64. 128 does not fit -- 136 KB of smem against Ada's 99 KB limit.
#define TILE_BK 64
#endif
constexpr int BM = TILE_BM, BN = TILE_BN, BK = TILE_BK;
constexpr int NSTAGE = 2;
constexpr int PAD = 8;
constexpr int LDA = BK + PAD;
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kWarpsM = 2, kWarpsN = kWarps / kWarpsM;
constexpr int kWarpM = BM / kWarpsM, kWarpN = BN / kWarpsN;
constexpr int kMFrag = kWarpM / 16, kNFrag = kWarpN / 8;
constexpr int kChunkPerRow = BK / 8;
constexpr int kThrPerRowA = kThreads / BM;
constexpr int kThrPerRowB = kThreads / BN;
constexpr int kSubA = kChunkPerRow / kThrPerRowA;
constexpr int kSubB = kChunkPerRow / kThrPerRowB;

static_assert(kN % BN == 0, "n must tile evenly");

// Blocks the SM-put dispatch variant takes away from the GEMM. The copy-engine
// variant needs one block; this one needs enough threads to saturate a PCIe
// flow, and every one of them is an SM not doing GEMM. That trade is the point
// of keeping the variant.
constexpr int kDispBlocks = 8;

// One unit of GEMM work: a BMxBN output tile of one expert, whose rows all came
// from one arrival group. `group` is what the fused kernel waits on.
struct TileDesc {
  int rowStart;
  int mRows;
  int expert;
  int group;
  int nTile;
};

__device__ __host__ __forceinline__ int destOf(int rank, int k) {
  return (rank + k) % NRANKS;
}
__device__ __host__ __forceinline__ int srcOf(int rank, int k) {
  return (rank - k + NRANKS) % NRANKS;
}
__device__ __host__ __forceinline__ int ascPeer(int rank, int i) {
  return (i < rank) ? i : i + 1;
}

// Routing is a pure function of (rank, token), so every rank can recompute
// every other rank's assignment. That is what makes the dispatch check exact
// rather than statistical: a row that lands in the wrong slot, from the wrong
// sender, or at the wrong offset is all caught.
__host__ __forceinline__ void routeOf(int s, int t, int* experts) {
  uint32_t h =
      (uint32_t)(s * 2654435761u) ^ (uint32_t)(t * 40503u) ^ 0x9e3779b9u;
  int n = 0;
  while (n < kTopK) {
    h ^= h << 13;
    h ^= h >> 17;
    h ^= h << 5;
    int const e = (int)(h % kE);
    bool dup = false;
    for (int i = 0; i < n; i++)
      if (experts[i] == e) dup = true;
    if (!dup) experts[n++] = e;
  }
}

// ---------------------------------------------------------------------------
// Staging: gather the rows this rank owes each destination.
// ---------------------------------------------------------------------------
__global__ void stageRows(__nv_bfloat16* __restrict__ dst,
                          __nv_bfloat16 const* __restrict__ x,
                          int const* __restrict__ tok, int nRows, int H) {
  int const r = blockIdx.x;
  if (r >= nRows) return;
  __nv_bfloat16 const* s = x + (size_t)tok[r] * H;
  __nv_bfloat16* d = dst + (size_t)r * H;
  for (int i = threadIdx.x; i < H / 8; i += blockDim.x)
    reinterpret_cast<float4*>(d)[i] = reinterpret_cast<float4 const*>(s)[i];
}

// The permute every non-fused row needs: materialise a contiguous A_e from the
// receive buffer. The fused kernel folds this into its A-operand cp.async and
// never runs it.
__global__ void permuteRows(__nv_bfloat16* __restrict__ dst,
                            __nv_bfloat16 const* __restrict__ src,
                            int const* __restrict__ rowIdx, int nRows, int H) {
  int const r = blockIdx.x;
  if (r >= nRows) return;
  __nv_bfloat16 const* s = src + (size_t)rowIdx[r] * H;
  __nv_bfloat16* d = dst + (size_t)r * H;
  for (int i = threadIdx.x; i < H / 8; i += blockDim.x)
    reinterpret_cast<float4*>(d)[i] = reinterpret_cast<float4 const*>(s)[i];
}

// ---------------------------------------------------------------------------
// Dispatch, standalone. Copy-engine rotation: step k sends to (r+k)%4 and
// receives from (r-k)%4, the (2,4,2) cross-socket split docs/alltoall.md proves
// optimal for this box. Ragged sizes are the only difference from that file --
// each flow carries only the rows that rank actually owes.
// ---------------------------------------------------------------------------
__global__ void dispatchCE(PChans pc, int rank, size_t const* sendOff,
                           size_t const* recvOff, size_t const* sendBytes,
                           int barriered) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  for (int k = 1; k <= NPEERS; k++) {
    int const d = destOf(rank, k);
    pc.c[d].putWithSignal(recvOff[rank], sendOff[d], sendBytes[d]);
    if (barriered) pc.c[srcOf(rank, k)].wait();
  }
  if (!barriered)
    for (int i = 0; i < NPEERS; i++) pc.c[ascPeer(rank, i)].wait();
}

// SM-issued put, one destination at a time with a grid sync between steps.
// Without the sync the blocks free-run and different blocks of the same grid
// address different peers, which is fan-out assembled by accident -- and
// docs/alltoall.md measures that at 5 GB/s against an 82 GB/s ceiling.
__global__ void __launch_bounds__(1024)
    dispatchSM(Chans ch, int rank, size_t const* sendOff, size_t const* recvOff,
               size_t const* sendBytes) {
  int const tid = blockIdx.x * blockDim.x + threadIdx.x;
  int const nth = gridDim.x * blockDim.x;
  for (int k = 1; k <= NPEERS; k++) {
    int const d = destOf(rank, k);
    ch.c[d].put<16, true>(recvOff[rank], sendOff[d], sendBytes[d], tid, nth);
    if (k < NPEERS) gSyncer.sync(gridDim.x);
  }
  gSyncer.sync(gridDim.x);
  if (tid < NPEERS) {
    int const p = ascPeer(rank, tid);
    ch.c[p].signal();
    ch.c[p].wait();
  }
  gSyncer.sync(gridDim.x);
}

// ---------------------------------------------------------------------------
// The GEMM, and the fused kernel that is the same GEMM with a dispatcher
// bolted into block 0 and an arrival wait in front of each tile.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smemAddr(void const* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
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

// Grouped GEMM over an arrival-ordered tile queue.
//
// `mode`: 0 = plain GEMM, no comm at all (the non-fused rows use this).
//         1 = fused, copy engine. Block 0 issues the proxy triggers and
//             publishes arrivals; every other block does GEMM.
//         2 = fused, SM put. Block 0 does the same over MemoryChannels.
//
// Tiles are enqueued group 0 first (this rank's own rows, which need no comm at
// all), so the workers have real work from cycle zero and only ever spin on the
// group actually in flight.
__global__ void __launch_bounds__(kThreads) fusedMoe(
    PChans pc, Chans ch, int rank, int mode, size_t const* __restrict__ sendOff,
    size_t const* __restrict__ recvOff, size_t const* __restrict__ sendBytes,
    uint32_t* __restrict__ arrived, TileDesc const* __restrict__ tiles,
    int nTiles, int* __restrict__ ctr, __nv_bfloat16 const* __restrict__ Aall,
    int const* __restrict__ rowIdx, __nv_bfloat16 const* __restrict__ W,
    __nv_bfloat16* __restrict__ Y, int K, int N) {
  if (mode != 0 && blockIdx.x < (mode == 2 ? kDispBlocks : 1)) {
    // Announce residency before doing anything. A gated GEMM on another stream
    // has to know the dispatcher actually got an SM before it floods the GPU
    // with spinning blocks -- otherwise the two race, and if the GEMM wins
    // there is nobody left to publish the arrivals it is waiting for. Slot
    // NRANKS is reserved for this; the arrival flags themselves are
    // 0..NRANKS-1.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      __threadfence_system();
      atomicExch(&arrived[NRANKS], 1u);
    }
    if (mode == 1) {
      // Copy-engine dispatch. One block, because
      // benchmarks/probes/moe_overlap_probe.cu shows the copy engine loses
      // nothing to a concurrent GEMM -- there is no reason to reserve SMs to
      // protect the fabric.
      if (threadIdx.x == 0) {
        for (int k = 1; k <= NPEERS; k++)
          pc.c[destOf(rank, k)].putWithSignal(recvOff[rank],
                                              sendOff[destOf(rank, k)],
                                              sendBytes[destOf(rank, k)]);
        for (int k = 1; k <= NPEERS; k++) {
          pc.c[srcOf(rank, k)].wait();
          // The bytes were written by a DMA engine, not by a thread. wait() is
          // a system-scope acquire, and this fence carries that ordering to the
          // worker blocks, which pick it up by spinning on `arrived` and
          // fencing after. One thread, three times a launch -- not the
          // grid-wide fence docs/allreduce.md prices at ~35 us.
          __threadfence_system();
          atomicExch(&arrived[k], 1u);
        }
      }
    } else {
      // SM-issued dispatch, the control that shows why the copy engine is the
      // right mechanism here: these blocks are SMs taken away from the GEMM,
      // and they need a grid sync per step or they free-run into fan-out.
      int const tid = blockIdx.x * blockDim.x + threadIdx.x;
      int const nth = kDispBlocks * blockDim.x;
      for (int k = 1; k <= NPEERS; k++) {
        int const d = destOf(rank, k);
        ch.c[d].put<16, true>(recvOff[rank], sendOff[d], sendBytes[d], tid,
                              nth);
        gSyncer.sync(kDispBlocks);
        if (tid == 0) ch.c[d].signal();
        int const q = srcOf(rank, k);
        if (tid == 0) {
          ch.c[q].wait();
          __threadfence_system();
          atomicExch(&arrived[k], 1u);
        }
        gSyncer.sync(kDispBlocks);
      }
    }
    return;
  }

  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16* sA = smem;
  __nv_bfloat16* sB = smem + NSTAGE * BM * LDA;

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
    if (mode != 0 && td.group != 0) {
      if (tid == 0)
        while (*(uint32_t volatile*)&arrived[td.group] == 0) {
        }
      __syncthreads();
      __threadfence();
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

    int const gRowA = (rowA < td.mRows) ? rowIdx[td.rowStart + rowA] : -1;

    // All inner-loop addressing is an affine function of the lane plus a
    // compile-time offset, hoisted out so the k loop has no 64-bit math.
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
      const int k0 = kStep * BK;
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
#pragma unroll
    for (int s = 0; s < NSTAGE - 1; s++) stage(s, s);

    for (int ks = 0; ks < nSteps; ks++) {
      int const nxt = ks + NSTAGE - 1;
      if (nxt < nSteps) stage(nxt, nxt % NSTAGE);
      // `wait_group n` is already satisfied when fewer than n groups are
      // pending, so near the tail this count has to fall or the last steps
      // compute from unfilled buffers -- silently.
      int const pending = (nxt < nSteps ? nxt : nSteps - 1) - ks;
      switch (pending) {
        case 0:
          asm volatile("cp.async.wait_group 0;\n" ::);
          break;
        default:
          asm volatile("cp.async.wait_group 1;\n" ::);
          break;
      }
      __syncthreads();

      int const buf = ks % NSTAGE;
      const uint32_t pA = aFrag + buf * kBufA;
      const uint32_t pB = bFrag + buf * kBufB;

#pragma unroll
      for (int kk = 0; kk < BK; kk += 16) {
        uint32_t a[kMFrag][4], b[kNFrag][2];
#pragma unroll
        for (int mi = 0; mi < kMFrag; mi++)
          ldm4(pA + mi * 16 * kRow + kk * 2, a[mi][0], a[mi][1], a[mi][2],
               a[mi][3]);
          // B is n-major, so one ldmatrix.x4 covers two 16x8 fragments.
#pragma unroll
        for (int nj = 0; nj < kNFrag; nj += 2) {
          uint32_t q0, q1, q2, q3;
          ldm4(pB + nj * 8 * kRow + kk * 2, q0, q1, q2, q3);
          b[nj][0] = q0;
          b[nj][1] = q2;
          b[nj + 1][0] = q1;
          b[nj + 1][1] = q3;
        }
#pragma unroll
        for (int mi = 0; mi < kMFrag; mi++)
#pragma unroll
          for (int nj = 0; nj < kNFrag; nj++)
            mma16816(acc[mi][nj][0], acc[mi][nj][1], acc[mi][nj][2],
                     acc[mi][nj][3], a[mi][0], a[mi][1], a[mi][2], a[mi][3],
                     b[nj][0], b[nj][1]);
      }
      __syncthreads();
    }

    int const cr = lane >> 2, cc = (lane & 3) << 1;
#pragma unroll
    for (int mi = 0; mi < kMFrag; mi++)
#pragma unroll
      for (int nj = 0; nj < kNFrag; nj++)
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
    __syncthreads();
  }
}

// Spin until a peer's dispatch has landed. Launched on the GEMM stream ahead of
// a cuBLAS call, this is how the non-fused pipelined rows express "start this
// group's GEMM the moment its rows arrive" without leaving the stream model.
__global__ void waitArrival(uint32_t const* arrived, int group) {
  if (threadIdx.x == 0)
    while (*(uint32_t volatile*)&arrived[group] == 0) {
    }
}
__global__ void clearFlags(uint32_t* arrived) {
  // Slot NRANKS is the dispatcher's "I am resident" flag, not an arrival group.
  if (threadIdx.x < NRANKS + 1)
    arrived[threadIdx.x] = (threadIdx.x == 0) ? 1u : 0u;
}

// Publish "gate group g's rows are permuted and readable". The fence is the
// release half of the pair; the gated GEMM's spin is the acquire half.
__global__ void publishFlag(uint32_t* flags, int g) {
  if (threadIdx.x == 0) {
    __threadfence_system();
    flags[g] = 1u;
  }
}
__global__ void clearGateFlags(uint32_t* flags, int n) {
  if ((int)threadIdx.x < n) flags[threadIdx.x] = 0u;
}

// ---------------------------------------------------------------------------
// Host-side scaffolding
// ---------------------------------------------------------------------------
// The `-grp` rows call cublasGemmGroupedBatchedEx, which issues all 16 experts
// as one grouped GEMM instead of 16 separate launches. Without them the
// baseline is artificially weak exactly where this kernel claims its win: at
// decode, M_e drops to single digits and 16 sequential launches each underfill
// the GPU, which is a property of how the baseline was called rather than of
// cuBLAS. Both are kept so the difference is visible rather than assumed.
// CUTLASS grouped GEMM, and the same kernel with the arrival gate.
//
// Two arrival groups, not four, because the group-major problem order that
// gating requires is what keeps a window of concurrent blocks on few enough
// experts to stay in L2: at 4 groups that ordering costs 2.6x at decode, at 2
// groups it costs 4% (benchmarks/data/cutlass_gemm_probe.txt).
//
// Tile shape is a build knob because the right answer moved. 128x128 was forced
// for a while by an SM co-residency problem: a 96 KB tile leaves no room for
// any other kernel on the SM, and back when the gather ran as a separate kernel
// that deadlocked. With gather-A folded in and the dispatcher living in block 0
// there is no second kernel, so the constraint is gone and the sweep's winner
// is available again -- 128x256 measured 1.20x grouped cuBLAS against 128x128's
// 1.16x on the group-major split (benchmarks/data/cutlass_gemm_probe.txt).
#ifndef FM_BM
#define FM_BM 128
#endif
#ifndef FM_BN
#define FM_BN 256
#endif

using CtElem = cutlass::bfloat16_t;
using CtKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    CtElem, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,
    CtElem, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,
    CtElem, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80, cutlass::gemm::GemmShape<FM_BM, FM_BN, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<CtElem, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4>::GemmKernel;
using CtGemm = cutlass::gemm::device::GemmGrouped<CtKernel>;
using CtGated = uccl_lite::GatedGrouped<CtKernel>;

// The same grouped kernel built with gather-A, so the GEMM reads the receive
// buffer's scattered rows through the row-index list instead of a permute
// kernel materialising them first.
//
// This is what makes the fused path schedulable. With a separate permute, that
// permute has to run *while* the gated GEMM spins, and a persistent GEMM whose
// blocks each hold most of an SM's shared memory cannot share an SM with
// anything -- measured: deadlock at prefill even with 12 SMs held back.
// Gather-A deletes the second kernel, so the only thing that still needs a free
// SM is the dispatcher, and one is enough. It costs 3.9% against the permute it
// replaces (benchmarks/data/cutlass_gemm_probe.txt).
//
// DefaultGemmGrouped hardcodes GatherA=false, so this reassembles what it does:
// DefaultGemm does expose the flag, and kInternalTranspose is false for
// RowMajor C, so the gather really lands on A.
using CtGatherBase = typename cutlass::gemm::kernel::DefaultGemm<
    CtElem, cutlass::layout::RowMajor, 8, CtElem, cutlass::layout::ColumnMajor,
    8, CtElem, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80, cutlass::gemm::GemmShape<FM_BM, FM_BN, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<CtElem, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 4, true,
    typename cutlass::gemm::device::DefaultGemmConfiguration<
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, CtElem, CtElem,
        CtElem, float>::Operator,
    cutlass::gemm::SharedMemoryClearOption::kNone, true, false,
    false>::GemmKernel;
using CtGatherKernel = cutlass::gemm::kernel::GemmGrouped<
    typename CtGatherBase::Mma, typename CtGatherBase::Epilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>;
// Block 0's job in the fused kernel: the ce-shift copy-engine dispatch, exactly
// as fusedMoe's mode 1 does it, publishing arrived[k] as each peer lands. Held
// by value in the kernel's Params.
struct MoeDispatch {
  static constexpr bool kEnabled = true;
  static constexpr int kBlocks = 1;
  PChans pc;
  int rank;
  size_t const* sendOff;
  size_t const* recvOff;
  size_t const* sendBytes;
  uint32_t* arrived;

  CUTLASS_DEVICE void operator()() {
    if (threadIdx.x != 0) return;
    for (int k = 1; k <= NPEERS; k++)
      pc.c[destOf(rank, k)].putWithSignal(
          recvOff[rank], sendOff[destOf(rank, k)], sendBytes[destOf(rank, k)]);
    for (int k = 1; k <= NPEERS; k++) {
      pc.c[srcOf(rank, k)].wait();
      // The bytes were written by a DMA engine, not a thread. wait() is a
      // system-scope acquire; this fence carries that ordering to the worker
      // blocks, which pick it up by spinning on `arrived`.
      __threadfence_system();
      atomicExch(&arrived[k], 1u);
    }
  }
};

using CtGatedGather = uccl_lite::GatedGrouped<CtGatherKernel, MoeDispatch>;

// The same gather-A GEMM with no dispatcher and no gate. Launched after the
// collective has finished, it is exactly the fused kernel's compute run
// sequentially -- which is what separates "our GEMM is faster than cuBLAS" from
// "the communication is hidden". Without this rung the two are
// indistinguishable in the ladder, and the fused row would get credit for both.
using CtUngatedGather = uccl_lite::GatedGrouped<CtGatherKernel>;

constexpr int kGateGroups = 2;

enum Variant {
  V_GEMM_ONLY = 0,
  V_GEMM_ONLY_GRP,
  V_MMA_ONLY,
  V_DISP_ONLY,
  V_NCCL_DISP_ONLY,
  V_NCCL_CUBLAS,
  V_NCCL_CUBLAS_GRP,
  V_NCCL_CUBLAS_PIPE,
  V_OURS_CUBLAS,
  V_OURS_CUBLAS_GRP,
  V_NCCL_CUBLAS_GRP_PIPE,
  V_OURS_CUBLAS_PIPE,
  V_OURS_CUTLASS,
  V_FUSED_CUTLASS,
  V_FUSED_CUTLASS_G,
  V_FUSED_CUTLASS_T,
  V_FUSED_CE,
  V_FUSED_SM,
  NVAR
};
static char const* kName[NVAR] = {"gemm-only",
                                  "gemm-only-grp",
                                  "mma-only",
                                  "disp-only",
                                  "nccl-disp-only",
                                  "nccl+cublas",
                                  "nccl+cublas-grp",
                                  "nccl+cublas-pipe",
                                  "ours+cublas",
                                  "ours+cublas-grp",
                                  "nccl+cublas-grp-pipe",
                                  "ours+cublas-pipe",
                                  "ours+cutlass",
                                  "fused-cutlass",
                                  "fused-cutlass-gather",
                                  "fused-cutlass-tile",
                                  "fused-ce",
                                  "fused-sm"};

struct Cfg {
  int minT = 8, maxT = 2048, factorT = 4;
  // Few iterations per visit and many rounds, because the variants are
  // interleaved: short visits keep a slow variant from throttling itself inside
  // its own measurement. See the timing block in main().
  int warmup = 2, iters = 4, rounds = 12;
  double settleMs = 2500.0;
  int smBlocks = 24, smThreads = 256;
  bool check = true;
  int only = -1;
  // -X <variant> drops a row from the run. Useful when one row is unstable and
  // the rest of the ladder still has to be measured.
  bool skip[NVAR] = {false};
  // SMs held back from the gated GEMM's grid. 1, so that grid + the dispatcher
  // block is exactly the SM count. It used to be 4, from a period when the
  // dispatcher was a separate kernel and reserving SMs looked like it mattered;
  // it never did -- the stalls being mitigated belonged to ours+cublas-pipe, a
  // different row entirely. With the dispatcher inside the grid, reserving
  // costs 5%: 1.24x at -R 1 against 1.18x at -R 4.
  int reserveSM = 1;
  // Where the four arrival groups are cut into two gate groups. -P 2 gives
  // {own,peer1} and {peer2,peer3}, -P 1 gives {own} and {peer1,peer2,peer3}.
  //
  // 2 is a real optimum, not a default nobody questioned. The obvious argument
  // for 1 is that at -P 2 every tile waits for peer 1, so the GPU idles for the
  // first third of the dispatch even though a quarter of the rows are this
  // rank's own and were never in flight. Measured, that argument loses:
  //
  //             gate rows    tiles/expert   T=2048   T=8192
  //   -P 1       190 / 570     2+5 = 154     1.280x   1.188x
  //   -P 2       380 / 380     3+3 = 132     1.293x   1.216x
  //   -P 3       570 / 190     5+2 = 154     1.114x   stalled
  //
  // An even split is worth more than an early start, because 380 rows is
  // exactly 3 tiles of BM=128 at 99% fill while 190 needs 2 at 74%. The uneven
  // splits pay 17% more tiles and only buy back the ~400 us head start.
  int gateSplit = 2;
};

// Everything routing-derived is precomputed here and uploaded before timing.
// A real server computes it from the gating logits; the point of this benchmark
// is dispatch and GEMM, and charging one row for a metadata exchange the other
// rows get for free would make the ladder mean nothing. The cost of exchanging
// it is small and separable -- see docs/fusion.md.
struct Plan {
  int T = 0;
  std::vector<int> sendTok[NRANKS];  // rows this rank owes each peer
  int nSend[NRANKS] = {0, 0, 0, 0};
  std::vector<int> recvTok[NRANKS];  // rows arriving from each source
  int nRecv[NRANKS] = {0, 0, 0, 0};
  std::vector<int> rowIdx;  // expert-major, group-minor
  int expOff[kELoc] = {0};
  int grpOff[kELoc][NRANKS] = {{0}};
  int grpCnt[kELoc][NRANKS] = {{0}};
  int totalRows = 0;
  std::vector<TileDesc> tiles;
};

static void buildPlan(Plan& p, int rank, int T, int cap) {
  p.T = T;
  int ex[kTopK];
  // What this rank sends: a token crosses to a peer once if any of its experts
  // lives there. That dedup is what makes the flows ragged and unequal, which
  // docs/alltoall.md lists as untested for a rotation schedule.
  for (int d = 0; d < NRANKS; d++) p.sendTok[d].clear();
  for (int t = 0; t < T; t++) {
    routeOf(rank, t, ex);
    bool want[NRANKS] = {false, false, false, false};
    for (int i = 0; i < kTopK; i++) want[ex[i] / kELoc] = true;
    for (int d = 0; d < NRANKS; d++)
      if (want[d]) p.sendTok[d].push_back(t);
  }
  for (int d = 0; d < NRANKS; d++) p.nSend[d] = (int)p.sendTok[d].size();

  // What arrives: recomputed from the same function, so it matches by
  // construction and the check below is exact rather than statistical.
  for (int s = 0; s < NRANKS; s++) {
    p.recvTok[s].clear();
    for (int t = 0; t < T; t++) {
      routeOf(s, t, ex);
      bool want = false;
      for (int i = 0; i < kTopK; i++)
        if (ex[i] / kELoc == rank) want = true;
      if (want) p.recvTok[s].push_back(t);
    }
    p.nRecv[s] = (int)p.recvTok[s].size();
  }

  // Row order: expert-major so cuBLAS sees each expert's rows contiguous, and
  // group-minor inside that so a (group, expert) span is contiguous too. One
  // layout serves both the grouped cuBLAS calls and the arrival-ordered tiles.
  p.rowIdx.clear();
  int pos = 0;
  for (int e = 0; e < kELoc; e++) {
    p.expOff[e] = pos;
    for (int g = 0; g < NRANKS; g++) {
      int const s = (g == 0) ? rank : srcOf(rank, g);
      p.grpOff[e][g] = pos - p.expOff[e];
      for (int j = 0; j < p.nRecv[s]; j++) {
        routeOf(s, p.recvTok[s][j], ex);
        for (int i = 0; i < kTopK; i++)
          if (ex[i] == rank * kELoc + e) {
            // Slots are indexed at the sweep-wide capacity, not at this T, so
            // one registration and one row-index space serve every size.
            p.rowIdx.push_back(s * cap + j);
            pos++;
            break;
          }
      }
      p.grpCnt[e][g] = pos - p.expOff[e] - p.grpOff[e][g];
    }
  }
  p.totalRows = pos;

  // Tiles. The obvious enumeration -- one tile range per (group, expert) -- is
  // a trap at these shapes: it forces a tile boundary at every arrival-group
  // edge, so with ~190 rows per (expert, group) and BM=128 the tiles run 74%
  // full and the tile count goes up 1.35x for the same work. Measured at
  // T=2048: 4656 us that way against 3519 for the same rows unsplit.
  //
  // Instead a tile covers a full BM span of the expert's rows and waits on the
  // group of its *last* row. Rows are group-ordered within an expert, so that
  // is the latest arrival the tile depends on, and tiles early in an expert's
  // range still depend only on early groups. Full tiles and arrival ordering
  // both, rather than one at the cost of the other.
  p.tiles.clear();
  for (int wave = 0; wave < NRANKS; wave++)
    for (int e = 0; e < kELoc; e++) {
      int const mE =
          p.grpCnt[e][0] + p.grpCnt[e][1] + p.grpCnt[e][2] + p.grpCnt[e][3];
      for (int mt = 0; mt * BM < mE; mt++) {
        int const mRows = std::min(BM, mE - mt * BM);
        // Group of the last row in this tile.
        int const last = mt * BM + mRows - 1;
        int g = 0, acc = 0;
        for (int gg = 0; gg < NRANKS; gg++) {
          acc += p.grpCnt[e][gg];
          if (last < acc) {
            g = gg;
            break;
          }
        }
        if (g != wave) continue;  // emit in arrival order
        for (int nt = 0; nt < kN / BN; nt++) {
          TileDesc d;
          d.rowStart = p.expOff[e] + mt * BM;
          d.mRows = mRows;
          d.expert = e;
          d.group = g;
          d.nTile = nt;
          p.tiles.push_back(d);
        }
      }
    }
}

// Deterministic activations, so every rank can reconstruct any other rank's
// tokens and the dispatch check can be exact.
static __nv_bfloat16 xval(int s, int t, int i) {
  const uint32_t h =
      (uint32_t)(s * 7919 + t) * 2654435761u + (uint32_t)i * 40503u;
  return __float2bfloat16((float)(h % 13) * 0.125f - 0.75f);
}

static void runRank(int rank, mscclpp::UniqueId id, Cfg const& cfg) {
  CUDACHECK(cudaSetDevice(rank));
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, NRANKS);
  bootstrap->initialize(id);
  mscclpp::Communicator comm(bootstrap);

  const size_t CAP = (size_t)cfg.maxT;  // rows per (src,dst) slot
  const size_t slotBytes = CAP * kH * sizeof(__nv_bfloat16);
  const size_t sendBase = 0, recvBase = NRANKS * slotBytes;
  const size_t bufBytes = 2 * NRANKS * slotBytes;

  // The proxy FIFO packs each offset into 32 bits (mscclpp fifo_device.hpp,
  // TriggerBitsOffset = 32) and the device-side assert that would catch an
  // overflow is compiled out in release builds, so it would corrupt silently.
  if (bufBytes >= (1UL << 32)) {
    std::fprintf(stderr, "-e %d too large: proxy offsets are 32-bit\n",
                 cfg.maxT);
    std::exit(1);
  }

  char* buf = nullptr;
  CUDACHECK(cudaMalloc(&buf, bufBytes));
  CUDACHECK(cudaMemset(buf, 0, bufBytes));
  auto regMem = comm.registerMemory(buf, bufBytes, mscclpp::Transport::CudaIpc);

  std::vector<std::shared_future<mscclpp::Connection>> connF(NRANKS);
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> memF(NRANKS);
  for (int i = 0; i < NPEERS; i++) {
    int const p = ascPeer(rank, i);
    comm.sendMemory(regMem, p, 0);
    memF[p] = comm.recvMemory(p, 0);
    connF[p] = comm.connect(mscclpp::Transport::CudaIpc, p);
  }
  std::vector<mscclpp::Connection> conns(NRANKS);
  for (int i = 0; i < NPEERS; i++)
    conns[ascPeer(rank, i)] = connF[ascPeer(rank, i)].get();

  std::vector<mscclpp::MemoryChannel> memChans(NRANKS);
  std::vector<mscclpp::PortChannel> portChans(NRANKS);
  auto proxy = std::make_shared<mscclpp::ProxyService>();
  mscclpp::MemoryId midLocal = proxy->addMemory(regMem);
  std::vector<std::shared_future<mscclpp::Semaphore>> semF(NRANKS);
  std::vector<mscclpp::SemaphoreId> sid(NRANKS);
  std::vector<mscclpp::MemoryId> midRemote(NRANKS);
  for (int i = 0; i < NPEERS; i++) {
    int const p = ascPeer(rank, i);
    semF[p] = comm.buildSemaphore(conns[p], p);
    sid[p] = proxy->buildAndAddSemaphore(comm, conns[p]);
    midRemote[p] = proxy->addMemory(memF[p].get());
  }
  Chans hChans{};
  PChans hPChans{};
  for (int i = 0; i < NPEERS; i++) {
    int const p = ascPeer(rank, i);
    mscclpp::MemoryChannel mc(semF[p].get(), memF[p].get(), regMem, buf);
    mscclpp::PortChannel pc =
        proxy->portChannel(sid[p], midRemote[p], midLocal);
    memChans[p] = mc;
    portChans[p] = pc;
    hChans.c[p] = memChans[p].deviceHandle();
    hPChans.c[p] = portChans[p].deviceHandle();
  }
  proxy->startProxy();

  // NCCL over the same four processes; the unique id rides the bootstrap that
  // is already there, so no MPI and no second rendezvous.
  ncclUniqueId nid;
  std::vector<ncclUniqueId> nids(NRANKS);
  if (rank == 0) NCCLCHECK(ncclGetUniqueId(&nid));
  nids[rank] = nid;
  bootstrap->allGather(nids.data(), sizeof(ncclUniqueId));
  ncclComm_t ncclComm;
  NCCLCHECK(ncclCommInitRank(&ncclComm, NRANKS, nids[0], rank));

  cublasHandle_t cub;
  CUBLASCHECK(cublasCreate(&cub));
  // CUTLASS argument arrays and the gate flags. Sized for the widest problem
  // list; the contents are rebuilt per size.
  constexpr int kCtMaxProb = kGateGroups * kELoc;
  cutlass::gemm::GemmCoord* dCtProb = nullptr;
  CtElem **dCtA = nullptr, **dCtB = nullptr, **dCtC = nullptr;
  int64_t *dCtLdA = nullptr, *dCtLdB = nullptr, *dCtLdC = nullptr;
  int* dCtGroupOf = nullptr;
  uint32_t* dGateFlags = nullptr;
  CUDACHECK(
      cudaMalloc(&dCtProb, kCtMaxProb * sizeof(cutlass::gemm::GemmCoord)));
  CUDACHECK(cudaMalloc(&dCtA, kCtMaxProb * sizeof(void*)));
  CUDACHECK(cudaMalloc(&dCtB, kCtMaxProb * sizeof(void*)));
  CUDACHECK(cudaMalloc(&dCtC, kCtMaxProb * sizeof(void*)));
  CUDACHECK(cudaMalloc(&dCtLdA, kCtMaxProb * sizeof(int64_t)));
  CUDACHECK(cudaMalloc(&dCtLdB, kCtMaxProb * sizeof(int64_t)));
  CUDACHECK(cudaMalloc(&dCtLdC, kCtMaxProb * sizeof(int64_t)));
  CUDACHECK(cudaMalloc(&dCtGroupOf, kCtMaxProb * sizeof(int)));
  CUDACHECK(cudaMalloc(&dGateFlags, kGateGroups * sizeof(uint32_t)));
  CtElem** dCtGatherA = nullptr;
  int const** dCtGatherIdx = nullptr;
  int* dCtArrivalOf =
      nullptr;  // problem index -> which arrival flag to wait on
  CUDACHECK(cudaMalloc(&dCtGatherA, kCtMaxProb * sizeof(void*)));
  CUDACHECK(cudaMalloc(&dCtGatherIdx, kCtMaxProb * sizeof(void*)));
  CUDACHECK(cudaMalloc(&dCtArrivalOf, kCtMaxProb * sizeof(int)));
  int const ctGatherSmem = int(sizeof(typename CtGatedGather::SharedStorage));
  CUDACHECK(cudaFuncSetAttribute(uccl_lite::gatedGroupedKernel<CtGatedGather>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 ctGatherSmem));
  // Separate template instantiation, so it needs the >48 KB opt-in of its own.
  // Missing it makes the launch fail silently and the row hang.
  CUDACHECK(cudaFuncSetAttribute(uccl_lite::gatedGroupedKernel<CtUngatedGather>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 ctGatherSmem));
  int const ctSmem = int(sizeof(typename CtGated::SharedStorage));
  CUDACHECK(cudaFuncSetAttribute(uccl_lite::gatedGroupedKernel<CtGated>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 ctSmem));

  cudaStream_t sMain, sComm, sPerm;
  CUDACHECK(cudaStreamCreateWithFlags(&sMain, cudaStreamNonBlocking));
  CUDACHECK(cudaStreamCreateWithFlags(&sComm, cudaStreamNonBlocking));
  // The gather has to run concurrently with the gated GEMM, not before it, so
  // it needs a stream of its own. See the fused-cutlass case for why this
  // cannot deadlock against a GPU full of spinning GEMM blocks.
  CUDACHECK(cudaStreamCreateWithFlags(&sPerm, cudaStreamNonBlocking));
  CUBLASCHECK(cublasSetStream(cub, sMain));
  cudaEvent_t evStep[NRANKS];
  for (auto& ev : evStep)
    CUDACHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));

  // Size everything from the largest plan so one allocation serves the sweep.
  Plan maxPlan;
  buildPlan(maxPlan, rank, cfg.maxT, (int)CAP);
  const size_t maxRows = (size_t)maxPlan.totalRows + BM;

  __nv_bfloat16 *x, *W, *Y, *Yref, *Ap;
  CUDACHECK(cudaMalloc(&x, CAP * kH * sizeof(__nv_bfloat16)));
  CUDACHECK(cudaMalloc(&W, (size_t)kELoc * kN * kH * sizeof(__nv_bfloat16)));
  CUDACHECK(cudaMalloc(&Y, maxRows * kN * sizeof(__nv_bfloat16)));
  CUDACHECK(cudaMalloc(&Yref, maxRows * kN * sizeof(__nv_bfloat16)));

  // Unsplit list: one problem per expert, plus the per-(problem, m-tile) group
  // table that lets a tile derive what it waits for without splitting anything.
  cutlass::gemm::GemmCoord* dTProb = nullptr;
  CtElem **dTA = nullptr, **dTB = nullptr, **dTC = nullptr;
  int const** dTIdx = nullptr;
  int64_t *dTLdA = nullptr, *dTLdB = nullptr, *dTLdC = nullptr;
  int *dTTileGroup = nullptr, *dTTileGroupOff = nullptr;
  CUDACHECK(cudaMalloc(&dTProb, kELoc * sizeof(cutlass::gemm::GemmCoord)));
  CUDACHECK(cudaMalloc(&dTA, kELoc * sizeof(void*)));
  CUDACHECK(cudaMalloc(&dTB, kELoc * sizeof(void*)));
  CUDACHECK(cudaMalloc(&dTC, kELoc * sizeof(void*)));
  CUDACHECK(cudaMalloc(&dTIdx, kELoc * sizeof(void*)));
  CUDACHECK(cudaMalloc(&dTLdA, kELoc * sizeof(int64_t)));
  CUDACHECK(cudaMalloc(&dTLdB, kELoc * sizeof(int64_t)));
  CUDACHECK(cudaMalloc(&dTLdC, kELoc * sizeof(int64_t)));
  CUDACHECK(cudaMalloc(&dTTileGroupOff, (kELoc + 1) * sizeof(int)));
  CUDACHECK(cudaMalloc(&dTTileGroup,
                       ((size_t)maxRows / 64 + 4 * kELoc) * sizeof(int)));

  CUDACHECK(cudaMalloc(&Ap, maxRows * kH * sizeof(__nv_bfloat16)));
  int *dRowIdx, *dSendTok, *dCtr;
  TileDesc* dTiles;
  uint32_t* dArrived;
  size_t *dSendOff, *dRecvOff, *dSendBytes;
  CUDACHECK(cudaMalloc(&dRowIdx, maxRows * sizeof(int)));
  CUDACHECK(cudaMalloc(&dSendTok, (size_t)NRANKS * CAP * sizeof(int)));
  CUDACHECK(
      cudaMalloc(&dTiles, (maxPlan.tiles.size() + 64) * sizeof(TileDesc)));
  CUDACHECK(cudaMalloc(&dCtr, sizeof(int)));
  CUDACHECK(cudaMalloc(&dArrived, (NRANKS + 1) * sizeof(uint32_t)));
  CUDACHECK(cudaMalloc(&dSendOff, NRANKS * sizeof(size_t)));
  CUDACHECK(cudaMalloc(&dRecvOff, NRANKS * sizeof(size_t)));
  CUDACHECK(cudaMalloc(&dSendBytes, NRANKS * sizeof(size_t)));

  {
    std::vector<__nv_bfloat16> hx(CAP * kH), hw((size_t)kELoc * kN * kH);
    for (size_t t = 0; t < CAP; t++)
      for (int i = 0; i < kH; i++) hx[t * kH + i] = xval(rank, (int)t, i);
    for (size_t i = 0; i < hw.size(); i++)
      hw[i] =
          __float2bfloat16((float)((i * 2654435761u) % 17) * 0.0625f - 0.5f);
    CUDACHECK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(W, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
  }

  const size_t smemBytes =
      (size_t)NSTAGE * (BM + BN) * LDA * sizeof(__nv_bfloat16);
  CUDACHECK(cudaFuncSetAttribute(
      fusedMoe, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemBytes));
  int nSM = 0, bpsm = 0;
  CUDACHECK(cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, 0));
  CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm, fusedMoe,
                                                          kThreads, smemBytes));
  if (bpsm < 1) bpsm = 1;
  // Every block co-resident, so the dispatcher in block 0 is guaranteed to be
  // scheduled. If it were not, the workers would spin on an arrival that no one
  // is ever going to publish.
  int const gridAll = bpsm * nSM;

  float const alpha = 1.0f, beta = 0.0f;

  // Descriptors for cublasGemmGroupedBatchedEx. Sizes and leading dimensions
  // are host-side; the operand pointer arrays are device-side. Refilled once
  // per sweep point, outside the timed region, the same way every other
  // routing-derived quantity is.
  cublasOperation_t gTransA[kELoc], gTransB[kELoc];
  int gM[kELoc], gN[kELoc], gK[kELoc], gLdA[kELoc], gLdB[kELoc], gLdC[kELoc],
      gGroupSize[kELoc];
  float gAlpha[kELoc], gBeta[kELoc];
  int gGroups = 0;
  __nv_bfloat16 const **dGA = nullptr, **dGB = nullptr;
  __nv_bfloat16** dGC = nullptr;
  CUDACHECK(cudaMalloc(&dGA, kELoc * sizeof(void*)));
  CUDACHECK(cudaMalloc(&dGB, kELoc * sizeof(void*)));
  CUDACHECK(cudaMalloc(&dGC, kELoc * sizeof(void*)));

  // The same, sliced per arrival group, for the pipelined row.
  cublasOperation_t pTransA[NRANKS][kELoc], pTransB[NRANKS][kELoc];
  int pM[NRANKS][kELoc], pN[NRANKS][kELoc], pK[NRANKS][kELoc];
  int pLdA[NRANKS][kELoc], pLdB[NRANKS][kELoc], pLdC[NRANKS][kELoc],
      pGroupSize[NRANKS][kELoc];
  float pAlpha[NRANKS][kELoc], pBeta[NRANKS][kELoc];
  int pGroups[NRANKS] = {0, 0, 0, 0};
  __nv_bfloat16 const*(*dPA[NRANKS]), *(*dPB[NRANKS]);
  __nv_bfloat16** dPC[NRANKS];
  for (int g = 0; g < NRANKS; g++) {
    CUDACHECK(cudaMalloc(&dPA[g], kELoc * sizeof(void*)));
    CUDACHECK(cudaMalloc(&dPB[g], kELoc * sizeof(void*)));
    CUDACHECK(cudaMalloc(&dPC[g], kELoc * sizeof(void*)));
  }

  if (rank == 0) {
    std::printf("# Fused MoE dispatch + GEMM, %d x L4, PCIe only, no NVLink\n",
                NRANKS);
    std::printf(
        "# DeepSeek-V2-Lite at EP=4: H=%d N=%d, %d experts, %d local, top-%d, "
        "bf16\n",
        kH, kN, kE, kELoc, kTopK);
    std::printf(
        "# GEMM tile %dx%dx%d, %d stages, %d threads, %d blocks (%d/SM x %d "
        "SM)\n",
        BM, BN, BK, NSTAGE, kThreads, gridAll, bpsm, nSM);
    std::printf("# time is the worst rank; dispatch+GEMM only, no combine\n");
    std::printf(
        "# %d rounds x %d iters, variants interleaved, min per variant, after "
        "%.0f ms settling:\n",
        cfg.rounds, cfg.iters, cfg.settleMs);
    std::printf(
        "# this L4 power-throttles from 2040 to 885 MHz under sustained load "
        "(72 W cap), so\n");
    std::printf(
        "# timing the rows one after another does not give comparable "
        "numbers\n#\n");
    std::printf("# %7s %8s %8s %18s %11s %10s %10s %9s %9s\n", "T/rank", "rows",
                "sendMiB", "variant", "time(us)", "TFLOP/s", "GB/s", "badrow",
                "relerr");
  }

  Plan p;
  for (int T = cfg.minT; T <= cfg.maxT; T *= cfg.factorT) {
    buildPlan(p, rank, T, (int)CAP);
    int const nTiles = (int)p.tiles.size();
    double const flops = 2.0 * (double)p.totalRows * kH * kN;

    size_t hSendOff[NRANKS], hRecvOff[NRANKS], hSendBytes[NRANKS];
    size_t egress = 0;
    for (int d = 0; d < NRANKS; d++) {
      hSendOff[d] = sendBase + (size_t)d * slotBytes;
      hRecvOff[d] = recvBase + (size_t)d * slotBytes;
      hSendBytes[d] = (size_t)p.nSend[d] * kH * sizeof(__nv_bfloat16);
      if (d != rank) egress += hSendBytes[d];
    }
    CUDACHECK(cudaMemcpy(dSendOff, hSendOff, sizeof(hSendOff),
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dRecvOff, hRecvOff, sizeof(hRecvOff),
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dSendBytes, hSendBytes, sizeof(hSendBytes),
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dRowIdx, p.rowIdx.data(),
                         p.rowIdx.size() * sizeof(int),
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dTiles, p.tiles.data(),
                         p.tiles.size() * sizeof(TileDesc),
                         cudaMemcpyHostToDevice));
    for (int d = 0; d < NRANKS; d++)
      if (p.nSend[d])
        CUDACHECK(cudaMemcpy(dSendTok + (size_t)d * CAP, p.sendTok[d].data(),
                             p.nSend[d] * sizeof(int), cudaMemcpyHostToDevice));

    {
      __nv_bfloat16 const *hA[kELoc], *hB[kELoc];
      __nv_bfloat16* hC[kELoc];
      gGroups = 0;
      for (int e = 0; e < kELoc; e++) {
        int const m =
            p.grpCnt[e][0] + p.grpCnt[e][1] + p.grpCnt[e][2] + p.grpCnt[e][3];
        if (!m) continue;
        int const off = p.expOff[e];
        int const i = gGroups++;
        gTransA[i] = CUBLAS_OP_T;
        gTransB[i] = CUBLAS_OP_N;
        gM[i] = kN;
        gN[i] = m;
        gK[i] = kH;
        gLdA[i] = kH;
        gLdB[i] = kH;
        gLdC[i] = kN;
        gGroupSize[i] = 1;
        gAlpha[i] = 1.0f;
        gBeta[i] = 0.0f;
        hA[i] = W + (size_t)e * kN * kH;
        hB[i] = Ap + (size_t)off * kH;
        hC[i] = Y + (size_t)off * kN;
      }
      if (gGroups) {
        CUDACHECK(cudaMemcpy(dGA, hA, gGroups * sizeof(void*),
                             cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(dGB, hB, gGroups * sizeof(void*),
                             cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(dGC, hC, gGroups * sizeof(void*),
                             cudaMemcpyHostToDevice));
      }
      // The same descriptors sliced by arrival group, so the GEMM for group g
      // can be issued the moment group g lands instead of waiting for all four.
      // Rows are expert-major/group-minor, so a (group, expert) span is
      // contiguous and needs no extra permute.
      for (int g = 0; g < NRANKS; g++) {
        pGroups[g] = 0;
        for (int e = 0; e < kELoc; e++) {
          int const m = p.grpCnt[e][g];
          if (!m) continue;
          int const off = p.expOff[e] + p.grpOff[e][g];
          int const i = pGroups[g]++;
          pM[g][i] = kN;
          pN[g][i] = m;
          pK[g][i] = kH;
          pLdA[g][i] = kH;
          pLdB[g][i] = kH;
          pLdC[g][i] = kN;
          pTransA[g][i] = CUBLAS_OP_T;
          pTransB[g][i] = CUBLAS_OP_N;
          pGroupSize[g][i] = 1;
          pAlpha[g][i] = 1.0f;
          pBeta[g][i] = 0.0f;
          hA[i] = W + (size_t)e * kN * kH;
          hB[i] = Ap + (size_t)off * kH;
          hC[i] = Y + (size_t)off * kN;
        }
        if (pGroups[g]) {
          CUDACHECK(cudaMemcpy(dPA[g], hA, pGroups[g] * sizeof(void*),
                               cudaMemcpyHostToDevice));
          CUDACHECK(cudaMemcpy(dPB[g], hB, pGroups[g] * sizeof(void*),
                               cudaMemcpyHostToDevice));
          CUDACHECK(cudaMemcpy(dPC[g], hC, pGroups[g] * sizeof(void*),
                               cudaMemcpyHostToDevice));
        }
      }
    }

    // ---- CUTLASS grouped GEMM, gated
    // ----------------------------------------- One problem per (gate group,
    // expert), ordered group-major. The four arrival groups are merged into
    // two: rows within an expert are laid out group-minor, so gate group gg
    // covers arrival groups 2gg and 2gg+1 and is still one contiguous span.
    // Merging is not a simplification, it is the result -- at four groups the
    // group-major order costs 2.6x in lost weight reuse, at two it costs 4%
    // (benchmarks/data/cutlass_gemm_probe.txt).
    //
    // Gate group gg is ready once arrival group 2gg+1 has landed *and* its rows
    // have been gathered into Ap, which is what the permute chain publishes.
    // Arrival groups [gateLo[gg], gateLo[gg+1]) form gate group gg, and that
    // gate waits on the last of them.
    int const gateLo[kGateGroups + 1] = {0, cfg.gateSplit, NRANKS};
    int kGateOf[kGateGroups];
    for (int gg = 0; gg < kGateGroups; gg++) kGateOf[gg] = gateLo[gg + 1] - 1;
    std::vector<cutlass::gemm::GemmCoord> ctProb(kGateGroups * kELoc);
    std::vector<int> ctGroupOf(kGateGroups * kELoc);
    std::vector<CtElem*> ctA(kGateGroups * kELoc), ctB(kGateGroups * kELoc),
        ctC(kGateGroups * kELoc);
    std::vector<int64_t> ctLdA(kGateGroups * kELoc), ctLdB(kGateGroups * kELoc),
        ctLdC(kGateGroups * kELoc);
    int ctN = 0;
    for (int gg = 0; gg < kGateGroups; gg++)
      for (int e = 0; e < kELoc; e++) {
        int const lo = p.grpOff[e][gateLo[gg]];
        int mr = 0;
        for (int g = gateLo[gg]; g < gateLo[gg + 1]; g++) mr += p.grpCnt[e][g];
        if (mr <= 0) continue;
        int const off = p.expOff[e] + lo;
        ctProb[ctN] = cutlass::gemm::GemmCoord(mr, kN, kH);
        ctGroupOf[ctN] = gg;
        ctA[ctN] = reinterpret_cast<CtElem*>(Ap) + (size_t)off * kH;
        ctB[ctN] = reinterpret_cast<CtElem*>(W) + (size_t)e * kN * kH;
        ctC[ctN] = reinterpret_cast<CtElem*>(Y) + (size_t)off * kN;
        ctLdA[ctN] = kH;
        ctLdB[ctN] = kH;
        ctLdC[ctN] = kN;
        ctN++;
      }
    CUDACHECK(cudaMemcpy(dCtProb, ctProb.data(), ctN * sizeof(ctProb[0]),
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dCtGroupOf, ctGroupOf.data(), ctN * sizeof(int),
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dCtA, ctA.data(), ctN * sizeof(void*),
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dCtB, ctB.data(), ctN * sizeof(void*),
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dCtC, ctC.data(), ctN * sizeof(void*),
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dCtLdA, ctLdA.data(), ctN * sizeof(int64_t),
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dCtLdB, ctLdB.data(), ctN * sizeof(int64_t),
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dCtLdC, ctLdC.data(), ctN * sizeof(int64_t),
                         cudaMemcpyHostToDevice));

    typename CtGemm::Arguments ctArgs(
        dCtProb, ctN,
        std::min(CtGemm::sufficient(ctProb.data(), ctN), nSM - cfg.reserveSM),
        typename CtGemm::EpilogueOutputOp::Params(1.0f, 0.0f), dCtA, dCtB, dCtC,
        dCtC, dCtLdA, dCtLdB, dCtLdC, dCtLdC, ctProb.data());
    CtGemm ctOp;
    CUTLASSCHECK(CtGemm::can_implement(ctArgs));
    CUTLASSCHECK(ctOp.initialize(ctArgs, nullptr, sMain));
    int const ctGrid =
        std::min(CtGemm::sufficient(ctProb.data(), ctN), nSM - cfg.reserveSM);
    typename CtGated::Params ctGatedParams(
        typename CtKernel::Params(ctArgs, nullptr, 0), dGateFlags, dCtGroupOf);

    // Gather build: same problems, but A is the receive buffer itself and each
    // problem carries its own slice of the row list. With no permute in the way
    // the gate can spin on the dispatcher's own arrival flags directly -- no
    // second set of flags, no third stream, nothing to publish.
    {
      std::vector<CtElem*> ga(ctN);
      std::vector<int const*> gi(ctN);
      std::vector<int> av(ctN);
      int idx = 0;
      for (int gg = 0; gg < kGateGroups; gg++)
        for (int e = 0; e < kELoc; e++) {
          int mr = 0;
          for (int g = gateLo[gg]; g < gateLo[gg + 1]; g++)
            mr += p.grpCnt[e][g];
          if (mr <= 0) continue;
          int const o = p.expOff[e] + p.grpOff[e][gateLo[gg]];
          ga[idx] = reinterpret_cast<CtElem*>(buf + recvBase);  // tensor base
          gi[idx] = dRowIdx + o;
          av[idx] = kGateOf[gg];  // the last arrival group this problem needs
          idx++;
        }
      CUDACHECK(cudaMemcpy(dCtGatherA, ga.data(), ctN * sizeof(void*),
                           cudaMemcpyHostToDevice));
      CUDACHECK(cudaMemcpy(dCtGatherIdx, gi.data(), ctN * sizeof(void*),
                           cudaMemcpyHostToDevice));
      CUDACHECK(cudaMemcpy(dCtArrivalOf, av.data(), ctN * sizeof(int),
                           cudaMemcpyHostToDevice));
    }
    typename CtGatherKernel::Arguments ctGArgs(
        dCtProb, ctN, ctGrid,
        typename CtGatherKernel::EpilogueOutputOp::Params(1.0f, 0.0f),
        dCtGatherA, dCtB, dCtC, dCtC, dCtLdA, dCtLdB, dCtLdC, dCtLdC,
        ctProb.data());
    // ---- unsplit list: one problem per expert, gated per tile ---------------
    // No split, so every problem is M_e rows tall and the visitor walks experts
    // in order -- full tiles and intact weight reuse. What a tile waits for is
    // derived from which m-tile it is: rows are ordered by source within an
    // expert, so the arrival group of a tile's *last* row is the latest arrival
    // it depends on. Tiles that fall entirely inside this rank's own rows wait
    // for nothing.
    constexpr int kTBM = CtGatherKernel::Mma::Shape::kM;
    std::vector<cutlass::gemm::GemmCoord> tProb;
    std::vector<CtElem*> tA, tB, tC;
    std::vector<int const*> tIdx;
    std::vector<int64_t> tLdA, tLdB, tLdC;
    std::vector<int> tTileGroup, tTileGroupOff;
    for (int e = 0; e < kELoc; e++) {
      int mE = 0;
      for (int g = 0; g < NRANKS; g++) mE += p.grpCnt[e][g];
      if (mE <= 0) continue;
      int const off = p.expOff[e];
      tTileGroupOff.push_back((int)tTileGroup.size());
      tProb.push_back(cutlass::gemm::GemmCoord(mE, kN, kH));
      tA.push_back(reinterpret_cast<CtElem*>(
          buf + recvBase));  // tensor base; rows via tIdx
      tIdx.push_back(dRowIdx + off);
      tB.push_back(reinterpret_cast<CtElem*>(W) + (size_t)e * kN * kH);
      tC.push_back(reinterpret_cast<CtElem*>(Y) + (size_t)off * kN);
      tLdA.push_back(kH);
      tLdB.push_back(kH);
      tLdC.push_back(kN);
      for (int mt = 0; mt * kTBM < mE; mt++) {
        int const last =
            std::min((mt + 1) * kTBM, mE) - 1;  // latest row this tile reads
        int acc = 0, g = 0;
        for (int gg = 0; gg < NRANKS; gg++) {
          acc += p.grpCnt[e][gg];
          if (last < acc) {
            g = gg;
            break;
          }
        }
        tTileGroup.push_back(g);
      }
    }
    int const tN = (int)tProb.size();
    CUDACHECK(cudaMemcpy(dTProb, tProb.data(), tN * sizeof(tProb[0]),
                         cudaMemcpyHostToDevice));
    CUDACHECK(
        cudaMemcpy(dTA, tA.data(), tN * sizeof(void*), cudaMemcpyHostToDevice));
    CUDACHECK(
        cudaMemcpy(dTB, tB.data(), tN * sizeof(void*), cudaMemcpyHostToDevice));
    CUDACHECK(
        cudaMemcpy(dTC, tC.data(), tN * sizeof(void*), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dTIdx, tIdx.data(), tN * sizeof(void*),
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dTLdA, tLdA.data(), tN * sizeof(int64_t),
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dTLdB, tLdB.data(), tN * sizeof(int64_t),
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dTLdC, tLdC.data(), tN * sizeof(int64_t),
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dTTileGroupOff, tTileGroupOff.data(), tN * sizeof(int),
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dTTileGroup, tTileGroup.data(),
                         tTileGroup.size() * sizeof(int),
                         cudaMemcpyHostToDevice));

    int const tGrid =
        std::min(CtGemm::sufficient(tProb.data(), tN), nSM - cfg.reserveSM);
    typename CtGatherKernel::Arguments tArgs(
        dTProb, tN, tGrid,
        typename CtGatherKernel::EpilogueOutputOp::Params(1.0f, 0.0f), dTA, dTB,
        dTC, dTC, dTLdA, dTLdB, dTLdC, dTLdC, tProb.data());

    MoeDispatch ctDisp{hPChans, rank, dSendOff, dRecvOff, dSendBytes, dArrived};
    typename CtGatedGather::Params ctGatherParams(
        typename CtGatherKernel::Params(ctGArgs, nullptr, 0), dArrived,
        dCtArrivalOf, dCtGatherIdx, ctDisp);

    // Same problems and pointers as the fused row, but arrived = nullptr, which
    // compiles the gate out entirely.
    typename CtUngatedGather::Params ungatedParams(
        typename CtGatherKernel::Params(ctGArgs, nullptr, 0), nullptr, nullptr,
        dCtGatherIdx);

    typename CtGatedGather::Params tParams(
        typename CtGatherKernel::Params(tArgs, nullptr, 0), dArrived, nullptr,
        dTIdx, ctDisp);
    tParams.tile_group = dTTileGroup;
    tParams.tile_group_off = dTTileGroupOff;

    // Gather gate group gg's rows into Ap. Per (gg, expert), because rows are
    // expert-major and only contiguous within one expert.
    auto permuteGate = [&](int gg, cudaStream_t st) {
      for (int e = 0; e < kELoc; e++) {
        int mr = 0;
        for (int g = gateLo[gg]; g < gateLo[gg + 1]; g++) mr += p.grpCnt[e][g];
        if (!mr) continue;
        const int off = p.expOff[e] + p.grpOff[e][gateLo[gg]];
        permuteRows<<<mr, 256, 0, st>>>(
            Ap + (size_t)off * kH,
            reinterpret_cast<__nv_bfloat16*>(buf + recvBase), dRowIdx + off, mr,
            kH);
      }
    };

    // Stage: gather the rows owed to each peer into the send slots, and this
    // rank's own rows straight into its receive slot so the GEMM sees one
    // uniform [4][CAP][H] view.
    auto stageAll = [&](cudaStream_t s) {
      for (int d = 0; d < NRANKS; d++) {
        if (!p.nSend[d]) continue;
        __nv_bfloat16* dst =
            (d == rank) ? reinterpret_cast<__nv_bfloat16*>(buf + hRecvOff[rank])
                        : reinterpret_cast<__nv_bfloat16*>(buf + hSendOff[d]);
        stageRows<<<p.nSend[d], 256, 0, s>>>(dst, x, dSendTok + (size_t)d * CAP,
                                             p.nSend[d], kH);
      }
    };

    auto ncclDispatchAll = [&](cudaStream_t s) {
      NCCLCHECK(ncclGroupStart());
      for (int i = 0; i < NPEERS; i++) {
        const int d = ascPeer(rank, i);
        if (p.nSend[d])
          NCCLCHECK(ncclSend(buf + hSendOff[d], (size_t)p.nSend[d] * kH,
                             ncclBfloat16, d, ncclComm, s));
        if (p.nRecv[d])
          NCCLCHECK(ncclRecv(buf + hRecvOff[d], (size_t)p.nRecv[d] * kH,
                             ncclBfloat16, d, ncclComm, s));
      }
      NCCLCHECK(ncclGroupEnd());
    };
    // One group call per rotation step, so each source's arrival is a separate
    // event the GEMM stream can wait on.
    auto ncclDispatchStep = [&](int k, cudaStream_t s) {
      const int d = destOf(rank, k), q = srcOf(rank, k);
      NCCLCHECK(ncclGroupStart());
      if (p.nSend[d])
        NCCLCHECK(ncclSend(buf + hSendOff[d], (size_t)p.nSend[d] * kH,
                           ncclBfloat16, d, ncclComm, s));
      if (p.nRecv[q])
        NCCLCHECK(ncclRecv(buf + hRecvOff[q], (size_t)p.nRecv[q] * kH,
                           ncclBfloat16, q, ncclComm, s));
      NCCLCHECK(ncclGroupEnd());
    };

    // Permute + cuBLAS for a contiguous span of arrival groups.
    auto cublasGroups = [&](int gLo, int gHi, cudaStream_t s) {
      for (int e = 0; e < kELoc; e++)
        for (int g = gLo; g < gHi; g++) {
          const int m = p.grpCnt[e][g];
          if (!m) continue;
          const int off = p.expOff[e] + p.grpOff[e][g];
          permuteRows<<<m, 256, 0, s>>>(
              Ap + (size_t)off * kH,
              reinterpret_cast<__nv_bfloat16*>(buf + recvBase), dRowIdx + off,
              m, kH);
          CUBLASCHECK(cublasGemmEx(
              cub, CUBLAS_OP_T, CUBLAS_OP_N, kN, m, kH, &alpha,
              W + (size_t)e * kN * kH, CUDA_R_16BF, kH, Ap + (size_t)off * kH,
              CUDA_R_16BF, kH, &beta, Y + (size_t)off * kN, CUDA_R_16BF, kN,
              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    };

    // All 16 experts as one grouped GEMM. Sizes and leading dimensions are
    // host arrays; the A/B/C pointer arrays live on the device, as with
    // cublasGemmBatchedEx. group_size is 1 per group because every expert has
    // its own M_e.
    auto cublasGrouped = [&](cudaStream_t s) {
      for (int e = 0; e < kELoc; e++) {
        const int m =
            p.grpCnt[e][0] + p.grpCnt[e][1] + p.grpCnt[e][2] + p.grpCnt[e][3];
        if (!m) continue;
        const int off = p.expOff[e];
        permuteRows<<<m, 256, 0, s>>>(
            Ap + (size_t)off * kH,
            reinterpret_cast<__nv_bfloat16*>(buf + recvBase), dRowIdx + off, m,
            kH);
      }
      CUBLASCHECK(cublasGemmGroupedBatchedEx(
          cub, gTransA, gTransB, gM, gN, gK, gAlpha, (const void* const*)dGA,
          CUDA_R_16BF, gLdA, (const void* const*)dGB, CUDA_R_16BF, gLdB, gBeta,
          (void* const*)dGC, CUDA_R_16BF, gLdC, gGroups, gGroupSize,
          CUBLAS_COMPUTE_32F));
    };

    // Permute + grouped GEMM for one arrival group.
    auto cublasGroupedOne = [&](int g, cudaStream_t s) {
      if (!pGroups[g]) return;
      for (int e = 0; e < kELoc; e++) {
        const int m = p.grpCnt[e][g];
        if (!m) continue;
        const int off = p.expOff[e] + p.grpOff[e][g];
        permuteRows<<<m, 256, 0, s>>>(
            Ap + (size_t)off * kH,
            reinterpret_cast<__nv_bfloat16*>(buf + recvBase), dRowIdx + off, m,
            kH);
      }
      CUBLASCHECK(cublasGemmGroupedBatchedEx(
          cub, pTransA[g], pTransB[g], pM[g], pN[g], pK[g], pAlpha[g],
          (const void* const*)dPA[g], CUDA_R_16BF, pLdA[g],
          (const void* const*)dPB[g], CUDA_R_16BF, pLdB[g], pBeta[g],
          (void* const*)dPC[g], CUDA_R_16BF, pLdC[g], pGroups[g], pGroupSize[g],
          CUBLAS_COMPUTE_32F));
    };

    // A dispatch-only launch (tiles == 0) needs exactly one block: block 0 is
    // the dispatcher and every other block exits immediately. Launching the
    // full persistent grid for it is not merely wasteful -- all gridAll blocks
    // reserve 72 KB of smem apiece, so they occupy every SM, and a concurrent
    // cublasGemmGroupedBatchedEx on the other stream then never makes progress.
    // The looped cublasGemmEx path tolerates it; the grouped one deadlocks.
    // `clear` exists because the flags cannot always be cleared on the same
    // stream the dispatcher runs on. When a gated GEMM reads them concurrently
    // from another stream, clearing on the dispatcher's stream races: the GEMM
    // can see last iteration's flags still set and start on rows that have not
    // arrived. Those callers clear on the GEMM's stream before forking.
    auto launchFused = [&](int mode, int tiles, cudaStream_t s,
                           bool clear = true) {
      const int grid = tiles ? gridAll : 1;
      if (clear) clearFlags<<<1, 32, 0, s>>>(dArrived);
      CUDACHECK(cudaMemsetAsync(dCtr, 0, sizeof(int), s));
      fusedMoe<<<grid, kThreads, smemBytes, s>>>(
          hPChans, hChans, rank, mode, dSendOff, dRecvOff, dSendBytes, dArrived,
          dTiles, tiles, dCtr, reinterpret_cast<__nv_bfloat16*>(buf + recvBase),
          dRowIdx, W, Y, kH, kN);
    };

    // One full dispatch+GEMM step. cublasSetStream is per handle, so the
    // pipelined rows retarget it rather than keeping two handles.
    auto runVariant = [&](int v) {
      switch (v) {
        case V_GEMM_ONLY:
          cublasGroups(0, NRANKS, sMain);
          break;
        case V_GEMM_ONLY_GRP:
#ifdef BISECT_SLICED
          // Same total work, issued as four grouped calls sliced by arrival
          // group instead of one. Prices the slicing on its own, with no
          // communication anywhere near it.
          for (int g = 0; g < NRANKS; g++) cublasGroupedOne(g, sMain);
#else
          cublasGrouped(sMain);
#endif
          break;
        case V_MMA_ONLY:
          launchFused(0, nTiles, sMain);
          break;
        case V_DISP_ONLY:
          stageAll(sMain);
          launchFused(1, 0, sMain);
          break;
        // The collective on its own. This exists so "ours is N x NCCL" is a
        // measured ratio rather than the difference of two noisy rows: taking
        // nccl+cublas-grp minus gemm-only-grp moved the answer between 1.31x
        // and 1.37x across two runs purely on GEMM noise.
        case V_NCCL_DISP_ONLY:
          stageAll(sMain);
          ncclDispatchAll(sMain);
          break;
        case V_NCCL_CUBLAS:
          stageAll(sMain);
          ncclDispatchAll(sMain);
          cublasGroups(0, NRANKS, sMain);
          break;
        case V_NCCL_CUBLAS_GRP:
          stageAll(sMain);
          ncclDispatchAll(sMain);
          cublasGrouped(sMain);
          break;
        case V_NCCL_CUBLAS_PIPE: {
          stageAll(sMain);
          CUDACHECK(cudaEventRecord(evStep[0], sMain));
          CUDACHECK(cudaStreamWaitEvent(sComm, evStep[0], 0));
          for (int k = 1; k <= NPEERS; k++) {
            ncclDispatchStep(k, sComm);
            CUDACHECK(cudaEventRecord(evStep[k], sComm));
          }
          cublasGroups(0, 1, sMain);  // our own rows need no comm
          for (int k = 1; k <= NPEERS; k++) {
            CUDACHECK(cudaStreamWaitEvent(sMain, evStep[k], 0));
            cublasGroups(k, k + 1, sMain);
          }
          break;
        }
        case V_OURS_CUBLAS:
          stageAll(sMain);
          launchFused(1, 0, sMain);
          cublasGroups(0, NRANKS, sMain);
          break;
        case V_OURS_CUBLAS_GRP:
          stageAll(sMain);
          launchFused(1, 0, sMain);
          cublasGrouped(sMain);
          break;
        case V_NCCL_CUBLAS_GRP_PIPE: {
          // How much overlap alone is worth, with no custom GEMM: one grouped
          // cuBLAS call per arrival group, each gated on that group landing.
          // Group 0 is this rank's own rows and needs no communication, so the
          // compute starts immediately and each later arrival lands while the
          // previous group is still running.
          //
          // The dispatch here is NCCL rather than ours, and that is forced
          // rather than chosen. Our dispatcher is a persistent kernel that
          // spins on peer arrivals, and running it concurrently with
          // cublasGemmGroupedBatchedEx deadlocks -- reproducibly, at every
          // size, with a single grouped call and with no spin kernel on the
          // compute stream at all. The looped cublasGemmEx path tolerates it
          // (ours+cublas-pipe runs fine); the grouped one does not. NCCL's
          // per-step group calls are ordinary stream work, so this row
          // expresses the same schedule with events and nothing spinning.
          stageAll(sMain);
          CUDACHECK(cudaEventRecord(evStep[0], sMain));
          CUDACHECK(cudaStreamWaitEvent(sComm, evStep[0], 0));
          for (int k = 1; k <= NPEERS; k++) {
            ncclDispatchStep(k, sComm);
            CUDACHECK(cudaEventRecord(evStep[k], sComm));
          }
          cublasGroupedOne(0, sMain);
          for (int k = 1; k <= NPEERS; k++) {
            CUDACHECK(cudaStreamWaitEvent(sMain, evStep[k], 0));
            cublasGroupedOne(k, sMain);
          }
          break;
        }
        case V_OURS_CUBLAS_PIPE: {
          stageAll(sMain);
          CUDACHECK(cudaEventRecord(evStep[0], sMain));
          CUDACHECK(cudaStreamWaitEvent(sComm, evStep[0], 0));
          launchFused(1, 0, sComm);
          cublasGroups(0, 1, sMain);
          for (int k = 1; k <= NPEERS; k++) {
            // A spin kernel is how a stream-ordered cuBLAS call expresses
            // "wait for this peer's rows" against a flag the dispatcher sets.
            waitArrival<<<1, 32, 0, sMain>>>(dArrived, k);
            cublasGroups(k, k + 1, sMain);
          }
          break;
        }
        // Our dispatch, then gather, then CUTLASS -- all sequential. This is
        // the reference point that says how much of any fused win is the GEMM
        // being better rather than the overlap working. Our collective, then
        // the fused kernel's own GEMM, strictly sequential: same gather-A
        // CUTLASS, same tile, no gate, no overlap. The rung between +a2a and
        // +fused.
        case V_OURS_CUTLASS:
          stageAll(sMain);
          launchFused(1, 0, sMain);  // dispatcher only, and it completes first
          uccl_lite::gatedGroupedKernel<CtUngatedGather>
              <<<ctGrid, CtUngatedGather::kThreadCount, ctGatherSmem, sMain>>>(
                  ungatedParams);
          break;

        // The fused row. Three streams, deliberately:
        //
        //   sComm  the dispatcher, one block, spinning on its three arrivals.
        //   sPerm  per gate group: wait for the group, gather it, publish it.
        //   sMain  the gated CUTLASS GEMM, whose tiles spin on those flags.
        //
        // This cannot deadlock the way our dispatcher does against grouped
        // cuBLAS, and the reason is co-residency rather than luck. The GEMM
        // launches nSM-1 blocks and each needs ~96 KB of the SM's 100 KB, so it
        // occupies 57 of 58 SMs and leaves one for the dispatcher. The permute
        // and wait kernels use no dynamic shared memory, so they schedule
        // alongside resident GEMM blocks rather than queueing behind them, and
        // they always terminate. No participant waits on something that is
        // waiting on it.
        case V_FUSED_CUTLASS: {
          stageAll(sMain);
          clearGateFlags<<<1, 32, 0, sMain>>>(dGateFlags, kGateGroups);
          CUDACHECK(cudaEventRecord(evStep[0], sMain));
          CUDACHECK(cudaStreamWaitEvent(sComm, evStep[0], 0));
          CUDACHECK(cudaStreamWaitEvent(sPerm, evStep[0], 0));
          clearFlags<<<1, 32, 0, sMain>>>(
              dArrived);  // ahead of the fork, not on sComm
          CUDACHECK(cudaEventRecord(evStep[3], sMain));
          CUDACHECK(cudaStreamWaitEvent(sComm, evStep[3], 0));
          CUDACHECK(cudaStreamWaitEvent(sPerm, evStep[3], 0));
          launchFused(1, 0, sComm,
                      /*clear=*/false);  // dispatcher only: tiles = 0
          for (int gg = 0; gg < kGateGroups; gg++) {
            waitArrival<<<1, 32, 0, sPerm>>>(dArrived, kGateOf[gg]);
            permuteGate(gg, sPerm);
            publishFlag<<<1, 32, 0, sPerm>>>(dGateFlags, gg);
          }
          uccl_lite::gatedGroupedKernel<CtGated>
              <<<ctGrid, CtGated::kThreadCount, ctSmem, sMain>>>(ctGatedParams);
          CUDACHECK(cudaEventRecord(evStep[1], sPerm));
          CUDACHECK(cudaStreamWaitEvent(sMain, evStep[1], 0));
          CUDACHECK(cudaEventRecord(evStep[2], sComm));
          CUDACHECK(cudaStreamWaitEvent(sMain, evStep[2], 0));
          break;
        }
        // The fused row that should work. Two streams and two kernels: the
        // dispatcher, and the gather-A gated GEMM. No permute, so nothing has
        // to be co-scheduled against a GPU full of spinning GEMM blocks -- the
        // dispatcher needs one free SM and that is the whole requirement.
        // Correct and, when it completes, the fastest row in the ladder: 3690
        // us against the 4341 us baseline at T=2048, or 1.18x, where the best
        // unfused path is 1.15x. It is also intermittent -- the same command
        // completed twice and then deadlocked twice -- so it is excluded from
        // the default sweep and has to be asked for. The remaining suspect is
        // the dispatcher failing to get an SM against a grid of spinning GEMM
        // blocks, but 4 reserved SMs should be enough for a 1-block dispatcher
        // and raising the reserve to 8 did not help, so that is not
        // established. One kernel, one stream. Block 0 dispatches,
        // blocks 1..N-1 run gated GEMM tiles that read the receive buffer
        // through gather-A. There is no second kernel to schedule and therefore
        // nothing that can be scheduled behind the grid waiting for it -- which
        // is what made every separate-kernel version of this deadlock.
        case V_FUSED_CUTLASS_G: {
          stageAll(sMain);
          clearFlags<<<1, 32, 0, sMain>>>(dArrived);
          uccl_lite::gatedGroupedKernel<CtGatedGather>
              <<<ctGrid + 1, CtGatedGather::kThreadCount, ctGatherSmem,
                 sMain>>>(ctGatherParams);
          break;
        }
        // Same kernel, gated per tile instead of per problem. One problem per
        // expert, so nothing is split: full tiles, expert-major order, intact
        // weight reuse -- and finer gating, since each m-tile waits only for
        // the group its own last row came from.
        case V_FUSED_CUTLASS_T: {
          stageAll(sMain);
          clearFlags<<<1, 32, 0, sMain>>>(dArrived);
          uccl_lite::gatedGroupedKernel<CtGatedGather>
              <<<tGrid + 1, CtGatedGather::kThreadCount, ctGatherSmem, sMain>>>(
                  tParams);
          break;
        }
        case V_FUSED_CE:
          stageAll(sMain);
          launchFused(1, nTiles, sMain);
          break;
        case V_FUSED_SM:
          stageAll(sMain);
          launchFused(2, nTiles, sMain);
          break;
        default:
          break;
      }
    };

    // Correctness first, for every variant, then timing for every variant. The
    // two passes are separate because the timing pass has to interleave the
    // variants and a check in the middle of that would perturb it.
    std::vector<long long> badRowOf(NVAR, -1);
    std::vector<double> relErrOf(NVAR, -1.0);
    for (int v = 0; v < NVAR; v++) {
      if (cfg.only >= 0 && v != cfg.only) continue;
      if (cfg.skip[v]) continue;

      long long badRow = -1;
      double relErr = -1.0;
      if (cfg.check && v != V_DISP_ONLY && v != V_NCCL_DISP_ONLY) {
        CUDACHECK(
            cudaMemsetAsync(buf + recvBase, 0, NRANKS * slotBytes, sMain));
        CUDACHECK(cudaMemsetAsync(Y, 0, maxRows * kN * 2, sMain));
        CUDACHECK(cudaStreamSynchronize(sMain));
        bootstrap->barrier();
        runVariant(v);
        CUDACHECK(cudaStreamSynchronize(sMain));
        CUDACHECK(cudaStreamSynchronize(sComm));
        bootstrap->barrier();

        // Dispatch, exactly. Every arrived row must be bit-identical to the
        // source rank's token, in the slot the plan says, from the sender the
        // plan says -- so a row in the wrong place is caught even though the
        // GEMM below can only be checked to a tolerance.
        if (v != V_GEMM_ONLY && v != V_GEMM_ONLY_GRP && v != V_MMA_ONLY) {
          std::vector<__nv_bfloat16> hr(NRANKS * CAP * kH);
          CUDACHECK(cudaMemcpy(hr.data(), buf + recvBase, hr.size() * 2,
                               cudaMemcpyDeviceToHost));
          badRow = 0;
          for (int s = 0; s < NRANKS; s++)
            for (int j = 0; j < p.nRecv[s]; j++) {
              int const t = p.recvTok[s][j];
              for (int i = 0; i < kH;
                   i += 97) {  // strided: full compare is 100x the runtime
                const __nv_bfloat16 got = hr[((size_t)s * CAP + j) * kH + i];
                if (__bfloat162float(got) != __bfloat162float(xval(s, t, i))) {
                  badRow++;
                  break;
                }
              }
            }
        }

        // GEMM, to a tolerance: bf16 with fp32 accumulate is not reproducible
        // across tile orders, so the reference is cuBLAS over the same rows.
        CUDACHECK(cudaMemsetAsync(Yref, 0, maxRows * kN * 2, sMain));
        for (int e = 0; e < kELoc; e++) {
          int const m =
              p.grpCnt[e][0] + p.grpCnt[e][1] + p.grpCnt[e][2] + p.grpCnt[e][3];
          if (!m) continue;
          int const off = p.expOff[e];
          permuteRows<<<m, 256, 0, sMain>>>(
              Ap + (size_t)off * kH,
              reinterpret_cast<__nv_bfloat16*>(buf + recvBase), dRowIdx + off,
              m, kH);
          CUBLASCHECK(cublasGemmEx(
              cub, CUBLAS_OP_T, CUBLAS_OP_N, kN, m, kH, &alpha,
              W + (size_t)e * kN * kH, CUDA_R_16BF, kH, Ap + (size_t)off * kH,
              CUDA_R_16BF, kH, &beta, Yref + (size_t)off * kN, CUDA_R_16BF, kN,
              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        CUDACHECK(cudaStreamSynchronize(sMain));
        std::vector<__nv_bfloat16> ha((size_t)p.totalRows * kN),
            hb((size_t)p.totalRows * kN);
        CUDACHECK(
            cudaMemcpy(ha.data(), Y, ha.size() * 2, cudaMemcpyDeviceToHost));
        CUDACHECK(
            cudaMemcpy(hb.data(), Yref, hb.size() * 2, cudaMemcpyDeviceToHost));
        relErr = 0.0;
        for (size_t i = 0; i < ha.size(); i++) {
          double const gv = __bfloat162float(ha[i]),
                       rv = __bfloat162float(hb[i]);
          double const d = std::abs(gv - rv) / (std::abs(rv) + 1e-3);
          if (d > relErr) relErr = d;
        }
        bootstrap->barrier();
      }
      badRowOf[v] = badRow;
      relErrOf[v] = relErr;
    }

    // Timing: settle, then interleave the variants round-robin and keep each
    // one's minimum.
    //
    // Timing each variant to completion in turn -- which is what this harness
    // used to do -- is not sound on this machine. An L4 is a 72 W card and
    // sustained bf16 work saturates that budget, so the SM clock swings between
    // 2040 and 885 MHz during a run. Whichever variant is measured first gets a
    // colder GPU, and the ladder's rows are then not comparable with each
    // other. benchmarks/probes/cutlass_gemm_probe.cu measures the effect
    // directly: the same cuBLAS call reads 2480 us alone and 2785 us after a
    // smaller size ran first in the same process. That is also the 8-12%
    // run-to-run sensitivity docs/fusion.md previously recorded as unexplained.
    //
    // Every rank runs the same schedule and the barrier keeps them in step, so
    // the interleaving is consistent across the node.
    std::vector<double> bestUs(NVAR, 1e30);
    {
      auto once = [&](int v) -> double {
        bootstrap->barrier();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < cfg.iters; i++) runVariant(v);
        CUDACHECK(cudaStreamSynchronize(sMain));
        CUDACHECK(cudaStreamSynchronize(sComm));
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count() /
               cfg.iters;
      };
      auto sweepOnce = [&]() {
        for (int v = 0; v < NVAR; v++) {
          if (cfg.only >= 0 && v != cfg.only) continue;
          if (cfg.skip[v]) continue;
          const double us = once(v);
          if (us < bestUs[v]) bestUs[v] = us;
        }
      };
      // Settle into the throttled steady state before anything counts. A cold
      // GPU and a settled one differ by ~15% here, and which of the two a
      // benchmark reports should not depend on how many sizes ran before it.
      //
      // The barrier inside the loop is not decoration. Without it the ranks are
      // free to drift, and for a cheap collective row -- disp-only at T=8 is
      // 42 us -- 2500 ms of settling is tens of thousands of un-barriered
      // dispatches, which deadlocks reproducibly. kSettleMaxPasses is the other
      // half of the same guard, bounding the work if a row is cheaper still.
      // 300 rather than "however many fit in settleMs". Beyond that the
      // ours+cublas-grp row starts hitting the deadlock docs/fusion.md
      // records between our spinning dispatcher and grouped cuBLAS: the more
      // dispatches a run issues, the likelier it is to bite. The full ladder
      // still settles for the whole 2500 ms, because a pass there is every
      // variant; only a single cheap row selected with -a settles less, and a
      // cheap row barely moves the power budget in the first place.
      constexpr int kSettleMaxPasses = 300;
      auto s0 = std::chrono::steady_clock::now();
      for (int pass = 0; pass < kSettleMaxPasses; pass++) {
        for (int i = 0; i < cfg.warmup; i++)
          for (int v = 0; v < NVAR; v++) {
            if (cfg.only >= 0 && v != cfg.only) continue;
            if (cfg.skip[v]) continue;
            runVariant(v);
          }
        CUDACHECK(cudaStreamSynchronize(sMain));
        CUDACHECK(cudaStreamSynchronize(sComm));
        bootstrap->barrier();
        if (std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - s0)
                .count() >= cfg.settleMs)
          break;
      }
      std::fill(bestUs.begin(), bestUs.end(), 1e30);
      bootstrap->barrier();
      for (int r = 0; r < cfg.rounds; r++) sweepOnce();
    }

    for (int v = 0; v < NVAR; v++) {
      if (cfg.only >= 0 && v != cfg.only) continue;
      if (cfg.skip[v]) continue;

      // The step is only as fast as its slowest participant.
      std::vector<double> all(NRANKS, 0.0);
      all[rank] = bestUs[v];
      bootstrap->allGather(all.data(), sizeof(double));
      double const worst = *std::max_element(all.begin(), all.end());
      std::vector<long long> bad(NRANKS, 0);
      bad[rank] = badRowOf[v];
      bootstrap->allGather(bad.data(), sizeof(long long));
      long long const worstBad = *std::max_element(bad.begin(), bad.end());
      std::vector<double> errs(NRANKS, 0.0);
      errs[rank] = relErrOf[v];
      bootstrap->allGather(errs.data(), sizeof(double));
      double const worstErr = *std::max_element(errs.begin(), errs.end());

      if (rank == 0) {
        std::printf("  %7d %8d %8.2f %18s %11.2f %10.2f %10.2f %9lld %9.2e\n",
                    T, p.totalRows, (double)egress / (1 << 20), kName[v], worst,
                    flops / (worst * 1e-6) / 1e12,
                    (double)egress * NRANKS / (worst * 1e-6) / 1e9, worstBad,
                    worstErr);
        std::fflush(stdout);
      }
      bootstrap->barrier();
    }
  }

  CUDACHECK(cudaStreamSynchronize(sMain));
  bootstrap->barrier();
  ncclCommDestroy(ncclComm);
  proxy->stopProxy();
  cublasDestroy(cub);
  CUDACHECK(cudaFree(buf));
}

int main(int argc, char** argv) {
  Cfg cfg;
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    auto next = [&]() -> char const* {
      return (i + 1 < argc) ? argv[++i] : "0";
    };
    if (k == "-b")
      cfg.minT = std::atoi(next());
    else if (k == "-e")
      cfg.maxT = std::atoi(next());
    else if (k == "-f")
      cfg.factorT = std::atoi(next());
    else if (k == "-n")
      cfg.iters = std::atoi(next());
    else if (k == "-w")
      cfg.warmup = std::atoi(next());
    else if (k == "-r")
      cfg.rounds = std::atoi(next());
    else if (k == "-S")
      cfg.settleMs = std::atof(next());
    else if (k == "-R")
      cfg.reserveSM = std::atoi(next());
    else if (k == "-P")
      cfg.gateSplit = std::atoi(next());
    else if (k == "-X") {
      const std::string want = next();
      for (int j = 0; j < NVAR; j++)
        if (want == kName[j]) cfg.skip[j] = true;
    } else if (k == "-B")
      cfg.smBlocks = std::atoi(next());
    else if (k == "-T")
      cfg.smThreads = std::atoi(next());
    else if (k == "-c")
      cfg.check = std::atoi(next()) != 0;
    else if (k == "-a") {
      const std::string want = next();
      cfg.only = -1;
      for (int j = 0; j < NVAR; j++)
        if (want == kName[j]) cfg.only = j;
      if (cfg.only < 0) {
        std::fprintf(stderr, "unknown variant %s\n", want.c_str());
        return 1;
      }
    } else {
      std::fprintf(stderr, "unknown arg %s\n", k.c_str());
      return 1;
    }
  }

  // Fork before touching CUDA so every rank gets a clean context; the unique id
  // is made in the parent and inherited, which removes the need for MPI.
  mscclpp::UniqueId id = mscclpp::TcpBootstrap::createUniqueId();
  std::vector<pid_t> kids;
  for (int r = 1; r < NRANKS; r++) {
    pid_t pid = fork();
    if (pid == 0) {
      runRank(r, id, cfg);
      _exit(0);
    }
    kids.push_back(pid);
  }
  runRank(0, id, cfg);
  int rc = 0;
  for (pid_t pid : kids) {
    int st = 0;
    waitpid(pid, &st, 0);
    if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) rc = 1;
  }
  return rc;
}
