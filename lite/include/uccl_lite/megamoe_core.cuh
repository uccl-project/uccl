#pragma once
// The reusable half of the fused MoE layer: shape constants, the ce-shift
// rotation, the CUTLASS grouped-GEMM configuration, the small kernels around
// the GEMMs, and the problem-list helper.
//
// Kernel behavior lives here; benchmarks/fusion/fused_moe_layer.cu owns the
// synthetic inputs, host routing plan, numerical references, and timing ladder.
// Application-specific bootstrap and allocation policy belong to the caller.
//
// Extracted from fused_moe_layer.cu verbatim -- no behaviour change. The gate
// on that claim is `-c 2` still passing and the sweep still reproducing the
// published 1.41-1.51x.

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
#include <mscclpp/port_channel.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <mscclpp/semaphore.hpp>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define NRANKS 4
#define NPEERS (NRANKS - 1)

#define CUDACHECK(cmd)                                                   \
  do {                                                                   \
    cudaError_t e_ = (cmd);                                              \
    if (e_ != cudaSuccess) {                                             \
      std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e_));                              \
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

using PortChan = mscclpp::PortChannelDeviceHandle;
struct PChans {
  PortChan c[NRANKS];
};

// The MoE shape, as build knobs so one file serves more than one model. Both
// shapes below have H=2048, which is why they share a tile.
//
//   default          DeepSeek-V2-Lite   I=1408, 64 experts, top-6
//   -DMOE_QWEN36     Qwen3.6-35B-A3B    I=512, 256 experts, top-8
//
// The second is a much harder shape for fusion to help with in absolute terms
// and a much easier one in relative terms: half the FLOPs per token and
// slightly more bytes, because a token still crosses as a 2048-wide row no
// matter how small the expert is. Expect a bigger speedup over a slower step.
#ifdef MOE_QWEN36
constexpr int kI = 512;
constexpr int kE = 256;
constexpr int kTopK = 8;
#else
constexpr int kI = 1408;  // moe_intermediate per expert; GEMM2's K
constexpr int kE = 64;
constexpr int kTopK = 6;
#endif
constexpr int kH = 2048;     // hidden; GEMM1's K and GEMM2's N
constexpr int kGU = 2 * kI;  // gate and up fused; GEMM1's N
constexpr int kELoc = kE / NRANKS;
static_assert(kGU % 256 == 0 && kH % 256 == 0, "N must tile evenly at BN=256");
static_assert(kI % 8 == 0 && kGU % 8 == 0, "alignment 8 for the bf16 operands");

__device__ __host__ __forceinline__ int destOf(int rank, int k) {
  return (rank + k) % NRANKS;
}
__device__ __host__ __forceinline__ int srcOf(int rank, int k) {
  return (rank - k + NRANKS) % NRANKS;
}
__device__ __host__ __forceinline__ int ascPeer(int rank, int i) {
  return (i < rank) ? i : i + 1;
}
// Arrival group g's rows came from this rank (g == 0) or from the peer that the
// ce-shift rotation receives from at step g.
__device__ __host__ __forceinline__ int srcOfGroup(int rank, int g) {
  return (g == 0) ? rank : srcOf(rank, g);
}
// ---------------------------------------------------------------------------
// CUTLASS: three grouped GEMMs off one configuration
// ---------------------------------------------------------------------------
// Tile shape from the sweep in benchmarks/data/cutlass_gemm_probe.txt:
// 128x256x32 at 4 stages measured 1.20x grouped cuBLAS on the group-major
// split, against 128x128's 1.16x. 96 KB of shared memory means one block per SM
// and no co-residency with anything else, which is exactly why the dispatcher
// has to live inside the grid rather than beside it.
#ifndef FM_BM
#define FM_BM 128
#endif
#ifndef FM_BN
#define FM_BN 256
#endif

using CtElem = cutlass::bfloat16_t;
using RowMajor = cutlass::layout::RowMajor;
using ColMajor = cutlass::layout::ColumnMajor;
using CtSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;
using CtEpilogue =
    cutlass::epilogue::thread::LinearCombination<CtElem, 8, float, float>;
using CtTile = cutlass::gemm::GemmShape<FM_BM, FM_BN, 32>;
using CtWarp = cutlass::gemm::GemmShape<64, 64, 32>;
using CtMma = cutlass::gemm::GemmShape<16, 8, 16>;

// Plain grouped GEMM. Used for GEMM2 (its A operand is already contiguous) and
// as the type whose device wrapper answers sufficient()/can_implement().
template <typename Unused = void>
struct CtPlainT {
  using Kernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
      CtElem, RowMajor, cutlass::ComplexTransform::kNone, 8, CtElem, ColMajor,
      cutlass::ComplexTransform::kNone, 8, CtElem, RowMajor, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, CtTile, CtWarp,
      CtMma, CtEpilogue, CtSwizzle, 4>::GemmKernel;
};
using CtKernel = typename CtPlainT<>::Kernel;
using CtDev = cutlass::gemm::device::GemmGrouped<CtKernel>;

// The same grouped kernel built with gather-A, so GEMM1 reads the receive
// buffer's scattered rows through a row-index list instead of a permute kernel
// materialising them first. DefaultGemmGrouped hardcodes GatherA=false, so this
// reassembles what it does from DefaultGemm, which exposes the flag.
// kInternalTranspose is false for RowMajor C, so the gather really lands on A.
using CtGatherBase = typename cutlass::gemm::kernel::DefaultGemm<
    CtElem, RowMajor, 8, CtElem, ColMajor, 8, CtElem, RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, CtTile, CtWarp, CtMma,
    CtEpilogue, CtSwizzle, 4, true,
    typename cutlass::gemm::device::DefaultGemmConfiguration<
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, CtElem, CtElem,
        CtElem, float>::Operator,
    cutlass::gemm::SharedMemoryClearOption::kNone, true, false,
    false>::GemmKernel;
using CtGatherKernel = cutlass::gemm::kernel::GemmGrouped<
    typename CtGatherBase::Mma, typename CtGatherBase::Epilogue, CtSwizzle,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>;

// Block 0's job in the fused kernel: the ce-shift copy-engine dispatch,
// publishing arrived[k] as each peer lands. Held by value in the kernel Params.
struct MoeDispatch {
  static constexpr bool kEnabled = true;
  static constexpr int kBlocks = 1;
  PChans pc;
  int rank;
  size_t const* sendOff;
  size_t const* recvOff;
  size_t const* sendBytes;
  uint32_t* arrived;
  int volatile* prog;  // 5x issuing put k, 6x waiting on k, 69 all arrived

  CUTLASS_DEVICE void operator()() {
    if (threadIdx.x != 0) return;
    for (int k = 1; k <= NPEERS; k++) {
      if (prog) *prog = 50 + k;
      pc.c[destOf(rank, k)].putWithSignal(
          recvOff[rank], sendOff[destOf(rank, k)], sendBytes[destOf(rank, k)]);
    }
    for (int k = 1; k <= NPEERS; k++) {
      if (prog) *prog = 60 + k;
      pc.c[srcOf(rank, k)].wait();
      // The bytes were written by a DMA engine, not a thread. wait() is a
      // system-scope acquire; this fence carries that ordering to the worker
      // blocks, which pick it up by spinning on `arrived`.
      __threadfence_system();
      atomicExch(&arrived[k], 1u);
    }
    if (prog) *prog = 69;
  }
};

using CtGuFused =
    uccl_lite::GatedGrouped<CtGatherKernel,
                            MoeDispatch>;  // dispatch + gated GEMM1
using CtGuPlain =
    uccl_lite::GatedGrouped<CtGatherKernel>;          // GEMM1, gather, no gate
using CtDnPlain = uccl_lite::GatedGrouped<CtKernel>;  // GEMM2

// How many arrival groups the gated GEMM1 merges into one gate group. Two, not
// four: the group-major problem order that gating requires is what keeps a
// window of concurrent blocks on few enough experts to stay in L2, and at four
// groups that ordering costs 2.6x at decode where at two it costs 4%
// (benchmarks/data/cutlass_gemm_probe.txt).
constexpr int kGateGroups = 2;

// ---------------------------------------------------------------------------
// The small kernels around the GEMMs
// ---------------------------------------------------------------------------
__device__ __forceinline__ float silu(float v) {
  return v / (1.0f + __expf(-v));
}

// Gather the rows this rank owes one destination into a contiguous send slot.
__global__ void stageRows(__nv_bfloat16* __restrict__ dst,
                          __nv_bfloat16 const* __restrict__ x,
                          int const* __restrict__ tok, int nRows) {
  int const r = blockIdx.x;
  if (r >= nRows) return;
  __nv_bfloat16 const* s = x + (size_t)tok[r] * kH;
  __nv_bfloat16* d = dst + (size_t)r * kH;
  for (int i = threadIdx.x; i < kH / 8; i += blockDim.x)
    reinterpret_cast<float4*>(d)[i] = reinterpret_cast<float4 const*>(s)[i];
}

// The permute every cuBLAS row needs: materialise a contiguous A_e from the
// receive buffer. The CUTLASS rows fold this into gather-A and never run it.
__global__ void permuteRows(__nv_bfloat16* __restrict__ dst,
                            __nv_bfloat16 const* __restrict__ src,
                            int const* __restrict__ rowIdx, int nRows) {
  int const r = blockIdx.x;
  if (r >= nRows) return;
  __nv_bfloat16 const* s = src + (size_t)rowIdx[r] * kH;
  __nv_bfloat16* d = dst + (size_t)r * kH;
  for (int i = threadIdx.x; i < kH / 8; i += blockDim.x)
    reinterpret_cast<float4*>(d)[i] = reinterpret_cast<float4 const*>(s)[i];
}

// h [rows, 2I] -> a [rows, I]. Bandwidth-bound -- 2 reads for every write and
// no arithmetic worth the name -- so the access width is the whole game. Eight
// bf16 per thread per access, not one: at I=1408 every row offset is a multiple
// of 16 bytes, so the float4 form is always legal here.
__global__ void swigluKernel(__nv_bfloat16* __restrict__ a,
                             __nv_bfloat16 const* __restrict__ h, int rows,
                             int I) {
  int const r = blockIdx.x;
  if (r >= rows) return;
  int const nv = I / 8;
  float4 const* hg = reinterpret_cast<float4 const*>(h + (size_t)r * 2 * I);
  float4 const* hu = reinterpret_cast<float4 const*>(h + (size_t)r * 2 * I + I);
  float4* av = reinterpret_cast<float4*>(a + (size_t)r * I);
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    float4 g = hg[i], u = hu[i], o;
    __nv_bfloat162 const* gp = reinterpret_cast<__nv_bfloat162 const*>(&g);
    __nv_bfloat162 const* up = reinterpret_cast<__nv_bfloat162 const*>(&u);
    __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&o);
    for (int k = 0; k < 4; k++) {
      const float2 gf = __bfloat1622float2(gp[k]);
      const float2 uf = __bfloat1622float2(up[k]);
      op[k] = __floats2bfloat162_rn(silu(gf.x) * uf.x, silu(gf.y) * uf.y);
    }
    av[i] = o;
  }
}

// Same, reading an fp32 h. The reference uses this so that neither path is
// charged for the other's rounding: a bf16 h is rounded twice, once out of
// GEMM1 and once out of SwiGLU, and comparing against that would reward
// whichever implementation happens to round the same way.
__global__ void swigluKernel32(__nv_bfloat16* __restrict__ a,
                               float const* __restrict__ h, int rows, int I) {
  int const r = blockIdx.x;
  if (r >= rows) return;
  int const nv = I / 4;
  float4 const* hg = reinterpret_cast<float4 const*>(h + (size_t)r * 2 * I);
  float4 const* hu = reinterpret_cast<float4 const*>(h + (size_t)r * 2 * I + I);
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    const float4 g = hg[i], u = hu[i];
    a[(size_t)r * I + 4 * i + 0] = __float2bfloat16(silu(g.x) * u.x);
    a[(size_t)r * I + 4 * i + 1] = __float2bfloat16(silu(g.y) * u.y);
    a[(size_t)r * I + 4 * i + 2] = __float2bfloat16(silu(g.z) * u.z);
    a[(size_t)r * I + 4 * i + 3] = __float2bfloat16(silu(g.w) * u.w);
  }
}

// The weighted local reduce, indexed by (source, token) slot. A token routed to
// three of this rank's experts has three rows of y to fold together with their
// softmax weights, and folding them *before* the return is what keeps the
// combine at one row per (token, source) instead of kTopK times it.
__global__ void weightedReduce(__nv_bfloat16* __restrict__ dst,
                               __nv_bfloat16 const* __restrict__ y,
                               int const* __restrict__ redOff,
                               int const* __restrict__ redRow,
                               float const* __restrict__ redW, int base,
                               int n) {
  int const j = blockIdx.x;
  if (j >= n) return;
  int const lo = redOff[base + j], hi = redOff[base + j + 1];
  int const nv = kH / 8;
  float4* dv = reinterpret_cast<float4*>(dst + (size_t)j * kH);
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    // The rows are scattered but each one is 4 KB contiguous, so what matters
    // is the access width inside a row, not the order of the rows.
    for (int q = lo; q < hi; q++) {
      const float4 v =
          reinterpret_cast<float4 const*>(y + (size_t)redRow[q] * kH)[i];
      __nv_bfloat162 const* vp = reinterpret_cast<__nv_bfloat162 const*>(&v);
      float const w = redW[q];
      for (int k = 0; k < 4; k++) {
        const float2 f = __bfloat1622float2(vp[k]);
        acc[2 * k] += w * f.x;
        acc[2 * k + 1] += w * f.y;
      }
    }
    float4 o;
    __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&o);
    for (int k = 0; k < 4; k++)
      op[k] = __floats2bfloat162_rn(acc[2 * k], acc[2 * k + 1]);
    dv[i] = o;
  }
}

// Scatter-add one peer's returned partials onto the tokens we sent it.
__global__ void accumulatePeer(__nv_bfloat16* __restrict__ out,
                               __nv_bfloat16 const* __restrict__ src,
                               int const* __restrict__ tok, int n) {
  int const j = blockIdx.x;
  if (j >= n) return;
  int const t = tok[j];
  int const nv = kH / 8;
  float4* ov = reinterpret_cast<float4*>(out + (size_t)t * kH);
  float4 const* sv = reinterpret_cast<float4 const*>(src + (size_t)j * kH);
  // One launch per source, and sendTok[d] holds each token once, so no two
  // blocks of a launch touch the same t and the read-modify-write is safe.
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    float4 o = ov[i];
    const float4 v = sv[i];
    __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&o);
    __nv_bfloat162 const* vp = reinterpret_cast<__nv_bfloat162 const*>(&v);
    for (int k = 0; k < 4; k++) {
      const float2 a = __bfloat1622float2(op[k]), b = __bfloat1622float2(vp[k]);
      op[k] = __floats2bfloat162_rn(a.x + b.x, a.y + b.y);
    }
    ov[i] = o;
  }
}

// Standalone dispatch, for the rows that do not fuse it. Copy-engine rotation:
// step k sends to (r+k)%4 and receives from (r-k)%4, the (2,4,2) cross-socket
// split docs/alltoall.md proves optimal for this box.
__global__ void dispatchCE(PChans pc, int rank, size_t const* sendOff,
                           size_t const* recvOff, size_t const* sendBytes,
                           int volatile* prog) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  for (int k = 1; k <= NPEERS; k++) {
    if (prog) *prog = 10 + k;
    pc.c[destOf(rank, k)].putWithSignal(recvOff[rank], sendOff[destOf(rank, k)],
                                        sendBytes[destOf(rank, k)]);
  }
  for (int k = 1; k <= NPEERS; k++) {
    if (prog) *prog = 20 + k;
    pc.c[srcOf(rank, k)].wait();
  }
  if (prog) *prog = 29;
  __threadfence_system();
}

// Combine is dispatch in reverse: step k returns to the rank we received from
// at step k, so the schedule stays a permutation and each peer sees exactly two
// signals and two waits across the layer.
//
// This kernel does *not* wait for anything. It posts descriptors and exits,
// which is the entire reason the combine overlaps for free: the transfer runs
// on the copy engine while the next chunk's GEMM owns every SM. Splitting the
// puts across chunks is what `-Q` controls.
__global__ void putCombine(PChans pc, int rank, size_t const* cSendOff,
                           size_t const* cRecvOff, size_t const* cSendBytes,
                           int kLo, int kHi, int volatile* prog) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  // Release: the reduce that produced these bytes ran on this stream, but the
  // reader is a DMA engine on another agent.
  __threadfence_system();
  for (int k = kLo; k < kHi; k++) {
    int const d = srcOf(rank, k);
    if (prog) *prog = 30 + k;
    pc.c[d].putWithSignal(cRecvOff[rank], cSendOff[d], cSendBytes[d]);
  }
  if (prog) *prog = 39;
}

__global__ void waitCombine(PChans pc, int rank, int volatile* prog) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  for (int k = 1; k <= NPEERS; k++) {
    if (prog) *prog = 40 + k;
    pc.c[destOf(rank, k)].wait();
  }
  if (prog) *prog = 49;
  __threadfence_system();
}

__global__ void clearFlags(uint32_t* arrived) {
  // Group 0 is this rank's own rows: staged locally, never in flight, so its
  // flag starts set.
  if (threadIdx.x < NRANKS) arrived[threadIdx.x] = (threadIdx.x == 0) ? 1u : 0u;
}

__global__ void zeroRows(__nv_bfloat16* __restrict__ out, int T) {
  int const t = blockIdx.x;
  if (t >= T) return;
  for (int i = threadIdx.x; i < kH / 8; i += blockDim.x)
    reinterpret_cast<float4*>(out + (size_t)t * kH)[i] =
        make_float4(0.f, 0.f, 0.f, 0.f);
}
// ---------------------------------------------------------------------------
// One CUTLASS grouped-GEMM problem list, host and device sides together.
// ---------------------------------------------------------------------------
struct ProbSet {
  cutlass::gemm::GemmCoord* dProb = nullptr;
  CtElem **dA = nullptr, **dB = nullptr, **dC = nullptr;
  int const** dIdx = nullptr;
  int64_t *dLdA = nullptr, *dLdB = nullptr, *dLdC = nullptr;
  int* dGroupOf = nullptr;

  std::vector<cutlass::gemm::GemmCoord> hProb;
  std::vector<CtElem*> hA, hB, hC;
  std::vector<int const*> hIdx;
  std::vector<int64_t> hLdA, hLdB, hLdC;
  std::vector<int> hGroupOf;

  void alloc(int cap) {
    CUDACHECK(cudaMalloc(&dProb, cap * sizeof(cutlass::gemm::GemmCoord)));
    CUDACHECK(cudaMalloc(&dA, cap * sizeof(void*)));
    CUDACHECK(cudaMalloc(&dB, cap * sizeof(void*)));
    CUDACHECK(cudaMalloc(&dC, cap * sizeof(void*)));
    CUDACHECK(cudaMalloc(&dIdx, cap * sizeof(void*)));
    CUDACHECK(cudaMalloc(&dLdA, cap * sizeof(int64_t)));
    CUDACHECK(cudaMalloc(&dLdB, cap * sizeof(int64_t)));
    CUDACHECK(cudaMalloc(&dLdC, cap * sizeof(int64_t)));
    CUDACHECK(cudaMalloc(&dGroupOf, cap * sizeof(int)));
  }
  void clear() {
    hProb.clear();
    hA.clear();
    hB.clear();
    hC.clear();
    hIdx.clear();
    hLdA.clear();
    hLdB.clear();
    hLdC.clear();
    hGroupOf.clear();
  }
  void add(int m, int n, int k, CtElem* a, CtElem* b, CtElem* c, int64_t lda,
           int64_t ldb, int64_t ldc, int const* idx, int group) {
    hProb.push_back(cutlass::gemm::GemmCoord(m, n, k));
    hA.push_back(a);
    hB.push_back(b);
    hC.push_back(c);
    hLdA.push_back(lda);
    hLdB.push_back(ldb);
    hLdC.push_back(ldc);
    hIdx.push_back(idx);
    hGroupOf.push_back(group);
  }
  int n() const { return (int)hProb.size(); }
  void upload() {
    if (!n()) return;
    int const c = n();
    CUDACHECK(cudaMemcpy(dProb, hProb.data(), c * sizeof(hProb[0]),
                         cudaMemcpyHostToDevice));
    CUDACHECK(
        cudaMemcpy(dA, hA.data(), c * sizeof(void*), cudaMemcpyHostToDevice));
    CUDACHECK(
        cudaMemcpy(dB, hB.data(), c * sizeof(void*), cudaMemcpyHostToDevice));
    CUDACHECK(
        cudaMemcpy(dC, hC.data(), c * sizeof(void*), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dIdx, hIdx.data(), c * sizeof(void*),
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dLdA, hLdA.data(), c * sizeof(int64_t),
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dLdB, hLdB.data(), c * sizeof(int64_t),
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dLdC, hLdC.data(), c * sizeof(int64_t),
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dGroupOf, hGroupOf.data(), c * sizeof(int),
                         cudaMemcpyHostToDevice));
  }
  // Blocks that keep every problem in the list busy without oversubscribing.
  int grid(int nSM, int reserve) const {
    if (!n()) return 1;
    return std::max(
        1, std::min(CtDev::sufficient(hProb.data(), n()), nSM - reserve));
  }
  template <typename KernelT>
  typename KernelT::Arguments args(int grid) const {
    return typename KernelT::Arguments(
        dProb, n(), grid,
        typename KernelT::EpilogueOutputOp::Params(1.0f, 0.0f), dA, dB, dC, dC,
        dLdA, dLdB, dLdC, dLdC,
        const_cast<cutlass::gemm::GemmCoord*>(hProb.data()));
  }
};

// ---------------------------------------------------------------------------
// The fused layer itself, as one reusable object.
//
// The caller supplies counts and device arrays through PlanView. Routing can
// come from the benchmark's host planner or a GPU planner; the kernel object
// does not depend on a particular application or framework.
// ---------------------------------------------------------------------------

// The host-visible half of a routing plan, plus pointers to the device half.
// Counts are host-side because the CUTLASS problem list is built on the host;
// that is the one D2H stop per layer, and it is deliberate.
struct PlanView {
  int T = 0, cap = 0, totalRows = 0;
  int nSend[NRANKS] = {0};
  int nRecv[NRANKS] = {0};
  int expOff[kELoc] = {0};
  int grpOff[kELoc][NRANKS] = {{0}};
  int grpCnt[kELoc][NRANKS] = {{0}};

  int const* dRowIdx = nullptr;   // expert-major, group-minor, length totalRows
  int const* dSendTok = nullptr;  // [NRANKS][cap]
  int const* dRedOff = nullptr;   // [NRANKS*cap + 1]
  int const* dRedRow = nullptr;   // [totalRows]
  float const* dRedW = nullptr;   // [totalRows]
};

struct MegaMoe {
  // ---- set once ----------------------------------------------------------
  int rank = 0, nSM = 0;
  size_t cap = 0, slot = 0;
  size_t dSend = 0, dRecv = 0, cSend = 0, cRecv = 0;
  PChans pc{};
  char* buf = nullptr;  // the registered 4-region buffer, caller-owned

  // Scratch between the stages. Caller-owned so the benchmark can share it
  // with the ladder rows it compares against.
  __nv_bfloat16 *h = nullptr, *a = nullptr, *y = nullptr;

  // Knobs. gateSplit is -P, chunks is -Q; the defaults are the measured
  // winners (benchmarks/data/moe_layer.txt), and reserveSM is the SM held back
  // so block 0's dispatcher is guaranteed to be scheduled.
  int gateSplit = 2, chunks = 3, reserveSM = 1;

  // ---- owned by us -------------------------------------------------------
  uint32_t* dArrived = nullptr;
  // The six offset arrays are caller-owned: the benchmark's other ladder rows
  // drive dispatchCE from the same arrays, so they cannot live in here.
  // Device arrays rather than kernel arguments precisely so the sizes can be
  // data-dependent without the host knowing them at launch time.
  size_t const *dSendOff = nullptr, *dRecvOff = nullptr, *dSendBytes = nullptr;
  size_t const *dCSendOff = nullptr, *dCRecvOff = nullptr,
               *dCSendBytes = nullptr;
  ProbSet psGu;
  ProbSet psDn[NRANKS];
  int guSmem = 0, dnSmem = 0;
  int volatile* dProg = nullptr;  // FML_TRACE progress word, or null
  // The collective gets its own stream. Enqueuing the compute kernels *behind*
  // a spinning dispatch on one stream is what reproduces the documented
  // comm-only/-Q 1 hang: dispatchCE alone completes, dispatchCE with work
  // queued behind it stalls at prog=21 with every put issued.
  cudaStream_t sComm = nullptr;
  cudaEvent_t eStaged = nullptr, eArrived = nullptr;

  // ---- rebuilt every configure() ----------------------------------------
  PlanView const* p = nullptr;
  __nv_bfloat16 const* x = nullptr;
  __nv_bfloat16* out = nullptr;
  int chunkLo[NRANKS + 1] = {0};
  int guGrid = 1, dnGrid[NRANKS] = {1};
  std::vector<typename CtDnPlain::Params> dnParams;
  std::unique_ptr<typename CtGuFused::Params> guParams;
  // Same problems and pointers with arrived=nullptr, which compiles the gate
  // out entirely. Drives the unfused rung, and gets the whole GPU since there
  // is no dispatcher block to reserve for.
  std::unique_ptr<typename CtGuPlain::Params> guPlainParams;
  int guGridFull = 1;

  // Buffer geometry for a given capacity. Kept static so a caller can size its
  // allocation before constructing anything.
  static size_t bufBytesFor(size_t capacity) {
    return 4 * NRANKS * capacity * kH * sizeof(__nv_bfloat16);
  }

  void allocOnce(size_t capacity, size_t maxRows, int numSM) {
    cap = capacity;
    nSM = numSM;
    slot = cap * kH * sizeof(__nv_bfloat16);
    dSend = 0;
    dRecv = NRANKS * slot;
    cSend = 2 * NRANKS * slot;
    cRecv = 3 * NRANKS * slot;
    CUDACHECK(cudaMalloc(&dArrived, NRANKS * sizeof(uint32_t)));
    CUDACHECK(cudaStreamCreateWithFlags(&sComm, cudaStreamNonBlocking));
    CUDACHECK(cudaEventCreateWithFlags(&eStaged, cudaEventDisableTiming));
    CUDACHECK(cudaEventCreateWithFlags(&eArrived, cudaEventDisableTiming));
    psGu.alloc(kGateGroups * kELoc);
    for (int q = 0; q < NRANKS; q++) psDn[q].alloc(kELoc);
    // Each gatedGroupedKernel<Op> is its own template instantiation and needs
    // its own >48 KB opt-in. Missing one makes the launch fail silently and the
    // row hang, which is a genuinely unpleasant afternoon.
    guSmem = int(sizeof(typename CtGuFused::SharedStorage));
    dnSmem = int(sizeof(typename CtDnPlain::SharedStorage));
    CUDACHECK(cudaFuncSetAttribute(uccl_lite::gatedGroupedKernel<CtGuFused>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   guSmem));
    CUDACHECK(cudaFuncSetAttribute(uccl_lite::gatedGroupedKernel<CtGuPlain>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   guSmem));
    CUDACHECK(cudaFuncSetAttribute(uccl_lite::gatedGroupedKernel<CtDnPlain>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   dnSmem));
    (void)maxRows;
  }

  // Build the two problem lists and the kernel Params for one step's plan.
  void configure(PlanView const& plan, __nv_bfloat16 const* xIn,
                 __nv_bfloat16 const* Wgu, __nv_bfloat16 const* Wdn,
                 __nv_bfloat16* outBuf) {
    p = &plan;
    x = xIn;
    out = outBuf;

    // GEMM1: one problem per (gate group, expert), group-major. Merging the
    // four arrival groups into two is not a simplification, it is the result:
    // at four groups the group-major order costs 2.6x in lost weight reuse, at
    // two it costs 4%.
    int const gateLo[kGateGroups + 1] = {0, gateSplit, NRANKS};
    psGu.clear();
    for (int gg = 0; gg < kGateGroups; gg++)
      for (int e = 0; e < kELoc; e++) {
        int mr = 0;
        for (int g = gateLo[gg]; g < gateLo[gg + 1]; g++) mr += p->grpCnt[e][g];
        if (mr <= 0) continue;
        int const off = p->expOff[e] + p->grpOff[e][gateLo[gg]];
        psGu.add(
            mr, kGU, kH, reinterpret_cast<CtElem*>(buf + dRecv),
            reinterpret_cast<CtElem*>(const_cast<__nv_bfloat16*>(Wgu)) +
                (size_t)e * kGU * kH,
            reinterpret_cast<CtElem*>(h) + (size_t)off * kGU, kH, kH, kGU,
            p->dRowIdx + off,
            gateLo[gg + 1] - 1);  // the last arrival group this problem needs
      }
    psGu.upload();
    guGrid = psGu.grid(nSM, reserveSM);
    auto guArgs = psGu.args<CtGatherKernel>(guGrid);
    MoeDispatch disp{pc, rank, dSendOff, dRecvOff, dSendBytes, dArrived, dProg};
    guParams.reset(new typename CtGuFused::Params(
        typename CtGatherKernel::Params(guArgs, nullptr, 0), dArrived,
        psGu.dGroupOf, psGu.dIdx, disp));

    guGridFull = psGu.grid(nSM, 0);
    auto guArgsFull = psGu.args<CtGatherKernel>(guGridFull);
    guPlainParams.reset(new typename CtGuPlain::Params(
        typename CtGatherKernel::Params(guArgsFull, nullptr, 0), nullptr,
        nullptr, psGu.dIdx));

    // GEMM2: one problem list per combine chunk. Rows are group-minor within an
    // expert, so a chunk's span is contiguous and GEMM2 needs neither a gather
    // nor a permute. The uneven boundary for Q=3 is deliberate -- it puts the
    // two cheap groups last, where the remaining compute is smallest.
    int const Q = chunks;
    for (int q = 0; q <= Q; q++) chunkLo[q] = (q * NRANKS + Q - 1) / Q;
    dnParams.clear();
    dnParams.reserve(Q);
    for (int q = 0; q < Q; q++) {
      psDn[q].clear();
      for (int e = 0; e < kELoc; e++) {
        int mr = 0;
        for (int g = chunkLo[q]; g < chunkLo[q + 1]; g++) mr += p->grpCnt[e][g];
        if (mr <= 0) continue;
        int const off = p->expOff[e] + p->grpOff[e][chunkLo[q]];
        psDn[q].add(mr, kH, kI, reinterpret_cast<CtElem*>(a) + (size_t)off * kI,
                    reinterpret_cast<CtElem*>(const_cast<__nv_bfloat16*>(Wdn)) +
                        (size_t)e * kH * kI,
                    reinterpret_cast<CtElem*>(y) + (size_t)off * kH, kI, kI, kH,
                    nullptr, 0);
      }
      psDn[q].upload();
      dnGrid[q] = psDn[q].grid(nSM, 0);
      auto dnArgs = psDn[q].args<CtKernel>(dnGrid[q]);
      dnParams.emplace_back(typename CtKernel::Params(dnArgs, nullptr, 0),
                            nullptr, nullptr, nullptr);
    }
  }

  // ---- the stages --------------------------------------------------------
  void stageAll(cudaStream_t s) {
    for (int d = 0; d < NRANKS; d++) {
      if (!p->nSend[d]) continue;
      // Own rows go straight into the receive slot, so the GEMM sees one
      // uniform [4][cap][H] view whether a row crossed PCIe or not.
      __nv_bfloat16* dst = (d == rank) ? reinterpret_cast<__nv_bfloat16*>(
                                             buf + dRecv + (size_t)rank * slot)
                                       : reinterpret_cast<__nv_bfloat16*>(
                                             buf + dSend + (size_t)d * slot);
      stageRows<<<p->nSend[d], 256, 0, s>>>(
          dst, x, p->dSendTok + (size_t)d * cap, p->nSend[d]);
    }
  }
  // SwiGLU over one chunk: per (chunk, expert), because rows are expert-major
  // and only contiguous within one expert.
  void swigluChunk(int q, cudaStream_t s) {
    for (int e = 0; e < kELoc; e++) {
      int mr = 0;
      for (int g = chunkLo[q]; g < chunkLo[q + 1]; g++) mr += p->grpCnt[e][g];
      if (!mr) continue;
      int const off = p->expOff[e] + p->grpOff[e][chunkLo[q]];
      swigluKernel<<<mr, 256, 0, s>>>(a + (size_t)off * kI,
                                      h + (size_t)off * kGU, mr, kI);
    }
  }
  void reduceChunk(int q, cudaStream_t s) {
    for (int g = chunkLo[q]; g < chunkLo[q + 1]; g++) {
      int const sr = srcOfGroup(rank, g);
      if (p->nRecv[sr])
        weightedReduce<<<p->nRecv[sr], 256, 0, s>>>(
            reinterpret_cast<__nv_bfloat16*>(buf + cSend + (size_t)sr * slot),
            y, p->dRedOff, p->dRedRow, p->dRedW, sr * (int)cap, p->nRecv[sr]);
    }
  }
  // Rotation steps this chunk releases. Group 0 is this rank's own rows: its
  // reduce writes cSend[rank] and finish() reads it locally, so there is no
  // step 0 to send.
  void putChunk(int q, cudaStream_t s) {
    int const kLo = std::max(1, chunkLo[q]), kHi = chunkLo[q + 1];
    if (kLo < kHi)
      putCombine<<<1, 32, 0, s>>>(pc, rank, dCSendOff, dCRecvOff, dCSendBytes,
                                  kLo, kHi, dProg);
  }
  void finish(cudaStream_t s) {
    zeroRows<<<p->T, 256, 0, s>>>(out, p->T);
    for (int d = 0; d < NRANKS; d++) {
      if (!p->nSend[d]) continue;
      __nv_bfloat16 const* src = (d == rank)
                                     ? reinterpret_cast<__nv_bfloat16*>(
                                           buf + cSend + (size_t)rank * slot)
                                     : reinterpret_cast<__nv_bfloat16*>(
                                           buf + cRecv + (size_t)d * slot);
      accumulatePeer<<<p->nSend[d], 256, 0, s>>>(
          out, src, p->dSendTok + (size_t)d * cap, p->nSend[d]);
    }
  }

  // The same layer, strictly sequential: a standalone ce-shift dispatch that
  // completes before the GEMM starts, and the gate compiled out. This is the
  // `ours+cutlass` rung -- it keeps the ragged All-to-All and the grouped
  // CUTLASS GEMMs but gives up the overlap, so it is worth about 1.15x rather
  // than 1.41x. It exists because it does not deadlock: nothing spins while
  // waiting for a copy that cannot be serviced.
  void runUnfused(cudaStream_t s) {
    stageAll(s);
    // Cross to sComm for the dispatch and back, so nothing is queued behind the
    // spinning kernel on the stream it occupies.
    CUDACHECK(cudaEventRecord(eStaged, s));
    CUDACHECK(cudaStreamWaitEvent(sComm, eStaged, 0));
    dispatchCE<<<1, 32, 0, sComm>>>(pc, rank, dSendOff, dRecvOff, dSendBytes,
                                    dProg);
    // Drain the dispatch on the host before enqueuing anything else.
    //
    // Bisect result: stageAll + dispatchCE alone completes; the same dispatch
    // with the ~58 compute launches queued behind it stalls at prog=21 with
    // every put issued. A separate stream does not help, which points at the
    // driver's per-context submission path rather than stream ordering -- the
    // launching thread blocks once the pending-launch queue fills, and the
    // proxy thread cannot get its copy in.
    //
    // Synchronising here costs the overlap this design exists for, so it is a
    // correctness crutch for the sequential rung, not the shipping answer.
    CUDACHECK(cudaStreamSynchronize(sComm));
    if (psGu.n())
      uccl_lite::gatedGroupedKernel<CtGuPlain>
          <<<guGridFull, CtGuPlain::kThreadCount, guSmem, s>>>(*guPlainParams);
    for (int q = 0; q < chunks; q++) {
      swigluChunk(q, s);
      if (psDn[q].n())
        uccl_lite::gatedGroupedKernel<CtDnPlain>
            <<<dnGrid[q], CtDnPlain::kThreadCount, dnSmem, s>>>(dnParams[q]);
      reduceChunk(q, s);
      putChunk(q, s);
    }
    waitCombine<<<1, 32, 0, s>>>(pc, rank, dProg);
    // Same crutch as the dispatch, one stage later: waitCombine spins while
    // finish()'s launches would queue behind it, and that is what stalls the
    // proxy. Drain first.
    CUDACHECK(cudaStreamSynchronize(s));
    finish(s);
  }

  // The whole layer. Four kernels' worth of dependency and nothing spins except
  // block 0 of the gated GEMM and the final wait.
  void run(cudaStream_t s) {
    stageAll(s);
    clearFlags<<<1, 32, 0, s>>>(dArrived);
    if (psGu.n())
      uccl_lite::gatedGroupedKernel<CtGuFused>
          <<<guGrid + 1, CtGuFused::kThreadCount, guSmem, s>>>(*guParams);
    for (int q = 0; q < chunks; q++) {
      swigluChunk(q, s);
      if (psDn[q].n())
        uccl_lite::gatedGroupedKernel<CtDnPlain>
            <<<dnGrid[q], CtDnPlain::kThreadCount, dnSmem, s>>>(dnParams[q]);
      reduceChunk(q, s);
      putChunk(q, s);
    }
    waitCombine<<<1, 32, 0, s>>>(pc, rank, dProg);
    finish(s);
  }
};
