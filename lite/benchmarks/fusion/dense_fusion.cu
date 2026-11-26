// Dense tensor-parallel fusion for the PCIe-only 4xL4 node.
//
//   AG + GEMM : recursive-doubling, NUMA-local first.  Block 0 drives the
//               copy engines while the remaining blocks run one unsplit,
//               tile-gated CUTLASS GEMM.
//
//   GEMM + RS : destination tiles are issued in the topology ring's consume
//               order and publish readiness immediately.  A small resident RS
//               grid consumes them while cuBLAS computes later tiles.
//
//   GEMM + AR : the same ready ring immediately all-gathers each completed RS
//               stripe, overlapping that tail with the next stripe's GEMM.
//
// The binary contains NCCL+cuBLAS baselines and diagnostic rungs using exactly
// the same tensors.  It is intentionally specialised to four ranks and the
// hierarchy used by src/collectives/allreduce.cu.

#include "cutlass_gated_grouped.h"
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/numeric_types.h>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/memory_channel_device.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <mscclpp/semaphore.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <sys/wait.h>
#include <unistd.h>

#define NRANKS 4

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

#define CUBLASCHECK(cmd)                                                  \
  do {                                                                    \
    cublasStatus_t s_ = (cmd);                                            \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                    \
      std::fprintf(stderr, "%s:%d cuBLAS error %d\n", __FILE__, __LINE__, \
                   (int)s_);                                              \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

#define NCCLCHECK(cmd)                                                   \
  do {                                                                   \
    ncclResult_t r_ = (cmd);                                             \
    if (r_ != ncclSuccess) {                                             \
      std::fprintf(stderr, "%s:%d NCCL error: %s\n", __FILE__, __LINE__, \
                   ncclGetErrorString(r_));                              \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

using PortChan = mscclpp::PortChannelDeviceHandle;
using MemChan = mscclpp::MemoryChannelDeviceHandle;
using Elem = cutlass::half_t;

__device__ mscclpp::DeviceSyncer gDenseSyncer;

#ifndef DENSE_BM
#define DENSE_BM 128
#endif
#ifndef DENSE_BN
#define DENSE_BN 256
#endif
#ifndef RS_BLOCKS
#define RS_BLOCKS 8
#endif
#ifndef READY_RS_BLOCKS
#define READY_RS_BLOCKS 3
#endif
#ifndef READY_AR_BLOCKS
#define READY_AR_BLOCKS 2
#endif
#ifndef READY_AR_THREADS
#define READY_AR_THREADS 256
#endif
#ifndef READY_AR_SUBS
#define READY_AR_SUBS 4
#endif
#ifndef AR_BLOCKS
#define AR_BLOCKS 24
#endif
#ifndef AR_THREADS
#define AR_THREADS 512
#endif
#ifndef AR_SUBS
#define AR_SUBS 4
#endif

using DenseKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    Elem, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8, Elem,
    cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8, Elem,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80, cutlass::gemm::GemmShape<DENSE_BM, DENSE_BN, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<Elem, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4>::GemmKernel;
using DenseGemm = cutlass::gemm::device::GemmGrouped<DenseKernel>;
using DensePlain = uccl_lite::GatedGrouped<DenseKernel>;

// Flux's SM80/89 search space includes a 64x256 CTA tile.  On L4 it is much
// better than the large-M 128x256 kernel for the M=128 decode/prefill boundary:
// the extra M wave exposes enough independent N tiles to cover AG latency.
using SmallDenseKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    Elem, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8, Elem,
    cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8, Elem,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<Elem, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4>::GemmKernel;
using SmallDenseGemm = cutlass::gemm::device::GemmGrouped<SmallDenseKernel>;
using SmallDensePlain = uccl_lite::GatedGrouped<SmallDenseKernel>;

// Small-M AR needs enough M waves to publish a useful contiguous half before
// the whole GEMM finishes.  A 32-row CTA supplies that parallelism without
// splitting the host GEMM launch, while communication remains a large,
// contiguous hierarchy transfer.
using TinyDenseKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    Elem, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8, Elem,
    cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8, Elem,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80, cutlass::gemm::GemmShape<32, 256, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<Elem, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4>::GemmKernel;
using TinyDenseGemm = cutlass::gemm::device::GemmGrouped<TinyDenseKernel>;
using TinyDensePlain = uccl_lite::GatedGrouped<TinyDenseKernel>;

static size_t alignUp(size_t x, size_t a = 256) { return (x + a - 1) / a * a; }

struct Cfg {
  int agM = 4096;
  int agN = 49152;
  int agK = 12288;
  int rsM = 4096;
  int rsN = 12288;
  int rsK = 49152;
  int rsStripes = 1;
  // Zero selects the L4-tuned size-dependent policy below.  Explicit values
  // remain useful for topology/shape sweeps.
  int readyRsBlocks = 0;
  int readyRsThreads = 0;
  int fusedArBlocks = 8;
  int fusedArGemmBlocks = 0;
  int fusedArTiles = 0;
  int warmup = 2;
  int iters = 7;
  bool check = true;
};

__global__ void fillHalf(__half* p, size_t n, int seed, float scale) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
    unsigned x = (unsigned)i * 1664525u + (unsigned)seed * 1013904223u;
    int v = (int)((x >> 24) % 17) - 8;
    p[i] = __float2half((float)v * scale);
  }
}

__global__ void clearAgFlags(uint32_t* f) {
  if (threadIdx.x == 0) {
    f[0] = 1;
    f[1] = 0;
    f[2] = 0;
  }
}

__global__ void waitAgFlag(const volatile uint32_t* f, int i) {
  if (threadIdx.x == 0)
    while (f[i] == 0) __nanosleep(128);
}

struct AgDispatch {
  static constexpr bool kEnabled = true;
  static constexpr int kBlocks = 1;
  PortChan intra;
  PortChan cross;
  int rank;
  size_t base;
  size_t rankBytes;
  uint32_t* arrived;

  CUTLASS_DEVICE void operator()() {
    if (threadIdx.x != 0) return;

    // Phase 1: exchange one rank inside the NUMA node.
    intra.putWithSignal(base + (size_t)rank * rankBytes,
                        base + (size_t)rank * rankBytes, rankBytes);
    intra.wait(-1);
    __threadfence_system();
    atomicExch(arrived + 1, 1u);

    // Phase 2: each GPU exchanges the now-complete two-rank NUMA slab with
    // its same-position partner in the other NUMA node.
    const size_t pair = base + (size_t)(rank & 2) * rankBytes;
    cross.putWithSignal(pair, pair, 2 * rankBytes);
    cross.wait(-1);
    __threadfence_system();
    atomicExch(arrived + 2, 1u);
  }
};

using AgFused = uccl_lite::GatedGrouped<DenseKernel, AgDispatch>;

__global__ void agOnly(AgDispatch d) { d(); }

__global__ void agIssueIntra(PortChan intra, int rank, size_t base,
                             size_t rankBytes) {
  if (threadIdx.x == 0)
    intra.putWithSignal(base + (size_t)rank * rankBytes,
                        base + (size_t)rank * rankBytes, rankBytes);
}

__global__ void agWaitIntraIssueCross(PortChan intra, PortChan cross, int rank,
                                      size_t base, size_t rankBytes) {
  if (threadIdx.x != 0) return;
  intra.wait(-1);
  __threadfence_system();
  size_t pair = base + (size_t)(rank & 2) * rankBytes;
  cross.putWithSignal(pair, pair, 2 * rankBytes);
}

__global__ void agWaitCross(PortChan cross) {
  if (threadIdx.x == 0) {
    cross.wait(-1);
    __threadfence_system();
  }
}

struct RsSlot {
  size_t data;
  size_t phase1;
  size_t temp;
  size_t phase2;
};

__device__ __forceinline__ void addHalf(__half* dst, __half const* a,
                                        __half const* b, size_t n, int tid,
                                        int nth) {
  union Pack128 {
    uint4 v;
    __half2 h2[4];
  } x, y, z;
  const size_t n8 = n / 8;
  uint4* d8 = reinterpret_cast<uint4*>(dst);
  uint4 const* a8 = reinterpret_cast<uint4 const*>(a);
  uint4 const* b8 = reinterpret_cast<uint4 const*>(b);
  for (size_t i = tid; i < n8; i += nth) {
    x.v = a8[i];
    y.v = b8[i];
#pragma unroll
    for (int j = 0; j < 4; j++) z.h2[j] = __hadd2(x.h2[j], y.h2[j]);
    d8[i] = z.v;
  }
  const size_t tail = n8 * 8;
  for (size_t i = tail + tid; i < n; i += nth) dst[i] = __hadd(a[i], b[i]);
}

__device__ __forceinline__ uint32_t
loadAcquireSys(const volatile uint32_t* ptr) {
  uint32_t value;
  asm volatile("ld.global.acquire.sys.b32 %0, [%1];\n"
               : "=r"(value)
               : "l"(ptr));
  return value;
}

__host__ __device__ __forceinline__ int ringRank(int pos) {
  pos &= 3;
  return pos < 2 ? pos : 5 - pos;  // positions: 0,1,3,2
}

__host__ __device__ __forceinline__ int ringPos(int rank) {
  return rank < 2 ? rank : 5 - rank;
}

// A single CUTLASS launch owns both sides of the pipeline.  Blocks [0,kBlocks)
// consume completed row tiles while the remaining blocks execute GEMM.  Unlike
// the earlier striped-cuBLAS prototype, this keeps one GEMM mainloop and
// publishes readiness directly after each CUTLASS epilogue.
struct ArHierDispatch {
  static constexpr bool kEnabled = true;
  static constexpr int kBlocks = 8;

  MemChan intra;
  MemChan cross;
  int rank;
  size_t dataOff;
  size_t scratchOff;
  int m;
  int n;
  int tileM;
  int mTiles;
  int nTiles;
  int tilesPerChunk;
  int activeBlocks;
  uint32_t* completed;

  CUTLASS_DEVICE void operator()() {
    if ((int)blockIdx.x >= activeBlocks) return;
    int const tid = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int const nth = activeBlocks * (int)blockDim.x;
    char* base = reinterpret_cast<char*>(intra.src_);
    __half* data = reinterpret_cast<__half*>(base + dataOff);
    __half* slotA = reinterpret_cast<__half*>(base + scratchOff);

    for (int mt = 0; mt < mTiles; mt += tilesPerChunk) {
      int const mtEnd = min(mTiles, mt + tilesPerChunk);
      int const rows = min((mtEnd - mt) * tileM, m - mt * tileM);
      const size_t nelem = (size_t)rows * n;
      const size_t halfN = nelem / 2;
      const size_t halfB = halfN * sizeof(__half);
      const size_t chunkOff = dataOff + (size_t)mt * tileM * n * sizeof(__half);
      int const myHalf = rank & 1;
      int const otherHalf = myHalf ^ 1;
      __half* chunk = data + (size_t)mt * tileM * n;
      __half* slotB = reinterpret_cast<__half*>(base + scratchOff + halfB);

      auto waitRows = [&](int rowBegin, int rowCount) {
        if (tid == 0) {
          const int first = rowBegin / tileM;
          const int last = (rowBegin + rowCount + tileM - 1) / tileM;
          for (int readyMt = first; readyMt < last; readyMt++)
            for (int nt = 0; nt < nTiles; nt++)
              while (loadAcquireSys(completed + readyMt * nTiles + nt) == 0)
                __nanosleep(64);
        }
        gDenseSyncer.sync(activeBlocks);
      };

      int const chunkRow = mt * tileM;
      int const halfRows = rows / 2;
      waitRows(chunkRow + otherHalf * halfRows, halfRows);

      intra.put<16, true>(scratchOff, chunkOff + (size_t)otherHalf * halfB,
                          halfB, tid, nth);
      gDenseSyncer.sync(activeBlocks);
      if (tid == 0) {
        intra.signal();
        intra.wait();
      }
      gDenseSyncer.sync(activeBlocks);
      waitRows(chunkRow + myHalf * halfRows, halfRows);
      addHalf(chunk + (size_t)myHalf * halfN, chunk + (size_t)myHalf * halfN,
              slotA, halfN, tid, nth);
      gDenseSyncer.sync(activeBlocks);

      cross.put<16, true>(scratchOff + halfB, chunkOff + (size_t)myHalf * halfB,
                          halfB, tid, nth);
      gDenseSyncer.sync(activeBlocks);
      if (tid == 0) {
        cross.signal();
        cross.wait();
      }
      gDenseSyncer.sync(activeBlocks);
      addHalf(chunk + (size_t)myHalf * halfN, chunk + (size_t)myHalf * halfN,
              slotB, halfN, tid, nth);
      gDenseSyncer.sync(activeBlocks);

      intra.put<16, true>(chunkOff + (size_t)myHalf * halfB,
                          chunkOff + (size_t)myHalf * halfB, halfB, tid, nth);
      gDenseSyncer.sync(activeBlocks);
      if (tid == 0) {
        intra.signal();
        intra.wait();
      }
      gDenseSyncer.sync(activeBlocks);
    }
  }
};

using ArFused = uccl_lite::GatedGrouped<DenseKernel, ArHierDispatch>;
using SmallArFused = uccl_lite::GatedGrouped<SmallDenseKernel, ArHierDispatch>;
using TinyArFused = uccl_lite::GatedGrouped<TinyDenseKernel, ArHierDispatch>;

__global__ void arHierOnly(ArHierDispatch d) { d(); }

// One stripe contains q rows for each of the four final destinations.  The
// hierarchy is recursive halving: retain the destination chunks whose low bit
// matches this rank, reduce with the NUMA peer, then retain the one whose high
// bit matches this rank and reduce with the cross-NUMA partner.
__global__ void gemmRsStripe(PortChan intra, PortChan cross, char* base,
                             RsSlot s, int rank, size_t chunkElems,
                             __half* finalOut) {
  int const tid = blockIdx.x * blockDim.x + threadIdx.x;
  int const nth = gridDim.x * blockDim.x;
  const size_t chunkBytes = chunkElems * sizeof(__half);
  __half* data = reinterpret_cast<__half*>(base + s.data);
  __half* p1 = reinterpret_cast<__half*>(base + s.phase1);
  __half* tmp = reinterpret_cast<__half*>(base + s.temp);
  __half* p2 = reinterpret_cast<__half*>(base + s.phase2);
  int const low = rank & 1;
  int const high = rank >> 1;

  // GEMM ran on another stream.  Publish every worker's output before block 0
  // hands the source addresses to the proxy copy engines.
  __threadfence_system();
  gDenseSyncer.sync(gridDim.x);
  if (tid == 0) {
    int const sendLow = low ^ 1;
    for (int h = 0; h < 2; h++) {
      int const dest = sendLow + 2 * h;
      intra.putWithSignal(s.phase1 + (size_t)h * chunkBytes,
                          s.data + (size_t)dest * chunkBytes, chunkBytes);
    }
    intra.wait(-1);
    intra.wait(-1);
    // Put and receive are full-duplex work.  Waiting for the inbound copies
    // before flushing lets them overlap; the flush still closes the sender's
    // buffer lifetime before the slot may be reused.
    intra.flush(-1);
  }
  gDenseSyncer.sync(gridDim.x);

  for (int h = 0; h < 2; h++) {
    int const dest = low + 2 * h;
    addHalf(tmp + (size_t)h * chunkElems, data + (size_t)dest * chunkElems,
            p1 + (size_t)h * chunkElems, chunkElems, tid, nth);
  }

  // Device release/acquire through the grid synchronizer followed by the
  // proxy FIFO's system-release store is sufficient here (the allreduce code
  // measures a redundant system fence at 30-50 us per handoff).
  __threadfence();
  gDenseSyncer.sync(gridDim.x);
  if (tid == 0) {
    int const sendHigh = high ^ 1;
    cross.putWithSignal(s.phase2, s.temp + (size_t)sendHigh * chunkBytes,
                        chunkBytes);
    cross.wait(-1);
    cross.flush(-1);
  }
  gDenseSyncer.sync(gridDim.x);
  addHalf(finalOut, tmp + (size_t)high * chunkElems, p2, chunkElems, tid, nth);
  gDenseSyncer.sync(gridDim.x);
}

// A denser two-stripe schedule.  Stripe h computes the two contiguous output
// rank chunks owned by NUMA node h.  After the intra-NUMA exchange each rank
// retains its low-bit chunk; only the opposite NUMA node sends across UPI, so
// the cross phase is one-way and lands directly at the final owners.
__global__ void gemmRsHalf(PortChan intra, PortChan cross, char* base, RsSlot s,
                           int rank, int ownerHigh, size_t chunkElems,
                           __half* finalOut) {
  int const tid = blockIdx.x * blockDim.x + threadIdx.x;
  int const nth = gridDim.x * blockDim.x;
  const size_t chunkBytes = chunkElems * sizeof(__half);
  __half* data = reinterpret_cast<__half*>(base + s.data);
  __half* p1 = reinterpret_cast<__half*>(base + s.phase1);
  __half* tmp = reinterpret_cast<__half*>(base + s.temp);
  __half* p2 = reinterpret_cast<__half*>(base + s.phase2);
  int const low = rank & 1;
  int const high = rank >> 1;

  __threadfence_system();
  gDenseSyncer.sync(gridDim.x);
  if (tid == 0) {
    intra.putWithSignal(s.phase1, s.data + (size_t)(low ^ 1) * chunkBytes,
                        chunkBytes);
    intra.wait(-1);
    intra.flush(-1);
  }
  gDenseSyncer.sync(gridDim.x);
  addHalf(tmp, data + (size_t)low * chunkElems, p1, chunkElems, tid, nth);

  __threadfence();
  gDenseSyncer.sync(gridDim.x);
  if (tid == 0) {
    if (high == ownerHigh) {
      cross.wait(-1);
    } else {
      cross.putWithSignal(s.phase2, s.temp, chunkBytes);
      cross.flush(-1);
    }
  }
  gDenseSyncer.sync(gridDim.x);
  if (high == ownerHigh) addHalf(finalOut, tmp, p2, chunkElems, tid, nth);
  gDenseSyncer.sync(gridDim.x);
}

// Large-message reduce-scatter follows the allreduce A3 ring because it uses
// both PCIe roots more evenly than the phase-serial hierarchy.  At each step a
// completed destination chunk advances once around 0->1->3->2->0 and is
// reduced immediately; after three steps rank r owns destination chunk r.
__global__ void gemmRsRing(PortChan intra, PortChan cross, char* base, RsSlot s,
                           int rank, size_t chunkElems, __half* finalOut) {
  int const tid = blockIdx.x * blockDim.x + threadIdx.x;
  int const nth = gridDim.x * blockDim.x;
  const size_t chunkBytes = chunkElems * sizeof(__half);
  __half* data = reinterpret_cast<__half*>(base + s.data);
  __half* scratch = reinterpret_cast<__half*>(base + s.phase1);
  int const pos = ringPos(rank);
  int const succ = ringRank(pos + 1);
  int const pred = ringRank(pos + 3);
  PortChan send = (succ == (rank ^ 1)) ? intra : cross;
  PortChan recv = (pred == (rank ^ 1)) ? intra : cross;

  for (int step = 0; step < NRANKS - 1; step++) {
    // Every step after the first sends values produced by all blocks in the
    // preceding reduction.  A device fence is not enough for the proxy DMA
    // engine under sustained large-message load; publish at system scope.
    __threadfence_system();
    gDenseSyncer.sync(gridDim.x);
    int const sendPos = (pos - step - 1) & 3;
    int const recvPos = (pos - step - 2) & 3;
    int const sendDest = ringRank(sendPos);
    int const recvDest = ringRank(recvPos);
    if (tid == 0) {
      send.putWithSignal(s.phase1 + (size_t)step * chunkBytes,
                         s.data + (size_t)sendDest * chunkBytes, chunkBytes);
      recv.wait(-1);
      send.flush(-1);
    }
    gDenseSyncer.sync(gridDim.x);
    __half* dst =
        (step == NRANKS - 2) ? finalOut : data + (size_t)recvDest * chunkElems;
    addHalf(dst, data + (size_t)recvDest * chunkElems,
            scratch + (size_t)step * chunkElems, chunkElems, tid, nth);
    gDenseSyncer.sync(gridDim.x);
  }
}

__global__ void publishRsReady(uint32_t* ready, int dst) {
  if (threadIdx.x == 0) {
    __threadfence_system();
    atomicExch(ready + dst, 1u);
  }
}

// Flux's PCIe GEMM+RS consumer does not wait for the whole GEMM: it consumes
// output regions as soon as their epilogues publish readiness.  cuBLAS cannot
// expose per-CTA epilogue callbacks, so use the largest efficient unit it does
// expose here -- one complete destination shard.  The four shard GEMMs are
// issued in ring-consumption order.  This resident kernel starts after shard 0
// and overlaps the three ring steps with the remaining shard GEMMs.  It lives
// on the communication stream but uses only READY_RS_BLOCKS CTAs, avoiding the
// all-SM waiter/GEMM scheduling cycle.
__global__ void gemmRsReadyRing(PortChan intra, PortChan cross, char* base,
                                size_t dataOff, size_t scratchOff, int rank,
                                int stripes, size_t chunkElems, uint32_t* ready,
                                __half* finalOut, int allReduceMode, int nsub) {
  int const tid = blockIdx.x * blockDim.x + threadIdx.x;
  int const nth = gridDim.x * blockDim.x;
  const size_t chunkBytes = chunkElems * sizeof(__half);
  __half* data = reinterpret_cast<__half*>(base + dataOff);
  __half* scratch = reinterpret_cast<__half*>(base + scratchOff);
  int const pos = ringPos(rank);
  int const succ = ringRank(pos + 1);
  int const pred = ringRank(pos + 3);
  PortChan send = (succ == (rank ^ 1)) ? intra : cross;
  PortChan recv = (pred == (rank ^ 1)) ? intra : cross;

  auto waitReady = [&](int stripe, int dst) {
    if (tid == 0) {
      volatile uint32_t* flag = ready + stripe * NRANKS + dst;
      while (*flag == 0) __nanosleep(128);
    }
    gDenseSyncer.sync(gridDim.x);
  };

  for (int stripe = 0; stripe < stripes; stripe++) {
    for (int step = 0; step < NRANKS - 1; step++) {
      int const sendDest = ringRank(pos - step - 1);
      int const recvDest = ringRank(pos - step - 2);
      const size_t sendChunk =
          ((size_t)sendDest * stripes + stripe) * chunkBytes;
      const size_t recvChunk =
          ((size_t)recvDest * stripes + stripe) * chunkElems;
      const size_t pieceElems =
          ((chunkElems + nsub - 1) / nsub + 7) & ~(size_t)7;

      // Publish the outgoing chunk, then wait for the independent local GEMM
      // tile while the predecessor's DMA is already in flight.
      if (step == 0) waitReady(stripe, sendDest);
      __threadfence_system();
      gDenseSyncer.sync(gridDim.x);
      if (tid == 0) {
        for (int piece = 0; piece < nsub; piece++) {
          const size_t off = (size_t)piece * pieceElems;
          if (off >= chunkElems) break;
          const size_t len = min(pieceElems, chunkElems - off);
          send.putWithSignal(
              scratchOff + ((size_t)step * chunkElems + off) * sizeof(__half),
              dataOff + sendChunk + off * sizeof(__half), len * sizeof(__half));
        }
      }

      waitReady(stripe, recvDest);
      __half* src = data + recvChunk;
      __half* dst = step == NRANKS - 2 && allReduceMode == 0
                        ? finalOut + (size_t)stripe * chunkElems
                        : src;
      for (int piece = 0; piece < nsub; piece++) {
        const size_t off = (size_t)piece * pieceElems;
        if (off >= chunkElems) break;
        const size_t len = min(pieceElems, chunkElems - off);
        if (tid == 0) recv.wait(-1);
        gDenseSyncer.sync(gridDim.x);
        addHalf(dst + off, src + off, scratch + (size_t)step * chunkElems + off,
                len, tid, nth);
        gDenseSyncer.sync(gridDim.x);
      }
    }

    if (allReduceMode == 1) {
      // Protect the successor's partial stripe, then immediately gather this
      // stripe while the producer computes the next one.
      if (tid == 0) {
        recv.signal();
        send.wait(-1);
      }
      gDenseSyncer.sync(gridDim.x);
      __threadfence_system();
      gDenseSyncer.sync(gridDim.x);
      for (int step = 0; step < NRANKS - 1; step++) {
        int const sendDest = ringRank(pos - step);
        const size_t off = ((size_t)sendDest * stripes + stripe) * chunkBytes;
        if (tid == 0) {
          send.putWithSignal(dataOff + off, dataOff + off, chunkBytes);
          recv.wait(-1);
        }
        gDenseSyncer.sync(gridDim.x);
      }
    }
  }

  if (allReduceMode == 2) {
    // Hierarchy is retained as a diagnostic.  Wait for all four ranks before
    // either pair can overwrite partial data on the other socket.
    if (tid == 0) {
      intra.signal();
      cross.signal();
      intra.wait(-1);
      cross.wait(-1);
    }
    gDenseSyncer.sync(gridDim.x);
    __threadfence_system();
    gDenseSyncer.sync(gridDim.x);

    const size_t ownOff = ((size_t)rank * stripes) * chunkBytes;
    if (tid == 0) {
      intra.putWithSignal(dataOff + ownOff, dataOff + ownOff,
                          (size_t)stripes * chunkBytes);
      intra.wait(-1);
    }
    gDenseSyncer.sync(gridDim.x);

    int const pairBase = rank & 2;
    const size_t pairOff = ((size_t)pairBase * stripes) * chunkBytes;
    if (tid == 0) {
      cross.putWithSignal(dataOff + pairOff, dataOff + pairOff,
                          (size_t)2 * stripes * chunkBytes);
      cross.wait(-1);
    }
    gDenseSyncer.sync(gridDim.x);
  }

  // No forwarded source tile is modified again, so all edges share one retire
  // at the end instead of serializing every step.  The final flush makes the
  // registered buffer safe to reuse without relying on an external barrier.
  if (tid == 0) {
    if (allReduceMode == 2) {
      intra.flush(-1);
      cross.flush(-1);
    } else {
      send.flush(-1);
    }
  }
  gDenseSyncer.sync(gridDim.x);
}

// Non-overlapped UCCL Lite A3 baseline.  The full GEMM has already completed
// before this kernel starts.  Each 24 MiB rank chunk is split into 6 MiB
// pieces so reduction of piece j overlaps DMA of piece j+1, matching the
// large-message schedule in src/collectives/allreduce.cu.
__global__ void pcieAllReduceRing(PortChan intra, PortChan cross, char* base,
                                  size_t dataOff, size_t scratchOff, int rank,
                                  size_t nelem, int nsub) {
  int const tid = blockIdx.x * blockDim.x + threadIdx.x;
  int const nth = gridDim.x * blockDim.x;
  const size_t chunkElems = nelem / NRANKS;
  const size_t chunkBytes = chunkElems * sizeof(__half);
  const size_t pieceElems = ((chunkElems + nsub - 1) / nsub + 7) & ~(size_t)7;
  __half* data = reinterpret_cast<__half*>(base + dataOff);
  __half* scratch = reinterpret_cast<__half*>(base + scratchOff);
  int const pos = ringPos(rank);
  int const succ = ringRank(pos + 1);
  int const pred = ringRank(pos + 3);
  PortChan send = (succ == (rank ^ 1)) ? intra : cross;
  PortChan recv = (pred == (rank ^ 1)) ? intra : cross;

  for (int step = 0; step < NRANKS - 1; step++) {
    int const sendDest = ringRank(pos - step - 1);
    int const recvDest = ringRank(pos - step - 2);
    __threadfence_system();
    gDenseSyncer.sync(gridDim.x);
    if (tid == 0) {
      for (int piece = 0; piece < nsub; piece++) {
        const size_t off = (size_t)piece * pieceElems;
        if (off >= chunkElems) break;
        const size_t len = min(pieceElems, chunkElems - off);
        send.putWithSignal(
            scratchOff + ((size_t)step * chunkElems + off) * sizeof(__half),
            dataOff + ((size_t)sendDest * chunkElems + off) * sizeof(__half),
            len * sizeof(__half));
      }
    }
    for (int piece = 0; piece < nsub; piece++) {
      const size_t off = (size_t)piece * pieceElems;
      if (off >= chunkElems) break;
      const size_t len = min(pieceElems, chunkElems - off);
      if (tid == 0) recv.wait(-1);
      gDenseSyncer.sync(gridDim.x);
      addHalf(data + (size_t)recvDest * chunkElems + off,
              data + (size_t)recvDest * chunkElems + off,
              scratch + (size_t)step * chunkElems + off, len, tid, nth);
      gDenseSyncer.sync(gridDim.x);
    }
  }

  // No rank may overwrite a successor's partial chunk before its RS ends.
  if (tid == 0) {
    recv.signal();
    send.wait(-1);
  }
  gDenseSyncer.sync(gridDim.x);
  __threadfence_system();
  gDenseSyncer.sync(gridDim.x);

  for (int step = 0; step < NRANKS - 1; step++) {
    int const sendDest = ringRank(pos - step);
    const size_t off = (size_t)sendDest * chunkBytes;
    if (tid == 0) {
      send.putWithSignal(dataOff + off, dataOff + off, chunkBytes);
      recv.wait(-1);
    }
    gDenseSyncer.sync(gridDim.x);
  }
  if (tid == 0) send.flush(-1);
  gDenseSyncer.sync(gridDim.x);
}

__global__ void compareHalf(__half const* a, __half const* b, size_t n,
                            unsigned* maxDiffBits, unsigned* maxRefBits,
                            unsigned long long* bad, float atol) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  float md = 0.0f, mr = 0.0f;
  unsigned long long nb = 0;
  for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
    float x = __half2float(a[i]);
    float y = __half2float(b[i]);
    float d = fabsf(x - y);
    md = fmaxf(md, d);
    mr = fmaxf(mr, fabsf(x));
    nb += d > atol + 0.01f * fabsf(x);
  }
  atomicMax(maxDiffBits, __float_as_uint(md));
  atomicMax(maxRefBits, __float_as_uint(mr));
  if (nb) atomicAdd(bad, nb);
}

struct CheckResult {
  float maxAbs;
  float maxRef;
  unsigned long long bad;
};

static CheckResult compare(__half const* ref, __half const* got, size_t n) {
  unsigned *dDiff = nullptr, *dRef = nullptr;
  unsigned long long* dBad = nullptr;
  CUDACHECK(cudaMalloc(&dDiff, sizeof(unsigned)));
  CUDACHECK(cudaMalloc(&dRef, sizeof(unsigned)));
  CUDACHECK(cudaMalloc(&dBad, sizeof(unsigned long long)));
  CUDACHECK(cudaMemset(dDiff, 0, sizeof(unsigned)));
  CUDACHECK(cudaMemset(dRef, 0, sizeof(unsigned)));
  CUDACHECK(cudaMemset(dBad, 0, sizeof(unsigned long long)));
  compareHalf<<<256, 256>>>(ref, got, n, dDiff, dRef, dBad, 0.05f);
  unsigned hDiff = 0, hRef = 0;
  unsigned long long hBad = 0;
  CUDACHECK(
      cudaMemcpy(&hDiff, dDiff, sizeof(unsigned), cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(&hRef, dRef, sizeof(unsigned), cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(&hBad, dBad, sizeof(unsigned long long),
                       cudaMemcpyDeviceToHost));
  CUDACHECK(cudaFree(dDiff));
  CUDACHECK(cudaFree(dRef));
  CUDACHECK(cudaFree(dBad));
  float fDiff, fRef;
  std::memcpy(&fDiff, &hDiff, sizeof(float));
  std::memcpy(&fRef, &hRef, sizeof(float));
  return {fDiff, fRef, hBad};
}

template <class F>
static double timeOne(F&& fn, std::shared_ptr<mscclpp::TcpBootstrap> bootstrap,
                      cudaStream_t stream) {
  bootstrap->barrier();
  auto t0 = std::chrono::steady_clock::now();
  fn();
  CUDACHECK(cudaStreamSynchronize(stream));
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

struct BenchRow {
  std::string name;
  std::function<void()> launch;
  std::vector<double> samples;
};

static std::vector<double> benchRows(
    char const* title, std::vector<BenchRow>& rows, int baseline, int rank,
    Cfg const& cfg, std::shared_ptr<mscclpp::TcpBootstrap> bootstrap,
    cudaStream_t stream) {
  for (int w = 0; w < cfg.warmup; w++) {
    for (size_t j = 0; j < rows.size(); j++) {
      size_t i = (j + w) % rows.size();
      bootstrap->barrier();
      rows[i].launch();
      CUDACHECK(cudaStreamSynchronize(stream));
    }
  }
  for (int r = 0; r < cfg.iters; r++) {
    for (size_t j = 0; j < rows.size(); j++) {
      size_t i = (j + r) % rows.size();
      rows[i].samples.push_back(timeOne(rows[i].launch, bootstrap, stream));
    }
  }

  std::vector<double> worst(rows.size());
  for (size_t i = 0; i < rows.size(); i++) {
    auto x = rows[i].samples;
    std::sort(x.begin(), x.end());
    double med = x[x.size() / 2];
    std::vector<double> all(NRANKS);
    all[rank] = med;
    bootstrap->allGather(all.data(), sizeof(double));
    worst[i] = *std::max_element(all.begin(), all.end());
  }
  if (rank == 0) {
    std::printf("\n# %s\n", title);
    std::printf("# %-27s %10s %10s\n", "variant", "ms", "speedup");
    for (size_t i = 0; i < rows.size(); i++)
      std::printf("  %-27s %10.3f %9.3fx\n", rows[i].name.c_str(), worst[i],
                  worst[baseline] / worst[i]);
    std::fflush(stdout);
  }
  bootstrap->barrier();
  return worst;
}

static void runRank(int rank, mscclpp::UniqueId id, Cfg const& cfg) {
  CUDACHECK(cudaSetDevice(rank));
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, NRANKS);
  bootstrap->initialize(id);
  mscclpp::Communicator comm(bootstrap);

  if (cfg.agM % NRANKS || cfg.agN % NRANKS || cfg.rsM % NRANKS ||
      cfg.rsK % NRANKS || cfg.rsN % 2 || (cfg.rsM / NRANKS) % cfg.rsStripes) {
    if (rank == 0)
      std::fprintf(
          stderr,
          "M/N/K must divide TP=4 and RS local-M must divide stripes\n");
    std::exit(1);
  }

  int const agLocalM = cfg.agM / NRANKS;
  int const agLocalN = cfg.agN / NRANKS;
  int const rsLocalM = cfg.rsM / NRANKS;
  int const rsLocalK = cfg.rsK / NRANKS;
  int const q = rsLocalM / cfg.rsStripes;
  const size_t agRankBytes = (size_t)agLocalM * cfg.agK * sizeof(__half);
  const size_t agBytes = (size_t)cfg.agM * cfg.agK * sizeof(__half);
  const size_t rsChunkBytes = (size_t)q * cfg.rsN * sizeof(__half);
  const size_t rsHalfChunkBytes = (size_t)rsLocalM * cfg.rsN * sizeof(__half);
  const size_t rsFullBytes = (size_t)cfg.rsM * cfg.rsN * sizeof(__half);
  // More workers sharply reduce the latency of model-sized (<=4 MiB) shards;
  // beyond that point their SM and memory-bandwidth interference with cuBLAS
  // outweighs the faster reduction.  This is the same shape-specialisation
  // principle as Flux's per-shape kernel table, applied to the RS consumer.
  int const tunedReadyRsBlocks =
      cfg.readyRsBlocks > 0
          ? cfg.readyRsBlocks
          : (rsChunkBytes <= (4u << 20) ? 8 : READY_RS_BLOCKS);
  int const tunedReadyRsThreads =
      cfg.readyRsThreads > 0 ? cfg.readyRsThreads
                             : (rsChunkBytes <= (4u << 20) ? 512 : 256);

  size_t agOff = 0;
  size_t next = alignUp(agBytes);
  RsSlot slots[2];
  for (int p = 0; p < 2; p++) {
    slots[p].data = next;
    next += std::max(4 * rsChunkBytes, 2 * rsHalfChunkBytes);
    slots[p].phase1 = next;
    next += std::max(3 * rsChunkBytes, rsHalfChunkBytes);
    slots[p].temp = next;
    next += std::max(2 * rsChunkBytes, rsHalfChunkBytes);
    slots[p].phase2 = next;
    next += std::max(rsChunkBytes, rsHalfChunkBytes);
    next = alignUp(next);
  }
  const size_t rsReadyDataOff = next;
  next += rsFullBytes;
  const size_t rsReadyScratchOff = next;
  next += 3 * rsHalfChunkBytes;
  int const arFlagTileM =
      cfg.rsM <= 128 ? 32 : (cfg.rsM <= 256 ? 64 : DENSE_BM);
  const size_t arFlagCount =
      (size_t)((cfg.rsM + arFlagTileM - 1) / arFlagTileM) *
      ((cfg.rsN + DENSE_BN - 1) / DENSE_BN);
  const size_t arFlagsOff = next;
  next += alignUp(arFlagCount * sizeof(uint32_t));
  const size_t arFusedScratchOff = next;
  next += rsFullBytes;
  next = alignUp(next);
  const size_t commBytes = next;
  if (commBytes >= (1ULL << 32)) {
    if (rank == 0)
      std::fprintf(stderr,
                   "registered buffer exceeds proxy's 32-bit offset field\n");
    std::exit(1);
  }

  char* cbuf = nullptr;
  CUDACHECK(cudaMalloc(&cbuf, commBytes));
  CUDACHECK(cudaMemset(cbuf, 0, commBytes));
  auto reg = comm.registerMemory(cbuf, commBytes, mscclpp::Transport::CudaIpc);

  int const peers[2] = {rank ^ 1, rank ^ 2};
  std::shared_future<mscclpp::Connection> connF[2];
  std::shared_future<mscclpp::RegisteredMemory> memF[2];
  for (int i = 0; i < 2; i++) {
    comm.sendMemory(reg, peers[i], 0);
    memF[i] = comm.recvMemory(peers[i], 0);
    connF[i] = comm.connect(mscclpp::Transport::CudaIpc, peers[i]);
  }
  mscclpp::Connection conns[2] = {connF[0].get(), connF[1].get()};
  std::shared_future<mscclpp::Semaphore> memSemF[2] = {
      comm.buildSemaphore(conns[0], peers[0]),
      comm.buildSemaphore(conns[1], peers[1])};
  mscclpp::MemoryChannel memIntraObj(memSemF[0].get(), memF[0].get(), reg,
                                     cbuf);
  mscclpp::MemoryChannel memCrossObj(memSemF[1].get(), memF[1].get(), reg,
                                     cbuf);
  MemChan memIntra = memIntraObj.deviceHandle();
  MemChan memCross = memCrossObj.deviceHandle();
  auto proxy = std::make_shared<mscclpp::ProxyService>();
  mscclpp::SemaphoreId sid[2];
  mscclpp::MemoryId remote[2];
  mscclpp::MemoryId local = proxy->addMemory(reg);
  for (int i = 0; i < 2; i++) {
    sid[i] = proxy->buildAndAddSemaphore(comm, conns[i]);
    remote[i] = proxy->addMemory(memF[i].get());
  }
  mscclpp::PortChannel pcIntraObj =
      proxy->portChannel(sid[0], remote[0], local);
  mscclpp::PortChannel pcCrossObj =
      proxy->portChannel(sid[1], remote[1], local);
  PortChan pcIntra = pcIntraObj.deviceHandle();
  PortChan pcCross = pcCrossObj.deviceHandle();
  proxy->startProxy();

  ncclUniqueId nid{};
  std::vector<ncclUniqueId> nids(NRANKS);
  if (rank == 0) NCCLCHECK(ncclGetUniqueId(&nid));
  nids[rank] = nid;
  bootstrap->allGather(nids.data(), sizeof(ncclUniqueId));
  ncclComm_t nccl;
  NCCLCHECK(ncclCommInitRank(&nccl, NRANKS, nids[0], rank));

  cudaStream_t mainStream, commStream;
  CUDACHECK(cudaStreamCreateWithFlags(&mainStream, cudaStreamNonBlocking));
  int leastPriority = 0, greatestPriority = 0;
  CUDACHECK(
      cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
  CUDACHECK(cudaStreamCreateWithPriority(&commStream, cudaStreamNonBlocking,
                                         greatestPriority));
  cublasHandle_t cub;
  CUBLASCHECK(cublasCreate(&cub));
  CUBLASCHECK(cublasSetStream(cub, mainStream));
  CUBLASCHECK(cublasSetMathMode(cub, CUBLAS_TENSOR_OP_MATH));

  __half *agB = nullptr, *agOut = nullptr, *agRef = nullptr;
  __half *rsA = nullptr, *rsB = nullptr, *rsPartial = nullptr, *rsOut = nullptr,
         *rsRef = nullptr;
  __half *arRef = nullptr, *arOut = nullptr;
  CUDACHECK(cudaMalloc(&agB, (size_t)cfg.agK * agLocalN * sizeof(__half)));
  CUDACHECK(cudaMalloc(&agOut, (size_t)cfg.agM * agLocalN * sizeof(__half)));
  CUDACHECK(cudaMalloc(&agRef, (size_t)cfg.agM * agLocalN * sizeof(__half)));
  CUDACHECK(cudaMalloc(&rsA, (size_t)cfg.rsM * rsLocalK * sizeof(__half)));
  CUDACHECK(cudaMalloc(&rsB, (size_t)rsLocalK * cfg.rsN * sizeof(__half)));
  CUDACHECK(cudaMalloc(&rsPartial, (size_t)cfg.rsM * cfg.rsN * sizeof(__half)));
  CUDACHECK(cudaMalloc(&rsOut, (size_t)rsLocalM * cfg.rsN * sizeof(__half)));
  CUDACHECK(cudaMalloc(&rsRef, (size_t)rsLocalM * cfg.rsN * sizeof(__half)));
  CUDACHECK(cudaMalloc(&arRef, (size_t)cfg.rsM * cfg.rsN * sizeof(__half)));
  CUDACHECK(cudaMalloc(&arOut, (size_t)cfg.rsM * cfg.rsN * sizeof(__half)));

  fillHalf<<<256, 256>>>(reinterpret_cast<__half*>(cbuf + agOff) +
                             (size_t)rank * agLocalM * cfg.agK,
                         (size_t)agLocalM * cfg.agK, 11 + rank, 0.0625f);
  fillHalf<<<256, 256>>>(agB, (size_t)cfg.agK * agLocalN, 101 + rank,
                         0.0078125f);
  fillHalf<<<256, 256>>>(rsA, (size_t)cfg.rsM * rsLocalK, 211 + rank, 0.03125f);
  fillHalf<<<256, 256>>>(rsB, (size_t)rsLocalK * cfg.rsN, 307 + rank,
                         0.0078125f);
  CUDACHECK(cudaDeviceSynchronize());

  auto cublasGemm = [&](int m, int n, int k, __half const* A, __half const* B,
                        __half* C) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLASCHECK(cublasGemmEx(cub, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, B,
                             CUDA_R_16F, k, A, CUDA_R_16F, k, &beta, C,
                             CUDA_R_16F, n, CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  };

  int nSM = 0;
  CUDACHECK(cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, rank));
  int const smemPlain = (int)sizeof(typename DensePlain::SharedStorage);
  int const smemFused = (int)sizeof(typename AgFused::SharedStorage);
  int const smemSmall = (int)sizeof(typename SmallDensePlain::SharedStorage);
  int const smemArFused = (int)sizeof(typename ArFused::SharedStorage);
  int const smemSmallArFused =
      (int)sizeof(typename SmallArFused::SharedStorage);
  int const smemTiny = (int)sizeof(typename TinyDensePlain::SharedStorage);
  int const smemTinyArFused = (int)sizeof(typename TinyArFused::SharedStorage);
  CUDACHECK(cudaFuncSetAttribute(uccl_lite::gatedGroupedKernel<DensePlain>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smemPlain));
  CUDACHECK(cudaFuncSetAttribute(uccl_lite::gatedGroupedKernel<AgFused>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smemFused));
  CUDACHECK(cudaFuncSetAttribute(uccl_lite::gatedGroupedKernel<SmallDensePlain>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smemSmall));
  CUDACHECK(cudaFuncSetAttribute(uccl_lite::gatedGroupedKernel<ArFused>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smemArFused));
  CUDACHECK(cudaFuncSetAttribute(uccl_lite::gatedGroupedKernel<SmallArFused>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smemSmallArFused));
  CUDACHECK(cudaFuncSetAttribute(uccl_lite::gatedGroupedKernel<TinyDensePlain>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smemTiny));
  CUDACHECK(cudaFuncSetAttribute(uccl_lite::gatedGroupedKernel<TinyArFused>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smemTinyArFused));

  // AG: one unsplit GEMM.  Only the M-tile gate varies by source rank.
  cutlass::gemm::GemmCoord* dAgProb = nullptr;
  Elem **dAgA = nullptr, **dAgB = nullptr, **dAgC = nullptr;
  int64_t *dAgLdA = nullptr, *dAgLdB = nullptr, *dAgLdC = nullptr;
  int *dAgTileGroup = nullptr, *dAgSmallTileGroup = nullptr,
      *dAgTileOff = nullptr;
  uint32_t* dAgFlags = nullptr;
  CUDACHECK(cudaMalloc(&dAgProb, sizeof(cutlass::gemm::GemmCoord)));
  CUDACHECK(cudaMalloc(&dAgA, sizeof(void*)));
  CUDACHECK(cudaMalloc(&dAgB, sizeof(void*)));
  CUDACHECK(cudaMalloc(&dAgC, sizeof(void*)));
  CUDACHECK(cudaMalloc(&dAgLdA, sizeof(int64_t)));
  CUDACHECK(cudaMalloc(&dAgLdB, sizeof(int64_t)));
  CUDACHECK(cudaMalloc(&dAgLdC, sizeof(int64_t)));
  CUDACHECK(cudaMalloc(&dAgTileOff, sizeof(int)));
  CUDACHECK(cudaMalloc(&dAgFlags, 3 * sizeof(uint32_t)));
  cutlass::gemm::GemmCoord agProb(cfg.agM, agLocalN, cfg.agK);
  Elem* hAgA = reinterpret_cast<Elem*>(cbuf + agOff);
  Elem* hAgB = reinterpret_cast<Elem*>(agB);
  Elem* hAgC = reinterpret_cast<Elem*>(agOut);
  int64_t agLdA = cfg.agK, agLdB = cfg.agK, agLdC = agLocalN;
  int zero = 0;
  CUDACHECK(
      cudaMemcpy(dAgProb, &agProb, sizeof(agProb), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(dAgA, &hAgA, sizeof(void*), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(dAgB, &hAgB, sizeof(void*), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(dAgC, &hAgC, sizeof(void*), cudaMemcpyHostToDevice));
  CUDACHECK(
      cudaMemcpy(dAgLdA, &agLdA, sizeof(int64_t), cudaMemcpyHostToDevice));
  CUDACHECK(
      cudaMemcpy(dAgLdB, &agLdB, sizeof(int64_t), cudaMemcpyHostToDevice));
  CUDACHECK(
      cudaMemcpy(dAgLdC, &agLdC, sizeof(int64_t), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(dAgTileOff, &zero, sizeof(int), cudaMemcpyHostToDevice));
  int const agMTiles = (cfg.agM + DENSE_BM - 1) / DENSE_BM;
  std::vector<int> agTileGroup(agMTiles);
  for (int mt = 0; mt < agMTiles; mt++) {
    int first = mt * DENSE_BM;
    int last = std::min(cfg.agM, first + DENSE_BM) - 1;
    int g = 0;
    for (int src = first / agLocalM; src <= last / agLocalM; src++)
      g = std::max(g, src == rank ? 0 : src == (rank ^ 1) ? 1 : 2);
    agTileGroup[mt] = g;
  }
  CUDACHECK(cudaMalloc(&dAgTileGroup, agMTiles * sizeof(int)));
  CUDACHECK(cudaMemcpy(dAgTileGroup, agTileGroup.data(), agMTiles * sizeof(int),
                       cudaMemcpyHostToDevice));
  int const agSmallMTiles = (cfg.agM + 63) / 64;
  std::vector<int> agSmallTileGroup(agSmallMTiles);
  for (int mt = 0; mt < agSmallMTiles; mt++) {
    int first = mt * 64;
    int last = std::min(cfg.agM, first + 64) - 1;
    int g = 0;
    for (int src = first / agLocalM; src <= last / agLocalM; src++)
      g = std::max(g, src == rank ? 0 : src == (rank ^ 1) ? 1 : 2);
    agSmallTileGroup[mt] = g;
  }
  CUDACHECK(cudaMalloc(&dAgSmallTileGroup, agSmallMTiles * sizeof(int)));
  CUDACHECK(cudaMemcpy(dAgSmallTileGroup, agSmallTileGroup.data(),
                       agSmallMTiles * sizeof(int), cudaMemcpyHostToDevice));
  int const agSufficient = DenseGemm::sufficient(&agProb, 1);
  int agGrid = std::min(agSufficient, nSM);
  // Flux's useful small-M idea is to keep one GEMM launch and let individual
  // M tiles wait for the AG phase they consume.  When communication runs in a
  // separate high-priority stream, leave one SM unoccupied by the persistent
  // GEMM grid so the tiny dispatcher is guaranteed to run.
  int agOverlapGrid = std::min(agSufficient, std::max(1, nSM - 1));
  typename DenseKernel::Arguments agArgs(
      dAgProb, 1, agGrid,
      typename DenseKernel::EpilogueOutputOp::Params(1.0f, 0.0f), dAgA, dAgB,
      dAgC, dAgC, dAgLdA, dAgLdB, dAgLdC, dAgLdC, &agProb);
  CUTLASSCHECK(DenseGemm::can_implement(agArgs));
  typename DensePlain::Params agPlain(
      typename DenseKernel::Params(agArgs, nullptr, 0), nullptr, nullptr);
  typename DenseKernel::Arguments agOverlapArgs(
      dAgProb, 1, agOverlapGrid,
      typename DenseKernel::EpilogueOutputOp::Params(1.0f, 0.0f), dAgA, dAgB,
      dAgC, dAgC, dAgLdA, dAgLdB, dAgLdC, dAgLdC, &agProb);
  CUTLASSCHECK(DenseGemm::can_implement(agOverlapArgs));
  typename DensePlain::Params agOverlap(
      typename DenseKernel::Params(agOverlapArgs, nullptr, 0), dAgFlags,
      nullptr);
  agOverlap.tile_group = dAgTileGroup;
  agOverlap.tile_group_off = dAgTileOff;
  agOverlap.tile_m_offset = (rank * agLocalM) / DENSE_BM;
  int const agSmallSufficient = SmallDenseGemm::sufficient(&agProb, 1);
  int const agSmallGrid = std::min(agSmallSufficient, std::max(1, nSM - 1));
  typename SmallDenseKernel::Arguments agSmallArgs(
      dAgProb, 1, agSmallGrid,
      typename SmallDenseKernel::EpilogueOutputOp::Params(1.0f, 0.0f), dAgA,
      dAgB, dAgC, dAgC, dAgLdA, dAgLdB, dAgLdC, dAgLdC, &agProb);
  CUTLASSCHECK(SmallDenseGemm::can_implement(agSmallArgs));
  typename SmallDensePlain::Params agSmall(
      typename SmallDenseKernel::Params(agSmallArgs, nullptr, 0), dAgFlags,
      nullptr);
  agSmall.tile_group = dAgSmallTileGroup;
  agSmall.tile_group_off = dAgTileOff;
  agSmall.tile_m_offset = (rank * agLocalM) / 64;
  AgDispatch agDisp{pcIntra, pcCross, rank, agOff, agRankBytes, dAgFlags};
  typename AgFused::Params agFused(
      typename DenseKernel::Params(agArgs, nullptr, 0), dAgFlags, nullptr,
      nullptr, agDisp);
  agFused.tile_group = dAgTileGroup;
  agFused.tile_group_off = dAgTileOff;
  agFused.tile_m_offset = (rank * agLocalM) / DENSE_BM;

  auto launchAgNccl = [&]() {
    NCCLCHECK(ncclAllGather(cbuf + agOff + (size_t)rank * agRankBytes,
                            cbuf + agOff, (size_t)agLocalM * cfg.agK, ncclHalf,
                            nccl, mainStream));
    cublasGemm(cfg.agM, agLocalN, cfg.agK,
               reinterpret_cast<__half*>(cbuf + agOff), agB, agOut);
  };
  auto launchAgHierCublas = [&]() {
    clearAgFlags<<<1, 32, 0, mainStream>>>(dAgFlags);
    agOnly<<<1, 32, 0, mainStream>>>(agDisp);
    cublasGemm(cfg.agM, agLocalN, cfg.agK,
               reinterpret_cast<__half*>(cbuf + agOff), agB, agOut);
  };
  auto launchAgHierCutlass = [&]() {
    clearAgFlags<<<1, 32, 0, mainStream>>>(dAgFlags);
    agOnly<<<1, 32, 0, mainStream>>>(agDisp);
    uccl_lite::gatedGroupedKernel<DensePlain>
        <<<agGrid, DensePlain::kThreadCount, smemPlain, mainStream>>>(agPlain);
  };
  auto launchAgFused = [&]() {
    clearAgFlags<<<1, 32, 0, mainStream>>>(dAgFlags);
    uccl_lite::gatedGroupedKernel<AgFused>
        <<<agGrid + 1, AgFused::kThreadCount, smemFused, mainStream>>>(agFused);
  };
  cudaEvent_t agFlagsCleared;
  CUDACHECK(cudaEventCreateWithFlags(&agFlagsCleared, cudaEventDisableTiming));
  auto launchAgSplitStream = [&]() {
    clearAgFlags<<<1, 32, 0, mainStream>>>(dAgFlags);
    CUDACHECK(cudaEventRecord(agFlagsCleared, mainStream));
    CUDACHECK(cudaStreamWaitEvent(commStream, agFlagsCleared, 0));
    agOnly<<<1, 32, 0, commStream>>>(agDisp);
    uccl_lite::gatedGroupedKernel<DensePlain>
        <<<agOverlapGrid, DensePlain::kThreadCount, smemPlain, mainStream>>>(
            agOverlap);
  };
  auto launchAgSmallSplitStream = [&]() {
    clearAgFlags<<<1, 32, 0, mainStream>>>(dAgFlags);
    CUDACHECK(cudaEventRecord(agFlagsCleared, mainStream));
    CUDACHECK(cudaStreamWaitEvent(commStream, agFlagsCleared, 0));
    agOnly<<<1, 32, 0, commStream>>>(agDisp);
    uccl_lite::gatedGroupedKernel<SmallDensePlain>
        <<<agSmallGrid, SmallDensePlain::kThreadCount, smemSmall, mainStream>>>(
            agSmall);
  };
  auto agChunkGemm = [&](int firstSrc, int nsrc) {
    cublasGemm(nsrc * agLocalM, agLocalN, cfg.agK,
               reinterpret_cast<__half*>(cbuf + agOff) +
                   (size_t)firstSrc * agLocalM * cfg.agK,
               agB, agOut + (size_t)firstSrc * agLocalM * agLocalN);
  };
  auto launchAgPipedCublas = [&]() {
    // No resident communication waiter is allowed to coexist with cuBLAS: that
    // scheduling cycle is the deadlock documented by fused_moe.cu.  Each tiny
    // state-transition kernel submits a CE transfer and exits; the transfer
    // itself remains asynchronous and overlaps the following GEMM.
    agIssueIntra<<<1, 32, 0, mainStream>>>(pcIntra, rank, agOff, agRankBytes);
    agChunkGemm(rank, 1);
    agWaitIntraIssueCross<<<1, 32, 0, mainStream>>>(pcIntra, pcCross, rank,
                                                    agOff, agRankBytes);
    agChunkGemm(rank ^ 1, 1);
    agWaitCross<<<1, 32, 0, mainStream>>>(pcCross);
    agChunkGemm((rank & 2) ^ 2, 2);
  };
  auto launchAgCublasOnly = [&]() {
    cublasGemm(cfg.agM, agLocalN, cfg.agK,
               reinterpret_cast<__half*>(cbuf + agOff), agB, agOut);
  };
  auto launchAgCutlassOnly = [&]() {
    uccl_lite::gatedGroupedKernel<DensePlain>
        <<<agGrid, DensePlain::kThreadCount, smemPlain, mainStream>>>(agPlain);
  };

  if (cfg.check) {
    launchAgNccl();
    CUDACHECK(cudaStreamSynchronize(mainStream));
    CUDACHECK(cudaMemcpyAsync(agRef, agOut,
                              (size_t)cfg.agM * agLocalN * sizeof(__half),
                              cudaMemcpyDeviceToDevice, mainStream));
    CUDACHECK(cudaStreamSynchronize(mainStream));
    bootstrap->barrier();
    launchAgPipedCublas();
    CUDACHECK(cudaStreamSynchronize(mainStream));
    CheckResult c = compare(agRef, agOut, (size_t)cfg.agM * agLocalN);
    std::vector<unsigned long long> bad(NRANKS);
    bad[rank] = c.bad;
    bootstrap->allGather(bad.data(), sizeof(unsigned long long));
    if (rank == 0)
      std::printf("# AG correctness: max_abs %.5g max_ref %.5g bad %llu\n",
                  c.maxAbs, c.maxRef,
                  *std::max_element(bad.begin(), bad.end()));
    bootstrap->barrier();
    launchAgSplitStream();
    CUDACHECK(cudaStreamSynchronize(mainStream));
    c = compare(agRef, agOut, (size_t)cfg.agM * agLocalN);
    bad[rank] = c.bad;
    bootstrap->allGather(bad.data(), sizeof(unsigned long long));
    if (rank == 0)
      std::printf(
          "# AG split-stream correctness: max_abs %.5g max_ref %.5g bad %llu\n",
          c.maxAbs, c.maxRef, *std::max_element(bad.begin(), bad.end()));
    bootstrap->barrier();
    if (cfg.agM <= 128) {
      launchAgSmallSplitStream();
      CUDACHECK(cudaStreamSynchronize(mainStream));
      c = compare(agRef, agOut, (size_t)cfg.agM * agLocalN);
      bad[rank] = c.bad;
      bootstrap->allGather(bad.data(), sizeof(unsigned long long));
      if (rank == 0)
        std::printf(
            "# AG small-tile correctness: max_abs %.5g max_ref %.5g bad %llu\n",
            c.maxAbs, c.maxRef, *std::max_element(bad.begin(), bad.end()));
      bootstrap->barrier();
    }
  }
  std::vector<BenchRow> agRows = {
      {"gemm-only-cuBLAS", launchAgCublasOnly, {}},
      {"gemm-only-CUTLASS", launchAgCutlassOnly, {}},
      {"NCCL-AG+cuBLAS", launchAgNccl, {}},
      {"hier-AG+cuBLAS", launchAgHierCublas, {}},
      {"hier-AG+CUTLASS", launchAgHierCutlass, {}},
      {"fused-AG+GEMM", launchAgFused, {}},
      {"split-stream-128x256", launchAgSplitStream, {}},
      {"split-stream-64x256", launchAgSmallSplitStream, {}},
      {"piped-AG+cuBLAS", launchAgPipedCublas, {}},
  };
  char agTitle[160];
  std::snprintf(agTitle, sizeof(agTitle),
                "AG+GEMM M=%d N=%d K=%d TP=4 (local N=%d)", cfg.agM, cfg.agN,
                cfg.agK, agLocalN);
  std::vector<double> agMs =
      benchRows(agTitle, agRows, 2, rank, cfg, bootstrap, mainStream);
  if (rank == 0) {
    int const selected = cfg.agM <= 128   ? 7
                         : cfg.agM <= 256 ? 6
                         : cfg.agM < 1024 ? 3
                                          : 8;
    std::printf("# AG dispatch: %s, %.3f ms, %.3fx vs NCCL baseline\n",
                agRows[selected].name.c_str(), agMs[selected],
                agMs[2] / agMs[selected]);
    std::fflush(stdout);
  }
  bootstrap->barrier();

  // RS CUTLASS descriptors.  Four q-row problems per stripe write directly to
  // the rank-destination chunks consumed by gemmRsStripe.
  int const nRsProb = cfg.rsStripes * NRANKS;
  cutlass::gemm::GemmCoord* dRsProb = nullptr;
  Elem **dRsA = nullptr, **dRsB = nullptr, **dRsC = nullptr;
  int64_t *dRsLdA = nullptr, *dRsLdB = nullptr, *dRsLdC = nullptr;
  CUDACHECK(cudaMalloc(&dRsProb, nRsProb * sizeof(cutlass::gemm::GemmCoord)));
  CUDACHECK(cudaMalloc(&dRsA, nRsProb * sizeof(void*)));
  CUDACHECK(cudaMalloc(&dRsB, nRsProb * sizeof(void*)));
  CUDACHECK(cudaMalloc(&dRsC, nRsProb * sizeof(void*)));
  CUDACHECK(cudaMalloc(&dRsLdA, nRsProb * sizeof(int64_t)));
  CUDACHECK(cudaMalloc(&dRsLdB, nRsProb * sizeof(int64_t)));
  CUDACHECK(cudaMalloc(&dRsLdC, nRsProb * sizeof(int64_t)));
  std::vector<cutlass::gemm::GemmCoord> rsProb(
      nRsProb, cutlass::gemm::GemmCoord(q, cfg.rsN, rsLocalK));
  std::vector<Elem*> rsAp(nRsProb), rsBp(nRsProb), rsCp(nRsProb);
  std::vector<int64_t> rsLdA(nRsProb, rsLocalK), rsLdB(nRsProb, rsLocalK),
      rsLdC(nRsProb, cfg.rsN);
  for (int s = 0; s < cfg.rsStripes; s++) {
    int p = s & 1;
    for (int d = 0; d < NRANKS; d++) {
      int i = s * NRANKS + d;
      rsAp[i] = reinterpret_cast<Elem*>(rsA) +
                ((size_t)d * rsLocalM + (size_t)s * q) * rsLocalK;
      rsBp[i] = reinterpret_cast<Elem*>(rsB);
      rsCp[i] = reinterpret_cast<Elem*>(cbuf + slots[p].data) +
                (size_t)d * q * cfg.rsN;
    }
  }
  CUDACHECK(cudaMemcpy(dRsProb, rsProb.data(), nRsProb * sizeof(rsProb[0]),
                       cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(dRsA, rsAp.data(), nRsProb * sizeof(void*),
                       cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(dRsB, rsBp.data(), nRsProb * sizeof(void*),
                       cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(dRsC, rsCp.data(), nRsProb * sizeof(void*),
                       cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(dRsLdA, rsLdA.data(), nRsProb * sizeof(int64_t),
                       cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(dRsLdB, rsLdB.data(), nRsProb * sizeof(int64_t),
                       cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(dRsLdC, rsLdC.data(), nRsProb * sizeof(int64_t),
                       cudaMemcpyHostToDevice));
  int rsGrid = std::min(DenseGemm::sufficient(rsProb.data(), NRANKS), nSM);
  std::vector<typename DensePlain::Params> rsParams;
  for (int s = 0; s < cfg.rsStripes; s++) {
    typename DenseKernel::Arguments a(
        dRsProb + s * NRANKS, NRANKS, rsGrid,
        typename DenseKernel::EpilogueOutputOp::Params(1.0f, 0.0f),
        dRsA + s * NRANKS, dRsB + s * NRANKS, dRsC + s * NRANKS,
        dRsC + s * NRANKS, dRsLdA + s * NRANKS, dRsLdB + s * NRANKS,
        dRsLdC + s * NRANKS, dRsLdC + s * NRANKS, rsProb.data() + s * NRANKS);
    CUTLASSCHECK(DenseGemm::can_implement(a));
    rsParams.emplace_back(typename DenseKernel::Params(a, nullptr, 0), nullptr,
                          nullptr);
  }

  // One unsplit GEMM for the fused AR path.  CUTLASS epilogues publish one flag
  // per (M,N) tile; the resident hierarchy dispatcher starts a row stripe as
  // soon as the half it must send is visible to PCIe.
  cutlass::gemm::GemmCoord* dArProb = nullptr;
  Elem **dArA = nullptr, **dArB = nullptr, **dArC = nullptr;
  int64_t *dArLdA = nullptr, *dArLdB = nullptr, *dArLdC = nullptr;
  CUDACHECK(cudaMalloc(&dArProb, sizeof(cutlass::gemm::GemmCoord)));
  CUDACHECK(cudaMalloc(&dArA, sizeof(void*)));
  CUDACHECK(cudaMalloc(&dArB, sizeof(void*)));
  CUDACHECK(cudaMalloc(&dArC, sizeof(void*)));
  CUDACHECK(cudaMalloc(&dArLdA, sizeof(int64_t)));
  CUDACHECK(cudaMalloc(&dArLdB, sizeof(int64_t)));
  CUDACHECK(cudaMalloc(&dArLdC, sizeof(int64_t)));
  cutlass::gemm::GemmCoord arProb(cfg.rsM, cfg.rsN, rsLocalK);
  Elem* hArA = reinterpret_cast<Elem*>(rsA);
  Elem* hArB = reinterpret_cast<Elem*>(rsB);
  Elem* hArC = reinterpret_cast<Elem*>(cbuf + rsReadyDataOff);
  int64_t arLdA = rsLocalK, arLdB = rsLocalK, arLdC = cfg.rsN;
  CUDACHECK(
      cudaMemcpy(dArProb, &arProb, sizeof(arProb), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(dArA, &hArA, sizeof(void*), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(dArB, &hArB, sizeof(void*), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(dArC, &hArC, sizeof(void*), cudaMemcpyHostToDevice));
  CUDACHECK(
      cudaMemcpy(dArLdA, &arLdA, sizeof(int64_t), cudaMemcpyHostToDevice));
  CUDACHECK(
      cudaMemcpy(dArLdB, &arLdB, sizeof(int64_t), cudaMemcpyHostToDevice));
  CUDACHECK(
      cudaMemcpy(dArLdC, &arLdC, sizeof(int64_t), cudaMemcpyHostToDevice));

  bool const arUseTiny = cfg.rsM <= 128;
  bool const arUseSmall = !arUseTiny && cfg.rsM <= 256;
  int const arTileM = arUseTiny ? 32 : (arUseSmall ? 64 : DENSE_BM);
  int const arMTiles = (cfg.rsM + arTileM - 1) / arTileM;
  int const arNTiles = (cfg.rsN + DENSE_BN - 1) / DENSE_BN;
  int const arActiveBlocks =
      std::min(cfg.fusedArBlocks, ArHierDispatch::kBlocks);
  const size_t arTileBytes = (size_t)arTileM * cfg.rsN * sizeof(__half);
  int const arAutoTiles = std::max(
      1,
      std::min(arMTiles, (int)((2u << 20) / std::max<size_t>(1, arTileBytes))));
  int const arTilesPerChunk = cfg.fusedArTiles > 0
                                  ? std::min(cfg.fusedArTiles, arMTiles)
                                  : (cfg.rsM <= 256 ? arMTiles : arAutoTiles);
  uint32_t* dArCompleted = nullptr;
  const size_t arCompletedBytes =
      (size_t)arMTiles * arNTiles * sizeof(uint32_t);
  dArCompleted = reinterpret_cast<uint32_t*>(cbuf + arFlagsOff);

  int const arDenseWorkers = std::min(DenseGemm::sufficient(&arProb, 1),
                                      std::max(1, nSM - arActiveBlocks));
  typename DenseKernel::Arguments arDenseArgs(
      dArProb, 1, arDenseWorkers,
      typename DenseKernel::EpilogueOutputOp::Params(1.0f, 0.0f), dArA, dArB,
      dArC, dArC, dArLdA, dArLdB, dArLdC, dArLdC, &arProb);
  CUTLASSCHECK(DenseGemm::can_implement(arDenseArgs));
  typename DensePlain::Params arDenseProducer(
      typename DenseKernel::Params(arDenseArgs, nullptr, 0), nullptr, nullptr);
  arDenseProducer.completed = dArCompleted;

  int const arSmallWorkerCap =
      cfg.fusedArGemmBlocks > 0 ? cfg.fusedArGemmBlocks : 2 * arNTiles;
  int const arSmallWorkers =
      std::min(SmallDenseGemm::sufficient(&arProb, 1),
               std::max(1, std::min(nSM - arActiveBlocks, arSmallWorkerCap)));
  typename SmallDenseKernel::Arguments arSmallArgs(
      dArProb, 1, arSmallWorkers,
      typename SmallDenseKernel::EpilogueOutputOp::Params(1.0f, 0.0f), dArA,
      dArB, dArC, dArC, dArLdA, dArLdB, dArLdC, dArLdC, &arProb);
  CUTLASSCHECK(SmallDenseGemm::can_implement(arSmallArgs));
  typename SmallDensePlain::Params arSmallProducer(
      typename SmallDenseKernel::Params(arSmallArgs, nullptr, 0), nullptr,
      nullptr);
  arSmallProducer.completed = dArCompleted;

  int const arTinyWorkers = std::min(TinyDenseGemm::sufficient(&arProb, 1),
                                     std::max(1, nSM - arActiveBlocks));
  typename TinyDenseKernel::Arguments arTinyArgs(
      dArProb, 1, arTinyWorkers,
      typename TinyDenseKernel::EpilogueOutputOp::Params(1.0f, 0.0f), dArA,
      dArB, dArC, dArC, dArLdA, dArLdB, dArLdC, dArLdC, &arProb);
  CUTLASSCHECK(TinyDenseGemm::can_implement(arTinyArgs));
  typename TinyDensePlain::Params arTinyProducer(
      typename TinyDenseKernel::Params(arTinyArgs, nullptr, 0), nullptr,
      nullptr);
  arTinyProducer.completed = dArCompleted;

  ArHierDispatch arDispatch{memIntra,       memCross,          rank,
                            rsReadyDataOff, arFusedScratchOff, cfg.rsM,
                            cfg.rsN,        arTileM,           arMTiles,
                            arNTiles,       arTilesPerChunk,   arActiveBlocks,
                            dArCompleted};
  typename ArFused::Params arDenseFused(
      typename DenseKernel::Params(arDenseArgs, nullptr, 0), nullptr, nullptr,
      nullptr, arDispatch);
  arDenseFused.completed = dArCompleted;
  typename SmallArFused::Params arSmallFused(
      typename SmallDenseKernel::Params(arSmallArgs, nullptr, 0), nullptr,
      nullptr, nullptr, arDispatch);
  arSmallFused.completed = dArCompleted;
  typename TinyArFused::Params arTinyFused(
      typename TinyDenseKernel::Params(arTinyArgs, nullptr, 0), nullptr,
      nullptr, nullptr, arDispatch);
  arTinyFused.completed = dArCompleted;
  if (cfg.rsM <= 256) {
    int const hierSendHalf = (rank & 1) ^ 1;
    int const hierTileOffset = hierSendHalf * (arMTiles / 2);
    arDenseFused.tile_m_offset = hierTileOffset;
    arSmallFused.tile_m_offset = hierTileOffset;
    arTinyFused.tile_m_offset = hierTileOffset;
  }

  auto launchArPersistentProducerOnly = [&]() {
    CUDACHECK(cudaMemsetAsync(dArCompleted, 0, arCompletedBytes, mainStream));
    if (arUseTiny)
      uccl_lite::gatedGroupedKernel<TinyDensePlain>
          <<<arTinyWorkers, TinyDensePlain::kThreadCount, smemTiny,
             mainStream>>>(arTinyProducer);
    else if (arUseSmall)
      uccl_lite::gatedGroupedKernel<SmallDensePlain>
          <<<arSmallWorkers, SmallDensePlain::kThreadCount, smemSmall,
             mainStream>>>(arSmallProducer);
    else
      uccl_lite::gatedGroupedKernel<DensePlain>
          <<<arDenseWorkers, DensePlain::kThreadCount, smemPlain, mainStream>>>(
              arDenseProducer);
  };
  auto launchArPersistentNoOverlap = [&]() {
    launchArPersistentProducerOnly();
    if (arUseTiny)
      arHierOnly<<<arActiveBlocks, TinyDensePlain::kThreadCount, 0,
                   mainStream>>>(arDispatch);
    else if (arUseSmall)
      arHierOnly<<<arActiveBlocks, SmallDensePlain::kThreadCount, 0,
                   mainStream>>>(arDispatch);
    else
      arHierOnly<<<arActiveBlocks, DensePlain::kThreadCount, 0, mainStream>>>(
          arDispatch);
  };
  auto launchArPersistentFused = [&]() {
    CUDACHECK(cudaMemsetAsync(dArCompleted, 0, arCompletedBytes, mainStream));
    if (arUseTiny)
      uccl_lite::gatedGroupedKernel<TinyArFused>
          <<<arTinyWorkers + ArHierDispatch::kBlocks, TinyArFused::kThreadCount,
             smemTinyArFused, mainStream>>>(arTinyFused);
    else if (arUseSmall)
      uccl_lite::gatedGroupedKernel<SmallArFused>
          <<<arSmallWorkers + ArHierDispatch::kBlocks,
             SmallArFused::kThreadCount, smemSmallArFused, mainStream>>>(
              arSmallFused);
    else
      uccl_lite::gatedGroupedKernel<ArFused>
          <<<arDenseWorkers + ArHierDispatch::kBlocks, ArFused::kThreadCount,
             smemArFused, mainStream>>>(arDenseFused);
  };
  std::vector<cudaEvent_t> ready(cfg.rsStripes), done(cfg.rsStripes);
  for (int s = 0; s < cfg.rsStripes; s++) {
    CUDACHECK(cudaEventCreateWithFlags(&ready[s], cudaEventDisableTiming));
    CUDACHECK(cudaEventCreateWithFlags(&done[s], cudaEventDisableTiming));
  }
  cudaEvent_t halfReady[2], halfDone[2];
  for (int s = 0; s < 2; s++) {
    CUDACHECK(cudaEventCreateWithFlags(&halfReady[s], cudaEventDisableTiming));
    CUDACHECK(cudaEventCreateWithFlags(&halfDone[s], cudaEventDisableTiming));
  }
  uint32_t* dRsReady = nullptr;
  CUDACHECK(
      cudaMalloc(&dRsReady, (size_t)cfg.rsStripes * NRANKS * sizeof(uint32_t)));
  auto rsGemmStripe = [&](int s, cudaStream_t st) {
    uccl_lite::gatedGroupedKernel<DensePlain>
        <<<rsGrid, DensePlain::kThreadCount, smemPlain, st>>>(rsParams[s]);
  };
  auto rsCublasStripe = [&](int s) {
    int p = s & 1;
    for (int d = 0; d < NRANKS; d++)
      cublasGemm(q, cfg.rsN, rsLocalK,
                 rsA + ((size_t)d * rsLocalM + (size_t)s * q) * rsLocalK, rsB,
                 reinterpret_cast<__half*>(cbuf + slots[p].data) +
                     (size_t)d * q * cfg.rsN);
  };
  auto rsCublasReadyTile = [&](int stripe, int dst) {
    cublasGemm(q, cfg.rsN, rsLocalK,
               rsA + ((size_t)dst * rsLocalM + (size_t)stripe * q) * rsLocalK,
               rsB,
               reinterpret_cast<__half*>(cbuf + rsReadyDataOff) +
                   ((size_t)dst * cfg.rsStripes + stripe) * q * cfg.rsN);
  };
  cublasOperation_t rsTransA[NRANKS], rsTransB[NRANKS];
  int rsGM[NRANKS], rsGN[NRANKS], rsGK[NRANKS];
  int rsGLdA[NRANKS], rsGLdB[NRANKS], rsGLdC[NRANKS], rsGroupSize[NRANKS];
  float rsAlpha[NRANKS], rsBeta[NRANKS];
  for (int d = 0; d < NRANKS; d++) {
    rsTransA[d] = CUBLAS_OP_T;
    rsTransB[d] = CUBLAS_OP_N;
    rsGM[d] = cfg.rsN;
    rsGN[d] = q;
    rsGK[d] = rsLocalK;
    rsGLdA[d] = rsLocalK;
    rsGLdB[d] = rsLocalK;
    rsGLdC[d] = cfg.rsN;
    rsGroupSize[d] = 1;
    rsAlpha[d] = 1.0f;
    rsBeta[d] = 0.0f;
  }
  auto rsCublasGroupedStripe = [&](int s) {
    CUBLASCHECK(cublasGemmGroupedBatchedEx(
        cub, rsTransA, rsTransB, rsGM, rsGN, rsGK, rsAlpha,
        (const void* const*)(dRsB + s * NRANKS), CUDA_R_16F, rsGLdA,
        (const void* const*)(dRsA + s * NRANKS), CUDA_R_16F, rsGLdB, rsBeta,
        (void* const*)(dRsC + s * NRANKS), CUDA_R_16F, rsGLdC, NRANKS,
        rsGroupSize, CUBLAS_COMPUTE_32F));
  };
  auto rsCommStripe = [&](int s, cudaStream_t st) {
    gemmRsStripe<<<RS_BLOCKS, 256, 0, st>>>(
        pcIntra, pcCross, cbuf, slots[s & 1], rank, (size_t)q * cfg.rsN,
        rsOut + (size_t)s * q * cfg.rsN);
  };
  auto rsRingCommStripe = [&](int s, cudaStream_t st) {
    gemmRsRing<<<RS_BLOCKS, 256, 0, st>>>(pcIntra, pcCross, cbuf, slots[s & 1],
                                          rank, (size_t)q * cfg.rsN,
                                          rsOut + (size_t)s * q * cfg.rsN);
  };
  auto launchRsBaseline = [&]() {
    cublasGemm(cfg.rsM, cfg.rsN, rsLocalK, rsA, rsB, rsPartial);
    NCCLCHECK(ncclReduceScatter(rsPartial, rsRef, (size_t)rsLocalM * cfg.rsN,
                                ncclHalf, ncclSum, nccl, mainStream));
  };
  auto launchArBaseline = [&]() {
    cublasGemm(cfg.rsM, cfg.rsN, rsLocalK, rsA, rsB, rsPartial);
    NCCLCHECK(ncclAllReduce(rsPartial, arRef, (size_t)cfg.rsM * cfg.rsN,
                            ncclHalf, ncclSum, nccl, mainStream));
  };
  auto launchRsCublasOnly = [&]() {
    cublasGemm(cfg.rsM, cfg.rsN, rsLocalK, rsA, rsB, rsPartial);
  };
  auto launchRsStripedOnly = [&]() {
    for (int s = 0; s < cfg.rsStripes; s++) rsGemmStripe(s, mainStream);
  };
  auto launchRsCublasStripedOnly = [&]() {
    for (int s = 0; s < cfg.rsStripes; s++) rsCublasStripe(s);
  };
  auto launchRsGroupedOnly = [&]() {
    for (int s = 0; s < cfg.rsStripes; s++) rsCublasGroupedStripe(s);
  };
  auto launchRsSequential = [&]() {
    for (int s = 0; s < cfg.rsStripes; s++) {
      rsGemmStripe(s, mainStream);
      rsCommStripe(s, mainStream);
    }
  };
  auto launchRsFused = [&]() {
    for (int s = 0; s < cfg.rsStripes; s++) {
      if (s >= 2) CUDACHECK(cudaStreamWaitEvent(mainStream, done[s - 2], 0));
      rsGemmStripe(s, mainStream);
      CUDACHECK(cudaEventRecord(ready[s], mainStream));
      CUDACHECK(cudaStreamWaitEvent(commStream, ready[s], 0));
      rsCommStripe(s, commStream);
      CUDACHECK(cudaEventRecord(done[s], commStream));
    }
    CUDACHECK(cudaStreamWaitEvent(mainStream, done.back(), 0));
  };
  auto launchRsCublasSequential = [&]() {
    for (int s = 0; s < cfg.rsStripes; s++) {
      rsCublasStripe(s);
      rsCommStripe(s, mainStream);
    }
  };
  auto launchRsCublasFused = [&]() {
    for (int s = 0; s < cfg.rsStripes; s++) {
      if (s >= 2) CUDACHECK(cudaStreamWaitEvent(mainStream, done[s - 2], 0));
      rsCublasStripe(s);
      CUDACHECK(cudaEventRecord(ready[s], mainStream));
      CUDACHECK(cudaStreamWaitEvent(commStream, ready[s], 0));
      rsCommStripe(s, commStream);
      CUDACHECK(cudaEventRecord(done[s], commStream));
    }
    CUDACHECK(cudaStreamWaitEvent(mainStream, done.back(), 0));
  };
  auto launchRsGroupedSequential = [&]() {
    for (int s = 0; s < cfg.rsStripes; s++) {
      rsCublasGroupedStripe(s);
      rsCommStripe(s, mainStream);
    }
  };
  auto launchRsGroupedFused = [&]() {
    for (int s = 0; s < cfg.rsStripes; s++) {
      if (s >= 2) CUDACHECK(cudaStreamWaitEvent(mainStream, done[s - 2], 0));
      rsCublasGroupedStripe(s);
      CUDACHECK(cudaEventRecord(ready[s], mainStream));
      CUDACHECK(cudaStreamWaitEvent(commStream, ready[s], 0));
      rsCommStripe(s, commStream);
      CUDACHECK(cudaEventRecord(done[s], commStream));
    }
    CUDACHECK(cudaStreamWaitEvent(mainStream, done.back(), 0));
  };
  auto launchRsRingSequential = [&]() {
    for (int s = 0; s < cfg.rsStripes; s++) {
      rsCublasStripe(s);
      rsRingCommStripe(s, mainStream);
    }
  };
  auto launchRsRingFused = [&]() {
    for (int s = 0; s < cfg.rsStripes; s++) {
      if (s >= 2) CUDACHECK(cudaStreamWaitEvent(mainStream, done[s - 2], 0));
      rsCublasStripe(s);
      CUDACHECK(cudaEventRecord(ready[s], mainStream));
      CUDACHECK(cudaStreamWaitEvent(commStream, ready[s], 0));
      rsRingCommStripe(s, commStream);
      CUDACHECK(cudaEventRecord(done[s], commStream));
    }
    CUDACHECK(cudaStreamWaitEvent(mainStream, done.back(), 0));
  };
  auto launchReadyCollective = [&](int blocks, int threads, int allReduceMode) {
    CUDACHECK(cudaMemsetAsync(dRsReady, 0,
                              (size_t)cfg.rsStripes * NRANKS * sizeof(uint32_t),
                              mainStream));
    const int pos = ringPos(rank);
    for (int stripe = 0; stripe < cfg.rsStripes; stripe++) {
      for (int stage = 0; stage < NRANKS; stage++) {
        const int dst = ringRank(pos - stage - 1);
        rsCublasReadyTile(stripe, dst);
        publishRsReady<<<1, 32, 0, mainStream>>>(dRsReady,
                                                 stripe * NRANKS + dst);
        if (stripe == 0 && stage == 0) {
          CUDACHECK(cudaEventRecord(ready[0], mainStream));
          CUDACHECK(cudaStreamWaitEvent(commStream, ready[0], 0));
          gemmRsReadyRing<<<blocks, threads, 0, commStream>>>(
              pcIntra, pcCross, cbuf, rsReadyDataOff, rsReadyScratchOff, rank,
              cfg.rsStripes, (size_t)q * cfg.rsN, dRsReady, rsOut,
              allReduceMode, allReduceMode == 1 ? READY_AR_SUBS : 1);
          CUDACHECK(cudaEventRecord(done[0], commStream));
        }
      }
    }
    CUDACHECK(cudaStreamWaitEvent(mainStream, done[0], 0));
  };
  auto launchRsReadyRing = [&]() {
    launchReadyCollective(tunedReadyRsBlocks, tunedReadyRsThreads, 0);
  };
  auto launchRsPcieNoOverlap = [&]() {
    cublasGemm(cfg.rsM, cfg.rsN, rsLocalK, rsA, rsB,
               reinterpret_cast<__half*>(cbuf + rsReadyDataOff));
    CUDACHECK(cudaMemsetAsync(dRsReady, 1,
                              (size_t)cfg.rsStripes * NRANKS * sizeof(uint32_t),
                              mainStream));
    gemmRsReadyRing<<<tunedReadyRsBlocks, tunedReadyRsThreads, 0, mainStream>>>(
        pcIntra, pcCross, cbuf, rsReadyDataOff, rsReadyScratchOff, rank,
        cfg.rsStripes, (size_t)q * cfg.rsN, dRsReady, rsOut, 0, 1);
  };
  auto launchArReadyRing = [&]() {
    launchReadyCollective(READY_AR_BLOCKS, READY_AR_THREADS, 1);
  };
  auto launchArReadyHier = [&]() {
    launchReadyCollective(READY_AR_BLOCKS, READY_AR_THREADS, 2);
  };
  auto launchArPcieNoOverlap = [&]() {
    cublasGemm(cfg.rsM, cfg.rsN, rsLocalK, rsA, rsB,
               reinterpret_cast<__half*>(cbuf + rsReadyDataOff));
    pcieAllReduceRing<<<AR_BLOCKS, AR_THREADS, 0, mainStream>>>(
        pcIntra, pcCross, cbuf, rsReadyDataOff, rsReadyScratchOff, rank,
        (size_t)cfg.rsM * cfg.rsN, AR_SUBS);
  };
  auto launchArNcclOnly = [&]() {
    NCCLCHECK(ncclAllReduce(rsPartial, arRef, (size_t)cfg.rsM * cfg.rsN,
                            ncclHalf, ncclSum, nccl, mainStream));
  };
  auto launchArPcieOnly = [&]() {
    pcieAllReduceRing<<<AR_BLOCKS, AR_THREADS, 0, mainStream>>>(
        pcIntra, pcCross, cbuf, rsReadyDataOff, rsReadyScratchOff, rank,
        (size_t)cfg.rsM * cfg.rsN, AR_SUBS);
  };
  auto launchArReadyProducerOnly = [&]() {
    CUDACHECK(cudaMemsetAsync(dRsReady, 0,
                              (size_t)cfg.rsStripes * NRANKS * sizeof(uint32_t),
                              mainStream));
    const int pos = ringPos(rank);
    for (int stripe = 0; stripe < cfg.rsStripes; stripe++) {
      for (int stage = 0; stage < NRANKS; stage++) {
        const int dst = ringRank(pos - stage - 1);
        rsCublasReadyTile(stripe, dst);
        publishRsReady<<<1, 32, 0, mainStream>>>(dRsReady,
                                                 stripe * NRANKS + dst);
      }
    }
  };
  auto launchArReadyCommOnly = [&]() {
    CUDACHECK(cudaMemsetAsync(dRsReady, 1,
                              (size_t)cfg.rsStripes * NRANKS * sizeof(uint32_t),
                              mainStream));
    gemmRsReadyRing<<<READY_AR_BLOCKS, READY_AR_THREADS, 0, mainStream>>>(
        pcIntra, pcCross, cbuf, rsReadyDataOff, rsReadyScratchOff, rank,
        cfg.rsStripes, (size_t)q * cfg.rsN, dRsReady, rsOut, 1, READY_AR_SUBS);
  };
  auto launchArHybrid = [&]() {
    launchRsReadyRing();
    NCCLCHECK(ncclAllGather(rsOut, arOut, (size_t)rsLocalM * cfg.rsN, ncclHalf,
                            nccl, mainStream));
  };
  auto rsHalfGemm = [&](int h) {
    cublasGemm(2 * rsLocalM, cfg.rsN, rsLocalK,
               rsA + (size_t)h * 2 * rsLocalM * rsLocalK, rsB,
               reinterpret_cast<__half*>(cbuf + slots[h].data));
  };
  auto rsHalfComm = [&](int h, cudaStream_t st) {
    gemmRsHalf<<<RS_BLOCKS, 256, 0, st>>>(pcIntra, pcCross, cbuf, slots[h],
                                          rank, h, (size_t)rsLocalM * cfg.rsN,
                                          rsOut);
  };
  auto launchRsHalfOnly = [&]() {
    rsHalfGemm(0);
    rsHalfGemm(1);
  };
  auto launchRsHalfSequential = [&]() {
    for (int h = 0; h < 2; h++) {
      rsHalfGemm(h);
      rsHalfComm(h, mainStream);
    }
  };
  auto launchRsHalfFused = [&]() {
    for (int h = 0; h < 2; h++) {
      rsHalfGemm(h);
      CUDACHECK(cudaEventRecord(halfReady[h], mainStream));
      CUDACHECK(cudaStreamWaitEvent(commStream, halfReady[h], 0));
      rsHalfComm(h, commStream);
      CUDACHECK(cudaEventRecord(halfDone[h], commStream));
    }
    CUDACHECK(cudaStreamWaitEvent(mainStream, halfDone[1], 0));
  };

  if (cfg.check) {
    launchRsBaseline();
    CUDACHECK(cudaStreamSynchronize(mainStream));
    bootstrap->barrier();
    launchRsReadyRing();
    CUDACHECK(cudaStreamSynchronize(mainStream));
    CheckResult c = compare(rsRef, rsOut, (size_t)rsLocalM * cfg.rsN);
    std::vector<unsigned long long> bad(NRANKS);
    bad[rank] = c.bad;
    bootstrap->allGather(bad.data(), sizeof(unsigned long long));
    if (rank == 0)
      std::printf("# RS correctness: max_abs %.5g max_ref %.5g bad %llu\n",
                  c.maxAbs, c.maxRef,
                  *std::max_element(bad.begin(), bad.end()));
    bootstrap->barrier();
    launchRsPcieNoOverlap();
    CUDACHECK(cudaStreamSynchronize(mainStream));
    c = compare(rsRef, rsOut, (size_t)rsLocalM * cfg.rsN);
    bad[rank] = c.bad;
    bootstrap->allGather(bad.data(), sizeof(unsigned long long));
    if (rank == 0)
      std::printf(
          "# RS correctness (UCCL Lite-no-overlap): max_abs %.5g "
          "max_ref %.5g bad %llu\n",
          c.maxAbs, c.maxRef, *std::max_element(bad.begin(), bad.end()));
    bootstrap->barrier();
  }

  std::vector<BenchRow> rsRows = {
      {"gemm-only-cuBLAS", launchRsCublasOnly, {}},
      {"gemm-only-striped-cuBLAS", launchRsCublasStripedOnly, {}},
      {"cuBLAS+NCCL-RS", launchRsBaseline, {}},
      {"cuBLAS+UCCL Lite-RS-no-ovlp", launchRsPcieNoOverlap, {}},
      {"cuBLAS-striped+hier-RS", launchRsCublasSequential, {}},
      {"fused-cuBLAS-GEMM+RS", launchRsCublasFused, {}},
      {"cuBLAS-striped+ring-RS", launchRsRingSequential, {}},
      {"fused-ring-GEMM+RS", launchRsRingFused, {}},
      {"ready-ring-GEMM+RS", launchRsReadyRing, {}},
  };
  char rsTitle[192];
  std::snprintf(
      rsTitle, sizeof(rsTitle),
      "GEMM+RS M=%d N=%d K=%d TP=4 (local K=%d, stripes=%d, q=%d, ready=%dx%d)",
      cfg.rsM, cfg.rsN, cfg.rsK, rsLocalK, cfg.rsStripes, q, tunedReadyRsBlocks,
      tunedReadyRsThreads);
  benchRows(rsTitle, rsRows, 2, rank, cfg, bootstrap, mainStream);

  if (cfg.check) {
    launchArBaseline();
    CUDACHECK(cudaStreamSynchronize(mainStream));
    auto checkAr = [&](char const* label, auto&& launch, __half const* out) {
      bootstrap->barrier();
      launch();
      CUDACHECK(cudaStreamSynchronize(mainStream));
      CheckResult c = compare(arRef, out, (size_t)cfg.rsM * cfg.rsN);
      std::vector<unsigned long long> bad(NRANKS);
      bad[rank] = c.bad;
      bootstrap->allGather(bad.data(), sizeof(unsigned long long));
      if (rank == 0)
        std::printf(
            "# AR correctness (%s): max_abs %.5g max_ref %.5g bad %llu\n",
            label, c.maxAbs, c.maxRef,
            *std::max_element(bad.begin(), bad.end()));
      bootstrap->barrier();
    };
    checkAr("ready-ring", launchArReadyRing,
            reinterpret_cast<__half*>(cbuf + rsReadyDataOff));
    checkAr("ready-hier", launchArReadyHier,
            reinterpret_cast<__half*>(cbuf + rsReadyDataOff));
    checkAr("UCCL Lite-no-overlap", launchArPcieNoOverlap,
            reinterpret_cast<__half*>(cbuf + rsReadyDataOff));
    checkAr("persistent-hier-no-overlap", launchArPersistentNoOverlap,
            reinterpret_cast<__half*>(cbuf + rsReadyDataOff));
    checkAr("persistent-epilogue-hier", launchArPersistentFused,
            reinterpret_cast<__half*>(cbuf + rsReadyDataOff));
    checkAr("ready-RS+NCCL-AG", launchArHybrid, arOut);
  }

  std::vector<BenchRow> arRows = {
      {"gemm-only-cuBLAS", launchRsCublasOnly, {}},
      {"cuBLAS+NCCL-AllReduce", launchArBaseline, {}},
      {"cuBLAS+UCCL Lite-AR-no-ovlp", launchArPcieNoOverlap, {}},
      {"ready-GEMM-producer-only", launchArReadyProducerOnly, {}},
      {"NCCL-AllReduce-only", launchArNcclOnly, {}},
      {"UCCL Lite-AllReduce-only", launchArPcieOnly, {}},
      {"ready-ring-AR-only", launchArReadyCommOnly, {}},
      {"ready-ring-GEMM+AR", launchArReadyRing, {}},
      {"ready-hier-GEMM+AR", launchArReadyHier, {}},
      {"ready-RS+NCCL-AG", launchArHybrid, {}},
      {"persistent-GEMM-only", launchArPersistentProducerOnly, {}},
      {"persistent+hier-AR-no-ovlp", launchArPersistentNoOverlap, {}},
      {"persistent-epilogue-GEMM+AR", launchArPersistentFused, {}},
  };
  char arTitle[192];
  std::snprintf(
      arTitle, sizeof(arTitle),
      "GEMM+AllReduce M=%d N=%d K=%d TP=4 (local K=%d, stripes=%d, q=%d)",
      cfg.rsM, cfg.rsN, cfg.rsK, rsLocalK, cfg.rsStripes, q);
  std::vector<double> arMs =
      benchRows(arTitle, arRows, 1, rank, cfg, bootstrap, mainStream);
  if (rank == 0) {
    double const fullGemm = arMs[0];
    double const readyProducer = arMs[3];
    double const pcieAr = arMs[5];
    double const readyAr = arMs[6];
    double const fused = arMs[7];
    double const absoluteIdeal = std::max(fullGemm, pcieAr);
    double const scheduleIdeal = std::max(readyProducer, readyAr);
    double const hidden = readyProducer + readyAr - fused;
    double const hideable = std::min(readyProducer, readyAr);
    double const overlap = std::max(0.0, std::min(1.0, hidden / hideable));
    std::printf(
        "# overlap: full_gemm %.3f ms, ready_producer %.3f ms, "
        "ready_comm %.3f ms\n",
        fullGemm, readyProducer, readyAr);
    std::printf(
        "# overlap: absolute_ideal %.3f ms, schedule_ideal %.3f ms, "
        "fused %.3f ms, exposed_tail %.3f ms, hidden %.1f%%\n",
        absoluteIdeal, scheduleIdeal, fused, fused - scheduleIdeal,
        100.0 * overlap);
    double const persistentGemm = arMs[10];
    double const persistentNoOverlap = arMs[11];
    double const persistentFused = arMs[12];
    double const persistentComm = persistentNoOverlap - persistentGemm;
    double const persistentHidden = persistentNoOverlap - persistentFused;
    double const persistentHideable = std::min(persistentGemm, persistentComm);
    std::printf(
        "# epilogue-overlap: gemm %.3f ms, inferred_comm %.3f ms, "
        "no_overlap %.3f ms, fused %.3f ms, hidden %.1f%%\n",
        persistentGemm, persistentComm, persistentNoOverlap, persistentFused,
        persistentHideable > 0 ? 100.0 * persistentHidden / persistentHideable
                               : 0.0);
    std::fflush(stdout);
  }
  bootstrap->barrier();

  CUDACHECK(cudaStreamSynchronize(mainStream));
  CUDACHECK(cudaStreamSynchronize(commStream));
  bootstrap->barrier();
  proxy->stopProxy();
  NCCLCHECK(ncclCommDestroy(nccl));
  CUBLASCHECK(cublasDestroy(cub));
  for (int s = 0; s < cfg.rsStripes; s++) {
    CUDACHECK(cudaEventDestroy(ready[s]));
    CUDACHECK(cudaEventDestroy(done[s]));
  }
  for (int s = 0; s < 2; s++) {
    CUDACHECK(cudaEventDestroy(halfReady[s]));
    CUDACHECK(cudaEventDestroy(halfDone[s]));
  }
  CUDACHECK(cudaFree(dRsReady));
  CUDACHECK(cudaEventDestroy(agFlagsCleared));
  CUDACHECK(cudaStreamDestroy(mainStream));
  CUDACHECK(cudaStreamDestroy(commStream));
}

int main(int argc, char** argv) {
  Cfg cfg;
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    auto next = [&]() -> char const* { return i + 1 < argc ? argv[++i] : "0"; };
    if (k == "--ag-m")
      cfg.agM = std::atoi(next());
    else if (k == "--ag-n")
      cfg.agN = std::atoi(next());
    else if (k == "--ag-k")
      cfg.agK = std::atoi(next());
    else if (k == "--rs-m")
      cfg.rsM = std::atoi(next());
    else if (k == "--rs-n")
      cfg.rsN = std::atoi(next());
    else if (k == "--rs-k")
      cfg.rsK = std::atoi(next());
    else if (k == "-s" || k == "--stripes")
      cfg.rsStripes = std::atoi(next());
    else if (k == "--ready-rs-blocks")
      cfg.readyRsBlocks = std::atoi(next());
    else if (k == "--ready-rs-threads")
      cfg.readyRsThreads = std::atoi(next());
    else if (k == "--fused-ar-blocks")
      cfg.fusedArBlocks = std::atoi(next());
    else if (k == "--fused-ar-gemm-blocks")
      cfg.fusedArGemmBlocks = std::atoi(next());
    else if (k == "--fused-ar-tiles")
      cfg.fusedArTiles = std::atoi(next());
    else if (k == "-w")
      cfg.warmup = std::atoi(next());
    else if (k == "-n")
      cfg.iters = std::atoi(next());
    else if (k == "-c")
      cfg.check = std::atoi(next()) != 0;
    else {
      std::fprintf(stderr, "unknown arg %s\n", k.c_str());
      return 1;
    }
  }
  if (cfg.rsStripes < 1 || cfg.readyRsBlocks < 0 || cfg.readyRsThreads < 0 ||
      cfg.fusedArBlocks < 1 || cfg.fusedArBlocks > ArHierDispatch::kBlocks ||
      cfg.fusedArGemmBlocks < 0 || cfg.fusedArTiles < 0 ||
      cfg.readyRsThreads > 1024 ||
      (cfg.readyRsThreads != 0 &&
       (cfg.readyRsThreads < 32 || cfg.readyRsThreads % 32)))
    return 1;

  mscclpp::UniqueId id = mscclpp::TcpBootstrap::createUniqueId();
  std::vector<pid_t> kids;
  for (int r = 1; r < NRANKS; r++) {
    pid_t p = fork();
    if (p == 0) {
      runRank(r, id, cfg);
      _exit(0);
    }
    kids.push_back(p);
  }
  runRank(0, id, cfg);
  int rc = 0;
  for (pid_t p : kids) {
    int st = 0;
    waitpid(p, &st, 0);
    if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) rc = 1;
  }
  return rc;
}
