#pragma once
// TP=4 / EP=1: all experts exist on all ranks, sharded along intermediate I.
// One down-GEMM grid contains both compute and combine/AllReduce consumers.
#include "cutlass_gated_grouped.h"
#include <cuda/atomic>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/memory_channel_device.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <cuda_bf16.h>

#ifndef TPM_BM
#define TPM_BM 128
#endif
#ifndef TPM_BN
#define TPM_BN 128
#endif
#ifndef TPM_COMM_BLOCKS
#define TPM_COMM_BLOCKS 8
#endif
#ifndef TPM_UP_M
#define TPM_UP_M 128
#endif
#ifndef TPM_UP_N
#define TPM_UP_N 256
#endif
#ifndef TPM_BK
#define TPM_BK 32
#endif
#ifndef TPM_WM
#define TPM_WM 16
#endif
#ifndef TPM_WN
#define TPM_WN 64
#endif
#ifndef TPM_STAGES
#define TPM_STAGES 3
#endif
#ifndef TPM_UP_K
#define TPM_UP_K 32
#endif
#ifndef TPM_UP_STAGES
#define TPM_UP_STAGES 4
#endif
#ifndef TPM_CE
#define TPM_CE 1
#endif
#ifndef TPM_RING
#define TPM_RING 1
#endif

namespace tpmoe {
using Bf = __nv_bfloat16;
using El = cutlass::bfloat16_t;
using Row = cutlass::layout::RowMajor;
using Col = cutlass::layout::ColumnMajor;
using Tile = cutlass::gemm::GemmShape<TPM_BM, TPM_BN, TPM_BK>;
using Warp = cutlass::gemm::GemmShape<TPM_WM, TPM_WN, TPM_BK>;
using Inst = cutlass::gemm::GemmShape<16, 8, 16>;
using Epi = cutlass::epilogue::thread::LinearCombination<El, 8, float, float>;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;
using Kernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    El, Row, cutlass::ComplexTransform::kNone, 8, El, Col,
    cutlass::ComplexTransform::kNone, 8, El, Row, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, Tile, Warp, Inst, Epi,
    Swizzle, TPM_STAGES>::GemmKernel;
using GatherBase = typename cutlass::gemm::kernel::DefaultGemm<
    El, Row, 8, El, Col, 8, El, Row, float, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80, cutlass::gemm::GemmShape<TPM_UP_M, TPM_UP_N, TPM_UP_K>,
    cutlass::gemm::GemmShape<(TPM_UP_M >= 128 ? 64
                                              : (TPM_UP_M >= 32 ? 32 : 16)),
                             (TPM_UP_N >= 128 ? 64 : 32), TPM_UP_K>,
    Inst, Epi, Swizzle, TPM_UP_STAGES, true,
    typename cutlass::gemm::device::DefaultGemmConfiguration<
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, El, El, El,
        float>::Operator,
    cutlass::gemm::SharedMemoryClearOption::kNone, true, false,
    false>::GemmKernel;
using Gather = cutlass::gemm::kernel::GemmGrouped<
    typename GatherBase::Mma, typename GatherBase::Epilogue, Swizzle,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>;
using Up = uccl_lite::GatedGrouped<Gather>;

__device__ inline unsigned long long timestamp() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// GEMM workers help combine ready output-channel stripes between GEMM tiles.
// Each task owns complete tokens, avoiding per-element atomics and preserving
// top-k order. Workers drain remaining combine tasks before returning from the
// same grid.
struct CombineTasks {
  static constexpr bool kEnabled = true;
  static constexpr int kRows = 16;
  int const *group, *inverse;
  float const* weights;
  Bf const* y;
  Bf* out;
  uint32_t *gemmReady, *expected, *next, *done;
  int T, H, K, chunk;
  unsigned long long* gemmTime;
  Bf const* shared;
  int rank;
  bool shardReady;
  __device__ int shards(int q) const {
    return shardReady && ((size_t)T * min(chunk, H - q * chunk)) % 32 == 0 ? 4
                                                                           : 1;
  }
  __device__ int vectors(int q) const {
    return T * (min(chunk, H - q * chunk) / 8) / shards(q);
  }
  __device__ int tasks(int q) const {
    int perTask = kRows * (min(chunk, H - q * chunk) / 8);
    return (vectors(q) + perTask - 1) / perTask;
  }
  __device__ uint32_t acquire(uint32_t* ptr) const {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> a(*ptr);
    return a.load(cuda::memory_order_acquire);
  }
  __device__ void service() const {
    __shared__ int chosen, job, dest;
    if (threadIdx.x == 0) {
      chosen = -1;
      for (int q = 0; q < (H + chunk - 1) / chunk; ++q) {
        if (acquire(gemmReady + q) != expected[q]) continue;
        int count = shards(q), pos = rank < 2 ? rank : 5 - rank;
        // Produce the first outbound shard, then the three local reduction
        // destinations, in the same order that this rank consumes them.
        for (int step = 0; step < count; ++step) {
          int p = (pos - step - 1) & 3,
              d = count == 1 ? 0 : (p < 2 ? p : 5 - p);
          int slot = q * 4 + d;
          if (acquire(next + slot) >= (uint32_t)tasks(q)) continue;
          int j = atomicAdd(next + slot, 1u);
          if (j < tasks(q)) {
            chosen = q;
            job = j;
            dest = d;
            break;
          }
        }
        if (chosen >= 0) break;
      }
    }
    __syncthreads();
    int q = chosen;
    if (q >= 0) {
      int cols = min(chunk, H - q * chunk), perTask = kRows * (cols / 8);
      int first = dest * vectors(q) + job * perTask;
      int count = min(perTask, vectors(q) - job * perTask);
      for (int j = threadIdx.x; j < count; j += blockDim.x) {
        int t = (first + j) / (cols / 8), c = (first + j) % (cols / 8);
        float sum[8] = {};
        for (int k = 0; k < K; ++k) {
          int r = inverse[t * K + k];
          uint4 raw =
              ((uint4 const*)y)[(size_t)r * (H / 8) + q * (chunk / 8) + c];
          float w = weights[t * K + k];
#pragma unroll
          for (int z = 0; z < 8; ++z)
            sum[z] += w * __bfloat162float(((Bf*)&raw)[z]);
        }
        uint4 raw;
#pragma unroll
        for (int z = 0; z < 8; ++z) ((Bf*)&raw)[z] = __float2bfloat16(sum[z]);
        // Match the framework's BF16 routed + shared addition before the TP
        // reduction. Shared experts use this same collective, not another AR.
        if (shared) {
          uint4 add =
              ((uint4 const*)shared)[(size_t)t * (H / 8) + q * (chunk / 8) + c];
#pragma unroll
          for (int z = 0; z < 4; ++z)
            ((__nv_bfloat162*)&raw)[z] =
                __hadd2(((__nv_bfloat162*)&raw)[z], ((__nv_bfloat162*)&add)[z]);
        }
        ((uint4*)out)[(size_t)q * chunk * T / 8 + first + j] = raw;
      }
      __threadfence();
      __syncthreads();
      if (threadIdx.x == 0) atomicAdd(done + q * 4 + dest, 1u);
    }
    __syncthreads();
  }
  __device__ void complete_tile(int p) const {
    __syncthreads();
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned long long t = gemmTime ? timestamp() : 0;
      int q = group[p];
      uint32_t old = atomicAdd(gemmReady + q, 1u);
      if (gemmTime && old + 1 == expected[q]) gemmTime[q] = t;
    }
    service();
  }
  __device__ void finish() const {
    __shared__ int finished;
    for (;;) {
      if (threadIdx.x == 0) {
        finished = 1;
        for (int q = 0; q < (H + chunk - 1) / chunk; ++q)
          for (int d = 0; d < shards(q); ++d)
            if (acquire(done + q * 4 + d) != (uint32_t)tasks(q)) finished = 0;
      }
      __syncthreads();
      if (finished) return;
      service();
    }
  }
};
using Plain =
    uccl_lite::GatedGrouped<Kernel, uccl_lite::NoDispatch, CombineTasks>;

struct Problems {
  cutlass::gemm::GemmCoord* shape;
  El **a, **b, **c;
  int64_t *lda, *ldb, *ldc;
  int const** indices;
  template <class K>
  typename K::Params params(int count, int grid) const {
    typename K::Arguments args(shape, count, grid,
                               typename K::EpilogueOutputOp::Params(1.f, 0.f),
                               a, b, c, c, lda, ldb, ldc, ldc);
    return typename K::Params(args, nullptr, 0);
  }
};

__global__ void histogram(int const* ids, int* counts, int T, int K) {
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < T * K;
       j += gridDim.x * blockDim.x)
    atomicAdd(counts + ids[j], 1);
}

// Dynamic problem sizes never leave the device; empty experts are zero-M
// CUTLASS problems. Storage is compact and needs exactly T*topk rows.
__global__ void makeProblems(int* count, int* off, int* cursor, int* group,
                             uint32_t* expected, uint32_t* ready,
                             uint32_t* next, uint32_t* done, uint32_t* epoch,
                             Problems up, Problems down, int* row, Bf const* x,
                             Bf const* w13, Bf const* w2, Bf* h, Bf* a, Bf* y,
                             int P, int E, int H, int I, int chunk,
                             int const* ids, int* inverse, int T, int K,
                             bool small) {
  // Small routing plans fit in one CTA. Keep the histogram in shared memory
  // and construct the scatter map here, avoiding separate resets/launches.
  __shared__ int localCount[256];
  if (epoch && threadIdx.x == 0) *epoch += P / E;
  if (small) {
    count = localCount;
    for (int e = threadIdx.x; e < E; e += blockDim.x) count[e] = 0;
    __syncthreads();
    for (int j = threadIdx.x; j < T * K; j += blockDim.x)
      atomicAdd(count + ids[j], 1);
    __syncthreads();
  }
  // This kernel has a single CTA. Its barrier orders all counter resets
  // before expected[] is updated, including when a graph changes routes.
  for (int q = threadIdx.x; q < P / E; q += blockDim.x) {
    expected[q] = 0;
    ready[q] = 0;
    for (int d = 0; d < 4; ++d) {
      next[q * 4 + d] = 0;
      done[q * 4 + d] = 0;
    }
  }
  if (E <= 256) {
    __shared__ int warpSum[8];
    int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    int n = threadIdx.x < E ? count[threadIdx.x] : 0, prefix = n;
#pragma unroll
    for (int delta = 1; delta < 32; delta *= 2) {
      int v = __shfl_up_sync(0xffffffff, prefix, delta);
      if (lane >= delta) prefix += v;
    }
    if (lane == 31) warpSum[warp] = prefix;
    __syncthreads();
    int base = 0;
    for (int w = 0; w < warp; ++w) base += warpSum[w];
    if (threadIdx.x < E) {
      off[threadIdx.x] = base + prefix - n;
      cursor[threadIdx.x] = base + prefix - n;
    }
    if (threadIdx.x == 255) off[E] = base + prefix;
  } else if (threadIdx.x == 0) {
    int sum = 0;
    for (int e = 0; e < E; ++e) {
      off[e] = sum;
      sum += count[e];
      cursor[e] = off[e];
    }
    off[E] = sum;
  }
  __syncthreads();
  for (int e = threadIdx.x; e < E; e += blockDim.x) {
    int m = count[e];
    int r = off[e];
    up.shape[e] = {m, 2 * I, H};
    up.a[e] = (El*)x;
    up.b[e] = (El*)w13 + (size_t)e * 2 * I * H;
    up.c[e] = (El*)h + (size_t)r * 2 * I;
    up.lda[e] = H;
    up.ldb[e] = H;
    up.ldc[e] = 2 * I;
    up.indices[e] = row + r;
  }
  for (int p = threadIdx.x; p < P; p += blockDim.x) {
    int e = p % E, q = p / E, m = count[e], r = off[e],
        cols = min(chunk, H - q * chunk);
    group[p] = q;
    if (m)
      atomicAdd(expected + q,
                ((m + TPM_BM - 1) / TPM_BM) * ((cols + TPM_BN - 1) / TPM_BN));
    down.shape[p] = {m, cols, I};
    down.a[p] = (El*)a + (size_t)r * I;
    down.b[p] = (El*)w2 + ((size_t)e * H + q * chunk) * I;
    down.c[p] = (El*)y + (size_t)r * H + q * chunk;
    down.lda[p] = I;
    down.ldb[p] = I;
    down.ldc[p] = H;
  }
  if (small) {
    __syncthreads();
    for (int j = threadIdx.x; j < T * K; j += blockDim.x) {
      int r = atomicAdd(cursor + ids[j], 1);
      row[r] = j / K;
      inverse[j] = r;
    }
  }
}

__global__ void scatter(int const* ids, int* cursor, int* row, int* inverse,
                        int T, int K) {
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < T * K;
       j += gridDim.x * blockDim.x) {
    int r = atomicAdd(cursor + ids[j], 1);
    row[r] = j / K;
    inverse[j] = r;
  }
}
__global__ void swiglu(Bf const* h, Bf* a, int rows, int I) {
  for (size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       j < (size_t)rows * I; j += (size_t)gridDim.x * blockDim.x) {
    size_t b = (j / I) * 2 * I + j % I;
    float gate = __bfloat162float(h[b]), up = __bfloat162float(h[b + I]);
    a[j] = __float2bfloat16(gate / (1.f + expf(-gate)) * up);
  }
}

__device__ mscclpp::DeviceSyncer consumerSync;
__device__ inline void addVector(Bf* dst, Bf const* src, size_t n, int tid,
                                 int nth) {
  for (size_t j = tid; j < n / 8; j += nth) {
    uint4 a = ((uint4*)dst)[j], b = ((uint4 const*)src)[j];
    auto* av = (__nv_bfloat162*)&a;
    auto* bv = (__nv_bfloat162*)&b;
#pragma unroll
    for (int k = 0; k < 4; ++k) av[k] = __hadd2(av[k], bv[k]);
    ((uint4*)dst)[j] = a;
  }
}
struct Consumer {
  static constexpr bool kEnabled = true;
  static constexpr int kBlocks = TPM_COMM_BLOCKS;
  mscclpp::MemoryChannelDeviceHandle intra, cross;
  mscclpp::PortChannelDeviceHandle ceIntra, ceCross;
  Bf* out;
  Bf* result;
  uint32_t* completed;
  int rank, T, H, chunk;
  size_t scratch;  // byte offset in registered output allocation
  bool wait;
  unsigned long long* ticks;
  bool shardReady;
  int blocks = kBlocks;
  bool sm = false;
  __device__ void awaitShard(int q, int d, int tid) {
    int cols = min(chunk, H - q * chunk);
    int count = shardReady && ((size_t)T * cols) % 32 == 0 ? 4 : 1;
    int vectors = T * (cols / 8) / count,
        perTask = CombineTasks::kRows * (cols / 8);
    if (wait && tid == 0) {
      cuda::atomic_ref<uint32_t, cuda::thread_scope_device> ready(
          completed[q * 4 + d]);
      while (ready.load(cuda::memory_order_acquire) !=
             (uint32_t)((vectors + perTask - 1) / perTask))
        __nanosleep(64);
    }
    consumerSync.sync(blocks);
  }

  __device__ void transfer(bool local, size_t target, size_t origin,
                           size_t bytes, int tid, int nth) {
#if TPM_CE
    if (!sm) {
      if (tid == 0) {
        auto pc = local ? ceIntra : ceCross;
        __threadfence_system();
        pc.putWithSignal(target, origin, bytes);
        pc.wait();
        // The following reduction can overwrite the sender's source. Receiving
        // the peer's transfer does not prove our outbound DMA finished.
        pc.flush();
      }
      consumerSync.sync(blocks);
      return;
    }
#endif
    auto mc = local ? intra : cross;
    mc.put<16, true>(target, origin, bytes, tid, nth);
    consumerSync.sync(blocks);
    if (tid == 0) {
      mc.signal();
      mc.wait();
    }
    consumerSync.sync(blocks);
  }

  __device__ int ringRank(int pos) {
    pos &= 3;
    return pos < 2 ? pos : 5 - pos;
  }
  __device__ void ring(size_t start, size_t elems, int q, int tid, int nth) {
    int const pos = rank < 2 ? rank : 5 - rank;
    bool sendLocal = ringRank(pos + 1) == (rank ^ 1);
    auto ms = sendLocal ? intra : cross, mr = sendLocal ? cross : intra;
    auto ps = sendLocal ? ceIntra : ceCross, pr = sendLocal ? ceCross : ceIntra;
    size_t shard = elems / 4, bytes = shard * sizeof(Bf);
    Bf* data = (Bf*)((char*)out + start);
    Bf* temp = (Bf*)((char*)out + scratch);
    // Per-step receive slots protect delayed readers. Forwarded source
    // shards are never modified again, so one final flush retires all sends.
    for (int step = 0; step < 3; ++step) {
      int sendDest = ringRank(pos - step - 1),
          recvDest = ringRank(pos - step - 2);
      if (shardReady && step == 0) {
        awaitShard(q, sendDest, tid);
        if (tid == 0 && ticks) ticks[q * 6 + 1] = timestamp();
      }
      __threadfence_system();
      consumerSync.sync(blocks);
#if TPM_CE
      int pieces = bytes >= (1u << 20) ? 4 : 1;
      size_t pieceElems = ((shard + pieces - 1) / pieces + 7) & ~size_t(7);
      if (tid == 0)
        for (int p = 0; p < pieces; ++p) {
          size_t off = p * pieceElems;
          if (off >= shard) break;
          size_t n = min(pieceElems, shard - off);
          ps.putWithSignal(scratch + (step * shard + off) * sizeof(Bf),
                           start + (sendDest * shard + off) * sizeof(Bf),
                           n * sizeof(Bf));
        }
      if (shardReady) awaitShard(q, recvDest, tid);
      for (int p = 0; p < pieces; ++p) {
        size_t off = p * pieceElems;
        if (off >= shard) break;
        size_t n = min(pieceElems, shard - off);
        if (tid == 0) pr.wait();
        consumerSync.sync(blocks);
        if (step == 0 && p == 0 && tid == 0 && ticks)
          ticks[q * 6 + 2] = timestamp();
        addVector(data + recvDest * shard + off, temp + step * shard + off, n,
                  tid, nth);
        consumerSync.sync(blocks);
      }
#else
      ms.put<16, true>(scratch + step * bytes, start + sendDest * bytes, bytes,
                       tid, nth);
      consumerSync.sync(blocks);
      if (tid == 0) {
        ms.signal();
        mr.wait();
      }
      consumerSync.sync(blocks);
      if (shardReady) awaitShard(q, recvDest, tid);
      if (step == 0 && tid == 0 && ticks) ticks[q * 6 + 2] = timestamp();
      addVector(data + recvDest * shard, temp + step * shard, shard, tid, nth);
      consumerSync.sync(blocks);
#endif
    }
    if (tid == 0 && ticks) ticks[q * 6 + 3] = timestamp();
    // The successor has consumed its last partial before AG overwrites it.
    if (tid == 0) {
      mr.signal();
      ms.wait();
    }
    consumerSync.sync(blocks);
    __threadfence_system();
    consumerSync.sync(blocks);
    for (int step = 0; step < 3; ++step) {
      size_t off = start + ringRank(pos - step) * bytes;
#if TPM_CE
      if (tid == 0) {
        ps.putWithSignal(off, off, bytes);
        pr.wait();
      }
      consumerSync.sync(blocks);
#else
      ms.put<16, true>(off, off, bytes, tid, nth);
      consumerSync.sync(blocks);
      if (tid == 0) {
        ms.signal();
        mr.wait();
      }
      consumerSync.sync(blocks);
#endif
    }
    if (tid == 0 && ticks) ticks[q * 6 + 4] = timestamp();
  }

  __device__ void operator()() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x, nth = blocks * blockDim.x;
    char* base = (char*)out;
    for (int first = 0, q = 0; first < H; first += chunk, ++q) {
      int cols = min(chunk, H - first);
      if (tid == 0 && ticks) ticks[q * 6] = timestamp();
      if (!shardReady || ((size_t)T * cols) % 32 != 0) awaitShard(q, 0, tid);
      if (tid == 0 && ticks) ticks[q * 6 + 1] = timestamp();
      if (tid == 0 && ticks) ticks[q * 6 + 2] = timestamp();
      // Recursive halving across the fast NUMA pair, then exchange only the
      // retained half across sockets, then local all-gather. Each rank sends
      // half an output chunk across the slow socket link, not the full chunk.
      size_t n = (size_t)T * cols / 2, bytes = n * sizeof(Bf);
      size_t start = (size_t)first * T * sizeof(Bf);
      int own = rank & 1, other = own ^ 1;
#if TPM_RING
      if (!sm && ((size_t)T * cols) % 32 == 0) {
        ring(start, (size_t)T * cols, q, tid, nth);
      } else {
#endif
        transfer(true, scratch, start + other * bytes, bytes, tid, nth);
        if (tid == 0 && ticks) ticks[q * 6 + 2] = timestamp();
        Bf* mine = out + (size_t)first * T + own * n;
        Bf* tmp = (Bf*)(base + scratch);
        addVector(mine, tmp, n, tid, nth);
        consumerSync.sync(blocks);
        if (tid == 0 && ticks) ticks[q * 6 + 3] = timestamp();
        transfer(false, scratch + bytes, start + own * bytes, bytes, tid, nth);
        tmp = (Bf*)(base + scratch + bytes);
        addVector(mine, tmp, n, tid, nth);
        consumerSync.sync(blocks);
        if (tid == 0 && ticks) ticks[q * 6 + 4] = timestamp();
        transfer(true, start + own * bytes, start + own * bytes, bytes, tid,
                 nth);
#if TPM_RING
      }
#endif
      // Restore [token, hidden] layout inside the same collective grid.
      for (int j = tid; j < T * (cols / 8); j += nth) {
        int t = j / (cols / 8), c = j % (cols / 8);
        ((uint4*)result)[(size_t)t * (H / 8) + first / 8 + c] =
            ((uint4 const*)out)[(size_t)first * T / 8 + j];
      }
      consumerSync.sync(blocks);
      // Scratch reuse: a fast rank cannot overwrite a slow peer's receive
      // half for the next chunk until both have finished reading this one.
      if (tid == 0) {
        cross.signal();
        cross.wait();
      }
      consumerSync.sync(blocks);
      if (tid == 0 && ticks) ticks[q * 6 + 5] = timestamp();
    }
#if TPM_RING && TPM_CE
    if (tid == 0 && !sm) {
      bool local = ringRank((rank < 2 ? rank : 5 - rank) + 1) == (rank ^ 1);
      auto send = local ? ceIntra : ceCross;
      send.flush();
    }
    consumerSync.sync(blocks);
#endif
  }
};
using Fused = uccl_lite::GatedGrouped<Kernel, Consumer, CombineTasks>;
__global__ void communicate(Consumer c) { c(); }
__global__ void unpack(Bf const* packed, Bf* result, int T, int H, int chunk) {
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < T * (H / 8);
       j += gridDim.x * blockDim.x) {
    int t = j / (H / 8), c = (j % (H / 8)) * 8, q = c / chunk,
        cols = min(chunk, H - q * chunk);
    ((uint4*)result)[j] =
        ((uint4 const*)packed)[(size_t)q * chunk * T / 8 +
                               (size_t)t * (cols / 8) + (c - q * chunk) / 8];
  }
}
}  // namespace tpmoe
