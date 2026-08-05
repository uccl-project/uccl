#pragma once

#include "high_perf.h"
#include "reduce_ops.h"
#include "reg_ops.h"
#include "tma_ops.h"

// Reduce ILP (16B loads in flight per thread) is a BUILD-TIME knob so the
// fully-unrolled kernel stays cheap to compile: REDUCE_ILP=4|8|16 (default
// 4). A runtime dispatch over 4/8/16 tripled cicc+ptxas time to ~20 min
// per file on B300; with a single compile-time value it is ~1-2 min.
#ifndef UK_REDUCE_ILP
#define UK_REDUCE_ILP 4
#endif
#if UK_REDUCE_ILP != 4 && UK_REDUCE_ILP != 8 && UK_REDUCE_ILP != 16
#error "UK_REDUCE_ILP must be 4, 8 or 16"
#endif

// Large-task TMA bulk reduce (sm_90+): REDUCE_SMEM_KB is the dynamic
// shared-memory budget per block (src+dst chunk buffers + mbarriers);
// TMA_REDUCE=1 enables the cp.async.bulk chunked path instead of the
// register/ILP-limited vector loop. Both are build-time so the kernel
// and the launch config stay consistent (no runtime size mismatch).
#ifndef UK_TMA_REDUCE
#define UK_TMA_REDUCE 0
#endif
#ifndef UK_TMA_WARPSPEC
#define UK_TMA_WARPSPEC 0
#endif
#ifndef UK_REDUCE_SMEM_KB
#define UK_REDUCE_SMEM_KB 4
#endif
#define UK_REDUCE_SMEM_BYTES (UK_REDUCE_SMEM_KB * 1024)

namespace UKernel {
namespace Device {
namespace {

// Vector type: 32B on SM80+, 16B otherwise
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
using Vec = ulonglong4;
static constexpr int kVEC_BYTES = 32;
#else
using Vec = uint4;
static constexpr int kVEC_BYTES = 16;
#endif

}  // anonymous namespace

template <typename T>
__device__ __forceinline__ void copy(void* dst, void const* src, size_t count,
                                     void* smem_buf) {
  int tid = threadIdx.x;
  int nthread = blockDim.x;
  size_t bytes = count * sizeof(T);

  // TMA path: small messages up to 4KB (one-shot load+store), and large
  // messages via chunked cp.async.bulk (one smem buffer sized to the full
  // dynamic-smem budget, so a 224KB build moves 224KB per chunk — twice
  // the reduce chunk). The in-place allreduce all-gather's Tmp->Output
  // shard copy (128MB/rank) is exactly this path; it replaced a plain
  // vectorized loop (~680 GB/s measured) with deeper async pipelining.
  if (is_tma_supported() && smem_buf != nullptr &&
      (reinterpret_cast<uintptr_t>(dst) & 0xF) == 0 &&
      (reinterpret_cast<uintptr_t>(src) & 0xF) == 0) {
    constexpr size_t kChunkBytes =
        (UK_REDUCE_SMEM_BYTES - sizeof(TmaSemaphore)) &
        ~static_cast<size_t>(31);
    // cp.async.bulk requires a multiple-of-16 size; a 4-byte allreduce
    // barrier (ncclBarrier) previously hit this with bytes=4 and crashed
    // the GPU context. Fall back to the plain path for odd sizes.
    if (bytes <= 4096 && bytes % 16 == 0) {
      if (tid == 0) {
        TmaSemaphore sem;
        tma_init_semaphore(sem, 1);
        tma_load<T>(smem_buf, src, bytes, sem);
        tma_wait_group<0>();
        tma_store<T>(dst, smem_buf, bytes);
      }
      __syncthreads();
      return;
    }
    if (kChunkBytes >= 32) {
      char* smem = static_cast<char*>(smem_buf);
      T* dst_t = static_cast<T*>(dst);
      T const* src_t = static_cast<T const*>(src);
      TmaSemaphore* sem = reinterpret_cast<TmaSemaphore*>(smem + kChunkBytes);
      size_t off = 0;
      while (off + kChunkBytes <= bytes) {
        if (tid == 0) {
          tma_init_semaphore(*sem, 1);
          tma_load<T>(smem, src_t + off / sizeof(T), kChunkBytes, *sem);
        }
        __syncthreads();
        if (tid == 0) {
          tma_wait(*sem, 0);
          tma_store<T>(dst_t + off / sizeof(T), smem, kChunkBytes);
          tma_wait_group<0>();
          tma_fence_async_global();
        }
        __syncthreads();
        off += kChunkBytes;
      }
      if (off < bytes) {
        // Tail (< kChunkBytes): vectorized loop, same as below.
        constexpr int NELTS_PER_VEC = kVEC_BYTES / (int)sizeof(T);
        size_t nvec = (bytes - off) / (sizeof(T) * NELTS_PER_VEC);
        Vec const* src_v = reinterpret_cast<Vec const*>(
            reinterpret_cast<char const*>(src_t) + off);
        Vec* dst_v = reinterpret_cast<Vec*>(
            reinterpret_cast<char*>(dst_t) + off);
        for (size_t vi = tid; vi < nvec; vi += nthread)
          dst_v[vi] = src_v[vi];
        if constexpr (NELTS_PER_VEC > 1) {
          size_t base = off / sizeof(T) + nvec * NELTS_PER_VEC;
          for (size_t i = base + tid; i < count; i += nthread)
            dst_t[i] = src_t[i];
        }
      }
      return;
    }
  }

  // Vectorized copy (kVEC_BYTES-byte loads through read-only cache)
  constexpr int NELTS_PER_VEC = kVEC_BYTES / (int)sizeof(T);
  size_t nvec = count / NELTS_PER_VEC;

  Vec const* src_v = reinterpret_cast<Vec const*>(src);
  Vec* dst_v = reinterpret_cast<Vec*>(dst);

  for (size_t vi = tid; vi < nvec; vi += nthread) dst_v[vi] = src_v[vi];

  // Scalar tail
  if constexpr (NELTS_PER_VEC > 1) {
    size_t base = nvec * NELTS_PER_VEC;
    T* dst_t = static_cast<T*>(dst);
    T const* src_t = static_cast<T const*>(src);
    for (size_t i = base + tid; i < count; i += nthread) dst_t[i] = src_t[i];
  }
}

// 16B-aligned typed vector for wide reduce loads/stores. All supported
// reduce dtypes have sizeof(T) in {1,2,4,8}, so N = 16/sizeof(T) is
// 16/8/4/2 and the struct is exactly 16B.
template <typename T, int N>
struct alignas(16) TypedVec {
  T e[N];
};

// Bulk vectorized dst[i] = dst[i] op src[i], compile-time op so the
// element loop fully unrolls and memory traffic stays 16B wide per
// thread (NCCL-style wide reduce; the previous scalar loop was neither
// vectorized nor coalesced).
template <typename T, int N, ReduceType op>
__device__ __forceinline__ void vec_read_reduce_store(T* dst, T const* src,
                                                      size_t count, int tid,
                                                      int nthread) {
  using V = TypedVec<T, N>;
  V const* svp = reinterpret_cast<V const*>(src);
  V* dvp = reinterpret_cast<V*>(dst);
  size_t nvec = count / static_cast<size_t>(N);
  // ILP: each thread keeps kVecInFlight independent 16B loads in flight
  // before storing. The element-wise reduce is latency-bound at low
  // unroll (measured: U=1 needs ~16 blocks for 150 GB/s, U=4 reaches
  // ~185 GB/s with 8 blocks on A40 — the DRAM ceiling), so fewer blocks
  // saturate the memory system. U = UK_REDUCE_ILP: B300's HBM3e
  // latency-bandwidth product is ~5x A40's, so it needs more bytes in
  // flight per block (rebuild with REDUCE_ILP=8/16 to sweep).
  constexpr int kVecInFlight = UK_REDUCE_ILP;
  size_t stride = static_cast<size_t>(nthread) * kVecInFlight;
  for (size_t base = static_cast<size_t>(tid); base < nvec;
       base += stride) {
    V sv[kVecInFlight];
    V dv[kVecInFlight];
    size_t idx[kVecInFlight];
#pragma unroll
    for (int u = 0; u < kVecInFlight; ++u)
      idx[u] = base + static_cast<size_t>(u) * nthread;
#pragma unroll
    for (int u = 0; u < kVecInFlight; ++u)
      if (idx[u] < nvec) sv[u] = svp[idx[u]];
#pragma unroll
    for (int u = 0; u < kVecInFlight; ++u)
      if (idx[u] < nvec) {
        dv[u] = dvp[idx[u]];
#pragma unroll
        for (int e = 0; e < N; ++e)
          dv[u].e[e] = apply_reduce(dv[u].e[e], sv[u].e[e], op);
      }
#pragma unroll
    for (int u = 0; u < kVecInFlight; ++u)
      if (idx[u] < nvec) dvp[idx[u]] = dv[u];
  }
  size_t base = nvec * static_cast<size_t>(N);
  for (size_t i = base + tid; i < count; i += nthread)
    dst[i] = apply_reduce(dst[i], src[i], op);
}

#if __CUDA_ARCH__ >= 900 && UK_TMA_REDUCE
// Chunked cp.async.bulk reduce for large tasks: per chunk, bulk-load src
// and dst into shared memory (mbarrier-tracked), reduce in smem, bulk-
// store dst back. In-flight bytes per block are bounded by smem, not by
// per-thread registers — this is the lever to reach native-class
// per-block throughput at low block counts. TmaSemaphores are carved out
// of the same smem buffer (mbarriers must live in shared memory).
template <typename T, ReduceType op>
__device__ __forceinline__ void tma_bulk_reduce_chunk(
    T* dst_t, T const* src_t, char* smem, size_t off_bytes, size_t len_bytes,
    int tid, int nthread) {
  constexpr size_t kChunkBytes =
      ((UK_REDUCE_SMEM_BYTES - 2 * sizeof(TmaSemaphore)) / 2) &
      ~static_cast<size_t>(31);
  T* smem_src = reinterpret_cast<T*>(smem);
  T* smem_dst = reinterpret_cast<T*>(smem + kChunkBytes);
  TmaSemaphore* sem_src = reinterpret_cast<TmaSemaphore*>(
      smem + 2 * kChunkBytes);
  TmaSemaphore* sem_dst = reinterpret_cast<TmaSemaphore*>(
      smem + 2 * kChunkBytes + sizeof(TmaSemaphore));
  size_t const e0 = off_bytes / sizeof(T);

  // mbarrier.init per chunk (count=1, phase 0). Phase-toggling across
  // chunks hangs at high chunk counts (observed at 512 chunks/tile), and
  // the barrier must be re-armed for every arrive.expect_tx — a fresh
  // init per chunk is what the earlier runs validated.
  if (tid == 0) {
    tma_init_semaphore(*sem_src, 1);
    tma_init_semaphore(*sem_dst, 1);
  }
  __syncthreads();
  if (tid == 0) {
    tma_load<T>(smem_src, src_t + e0, len_bytes, *sem_src);
    tma_load<T>(smem_dst, dst_t + e0, len_bytes, *sem_dst);
  }
  __syncthreads();
  if (tid == 0) {
    tma_wait(*sem_src, 0);
    tma_wait(*sem_dst, 0);
  }
  __syncthreads();
  size_t const n = len_bytes / sizeof(T);
  // Vectorize the smem reduce: one 16B TypedVec per thread per iteration
  // instead of scalar element ops (4x fewer smem accesses / loop iters on
  // fp32). kChunkBytes is 32B-aligned so both chunk buffers are 16B
  // aligned for TypedVec; the scalar tail covers any remainder.
  constexpr int kVec = 16 / static_cast<int>(sizeof(T));
  if (kVec > 1) {
    using V = TypedVec<T, kVec>;
    V* sv = reinterpret_cast<V*>(smem_src);
    V* dv = reinterpret_cast<V*>(smem_dst);
    size_t const nvec = n / static_cast<size_t>(kVec);
    for (size_t i = static_cast<size_t>(tid); i < nvec;
         i += static_cast<size_t>(nthread)) {
      V const s = sv[i];
      V d = dv[i];
#pragma unroll
      for (int e = 0; e < kVec; ++e)
        d.e[e] = apply_reduce(d.e[e], s.e[e], op);
      dv[i] = d;
    }
    for (size_t i = nvec * static_cast<size_t>(kVec) +
                    static_cast<size_t>(tid);
         i < n; i += static_cast<size_t>(nthread))
      smem_dst[i] = apply_reduce(smem_dst[i], smem_src[i], op);
  } else {
    for (size_t i = static_cast<size_t>(tid); i < n;
         i += static_cast<size_t>(nthread))
      smem_dst[i] = apply_reduce(smem_dst[i], smem_src[i], op);
  }
  __syncthreads();
  if (tid == 0) {
    tma_store<T>(dst_t + e0, smem_dst, len_bytes);
    // Wait for THIS thread's bulk-store group before the next chunk's
    // TMA loads reuse the smem buffer (non-issuing threads have no group
    // and return immediately; the barrier below fences the reuse).
    tma_wait_group<0>();
    // Make the async-proxy writes visible to generic-proxy agents (host
    // memsets, other blocks' loads) — without this, a store can land after
    // a later memset/load and corrupt data (observed: warmup's store raced
    // the pre-timing memset, leaving dst = warmup + rounds instead of
    // rounds).
    tma_fence_async_global();
  }
  __syncthreads();
}

#if __CUDA_ARCH__ >= 900 && UK_TMA_REDUCE && UK_TMA_WARPSPEC
#ifndef UK_WARPSPEC_DEBUG
#define UK_WARPSPEC_DEBUG 0
#endif

// Debug: print pipeline progress from block 0 (first few + last chunks).
// Not for production use; enabled with -DUK_WARPSPEC_DEBUG=1.
__device__ __forceinline__ void ws_dbg(int bid, int c, int nfull,
                                       char const* tag) {
#if UK_WARPSPEC_DEBUG
  if (bid == 0 && (c < 6 || c >= nfull - 2) && c >= 0)
    printf("[ws] %s c=%d/%d\n", tag, c, nfull);
#endif
}

// Warp-spec pipeline depth. 4 slots gives the deepest overlap but shrinks
// the per-stage chunk to (smem - slots*2*barriers) / (2*slots), and every
// chunk pays fixed TMA-op + barrier costs — at 224KB, 4 slots = 28.6KB
// chunks and the fixed overhead dominates (measured ~5.4us/chunk -> 178
// GB/s at BLK=32 vs 443 GB/s single-buffer). Fewer slots amortize the
// overhead over bigger chunks; sweep 4/3/2/1 to find the knee.
constexpr int kWSNSlots = 4;
constexpr size_t kWSChunkBytes =
    ((UK_REDUCE_SMEM_BYTES - kWSNSlots * 2 * sizeof(TmaSemaphore)) /
     (2 * kWSNSlots)) &
    ~static_cast<size_t>(31);

// Warp-specialized TMA reduce: producer warp (warp 0, lane 0) issues the
// bulk loads and stores while consumer warps (1..7) reduce in shared
// memory — load[N+kNSlots]/store[N] overlap reduce[N] continuously
// (FlashAttention-3 / CUTLASS / DeepEP producer-consumer pattern).
//
// Synchronization protocol (canonical; see
// docs/warp_spec_reduce_design.md): all mbarriers are initialized ONCE
// per task, then every use toggles the phase. Chunk c uses slot
// s = c % kNSlots for the (c / kNSlots)-th time, so both full[s] and
// done[s] complete exactly that many times and the parity to wait on is
// (c / kNSlots) & 1 — phases are derived from the chunk index, no
// per-stage state needed. The previous re-init-per-chunk version was
// racy: without __syncthreads() around every reuse, a consumer parked in
// try_wait can observe a freshly re-initialized barrier.
__device__ __forceinline__ void ws_slot_init(char* slot) {
  constexpr size_t kChunkBytes = kWSChunkBytes;
  TmaSemaphore* ready =
      reinterpret_cast<TmaSemaphore*>(slot + 2 * kChunkBytes);
  TmaSemaphore* cdone = reinterpret_cast<TmaSemaphore*>(
      slot + 2 * kChunkBytes + sizeof(TmaSemaphore));
  // ready receives TWO arrive.expect_tx per use (src + dst loads), so its
  // count is 2; cdone receives exactly one arrive per use (count=1). An
  // arrive beyond count carries into the next phase and corrupts it.
  tma_init_semaphore(*ready, 2);
  tma_init_semaphore(*cdone, 1);
}

__device__ __forceinline__ void ws_fence_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

template <typename T>
__device__ __forceinline__ void ws_slot_load(T* dst_t, T const* src_t,
                                             char* slot, size_t off_bytes,
                                             size_t len_bytes) {
  constexpr size_t kChunkBytes = kWSChunkBytes;
  T* smem_src = reinterpret_cast<T*>(slot);
  T* smem_dst = reinterpret_cast<T*>(slot + kChunkBytes);
  TmaSemaphore* ready =
      reinterpret_cast<TmaSemaphore*>(slot + 2 * kChunkBytes);
  size_t const e0 = off_bytes / sizeof(T);
  tma_load<T>(smem_src, src_t + e0, len_bytes, *ready);
  tma_load<T>(smem_dst, dst_t + e0, len_bytes, *ready);
}

__device__ __forceinline__ void ws_slot_wait_ready(char* slot, int phase) {
  constexpr size_t kChunkBytes = kWSChunkBytes;
  TmaSemaphore* ready =
      reinterpret_cast<TmaSemaphore*>(slot + 2 * kChunkBytes);
  tma_wait(*ready, phase);
}

__device__ __forceinline__ void ws_slot_arrive_done(char* slot) {
  constexpr size_t kChunkBytes = kWSChunkBytes;
  TmaSemaphore* cdone = reinterpret_cast<TmaSemaphore*>(
      slot + 2 * kChunkBytes + sizeof(TmaSemaphore));
  uint32_t sem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(cdone));
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(sem_ptr));
}

__device__ __forceinline__ void ws_slot_wait_done(char* slot, int phase) {
  constexpr size_t kChunkBytes = kWSChunkBytes;
  TmaSemaphore* cdone = reinterpret_cast<TmaSemaphore*>(
      slot + 2 * kChunkBytes + sizeof(TmaSemaphore));
  tma_wait(*cdone, phase);
}

template <typename T>
__device__ __forceinline__ void ws_slot_store(T* dst_t, char* slot,
                                              size_t off_bytes,
                                              size_t len_bytes) {
  constexpr size_t kChunkBytes = kWSChunkBytes;
  T* smem_dst = reinterpret_cast<T*>(slot + kChunkBytes);
  size_t const e0 = off_bytes / sizeof(T);
  tma_store<T>(dst_t + e0, smem_dst, len_bytes);
  tma_wait_group<0>();
  tma_fence_async_global();
}

template <typename T, ReduceType op>
__device__ __forceinline__ void ws_slot_reduce(char* slot, size_t len_bytes,
                                               int consumer_tid,
                                               int nconsumer) {
  constexpr size_t kChunkBytes = kWSChunkBytes;
  T* smem_src = reinterpret_cast<T*>(slot);
  T* smem_dst = reinterpret_cast<T*>(slot + kChunkBytes);
  size_t const n = len_bytes / sizeof(T);
  for (size_t i = static_cast<size_t>(consumer_tid); i < n; i += nconsumer)
    smem_dst[i] = apply_reduce(smem_dst[i], smem_src[i], op);
}

template <typename T, ReduceType op>
__device__ __forceinline__ bool tma_bulk_reduce_warp_spec(
    void* dst, void const* src, size_t count, void* smem_buf) {
  constexpr int kNSlots = kWSNSlots;
  constexpr size_t kChunkBytes = kWSChunkBytes;
  constexpr size_t kSlotBytes = 2 * kChunkBytes + 2 * sizeof(TmaSemaphore);
  // Not usable (smem too small): caller falls back to the single-buffered
  // path.
  if (kChunkBytes < 32) return false;

  char* smem = static_cast<char*>(smem_buf);
  T* dst_t = static_cast<T*>(dst);
  T const* src_t = static_cast<T const*>(src);
  size_t const bytes = count * sizeof(T);
  size_t const nfull = bytes / kChunkBytes;

  int const tid = threadIdx.x;
  int const nthread = blockDim.x;
  // The consumer named barrier count is a compile-time 224 (7 warps); the
  // persistent kernel always launches 256 threads. Anything else falls
  // back to the single-buffered path below.
  if (nthread != 256) return false;
  int const warp = tid >> 5;
  int const lane = tid & 31;
  int const nconsumer = nthread - 32;
  int const consumer_tid = tid - 32;

  // Init all 8 mbarriers once per task; fence + block barrier make them
  // visible to every warp before the pipeline starts.
  if (warp == 0 && lane == 0)
    for (int s = 0; s < kNSlots; ++s)
      ws_slot_init(smem + s * kSlotBytes);
  ws_fence_init();
  __syncthreads();

  if (nfull > 0) {
    if (warp == 0) {
      if (lane == 0) {
        for (int c = 0; c < kNSlots && c < static_cast<int>(nfull); ++c)
          ws_slot_load<T>(dst_t, src_t, smem + (c % kNSlots) * kSlotBytes,
                          c * kChunkBytes, kChunkBytes);
        ws_dbg(blockIdx.x, kNSlots - 1, static_cast<int>(nfull), "P prefill");
        for (int c = 0; c < static_cast<int>(nfull); ++c) {
          char* slot = smem + (c % kNSlots) * kSlotBytes;
          int const phase = static_cast<int>((c / kNSlots) & 1);
          ws_dbg(blockIdx.x, c, static_cast<int>(nfull), "P wait_done");
          ws_slot_wait_done(slot, phase);
          ws_dbg(blockIdx.x, c, static_cast<int>(nfull), "P store");
          ws_slot_store<T>(dst_t, slot, c * kChunkBytes, kChunkBytes);
          int const c2 = c + kNSlots;
          if (c2 < static_cast<int>(nfull)) {
            ws_slot_load<T>(dst_t, src_t, slot, c2 * kChunkBytes, kChunkBytes);
            ws_dbg(blockIdx.x, c2, static_cast<int>(nfull), "P load");
          }
        }
      }
    } else {
      for (int c = 0; c < static_cast<int>(nfull); ++c) {
        char* slot = smem + (c % kNSlots) * kSlotBytes;
        int const phase = static_cast<int>((c / kNSlots) & 1);
        if (warp == 1 && lane == 0)
          ws_dbg(blockIdx.x, c, static_cast<int>(nfull), "C wait_ready");
        ws_slot_wait_ready(slot, phase);
        if (warp == 1 && lane == 0)
          ws_dbg(blockIdx.x, c, static_cast<int>(nfull), "C reduce");
        ws_slot_reduce<T, op>(slot, kChunkBytes, consumer_tid, nconsumer);
        // All 224 consumers finished the stage; exactly ONE thread then
        // signals cdone (count=1) — a per-warp arrive would over-arrive
        // and corrupt the next phase. bar.sync's count must be immediate.
        asm volatile("bar.sync 1, 224;\n" ::: "memory");
        if (warp == 1 && lane == 0) {
          ws_dbg(blockIdx.x, c, static_cast<int>(nfull), "C arrive");
          ws_slot_arrive_done(slot);
        }
      }
    }
  }

  // Tail (< kChunkBytes) via the ILP vector path on all threads; also
  // covers the whole buffer when nfull == 0 (odd/undersized slices).
  size_t off = nfull * kChunkBytes;
  if (off < bytes) {
    constexpr int kVec = 16 / static_cast<int>(sizeof(T));
    size_t tail_count = (bytes - off) / sizeof(T);
    if (kVec > 1 && (reinterpret_cast<uintptr_t>(dst_t + off / sizeof(T)) &
                     0xF) == 0 &&
        (reinterpret_cast<uintptr_t>(src_t + off / sizeof(T)) & 0xF) == 0) {
      vec_read_reduce_store<T, kVec, op>(dst_t + off / sizeof(T),
                                         src_t + off / sizeof(T), tail_count,
                                         tid, nthread);
    } else {
      for (size_t i = off / sizeof(T); i < count; ++i)
        dst_t[i] = apply_reduce(dst_t[i], src_t[i], op);
    }
  }
  return true;
}
#endif

template <typename T, ReduceType op>
__device__ __forceinline__ void tma_bulk_reduce(void* dst, void const* src,
                                                size_t count,
                                                void* smem_buf) {
#if __CUDA_ARCH__ >= 900 && UK_TMA_REDUCE && UK_TMA_WARPSPEC
  if (tma_bulk_reduce_warp_spec<T, op>(dst, src, count, smem_buf)) return;
#endif
  constexpr size_t kChunkBytes =
      ((UK_REDUCE_SMEM_BYTES - 2 * sizeof(TmaSemaphore)) / 2) &
      ~static_cast<size_t>(31);
  if (kChunkBytes < 32) return;  // smem too small; caller falls back

  char* smem = static_cast<char*>(smem_buf);
  int tid = threadIdx.x;
  int nthread = blockDim.x;

  T* dst_t = static_cast<T*>(dst);
  T const* src_t = static_cast<T const*>(src);
  size_t bytes = count * sizeof(T);

  size_t off = 0;
  while (off + kChunkBytes <= bytes) {
    tma_bulk_reduce_chunk<T, op>(dst_t, src_t, smem, off, kChunkBytes, tid,
                                 nthread);
    off += kChunkBytes;
  }
  if (off < bytes) {
    // Tail (< kChunkBytes): TMA bulk on odd-sized final chunks produced
    // garbage (observed: 130KB wrong clusters per 512KB block slice,
    // starting at the tail chunk). Fall back to the ILP vector path —
    // the tail is at most one chunk, so the throughput impact is nil.
    constexpr int kVec = 16 / static_cast<int>(sizeof(T));
    size_t tail_count = (bytes - off) / sizeof(T);
    if (kVec > 1 && (reinterpret_cast<uintptr_t>(dst_t + off / sizeof(T)) &
                     0xF) == 0 &&
        (reinterpret_cast<uintptr_t>(src_t + off / sizeof(T)) & 0xF) == 0) {
      vec_read_reduce_store<T, kVec, op>(dst_t + off / sizeof(T),
                                         src_t + off / sizeof(T), tail_count,
                                         tid, nthread);
    } else {
      for (size_t i = off / sizeof(T); i < count; ++i)
        dst_t[i] = apply_reduce(dst_t[i], src_t[i], op);
    }
  }
}
#endif

template <typename T, ReduceType op>
__device__ __forceinline__ void read_reduce_store_op(void* dst, void const* src,
                                                     size_t count,
                                                     void* smem_buf) {
  int tid = threadIdx.x;
  int nthread = blockDim.x;
  size_t bytes = count * sizeof(T);

  // cp.async.bulk needs multiple-of-16 size and 16B-aligned addresses;
  // the 4-byte ncclBarrier allreduce used to crash here.
  if (is_tma_supported() && smem_buf != nullptr && bytes <= 4096 &&
      bytes % 16 == 0 &&
      (reinterpret_cast<uintptr_t>(dst) & 0xF) == 0 &&
      (reinterpret_cast<uintptr_t>(src) & 0xF) == 0) {
    T* dst_ptr = static_cast<T*>(dst);
    T const* src_ptr = static_cast<T const*>(src);
    T* temp_result = static_cast<T*>(smem_buf);

    if (tid == 0) {
      TmaSemaphore sem_dst;
      tma_init_semaphore(sem_dst, 1);
      tma_load<T>(smem_buf, dst_ptr, bytes, sem_dst);
      tma_wait_group<0>();
    }
    __syncthreads();

    size_t chunk = (count + nthread - 1) / nthread;
    size_t start = tid * chunk;
    size_t end = (tid + 1 == nthread) ? count : start + chunk;
    for (size_t i = start; i < end; ++i)
      temp_result[i] = apply_reduce(temp_result[i], src_ptr[i], op);

    __syncthreads();

    if (tid == 0) tma_store<T>(dst, smem_buf, bytes);
    return;
  }

  T* dst_ptr = static_cast<T*>(dst);
  T const* src_ptr = static_cast<T const*>(src);
#if __CUDA_ARCH__ >= 900 && UK_TMA_REDUCE
  // Large-task TMA bulk path: chunks of smem-sized blocks via cp.async.bulk.
  if (smem_buf != nullptr &&
      (reinterpret_cast<uintptr_t>(dst_ptr) & 0xF) == 0 &&
      (reinterpret_cast<uintptr_t>(src_ptr) & 0xF) == 0) {
    tma_bulk_reduce<T, op>(dst_ptr, src_ptr, count, smem_buf);
    return;
  }
#endif
  constexpr int kVec = 16 / static_cast<int>(sizeof(T));
  if (kVec > 1 &&
      (reinterpret_cast<uintptr_t>(dst_ptr) & 0xF) == 0 &&
      (reinterpret_cast<uintptr_t>(src_ptr) & 0xF) == 0) {
    vec_read_reduce_store<T, kVec, op>(dst_ptr, src_ptr, count, tid, nthread);
  } else {
    // Unaligned pointers (odd tile/block offsets): coalesced scalar path.
    for (size_t i = tid; i < count; i += nthread)
      dst_ptr[i] = apply_reduce(dst_ptr[i], src_ptr[i], op);
  }
}

// dst[i] = dst[i] op src[i] over [0, count). Runtime op dispatched to a
// compile-time specialization so the vector loop fully unrolls.
template <typename T>
__device__ __forceinline__ void read_reduce_store(void* dst, void const* src,
                                                  size_t count, ReduceType op,
                                                  void* smem_buf) {
  switch (op) {
    case ReduceType::Sum:
      read_reduce_store_op<T, ReduceType::Sum>(dst, src, count, smem_buf);
      break;
    case ReduceType::Prod:
      read_reduce_store_op<T, ReduceType::Prod>(dst, src, count, smem_buf);
      break;
    case ReduceType::Max:
      read_reduce_store_op<T, ReduceType::Max>(dst, src, count, smem_buf);
      break;
    case ReduceType::Min:
      read_reduce_store_op<T, ReduceType::Min>(dst, src, count, smem_buf);
      break;
    case ReduceType::BitwiseAnd:
      read_reduce_store_op<T, ReduceType::BitwiseAnd>(dst, src, count,
                                                      smem_buf);
      break;
    default:
      read_reduce_store_op<T, ReduceType::Sum>(dst, src, count, smem_buf);
      break;
  }
}

}  // namespace Device
}  // namespace UKernel
