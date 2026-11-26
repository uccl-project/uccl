#pragma once
// The routing plan, on the GPU.
//
// Routing keys are bounded by destinations, experts, and NRANKS*cap slots.
// Histogram, scan, and scatter replace general-purpose sorting. The caller
// supplies persistent workspace and performs the inter-rank metadata exchanges.
// The ordering exception used by inverse-routing construction is documented
// at planCsrSort.

#include "megamoe_core.cuh"
#include <cub/cub.cuh>

namespace megamoe {

// One block for the whole token scan; T is capped at `cap` (<= 32768) and the
// work is trivial, so the alternative -- a multi-block scan with its own
// cross-block pass -- would cost more in launches than it saves.
constexpr int kPlanBlock = 1024;

// A caller may pad topk_ids with out-of-range sentinels for unused routes.
// Remap round-robin over all experts with weight zero: a zero-weighted row
// contributes nothing, so this is correct, and it keeps every array a fixed
// [T, kTopK] shape instead of forcing a data-dependent compaction. Round-robin
// rather than a single local expert because a capacity-profiling call can send
// topk_ids that are ENTIRELY -1, and collapsing those onto one rank would make
// every peer's transfer zero bytes.
__device__ __forceinline__ int fixId(int id, long long flat) {
  return (id < 0 || id >= kE) ? int(flat % kE) : id;
}

// ---------------------------------------------------------------------------
// 1. The local half: what this rank sends, and where each row lands.
//    Replaces _local()'s scatter, argsort compaction, cumsum and bincount.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kPlanBlock)
    planLocal(int const* __restrict__ topkIds,
              float const* __restrict__ topkWts, int T, int cap,
              int* __restrict__ idsFix, float* __restrict__ wtsFix,
              int* __restrict__ jOf, int* __restrict__ sendTok,
              int* __restrict__ nSend, int* __restrict__ cntFlat) {
  using Scan = cub::BlockScan<int, kPlanBlock>;
  __shared__ typename Scan::TempStorage temp;
  __shared__ int sCnt[kE];
  __shared__ int sRunning;

  for (int e = threadIdx.x; e < kE; e += kPlanBlock) sCnt[e] = 0;
  __syncthreads();

  // Pass 1: fix the sentinels, record the per-token destination mask in jOf as
  // a scratch bitfield, and histogram the experts.
  for (int t = threadIdx.x; t < T; t += kPlanBlock) {
    int want = 0;
    for (int k = 0; k < kTopK; k++) {
      long long const flat = (long long)t * kTopK + k;
      int const id = fixId(topkIds[flat], flat);
      bool const bad = topkIds[flat] < 0 || topkIds[flat] >= kE;
      idsFix[flat] = id;
      wtsFix[flat] = bad ? 0.0f : topkWts[flat];
      want |= 1 << (id / kELoc);
      atomicAdd(&sCnt[id], 1);
    }
    jOf[(size_t)t * NRANKS] =
        want;  // scratch; overwritten per destination below
  }
  __syncthreads();
  for (int e = threadIdx.x; e < kE; e += kPlanBlock)
    atomicAdd(&cntFlat[e], sCnt[e]);

  // Pass 2: per destination, an exclusive scan over tokens gives both the
  // compacted sendTok list and the slot index j -- in ascending token order,
  // which is exactly what the C++ loop over t produces and what the receiver's
  // layout assumes.
  for (int d = 0; d < NRANKS; d++) {
    if (threadIdx.x == 0) sRunning = 0;
    __syncthreads();
    for (int base = 0; base < T; base += kPlanBlock) {
      int const t = base + threadIdx.x;
      int const bit = (t < T) ? ((jOf[(size_t)t * NRANKS] >> d) & 1) : 0;
      int pos = 0, total = 0;
      Scan(temp).ExclusiveSum(bit, pos, total);
      if (t < T && bit) {
        int const j = sRunning + pos;
        sendTok[(size_t)d * cap + j] = t;
        // jOf[t][d] is written after the mask bitfield has been consumed for
        // this token, so d must be walked in ascending order -- it is.
        if (d > 0) jOf[(size_t)t * NRANKS + d] = j;
      }
      __syncthreads();
      if (threadIdx.x == 0) sRunning += total;
      __syncthreads();
    }
    if (threadIdx.x == 0) nSend[d] = sRunning;
    __syncthreads();
  }
  // d == 0 shares its slot with the mask bitfield, so it is recomputed last.
  if (threadIdx.x == 0) sRunning = 0;
  __syncthreads();
  for (int base = 0; base < T; base += kPlanBlock) {
    int const t = base + threadIdx.x;
    int want = 0;
    if (t < T) {
      for (int k = 0; k < kTopK; k++)
        want |= 1 << (idsFix[(long long)t * kTopK + k] / kELoc);
    }
    int const bit = (t < T) ? (want & 1) : 0;
    int pos = 0, total = 0;
    Scan(temp).ExclusiveSum(bit, pos, total);
    if (t < T) jOf[(size_t)t * NRANKS] = bit ? (sRunning + pos) : 0;
    __syncthreads();
    if (threadIdx.x == 0) sRunning += total;
    __syncthreads();
  }
}

// The 4 KB the prologue exchanges: per destination, its kELoc incidence counts
// followed by the dedup'd row count. Flat in T by construction, which is what
// keeps the prologue's cost independent of batch size.
__global__ void planHead(int const* __restrict__ cntFlat,
                         int const* __restrict__ nSend,
                         int* __restrict__ head) {
  int const i = threadIdx.x;
  if (i >= NRANKS * (kELoc + 1)) return;
  int const d = i / (kELoc + 1), c = i % (kELoc + 1);
  head[i] = (c == kELoc) ? nSend[d] : cntFlat[d * kELoc + c];
}

// ---------------------------------------------------------------------------
// 2. Offsets, from the gathered count matrix alone. 16x4 of everything, so one
//    block. Also packs the single D2H payload the host needs.
// ---------------------------------------------------------------------------
__global__ void planOffsets(int const* __restrict__ allCnt,
                            int const* __restrict__ nSendIn, int rank,
                            int* __restrict__ grpCnt, int* __restrict__ expOff,
                            int* __restrict__ grpOff, int* __restrict__ nRecv,
                            int* __restrict__ recvCounts,
                            int* __restrict__ packed) {
  // allCnt[s][d][0..kELoc) incidences, allCnt[s][d][kELoc] dedup'd rows.
  int const stride = kELoc + 1;
  if (threadIdx.x != 0) return;

  int perExp[kELoc];
  for (int e = 0; e < kELoc; e++) {
    int tot = 0;
    for (int g = 0; g < NRANKS; g++) {
      int const s = srcOfGroup(rank, g);
      int const c = allCnt[((size_t)s * NRANKS + rank) * stride + e];
      grpCnt[e * NRANKS + g] = c;
      grpOff[e * NRANKS + g] = tot;
      tot += c;
    }
    perExp[e] = tot;
  }
  int run = 0;
  for (int e = 0; e < kELoc; e++) {
    expOff[e] = run;
    run += perExp[e];
  }
  for (int s = 0; s < NRANKS; s++)
    nRecv[s] = allCnt[((size_t)s * NRANKS + rank) * stride + kELoc];

  // Layout mirrors megamoe_plan.py's pack: nSend, nRecv, expOff, grpOff,
  // grpCnt, totalRows, then sendCounts and recvCounts for all_to_all_single.
  int p = 0;
  for (int d = 0; d < NRANKS; d++) packed[p++] = nSendIn[d];
  for (int s = 0; s < NRANKS; s++) packed[p++] = nRecv[s];
  for (int e = 0; e < kELoc; e++) packed[p++] = expOff[e];
  for (int i = 0; i < kELoc * NRANKS; i++) packed[p++] = grpOff[i];
  for (int i = 0; i < kELoc * NRANKS; i++) packed[p++] = grpCnt[i];
  packed[p++] = run;  // totalRows
  for (int d = 0; d < NRANKS; d++) {
    int c = 0;
    for (int e = 0; e < kELoc; e++)
      c += allCnt[((size_t)rank * NRANKS + d) * stride + e];
    packed[p++] = c;  // sendCounts[d]
  }
  for (int s = 0; s < NRANKS; s++) {
    int c = 0;
    for (int e = 0; e < kELoc; e++)
      c += allCnt[((size_t)s * NRANKS + rank) * stride + e];
    recvCounts[s] = c;
    packed[p++] = c;  // recvCounts[s]
  }
}

// ---------------------------------------------------------------------------
// 3. The sender payload: a counting sort of the N = T*kTopK incidences by
//    global expert id. Replaces torch.argsort(ids, stable=True).
//    Three kernels -- per-block histogram, cross-block scan, ordered scatter.
// ---------------------------------------------------------------------------
constexpr int kSortBlock = 256;

__global__ void planPayloadCount(int const* __restrict__ idsFix, int N,
                                 int* __restrict__ binCount) {
  __shared__ int s[kE];
  for (int e = threadIdx.x; e < kE; e += blockDim.x) s[e] = 0;
  __syncthreads();
  int const lo = blockIdx.x * kSortBlock;
  int const i = lo + threadIdx.x;
  if (i < N) atomicAdd(&s[idsFix[i]], 1);
  __syncthreads();
  for (int e = threadIdx.x; e < kE; e += blockDim.x)
    binCount[(size_t)blockIdx.x * kE + e] = s[e];
}

// One thread per expert: exclusive scan of that expert's per-block counts, on
// top of the expert's own global start. kE threads, kE*nBlocks work.
__global__ void planPayloadScan(int* __restrict__ binCount,
                                int const* __restrict__ cntFlat, int nBlocks) {
  int const e = threadIdx.x;
  if (e >= kE) return;
  __shared__ int start[kE];
  start[e] = 0;
  __syncthreads();
  if (e == 0) {
    int run = 0;
    for (int k = 0; k < kE; k++) {
      start[k] = run;
      run += cntFlat[k];
    }
  }
  __syncthreads();
  int run = start[e];
  for (int b = 0; b < nBlocks; b++) {
    int const c = binCount[(size_t)b * kE + e];
    binCount[(size_t)b * kE + e] = run;
    run += c;
  }
}

__global__ void planPayloadScatter(int const* __restrict__ idsFix,
                                   float const* __restrict__ wtsFix,
                                   int const* __restrict__ jOf,
                                   int const* __restrict__ binStart, int N,
                                   int rank, int cap, int* __restrict__ slotOut,
                                   float* __restrict__ wOut) {
  __shared__ int sCur[kE];
  __shared__ int sId[kSortBlock];
  __shared__ int sPos[kSortBlock];
  for (int e = threadIdx.x; e < kE; e += blockDim.x)
    sCur[e] = binStart[(size_t)blockIdx.x * kE + e];
  int const lo = blockIdx.x * kSortBlock;
  int const n = min(kSortBlock, N - lo);
  if (threadIdx.x < n) sId[threadIdx.x] = idsFix[lo + threadIdx.x];
  __syncthreads();

  // One thread places the chunk in ascending index order. It reads only shared
  // memory, so the serial stretch is ~30 cycles an element, and the blocks run
  // concurrently. Ascending order is what makes this a STABLE counting sort,
  // which is what the receiver's layout assumes.
  if (threadIdx.x == 0)
    for (int i = 0; i < n; i++) sPos[i] = sCur[sId[i]]++;
  __syncthreads();

  if (threadIdx.x < n) {
    int const i = lo + threadIdx.x;
    int const d = sId[threadIdx.x] / kELoc;
    int const t = i / kTopK;
    slotOut[sPos[threadIdx.x]] = rank * cap + jOf[(size_t)t * NRANKS + d];
    wOut[sPos[threadIdx.x]] = wtsFix[i];
  }
}

// ---------------------------------------------------------------------------
// 4. Receiver scatter. The destination of each source's packed block is a pure
//    function of the counts, so this needs no data at all beyond them: one
//    block per (source, expert) pair copies one contiguous run.
// ---------------------------------------------------------------------------
__global__ void planScatter(
    int const* __restrict__ rslot, float const* __restrict__ rw,
    int const* __restrict__ grpCnt, int const* __restrict__ expOff,
    int const* __restrict__ grpOff, int const* __restrict__ recvCounts,
    int rank, int* __restrict__ rowIdx, float* __restrict__ rowW) {
  int const s = blockIdx.x / kELoc;
  int const e = blockIdx.x % kELoc;
  // Where source s's block begins in the received buffer.
  int srcBase = 0;
  for (int q = 0; q < s; q++) srcBase += recvCounts[q];
  // Within it, blocks are expert-major.
  int g = 0;
  for (int q = 0; q < NRANKS; q++)
    if (srcOfGroup(rank, q) == s) g = q;
  int within = 0;
  for (int q = 0; q < e; q++) within += grpCnt[q * NRANKS + g];

  int const n = grpCnt[e * NRANKS + g];
  int const dst = expOff[e] + grpOff[e * NRANKS + g];
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    rowIdx[dst + i] = rslot[srcBase + within + i];
    rowW[dst + i] = rw[srcBase + within + i];
  }
}

// ---------------------------------------------------------------------------
// 5. The reduce CSR: group rows by their (source, token) slot.
//    Key space is NRANKS*cap, too large for a shared histogram, so the scatter
//    uses global atomics and a tiny sort fixes the order afterwards.
// ---------------------------------------------------------------------------
__global__ void planCsrCount(int const* __restrict__ rowIdx, int totalRows,
                             int* __restrict__ cnt) {
  for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < totalRows;
       r += gridDim.x * blockDim.x)
    atomicAdd(&cnt[rowIdx[r]], 1);
}

__global__ void __launch_bounds__(kPlanBlock)
    planCsrScan(int const* __restrict__ cnt, int nSlots,
                int* __restrict__ redOff, int* __restrict__ cur) {
  using Scan = cub::BlockScan<int, kPlanBlock>;
  __shared__ typename Scan::TempStorage temp;
  __shared__ int sRunning;
  if (threadIdx.x == 0) sRunning = 0;
  __syncthreads();
  for (int base = 0; base < nSlots; base += kPlanBlock) {
    int const i = base + threadIdx.x;
    int const c = (i < nSlots) ? cnt[i] : 0;
    int pos = 0, total = 0;
    Scan(temp).ExclusiveSum(c, pos, total);
    if (i < nSlots) {
      redOff[i] = sRunning + pos;
      cur[i] = sRunning + pos;
    }
    __syncthreads();
    if (threadIdx.x == 0) sRunning += total;
    __syncthreads();
  }
  if (threadIdx.x == 0) redOff[nSlots] = sRunning;
}

__global__ void planCsrScatter(int const* __restrict__ rowIdx,
                               float const* __restrict__ rowW, int totalRows,
                               int* __restrict__ cur, int* __restrict__ redRow,
                               float* __restrict__ redW) {
  for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < totalRows;
       r += gridDim.x * blockDim.x) {
    int const p = atomicAdd(&cur[rowIdx[r]], 1);
    redRow[p] = r;
    redW[p] = rowW[r];
  }
}

// atomicAdd above assigns valid but arbitrary order within a slot. The reduce
// that consumes this is a sum, so order does not affect correctness -- but it
// does affect bit-exactness against the torch builder and run-to-run
// reproducibility. Each slot holds at most kTopK rows (a token feeds at most
// kTopK of this rank's experts), so one thread per slot can insertion-sort it
// back to ascending row order for a few dozen cycles.
__global__ void planCsrSort(int const* __restrict__ redOff, int nSlots,
                            int* __restrict__ redRow,
                            float* __restrict__ redW) {
  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nSlots) return;
  int const lo = redOff[i], hi = redOff[i + 1];
  for (int a = lo + 1; a < hi; a++) {
    int const kr = redRow[a];
    float const kw = redW[a];
    int b = a - 1;
    while (b >= lo && redRow[b] > kr) {
      redRow[b + 1] = redRow[b];
      redW[b + 1] = redW[b];
      b--;
    }
    redRow[b + 1] = kr;
    redW[b + 1] = kw;
  }
}

}  // namespace megamoe
