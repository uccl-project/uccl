// NUMA-aware AllReduce for a PCIe-only 4-GPU node, built on MSCCL++.
//
// Topology assumed (4x NVIDIA L4, PCIe Gen4 x16 each, no NVLink, dual socket):
//     NUMA0: GPU0 GPU1     NUMA1: GPU2 GPU3
//   intra-NUMA peer of r = r^1     cross-NUMA peer of r = r^2
//
// Three algorithms. A1 and A2 each come in a bandwidth-oriented (`put`), a
// latency-oriented (LL packet), and a copy-engine (proxy) form; A3 is
// copy-engine only.
//
//   A1 "hier"  intra-NUMA ReduceScatter -> cross-NUMA AllReduce -> intra-NUMA
//   AllGather
//              Every rank moves N/2 bytes per phase, 1.5N total, which is the
//              bandwidth-optimal volume for 4 ranks. Only phase 2 crosses the
//              UPI, so the inter-socket link carries N per direction instead of
//              the 1.5N a ring puts on it.
//
//   A2 "tree"  intra-NUMA reduce to leader -> cross-NUMA reduce between leaders
//              -> broadcast back. Leaders move 2N, so it is not
//              bandwidth-optimal, but the dependency chain is short.
//
//   A3 "ring"  ring ReduceScatter + AllGather over 0 -> 1 -> 3 -> 2 -> 0.
//   Same 1.5N
//              per rank as A1, but the cross-socket traffic is smeared over the
//              whole run instead of concentrated in one phase, so the socket
//              link never becomes the constraint and every PCIe link stays
//              saturated. Wins above 4 MiB. A hierarchical algorithm costs the
//              sum of its per-phase bottlenecks; a ring costs the max over its
//              links.
//
// Every transfer is strictly pairwise: no GPU ever addresses two peers at once.
// That matters here because on this fabric a GPU that holds more than one
// concurrent remote relationship collapses: one GPU writing 2 peer destinations
// from a kernel drops 19.2 -> 3.1 GB/s. Copy engines are exempt.
//
// Build: see Makefile

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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <sys/wait.h>
#include <unistd.h>

#define NRANKS 4

#define CUDACHECK(cmd)                                                   \
  do {                                                                   \
    cudaError_t e = (cmd);                                               \
    if (e != cudaSuccess) {                                              \
      std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e));                               \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

using Chan = mscclpp::MemoryChannelDeviceHandle;
using PortChan = mscclpp::PortChannelDeviceHandle;
using Pkt = mscclpp::LL16Packet;

__device__ mscclpp::DeviceSyncer gSyncer;

// Byte offsets inside the symmetric per-rank buffer. Scratch regions are
// double-buffered on iteration parity so that a rank running ahead can never
// overwrite a slot its peer is still reducing from.
//
// The proxy FIFO packs srcOffset/dstOffset into 32 bits each
// (mscclpp fifo_device.hpp: TriggerBitsOffset = 32), so an offset at or past
// 4 GiB into a registered region wraps silently -- the device-side asserts that
// would catch it are compiled out in release builds. The buffer is therefore
// registered to the proxy as three sub-regions, and the copy-engine kernels use
// region-relative offsets for everything above `scr`:
//     regA = [0, 3N)  data + scr      regB = [3N, 5N) scr2     regC = [5N, 7N)
//     tmp
// so the largest offset any trigger can carry is just under 3N.
struct Layout {
  size_t data;  // in-place AllReduce buffer, N bytes            (regA)
  size_t scr;   // scratch A (A1: both half-slots; A2: from leaf) (regA)
  size_t
      scr2;    // scratch B (A2: from cross leader)             (regB), absolute
  size_t tmp;  // reduce output for the copy-engine forms       (regC), absolute
  size_t scr2Rel;  // scr2 relative to regB, for proxy triggers
  size_t tmpRel;   // tmp  relative to regC, for proxy triggers
  size_t pkt1;     // packet region, phase 1
  size_t pkt2;     // packet region, phase 2
  size_t pkt3;     // packet region, phase 3
  size_t flatScratch;  // four full-size source slots for serial flat exchange
  size_t flatStride;
  size_t flatFlags;
};

__device__ __forceinline__ uint2 f2u(float2 v) {
  return make_uint2(__float_as_uint(v.x), __float_as_uint(v.y));
}
__device__ __forceinline__ float2 u2f(uint2 v) {
  return make_float2(__uint_as_float(v.x), __uint_as_float(v.y));
}

// dst += src, over nelem floats, spread across the whole grid.
__device__ __forceinline__ void addInto(float* dst, float const* src,
                                        size_t nelem, int tid, int nthreads) {
  const size_t n4 = nelem / 4;
  float4* d4 = reinterpret_cast<float4*>(dst);
  float4 const* s4 = reinterpret_cast<float4 const*>(src);
  for (size_t i = tid; i < n4; i += nthreads) {
    float4 a = d4[i], b = s4[i];
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    d4[i] = a;
  }
  for (size_t i = n4 * 4 + tid; i < nelem; i += nthreads) dst[i] += src[i];
}

// dst = a + b. Out-of-place so that `a` can stay readable by an in-flight DMA.
//
// WT selects a write-through store (`st.global.wt`) for the output. It was
// added to test the theory that the handover fence costs what it does because
// it must write back dirty L2. That theory is WRONG: write-through changes
// nothing measurable (st.global.wt.v4.f32 is confirmed in the PTX, so the test
// did run). The fence's cost is its .sys scope, not the data -- see
// publishToEngines. Kept as an instrument; the default is off.
template <bool WT>
__device__ __forceinline__ void addOut(float* dst, float const* a,
                                       float const* b, size_t nelem, int tid,
                                       int nthreads) {
  const size_t n4 = nelem / 4;
  float4* d4 = reinterpret_cast<float4*>(dst);
  float4 const* a4 = reinterpret_cast<float4 const*>(a);
  float4 const* b4 = reinterpret_cast<float4 const*>(b);
  for (size_t i = tid; i < n4; i += nthreads) {
    float4 x = a4[i], y = b4[i];
    x.x += y.x;
    x.y += y.y;
    x.z += y.z;
    x.w += y.w;
    if constexpr (WT)
      __stwt(&d4[i], x);
    else
      d4[i] = x;
  }
  for (size_t i = n4 * 4 + tid; i < nelem; i += nthreads) {
    float v = a[i] + b[i];
    if constexpr (WT)
      __stwt(&dst[i], v);
    else
      dst[i] = v;
  }
}

// ---------------------------------------------------------------------------
// A1: intra-NUMA ReduceScatter -> cross-NUMA AllReduce -> intra-NUMA AllGather
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(1024)
    arHierPut(Chan chIntra, Chan chCross, int rank, size_t nelem, Layout L) {
  int const tid = blockIdx.x * blockDim.x + threadIdx.x;
  int const nth = gridDim.x * blockDim.x;
  const size_t halfN = nelem / 2;
  const size_t halfB = halfN * sizeof(float);
  int const myHalf = rank & 1;  // ranks 0,2 own half 0; ranks 1,3 own half 1
  int const otHalf = myHalf ^ 1;

  char* base = reinterpret_cast<char*>(chIntra.src_);
  float* data = reinterpret_cast<float*>(base + L.data);
  float* slotA = reinterpret_cast<float*>(base + L.scr);
  float* slotB = reinterpret_cast<float*>(base + L.scr + halfB);

  // Phase 1 - intra-NUMA ReduceScatter: hand my peer the half it owns.
  chIntra.put<16, true>(L.scr, L.data + otHalf * halfB, halfB, tid, nth);
  gSyncer.sync(gridDim.x);
  if (tid == 0) {
    chIntra.signal();
    chIntra.wait();
  }
  gSyncer.sync(gridDim.x);
  addInto(data + myHalf * halfN, slotA, halfN, tid, nth);
  gSyncer.sync(gridDim.x);

  // Phase 2 - cross-NUMA AllReduce of the half I own. This is the only UPI
  // traffic.
  chCross.put<16, true>(L.scr + halfB, L.data + myHalf * halfB, halfB, tid,
                        nth);
  gSyncer.sync(gridDim.x);
  if (tid == 0) {
    chCross.signal();
    chCross.wait();
  }
  gSyncer.sync(gridDim.x);
  addInto(data + myHalf * halfN, slotB, halfN, tid, nth);
  gSyncer.sync(gridDim.x);

  // Phase 3 - intra-NUMA AllGather: my half is final, write it straight into
  // the peer.
  chIntra.put<16, true>(L.data + myHalf * halfB, L.data + myHalf * halfB, halfB,
                        tid, nth);
  gSyncer.sync(gridDim.x);
  if (tid == 0) {
    chIntra.signal();
    chIntra.wait();
  }
  gSyncer.sync(gridDim.x);
}

// LL form: the flag carried in each packet replaces every barrier, and each
// thread owns one 8-byte unit end to end, so no grid sync is needed either.
__global__ void __launch_bounds__(1024)
    arHierLL(Chan chIntra, Chan chCross, int rank, size_t nelem, Layout L,
             uint32_t flag) {
  int const tid = blockIdx.x * blockDim.x + threadIdx.x;
  int const nth = gridDim.x * blockDim.x;
  const size_t halfN = nelem / 2;
  const size_t nUnit = halfN / 2;  // one LL16 packet carries 2 floats
  int const myHalf = rank & 1;
  int const otHalf = myHalf ^ 1;

  char* base = reinterpret_cast<char*>(chIntra.src_);
  char* pIntra = reinterpret_cast<char*>(chIntra.dst_);
  char* pCross = reinterpret_cast<char*>(chCross.dst_);
  float* data = reinterpret_cast<float*>(base + L.data);

  float2* mine = reinterpret_cast<float2*>(data + myHalf * halfN);
  float2* other = reinterpret_cast<float2*>(data + otHalf * halfN);

  Pkt* txP1 = reinterpret_cast<Pkt*>(pIntra + L.pkt1);
  Pkt* rxP1 = reinterpret_cast<Pkt*>(base + L.pkt1);
  Pkt* txP2 = reinterpret_cast<Pkt*>(pCross + L.pkt2);
  Pkt* rxP2 = reinterpret_cast<Pkt*>(base + L.pkt2);
  Pkt* txP3 = reinterpret_cast<Pkt*>(pIntra + L.pkt3);
  Pkt* rxP3 = reinterpret_cast<Pkt*>(base + L.pkt3);

  for (size_t i = tid; i < nUnit; i += nth) {
    txP1[i].write(f2u(other[i]), flag);  // ReduceScatter, intra-NUMA
    float2 acc = mine[i];
    float2 b = u2f(rxP1[i].read(flag));
    acc.x += b.x;
    acc.y += b.y;

    txP2[i].write(f2u(acc), flag);  // AllReduce, cross-NUMA
    float2 c = u2f(rxP2[i].read(flag));
    acc.x += c.x;
    acc.y += c.y;
    mine[i] = acc;

    txP3[i].write(f2u(acc), flag);  // AllGather, intra-NUMA
    other[i] = u2f(rxP3[i].read(flag));
  }
}

// ---------------------------------------------------------------------------
// A2: NUMA-local reduce to leader -> cross-NUMA reduce -> broadcast back
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(1024)
    arTreePut(Chan chIntra, Chan chCross, int rank, size_t nelem, Layout L) {
  int const tid = blockIdx.x * blockDim.x + threadIdx.x;
  int const nth = gridDim.x * blockDim.x;
  const size_t bytes = nelem * sizeof(float);
  bool const leader = ((rank & 1) == 0);  // GPU0 leads NUMA0, GPU2 leads NUMA1

  char* base = reinterpret_cast<char*>(chIntra.src_);
  float* data = reinterpret_cast<float*>(base + L.data);
  float* fromLeaf = reinterpret_cast<float*>(base + L.scr);
  float* fromPeer = reinterpret_cast<float*>(base + L.scr2);

  if (!leader) {
    chIntra.put<16, true>(L.scr, L.data, bytes, tid, nth);
    gSyncer.sync(gridDim.x);
    if (tid == 0) {
      chIntra.signal();
      chIntra.wait();
    }
    gSyncer.sync(gridDim.x);
    return;
  }

  // Phase 1 - absorb the leaf and reduce locally.
  if (tid == 0) chIntra.wait();
  gSyncer.sync(gridDim.x);
  addInto(data, fromLeaf, nelem, tid, nth);
  gSyncer.sync(gridDim.x);

  // Phase 2 - the two leaders exchange full buffers across the UPI.
  chCross.put<16, true>(L.scr2, L.data, bytes, tid, nth);
  gSyncer.sync(gridDim.x);
  if (tid == 0) {
    chCross.signal();
    chCross.wait();
  }
  gSyncer.sync(gridDim.x);
  addInto(data, fromPeer, nelem, tid, nth);
  gSyncer.sync(gridDim.x);

  // Phase 3 - broadcast the result back down.
  chIntra.put<16, true>(L.data, L.data, bytes, tid, nth);
  gSyncer.sync(gridDim.x);
  if (tid == 0) chIntra.signal();
  gSyncer.sync(gridDim.x);
}

__global__ void __launch_bounds__(1024)
    arTreeLL(Chan chIntra, Chan chCross, int rank, size_t nelem, Layout L,
             uint32_t flag) {
  int const tid = blockIdx.x * blockDim.x + threadIdx.x;
  int const nth = gridDim.x * blockDim.x;
  const size_t nUnit = nelem / 2;
  bool const leader = ((rank & 1) == 0);

  char* base = reinterpret_cast<char*>(chIntra.src_);
  char* pIntra = reinterpret_cast<char*>(chIntra.dst_);
  char* pCross = reinterpret_cast<char*>(chCross.dst_);
  float2* d = reinterpret_cast<float2*>(base + L.data);

  Pkt* txP1 = reinterpret_cast<Pkt*>(pIntra + L.pkt1);
  Pkt* rxP1 = reinterpret_cast<Pkt*>(base + L.pkt1);
  Pkt* txP2 = reinterpret_cast<Pkt*>(pCross + L.pkt2);
  Pkt* rxP2 = reinterpret_cast<Pkt*>(base + L.pkt2);
  Pkt* txP3 = reinterpret_cast<Pkt*>(pIntra + L.pkt3);
  Pkt* rxP3 = reinterpret_cast<Pkt*>(base + L.pkt3);

  if (!leader) {
    for (size_t i = tid; i < nUnit; i += nth) {
      txP1[i].write(f2u(d[i]), flag);
      d[i] = u2f(rxP3[i].read(flag));
    }
  } else {
    for (size_t i = tid; i < nUnit; i += nth) {
      float2 acc = d[i];
      float2 b = u2f(rxP1[i].read(flag));
      acc.x += b.x;
      acc.y += b.y;  // NUMA-local sum

      txP2[i].write(f2u(acc), flag);
      float2 c = u2f(rxP2[i].read(flag));
      acc.x += c.x;
      acc.y += c.y;  // full sum
      d[i] = acc;

      txP3[i].write(f2u(acc), flag);
    }
  }
}

// ---------------------------------------------------------------------------
// Copy-engine forms. Transfers are handed to the proxy (which issues
// cudaMemcpyAsync, i.e. the DMA engines) while the SMs reduce the chunk that
// arrived previously. Copy engines reach 22.6 GB/s here against 19.2 GB/s for
// SM-issued writes, and overlapping the reduce hides it almost entirely
// (measured: staged and pipelined reaches 97.4% of the no-reduction ceiling,
// against 16x slower for accumulating all peers in one kernel).
// ---------------------------------------------------------------------------
__device__ __forceinline__ size_t chunkElems(size_t total, int nchunk) {
  return (((total + nchunk - 1) / nchunk) + 31) & ~(size_t)31;  // 128 B aligned
}

// Publish this block's stores so the DMA engines see them, then meet the rest
// of the grid. Every thread must fence: a fence on the triggering thread alone
// only orders that thread's own stores, which silently leaves the reduce output
// of the other 12k threads unpublished when the proxy starts reading. fence: 0
// = system scope (safe default), 1 = device scope, 2 = none.
//
// Measured: mode 0 costs ~35 us per call, modes 1 and 2 are free. Modes 1 and 2
// are bit-identical because DeviceSyncer::sync already performs a device-scope
// release/acquire across blocks, so __threadfence() adds nothing on top of it.
// The entire cost is therefore the .sys scope, not the data being ordered --
// write-through stores (WT above) make no difference at all.
//
// Mode 0 is probably redundant: mscclpp's fifo_.push() compiles to
// st.global.release.sys, and release is cumulative, so the chain
//   all threads' writes -> device release/acquire (syncer) -> tid 0
//   -> system release (FIFO push) -> proxy thread
// already orders the data at system scope before the proxy can act on the
// trigger. The default stays at 0 because the correctness harness cannot
// distinguish the three modes (mode 2, with no ordering at all, also passes),
// and for A1 the difference is worth 0-2%. Mode 1 matters for schedules that
// hand over per chunk rather than per phase, where mode 0's cost is multiplied
// by the chunk count.
__device__ __forceinline__ void publishToEngines(int fence) {
  if (fence == 0)
    __threadfence_system();
  else if (fence == 1)
    __threadfence();
  gSyncer.sync(gridDim.x);
}

template <bool WT>
__global__ void __launch_bounds__(1024)
    arHierPort(PortChan chIntra, PortChan chCross, float* base, int rank,
               size_t nelem, Layout L, int nchunk, bool pipelined, int fence) {
  int const tid = blockIdx.x * blockDim.x + threadIdx.x;
  int const nth = gridDim.x * blockDim.x;
  const size_t halfN = nelem / 2;
  const size_t halfB = halfN * sizeof(float);
  int const myHalf = rank & 1;
  int const otHalf = myHalf ^ 1;

  float* data =
      reinterpret_cast<float*>(reinterpret_cast<char*>(base) + L.data);
  float* slotA =
      reinterpret_cast<float*>(reinterpret_cast<char*>(base) + L.scr);
  float* slotB =
      reinterpret_cast<float*>(reinterpret_cast<char*>(base) + L.scr + halfB);
  float* tmp = reinterpret_cast<float*>(reinterpret_cast<char*>(base) + L.tmp);

  const size_t cN = chunkElems(halfN, nchunk);
  int const nc = (int)((halfN + cN - 1) / cN);

  // A proxy copy drains on its own schedule and the *sender* is never told
  // when. So no region may be handed to the engines and then overwritten
  // locally: the buffer each phase sends from is one nothing later writes.
  // Phase 1 therefore reduces out-of-place into tmp, phase 2 sends tmp and
  // reduces into data, and phase 3 sends data. Reducing in place and copying
  // back at the end instead lets the peer read the phase-3 value out of the
  // phase-2 send buffer.
  //
  // `pipelined` decides *when* each chunk's outbound copy is handed over. Note
  // it is not free: handing over per chunk needs a grid-wide
  // __threadfence_system() per chunk rather than one per phase, and on this box
  // that fence costs 30-50us because peer memory forces an L2 write-back to
  // system scope.

  // Phase 1 - intra-NUMA ReduceScatter, chunked. The outbound copy reads
  // data[otHalf] while the reduce writes tmp, so they never overlap.
  if (tid == 0) {
    __threadfence_system();
    for (int c = 0; c < nc; c++) {
      size_t off = (size_t)c * cN, len = min(cN, halfN - off);
      chIntra.putWithSignal(L.scr + off * sizeof(float),
                            L.data + otHalf * halfB + off * sizeof(float),
                            len * sizeof(float));
    }
  }
  for (int c = 0; c < nc; c++) {
    size_t off = (size_t)c * cN, len = min(cN, halfN - off);
    if (tid == 0) chIntra.wait();
    gSyncer.sync(gridDim.x);
    addOut<WT>(tmp + off, data + myHalf * halfN + off, slotA + off, len, tid,
               nth);
    // Phase 2 - cross-NUMA, the only UPI traffic.
    if (pipelined) {
      publishToEngines(fence);
      if (tid == 0)
        chCross.putWithSignal(L.scr + halfB + off * sizeof(float),
                              L.tmpRel + off * sizeof(float),
                              len * sizeof(float));
    } else {
      gSyncer.sync(gridDim.x);
    }
  }

  if (!pipelined) {
    publishToEngines(fence);
    if (tid == 0) {
      for (int c = 0; c < nc; c++) {
        size_t off = (size_t)c * cN, len = min(cN, halfN - off);
        chCross.putWithSignal(L.scr + halfB + off * sizeof(float),
                              L.tmpRel + off * sizeof(float),
                              len * sizeof(float));
      }
    }
  }

  for (int c = 0; c < nc; c++) {
    size_t off = (size_t)c * cN, len = min(cN, halfN - off);
    if (tid == 0) chCross.wait();
    gSyncer.sync(gridDim.x);
    addOut<WT>(data + myHalf * halfN + off, tmp + off, slotB + off, len, tid,
               nth);
    // Phase 3 - intra-NUMA AllGather. data[myHalf] is final and nothing writes
    // it again, so it goes straight to the engines; the inbound half lands in
    // data[otHalf], which no in-flight copy of mine is reading. Safe to start
    // early: every peer phase-1 copy has already landed, or the loop above
    // would still be waiting on it.
    if (pipelined) {
      publishToEngines(fence);
      if (tid == 0)
        chIntra.putWithSignal(L.data + myHalf * halfB + off * sizeof(float),
                              L.data + myHalf * halfB + off * sizeof(float),
                              len * sizeof(float));
    } else {
      gSyncer.sync(gridDim.x);
    }
  }

  if (!pipelined) publishToEngines(fence);
  if (tid == 0) {
    if (!pipelined) {
      for (int c = 0; c < nc; c++) {
        size_t off = (size_t)c * cN, len = min(cN, halfN - off);
        chIntra.putWithSignal(L.data + myHalf * halfB + off * sizeof(float),
                              L.data + myHalf * halfB + off * sizeof(float),
                              len * sizeof(float));
      }
    }
    for (int c = 0; c < nc; c++) chIntra.wait();
  }
  gSyncer.sync(gridDim.x);
}

template <bool WT>
__global__ void __launch_bounds__(1024)
    arTreePort(PortChan chIntra, PortChan chCross, float* base, int rank,
               size_t nelem, Layout L, int nchunk, int fence) {
  int const tid = blockIdx.x * blockDim.x + threadIdx.x;
  int const nth = gridDim.x * blockDim.x;
  bool const leader = ((rank & 1) == 0);

  float* data =
      reinterpret_cast<float*>(reinterpret_cast<char*>(base) + L.data);
  float* fromLeaf =
      reinterpret_cast<float*>(reinterpret_cast<char*>(base) + L.scr);
  float* fromPeer =
      reinterpret_cast<float*>(reinterpret_cast<char*>(base) + L.scr2);
  float* tmp = reinterpret_cast<float*>(reinterpret_cast<char*>(base) + L.tmp);

  const size_t cN = chunkElems(nelem, nchunk);
  int const nc = (int)((nelem + cN - 1) / cN);

  if (!leader) {
    if (tid == 0) {
      __threadfence_system();
      for (int c = 0; c < nc; c++) {
        size_t off = (size_t)c * cN, len = min(cN, nelem - off);
        chIntra.putWithSignal(L.scr + off * sizeof(float),
                              L.data + off * sizeof(float),
                              len * sizeof(float));
      }
      for (int c = 0; c < nc; c++) chIntra.wait();  // the broadcast coming back
    }
    gSyncer.sync(gridDim.x);
    return;
  }

  // Phase 1 - absorb the leaf, reducing each chunk as it lands. Out-of-place
  // into tmp for the same reason as A1: phase 2 sends what this writes.
  for (int c = 0; c < nc; c++) {
    size_t off = (size_t)c * cN, len = min(cN, nelem - off);
    if (tid == 0) chIntra.wait();
    gSyncer.sync(gridDim.x);
    addOut<WT>(tmp + off, data + off, fromLeaf + off, len, tid, nth);
    gSyncer.sync(gridDim.x);
  }

  // Phase 2 - the two leaders exchange full buffers across the UPI, sending
  // from tmp so the reduce can land the final result straight into data.
  publishToEngines(fence);
  if (tid == 0) {
    for (int c = 0; c < nc; c++) {
      size_t off = (size_t)c * cN, len = min(cN, nelem - off);
      chCross.putWithSignal(L.scr2Rel + off * sizeof(float),
                            L.tmpRel + off * sizeof(float),
                            len * sizeof(float));
    }
  }
  for (int c = 0; c < nc; c++) {
    size_t off = (size_t)c * cN, len = min(cN, nelem - off);
    if (tid == 0) chCross.wait();
    gSyncer.sync(gridDim.x);
    addOut<WT>(data + off, tmp + off, fromPeer + off, len, tid, nth);
    gSyncer.sync(gridDim.x);
  }

  // Phase 3 - broadcast the result back down. data is final and nothing writes
  // it again, so it is safe to hand to the engines.
  publishToEngines(fence);
  if (tid == 0) {
    for (int c = 0; c < nc; c++) {
      size_t off = (size_t)c * cN, len = min(cN, nelem - off);
      chIntra.putWithSignal(L.data + off * sizeof(float),
                            L.data + off * sizeof(float), len * sizeof(float));
    }
  }
  gSyncer.sync(gridDim.x);
}

// ---------------------------------------------------------------------------
// A3: ring ReduceScatter + AllGather on the copy engines.
//
// Ring 0 -> 1 -> 3 -> 2 -> 0. On a 2+2 NUMA split *any* directed 4-cycle
// crosses the socket boundary exactly twice, and necessarily once in each
// direction, so the inter-socket link never carries more than one flow per
// direction. That is the entire point. A1's phase 2 puts two flows per
// direction, throttling each to 15.8 GB/s while the PCIe links idle at 77%; a
// ring smears the same traffic over the whole run and stays PCIe-bound
// throughout:
//
//   A1:   (N/2)/20.5 + (N/2)/15.8 + (N/2)/20.5 = 0.0804N -> 12.44 GB/s
//   ring:  1.5N / 20.46                        = 0.0733N -> 13.64 GB/s
//
// Every GPU holds exactly two remote relationships -- sends to succ, receives
// from pred -- so the fan-out rule is respected, and the transport is the copy
// engine rather than the SM path NCCL uses intra-node (22.6 vs 19.2 GB/s
// useful, 92% vs 71% wire efficiency). Ring traffic pattern plus copy-engine
// transport; NCCL takes only the first half.
//
// 2(n-1) = 6 steps. Step `step` sends chunk cSend and receives chunk cSend-1:
// in the first n-1 steps the arrival is reduced in, in the last n-1 it is
// already final and simply lands. Each chunk is cut into `nsub` pieces so the
// reduce of piece j overlaps the transfer of piece j+1 -- which is only
// affordable because the per-handover fence is no longer a system-scope one
// (see publishToEngines).
//
// `flatAg` switches the second half, and is what A5 is. All six ring steps are
// dependent -- step k+1 sends what step k just reduced -- but that is only true
// of the ReduceScatter half. Once the ReduceScatter ends every rank holds one
// final chunk, and the three sends that publish it to the three peers have no
// dependency on one another at all, so they need not be three round trips. With
// flatAg they become three triggers pushed back to back into the proxy FIFO and
// waited on once: 6 dependent round trips become 4.
//
// It costs three destinations where the ring had one, and on this fabric a GPU
// addressing two peers at once collapses. It is safe here for the reason the
// All-to-All is: the proxy drains every trigger from one rank onto a single
// CudaIpcStream, so destinations are visited strictly one at a time, and a
// switch at that granularity is cheap. What is ruinous is a *drifting* second
// destination, which a FIFO cannot produce (docs/alltoall.md).
struct AgChans {
  PortChan c[NRANKS];  // indexed by peer rank; the self slot is never touched
};

__global__ void __launch_bounds__(1024)
    arRingPort(PortChan chSend, PortChan chRecv, AgChans ag, float* base,
               int rank, size_t nelem, Layout L, int pos, int nsub, int fence,
               int flatAg) {
  int const tid = blockIdx.x * blockDim.x + threadIdx.x;
  int const nth = gridDim.x * blockDim.x;

  float* data =
      reinterpret_cast<float*>(reinterpret_cast<char*>(base) + L.data);
  float* scr = reinterpret_cast<float*>(reinterpret_cast<char*>(base) + L.scr);

  // Four chunks, 128 B aligned; chunk c is [c*cN, min((c+1)*cN, nelem)).
  const size_t cN = ((((nelem + NRANKS - 1) / NRANKS) + 31) & ~(size_t)31);
  const size_t cB = cN * sizeof(float);
  const size_t sN = (((cN + nsub - 1) / nsub) + 31) & ~(size_t)31;

  int const nstep = flatAg ? (NRANKS - 1) : 2 * (NRANKS - 1);
  for (int step = 0; step < nstep; step++) {
    bool const rs = (step < NRANKS - 1);
    int const s = rs ? step : step - (NRANKS - 1);
    int const cSend = ((rs ? pos - s : pos + 1 - s) % NRANKS + NRANKS) % NRANKS;
    int const cRecv = ((cSend - 1) % NRANKS + NRANKS) % NRANKS;
    const size_t sOff = (size_t)cSend * cN, rOff = (size_t)cRecv * cN;
    const size_t sLen = sOff >= nelem ? 0 : min(cN, nelem - sOff);
    const size_t rLen = rOff >= nelem ? 0 : min(cN, nelem - rOff);

    // data[cSend] is whatever the previous step reduced; publish it, then hand
    // the whole chunk over piece by piece. Nothing writes data[cSend] again
    // until it has been sent, so it is never a send source under a later write.
    publishToEngines(fence);
    if (tid == 0) {
      for (int j = 0; j < nsub; j++) {
        const size_t o = (size_t)j * sN;
        if (o >= sLen) break;
        const size_t l = min(sN, sLen - o);
        if (rs)  // ReduceScatter: land in the successor's scratch slot for this
                 // step
          chSend.putWithSignal(L.scr + (size_t)s * cB + o * sizeof(float),
                               L.data + (sOff + o) * sizeof(float),
                               l * sizeof(float));
        else  // AllGather: the value is final, write straight into its data
          chSend.putWithSignal(L.data + (sOff + o) * sizeof(float),
                               L.data + (sOff + o) * sizeof(float),
                               l * sizeof(float));
      }
    }

    // The predecessor sends its chunk cSend, which is exactly my cRecv, cut the
    // same way, so the signal counts match piece for piece.
    for (int j = 0; j < nsub; j++) {
      const size_t o = (size_t)j * sN;
      if (o >= rLen) break;
      const size_t l = min(sN, rLen - o);
      if (tid == 0) chRecv.wait();
      if (rs) {
        gSyncer.sync(gridDim.x);
        addInto(data + rOff + o, scr + (size_t)s * cN + o, l, tid, nth);
        gSyncer.sync(gridDim.x);
      }
    }
    if (!rs)
      gSyncer.sync(
          gridDim.x);  // tid 0's waits must land before the kernel exits
  }

  if (flatAg) {
    // The last ReduceScatter step reduced into chunk (pos+1): that is the chunk
    // this rank owns outright, and the one every peer is missing from it.
    int const cOwn = (pos + 1) % NRANKS;
    const size_t oOff = (size_t)cOwn * cN;
    const size_t oLen = oOff >= nelem ? 0 : min(cN, nelem - oOff);

    publishToEngines(fence);
    if (tid == 0) {
      // Peers are walked in rotation, so that at each position in the FIFO the
      // four ranks are addressing four different destinations rather than
      // piling onto one. Same schedule, same reason, as the All-to-All.
      //
      // The trigger is issued even when oLen is zero, because the peer's wait
      // count cannot depend on a length only this rank knows.
      for (int k = 1; k < NRANKS; k++) {
        int const d = (rank + k) % NRANKS;
        ag.c[d].putWithSignal(L.data + oOff * sizeof(float),
                              L.data + oOff * sizeof(float),
                              oLen * sizeof(float));
      }
      // All three are already in flight; one drain, not three round trips.
      for (int k = 1; k < NRANKS; k++)
        ag.c[(rank - k + NRANKS) % NRANKS].wait();
    }
    gSyncer.sync(gridDim.x);
  }
}

// ---------------------------------------------------------------------------
// A4: one-phase all-broadcast. Every rank writes its whole buffer to all three
// peers at once, then reduces the four copies locally. One hop instead of
// three.
//
// Egress is 3N per rank against A1's 1.5N, and LL doubles that again, so this
// is bandwidth-hopeless -- and it addresses three peers from one kernel, which
// is exactly the fan-out pattern that collapses 19.2 -> 2.1 GB/s. Neither
// matters below a few KB: there is no bandwidth to lose, and the collapse is a
// starvation effect that needs a saturated link to show up. What is left is
// latency, where one dependent hop beats three.
//
// Sender r writing to peer r^k uses packet slot k-1, and XOR is symmetric, so
// the receiver finds peer r^k's data in its own slot k-1. No agreement needed.
__global__ void __launch_bounds__(1024)
    arFlatLL(Chan c1, Chan c2, Chan c3, int rank, size_t nelem, Layout L,
             uint32_t flag) {
  int const tid = blockIdx.x * blockDim.x + threadIdx.x;
  int const nth = gridDim.x * blockDim.x;
  const size_t nUnit = nelem / 2;  // one LL16 packet carries 2 floats

  char* base = reinterpret_cast<char*>(c1.src_);
  float2* d = reinterpret_cast<float2*>(base + L.data);

  Pkt* tx0 = reinterpret_cast<Pkt*>(reinterpret_cast<char*>(c1.dst_) + L.pkt1);
  Pkt* tx1 = reinterpret_cast<Pkt*>(reinterpret_cast<char*>(c2.dst_) + L.pkt2);
  Pkt* tx2 = reinterpret_cast<Pkt*>(reinterpret_cast<char*>(c3.dst_) + L.pkt3);
  Pkt* rx0 = reinterpret_cast<Pkt*>(base + L.pkt1);
  Pkt* rx1 = reinterpret_cast<Pkt*>(base + L.pkt2);
  Pkt* rx2 = reinterpret_cast<Pkt*>(base + L.pkt3);

  for (size_t i = tid; i < nUnit; i += nth) {
    float2 v = d[i];
    const uint2 mine = f2u(v);
    tx0[i].write(mine, flag);
    tx1[i].write(mine, flag);
    tx2[i].write(mine, flag);
    float2 a = u2f(rx0[i].read(flag));
    float2 b = u2f(rx1[i].read(flag));
    float2 c = u2f(rx2[i].read(flag));
    v.x += a.x + b.x + c.x;
    v.y += a.y + b.y + c.y;
    d[i] = v;
  }
}

__device__ void releaseFlatFlag(uint32_t* address, uint32_t value) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::
                   "l"(address), "r"(value) : "memory");
}
__device__ uint32_t acquireFlatFlag(uint32_t const* address) {
  uint32_t value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" :
                   "=r"(value) : "l"(address) : "memory");
  return value;
}
__global__ void arFlatSerialFence(Chan c1, Chan c2, Chan c3, int rank,
                                  size_t nelem, Layout L, uint32_t flag) {
  char* base = reinterpret_cast<char*>(c1.src_);
  char* remote[4] = {nullptr, nullptr, nullptr, nullptr};
  remote[rank ^ 1] = reinterpret_cast<char*>(c1.dst_);
  remote[rank ^ 2] = reinterpret_cast<char*>(c2.dst_);
  remote[rank ^ 3] = reinterpret_cast<char*>(c3.dst_);
  float* data = reinterpret_cast<float*>(base + L.data);
  for (int phase = 1; phase < NRANKS; ++phase) {
    int dest = (rank + phase) % NRANKS;
    float* to = reinterpret_cast<float*>(
        remote[dest] + L.flatScratch + size_t(rank) * L.flatStride);
    auto const* in = reinterpret_cast<uint4 const*>(data);
    auto* out = reinterpret_cast<uint4*>(to);
    for (size_t i = threadIdx.x; i < nelem / 4; i += blockDim.x)
      out[i] = in[i];
    __syncthreads();
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int dest = 0; dest < NRANKS; ++dest)
      if (dest != rank)
        releaseFlatFlag(reinterpret_cast<uint32_t*>(remote[dest] + L.flatFlags) + rank,
                        flag);
    auto* ownFlags = reinterpret_cast<uint32_t*>(base + L.flatFlags);
    for (int source = 0; source < NRANKS; ++source) {
      if (source == rank) continue;
      while (acquireFlatFlag(ownFlags + source) != flag)
        __nanosleep(64);
    }
  }
  __syncthreads();
  auto* own = reinterpret_cast<float4*>(data);
  for (size_t i = threadIdx.x; i < nelem / 4; i += blockDim.x) {
    float4 value = own[i];
    for (int source = 0; source < NRANKS; ++source) {
      if (source == rank) continue;
      auto const* scratch = reinterpret_cast<float4 const*>(
          base + L.flatScratch + size_t(source) * L.flatStride);
      float4 remoteValue = scratch[i];
      value.x += remoteValue.x;
      value.y += remoteValue.y;
      value.z += remoteValue.z;
      value.w += remoteValue.w;
    }
    own[i] = value;
  }
}
__global__ void arFlatSerialLL(Chan c1, Chan c2, Chan c3, int rank,
                               size_t nelem, Layout L, uint32_t flag) {
  char* base = reinterpret_cast<char*>(c1.src_);
  char* remote[4] = {nullptr, nullptr, nullptr, nullptr};
  remote[rank ^ 1] = reinterpret_cast<char*>(c1.dst_);
  remote[rank ^ 2] = reinterpret_cast<char*>(c2.dst_);
  remote[rank ^ 3] = reinterpret_cast<char*>(c3.dst_);
  size_t offsets[4] = {0, 0, 0, 0};
  offsets[rank ^ 1] = L.pkt1;
  offsets[rank ^ 2] = L.pkt2;
  offsets[rank ^ 3] = L.pkt3;
  auto* data = reinterpret_cast<float2*>(base + L.data);
  size_t units = nelem / 2;
  for (int phase = 1; phase < NRANKS; ++phase) {
    int dest = (rank + phase) % NRANKS;
    auto* tx = reinterpret_cast<Pkt*>(remote[dest] + offsets[dest]);
    for (size_t i = threadIdx.x; i < units; i += blockDim.x)
      tx[i].write(f2u(data[i]), flag);
    __syncthreads();
  }
  for (size_t i = threadIdx.x; i < units; i += blockDim.x) {
    float2 sum = data[i];
    for (int source = 0; source < NRANKS; ++source) {
      if (source == rank) continue;
      auto const* rx = reinterpret_cast<Pkt const*>(base + offsets[source]);
      float2 value = u2f(rx[i].read(flag));
      sum.x += value.x;
      sum.y += value.y;
    }
    data[i] = sum;
  }
}

// Launch-overhead probe: same launch shape, no work. Bounds every other number.
__global__ void __launch_bounds__(1024) arNull() {}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------
enum Algo {
  A0_NULL = 0,
  A1_SM,
  A1_LL,
  A1_CE,
  A1_CP,
  A2_SM,
  A2_LL,
  A2_CE,
  A3_RING,
  A4_FLAT,
  A5_RSFLAT,
  A6_FLAT_SERIAL,
  A7_FLAT_SERIAL_LL,
  NALGO
};
static char const* kAlgoName[NALGO] = {
    "A0-null",    "A1-hier-sm", "A1-hier-ll",  "A1-hier-ce",
    "A1-hier-cp", "A2-tree-sm", "A2-tree-ll",  "A2-tree-ce",
    "A3-ring-ce", "A4-flat-ll", "A5-rsflat-ce", "A6-flat-serial",
    "A7-flat-serial-ll"};

struct Cfg {
  size_t minBytes = 16;
  size_t maxBytes = 256UL << 20;
  int warmup = 10;
  int iters = 20;
  int blocks = 24;
  int threads = 512;
  int llBlocks = 24;
  int llThreads = 1024;
  size_t llMaxBytes =
      16UL << 10;  // LL doubles the wire bytes; above this `put` wins
  size_t chunkBytes = 1UL << 20;  // copy-engine pipeline granularity
  int fence = 0;              // publishToEngines scope: 0 sys, 1 device, 2 none
  size_t ringMinBytes = 512;  // below this the 4-way chunking degenerates
  size_t ringChunkBytes = 8UL
                          << 20;  // ring piece size; measured optimum, see docs
  size_t flatMaxBytes =
      64UL << 10;     // one-phase all-broadcast: 3N egress, fan-out bound
  size_t serialMaxBytes = 0;
  bool proxy = true;  // -P 0 leaves the proxy threads unstarted (LL/put only)
  bool wt = false;  // write-through the reduce output (measurement instrument:
                    // no effect)
  bool check = true;
};

static size_t roundUp(size_t v, size_t a) { return ((v + a - 1) / a) * a; }

static void runRank(int rank, mscclpp::UniqueId id, Cfg const& cfg) {
  CUDACHECK(cudaSetDevice(rank));

  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, NRANKS);
  bootstrap->initialize(id);
  mscclpp::Communicator comm(bootstrap);

  const size_t N = cfg.maxBytes;  // data region size
  // LL16 doubles the payload, and A4 uses the packet regions too, so they must
  // be sized for whichever cutoff is larger -- otherwise pkt1 overruns into
  // pkt2.
  const size_t pktSlot = roundUp(
      2 * std::max({cfg.llMaxBytes, cfg.flatMaxBytes, cfg.serialMaxBytes}),
      4096);
  // [data N][scr par0/1 2N][scr2 par0/1 2N][tmp par0/1 2N][pkt1 x2][pkt2
  // x2][pkt3 x2]
  const size_t scrBase = N, scr2Base = 3 * N, tmpBase = 5 * N, pktBase = 7 * N;
  const size_t bufBytes = pktBase + 6 * pktSlot;

  // The proxy trigger's offset field is 32 bits wide and the largest sub-region
  // handed to it is regA = 3N. Fail loudly rather than wrap silently.
  if (3 * N > (1UL << 32)) {
    std::fprintf(stderr,
                 "-e %zu too large: proxy offsets are 32-bit, max is %zu\n", N,
                 (1UL << 32) / 3);
    std::exit(1);
  }

  char* buf = nullptr;
  CUDACHECK(cudaMalloc(&buf, bufBytes));
  CUDACHECK(cudaMemset(buf, 0, bufBytes));

  auto regMem = comm.registerMemory(buf, bufBytes, mscclpp::Transport::CudaIpc);
  // Sub-region views for the proxy, so no trigger ever carries an offset that
  // overflows its 32-bit field. See the comment on Layout.
  auto regA = comm.registerMemory(buf, 3 * N, mscclpp::Transport::CudaIpc);
  auto regB =
      comm.registerMemory(buf + scr2Base, 2 * N, mscclpp::Transport::CudaIpc);
  auto regC =
      comm.registerMemory(buf + tmpBase, 2 * N, mscclpp::Transport::CudaIpc);

  int const peers[3] = {rank ^ 1, rank ^ 2,
                        rank ^ 3};  // intra-NUMA, cross-NUMA, cross-NUMA
  std::vector<std::shared_future<mscclpp::Connection>> connF;
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> memF, memAF, memBF;
  for (int p : peers) {
    comm.sendMemory(regMem, p, 0);
    comm.sendMemory(regA, p, 1);
    comm.sendMemory(regB, p, 2);
    memF.push_back(comm.recvMemory(p, 0));
    memAF.push_back(comm.recvMemory(p, 1));
    memBF.push_back(comm.recvMemory(p, 2));
    connF.push_back(comm.connect(mscclpp::Transport::CudaIpc, p));
  }
  std::vector<mscclpp::Connection> conns;
  for (auto& f : connF) conns.push_back(f.get());
  std::vector<std::shared_future<mscclpp::Semaphore>> semF;
  for (int i = 0; i < 3; i++)
    semF.push_back(comm.buildSemaphore(conns[i], peers[i]));

  // Kept alive for the whole run: the device handles below alias their state.
  mscclpp::MemoryChannel chIntraObj(semF[0].get(), memF[0].get(), regMem, buf);
  mscclpp::MemoryChannel chCrossObj(semF[1].get(), memF[1].get(), regMem, buf);
  mscclpp::MemoryChannel chDiagObj(semF[2].get(), memF[2].get(), regMem, buf);
  Chan hIntra = chIntraObj.deviceHandle();
  Chan hCross = chCrossObj.deviceHandle();
  Chan hDiag = chDiagObj.deviceHandle();

  // Proxy-backed channels: the same two peers, but transfers run on the DMA
  // engines. One channel per (destination region, source region) pair; channels
  // sharing a SemaphoreId share its signal/wait counter, so the phase ordering
  // is unaffected by the split.
  auto proxy = std::make_shared<mscclpp::ProxyService>();
  mscclpp::SemaphoreId sidIntra = proxy->buildAndAddSemaphore(comm, conns[0]);
  mscclpp::SemaphoreId sidCross = proxy->buildAndAddSemaphore(comm, conns[1]);
  mscclpp::SemaphoreId sidDiag = proxy->buildAndAddSemaphore(comm, conns[2]);
  mscclpp::MemoryId midLocalA = proxy->addMemory(regA);
  mscclpp::MemoryId midLocalC = proxy->addMemory(regC);
  mscclpp::MemoryId midIntraA = proxy->addMemory(memAF[0].get());
  mscclpp::MemoryId midCrossA = proxy->addMemory(memAF[1].get());
  mscclpp::MemoryId midCrossB = proxy->addMemory(memBF[1].get());
  mscclpp::MemoryId midDiagA = proxy->addMemory(memAF[2].get());
  // A1/A2 phases 1 and 3: data <-> scr, both inside regA.
  mscclpp::PortChannel pcIntraObj =
      proxy->portChannel(sidIntra, midIntraA, midLocalA);
  // A1 phase 2: my tmp (regC) -> peer's scr (regA).
  mscclpp::PortChannel pcCrossTmpObj =
      proxy->portChannel(sidCross, midCrossA, midLocalC);
  // A2 phase 2: my tmp (regC) -> peer's scr2 (regB).
  mscclpp::PortChannel pcCrossS2Obj =
      proxy->portChannel(sidCross, midCrossB, midLocalC);
  // Ring 0 -> 1 -> 3 -> 2 -> 0. succ(r) is r^1 for ranks 0 and 3, r^2
  // otherwise; pred is whichever of the two it is not. Both connections already
  // exist, so the ring needs no extra handshake -- only a send channel (to
  // succ) and a channel whose semaphore belongs to the pred connection, for
  // wait().
  bool const succIsIntra = (rank == 0 || rank == 3);
  const mscclpp::SemaphoreId sidSucc = succIsIntra ? sidIntra : sidCross;
  const mscclpp::SemaphoreId sidPred = succIsIntra ? sidCross : sidIntra;
  const mscclpp::MemoryId midSuccA = succIsIntra ? midIntraA : midCrossA;
  const mscclpp::MemoryId midPredA = succIsIntra ? midCrossA : midIntraA;
  mscclpp::PortChannel pcRingSendObj =
      proxy->portChannel(sidSucc, midSuccA, midLocalA);
  mscclpp::PortChannel pcRingRecvObj =
      proxy->portChannel(sidPred, midPredA, midLocalA);
  PortChan pRingSend = pcRingSendObj.deviceHandle();
  PortChan pRingRecv = pcRingRecvObj.deviceHandle();
  int const ringPos = (rank == 0) ? 0 : (rank == 1) ? 1 : (rank == 2) ? 3 : 2;

  // A5's flat AllGather addresses all three peers, so it needs a data->data
  // channel to each. All three land in regA, the same shape as pcIntraObj, and
  // they reuse the ring's semaphores -- signal and wait are counters, so
  // sharing a SemaphoreId with the ring phase costs nothing as long as every
  // signal is matched by exactly one wait, which the symmetric schedule
  // guarantees.
  mscclpp::PortChannel pcAgObj[NRANKS - 1] = {
      proxy->portChannel(sidIntra, midIntraA, midLocalA),
      proxy->portChannel(sidCross, midCrossA, midLocalA),
      proxy->portChannel(sidDiag, midDiagA, midLocalA)};
  AgChans hAg{};
  for (int i = 0; i < NRANKS - 1; i++)
    hAg.c[peers[i]] = pcAgObj[i].deviceHandle();

  PortChan pIntra = pcIntraObj.deviceHandle();
  PortChan pCrossTmp = pcCrossTmpObj.deviceHandle();
  PortChan pCrossS2 = pcCrossS2Obj.deviceHandle();
  if (cfg.proxy) proxy->startProxy();

  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Host mirrors for correctness checking.
  std::vector<float> hostIn, hostOut;

  auto makeLayout = [&](int parity) {
    Layout L;
    L.data = 0;
    L.scr = scrBase + (size_t)parity * N;
    L.scr2 = scr2Base + (size_t)parity * N;
    L.tmp = tmpBase + (size_t)parity * N;
    L.scr2Rel = L.scr2 - scr2Base;
    L.tmpRel = L.tmp - tmpBase;
    L.pkt1 = pktBase + 0 * pktSlot;
    L.pkt2 = pktBase + 2 * pktSlot;
    L.pkt3 = pktBase + 4 * pktSlot;
    L.flatScratch = scrBase;
    L.flatStride = N;
    L.flatFlags = pktBase + 5 * pktSlot;
    return L;
  };

  uint32_t flag = 1;  // strictly increasing across every launch, so stale
                      // packets never match

  auto launch = [&](Algo a, size_t nelem, int iter) {
    Layout L = makeLayout(iter & 1);
    // Pipeline depth: enough chunks to hide the reduce, few enough to amortise
    // the proxy trigger.
    auto chunksFor = [&](size_t bytes) {
      return (int)std::max<size_t>(
          1, std::min<size_t>(64, bytes / cfg.chunkBytes));
    };
    switch (a) {
      case A1_SM:
        arHierPut<<<cfg.blocks, cfg.threads, 0, stream>>>(hIntra, hCross, rank,
                                                          nelem, L);
        break;
      case A1_LL:
        arHierLL<<<cfg.llBlocks, cfg.llThreads, 0, stream>>>(
            hIntra, hCross, rank, nelem, L, flag++);
        break;
      case A1_CE:
        if (cfg.wt)
          arHierPort<true><<<cfg.blocks, cfg.threads, 0, stream>>>(
              pIntra, pCrossTmp, (float*)buf, rank, nelem, L,
              chunksFor(nelem * sizeof(float) / 2), false, cfg.fence);
        else
          arHierPort<false><<<cfg.blocks, cfg.threads, 0, stream>>>(
              pIntra, pCrossTmp, (float*)buf, rank, nelem, L,
              chunksFor(nelem * sizeof(float) / 2), false, cfg.fence);
        break;
      case A1_CP:
        if (cfg.wt)
          arHierPort<true><<<cfg.blocks, cfg.threads, 0, stream>>>(
              pIntra, pCrossTmp, (float*)buf, rank, nelem, L,
              chunksFor(nelem * sizeof(float) / 2), true, cfg.fence);
        else
          arHierPort<false><<<cfg.blocks, cfg.threads, 0, stream>>>(
              pIntra, pCrossTmp, (float*)buf, rank, nelem, L,
              chunksFor(nelem * sizeof(float) / 2), true, cfg.fence);
        break;
      case A2_SM:
        arTreePut<<<cfg.blocks, cfg.threads, 0, stream>>>(hIntra, hCross, rank,
                                                          nelem, L);
        break;
      case A2_LL:
        arTreeLL<<<cfg.llBlocks, cfg.llThreads, 0, stream>>>(
            hIntra, hCross, rank, nelem, L, flag++);
        break;
      case A2_CE:
        if (cfg.wt)
          arTreePort<true><<<cfg.blocks, cfg.threads, 0, stream>>>(
              pIntra, pCrossS2, (float*)buf, rank, nelem, L,
              chunksFor(nelem * sizeof(float)), cfg.fence);
        else
          arTreePort<false><<<cfg.blocks, cfg.threads, 0, stream>>>(
              pIntra, pCrossS2, (float*)buf, rank, nelem, L,
              chunksFor(nelem * sizeof(float)), cfg.fence);
        break;
      case A3_RING:
      case A5_RSFLAT: {
        // One piece per chunkBytes of chunk, so the reduce of piece j overlaps
        // the transfer of piece j+1. Capped: more pieces only add triggers.
        size_t chunkB = nelem * sizeof(float) / NRANKS;
        int nsub = (int)std::max<size_t>(
            1, std::min<size_t>(16, chunkB / cfg.ringChunkBytes));
        arRingPort<<<cfg.blocks, cfg.threads, 0, stream>>>(
            pRingSend, pRingRecv, hAg, (float*)buf, rank, nelem, L, ringPos,
            nsub, cfg.fence, a == A5_RSFLAT);
        break;
      }
      case A4_FLAT:
        arFlatLL<<<cfg.llBlocks, cfg.llThreads, 0, stream>>>(
            hIntra, hCross, hDiag, rank, nelem, L, flag++);
        break;
      case A6_FLAT_SERIAL:
        arFlatSerialFence<<<1, 256, 0, stream>>>(
            hIntra, hCross, hDiag, rank, nelem, L, flag++);
        break;
      case A7_FLAT_SERIAL_LL:
        arFlatSerialLL<<<1, 256, 0, stream>>>(
            hIntra, hCross, hDiag, rank, nelem, L, flag++);
        break;
      case A0_NULL:
        arNull<<<cfg.llBlocks, cfg.llThreads, 0, stream>>>();
        break;
      default:
        break;
    }
  };

  // Deterministic, position-sensitive fill: rank r contributes
  // (r+1)*(1+(j%97)). Ranks sum to 10, so the expected result is 10*(1+(j%97)).
  auto fill = [&](size_t nelem) {
    hostIn.resize(nelem);
    for (size_t j = 0; j < nelem; j++)
      hostIn[j] = (float)(rank + 1) * (float)(1 + (j % 97));
    CUDACHECK(cudaMemcpy(buf, hostIn.data(), nelem * sizeof(float),
                         cudaMemcpyHostToDevice));
  };
  auto verify = [&](size_t nelem) -> double {
    hostOut.resize(nelem);
    CUDACHECK(cudaMemcpy(hostOut.data(), buf, nelem * sizeof(float),
                         cudaMemcpyDeviceToHost));
    double worst = 0.0;
    long long firstBad = -1, nbad = 0;
    for (size_t j = 0; j < nelem; j++) {
      double want = 10.0 * (double)(1 + (j % 97));
      double err = std::fabs((double)hostOut[j] - want) / want;
      if (err > 1e-5) {
        if (firstBad < 0) firstBad = (long long)j;
        nbad++;
      }
      worst = std::max(worst, err);
    }
    if (firstBad >= 0 && getenv("AR_DEBUG")) {
      size_t j = (size_t)firstBad;
      std::fprintf(
          stderr,
          "[rank %d] bad=%lld/%zu first j=%zu half=%d got=%.1f want=%.1f "
          "lastbadrun_end=%zu\n",
          rank, nbad, nelem, j, (int)(j >= nelem / 2), (double)hostOut[j],
          10.0 * (double)(1 + (j % 97)), (size_t)(firstBad + nbad));
    }
    return worst;
  };

  if (rank == 0) {
    std::printf("# MSCCL++ NUMA-aware AllReduce, %d x L4, float32\n", NRANKS);
    std::printf(
        "# put: %d blocks x %d threads   LL: %d blocks x %d threads   LL "
        "cutoff %zu B   CE chunk %zu B\n",
        cfg.blocks, cfg.threads, cfg.llBlocks, cfg.llThreads, cfg.llMaxBytes,
        cfg.chunkBytes);
    std::printf("#\n# %12s %12s %14s %10s %10s %10s %10s\n", "size(B)", "count",
                "algo", "time(us)", "algbw", "busbw", "maxrelerr");
  }

  for (size_t bytes = cfg.minBytes; bytes <= cfg.maxBytes; bytes *= 2) {
    const size_t nelem = bytes / sizeof(float);
    if (nelem < 4) continue;  // A1 needs >= 1 packet unit per half

    for (int a = 0; a < NALGO; a++) {
      bool const isLL = (a == A1_LL || a == A2_LL || a == A4_FLAT);
      if (isLL && bytes > cfg.flatMaxBytes && a == A4_FLAT) continue;
      if ((a == A6_FLAT_SERIAL || a == A7_FLAT_SERIAL_LL) &&
          bytes > cfg.serialMaxBytes) continue;
      if ((a == A1_LL || a == A2_LL) && bytes > cfg.llMaxBytes) continue;
      if (a == A0_NULL && bytes > cfg.minBytes)
        continue;  // launch floor: measure once
      bool const usesProxy = (a == A1_CE || a == A1_CP || a == A2_CE ||
                              a == A3_RING || a == A5_RSFLAT);
      if (usesProxy && !cfg.proxy) continue;
      if ((a == A3_RING || a == A5_RSFLAT) && bytes < cfg.ringMinBytes)
        continue;

      double worstErr = -1.0;
      double timedSum = 0.0;
      for (int i = 0; i < cfg.warmup + cfg.iters; ++i) {
        fill(nelem);
        bootstrap->barrier();
        auto t0 = std::chrono::steady_clock::now();
        launch((Algo)a, nelem, i);
        CUDACHECK(cudaStreamSynchronize(stream));
        auto t1 = std::chrono::steady_clock::now();
        double us =
            std::chrono::duration<double, std::micro>(t1 - t0).count();
        double relerr = cfg.check && a != A0_NULL ? verify(nelem) : -1.0;
        std::vector<double> all(NRANKS);
        all[rank] = us;
        bootstrap->allGather(all.data(), sizeof(double));
        double worst = *std::max_element(all.begin(), all.end());
        std::vector<double> errs(NRANKS);
        errs[rank] = relerr;
        bootstrap->allGather(errs.data(), sizeof(double));
        worstErr = std::max(worstErr, *std::max_element(errs.begin(), errs.end()));
        if (rank == 0 && i >= cfg.warmup)
          std::printf("# sample,%zu,%s,%d,%.3f,%.3e\n", bytes,
                      kAlgoName[a], i - cfg.warmup + 1, worst,
                      *std::max_element(errs.begin(), errs.end()));
        if (i >= cfg.warmup) timedSum += worst;
      }
      double worst = timedSum / cfg.iters;

      if (rank == 0) {
        double algbw = (double)bytes / (worst * 1e-6) / 1e9;
        std::printf("  %12zu %12zu %14s %10.2f %10.2f %10.2f %10.1e\n", bytes,
                    nelem, kAlgoName[a], worst, algbw, algbw * 1.5, worstErr);
        std::fflush(stdout);
      }
      bootstrap->barrier();
    }
  }

  CUDACHECK(cudaStreamSynchronize(stream));
  bootstrap->barrier();
  if (cfg.proxy) proxy->stopProxy();
  CUDACHECK(cudaStreamDestroy(stream));
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
      cfg.minBytes = std::strtoull(next(), nullptr, 0);
    else if (k == "-e")
      cfg.maxBytes = std::strtoull(next(), nullptr, 0);
    else if (k == "-n")
      cfg.iters = std::atoi(next());
    else if (k == "-w")
      cfg.warmup = std::atoi(next());
    else if (k == "-B")
      cfg.blocks = std::atoi(next());
    else if (k == "-T")
      cfg.threads = std::atoi(next());
    else if (k == "-L")
      cfg.llBlocks = std::atoi(next());
    else if (k == "-t")
      cfg.llThreads = std::atoi(next());
    else if (k == "-l")
      cfg.llMaxBytes = std::strtoull(next(), nullptr, 0);
    else if (k == "-k")
      cfg.chunkBytes = std::strtoull(next(), nullptr, 0);
    else if (k == "-c")
      cfg.check = std::atoi(next()) != 0;
    else if (k == "-F")
      cfg.fence = std::atoi(next());
    else if (k == "-r")
      cfg.ringMinBytes = std::strtoull(next(), nullptr, 0);
    else if (k == "-K")
      cfg.ringChunkBytes = std::strtoull(next(), nullptr, 0);
    else if (k == "-f")
      cfg.flatMaxBytes = std::strtoull(next(), nullptr, 0);
    else if (k == "-S")
      cfg.serialMaxBytes = std::strtoull(next(), nullptr, 0);
    else if (k == "-P")
      cfg.proxy = std::atoi(next()) != 0;
    else if (k == "-W")
      cfg.wt = std::atoi(next()) != 0;
    else {
      std::fprintf(stderr, "unknown arg %s\n", k.c_str());
      return 1;
    }
  }

  // Fork before touching CUDA so every rank gets a clean context. The unique id
  // is created in the parent and inherited, which removes the need for MPI.
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
