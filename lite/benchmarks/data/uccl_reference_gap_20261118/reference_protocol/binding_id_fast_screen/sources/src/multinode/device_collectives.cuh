#pragma once
// Shared verbatim with the standalone GPU hierarchy benchmark. Host/MPI
// orchestration and experimental host-packet algorithms remain in the harness.
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/memory_channel_device.hpp>
#include <mscclpp/packet_device.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <mscclpp/semaphore.hpp>
#include <mscclpp/utils.hpp>
#include <cuda_runtime.h>
#ifdef UCCL_LITE_PACKET_TIMELINE
#include <flashinfer/profiler.cuh>
#endif
namespace uccl_lite::multinode_kernels {
constexpr int kNodes = 2;
constexpr int kLocalRanks = 4;
constexpr int kWorld = 8;
using PortChan = mscclpp::PortChannelDeviceHandle;
using MemChan = mscclpp::MemoryChannelDeviceHandle;
__device__ mscclpp::DeviceSyncer gMultiNodeSyncer;

struct LocalAgChans {
  PortChan c[kLocalRanks];
};
struct LocalDirectChans {
  MemChan c[kLocalRanks];
};

struct Layout {
  size_t data;
  size_t localScratch;
  size_t ibSend;
  size_t ibScratch;
  size_t immutableInput;
};

__device__ __forceinline__ uint4 loadClientQuad(uint32_t const* data,
                                                size_t i) {
  if ((reinterpret_cast<uintptr_t>(data) & 15) == 0)
    return reinterpret_cast<uint4 const*>(data)[i];
  uint4 value;
  asm volatile("ld.global.u32 %0, [%1];"
               : "=r"(value.x)
               : "l"(data + 4 * i)
               : "memory");
  asm volatile("ld.global.u32 %0, [%1];"
               : "=r"(value.y)
               : "l"(data + 4 * i + 1)
               : "memory");
  asm volatile("ld.global.u32 %0, [%1];"
               : "=r"(value.z)
               : "l"(data + 4 * i + 2)
               : "memory");
  asm volatile("ld.global.u32 %0, [%1];"
               : "=r"(value.w)
               : "l"(data + 4 * i + 3)
               : "memory");
  return value;
}
__device__ __forceinline__ void storeClientQuad(uint32_t* data, size_t i,
                                                uint4 value) {
  if ((reinterpret_cast<uintptr_t>(data) & 15) == 0) {
    reinterpret_cast<uint4*>(data)[i] = value;
    return;
  }
  asm volatile("st.global.u32 [%0], %1;" ::"l"(data + 4 * i), "r"(value.x)
               : "memory");
  asm volatile("st.global.u32 [%0], %1;" ::"l"(data + 4 * i + 1), "r"(value.y)
               : "memory");
  asm volatile("st.global.u32 [%0], %1;" ::"l"(data + 4 * i + 2), "r"(value.z)
               : "memory");
  asm volatile("st.global.u32 [%0], %1;" ::"l"(data + 4 * i + 3), "r"(value.w)
               : "memory");
}
__device__ __forceinline__ float4 loadClientFloatQuad(uint32_t const* data,
                                                      size_t i) {
  uint4 v = loadClientQuad(data, i);
  return make_float4(__uint_as_float(v.x), __uint_as_float(v.y),
                     __uint_as_float(v.z), __uint_as_float(v.w));
}
__device__ __forceinline__ uint4 floatQuadBits(float4 v) {
  return make_uint4(__float_as_uint(v.x), __float_as_uint(v.y),
                    __float_as_uint(v.z), __float_as_uint(v.w));
}

// The caller selects the aligned specialization from actual pointer bits.
// All ring chunk/piece offsets preserve 16-byte alignment.
template <bool Aligned>
__device__ __forceinline__ uint4 loadClientQuad(uint32_t const* data,
                                                size_t i) {
  if constexpr (Aligned)
    return reinterpret_cast<uint4 const*>(data)[i];
  else
    return loadClientQuad(data, i);
}
template <bool Aligned>
__device__ __forceinline__ float4 loadClientFloatQuad(uint32_t const* data,
                                                      size_t i) {
  uint4 value = loadClientQuad<Aligned>(data, i);
  return make_float4(__uint_as_float(value.x), __uint_as_float(value.y),
                     __uint_as_float(value.z), __uint_as_float(value.w));
}
template <bool Aligned>
__device__ __forceinline__ void storeClientQuad(uint32_t* data, size_t i,
                                                uint4 value) {
  if constexpr (Aligned)
    reinterpret_cast<uint4*>(data)[i] = value;
  else
    storeClientQuad(data, i, value);
}

// Small-message variant: one resident CTA directly pushes local shards through
// CUDA IPC while the corresponding IB transfer is in flight.
__device__ __forceinline__ void addOut(float* dst, float const* a,
                                       float const* b, size_t n, int tid,
                                       int nth) {
  size_t n4 = n / 4;
  float4* d4 = reinterpret_cast<float4*>(dst);
  float4 const* a4 = reinterpret_cast<float4 const*>(a);
  float4 const* b4 = reinterpret_cast<float4 const*>(b);
  for (size_t i = tid; i < n4; i += nth) {
    float4 x = a4[i], y = b4[i];
    x.x += y.x;
    x.y += y.y;
    x.z += y.z;
    x.w += y.w;
    d4[i] = x;
  }
  for (size_t i = n4 * 4 + tid; i < n; i += nth) dst[i] = a[i] + b[i];
}

__device__ __forceinline__ void addInto(float* dst, float const* src, size_t n,
                                        int tid, int nth) {
  size_t n4 = n / 4;
  float4* d4 = reinterpret_cast<float4*>(dst);
  float4 const* s4 = reinterpret_cast<float4 const*>(src);
  for (size_t i = tid; i < n4; i += nth) {
    float4 x = d4[i], y = s4[i];
    x.x += y.x;
    x.y += y.y;
    x.z += y.z;
    x.w += y.w;
    d4[i] = x;
  }
  for (size_t i = n4 * 4 + tid; i < n; i += nth) dst[i] += src[i];
}

// DeviceSyncer provides the device release/acquire.  The proxy FIFO push is a
// system-scope release, which makes the preceding GPU stores visible to DMA.
__device__ __forceinline__ void publish() { gMultiNodeSyncer.sync(gridDim.x); }

// GPU-driven local ring transport. No host CUDA submission is needed while
// the caller may be blocked inside a pageable D2H copy or other CUDA API.
struct DirectRingChan {
  MemChan channel;
  __device__ void wait() { channel.wait(); }
  __device__ void flush() {}
};
__device__ __forceinline__ void putLocal(PortChan send, size_t dst, size_t src,
                                         size_t bytes, int tid) {
  if (tid == 0) send.putWithSignal(dst, src, bytes);
}
__device__ __forceinline__ void putLocal(DirectRingChan send, size_t dst,
                                         size_t src, size_t bytes, int tid) {
  auto* output = reinterpret_cast<uint4*>(
      reinterpret_cast<char*>(send.channel.dst_) + dst);
  auto const* input = reinterpret_cast<uint4 const*>(
      reinterpret_cast<char const*>(send.channel.src_) + src);
  int stride = gridDim.x * blockDim.x;
  for (size_t i = tid; i < bytes / sizeof(uint4); i += stride)
    output[i] = input[i];
  publish();
  if (tid == 0) send.channel.signal();
}

// Direct RS can read the original local input without a whole-tensor copy.
// Only the registered destination scratch is remotely accessed.
struct ClientRingChan {
  MemChan channel;
  uint32_t const* input;
  size_t inputBytes;
  __device__ void wait() { channel.wait(); }
  __device__ void flush() {}
};
template <class Channel>
__device__ __forceinline__ float const* ringInput(Channel,
                                                  float const* staged) {
  return staged;
}
__device__ __forceinline__ float const* ringInput(ClientRingChan send,
                                                  float const*) {
  return reinterpret_cast<float const*>(send.input);
}
__device__ __forceinline__ void putLocal(ClientRingChan send, size_t dst,
                                         size_t src, size_t bytes, int tid) {
  auto* output = reinterpret_cast<uint4*>(
      reinterpret_cast<char*>(send.channel.dst_) + dst);
  auto const* input =
      src < send.inputBytes
          ? send.input + src / 4
          : reinterpret_cast<uint32_t const*>(
                reinterpret_cast<char const*>(send.channel.src_) + src);
  int stride = gridDim.x * blockDim.x;
  for (size_t i = tid; i < bytes / sizeof(uint4); i += stride)
    output[i] = loadClientQuad(input, i);
  publish();
  if (tid == 0) send.channel.signal();
}
template <class Channel>
__device__ __forceinline__ void addLocalOut(Channel, float* dst,
                                            float const* own, float const* peer,
                                            size_t n, int tid, int nth) {
  addOut(dst, own, peer, n, tid, nth);
}
template <class Channel>
__device__ __forceinline__ void addLocalInto(Channel, float* dst,
                                             float const* own, size_t n,
                                             int tid, int nth) {
  addInto(dst, own, n, tid, nth);
}
template <bool AlignedIO = false>
__device__ __forceinline__ void addLocalOut(ClientRingChan, float* dst,
                                            float const* own, float const* peer,
                                            size_t n, int tid, int nth) {
  for (size_t i = tid; i < n / 4; i += nth) {
    float4 x = loadClientFloatQuad<AlignedIO>(
        reinterpret_cast<uint32_t const*>(own), i);
    float4 y = reinterpret_cast<float4 const*>(peer)[i];
    x.x += y.x;
    x.y += y.y;
    x.z += y.z;
    x.w += y.w;
    reinterpret_cast<float4*>(dst)[i] = x;
  }
  for (size_t i = n / 4 * 4 + tid; i < n; i += nth) dst[i] = own[i] + peer[i];
}
__device__ __forceinline__ void addLocalInto(ClientRingChan, float* dst,
                                             float const* own, size_t n,
                                             int tid, int nth) {
  for (size_t i = tid; i < n / 4; i += nth) {
    float4 x = reinterpret_cast<float4*>(dst)[i];
    float4 y = loadClientFloatQuad(reinterpret_cast<uint32_t const*>(own), i);
    x.x += y.x;
    x.y += y.y;
    x.z += y.z;
    x.w += y.w;
    reinterpret_cast<float4*>(dst)[i] = x;
  }
  for (size_t i = n / 4 * 4 + tid; i < n; i += nth) dst[i] += own[i];
}

// Local ring order is 0 -> 1 -> 3 -> 2 -> 0.  It crosses the NUMA boundary once
// in each direction, so the UPI is never oversubscribed by two same-direction
// flows.  pos is the rank's position in this ring.
template <class LocalChan, bool ClientIO = false, bool AlignedIO = false>
__device__ void localReduceScatterPiece(
    LocalChan send, LocalChan recv, float* data, float* scratch, float* ibSend,
    size_t nelem, int pos, size_t pieceOff, size_t pieceLen, size_t chunkElems,
    size_t dataByteOffset, size_t scratchByteOffset, int tid, int nth,
    uint32_t const* clientInput = nullptr) {
  const size_t chunkBytes = chunkElems * sizeof(float);
  if constexpr (ClientIO) {
    size_t firstOffset = size_t(pos) * chunkElems + pieceOff;
    size_t firstLength =
        firstOffset >= nelem ? 0 : min(pieceLen, nelem - firstOffset);
    auto* firstTarget = reinterpret_cast<uint32_t*>(
        reinterpret_cast<char*>(send.channel.dst_) + scratchByteOffset +
        pieceOff * sizeof(float));
    for (size_t i = tid; i < firstLength / 4; i += nth)
      reinterpret_cast<uint4*>(firstTarget)[i] =
          loadClientQuad<AlignedIO>(clientInput + firstOffset, i);
    for (size_t i = firstLength / 4 * 4 + tid; i < firstLength; i += nth)
      firstTarget[i] = clientInput[firstOffset + i];
    publish();
    if (tid == 0) {
      send.channel.signal();
      recv.wait();
    }
    publish();
    for (int step = 0; step < kLocalRanks - 1; ++step) {
      size_t receiveOffset =
          size_t((pos - step - 1 + kLocalRanks) % kLocalRanks) * chunkElems +
          pieceOff;
      size_t length =
          receiveOffset >= nelem ? 0 : min(pieceLen, nelem - receiveOffset);
      float* destination =
          step == kLocalRanks - 2
              ? ibSend + pieceOff
              : reinterpret_cast<float*>(
                    reinterpret_cast<char*>(send.channel.dst_) +
                    scratchByteOffset + size_t(step + 1) * chunkBytes +
                    pieceOff * sizeof(float));
      // Forward the sum from registers instead of writing and re-reading a
      // local intermediate tensor. Only stable registered scratch is remote.
      if (length)
        addLocalOut<AlignedIO>(
            ClientRingChan{}, destination,
            reinterpret_cast<float const*>(clientInput) + receiveOffset,
            scratch + size_t(step) * chunkElems + pieceOff, length, tid, nth);
      publish();
      if (step != kLocalRanks - 2) {
        if (tid == 0) {
          send.channel.signal();
          recv.wait();
        }
        publish();
      }
    }
  } else {
    for (int step = 0; step < kLocalRanks - 1; step++) {
      int cSend = (pos - step + kLocalRanks) % kLocalRanks;
      int cRecv = (cSend - 1 + kLocalRanks) % kLocalRanks;
      size_t sendOff = (size_t)cSend * chunkElems + pieceOff;
      size_t recvOff = (size_t)cRecv * chunkElems + pieceOff;
      size_t sendLen = sendOff >= nelem ? 0 : min(pieceLen, nelem - sendOff);
      size_t recvLen = recvOff >= nelem ? 0 : min(pieceLen, nelem - recvOff);

      publish();
      putLocal(send,
               scratchByteOffset + (size_t)step * chunkBytes +
                   pieceOff * sizeof(float),
               dataByteOffset + sendOff * sizeof(float),
               sendLen * sizeof(float), tid);
      if (tid == 0) recv.wait();
      gMultiNodeSyncer.sync(gridDim.x);
      if (recvLen) {
        if (step == kLocalRanks - 2)
          addOut(ibSend + pieceOff, data + recvOff,
                 scratch + (size_t)step * chunkElems + pieceOff, recvLen, tid,
                 nth);
        else
          addInto(data + recvOff,
                  scratch + (size_t)step * chunkElems + pieceOff, recvLen, tid,
                  nth);
      }
      gMultiNodeSyncer.sync(gridDim.x);
    }
  }
}

// Three receive slots per half preserve each outbound partial until the local
// proxy has finished reading it. The original standard-layout input is never
// modified, so the next collective may use it without a reset or repack.
template <class LocalChan>
__device__ void localReduceScatterDirectHalf(LocalChan send, LocalChan recv,
                                             float const* input, float* scratch,
                                             float* partial, size_t shard,
                                             int half, int ringPos,
                                             size_t pieceOff, size_t pieceLen,
                                             Layout layout, int tid, int nth) {
  constexpr int ownerForChunk[kLocalRanks] = {2, 0, 1, 3};
  size_t bytes = pieceLen * sizeof(float);
  for (int step = 0; step < kLocalRanks - 1; ++step) {
    int cSend = (ringPos - step + kLocalRanks) % kLocalRanks;
    int cRecv = (cSend - 1 + kLocalRanks) % kLocalRanks;
    size_t receiveOffset =
        size_t(half * (kLocalRanks - 1) + step) * shard + pieceOff;
    size_t sourceOffset =
        step == 0
            ? size_t(ownerForChunk[cSend] + half * kLocalRanks) * shard +
                  pieceOff
            : size_t(half * (kLocalRanks - 1) + step - 1) * shard + pieceOff;
    size_t sourceBase = step == 0 ? layout.data : layout.localScratch;
    publish();
    putLocal(send, layout.localScratch + receiveOffset * sizeof(float),
             sourceBase + sourceOffset * sizeof(float), bytes, tid);
    if (tid == 0) recv.wait();
    gMultiNodeSyncer.sync(gridDim.x);
    float const* own =
        input + size_t(ownerForChunk[cRecv] + half * kLocalRanks) * shard +
        pieceOff;
    float* received = scratch + receiveOffset;
    if (step == kLocalRanks - 2)
      addLocalOut(send, partial + size_t(half) * shard + pieceOff, own,
                  received, pieceLen, tid, nth);
    else
      addLocalInto(send, received, own, pieceLen, tid, nth);
    gMultiNodeSyncer.sync(gridDim.x);
  }
}

template <class LocalChan>
__device__ void localAllGatherPiece(LocalChan send, LocalChan recv, float* data,
                                    size_t nelem, int pos, size_t pieceOff,
                                    size_t pieceLen, size_t chunkElems,
                                    size_t dataByteOffset, int tid) {
  for (int step = 0; step < kLocalRanks - 1; step++) {
    int cSend = (pos + 1 - step + kLocalRanks) % kLocalRanks;
    size_t sendOff = (size_t)cSend * chunkElems + pieceOff;
    size_t sendLen = sendOff >= nelem ? 0 : min(pieceLen, nelem - sendOff);

    // Only the first AllGather hop sends bytes produced by SM stores. Later
    // hops forward bytes that arrived through the same CUDA-IPC proxy and are
    // already ordered by wait(); the end-of-step sync aligns every block.
    if (step == 0) publish();
    putLocal(send, dataByteOffset + sendOff * sizeof(float),
             dataByteOffset + sendOff * sizeof(float), sendLen * sizeof(float),
             tid);
    if (tid == 0) recv.wait();
    gMultiNodeSyncer.sync(gridDim.x);
  }
}

// The standalone host-router harness uses zero-based registered staging.
template <class LocalChan>
__device__ void localAllGatherPiece(LocalChan send, LocalChan recv, float* data,
                                    size_t nelem, int pos, size_t pieceOff,
                                    size_t pieceLen, size_t chunkElems,
                                    int tid) {
  localAllGatherPiece(send, recv, data, nelem, pos, pieceOff, pieceLen,
                      chunkElems, size_t(0), tid);
}

template <class LocalChan, bool AlignedIO = false>
__device__ void localAllGatherClientPiece(LocalChan send, LocalChan recv,
                                          float* data, size_t nelem, int pos,
                                          size_t pieceOff, size_t pieceLen,
                                          size_t chunkElems, size_t dataOffset,
                                          int tid, int nth,
                                          uint32_t* clientOutput) {
  for (int step = 0; step < kLocalRanks - 1; ++step) {
    size_t offset =
        size_t((pos + 1 - step + kLocalRanks) % kLocalRanks) * chunkElems +
        pieceOff;
    size_t length = offset >= nelem ? 0 : min(pieceLen, nelem - offset);
    auto const* source = reinterpret_cast<uint4 const*>(data + offset);
    auto* target =
        reinterpret_cast<uint4*>(reinterpret_cast<char*>(send.channel.dst_) +
                                 dataOffset + offset * sizeof(float));
    for (size_t i = tid; i < length / 4; i += nth) {
      uint4 value = source[i];
      target[i] = value;
      storeClientQuad<AlignedIO>(clientOutput + offset, i, value);
    }
    for (size_t i = length / 4 * 4 + tid; i < length; i += nth) {
      uint32_t value = reinterpret_cast<uint32_t const*>(data + offset)[i];
      reinterpret_cast<uint32_t*>(target)[i] = value;
      clientOutput[offset + i] = value;
    }
    publish();
    if (tid == 0) {
      send.channel.signal();
      recv.wait();
    }
    publish();
  }
  // The last received chunk is not forwarded; copy only that remainder.
  size_t offset = size_t((pos + 2) % kLocalRanks) * chunkElems + pieceOff;
  size_t length = offset >= nelem ? 0 : min(pieceLen, nelem - offset);
  for (size_t i = tid; i < length / 4; i += nth)
    storeClientQuad<AlignedIO>(
        clientOutput + offset, i,
        reinterpret_cast<uint4 const*>(data + offset)[i]);
  for (size_t i = length / 4 * 4 + tid; i < length; i += nth)
    clientOutput[offset + i] =
        reinterpret_cast<uint32_t const*>(data + offset)[i];
}

__device__ void localFlatAllGatherPiece(LocalAgChans chans, float* data,
                                        size_t ownBase, size_t pieceOff,
                                        size_t pieceLen, int localRank,
                                        size_t dataByteOffset, int tid) {
  publish();
  if (tid == 0) {
    size_t byteOff = dataByteOffset + (ownBase + pieceOff) * sizeof(float);
    size_t bytes = pieceLen * sizeof(float);
    for (int k = 1; k < kLocalRanks; k++) {
      int peer = (localRank + k) % kLocalRanks;
      chans.c[peer].putWithSignal(byteOff, byteOff, bytes);
    }
    for (int k = 1; k < kLocalRanks; k++) {
      int peer = (localRank - k + kLocalRanks) % kLocalRanks;
      chans.c[peer].wait();
    }
  }
  gMultiNodeSyncer.sync(gridDim.x);
}

template <class LocalChan, bool ClientIO = false, bool AlignedIO = false>
__global__ void __launch_bounds__(1024)
    hierarchicalAllReduce(LocalChan localSend, LocalChan localRecv,
                          LocalAgChans localAg, PortChan ib, float* base,
                          size_t nelem, Layout layout, int localRank,
                          int ringPos, int pieces, int pipelineDistance,
                          bool flatAg, uint32_t const* clientInput = nullptr,
                          uint32_t* clientOutput = nullptr,
                          bool retireSends = true) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nth = gridDim.x * blockDim.x;
  float* data =
      reinterpret_cast<float*>(reinterpret_cast<char*>(base) + layout.data);
  float* localScratch = reinterpret_cast<float*>(reinterpret_cast<char*>(base) +
                                                 layout.localScratch);
  float* ibSend =
      reinterpret_cast<float*>(reinterpret_cast<char*>(base) + layout.ibSend);
  float* ibScratch = reinterpret_cast<float*>(reinterpret_cast<char*>(base) +
                                              layout.ibScratch);

  size_t chunkElems =
      ((nelem + kLocalRanks - 1) / kLocalRanks + 31) & ~(size_t)31;
  size_t ownChunk = (size_t)((ringPos + 1) % kLocalRanks);
  size_t ownBase = ownChunk * chunkElems;
  size_t ownLen = ownBase >= nelem ? 0 : min(chunkElems, nelem - ownBase);
  size_t pieceElems = ((chunkElems + pieces - 1) / pieces + 31) & ~(size_t)31;

  // A bounded software pipeline.  The IB write for piece p remains in flight
  // while local RS works on later pieces.  Once the chosen distance is reached,
  // reduce and locally publish the oldest completed piece.
  for (int p = 0; p < pieces + pipelineDistance; p++) {
    if (p < pieces) {
      size_t off = (size_t)p * pieceElems;
      size_t len = off >= chunkElems ? 0 : min(pieceElems, chunkElems - off);
      localReduceScatterPiece<LocalChan, ClientIO, AlignedIO>(
          localSend, localRecv, data, localScratch, ibSend, nelem, ringPos, off,
          len, chunkElems, layout.data, layout.localScratch, tid, nth,
          clientInput);
      // The final local RS step writes directly into the immutable IB slot, so
      // no extra staging copy is needed and inbound reduction cannot race the
      // outbound RDMA read.
      publish();
      if (tid == 0) {
        size_t owned = off >= ownLen ? 0 : min(len, ownLen - off);
        ib.putWithSignal(layout.ibScratch + off * sizeof(float),
                         layout.ibSend + off * sizeof(float),
                         owned * sizeof(float));
      }
    }

    int done = p - pipelineDistance;
    if (done >= 0 && done < pieces) {
      size_t off = (size_t)done * pieceElems;
      size_t len = off >= chunkElems ? 0 : min(pieceElems, chunkElems - off);
      if (tid == 0) ib.wait();
      gMultiNodeSyncer.sync(gridDim.x);
      size_t owned = off >= ownLen ? 0 : min(len, ownLen - off);
      if (owned)
        addOut(data + ownBase + off, ibSend + off, ibScratch + off, owned, tid,
               nth);
      gMultiNodeSyncer.sync(gridDim.x);
      if constexpr (ClientIO)
        localAllGatherClientPiece<LocalChan, AlignedIO>(
            localSend, localRecv, data, nelem, ringPos, off, len, chunkElems,
            layout.data, tid, nth, clientOutput);
      else if (flatAg)
        localFlatAllGatherPiece(localAg, data, ownBase, off, len, localRank,
                                layout.data, tid);
      else
        localAllGatherPiece(localSend, localRecv, data, nelem, ringPos, off,
                            len, chunkElems, layout.data, tid);
    }
  }
  if (tid == 0) {
    // CUDA-IPC copies do not consume an IB completion queue and need no flush.
    // Drain the IB QP once per collective to bound its CQ and protect reuse of
    // the source buffer by the next launch.
    if (retireSends) ib.flush();
  }
}

// Every node receives one remote shard per local rank over IB, then spreads
// those four shards over local CUDA-IPC channels. Each global-rank output
// interval has one writer, and the input slot stays immutable until flush.
__global__ void __launch_bounds__(1024)
    hierarchicalAllGather(LocalAgChans localAg, PortChan ib, float* base,
                          size_t nelem, Layout layout, int rank,
                          int localRank) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nth = gridDim.x * blockDim.x;
  size_t shard = nelem / kWorld;
  size_t ownOff = size_t(rank) * shard;
  size_t remoteOff = size_t(rank ^ kLocalRanks) * shard;
  float* output =
      reinterpret_cast<float*>(reinterpret_cast<char*>(base) + layout.data);
  float* input =
      reinterpret_cast<float*>(reinterpret_cast<char*>(base) + layout.ibSend);
  for (size_t i = tid; i < shard; i += nth) output[ownOff + i] = input[i];
  publish();
  if (tid == 0) {
    size_t bytes = shard * sizeof(float);
    ib.putWithSignal(layout.data + ownOff * sizeof(float), layout.ibSend,
                     bytes);
    for (int peer = 0; peer < kLocalRanks; ++peer)
      if (peer != localRank)
        localAg.c[peer].putWithSignal(layout.data + ownOff * sizeof(float),
                                      layout.ibSend, bytes);
    ib.wait();
  }
  gMultiNodeSyncer.sync(gridDim.x);
  publish();
  if (tid == 0) {
    size_t bytes = shard * sizeof(float);
    for (int peer = 0; peer < kLocalRanks; ++peer)
      if (peer != localRank)
        localAg.c[peer].putWithSignal(layout.data + remoteOff * sizeof(float),
                                      layout.data + remoteOff * sizeof(float),
                                      bytes);
    for (int peer = 0; peer < kLocalRanks; ++peer)
      if (peer != localRank) {
        localAg.c[peer].wait();
        localAg.c[peer].wait();
      }
    ib.flush();
    for (int peer = 0; peer < kLocalRanks; ++peer)
      if (peer != localRank) localAg.c[peer].flush();
  }
}

constexpr size_t kPacketSlotBytes = 64UL << 10;
constexpr size_t kLocalPacketBytes = 256UL << 10;
#ifdef UCCL_LITE_AG_READY_FLAGS
// Dedicated storage: user payload can never masquerade as a ready epoch.
constexpr size_t kPacketReadyBytes = 128;
constexpr size_t kPacketBankBytes =
    2 * kPacketSlotBytes + kLocalPacketBytes + kPacketReadyBytes;
__device__ __forceinline__ uint32_t* packetReadyFlag(char* base, Layout layout,
                                                     int sender) {
  return reinterpret_cast<uint32_t*>(base + layout.immutableInput +
                                     2 * kPacketSlotBytes + kLocalPacketBytes +
                                     size_t(sender) * 32);
}
__device__ __forceinline__ void packetLocalReady(LocalDirectChans local,
                                                 char* base, Layout layout,
                                                 int localRank, uint32_t flag) {
  if (threadIdx.x < kLocalRanks && int(threadIdx.x) != localRank) {
    int peer = threadIdx.x;
    auto* remote = packetReadyFlag(reinterpret_cast<char*>(local.c[peer].dst_),
                                   layout, localRank);
    auto* incoming = packetReadyFlag(base, layout, peer);
    // Exactly one sender owns each flag. The surrounding CTA barriers and
    // system release/acquire preserve the existing payload publication order.
    mscclpp::atomicStore<uint32_t, mscclpp::scopeSystem>(
        remote, flag, mscclpp::memoryOrderRelease);
    POLL_MAYBE_JAILBREAK((mscclpp::atomicLoad<uint32_t, mscclpp::scopeSystem>(
                              incoming, mscclpp::memoryOrderAcquire) != flag),
                         10000000);
  }
}
#else
constexpr size_t kPacketBankBytes = 2 * kPacketSlotBytes + kLocalPacketBytes;
#endif
#if defined(UCCL_LITE_EARLY_PACKET_FLUSH) || \
    defined(UCCL_LITE_AR_EARLY_FLUSH) || defined(UCCL_LITE_AG_DIRECT128)
__device__ __forceinline__ uint64_t beginPacketFlush(PortChan ib) {
  return ib.fifo_.push({mscclpp::TriggerSync, 0, 0, 0, 0, 0, ib.semaphoreId_});
}
__device__ __forceinline__ void endPacketFlush(PortChan ib, uint64_t ticket) {
  mscclpp::detail::waitFlush(ib.flushDonePos_, ticket, 10000000);
}
#endif
__device__ __forceinline__ uint64_t packetSend(PortChan ib, char* base,
                                               Layout layout,
                                               size_t sourceOffset,
                                               size_t bytes, uint32_t flag,
                                               bool requestFlush = false) {
  auto const* source = reinterpret_cast<uint2 const*>(base + sourceOffset);
  auto* packets =
      reinterpret_cast<mscclpp::LL16Packet*>(base + layout.immutableInput);
  for (size_t i = threadIdx.x; i < bytes / sizeof(uint2); i += blockDim.x)
    packets[i].write(source[i], flag);
#ifndef UCCL_LITE_PACKET_FIFO_PUBLISH
  __threadfence_system();
#endif
  // Every producer reaches this barrier before the leader's PortChannel put.
  // MSCCL++ FIFO push publishes with st.global.release.sys; its CPU consumer
  // acquires that commit before posting the RDMA write. The guarded variant
  // relies on this cumulative publication instead of an extra per-lane fence.
  __syncthreads();
  if (threadIdx.x == 0) {
#if defined(UCCL_LITE_AG_COALESCED_FLUSH) || defined(UCCL_LITE_AR_EARLY_FLUSH)
    if (requestFlush)
      return ib.fifo_.push({mscclpp::TriggerData | mscclpp::TriggerSync,
                            ib.dst_, layout.immutableInput + kPacketSlotBytes,
                            ib.src_, layout.immutableInput, bytes * 2,
                            ib.semaphoreId_});
#endif
    ib.put(layout.immutableInput + kPacketSlotBytes, layout.immutableInput,
           bytes * 2);
  }
  return 0;
}
__device__ __forceinline__ void packetReceive(char* base, Layout layout,
                                              size_t destinationOffset,
                                              size_t bytes, uint32_t flag) {
  auto const* packets = reinterpret_cast<mscclpp::LL16Packet const*>(
      base + layout.immutableInput + kPacketSlotBytes);
  auto* destination = reinterpret_cast<uint2*>(base + destinationOffset);
  for (size_t i = threadIdx.x; i < bytes / sizeof(uint2); i += blockDim.x)
    destination[i] = packets[i].read(flag);
  __syncthreads();
}

// Caller barriers publish the CTA's stores and acquire all peer completions.
// Independent lanes avoid serializing the three system-scope ready operations.
__device__ __forceinline__ void serialLocalReady(LocalDirectChans local,
                                                 int localRank) {
  if (threadIdx.x < kLocalRanks && int(threadIdx.x) != localRank) {
    local.c[threadIdx.x].signal();
    local.c[threadIdx.x].wait();
  }
}

template <bool ClientIO = false, bool Packet = false>
__device__ __forceinline__ void hierarchicalAllGatherSerialBody(
    LocalDirectChans local, PortChan ib, float* base, size_t nelem,
    Layout layout, int rank, int localRank,
    uint32_t const* clientInput = nullptr, uint32_t* clientOutput = nullptr,
    uint32_t packetFlag = 0, bool packetDrain = true
#ifdef UCCL_LITE_PACKET_TIMELINE
    ,
    uint64_t* serialTimeline = nullptr
#endif
) {
#ifdef UCCL_LITE_PACKET_TIMELINE
  using namespace flashinfer;
  struct {
    uint64_t* profiler_buffer;
  } traceParams{serialTimeline};
  struct {
    PROFILER_CLOSURE_PARAMS_DECL
  } serialTrace{};
  if (serialTimeline) {
    PROFILER_INIT(traceParams, unused, serialTrace, 0, 1, threadIdx.x == 0);
  }
#define UCCL_LITE_SERIAL_START(event) PROFILER_EVENT_START(serialTrace, event)
#define UCCL_LITE_SERIAL_END(event) PROFILER_EVENT_END(serialTrace, event)
#else
#define UCCL_LITE_SERIAL_START(event)
#define UCCL_LITE_SERIAL_END(event)
#endif
  UCCL_LITE_SERIAL_START(0);
  size_t shard = nelem / kWorld;
  size_t units = shard / 4;
  size_t ownOff = size_t(rank) * shard;
  size_t remoteOff = size_t(rank ^ kLocalRanks) * shard;
  char* bytes = reinterpret_cast<char*>(base);
  auto* output = reinterpret_cast<float*>(bytes + layout.data);
  auto const* input = reinterpret_cast<uint4 const*>(bytes + layout.ibSend);
  auto* self = reinterpret_cast<uint4*>(output + ownOff);
  for (size_t i = threadIdx.x; i < units; i += blockDim.x) {
    uint4 value;
    if constexpr (ClientIO) {
      value = loadClientQuad(clientInput, i);
      reinterpret_cast<uint4*>(bytes + layout.ibSend)[i] = value;
    } else
      value = input[i];
    self[i] = value;
  }
  UCCL_LITE_SERIAL_END(0);
  __syncthreads();
  UCCL_LITE_SERIAL_START(1);
  uint64_t flushTicket = 0;
  bool directRetirement = false;
#ifdef UCCL_LITE_AG_DIRECT128
  if constexpr (ClientIO && !Packet) directRetirement = nelem == 32768;
#endif
  if constexpr (Packet) {
#if defined(UCCL_LITE_EARLY_PACKET_FLUSH) && \
    defined(UCCL_LITE_AG_COALESCED_FLUSH)
    flushTicket =
        packetSend(ib, bytes, layout, layout.ibSend, shard * sizeof(float),
                   packetFlag, ClientIO && packetDrain);
#else
    packetSend(ib, bytes, layout, layout.ibSend, shard * sizeof(float),
               packetFlag);
#endif
  } else if (threadIdx.x == 0) {
#ifdef UCCL_LITE_AG_DIRECT128
    if (directRetirement)
      // Raw data plus the existing MSCCL++ receive signal and send-retirement
      // request. This is the only IB write; local fan-out cannot alter ibSend.
      flushTicket = ib.fifo_.push(
          {mscclpp::TriggerData | mscclpp::TriggerFlag | mscclpp::TriggerSync,
           ib.dst_, layout.data + ownOff * sizeof(float), ib.src_,
           layout.ibSend, shard * sizeof(float), ib.semaphoreId_});
    else
#endif
      ib.putWithSignal(layout.data + ownOff * sizeof(float), layout.ibSend,
                       shard * sizeof(float));
  }
#ifdef UCCL_LITE_EARLY_PACKET_FLUSH
#ifndef UCCL_LITE_AG_COALESCED_FLUSH
  if constexpr (Packet && ClientIO)
    if (!threadIdx.x && packetDrain) flushTicket = beginPacketFlush(ib);
#endif
#endif
  UCCL_LITE_SERIAL_END(1);
  UCCL_LITE_SERIAL_START(2);
#ifdef UCCL_LITE_AG128_PAIR_EXCHANGE
  if constexpr (ClientIO && Packet) {
    if (nelem * sizeof(float) == (128UL << 10)) {
      packetReceive(bytes, layout, layout.data + remoteOff * sizeof(float),
                    shard * sizeof(float), packetFlag);
      // Each rank owns {rank, rank^4}. Exchange one pair across NUMA, then
      // two pairs within NUMA. Sources/destinations are disjoint per phase.
      // Finishing the second phase requires every local and remote rank to
      // enter this call; the existing two-bank reader-reuse proof still holds.
      for (int phase = 0; phase < 2; ++phase) {
        int peer = localRank ^ (phase ? 1 : 2);
        for (int bundle = 0; bundle <= phase; ++bundle)
          for (int half = 0; half < 2; ++half) {
            int owner = localRank ^ (bundle ? 2 : 0);
            int global = ((rank - localRank) ^ (half * kLocalRanks)) + owner;
            size_t offset = size_t(global) * units;
            auto const* source =
                reinterpret_cast<uint4 const*>(output) + offset;
            auto* target =
                reinterpret_cast<uint4*>(
                    reinterpret_cast<char*>(local.c[peer].dst_) + layout.data) +
                offset;
            for (size_t i = threadIdx.x; i < units; i += blockDim.x)
              target[i] = source[i];
          }
        __syncthreads();
        if (threadIdx.x == 0) {
          local.c[peer].signal();
          local.c[peer].wait();
        }
        __syncthreads();
      }
      for (size_t i = threadIdx.x; i < nelem / 4; i += blockDim.x)
        storeClientQuad(clientOutput, i,
                        reinterpret_cast<uint4 const*>(output)[i]);
      if (threadIdx.x == 0 && packetDrain) {
#ifdef UCCL_LITE_EARLY_PACKET_FLUSH
        endPacketFlush(ib, flushTicket);
#else
        ib.flush();
#endif
      }
      UCCL_LITE_SERIAL_END(2);
      return;
    }
  }
#endif
#ifdef UCCL_LITE_AG_PARALLEL_FANOUT
  if constexpr (ClientIO && Packet) {
    // Each (peer, quad) has one writer. Publish only after all peers' stores.
    for (size_t j = threadIdx.x; j < units * (kLocalRanks - 1);
         j += blockDim.x) {
      int peer = (localRank + 1 + j / units) % kLocalRanks;
      auto* target =
          reinterpret_cast<uint4*>(reinterpret_cast<char*>(local.c[peer].dst_) +
                                   layout.data + ownOff * sizeof(float));
      target[j % units] = input[j % units];
    }
    __syncthreads();
  } else
#endif
    for (int step = 1; step < kLocalRanks; ++step) {
      int peer = (localRank + step) % kLocalRanks;
      auto* target =
          reinterpret_cast<uint4*>(reinterpret_cast<char*>(local.c[peer].dst_) +
                                   layout.data + ownOff * sizeof(float));
      for (size_t i = threadIdx.x; i < units; i += blockDim.x) {
        uint4 value = input[i];
        target[i] = value;
#ifdef UCCL_LITE_AG_FUSED_OWN
#if defined(UCCL_LITE_AG_PARALLEL_FANOUT) || defined(UCCL_LITE_AG_EARLY_OWN)
#error "Fused own-output is a separate serial fan-out experiment"
#endif
        if constexpr (ClientIO && Packet)
          if (nelem == 32768 && step == kLocalRanks - 1)
            storeClientQuad(clientOutput, ownOff / 4 + i, value);
#endif
      }
#ifdef UCCL_LITE_AG_COMPACT_BARRIERS
      if constexpr (ClientIO && Packet) {
        // Peer destinations are disjoint and the source stays immutable.
        // Keep one barrier after all writes, before any ready publication.
        if (step == kLocalRanks - 1) __syncthreads();
      } else
#endif
        __syncthreads();
    }
  UCCL_LITE_SERIAL_END(2);
  UCCL_LITE_SERIAL_START(3);
  if (threadIdx.x == 0) {
    // Client IO consumes data only after both local and remote fan-outs.
    // One final ready signal is sufficient; retain the standalone protocol.
    if constexpr (!ClientIO)
      for (int peer = 0; peer < kLocalRanks; ++peer)
        if (peer != localRank) local.c[peer].signal();
    if constexpr (!Packet) ib.wait();
  }
  if constexpr (Packet)
    packetReceive(bytes, layout, layout.data + remoteOff * sizeof(float),
                  shard * sizeof(float), packetFlag);
#ifdef UCCL_LITE_AG_COMPACT_BARRIERS
  // packetReceive already ends with the same CTA barrier.
  if constexpr (!(ClientIO && Packet))
#endif
    __syncthreads();
  UCCL_LITE_SERIAL_END(3);
  UCCL_LITE_SERIAL_START(4);
  auto const* received = reinterpret_cast<uint4 const*>(output + remoteOff);
#ifdef UCCL_LITE_AG_PARALLEL_FANOUT
  if constexpr (ClientIO && Packet) {
    for (size_t j = threadIdx.x; j < units * (kLocalRanks - 1);
         j += blockDim.x) {
      int peer = (localRank + 1 + j / units) % kLocalRanks;
      auto* target =
          reinterpret_cast<uint4*>(reinterpret_cast<char*>(local.c[peer].dst_) +
                                   layout.data + remoteOff * sizeof(float));
      target[j % units] = received[j % units];
    }
    __syncthreads();
  } else
#endif
    for (int step = 1; step < kLocalRanks; ++step) {
      int peer = (localRank + step) % kLocalRanks;
      auto* target =
          reinterpret_cast<uint4*>(reinterpret_cast<char*>(local.c[peer].dst_) +
                                   layout.data + remoteOff * sizeof(float));
      for (size_t i = threadIdx.x; i < units; i += blockDim.x) {
        uint4 value = received[i];
        target[i] = value;
#ifdef UCCL_LITE_AG_FUSED_OWN
        if constexpr (ClientIO && Packet)
          if (nelem == 32768 && step == kLocalRanks - 1)
            storeClientQuad(clientOutput, remoteOff / 4 + i, value);
#endif
      }
#ifdef UCCL_LITE_AG_EARLY_OWN
#if defined(UCCL_LITE_AG_PARALLEL_FANOUT)
#error "Early own-output experiment requires serial peer fan-out"
#endif
      if constexpr (ClientIO && Packet) {
        if (nelem == 32768 && step == kLocalRanks - 1) {
          // Both owned shards are ready. Copy them before the existing last
          // fan-out barrier so their stores overlap peer readiness waiting.
          auto const* ownValues =
              reinterpret_cast<uint4 const*>(output + ownOff);
          for (size_t i = threadIdx.x; i < units; i += blockDim.x) {
            storeClientQuad(clientOutput, ownOff / 4 + i, ownValues[i]);
            storeClientQuad(clientOutput, remoteOff / 4 + i, received[i]);
          }
        }
      }
#endif
#ifdef UCCL_LITE_AG_COMPACT_BARRIERS
      if constexpr (ClientIO && Packet) {
        if (step == kLocalRanks - 1) __syncthreads();
      } else
#endif
        __syncthreads();
    }
  UCCL_LITE_SERIAL_END(4);
  UCCL_LITE_SERIAL_START(5);
  if constexpr (ClientIO) {
#ifdef UCCL_LITE_AG_READY_FLAGS
    if constexpr (Packet)
      packetLocalReady(local, bytes, layout, localRank, packetFlag);
    else
#endif
      serialLocalReady(local, localRank);
    if (threadIdx.x == 0 && (!Packet || packetDrain)) {
#ifdef UCCL_LITE_EARLY_PACKET_FLUSH
      if constexpr (Packet) {
#ifndef UCCL_LITE_AG_OVERLAP_OUTPUT
        endPacketFlush(ib, flushTicket);
#endif
      } else if (!directRetirement)
        ib.flush();
#else
      if (!directRetirement) ib.flush();
#endif
    }
  } else if (threadIdx.x == 0) {
    for (int peer = 0; peer < kLocalRanks; ++peer)
      if (peer != localRank) local.c[peer].signal();
    for (int peer = 0; peer < kLocalRanks; ++peer)
      if (peer != localRank) {
        local.c[peer].wait();
        local.c[peer].wait();
      }
    if (!Packet || packetDrain) ib.flush();
  }
  if constexpr (ClientIO) {
    __syncthreads();
    UCCL_LITE_SERIAL_END(5);
    UCCL_LITE_SERIAL_START(6);
    for (size_t i = threadIdx.x; i < nelem / 4; i += blockDim.x) {
#if defined(UCCL_LITE_AG_EARLY_OWN) || defined(UCCL_LITE_AG_FUSED_OWN)
      if constexpr (Packet)
        if (nelem == 32768 && int((i >> 10) & 3) == localRank) continue;
#endif
      storeClientQuad(clientOutput, i,
                      reinterpret_cast<uint4 const*>(output)[i]);
    }
    UCCL_LITE_SERIAL_END(6);
    UCCL_LITE_SERIAL_START(7);
#if defined(UCCL_LITE_EARLY_PACKET_FLUSH) && \
    defined(UCCL_LITE_AG_OVERLAP_OUTPUT)
    // Output copying does not modify the IB send packets. The kernel still
    // waits for the same final send ticket before reporting completion.
    if constexpr (Packet)
      if (threadIdx.x == 0 && packetDrain) endPacketFlush(ib, flushTicket);
#endif
#ifdef UCCL_LITE_AG_DIRECT128
    if (directRetirement && threadIdx.x == 0) endPacketFlush(ib, flushTicket);
#endif
    UCCL_LITE_SERIAL_END(7);
  }
#undef UCCL_LITE_SERIAL_START
#undef UCCL_LITE_SERIAL_END
}

template <bool ClientIO = false>
__global__ void hierarchicalAllGatherSerial(
    LocalDirectChans local, PortChan ib, float* base, size_t nelem,
    Layout layout, int rank, int localRank,
    uint32_t const* clientInput = nullptr, uint32_t* clientOutput = nullptr) {
  hierarchicalAllGatherSerialBody<ClientIO>(local, ib, base, nelem, layout,
                                            rank, localRank, clientInput,
                                            clientOutput);
}

// Chunked all-gather: IB has an independent proxy and a bounded lookahead.
// Receive into alternating immutable staging slots, so the next invocation
// cannot overwrite a chunk while its local fan-out still reads it. The output
// is never a forwarding source. All outgoing local DMA is drained before
// return.
__device__ __forceinline__ void agPipelineSync() {
  if (gridDim.x == 1)
    __syncthreads();
  else
    publish();
}

template <bool CopyEngine, bool PairRelay = false, bool DualRdma = false>
__global__ void hierarchicalAllGatherPipeline(
    LocalDirectChans local, LocalAgChans dma, PortChan ib, PortChan duplicateIb,
    float* base, size_t nelem, Layout layout, int rank, int localRank,
    size_t receiveOffset, size_t pieceBytes, int depth, size_t relayOffset,
    size_t relayHalfBytes) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nth = gridDim.x * blockDim.x;
  char* bytes = reinterpret_cast<char*>(base);
  size_t shardBytes = nelem / kWorld * sizeof(float);
  size_t ownOffset = layout.data + size_t(rank) * shardBytes;
  size_t remoteOffset = layout.data + size_t(rank ^ kLocalRanks) * shardBytes;
  int pieces = int((shardBytes + pieceBytes - 1) / pieceBytes);
  depth = min(depth, pieces);
  if (tid == 0)
    for (int p = 0; p < depth; ++p) {
      size_t off = size_t(p) * pieceBytes;
      ib.putWithSignal(receiveOffset + off, layout.ibSend + off,
                       min(pieceBytes, shardBytes - off));
      if constexpr (DualRdma)
        duplicateIb.putWithSignal(relayOffset + off, layout.ibSend + off,
                                  min(pieceBytes, shardBytes - off));
    }
  for (int p = 0; p < pieces; ++p) {
    size_t off = size_t(p) * pieceBytes;
    size_t count = min(pieceBytes, shardBytes - off);
    auto const* input =
        reinterpret_cast<uint4 const*>(bytes + layout.ibSend + off);
    auto* own = reinterpret_cast<uint4*>(bytes + ownOffset + off);
    for (size_t i = tid; i < count / sizeof(uint4); i += nth) own[i] = input[i];
    agPipelineSync();
    for (int step = 1; step < kLocalRanks; ++step) {
      int peer = (localRank + step) % kLocalRanks;
      if constexpr (CopyEngine) {
        if (tid == 0) {
          if constexpr (PairRelay) {
            if (peer == (localRank ^ 1))
              dma.c[peer].put(ownOffset + off, layout.ibSend + off, count);
            if (peer == (localRank ^ 2))
              dma.c[peer].put(relayOffset + off, layout.ibSend + off, count);
          } else {
            dma.c[peer].put(ownOffset + off, layout.ibSend + off, count);
          }
        }
      } else {
        auto* target = reinterpret_cast<uint4*>(
            reinterpret_cast<char*>(local.c[peer].dst_) + ownOffset + off);
        for (size_t i = tid; i < count / sizeof(uint4); i += nth)
          target[i] = input[i];
        agPipelineSync();
      }
    }
    if (tid == 0) {
      ib.wait();
      // Refill lookahead as soon as this receive completes, before fan-out.
      int next = p + depth;
      if (next < pieces) {
        size_t nextOff = size_t(next) * pieceBytes;
        ib.putWithSignal(receiveOffset + nextOff, layout.ibSend + nextOff,
                         min(pieceBytes, shardBytes - nextOff));
        if constexpr (DualRdma)
          duplicateIb.putWithSignal(relayOffset + nextOff,
                                    layout.ibSend + nextOff,
                                    min(pieceBytes, shardBytes - nextOff));
      }
    }
    agPipelineSync();
    auto const* received =
        reinterpret_cast<uint4 const*>(bytes + receiveOffset + off);
    auto* self = reinterpret_cast<uint4*>(bytes + remoteOffset + off);
    for (size_t i = tid; i < count / sizeof(uint4); i += nth)
      self[i] = received[i];
    agPipelineSync();
    for (int step = 1; step < kLocalRanks; ++step) {
      int peer = (localRank + step) % kLocalRanks;
      if constexpr (DualRdma) {
        if (peer == (localRank ^ 1)) continue;
      }
      if constexpr (CopyEngine) {
        if (tid == 0) {
          if constexpr (PairRelay) {
            if (peer == (localRank ^ 1))
              dma.c[peer].put(remoteOffset + off, receiveOffset + off, count);
            if (peer == (localRank ^ 2))
              dma.c[peer].put(relayOffset + relayHalfBytes + off,
                              receiveOffset + off, count);
          } else {
            dma.c[peer].put(remoteOffset + off, receiveOffset + off, count);
          }
        }
      } else {
        auto* target = reinterpret_cast<uint4*>(
            reinterpret_cast<char*>(local.c[peer].dst_) + remoteOffset + off);
        for (size_t i = tid; i < count / sizeof(uint4); i += nth)
          target[i] = received[i];
        agPipelineSync();
      }
    }
    if constexpr (DualRdma) {
      if (tid == 0) duplicateIb.wait();
      agPipelineSync();
      auto const* duplicate =
          reinterpret_cast<uint4 const*>(bytes + relayOffset + off);
      auto* duplicateOutput = reinterpret_cast<uint4*>(
          bytes + layout.data + size_t(rank ^ kLocalRanks ^ 1) * shardBytes +
          off);
      for (size_t i = tid; i < count / sizeof(uint4); i += nth)
        duplicateOutput[i] = duplicate[i];
      agPipelineSync();
    }
    if constexpr (PairRelay) {
      // The cross-NUMA peer sends two owned shards into immutable staging.
      // Relay them only within this NUMA pair. Each source crosses UPI once.
      if (tid == 0) {
        dma.c[localRank ^ 2].signal();
        dma.c[localRank ^ 2].wait();
      }
      agPipelineSync();
      for (int half = 0; half < 2; ++half) {
        size_t src = relayOffset + size_t(half) * relayHalfBytes + off;
        size_t dst = layout.data +
                     size_t((rank ^ 2) ^ (half * kLocalRanks)) * shardBytes +
                     off;
        auto const* relay = reinterpret_cast<uint4 const*>(bytes + src);
        auto* selfRelay = reinterpret_cast<uint4*>(bytes + dst);
        for (size_t i = tid; i < count / sizeof(uint4); i += nth)
          selfRelay[i] = relay[i];
        agPipelineSync();
        if (tid == 0) dma.c[localRank ^ 1].put(dst, src, count);
      }
    }
  }
  if (tid == 0) {
    // No local consumer reads output until collective completion. One final
    // notification per peer covers all chunks, avoiding per-copy doorbells.
    for (int peer = 0; peer < kLocalRanks; ++peer) {
      if (peer == localRank) continue;
      if constexpr (PairRelay) {
        if (peer != (localRank ^ 1)) continue;
      }
      if constexpr (CopyEngine)
        dma.c[peer].signal();
      else
        local.c[peer].signal();
    }
    for (int peer = 0; peer < kLocalRanks; ++peer) {
      if (peer == localRank) continue;
      if constexpr (PairRelay) {
        if (peer != (localRank ^ 1)) continue;
      }
      if constexpr (CopyEngine) {
        dma.c[peer].wait();
        dma.c[peer].flush();
      } else {
        local.c[peer].wait();
      }
    }
    if constexpr (PairRelay) dma.c[localRank ^ 2].flush();
    if constexpr (DualRdma) duplicateIb.flush();
    ib.flush();
  }
}

// Recursive doubling over logical pairs {local shard, remote shard}, stored
// directly in standard output layout to avoid full-output packing/unpacking.
// By default XOR-2 exchanges one logical pair across NUMA, then XOR-1 two pairs
// within NUMA. Ready/data handshakes and send flushes keep one P2P payload
// ingress and egress per GPU, including when ranks advance at different speeds.
template <bool CopyEngine>
__global__ void hierarchicalAllGatherMatched(
    LocalDirectChans direct, LocalAgChans local, PortChan ib, float* base,
    size_t nelem, Layout layout, int rank, int localRank, size_t receiveOffset,
    size_t pieceBytes, int depth, int firstMask) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nth = gridDim.x * blockDim.x;
  char* bytes = reinterpret_cast<char*>(base);
  size_t shardBytes = nelem / kWorld * sizeof(float);
  int pieces = int((shardBytes + pieceBytes - 1) / pieceBytes);
  depth = min(depth, pieces);
  if (tid == 0)
    for (int p = 0; p < depth; ++p) {
      size_t off = size_t(p) * pieceBytes;
      ib.putWithSignal(receiveOffset + off, layout.ibSend + off,
                       min(pieceBytes, shardBytes - off));
    }
  for (int p = 0; p < pieces; ++p) {
    size_t off = size_t(p) * pieceBytes;
    size_t count = min(pieceBytes, shardBytes - off);
    if (tid == 0) {
      ib.wait();
      if (p + depth < pieces) {
        size_t next = size_t(p + depth) * pieceBytes;
        ib.putWithSignal(receiveOffset + next, layout.ibSend + next,
                         min(pieceBytes, shardBytes - next));
      }
    }
    agPipelineSync();
    for (int half = 0; half < 2; ++half) {
      auto const* source = reinterpret_cast<uint4 const*>(
          bytes + (half ? receiveOffset : layout.ibSend) + off);
      auto* dest = reinterpret_cast<uint4*>(
          bytes + layout.data +
          size_t(rank ^ (half * kLocalRanks)) * shardBytes + off);
      for (size_t i = tid; i < count / sizeof(uint4); i += nth)
        dest[i] = source[i];
    }
    agPipelineSync();
    for (int phase = 0; phase < 2; ++phase) {
      int peer = localRank ^ (phase == 0 ? firstMask : (3 ^ firstMask));
      if (tid == 0) {
        // Ready is GPU-driven even for CE payloads. Both endpoints must have
        // completed prior reads and sends before either writes this phase.
        direct.c[peer].signal();
        direct.c[peer].wait();
      }
      agPipelineSync();
      for (int bundle = 0; bundle <= phase; ++bundle) {
        int owner = bundle == 0 ? localRank : localRank ^ firstMask;
        for (int half = 0; half < 2; ++half) {
          int global = ((rank - localRank) ^ (half * kLocalRanks)) + owner;
          size_t src = layout.data + size_t(global) * shardBytes + off;
          // Source and incoming destination shard sets are disjoint within
          // each phase. Ready/data handshakes guard forwarding and reuse.
          if constexpr (CopyEngine) {
            if (tid == 0) local.c[peer].put(src, src, count);
          } else {
            auto const* from = reinterpret_cast<uint4 const*>(bytes + src);
            auto* to = reinterpret_cast<uint4*>(
                reinterpret_cast<char*>(direct.c[peer].dst_) + src);
            for (size_t i = tid; i < count / sizeof(uint4); i += nth)
              to[i] = from[i];
          }
          agPipelineSync();
        }
      }
      if (tid == 0) {
        if constexpr (CopyEngine) {
          local.c[peer].signal();
          local.c[peer].wait();
          local.c[peer].flush();
        } else {
          direct.c[peer].signal();
          direct.c[peer].wait();
        }
      }
      agPipelineSync();
    }
  }
  if (tid == 0) ib.flush();
}

template <class LocalChan>
__device__ void gatherFourRing(LocalChan send, LocalChan recv,
                               size_t baseOffset, size_t shard, int nodeBase,
                               int ringPos, int tid) {
  constexpr int order[kLocalRanks] = {0, 1, 3, 2};
  for (int step = 0; step < kLocalRanks - 1; ++step) {
    int owner = order[(ringPos - step + kLocalRanks) % kLocalRanks];
    size_t off = size_t(nodeBase + owner) * shard * sizeof(float);
    publish();
    putLocal(send, baseOffset + off, baseOffset + off, shard * sizeof(float),
             tid);
    if (tid == 0) recv.wait();
    gMultiNodeSyncer.sync(gridDim.x);
  }
}

// Ring forwarding avoids the flat multicast's simultaneous fan-out. The
// first local ring runs while the partner rank's IB transfer is in flight.
template <class LocalChan>
__global__ void __launch_bounds__(1024)
    hierarchicalAllGatherRing(LocalChan localSend, LocalChan localRecv,
                              PortChan ib, float* base, size_t nelem,
                              Layout layout, int rank, int localRank,
                              int ringPos) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nth = gridDim.x * blockDim.x;
  size_t shard = nelem / kWorld;
  int nodeBase = rank - localRank;
  int remoteBase = nodeBase ^ kLocalRanks;
  float* output =
      reinterpret_cast<float*>(reinterpret_cast<char*>(base) + layout.data);
  float* input =
      reinterpret_cast<float*>(reinterpret_cast<char*>(base) + layout.ibSend);
  auto* own = output + size_t(rank) * shard;
  if (((reinterpret_cast<uintptr_t>(own) | reinterpret_cast<uintptr_t>(input)) &
       15) == 0) {
    for (size_t i = tid; i < shard / 4; i += nth)
      reinterpret_cast<uint4*>(own)[i] =
          reinterpret_cast<uint4 const*>(input)[i];
    for (size_t i = shard / 4 * 4 + tid; i < shard; i += nth) own[i] = input[i];
  } else {
    for (size_t i = tid; i < shard; i += nth) own[i] = input[i];
  }
  publish();
  if (tid == 0)
    ib.putWithSignal(layout.data + size_t(rank) * shard * sizeof(float),
                     layout.ibSend, shard * sizeof(float));
  gatherFourRing(localSend, localRecv, layout.data, shard, nodeBase, ringPos,
                 tid);
  if (tid == 0) ib.wait();
  gMultiNodeSyncer.sync(gridDim.x);
  gatherFourRing(localSend, localRecv, layout.data, shard, remoteBase, ringPos,
                 tid);
  if (tid == 0) {
    ib.flush();
    localSend.flush();
  }
}

// First reduce both nodes' destination shards among the four local sources.
// The two half-shards owned by local rank r represent global destinations r
// and r+4. Exchange the remote-owned half with the partner rank on the other
// node, then add its partial to the local-owned half in place.
__global__ void __launch_bounds__(1024)
    hierarchicalReduceScatter(LocalAgChans localAg, PortChan ib, float* base,
                              size_t nelem, Layout layout, int rank,
                              int localRank) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nth = gridDim.x * blockDim.x;
  size_t shard = nelem / kWorld;
  size_t bytes = shard * sizeof(float);
  float* input =
      reinterpret_cast<float*>(reinterpret_cast<char*>(base) + layout.data);
  float* scratch = reinterpret_cast<float*>(reinterpret_cast<char*>(base) +
                                            layout.localScratch);
  float* partial =
      reinterpret_cast<float*>(reinterpret_cast<char*>(base) + layout.ibSend);
  float* remote = reinterpret_cast<float*>(reinterpret_cast<char*>(base) +
                                           layout.ibScratch);
  publish();
  if (tid == 0) {
    for (int dest = 0; dest < kLocalRanks; ++dest)
      if (dest != localRank)
        for (int half = 0; half < kNodes; ++half)
          localAg.c[dest].putWithSignal(
              layout.localScratch + size_t(localRank * kNodes + half) * bytes,
              layout.data + size_t(dest + half * kLocalRanks) * bytes, bytes);
    for (int peer = 0; peer < kLocalRanks; ++peer)
      if (peer != localRank) {
        localAg.c[peer].wait();
        localAg.c[peer].wait();
      }
  }
  gMultiNodeSyncer.sync(gridDim.x);
  for (int half = 0; half < kNodes; ++half)
    for (size_t i = tid; i < shard; i += nth) {
      float sum = 0;
      for (int source = 0; source < kLocalRanks; ++source) {
        float value =
            source == localRank
                ? input[size_t(localRank + half * kLocalRanks) * shard + i]
                : scratch[size_t(source * kNodes + half) * shard + i];
        sum += value;
      }
      partial[size_t(half) * shard + i] = sum;
    }
  publish();
  int ownHalf = rank < kLocalRanks ? 0 : 1;
  if (tid == 0) {
    int sendHalf = ownHalf ^ 1;
    ib.putWithSignal(layout.ibScratch, layout.ibSend + size_t(sendHalf) * bytes,
                     bytes);
    ib.wait();
  }
  gMultiNodeSyncer.sync(gridDim.x);
  for (size_t i = tid; i < shard; i += nth)
    partial[size_t(ownHalf) * shard + i] += remote[i];
  gMultiNodeSyncer.sync(gridDim.x);
  if (tid == 0) {
    ib.flush();
    for (int peer = 0; peer < kLocalRanks; ++peer)
      if (peer != localRank) localAg.c[peer].flush();
  }
}

template <bool ClientIO = false, bool Packet = false>
__device__ __forceinline__ void hierarchicalReduceScatterSerialBody(
    LocalDirectChans local, PortChan ib, float* base, size_t nelem,
    Layout layout, int rank, int localRank,
    uint32_t const* clientInput = nullptr, uint32_t* clientOutput = nullptr,
    uint32_t packetFlag = 0, bool packetDrain = true) {
  size_t shard = nelem / kWorld;
  size_t units = shard / 4;
  char* bytes = reinterpret_cast<char*>(base);
  auto const* input = reinterpret_cast<float const*>(bytes + layout.data);
  auto* scratch = reinterpret_cast<float*>(bytes + layout.localScratch);
  auto* partial = reinterpret_cast<float*>(bytes + layout.ibSend);
  auto const* remote = reinterpret_cast<float const*>(bytes + layout.ibScratch);
  for (int step = 1; step < kLocalRanks; ++step) {
    int dest = (localRank + step) % kLocalRanks;
    for (int half = 0; half < kNodes; ++half) {
      auto const* from = reinterpret_cast<uint4 const*>(
          input + size_t(dest + half * kLocalRanks) * shard);
      auto* target = reinterpret_cast<uint4*>(
          reinterpret_cast<char*>(local.c[dest].dst_) + layout.localScratch +
          size_t(localRank * kNodes + half) * shard * sizeof(float));
      for (size_t i = threadIdx.x; i < units; i += blockDim.x) {
        if constexpr (ClientIO)
          target[i] = loadClientQuad(
              clientInput, size_t(dest + half * kLocalRanks) * units + i);
        else
          target[i] = from[i];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    for (int peer = 0; peer < kLocalRanks; ++peer)
      if (peer != localRank) local.c[peer].signal();
    for (int peer = 0; peer < kLocalRanks; ++peer)
      if (peer != localRank) local.c[peer].wait();
  }
  __syncthreads();
  for (int half = 0; half < kNodes; ++half) {
    auto const* own = reinterpret_cast<float4 const*>(
        input + size_t(localRank + half * kLocalRanks) * shard);
    auto const* received = reinterpret_cast<float4 const*>(scratch);
    auto* out = reinterpret_cast<float4*>(partial + size_t(half) * shard);
    for (size_t i = threadIdx.x; i < units; i += blockDim.x) {
      float4 value;
      if constexpr (ClientIO)
        value = loadClientFloatQuad(
            clientInput, size_t(localRank + half * kLocalRanks) * units + i);
      else
        value = own[i];
      for (int source = 0; source < kLocalRanks; ++source) {
        if (source == localRank) continue;
        float4 peer = received[size_t(source * kNodes + half) * units + i];
        value.x += peer.x;
        value.y += peer.y;
        value.z += peer.z;
        value.w += peer.w;
      }
      out[i] = value;
    }
  }
  __syncthreads();
  int ownHalf = rank < kLocalRanks ? 0 : 1;
  if constexpr (Packet) {
    packetSend(ib, bytes, layout,
               layout.ibSend + size_t(ownHalf ^ 1) * shard * sizeof(float),
               shard * sizeof(float), packetFlag);
    packetReceive(bytes, layout, layout.ibScratch, shard * sizeof(float),
                  packetFlag);
  } else if (threadIdx.x == 0) {
    int sendHalf = ownHalf ^ 1;
    ib.putWithSignal(layout.ibScratch,
                     layout.ibSend + size_t(sendHalf) * shard * sizeof(float),
                     shard * sizeof(float));
    ib.wait();
  }
  __syncthreads();
  auto* result = reinterpret_cast<float4*>(partial + size_t(ownHalf) * shard);
  auto const* peerResult = reinterpret_cast<float4 const*>(remote);
  for (size_t i = threadIdx.x; i < units; i += blockDim.x) {
    float4 value = result[i], peer = peerResult[i];
    value.x += peer.x;
    value.y += peer.y;
    value.z += peer.z;
    value.w += peer.w;
    result[i] = value;
    if constexpr (ClientIO)
      storeClientQuad(clientOutput, i, floatQuadBits(value));
  }
  __syncthreads();
  if (threadIdx.x == 0)
    if (!Packet || packetDrain) ib.flush();
}

template <bool ClientIO = false>
__global__ void hierarchicalReduceScatterSerial(
    LocalDirectChans local, PortChan ib, float* base, size_t nelem,
    Layout layout, int rank, int localRank,
    uint32_t const* clientInput = nullptr, uint32_t* clientOutput = nullptr) {
  hierarchicalReduceScatterSerialBody<ClientIO>(local, ib, base, nelem, layout,
                                                rank, localRank, clientInput,
                                                clientOutput);
}

// Experimental remote-first ReduceScatter with push and pull local stages.
// Inputs stay immutable until the exit handshake. Each destination owns two
// disjoint partials; only the remote half is exposed to IB, and it is never
// changed before flush. The remote half is produced first so its transfer
// overlaps reduction of the local half.
template <bool Pull>
__global__ void hierarchicalReduceScatterOverlap(LocalDirectChans local,
                                                 PortChan ib, float* base,
                                                 size_t nelem, Layout layout,
                                                 int rank, int localRank,
                                                 size_t receiveOffset) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nth = gridDim.x * blockDim.x;
  size_t shard = nelem / kWorld;
  size_t units = shard / 4;
  char* bytes = reinterpret_cast<char*>(base);
  auto* partial = reinterpret_cast<float4*>(bytes + layout.ibSend);
  auto const* remote = reinterpret_cast<float4 const*>(bytes + receiveOffset);
  int ownHalf = rank / kLocalRanks;
  if (!Pull) {
    // Push coalesced writes to each destination; cross-socket PCIe reads can
    // have substantially lower throughput than writes on this testbed.
    for (int step = 1; step < kLocalRanks; ++step) {
      int dest = (localRank + step) % kLocalRanks;
      for (int half = 0; half < kNodes; ++half) {
        auto const* from =
            reinterpret_cast<float4 const*>(bytes + layout.data) +
            size_t(dest + half * kLocalRanks) * units;
        auto* to = reinterpret_cast<float4*>(
                       reinterpret_cast<char*>(local.c[dest].dst_) +
                       layout.localScratch) +
                   size_t(localRank * kNodes + half) * units;
        for (size_t i = tid; i < units; i += nth) to[i] = from[i];
      }
      if (gridDim.x == 1)
        __syncthreads();
      else
        publish();
    }
  }
  // Publish this invocation's input before any peer loads it.
  if (gridDim.x == 1)
    __syncthreads();
  else
    publish();
  if (tid == 0) {
    for (int peer = 0; peer < kLocalRanks; ++peer)
      if (peer != localRank) local.c[peer].signal();
    for (int peer = 0; peer < kLocalRanks; ++peer)
      if (peer != localRank) local.c[peer].wait();
  }
  if (gridDim.x == 1)
    __syncthreads();
  else
    publish();
  for (int phase = 0; phase < kNodes; ++phase) {
    int half = phase == 0 ? (ownHalf ^ 1) : ownHalf;
    size_t offset = size_t(localRank + half * kLocalRanks) * units;
    for (size_t i = tid; i < units; i += nth) {
      auto const* own = reinterpret_cast<float4 const*>(bytes + layout.data);
      float4 sum = own[offset + i];
      // Visit the same-NUMA pair before the other NUMA group. All destinations
      // use a fixed source order so reduction order does not depend on timing.
      for (int step = 0; step < kLocalRanks; ++step) {
        int source = step ^ (localRank & 2);
        if (source == localRank) continue;
        char const* sourceBase =
            source == localRank
                ? bytes
                : reinterpret_cast<char const*>(local.c[source].dst_);
        auto const* input =
            reinterpret_cast<float4 const*>(sourceBase + layout.data);
        auto const* scratch =
            reinterpret_cast<float4 const*>(bytes + layout.localScratch);
        float4 value =
            (Pull || source == localRank)
                ? input[offset + i]
                : scratch[size_t(source * kNodes + half) * units + i];
        sum.x += value.x;
        sum.y += value.y;
        sum.z += value.z;
        sum.w += value.w;
      }
      partial[size_t(half) * units + i] = sum;
    }
    if (gridDim.x == 1)
      __syncthreads();
    else
      publish();
    if (phase == 0 && tid == 0)
      ib.putWithSignal(receiveOffset,
                       layout.ibSend + size_t(half) * shard * sizeof(float),
                       shard * sizeof(float));
  }
  // Retire local readers while IB is in flight. This protects input/scratch
  // reuse without placing another local handshake after network completion.
  if (tid == 0) {
    for (int peer = 0; peer < kLocalRanks; ++peer)
      if (peer != localRank) local.c[peer].signal();
    for (int peer = 0; peer < kLocalRanks; ++peer)
      if (peer != localRank) local.c[peer].wait();
    ib.wait();
  }
  if (gridDim.x == 1)
    __syncthreads();
  else
    publish();
  for (size_t i = tid; i < units; i += nth) {
    float4 value = partial[size_t(ownHalf) * units + i], peer = remote[i];
    value.x += peer.x;
    value.y += peer.y;
    value.z += peer.z;
    value.w += peer.w;
    partial[size_t(ownHalf) * units + i] = value;
  }
  if (gridDim.x == 1)
    __syncthreads();
  else
    publish();
  if (tid == 0) ib.flush();
}

template <bool ClientIO = false, bool Packet = false>
__device__ __forceinline__ void hierarchicalAllReduceSerialBody(
    LocalDirectChans local, PortChan ib, float* base, size_t nelem,
    Layout layout, int localRank, int ringPos,
    uint32_t const* clientInput = nullptr, uint32_t* clientOutput = nullptr,
    uint32_t packetFlag = 0, bool packetDrain = true) {
  constexpr int ownerChunk[kLocalRanks] = {1, 2, 0, 3};
  size_t chunk = nelem / kLocalRanks;
  size_t units = chunk / 4;
  int ownChunk = (ringPos + 1) % kLocalRanks;
  char* bytes = reinterpret_cast<char*>(base);
  auto* data = reinterpret_cast<float*>(bytes + layout.data);
  auto* scratch = reinterpret_cast<float*>(bytes + layout.localScratch);
  auto* partial = reinterpret_cast<float*>(bytes + layout.ibSend);
  auto const* remote = reinterpret_cast<float const*>(bytes + layout.ibScratch);
  for (int step = 1; step < kLocalRanks; ++step) {
    int dest = (localRank + step) % kLocalRanks;
    auto const* from =
        reinterpret_cast<uint4 const*>(data + size_t(ownerChunk[dest]) * chunk);
    auto* target = reinterpret_cast<uint4*>(
        reinterpret_cast<char*>(local.c[dest].dst_) + layout.localScratch +
        size_t(localRank) * chunk * sizeof(float));
    for (size_t i = threadIdx.x; i < units; i += blockDim.x) {
      if constexpr (ClientIO)
        target[i] =
            loadClientQuad(clientInput, size_t(ownerChunk[dest]) * units + i);
      else
        target[i] = from[i];
    }
#ifdef UCCL_LITE_AR_COMPACT_BARRIERS
    if constexpr (ClientIO) {
      if (step == kLocalRanks - 1) __syncthreads();
    } else
#endif
      __syncthreads();
  }
  if constexpr (ClientIO)
    serialLocalReady(local, localRank);
  else if (threadIdx.x == 0) {
    for (int peer = 0; peer < kLocalRanks; ++peer)
      if (peer != localRank) local.c[peer].signal();
    for (int peer = 0; peer < kLocalRanks; ++peer)
      if (peer != localRank) local.c[peer].wait();
  }
  __syncthreads();
  auto const* own =
      reinterpret_cast<float4 const*>(data + size_t(ownChunk) * chunk);
  auto const* received = reinterpret_cast<float4 const*>(scratch);
  auto* out = reinterpret_cast<float4*>(partial);
  for (size_t i = threadIdx.x; i < units; i += blockDim.x) {
    float4 value;
    if constexpr (ClientIO)
      value = loadClientFloatQuad(clientInput, size_t(ownChunk) * units + i);
    else
      value = own[i];
    for (int source = 0; source < kLocalRanks; ++source) {
      if (source == localRank) continue;
      float4 peer = received[size_t(source) * units + i];
      value.x += peer.x;
      value.y += peer.y;
      value.z += peer.z;
      value.w += peer.w;
    }
    out[i] = value;
  }
#ifdef UCCL_LITE_AR_COMPACT_BARRIERS
  // This serial body has one CTA. The following PortChannel publication
  // provides system release after every producer reaches this CTA barrier.
  if constexpr (ClientIO)
    __syncthreads();
  else
#endif
    publish();
#ifdef UCCL_LITE_AR_EARLY_FLUSH
  uint64_t retireTicket = 0;
#endif
  if constexpr (Packet) {
#ifdef UCCL_LITE_AR_EARLY_FLUSH
    if constexpr (ClientIO)
      retireTicket = packetSend(ib, bytes, layout, layout.ibSend,
                                chunk * sizeof(float), packetFlag, packetDrain);
    else
#endif
      packetSend(ib, bytes, layout, layout.ibSend, chunk * sizeof(float),
                 packetFlag);
    packetReceive(bytes, layout, layout.ibScratch, chunk * sizeof(float),
                  packetFlag);
  } else if (threadIdx.x == 0) {
#ifdef UCCL_LITE_AR_EARLY_FLUSH
    if constexpr (ClientIO)
      // This is the serial algorithm's only IB write. Request its retirement
      // now, then overlap local fan-out and caller-output copying with it.
      retireTicket = ib.fifo_.push(
          {mscclpp::TriggerData | mscclpp::TriggerFlag | mscclpp::TriggerSync,
           ib.dst_, layout.ibScratch, ib.src_, layout.ibSend,
           chunk * sizeof(float), ib.semaphoreId_});
    else
#endif
      ib.putWithSignal(layout.ibScratch, layout.ibSend, chunk * sizeof(float));
    ib.wait();
  }
  __syncthreads();
  auto const* other = reinterpret_cast<float4 const*>(remote);
  auto* ownResult = reinterpret_cast<float4*>(data + size_t(ownChunk) * chunk);
  for (size_t i = threadIdx.x; i < units; i += blockDim.x) {
    float4 value = out[i], peer = other[i];
    value.x += peer.x;
    value.y += peer.y;
    value.z += peer.z;
    value.w += peer.w;
    ownResult[i] = value;
  }
  __syncthreads();
  for (int step = 1; step < kLocalRanks; ++step) {
    int peer = (localRank + step) % kLocalRanks;
    auto* target = reinterpret_cast<float4*>(
        reinterpret_cast<char*>(local.c[peer].dst_) + layout.data +
        size_t(ownChunk) * chunk * sizeof(float));
    for (size_t i = threadIdx.x; i < units; i += blockDim.x)
      target[i] = ownResult[i];
#ifdef UCCL_LITE_AR_COMPACT_BARRIERS
    if constexpr (ClientIO) {
      if (step == kLocalRanks - 1) __syncthreads();
    } else
#endif
      __syncthreads();
  }
  if constexpr (ClientIO)
    serialLocalReady(local, localRank);
  else if (threadIdx.x == 0) {
    for (int peer = 0; peer < kLocalRanks; ++peer)
      if (peer != localRank) local.c[peer].signal();
    for (int peer = 0; peer < kLocalRanks; ++peer)
      if (peer != localRank) local.c[peer].wait();
  }
#ifdef UCCL_LITE_AR_EARLY_FLUSH
  if constexpr (!ClientIO)
#endif
    if (threadIdx.x == 0 && (!Packet || packetDrain)) ib.flush();
  if constexpr (ClientIO) {
    __syncthreads();
    for (size_t i = threadIdx.x; i < nelem / 4; i += blockDim.x)
      storeClientQuad(clientOutput, i, reinterpret_cast<uint4 const*>(data)[i]);
#ifdef UCCL_LITE_AR_EARLY_FLUSH
    // Copying data/output does not mutate the IB source in partial. The same
    // send must still retire before this kernel can complete.
    if (threadIdx.x == 0 && (!Packet || packetDrain))
      endPacketFlush(ib, retireTicket);
#endif
  }
}

template <bool ClientIO = false>
__global__ void hierarchicalAllReduceSerial(
    LocalDirectChans local, PortChan ib, float* base, size_t nelem,
    Layout layout, int localRank, int ringPos,
    uint32_t const* clientInput = nullptr, uint32_t* clientOutput = nullptr) {
  hierarchicalAllReduceSerialBody<ClientIO>(local, ib, base, nelem, layout,
                                            localRank, ringPos, clientInput,
                                            clientOutput);
}

// Reorder a standard rank-ordered tensor into the ring's four owner chunks.
// Each block handles one contiguous source and destination interval.
__global__ void packReduceScatterRing(float const* input, float* packed,
                                      size_t shard) {
  constexpr int ownerForChunk[kLocalRanks] = {2, 0, 1, 3};
  int segment = blockIdx.y;
  int chunk = segment / kNodes;
  int half = segment % kNodes;
  size_t source = size_t(ownerForChunk[chunk] + half * kLocalRanks) * shard;
  size_t destination = size_t(segment) * shard;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < shard;
       i += size_t(gridDim.x) * blockDim.x)
    packed[destination + i] = input[source + i];
}

// The immutable standard-layout source is separate from the mutable ring
// chunks so repeated invocations cannot consume a prior reduction's output.
template <class LocalChan>
__global__ void __launch_bounds__(1024)
    hierarchicalReduceScatterRing(LocalChan localSend, LocalChan localRecv,
                                  PortChan ib, float* base, size_t nelem,
                                  Layout layout, int rank, int ringPos,
                                  bool packedReady, int pieces,
                                  float* clientOutput = nullptr) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nth = gridDim.x * blockDim.x;
  size_t shard = nelem / kWorld;
  size_t chunkElems = 2 * shard;
  float const* input = reinterpret_cast<float const*>(
      reinterpret_cast<char*>(base) + layout.immutableInput);
  float* packed = reinterpret_cast<float*>(reinterpret_cast<char*>(base) +
                                           layout.localScratch);
  float* scratch =
      reinterpret_cast<float*>(reinterpret_cast<char*>(base) + layout.data);
  float* partial =
      reinterpret_cast<float*>(reinterpret_cast<char*>(base) + layout.ibSend);
  float* remote = reinterpret_cast<float*>(reinterpret_cast<char*>(base) +
                                           layout.ibScratch);
  if (!packedReady) {
    constexpr int ownerForChunk[kLocalRanks] = {2, 0, 1, 3};
    for (size_t i = tid; i < nelem; i += nth) {
      int chunk = int(i / chunkElems);
      size_t inChunk = i % chunkElems;
      int half = int(inChunk / shard);
      size_t inShard = inChunk % shard;
      packed[i] =
          input[size_t(ownerForChunk[chunk] + half * kLocalRanks) * shard +
                inShard];
    }
    gMultiNodeSyncer.sync(gridDim.x);
  }
  int ownHalf = rank < kLocalRanks ? 0 : 1;
  int sendHalf = ownHalf ^ 1;
  if (pieces == 1) {
    localReduceScatterPiece(localSend, localRecv, packed, scratch, partial,
                            nelem, ringPos, 0, chunkElems, chunkElems,
                            layout.localScratch, layout.data, tid, nth);
    publish();
    if (tid == 0) {
      ib.putWithSignal(layout.ibScratch,
                       layout.ibSend + size_t(sendHalf) * shard * sizeof(float),
                       shard * sizeof(float));
      ib.wait();
    }
    gMultiNodeSyncer.sync(gridDim.x);
    for (size_t i = tid; i < shard; i += nth) {
      float value = partial[size_t(ownHalf) * shard + i] + remote[i];
      if (clientOutput)
        clientOutput[i] = value;
      else
        partial[size_t(ownHalf) * shard + i] = value;
    }
    gMultiNodeSyncer.sync(gridDim.x);
  } else {
    size_t pieceElems = ((shard + pieces - 1) / pieces + 31) & ~(size_t)31;
    for (int p = 0; p < pieces; ++p) {
      size_t off = size_t(p) * pieceElems;
      if (off >= shard) break;
      size_t len = min(pieceElems, shard - off);
      size_t sendOff = size_t(sendHalf) * shard + off;
      size_t ownOff = size_t(ownHalf) * shard + off;
      localReduceScatterPiece(localSend, localRecv, packed, scratch, partial,
                              nelem, ringPos, sendOff, len, chunkElems,
                              layout.localScratch, layout.data, tid, nth);
      publish();
      if (tid == 0)
        ib.putWithSignal(layout.ibScratch + off * sizeof(float),
                         layout.ibSend + sendOff * sizeof(float),
                         len * sizeof(float));
      localReduceScatterPiece(localSend, localRecv, packed, scratch, partial,
                              nelem, ringPos, ownOff, len, chunkElems,
                              layout.localScratch, layout.data, tid, nth);
      if (tid == 0) ib.wait();
      gMultiNodeSyncer.sync(gridDim.x);
      for (size_t i = tid; i < len; i += nth) {
        float value = partial[ownOff + i] + remote[off + i];
        if (clientOutput)
          clientOutput[off + i] = value;
        else
          partial[ownOff + i] = value;
      }
      gMultiNodeSyncer.sync(gridDim.x);
    }
  }
  if (tid == 0) {
    ib.flush();
    localSend.flush();
  }
}

template <class LocalChan>
__global__ void __launch_bounds__(1024)
    hierarchicalReduceScatterDirectRing(LocalChan localSend,
                                        LocalChan localRecv, PortChan ib,
                                        float* base, size_t nelem,
                                        Layout layout, int rank, int ringPos,
                                        int pieces,
                                        float* clientOutput = nullptr) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nth = gridDim.x * blockDim.x;
  size_t shard = nelem / kWorld;
  float const* input =
      ringInput(localSend, reinterpret_cast<float const*>(
                               reinterpret_cast<char*>(base) + layout.data));
  float* scratch = reinterpret_cast<float*>(reinterpret_cast<char*>(base) +
                                            layout.localScratch);
  float* partial =
      reinterpret_cast<float*>(reinterpret_cast<char*>(base) + layout.ibSend);
  float* remote = reinterpret_cast<float*>(reinterpret_cast<char*>(base) +
                                           layout.ibScratch);
  int ownHalf = rank < kLocalRanks ? 0 : 1;
  int sendHalf = ownHalf ^ 1;
  size_t pieceElems = ((shard + pieces - 1) / pieces + 31) & ~(size_t)31;
  for (int p = 0; p < pieces; ++p) {
    size_t off = size_t(p) * pieceElems;
    if (off >= shard) break;
    size_t len = min(pieceElems, shard - off);
    localReduceScatterDirectHalf(localSend, localRecv, input, scratch, partial,
                                 shard, sendHalf, ringPos, off, len, layout,
                                 tid, nth);
    publish();
    if (tid == 0)
      ib.putWithSignal(
          layout.ibScratch + off * sizeof(float),
          layout.ibSend + (size_t(sendHalf) * shard + off) * sizeof(float),
          len * sizeof(float));
    localReduceScatterDirectHalf(localSend, localRecv, input, scratch, partial,
                                 shard, ownHalf, ringPos, off, len, layout, tid,
                                 nth);
    if (tid == 0) ib.wait();
    gMultiNodeSyncer.sync(gridDim.x);
    // Every local contribution for this piece has been read before this
    // store. The next piece accesses disjoint offsets, including in-place RS.
    for (size_t i = tid; i < len; i += nth) {
      float value =
          partial[size_t(ownHalf) * shard + off + i] + remote[off + i];
      if (clientOutput)
        clientOutput[off + i] = value;
      else
        partial[size_t(ownHalf) * shard + off + i] = value;
    }
    gMultiNodeSyncer.sync(gridDim.x);
  }
  if (tid == 0) {
    ib.flush();
    localSend.flush();
  }
}

}  // namespace uccl_lite::multinode_kernels
