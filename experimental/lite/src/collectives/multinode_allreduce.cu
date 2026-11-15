// Two-node, four-GPU-per-node hierarchical AllReduce for PCIe-only GPUs.
//
// Schedule (H1):
//   1. topology-aware four-GPU ring ReduceScatter inside each node;
//   2. four parallel, rail-local IB exchanges between equal local ranks;
//   3. reduce the remote shard and ring AllGather inside each node.
//
// H2 pipelines the three stages by splitting each rank shard into pieces. While
// the IB channel moves piece p, the local copy engines work on other pieces. It
// is intended for medium/large messages; H1 avoids the extra triggers at small
// sizes.  Both algorithms move the minimum N bytes in each direction between
// two nodes, spread over the two HCAs without routing GPU traffic over UPI.

#include "uccl_lite/multinode_host_gather.cuh"
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/memory_channel_device.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <mscclpp/semaphore.hpp>
#include <mscclpp/utils.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <mpi.h>

constexpr int kNodes = 2;
constexpr int kLocalRanks = 4;
constexpr int kWorld = kNodes * kLocalRanks;

#define CUDACHECK(cmd)                                                   \
  do {                                                                   \
    cudaError_t e_ = (cmd);                                              \
    if (e_ != cudaSuccess) {                                             \
      std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e_));                              \
      MPI_Abort(MPI_COMM_WORLD, 1);                                      \
    }                                                                    \
  } while (0)

#define MPICHECK(cmd)                                                         \
  do {                                                                        \
    int e_ = (cmd);                                                           \
    if (e_ != MPI_SUCCESS) {                                                  \
      char s_[MPI_MAX_ERROR_STRING];                                          \
      int n_ = 0;                                                             \
      MPI_Error_string(e_, s_, &n_);                                          \
      std::fprintf(stderr, "%s:%d MPI error: %.*s\n", __FILE__, __LINE__, n_, \
                   s_);                                                       \
      MPI_Abort(MPI_COMM_WORLD, e_);                                          \
    }                                                                         \
  } while (0)

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

// Local ring order is 0 -> 1 -> 3 -> 2 -> 0.  It crosses the NUMA boundary once
// in each direction, so the UPI is never oversubscribed by two same-direction
// flows.  pos is the rank's position in this ring.
__device__ void localReduceScatterPiece(
    PortChan send, PortChan recv, float* data, float* scratch, float* ibSend,
    size_t nelem, int pos, size_t pieceOff, size_t pieceLen, size_t chunkElems,
    size_t dataByteOffset, size_t scratchByteOffset, int tid, int nth) {
  const size_t chunkBytes = chunkElems * sizeof(float);
  for (int step = 0; step < kLocalRanks - 1; step++) {
    int cSend = (pos - step + kLocalRanks) % kLocalRanks;
    int cRecv = (cSend - 1 + kLocalRanks) % kLocalRanks;
    size_t sendOff = (size_t)cSend * chunkElems + pieceOff;
    size_t recvOff = (size_t)cRecv * chunkElems + pieceOff;
    size_t sendLen = sendOff >= nelem ? 0 : min(pieceLen, nelem - sendOff);
    size_t recvLen = recvOff >= nelem ? 0 : min(pieceLen, nelem - recvOff);

    publish();
    if (tid == 0)
      send.putWithSignal(scratchByteOffset + (size_t)step * chunkBytes +
                             pieceOff * sizeof(float),
                         dataByteOffset + sendOff * sizeof(float),
                         sendLen * sizeof(float));
    if (tid == 0) recv.wait();
    gMultiNodeSyncer.sync(gridDim.x);
    if (recvLen) {
      if (step == kLocalRanks - 2)
        addOut(ibSend + pieceOff, data + recvOff,
               scratch + (size_t)step * chunkElems + pieceOff, recvLen, tid,
               nth);
      else
        addInto(data + recvOff, scratch + (size_t)step * chunkElems + pieceOff,
                recvLen, tid, nth);
    }
    gMultiNodeSyncer.sync(gridDim.x);
  }
}

// Three receive slots per half preserve each outbound partial until the local
// proxy has finished reading it. The original standard-layout input is never
// modified, so the next collective may use it without a reset or repack.
__device__ void localReduceScatterDirectHalf(
    PortChan send, PortChan recv, float const* input, float* scratch,
    float* partial, size_t shard, int half, int ringPos, size_t pieceOff,
    size_t pieceLen, Layout layout, int tid, int nth) {
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
    if (tid == 0) {
      send.putWithSignal(layout.localScratch +
                             receiveOffset * sizeof(float),
                         sourceBase + sourceOffset * sizeof(float), bytes);
      recv.wait();
    }
    gMultiNodeSyncer.sync(gridDim.x);
    float const* own =
        input + size_t(ownerForChunk[cRecv] + half * kLocalRanks) * shard +
        pieceOff;
    float* received = scratch + receiveOffset;
    if (step == kLocalRanks - 2)
      addOut(partial + size_t(half) * shard + pieceOff, own, received,
             pieceLen, tid, nth);
    else
      addInto(received, own, pieceLen, tid, nth);
    gMultiNodeSyncer.sync(gridDim.x);
  }
}

__device__ void localAllGatherPiece(PortChan send, PortChan recv, float* data,
                                    size_t nelem, int pos, size_t pieceOff,
                                    size_t pieceLen, size_t chunkElems,
                                    int tid) {
  for (int step = 0; step < kLocalRanks - 1; step++) {
    int cSend = (pos + 1 - step + kLocalRanks) % kLocalRanks;
    size_t sendOff = (size_t)cSend * chunkElems + pieceOff;
    size_t sendLen = sendOff >= nelem ? 0 : min(pieceLen, nelem - sendOff);

    // Only the first AllGather hop sends bytes produced by SM stores. Later
    // hops forward bytes that arrived through the same CUDA-IPC proxy and are
    // already ordered by wait(); the end-of-step sync aligns every block.
    if (step == 0) publish();
    if (tid == 0)
      send.putWithSignal(sendOff * sizeof(float), sendOff * sizeof(float),
                         sendLen * sizeof(float));
    if (tid == 0) recv.wait();
    gMultiNodeSyncer.sync(gridDim.x);
  }
}

__device__ void localFlatAllGatherPiece(LocalAgChans chans, float* data,
                                        size_t ownBase, size_t pieceOff,
                                        size_t pieceLen, int localRank,
                                        int tid) {
  publish();
  if (tid == 0) {
    size_t byteOff = (ownBase + pieceOff) * sizeof(float);
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

__global__ void __launch_bounds__(1024)
    hierarchicalAllReduce(PortChan localSend, PortChan localRecv,
                          LocalAgChans localAg, PortChan ib, float* base,
                          size_t nelem, Layout layout, int localRank,
                          int ringPos, int pieces, int pipelineDistance,
                          bool flatAg) {
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
  size_t pieceElems = ((ownLen + pieces - 1) / pieces + 31) & ~(size_t)31;

  // A bounded software pipeline.  The IB write for piece p remains in flight
  // while local RS works on later pieces.  Once the chosen distance is reached,
  // reduce and locally publish the oldest completed piece.
  for (int p = 0; p < pieces + pipelineDistance; p++) {
    if (p < pieces) {
      size_t off = (size_t)p * pieceElems;
      size_t len = off >= ownLen ? 0 : min(pieceElems, ownLen - off);
      localReduceScatterPiece(localSend, localRecv, data, localScratch, ibSend,
                              nelem, ringPos, off, len, chunkElems,
                              layout.data, layout.localScratch, tid, nth);
      // The final local RS step writes directly into the immutable IB slot, so
      // no extra staging copy is needed and inbound reduction cannot race the
      // outbound RDMA read.
      publish();
      if (tid == 0)
        ib.putWithSignal(layout.ibScratch + off * sizeof(float),
                         layout.ibSend + off * sizeof(float),
                         len * sizeof(float));
    }

    int done = p - pipelineDistance;
    if (done >= 0 && done < pieces) {
      size_t off = (size_t)done * pieceElems;
      size_t len = off >= ownLen ? 0 : min(pieceElems, ownLen - off);
      if (tid == 0) ib.wait();
      gMultiNodeSyncer.sync(gridDim.x);
      if (len)
        addOut(data + ownBase + off, ibSend + off, ibScratch + off, len, tid,
               nth);
      gMultiNodeSyncer.sync(gridDim.x);
      if (flatAg)
        localFlatAllGatherPiece(localAg, data, ownBase, off, len, localRank,
                                tid);
      else
        localAllGatherPiece(localSend, localRecv, data, nelem, ringPos, off,
                            len, chunkElems, tid);
    }
  }
  if (tid == 0) {
    // CUDA-IPC copies do not consume an IB completion queue and need no flush.
    // Drain the IB QP once per collective to bound its CQ and protect reuse of
    // the source buffer by the next launch.
    ib.flush();
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

// Small-message variant: one resident CTA directly pushes local shards through
// CUDA IPC while the corresponding IB transfer is in flight.
__global__ void hierarchicalAllGatherSerial(LocalDirectChans local,
                                             PortChan ib, float* base,
                                             size_t nelem, Layout layout,
                                             int rank, int localRank) {
  size_t shard = nelem / kWorld;
  size_t units = shard / 4;
  size_t ownOff = size_t(rank) * shard;
  size_t remoteOff = size_t(rank ^ kLocalRanks) * shard;
  char* bytes = reinterpret_cast<char*>(base);
  auto* output = reinterpret_cast<float*>(bytes + layout.data);
  auto const* input = reinterpret_cast<uint4 const*>(bytes + layout.ibSend);
  auto* self = reinterpret_cast<uint4*>(output + ownOff);
  for (size_t i = threadIdx.x; i < units; i += blockDim.x)
    self[i] = input[i];
  __syncthreads();
  if (threadIdx.x == 0)
    ib.putWithSignal(layout.data + ownOff * sizeof(float), layout.ibSend,
                     shard * sizeof(float));
  for (int step = 1; step < kLocalRanks; ++step) {
    int peer = (localRank + step) % kLocalRanks;
    auto* target = reinterpret_cast<uint4*>(
        reinterpret_cast<char*>(local.c[peer].dst_) + layout.data +
        ownOff * sizeof(float));
    for (size_t i = threadIdx.x; i < units; i += blockDim.x)
      target[i] = input[i];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    for (int peer = 0; peer < kLocalRanks; ++peer)
      if (peer != localRank) local.c[peer].signal();
    ib.wait();
  }
  __syncthreads();
  auto const* received = reinterpret_cast<uint4 const*>(output + remoteOff);
  for (int step = 1; step < kLocalRanks; ++step) {
    int peer = (localRank + step) % kLocalRanks;
    auto* target = reinterpret_cast<uint4*>(
        reinterpret_cast<char*>(local.c[peer].dst_) + layout.data +
        remoteOff * sizeof(float));
    for (size_t i = threadIdx.x; i < units; i += blockDim.x)
      target[i] = received[i];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    for (int peer = 0; peer < kLocalRanks; ++peer)
      if (peer != localRank) local.c[peer].signal();
    for (int peer = 0; peer < kLocalRanks; ++peer)
      if (peer != localRank) {
        local.c[peer].wait();
        local.c[peer].wait();
      }
    ib.flush();
  }
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

__device__ void gatherFourRing(PortChan send, PortChan recv, size_t baseOffset,
                               size_t shard, int nodeBase, int ringPos,
                               int tid) {
  constexpr int order[kLocalRanks] = {0, 1, 3, 2};
  for (int step = 0; step < kLocalRanks - 1; ++step) {
    int owner = order[(ringPos - step + kLocalRanks) % kLocalRanks];
    size_t off = size_t(nodeBase + owner) * shard * sizeof(float);
    publish();
    if (tid == 0) {
      send.putWithSignal(baseOffset + off, baseOffset + off,
                         shard * sizeof(float));
      recv.wait();
    }
    gMultiNodeSyncer.sync(gridDim.x);
  }
}

// Ring forwarding avoids the flat multicast's simultaneous fan-out. The
// first local ring runs while the partner rank's IB transfer is in flight.
__global__ void __launch_bounds__(1024)
    hierarchicalAllGatherRing(PortChan localSend, PortChan localRecv,
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
  for (size_t i = tid; i < shard; i += nth)
    output[size_t(rank) * shard + i] = input[i];
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
  float* partial = reinterpret_cast<float*>(reinterpret_cast<char*>(base) +
                                            layout.ibSend);
  float* remote = reinterpret_cast<float*>(reinterpret_cast<char*>(base) +
                                           layout.ibScratch);
  publish();
  if (tid == 0) {
    for (int dest = 0; dest < kLocalRanks; ++dest)
      if (dest != localRank)
        for (int half = 0; half < kNodes; ++half)
          localAg.c[dest].putWithSignal(
              layout.localScratch +
                  size_t(localRank * kNodes + half) * bytes,
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
        float value = source == localRank
                          ? input[size_t(localRank + half * kLocalRanks) *
                                      shard +
                                  i]
                          : scratch[size_t(source * kNodes + half) * shard +
                                    i];
        sum += value;
      }
      partial[size_t(half) * shard + i] = sum;
    }
  publish();
  int ownHalf = rank < kLocalRanks ? 0 : 1;
  if (tid == 0) {
    int sendHalf = ownHalf ^ 1;
    ib.putWithSignal(layout.ibScratch,
                     layout.ibSend + size_t(sendHalf) * bytes, bytes);
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

__global__ void hierarchicalReduceScatterSerial(LocalDirectChans local,
                                                  PortChan ib, float* base,
                                                  size_t nelem, Layout layout,
                                                  int rank, int localRank) {
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
          reinterpret_cast<char*>(local.c[dest].dst_) +
          layout.localScratch +
          size_t(localRank * kNodes + half) * shard * sizeof(float));
      for (size_t i = threadIdx.x; i < units; i += blockDim.x)
        target[i] = from[i];
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
      float4 value = own[i];
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
  if (threadIdx.x == 0) {
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
  }
  __syncthreads();
  if (threadIdx.x == 0) ib.flush();
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

__global__ void hierarchicalAllReduceSerial(LocalDirectChans local,
                                             PortChan ib, float* base,
                                             size_t nelem, Layout layout,
                                             int localRank, int ringPos) {
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
    auto const* from = reinterpret_cast<uint4 const*>(
        data + size_t(ownerChunk[dest]) * chunk);
    auto* target = reinterpret_cast<uint4*>(
        reinterpret_cast<char*>(local.c[dest].dst_) +
        layout.localScratch + size_t(localRank) * chunk * sizeof(float));
    for (size_t i = threadIdx.x; i < units; i += blockDim.x)
      target[i] = from[i];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    for (int peer = 0; peer < kLocalRanks; ++peer)
      if (peer != localRank) local.c[peer].signal();
    for (int peer = 0; peer < kLocalRanks; ++peer)
      if (peer != localRank) local.c[peer].wait();
  }
  __syncthreads();
  auto const* own = reinterpret_cast<float4 const*>(
      data + size_t(ownChunk) * chunk);
  auto const* received = reinterpret_cast<float4 const*>(scratch);
  auto* out = reinterpret_cast<float4*>(partial);
  for (size_t i = threadIdx.x; i < units; i += blockDim.x) {
    float4 value = own[i];
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
  publish();
  if (threadIdx.x == 0) {
    ib.putWithSignal(layout.ibScratch, layout.ibSend,
                     chunk * sizeof(float));
    ib.wait();
  }
  __syncthreads();
  auto const* other = reinterpret_cast<float4 const*>(remote);
  auto* ownResult = reinterpret_cast<float4*>(
      data + size_t(ownChunk) * chunk);
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
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    for (int peer = 0; peer < kLocalRanks; ++peer)
      if (peer != localRank) local.c[peer].signal();
    for (int peer = 0; peer < kLocalRanks; ++peer)
      if (peer != localRank) local.c[peer].wait();
    ib.flush();
  }
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
__global__ void __launch_bounds__(1024)
    hierarchicalReduceScatterRing(PortChan localSend, PortChan localRecv,
                                  PortChan ib, float* base, size_t nelem,
                                  Layout layout, int rank, int ringPos,
                                  bool packedReady, int pieces) {
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
  float* partial = reinterpret_cast<float*>(reinterpret_cast<char*>(base) +
                                            layout.ibSend);
  float* remote = reinterpret_cast<float*>(reinterpret_cast<char*>(base) +
                                           layout.ibScratch);
  if (!packedReady) {
    constexpr int ownerForChunk[kLocalRanks] = {2, 0, 1, 3};
    for (size_t i = tid; i < nelem; i += nth) {
      int chunk = int(i / chunkElems);
      size_t inChunk = i % chunkElems;
      int half = int(inChunk / shard);
      size_t inShard = inChunk % shard;
      packed[i] = input[size_t(ownerForChunk[chunk] + half * kLocalRanks) *
                            shard +
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
    for (size_t i = tid; i < shard; i += nth)
      partial[size_t(ownHalf) * shard + i] += remote[i];
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
      for (size_t i = tid; i < len; i += nth)
        partial[ownOff + i] += remote[off + i];
      gMultiNodeSyncer.sync(gridDim.x);
    }
  }
  if (tid == 0) {
    ib.flush();
    localSend.flush();
  }
}

__global__ void __launch_bounds__(1024)
    hierarchicalReduceScatterDirectRing(PortChan localSend,
                                        PortChan localRecv, PortChan ib,
                                        float* base, size_t nelem, Layout layout,
                                        int rank, int ringPos, int pieces) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nth = gridDim.x * blockDim.x;
  size_t shard = nelem / kWorld;
  float const* input = reinterpret_cast<float const*>(
      reinterpret_cast<char*>(base) + layout.data);
  float* scratch = reinterpret_cast<float*>(reinterpret_cast<char*>(base) +
                                            layout.localScratch);
  float* partial = reinterpret_cast<float*>(reinterpret_cast<char*>(base) +
                                            layout.ibSend);
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
      ib.putWithSignal(layout.ibScratch + off * sizeof(float),
                       layout.ibSend +
                           (size_t(sendHalf) * shard + off) * sizeof(float),
                       len * sizeof(float));
    localReduceScatterDirectHalf(localSend, localRecv, input, scratch, partial,
                                 shard, ownHalf, ringPos, off, len, layout,
                                 tid, nth);
    if (tid == 0) ib.wait();
    gMultiNodeSyncer.sync(gridDim.x);
    for (size_t i = tid; i < len; i += nth)
      partial[size_t(ownHalf) * shard + off + i] += remote[off + i];
    gMultiNodeSyncer.sync(gridDim.x);
  }
  if (tid == 0) {
    ib.flush();
    localSend.flush();
  }
}

__global__ void nullKernel() {}

struct Config {
  bool allGather = false;
  bool reduceScatter = false;
  bool arAuto = false;
  bool hostGather = false;
  int hostRepeatCheck = 0;
  size_t hostPieceBytes = 256UL << 10;
  bool agRing = false;
  bool agAuto = false;
  bool agSerialLocal = false;
  int agPipeline =
      0;  // 1: SM, 2: CE, 3: CE relay, 4: matched CE, 5: matched SM.
  bool agDualRdma = false;
  int agFirstMask = 2;
  size_t agPieceBytes = 64UL << 10;
  int agDepth = 2;
  bool rsRing = false;
  bool rsAuto = false;
  bool rsSerialLocal = false;
  bool rsPullLocal = false;
  bool rsOverlapLocal = false;
  bool rsFastPack = false;
  bool rsDirectRing = false;
  int rsPieces = 1;
  size_t rsTargetPieceBytes = 0;
  bool gpuTiming = false;
  bool checkEach = false;
  size_t minBytes = 1UL << 10;
  size_t maxBytes = 1UL << 30;
  int warmup = 10;
  int iters = 20;
  int blocks = 24;
  int rsFlatBlocks = 24;
  int threads = 512;
  size_t pipelineMinBytes = 4UL << 20;
  size_t pieceBytes = 2UL << 20;
  int pipelineDistance = 2;
  bool flatAg = false;
  bool arSerialLocal = false;
  bool check = true;
  std::array<int, kLocalRanks> railMap{0, 0, 1, 1};
};

static constexpr size_t kSerialArMaxBytes = 1UL << 20;
// On the measured two-node L4 topology, avoid the host phase handoff for
// small AR messages. Larger messages retain the explicit hybrid path.
static constexpr size_t kHybridSerialArMaxBytes = 1UL << 18;
static constexpr size_t kSerialAgMaxBytes = 2UL << 20;
static constexpr size_t kSerialRsMaxBytes = 1UL << 20;

static bool autoPackedRs(size_t bytes) {
  return bytes >= (1UL << 20) && bytes <= (16UL << 20);
}

static bool autoDirectRs(size_t bytes) {
  return bytes == 1024 || bytes >= (32UL << 20);
}

static std::array<int, kLocalRanks> parseRailMap(std::string const& text) {
  std::array<int, kLocalRanks> out{};
  std::stringstream ss(text);
  std::string item;
  for (int i = 0; i < kLocalRanks; i++) {
    if (!std::getline(ss, item, ','))
      throw std::runtime_error("rail map needs four entries");
    out[i] = std::stoi(item);
  }
  if (std::getline(ss, item, ','))
    throw std::runtime_error("rail map needs four entries");
  return out;
}

static mscclpp::Transport ibTransport(int index) {
  if (index < 0 || index >= 8)
    throw std::runtime_error("IB rail index must be in [0,7]");
  return static_cast<mscclpp::Transport>(
      static_cast<int>(mscclpp::Transport::IB0) + index);
}

static int ringPosition(int localRank) {
  constexpr int pos[kLocalRanks] = {0, 1, 3, 2};
  return pos[localRank];
}

static int ringSuccessor(int localRank) {
  constexpr int succ[kLocalRanks] = {1, 3, 0, 2};
  return succ[localRank];
}

static int ringPredecessor(int localRank) {
  constexpr int pred[kLocalRanks] = {2, 0, 3, 1};
  return pred[localRank];
}

int main(int argc, char** argv) {
  // UCX is selected by the explicit hybrid runner; route each process to
  // its NUMA-local HCA before MPI creates its endpoints.
  bool hostRequested = false;
  for (int i = 1; i + 1 < argc; ++i)
    if (std::string(argv[i]) == "--host-gather" && std::atoi(argv[i + 1]))
      hostRequested = true;
  if (hostRequested) {
    char const* local = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    int gpu = local ? std::atoi(local) : 0;
    if (!std::getenv("UCX_NET_DEVICES"))
      setenv("UCX_NET_DEVICES", gpu < 2 ? "mlx5_0:1" : "mlx5_1:1", 1);
    if (!std::getenv("UCX_TLS"))
      setenv("UCX_TLS", "rc_mlx5,ud_mlx5,sm,self", 1);
  }
  MPICHECK(MPI_Init(&argc, &argv));
  int rank = 0, world = 0;
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world));
  if (world != kWorld) {
    if (rank == 0)
      std::fprintf(stderr,
                   "requires exactly %d MPI ranks (two nodes x four GPUs)\n",
                   kWorld);
    MPI_Finalize();
    return 2;
  }

  Config cfg;
  try {
    for (int i = 1; i < argc; i++) {
      std::string key = argv[i];
      auto next = [&]() -> char const* {
        if (++i >= argc) throw std::runtime_error("missing value for " + key);
        return argv[i];
      };
      if (key == "-b")
        cfg.minBytes = std::strtoull(next(), nullptr, 0);
      else if (key == "-e")
        cfg.maxBytes = std::strtoull(next(), nullptr, 0);
      else if (key == "-w")
        cfg.warmup = std::atoi(next());
      else if (key == "-n")
        cfg.iters = std::atoi(next());
      else if (key == "-B")
        cfg.blocks = std::atoi(next());
      else if (key == "-T")
        cfg.threads = std::atoi(next());
      else if (key == "-p")
        cfg.pipelineMinBytes = std::strtoull(next(), nullptr, 0);
      else if (key == "-k")
        cfg.pieceBytes = std::strtoull(next(), nullptr, 0);
      else if (key == "-d")
        cfg.pipelineDistance = std::atoi(next());
      else if (key == "-a")
        cfg.flatAg = std::atoi(next()) != 0;
      else if (key == "--host-repeat-check")
        cfg.hostRepeatCheck = std::atoi(next());
      else if (key == "--host-gather")
        cfg.hostGather = std::atoi(next()) != 0;
      else if (key == "--host-piece-bytes")
        cfg.hostPieceBytes = std::strtoull(next(), nullptr, 0);
      else if (key == "--ar-auto")
        cfg.arAuto = std::atoi(next()) != 0;
      else if (key == "--ar-serial-local")
        cfg.arSerialLocal = std::atoi(next()) != 0;
      else if (key == "-c")
        cfg.check = std::atoi(next()) != 0;
      else if (key == "--rail-map")
        cfg.railMap = parseRailMap(next());
      else if (key == "--op") {
        std::string op = next();
        cfg.allGather = op == "allgather";
        cfg.reduceScatter = op == "reducescatter";
        if (op != "allreduce" && !cfg.allGather && !cfg.reduceScatter)
          throw std::runtime_error(
              "--op must be allreduce, allgather, or reducescatter");
      } else if (key == "--ag-ring")
        cfg.agRing = std::atoi(next()) != 0;
      else if (key == "--ag-auto")
        cfg.agAuto = std::atoi(next()) != 0;
      else if (key == "--ag-pipeline") {
        std::string mode = next();
        if (mode != "sm" && mode != "ce" && mode != "ce-pair" &&
            mode != "matched" && mode != "matched-sm")
          throw std::runtime_error(
              "--ag-pipeline must be sm, ce, ce-pair, matched, or matched-sm");
        cfg.agPipeline = mode == "sm"        ? 1
                         : mode == "ce"      ? 2
                         : mode == "ce-pair" ? 3
                         : mode == "matched" ? 4
                                             : 5;
      } else if (key == "--ag-piece-bytes")
        cfg.agPieceBytes = std::strtoull(next(), nullptr, 0);
      else if (key == "--ag-first-mask")
        cfg.agFirstMask = std::atoi(next());
      else if (key == "--ag-dual-rdma")
        cfg.agDualRdma = std::atoi(next()) != 0;
      else if (key == "--ag-depth")
        cfg.agDepth = std::atoi(next());
      else if (key == "--ag-serial-local")
        cfg.agSerialLocal = std::atoi(next()) != 0;
      else if (key == "--rs-ring")
        cfg.rsRing = std::atoi(next()) != 0;
      else if (key == "--rs-auto")
        cfg.rsAuto = std::atoi(next()) != 0;
      else if (key == "--rs-overlap-local")
        cfg.rsOverlapLocal = std::atoi(next()) != 0;
      else if (key == "--rs-pull-local")
        cfg.rsPullLocal = std::atoi(next()) != 0;
      else if (key == "--rs-serial-local")
        cfg.rsSerialLocal = std::atoi(next()) != 0;
      else if (key == "--rs-flat-blocks")
        cfg.rsFlatBlocks = std::atoi(next());
      else if (key == "--rs-fast-pack")
        cfg.rsFastPack = std::atoi(next()) != 0;
      else if (key == "--rs-direct-ring")
        cfg.rsDirectRing = std::atoi(next()) != 0;
      else if (key == "--rs-pieces")
        cfg.rsPieces = std::atoi(next());
      else if (key == "--rs-piece-bytes")
        cfg.rsTargetPieceBytes = std::strtoull(next(), nullptr, 0);
      else if (key == "--gpu-timing")
        cfg.gpuTiming = std::atoi(next()) != 0;
      else if (key == "--check-each")
        cfg.checkEach = std::atoi(next()) != 0;
      else
        throw std::runtime_error("unknown argument " + key);
    }
  } catch (std::exception const& e) {
    if (rank == 0) std::fprintf(stderr, "%s\n", e.what());
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.arAuto && (cfg.allGather || cfg.reduceScatter)) {
    if (rank == 0) std::fprintf(stderr, "--ar-auto requires AllReduce\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.arSerialLocal && (cfg.allGather || cfg.reduceScatter)) {
    if (rank == 0)
      std::fprintf(stderr, "--ar-serial-local requires AllReduce\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if ((cfg.rsPullLocal || cfg.rsOverlapLocal) &&
      (!cfg.reduceScatter || cfg.rsAuto || cfg.rsRing || cfg.rsSerialLocal ||
       cfg.rsFastPack || cfg.rsDirectRing ||
       (cfg.rsPullLocal && cfg.rsOverlapLocal))) {
    if (rank == 0)
      std::fprintf(
          stderr, "overlap/pull requires ReduceScatter and no other RS mode\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.agFirstMask != 1 && cfg.agFirstMask != 2) {
    if (rank == 0) std::fprintf(stderr, "--ag-first-mask must be 1 or 2\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.agDualRdma && cfg.agPipeline != 1 && cfg.agPipeline != 2) {
    if (rank == 0)
      std::fprintf(stderr, "--ag-dual-rdma requires --ag-pipeline sm or ce\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.agPipeline &&
      (!cfg.allGather || cfg.agAuto || cfg.agRing || cfg.agSerialLocal ||
       cfg.agPieceBytes < 512 || cfg.agPieceBytes % 16 || cfg.agDepth < 1 ||
       cfg.agDepth > 64 ||
       (cfg.maxBytes / kWorld + cfg.agPieceBytes - 1) / cfg.agPieceBytes >
           64)) {
    if (rank == 0)
      std::fprintf(stderr,
                   "AG pipeline requires exclusive AllGather mode, aligned "
                   "pieces >=512 bytes, <=64 pieces, depth 1..64\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.agAuto) cfg.agRing = true;
  if (cfg.rsAuto) {
    cfg.rsRing = true;
    cfg.rsDirectRing = true;
  }
  if (cfg.agRing && !cfg.allGather) {
    if (rank == 0)
      std::fprintf(stderr, "--ag-ring is only valid for --op allgather\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.agSerialLocal && !cfg.allGather) {
    if (rank == 0)
      std::fprintf(stderr, "--ag-serial-local requires AllGather\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.rsSerialLocal && !cfg.reduceScatter) {
    if (rank == 0)
      std::fprintf(stderr, "--rs-serial-local requires ReduceScatter\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.rsRing && !cfg.reduceScatter) {
    if (rank == 0)
      std::fprintf(stderr,
                   "--rs-ring is only valid for --op reducescatter\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.rsFastPack && !cfg.rsRing) {
    if (rank == 0)
      std::fprintf(stderr, "--rs-fast-pack requires --rs-ring 1\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.rsDirectRing && (!cfg.rsRing || cfg.rsFastPack)) {
    if (rank == 0)
      std::fprintf(stderr,
                   "--rs-direct-ring requires --rs-ring 1 and no fast pack\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.rsPieces < 1 || cfg.rsPieces > 32 ||
      (cfg.rsPieces > 1 && !cfg.rsRing)) {
    if (rank == 0)
      std::fprintf(stderr, "--rs-pieces must be 1..32 and requires ring\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.rsTargetPieceBytes &&
      (!cfg.rsDirectRing || cfg.rsPieces != 1 ||
       cfg.rsTargetPieceBytes < 512 || cfg.rsTargetPieceBytes % 512)) {
    if (rank == 0)
      std::fprintf(stderr,
                   "--rs-piece-bytes requires direct ring, default "
                   "--rs-pieces, and 512-byte alignment\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.minBytes < 1024 || cfg.maxBytes < cfg.minBytes ||
      cfg.minBytes % 512 || cfg.maxBytes % 512 || cfg.warmup < 0 ||
      cfg.iters < 1 || cfg.blocks < 1 || cfg.rsFlatBlocks < 1 ||
      cfg.threads < 1 ||
      cfg.threads > 1024 || cfg.pieceBytes < 512 ||
      cfg.pipelineDistance < 1) {
    if (rank == 0)
      std::fprintf(stderr, "invalid size, launch, or iteration settings\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  bool hybridRs = cfg.hostGather && !cfg.allGather;
  if (cfg.hostRepeatCheck < 0 || cfg.hostRepeatCheck > 8 ||
      (cfg.hostRepeatCheck && (!hybridRs || cfg.reduceScatter))) {
    if (rank == 0)
      std::fprintf(
          stderr,
          "--host-repeat-check requires hybrid AllReduce and count 1..8\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cfg.hostGather) {
    if (cfg.reduceScatter || cfg.gpuTiming || cfg.arAuto || cfg.agAuto ||
        cfg.agPipeline || cfg.agRing || cfg.agSerialLocal ||
        cfg.arSerialLocal || cfg.rsAuto || cfg.rsRing || cfg.rsSerialLocal ||
        cfg.rsPullLocal || cfg.rsOverlapLocal || cfg.maxBytes > (1UL << 30) ||
        cfg.hostPieceBytes < 128 || cfg.hostPieceBytes > (16UL << 20) ||
        cfg.hostPieceBytes % 128) {
      if (rank == 0)
        std::fprintf(stderr,
                     "--host-gather requires exclusive eager AR/AG, host "
                     "timing and aligned host pieces\n");
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
    if (hybridRs) {
      cfg.rsAuto = cfg.rsRing = cfg.rsDirectRing = true;
      cfg.rsTargetPieceBytes = 8UL << 20;
      cfg.blocks = 96;
      cfg.threads = 512;
    }
  }

  MPI_Comm localComm;
  MPICHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                               MPI_INFO_NULL, &localComm));
  int localRank = 0, localSize = 0;
  MPICHECK(MPI_Comm_rank(localComm, &localRank));
  MPICHECK(MPI_Comm_size(localComm, &localSize));
  if (localSize != kLocalRanks) {
    std::fprintf(stderr, "rank %d: expected four ranks per node, got %d\n",
                 rank, localSize);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  CUDACHECK(cudaSetDevice(localRank));
  cudaDeviceProp localProperty{};
  CUDACHECK(cudaGetDeviceProperties(&localProperty, localRank));
  int residentBlocks = 0;
  if (cfg.allGather) {
    if (cfg.agPipeline == 5)
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &residentBlocks, hierarchicalAllGatherMatched<false>, cfg.threads,
          0));
    else if (cfg.agPipeline == 4)
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &residentBlocks, hierarchicalAllGatherMatched<true>, cfg.threads, 0));
    else if (cfg.agDualRdma && cfg.agPipeline == 1)
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &residentBlocks, hierarchicalAllGatherPipeline<false, false, true>,
          cfg.threads, 0));
    else if (cfg.agDualRdma)
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &residentBlocks, hierarchicalAllGatherPipeline<true, false, true>,
          cfg.threads, 0));
    else if (cfg.agPipeline == 1)
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &residentBlocks, hierarchicalAllGatherPipeline<false>, cfg.threads,
          0));
    else if (cfg.agPipeline == 3)
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &residentBlocks, hierarchicalAllGatherPipeline<true, true>,
          cfg.threads, 0));
    else if (cfg.agPipeline == 2)
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &residentBlocks, hierarchicalAllGatherPipeline<true>, cfg.threads,
          0));
    else if (cfg.agRing)
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &residentBlocks, hierarchicalAllGatherRing, cfg.threads, 0));
    else
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &residentBlocks, hierarchicalAllGather, cfg.threads, 0));
    if (cfg.agAuto) {
      int flatResidentBlocks = 0;
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &flatResidentBlocks, hierarchicalAllGather, cfg.threads, 0));
      residentBlocks = std::min(residentBlocks, flatResidentBlocks);
    }
  } else if (cfg.reduceScatter || hybridRs) {
    if (cfg.rsPullLocal)
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &residentBlocks, hierarchicalReduceScatterOverlap<true>, cfg.threads,
          0));
    else if (cfg.rsOverlapLocal)
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &residentBlocks, hierarchicalReduceScatterOverlap<false>, cfg.threads,
          0));
    else if (cfg.rsDirectRing)
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &residentBlocks, hierarchicalReduceScatterDirectRing, cfg.threads,
          0));
    else if (cfg.rsRing)
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &residentBlocks, hierarchicalReduceScatterRing, cfg.threads, 0));
    else
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &residentBlocks, hierarchicalReduceScatter, cfg.threads, 0));
    if (cfg.rsAuto) {
      int flatResidentBlocks = 0;
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &flatResidentBlocks, hierarchicalReduceScatter, cfg.threads, 0));
      residentBlocks = std::min(residentBlocks, flatResidentBlocks);
      int packedResidentBlocks = 0;
      CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &packedResidentBlocks, hierarchicalReduceScatterRing, cfg.threads,
          0));
      residentBlocks = std::min(residentBlocks, packedResidentBlocks);
    }
  } else
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &residentBlocks, hierarchicalAllReduce, cfg.threads, 0));
  if (cfg.blocks > residentBlocks * localProperty.multiProcessorCount) {
    std::fprintf(stderr,
                 "rank %d: grid %d blocks exceeds resident limit %d for "
                 "this cooperative kernel\n",
                 rank, cfg.blocks,
                 residentBlocks * localProperty.multiProcessorCount);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  int rail = cfg.railMap[localRank];
  int nIb = mscclpp::getIBDeviceCount();
  if (rail >= nIb) {
    std::fprintf(stderr,
                 "rank %d: rail %d requested but MSCCL++ sees %d IB devices; "
                 "build with MSCCLPP_USE_IB=ON\n",
                 rank, rail, nIb);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, world);
  mscclpp::UniqueId id{};
  if (rank == 0) id = mscclpp::TcpBootstrap::createUniqueId();
  MPICHECK(MPI_Bcast(id.data(), id.size(), MPI_BYTE, 0, MPI_COMM_WORLD));
  bootstrap->initialize(id, 120);
  mscclpp::Communicator comm(bootstrap);

  size_t dataBytes = cfg.maxBytes;
  size_t localScratchBytes = dataBytes;
  size_t ibSlotBytes = (dataBytes + kLocalRanks - 1) / kLocalRanks + 4096;
  Layout layout{0, dataBytes, dataBytes + localScratchBytes,
                dataBytes + localScratchBytes + ibSlotBytes, 0};
  size_t bufferBytes = layout.ibScratch + ibSlotBytes;
  if (!hybridRs && cfg.rsRing && (!cfg.rsDirectRing || cfg.rsAuto)) {
    layout.immutableInput = bufferBytes;
    bufferBytes += dataBytes;
  }
  if (bufferBytes >= (1ULL << 32)) {
    if (rank == 0)
      std::fprintf(
          stderr,
          "registered region exceeds PortChannel's 32-bit offset limit\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  char* buffer = nullptr;
  CUDACHECK(cudaMalloc(&buffer, bufferBytes));
  CUDACHECK(cudaMemset(buffer, 0, bufferBytes));

  mscclpp::TransportFlags transports = mscclpp::Transport::CudaIpc;
  transports |= ibTransport(rail);
  auto localMemory = comm.registerMemory(buffer, bufferBytes, transports);

  int nodeBase = rank - localRank;
  int succ = nodeBase + ringSuccessor(localRank);
  int pred = nodeBase + ringPredecessor(localRank);
  int diag = nodeBase + (localRank ^ 3);
  int cross = rank ^ kLocalRanks;
  std::array<int, 4> peers{succ, pred, diag, cross};
  std::array<mscclpp::Transport, 4> peerTransports{
      mscclpp::Transport::CudaIpc, mscclpp::Transport::CudaIpc,
      mscclpp::Transport::CudaIpc, ibTransport(rail)};
  std::array<std::shared_future<mscclpp::Connection>, 4> connectionFutures;
  std::array<std::shared_future<mscclpp::RegisteredMemory>, 4> memoryFutures;
  for (int i = 0; i < 4; i++) {
    connectionFutures[i] =
        comm.connect(peerTransports[i], peers[i], i < 3 ? 100 : 102);
    comm.sendMemory(localMemory, peers[i], i < 3 ? 200 : 202);
    memoryFutures[i] = comm.recvMemory(peers[i], i < 3 ? 200 : 202);
  }

  std::shared_future<mscclpp::Connection> duplicateConnectionFuture;
  std::shared_future<mscclpp::RegisteredMemory> duplicateMemoryFuture;
  if (cfg.agDualRdma) {
    duplicateConnectionFuture = comm.connect(ibTransport(rail), cross ^ 1, 104);
    comm.sendMemory(localMemory, cross ^ 1, 204);
    duplicateMemoryFuture = comm.recvMemory(cross ^ 1, 204);
  }

  std::array<mscclpp::Connection, 4> connections;
  std::array<mscclpp::RegisteredMemory, 4> remoteMemory;
  for (int i = 0; i < 4; i++) {
    connections[i] = connectionFutures[i].get();
    remoteMemory[i] = memoryFutures[i].get();
  }

  // Local DMA and IB must have independent proxy threads/FIFOs; sharing one
  // ProxyService serializes trigger submission and defeats the software
  // pipeline even though the PCIe copy engines and NIC can run concurrently.
  auto localProxy = std::make_shared<mscclpp::ProxyService>();
  auto ibProxy = std::make_shared<mscclpp::ProxyService>();
  std::array<std::shared_future<mscclpp::Semaphore>, 4> semaphoreFutures;
  for (int i = 0; i < 4; i++)
    semaphoreFutures[i] =
        comm.buildSemaphore(connections[i], peers[i], i < 3 ? 300 : 302);
  auto localSendSemaphore = localProxy->addSemaphore(semaphoreFutures[0].get());
  auto localRecvSemaphore = localProxy->addSemaphore(semaphoreFutures[1].get());
  auto localDiagSemaphore = localProxy->addSemaphore(semaphoreFutures[2].get());
  auto ibSemaphore = ibProxy->addSemaphore(semaphoreFutures[3].get());
  auto localLocalMemoryId = localProxy->addMemory(localMemory);
  auto localSendMemoryId = localProxy->addMemory(remoteMemory[0]);
  auto localRecvMemoryId = localProxy->addMemory(remoteMemory[1]);
  auto localDiagMemoryId = localProxy->addMemory(remoteMemory[2]);
  auto ibLocalMemoryId = ibProxy->addMemory(localMemory);
  auto ibRemoteMemoryId = ibProxy->addMemory(remoteMemory[3]);

  // Separate objects are retained because their device handles alias proxy
  // state.
  auto localSendObject = localProxy->portChannel(
      localSendSemaphore, localSendMemoryId, localLocalMemoryId);
  auto localRecvObject = localProxy->portChannel(
      localRecvSemaphore, localRecvMemoryId, localLocalMemoryId);
  auto localDiagObject = localProxy->portChannel(
      localDiagSemaphore, localDiagMemoryId, localLocalMemoryId);
  auto ibObject =
      ibProxy->portChannel(ibSemaphore, ibRemoteMemoryId, ibLocalMemoryId);
  PortChan duplicateIb{};
  std::unique_ptr<mscclpp::PortChannel> duplicateObject;
  if (cfg.agDualRdma) {
    auto connection = duplicateConnectionFuture.get();
    auto memory = duplicateMemoryFuture.get();
    auto semaphore = comm.buildSemaphore(connection, cross ^ 1, 304);
    auto semaphoreId = ibProxy->addSemaphore(semaphore.get());
    auto memoryId = ibProxy->addMemory(memory);
    duplicateObject = std::make_unique<mscclpp::PortChannel>(
        ibProxy->portChannel(semaphoreId, memoryId, ibLocalMemoryId));
    duplicateIb = duplicateObject->deviceHandle();
  }
  PortChan localSend = localSendObject.deviceHandle();
  PortChan localRecv = localRecvObject.deviceHandle();
  LocalAgChans localAg{};
  localAg.c[ringSuccessor(localRank)] = localSendObject.deviceHandle();
  localAg.c[ringPredecessor(localRank)] = localRecvObject.deviceHandle();
  localAg.c[localRank ^ 3] = localDiagObject.deviceHandle();
  LocalDirectChans localDirect{};
  std::array<std::unique_ptr<mscclpp::MemoryChannel>, 3> directObjects;
  bool needsDirectLocal =
      cfg.agPipeline == 1 || cfg.agPipeline >= 4 || cfg.rsOverlapLocal ||
      cfg.rsPullLocal || cfg.agSerialLocal || cfg.rsSerialLocal ||
      cfg.arSerialLocal ||
      (cfg.agAuto && cfg.allGather && cfg.minBytes <= kSerialAgMaxBytes) ||
      (cfg.rsAuto && (cfg.reduceScatter || hybridRs) &&
       cfg.minBytes <= kSerialRsMaxBytes) ||
      (cfg.arAuto && !cfg.allGather && !cfg.reduceScatter &&
       cfg.minBytes <= kSerialArMaxBytes);
  if (needsDirectLocal) {
    std::array<std::shared_future<mscclpp::Semaphore>, 3> directSemaphores;
    for (int i = 0; i < 3; ++i)
      directSemaphores[i] =
          comm.buildSemaphore(connections[i], peers[i], 320);
    for (int i = 0; i < 3; ++i) {
      directObjects[i] = std::make_unique<mscclpp::MemoryChannel>(
          directSemaphores[i].get(), remoteMemory[i], localMemory, buffer);
      localDirect.c[peers[i] - nodeBase] = directObjects[i]->deviceHandle();
    }
  }
  PortChan ib = ibObject.deviceHandle();
  localProxy->startProxy();
  ibProxy->startProxy();

  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cudaEvent_t timingStart = nullptr, timingEnd = nullptr;
  if (cfg.gpuTiming) {
    CUDACHECK(cudaEventCreate(&timingStart));
    CUDACHECK(cudaEventCreate(&timingEnd));
  }
  std::vector<float> hostIn, hostOut;

  auto fill = [&](size_t count, int generation) {
    size_t inputCount = cfg.allGather ? count / kWorld : count;
    hostIn.resize(inputCount);
    for (size_t i = 0; i < inputCount; i++) {
      hostIn[i] = (float)(rank + 1 + generation * kWorld) *
                  (float)(1 + i % 97);
    }
    if (cfg.allGather)
      CUDACHECK(cudaMemsetAsync(buffer + layout.data, 0xff,
                                count * sizeof(float), stream));
    size_t inputOffset = cfg.allGather ? layout.ibSend : layout.data;
    bool serialRs = cfg.rsSerialLocal ||
                    (cfg.rsAuto && count * sizeof(float) <=
                                      kSerialRsMaxBytes);
    if (cfg.reduceScatter && !serialRs && cfg.rsRing &&
        (!cfg.rsDirectRing ||
         (cfg.rsAuto && autoPackedRs(count * sizeof(float)))))
      inputOffset = layout.immutableInput;
    CUDACHECK(cudaMemcpyAsync(buffer + inputOffset,
                              hostIn.data(), inputCount * sizeof(float),
                              cudaMemcpyHostToDevice, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
  };
  auto verify = [&](size_t count, int generation) {
    size_t outputCount = cfg.reduceScatter ? count / kWorld : count;
    char* output = buffer;
    if (cfg.reduceScatter)
      output += layout.ibSend +
                size_t(rank < kLocalRanks ? 0 : 1) * outputCount *
                    sizeof(float);
    hostOut.resize(outputCount);
    CUDACHECK(cudaMemcpy(hostOut.data(), output,
                         outputCount * sizeof(float),
                         cudaMemcpyDeviceToHost));
    double worst = 0.0;
    size_t firstBad = outputCount;
    double rankSum = (double)kWorld * (kWorld + 1) / 2.0 +
                     generation * kWorld * kWorld;
    for (size_t i = 0; i < outputCount; i++) {
      double want = cfg.allGather
                        ? (double)(i / (count / kWorld) + 1 +
                                   generation * kWorld) *
                              (double)(1 + (i % (count / kWorld)) % 97)
                    : cfg.reduceScatter
                        ? rankSum *
                              (double)(1 +
                                       (size_t(rank) * outputCount + i) % 97)
                        : rankSum * (double)(1 + i % 97);
      double error = std::fabs((double)hostOut[i] - want) / want;
      if ((!std::isfinite(error) || error > 1e-5) && firstBad == outputCount)
        firstBad = i;
      worst = std::max(worst, std::isfinite(error) ? error : INFINITY);
    }
    if (firstBad != outputCount) {
      double want = cfg.allGather
                        ? (double)(firstBad / (count / kWorld) + 1 +
                                   generation * kWorld) *
                              (double)(1 + (firstBad % (count / kWorld)) % 97)
                    : cfg.reduceScatter
                        ? rankSum *
                              (double)(1 + (size_t(rank) * outputCount +
                                            firstBad) %
                                           97)
                        : rankSum * (double)(1 + firstBad % 97);
      std::fprintf(
          stderr,
          "rank %d first bad index %zu got %.1f want %.1f relerr %.3e\n", rank,
          firstBad, (double)hostOut[firstBad], want,
          std::fabs((double)hostOut[firstBad] - want) / want);
    }
    return worst;
  };
  auto useAgRing = [&](size_t bytes) {
    return cfg.agAuto ? bytes == 1024 || bytes >= (1UL << 20) : cfg.agRing;
  };
  auto useAgSerial = [&](size_t bytes) {
    return cfg.agSerialLocal ||
           (cfg.agAuto && bytes <= kSerialAgMaxBytes);
  };
  auto useRsSerial = [&](size_t bytes) {
    return cfg.rsSerialLocal ||
           (cfg.rsAuto && bytes <= kSerialRsMaxBytes);
  };
  auto useArSerial = [&](size_t bytes) {
    return cfg.arSerialLocal ||
           (cfg.arAuto && bytes <= kSerialArMaxBytes);
  };
  auto useRsDirectRing = [&](size_t bytes) {
    return cfg.rsAuto ? autoDirectRs(bytes)
                      : cfg.rsDirectRing;
  };
  auto useArFlatGather = [&](size_t bytes) {
    return cfg.arAuto ? bytes <= (1UL << 18) : cfg.flatAg;
  };
  // Two receive slots prevent a faster partner's next call from overwriting
  // data still being consumed. A partner cannot reach call i+2 until it has
  // received call i+1, by which time this rank has finished consuming call i.
  // The allocation already reserves two maximum-size RS shards for IB scratch.
  std::unique_ptr<uccl_lite::host_gather::Gather> hostGather;
  if (cfg.hostGather &&
      (cfg.allGather || cfg.maxBytes > kHybridSerialArMaxBytes))
    hostGather = std::make_unique<uccl_lite::host_gather::Gather>(
        rank, localComm, cfg.maxBytes / kWorld, cfg.hostPieceBytes);
  unsigned overlapSequence = 0;
  unsigned agSequence = 0;
  auto launch = [&](size_t count, size_t bytes) {
    CUDACHECK(cudaSetDevice(localRank));
    if (hostGather) uccl_lite::host_gather::clearMpiProbeStatus();
    if (cfg.allGather) {
      if (hostGather) {
        hostGather->run(buffer + layout.ibSend, buffer + layout.data,
                        bytes / kWorld);
        return;
      }
      if (cfg.agPipeline) {
        size_t slot = agSequence++ & 1;
        size_t receiveOffset =
            layout.ibScratch + slot * (cfg.maxBytes / kWorld);
        size_t relayHalfBytes = cfg.maxBytes / kWorld;
        size_t relayOffset = layout.localScratch + slot * 2 * relayHalfBytes;
        if (cfg.agPipeline == 5)
          hierarchicalAllGatherMatched<false>
              <<<cfg.blocks, cfg.threads, 0, stream>>>(
                  localDirect, localAg, ib, reinterpret_cast<float*>(buffer),
                  count, layout, rank, localRank, receiveOffset,
                  cfg.agPieceBytes, cfg.agDepth, cfg.agFirstMask);
        else if (cfg.agPipeline == 4)
          hierarchicalAllGatherMatched<true>
              <<<cfg.blocks, cfg.threads, 0, stream>>>(
                  localDirect, localAg, ib, reinterpret_cast<float*>(buffer),
                  count, layout, rank, localRank, receiveOffset,
                  cfg.agPieceBytes, cfg.agDepth, cfg.agFirstMask);
        else if (cfg.agDualRdma && cfg.agPipeline == 1)
          hierarchicalAllGatherPipeline<false, false, true>
              <<<cfg.blocks, cfg.threads, 0, stream>>>(
                  localDirect, localAg, ib, duplicateIb,
                  reinterpret_cast<float*>(buffer), count, layout, rank,
                  localRank, receiveOffset, cfg.agPieceBytes, cfg.agDepth,
                  relayOffset, relayHalfBytes);
        else if (cfg.agDualRdma)
          hierarchicalAllGatherPipeline<true, false, true>
              <<<cfg.blocks, cfg.threads, 0, stream>>>(
                  localDirect, localAg, ib, duplicateIb,
                  reinterpret_cast<float*>(buffer), count, layout, rank,
                  localRank, receiveOffset, cfg.agPieceBytes, cfg.agDepth,
                  relayOffset, relayHalfBytes);
        else if (cfg.agPipeline == 3)
          hierarchicalAllGatherPipeline<true, true>
              <<<cfg.blocks, cfg.threads, 0, stream>>>(
                  localDirect, localAg, ib, duplicateIb,
                  reinterpret_cast<float*>(buffer), count, layout, rank,
                  localRank, receiveOffset, cfg.agPieceBytes, cfg.agDepth,
                  relayOffset, relayHalfBytes);
        else if (cfg.agPipeline == 2)
          hierarchicalAllGatherPipeline<true>
              <<<cfg.blocks, cfg.threads, 0, stream>>>(
                  localDirect, localAg, ib, duplicateIb,
                  reinterpret_cast<float*>(buffer), count, layout, rank,
                  localRank, receiveOffset, cfg.agPieceBytes, cfg.agDepth,
                  relayOffset, relayHalfBytes);
        else
          hierarchicalAllGatherPipeline<false>
              <<<cfg.blocks, cfg.threads, 0, stream>>>(
                  localDirect, localAg, ib, duplicateIb,
                  reinterpret_cast<float*>(buffer), count, layout, rank,
                  localRank, receiveOffset, cfg.agPieceBytes, cfg.agDepth,
                  relayOffset, relayHalfBytes);
      } else if (useAgSerial(bytes))
        hierarchicalAllGatherSerial<<<1, 256, 0, stream>>>(
            localDirect, ib, reinterpret_cast<float*>(buffer), count,
            layout, rank, localRank);
      else if (useAgRing(bytes))
        hierarchicalAllGatherRing<<<cfg.blocks, cfg.threads, 0, stream>>>(
            localSend, localRecv, ib, reinterpret_cast<float*>(buffer), count,
            layout, rank, localRank, ringPosition(localRank));
      else
        hierarchicalAllGather<<<cfg.blocks, cfg.threads, 0, stream>>>(
            localAg, ib, reinterpret_cast<float*>(buffer), count, layout,
            rank, localRank);
      return;
    }
    if (hybridRs && bytes <= kHybridSerialArMaxBytes) {
      hierarchicalAllReduceSerial<<<1, 256, 0, stream>>>(
          localDirect, ib, reinterpret_cast<float*>(buffer), count, layout,
          localRank, ringPosition(localRank));
      return;
    }
    if (cfg.reduceScatter || hybridRs) {
      if (cfg.rsPullLocal)
        hierarchicalReduceScatterOverlap<true>
            <<<cfg.blocks, cfg.threads, 0, stream>>>(
                localDirect, ib, reinterpret_cast<float*>(buffer), count,
                layout, rank, localRank,
                layout.ibScratch +
                    size_t(overlapSequence++ & 1) * (cfg.maxBytes / kWorld));
      else if (cfg.rsOverlapLocal)
        hierarchicalReduceScatterOverlap<false>
            <<<cfg.blocks, cfg.threads, 0, stream>>>(
                localDirect, ib, reinterpret_cast<float*>(buffer), count,
                layout, rank, localRank,
                layout.ibScratch +
                    size_t(overlapSequence++ & 1) * (cfg.maxBytes / kWorld));
      else if (useRsSerial(bytes))
        hierarchicalReduceScatterSerial<<<1, 256, 0, stream>>>(
            localDirect, ib, reinterpret_cast<float*>(buffer), count,
            layout, rank, localRank);
      else if (useRsDirectRing(bytes)) {
        int availablePieces =
            int(std::max<size_t>(1, count / kWorld / 32));
        int requestedPieces = cfg.rsTargetPieceBytes
                                  ? int(std::max<size_t>(
                                        2, (bytes / kWorld +
                                            cfg.rsTargetPieceBytes - 1) /
                                               cfg.rsTargetPieceBytes))
                                  : cfg.rsPieces;
        int pieces = std::min(requestedPieces, availablePieces);
        hierarchicalReduceScatterDirectRing<<<cfg.blocks, cfg.threads, 0,
                                              stream>>>(
            localSend, localRecv, ib, reinterpret_cast<float*>(buffer), count,
            layout, rank, ringPosition(localRank), pieces);
      } else if (cfg.rsRing && (!cfg.rsAuto || autoPackedRs(bytes))) {
        int availablePieces =
            int(std::max<size_t>(1, count / kWorld / 32));
        int pieces = cfg.rsAuto ? 1 : std::min(cfg.rsPieces, availablePieces);
        bool fastPack = cfg.rsAuto || cfg.rsFastPack;
        if (fastPack) {
          size_t shard = count / kWorld;
          int blocks = int(std::min<size_t>(128, (shard + 255) / 256));
          packReduceScatterRing<<<dim3(blocks, kWorld), 256, 0, stream>>>(
              reinterpret_cast<float*>(
                  buffer + (hybridRs ? layout.data : layout.immutableInput)),
              reinterpret_cast<float*>(buffer + layout.localScratch), shard);
          if (hybridRs) {
            // Packed RS receives into data. Every GPU must finish reading its
            // in-place AR input before any peer can overwrite that scratch.
            CUDACHECK(cudaGetLastError());
            CUDACHECK(cudaStreamSynchronize(stream));
            MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
          }
        }
        hierarchicalReduceScatterRing<<<cfg.blocks, cfg.threads, 0, stream>>>(
            localSend, localRecv, ib, reinterpret_cast<float*>(buffer), count,
            layout, rank, ringPosition(localRank), fastPack, pieces);
      } else
        hierarchicalReduceScatter<<<cfg.rsAuto ? cfg.rsFlatBlocks : cfg.blocks,
                                    cfg.threads,
                                    0, stream>>>(
            localAg, ib, reinterpret_cast<float*>(buffer), count, layout,
            rank, localRank);
      if (hybridRs) {
        CUDACHECK(cudaGetLastError());
        CUDACHECK(cudaStreamSynchronize(stream));
        hostGather->run(buffer + layout.ibSend +
                            size_t(rank / kLocalRanks) * (bytes / kWorld),
                        buffer + layout.data, bytes / kWorld);
      }
      return;
    }
    size_t shardBytes = (bytes + kLocalRanks - 1) / kLocalRanks;
    int pieces =
        bytes >= cfg.pipelineMinBytes
            ? (int)std::max<size_t>(
                  2, std::min<size_t>(32, (shardBytes + cfg.pieceBytes - 1) /
                                              cfg.pieceBytes))
            : 1;
    int distance =
        pieces == 1 ? 1 : std::max(1, std::min(cfg.pipelineDistance, pieces));
    if (useArSerial(bytes))
      hierarchicalAllReduceSerial<<<1, 256, 0, stream>>>(
          localDirect, ib, reinterpret_cast<float*>(buffer), count, layout,
          localRank, ringPosition(localRank));
    else
      hierarchicalAllReduce<<<cfg.blocks, cfg.threads, 0, stream>>>(
          localSend, localRecv, localAg, ib, reinterpret_cast<float*>(buffer),
          count, layout, localRank, ringPosition(localRank), pieces, distance,
          useArFlatGather(bytes));
  };

  char host[MPI_MAX_PROCESSOR_NAME];
  int hostLen = 0;
  MPI_Get_processor_name(host, &hostLen);
  char line[512]{};
  std::snprintf(line, sizeof(line),
                "# rank %d node %.*s gpu %d %s rail %d (%s)\n", rank, hostLen,
                host, localRank, localProperty.name, rail,
                mscclpp::getIBDeviceName(ibTransport(rail)).c_str());
  std::vector<char> lines(rank == 0 ? world * sizeof(line) : 0);
  MPICHECK(MPI_Gather(line, sizeof(line), MPI_BYTE, lines.data(), sizeof(line),
                      MPI_BYTE, 0, MPI_COMM_WORLD));
  if (rank == 0) {
    std::printf("# UCCL Lite two-node hierarchical %s, float32\n",
                cfg.allGather      ? "AllGather"
                : cfg.reduceScatter ? "ReduceScatter"
                                    : "AllReduce");
    std::printf("# timing=%s; slowest of eight ranks\n",
                cfg.gpuTiming ? "CUDA event" : "host enqueue+completion");
    for (int r = 0; r < world; r++)
      std::fputs(lines.data() + r * sizeof(line), stdout);
    if (cfg.hostGather)
      std::printf(
          "# explicit host packet gather; chunk=%zu; slots=3; GPU "
          "reduction=%s; timing includes phase handoff\n",
          cfg.hostPieceBytes,
          hybridRs ? "tuned RS 96x512, 8MiB target" : "none");
    if (cfg.allGather || cfg.reduceScatter)
      std::printf("# local %s, remote-rank IB exchange; rail map "
                  "%d,%d,%d,%d; RS pieces %d\n",
                  cfg.allGather ? (cfg.agSerialLocal ? "serial direct local AG"
                                   : cfg.agAuto ? "adaptive ring/multicast"
                                   : cfg.agRing ? "ring" : "multicast")
                                : (cfg.rsSerialLocal ? "serial direct local reduction"
                                   : cfg.rsAuto ? "adaptive direct/packed/flat reduction"
                                   : cfg.rsDirectRing ? "direct ring reduction"
                                   : cfg.rsRing ? (cfg.rsFastPack
                                                    ? "ring reduction + GPU pack"
                                                    : "ring reduction")
                                              : "flat reduction"),
                  cfg.railMap[0], cfg.railMap[1], cfg.railMap[2],
                  cfg.railMap[3], cfg.rsPieces);
    if (cfg.rsTargetPieceBytes && rank == 0)
      std::printf("# RS target piece %zu B per destination shard\n",
                  cfg.rsTargetPieceBytes);
    else
      std::printf(
          "# local ring 0->1->3->2->0; AG %s; rail map %d,%d,%d,%d; "
          "piece %zu B; pipeline >= %zu B\n",
          cfg.arSerialLocal ? "serial direct local AR"
          : cfg.arAuto ? "adaptive flat/ring"
                     : cfg.flatAg ? "flat" : "ring",
          cfg.railMap[0], cfg.railMap[1],
          cfg.railMap[2], cfg.railMap[3], cfg.pieceBytes,
          cfg.pipelineMinBytes);
    std::printf("# %12s %12s %8s %10s %10s %10s %10s\n", "size(B)", "count",
                "pieces", "time(us)", "algbw", "busbw", "maxrelerr");
  }

  for (size_t bytes = cfg.minBytes; bytes <= cfg.maxBytes; bytes *= 2) {
    bytes =
        (bytes + 511) & ~(size_t)511;  // divisible into aligned four-way chunks
    size_t count = bytes / sizeof(float);
    size_t shardBytes = bytes / kLocalRanks;
    int pieces =
        (cfg.reduceScatter && useRsSerial(bytes)) ? 1
        : (cfg.rsRing && (!cfg.rsAuto || useRsDirectRing(bytes)))
            ? std::min(
                  cfg.rsTargetPieceBytes
                      ? int(std::max<size_t>(
                            2, (bytes / kWorld +
                                cfg.rsTargetPieceBytes - 1) /
                                   cfg.rsTargetPieceBytes))
                      : cfg.rsPieces,
                  int(std::max<size_t>(1, count / kWorld / 32)))
        : (cfg.allGather || cfg.reduceScatter) ? 1
        : bytes >= cfg.pipelineMinBytes
            ? (int)std::max<size_t>(
                  2, std::min<size_t>(32, (shardBytes + cfg.pieceBytes - 1) /
                                              cfg.pieceBytes))
            : 1;
    if (hybridRs) {
      pieces =
          useRsSerial(bytes) || !useRsDirectRing(bytes)
              ? 1
              : std::min(int(std::max<size_t>(
                             2, (bytes / kWorld + cfg.rsTargetPieceBytes - 1) /
                                    cfg.rsTargetPieceBytes)),
                         int(std::max<size_t>(1, count / kWorld / 32)));
    }
    if (rank == 0) {
      char const* selected =
          cfg.hostGather ? (hybridRs ? (bytes <= kHybridSerialArMaxBytes
                                          ? "serial-local" : "gpu-rs-host-ag")
                                      : "host-ag-eight-rank")
          : cfg.allGather
              ? (cfg.agPipeline == 5   ? "matched-sm"
                 : cfg.agPipeline == 4 ? "matched-ce"
                 : cfg.agDualRdma
                     ? (cfg.agPipeline == 1 ? "dual-rdma-sm" : "dual-rdma-ce")
                 : cfg.agPipeline == 3 ? "pipeline-ce-pair"
                 : cfg.agPipeline == 2 ? "pipeline-ce"
                 : cfg.agPipeline == 1 ? "pipeline-sm"
                 : useAgSerial(bytes)  ? "serial-local"
                 : useAgRing(bytes)    ? "ring"
                                       : "multicast")
          : cfg.reduceScatter
              ? (cfg.rsOverlapLocal       ? "overlap-local"
                 : cfg.rsPullLocal        ? "pull-local"
                 : useRsSerial(bytes)     ? "serial-local"
                 : useRsDirectRing(bytes) ? "direct-ring"
                 : autoPackedRs(bytes)    ? "packed-ring"
                                          : "flat")
              : (useArSerial(bytes)       ? "serial-local"
                 : useArFlatGather(bytes) ? "flat-local-gather"
                                          : "ring-local-gather");
      std::printf("# selected,%s,%zu,%s\n",
                  cfg.allGather ? "allgather"
                  : cfg.reduceScatter ? "reducescatter" : "allreduce",
                  bytes, selected);
      if (cfg.agPipeline)
        std::printf(
            "# "
            "ag_pipeline,piece_bytes=%zu,pieces=%zu,depth=%d,blocks=%d,threads="
            "%d,slots=2,dual=%d,first_mask=%d\n",
            cfg.agPieceBytes,
            (bytes / kWorld + cfg.agPieceBytes - 1) / cfg.agPieceBytes,
            cfg.agDepth, cfg.blocks, cfg.threads, int(cfg.agDualRdma),
            cfg.agFirstMask);
    }
    double relerr = -1.0;
    if (cfg.hostRepeatCheck) {
      fill(count, 1);
      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
      for (int repeat = 0; repeat < cfg.hostRepeatCheck; ++repeat)
        launch(count, bytes);
      CUDACHECK(cudaStreamSynchronize(stream));
      hostOut.resize(count);
      CUDACHECK(cudaMemcpy(hostOut.data(), buffer + layout.data, bytes,
                           cudaMemcpyDeviceToHost));
      double factor = (double(kWorld) * (kWorld + 1) / 2 + kWorld * kWorld) *
                      std::pow(double(kWorld), cfg.hostRepeatCheck - 1);
      double error = 0;
      for (size_t i = 0; i < count; ++i) {
        double want = factor * (1 + i % 97);
        double e = std::fabs(double(hostOut[i]) - want) / want;
        error = std::max(error, std::isfinite(e) ? e : INFINITY);
      }
      double worst = 0;
      MPICHECK(MPI_Allreduce(&error, &worst, 1, MPI_DOUBLE, MPI_MAX,
                             MPI_COMM_WORLD));
      if (rank == 0)
        std::printf("# host_repeat_check,%zu,%d,%.3e\n", bytes,
                    cfg.hostRepeatCheck, worst);
      if (worst > 1e-5) MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (cfg.check) {
      int generations = (cfg.allGather || cfg.reduceScatter) ? 2 : 1;
      for (int generation = 0; generation < generations; ++generation) {
        fill(count, generation);
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        launch(count, bytes);
        CUDACHECK(cudaStreamSynchronize(stream));
        relerr = std::max(relerr, verify(count, generation));
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
      }
    }

    double us = 0;
    if (cfg.checkEach) {
      double timedSum = 0;
      for (int i = 0; i < cfg.warmup + cfg.iters; ++i) {
        int generation = 10 + i;
        fill(count, generation);
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        if (cfg.gpuTiming) CUDACHECK(cudaEventRecord(timingStart, stream));
        auto begin = std::chrono::steady_clock::now();
        launch(count, bytes);
        if (cfg.gpuTiming) CUDACHECK(cudaEventRecord(timingEnd, stream));
        CUDACHECK(cudaStreamSynchronize(stream));
        auto end = std::chrono::steady_clock::now();
        double localUs =
            std::chrono::duration<double, std::micro>(end - begin).count();
        if (cfg.gpuTiming) {
          float ms = 0;
          CUDACHECK(cudaEventElapsedTime(&ms, timingStart, timingEnd));
          localUs = double(ms) * 1000;
        }
        double localErr = cfg.check ? verify(count, generation) : -1;
        relerr = std::max(relerr, localErr);
        double callUs = 0, callErr = 0;
        MPICHECK(MPI_Reduce(&localUs, &callUs, 1, MPI_DOUBLE, MPI_MAX, 0,
                            MPI_COMM_WORLD));
        MPICHECK(MPI_Reduce(&localErr, &callErr, 1, MPI_DOUBLE, MPI_MAX, 0,
                            MPI_COMM_WORLD));
        if (rank == 0 && i >= cfg.warmup) {
          timedSum += callUs;
          std::printf("# sample,%s,%zu,%d,%.3f,%.3e\n",
                      cfg.allGather ? "allgather" : cfg.reduceScatter
                                                  ? "reducescatter"
                                                  : "allreduce",
                      bytes, i - cfg.warmup + 1, callUs, callErr);
        }
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
      }
      if (rank == 0) us = timedSum / cfg.iters;
    } else {
      fill(count, (cfg.allGather || cfg.reduceScatter) ? 1 : 0);
      for (int i = 0; i < cfg.warmup; i++) launch(count, bytes);
      CUDACHECK(cudaStreamSynchronize(stream));
      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
      if (cfg.gpuTiming) CUDACHECK(cudaEventRecord(timingStart, stream));
      auto begin = std::chrono::steady_clock::now();
      for (int i = 0; i < cfg.iters; i++) launch(count, bytes);
      if (cfg.gpuTiming) CUDACHECK(cudaEventRecord(timingEnd, stream));
      CUDACHECK(cudaStreamSynchronize(stream));
      auto end = std::chrono::steady_clock::now();
      if (cfg.check && (cfg.allGather || cfg.reduceScatter)) {
        relerr = std::max(relerr, verify(count, 1));
        fill(count, 2);
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        launch(count, bytes);
        CUDACHECK(cudaStreamSynchronize(stream));
        relerr = std::max(relerr, verify(count, 2));
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
      }
      us = std::chrono::duration<double, std::micro>(end - begin).count() /
           cfg.iters;
      if (cfg.gpuTiming) {
        float ms = 0;
        CUDACHECK(cudaEventElapsedTime(&ms, timingStart, timingEnd));
        us = double(ms) * 1000 / cfg.iters;
      }
    }
    double worstUs = 0.0, worstErr = 0.0;
    MPICHECK(
        MPI_Reduce(&us, &worstUs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce(&relerr, &worstErr, 1, MPI_DOUBLE, MPI_MAX, 0,
                        MPI_COMM_WORLD));
    int failed = rank == 0 && cfg.check && worstErr > 1e-5;
    MPICHECK(MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD));
    if (failed) MPI_Abort(MPI_COMM_WORLD, 1);
    if (rank == 0) {
      double algbw = (double)bytes / (worstUs * 1e-6) / 1e9;
      double busbw = algbw *
                     ((cfg.allGather || cfg.reduceScatter) ? 1.0 : 2.0) *
                     (kWorld - 1) / kWorld;
      std::printf("  %12zu %12zu %8d %10.2f %10.2f %10.2f %10.1e\n", bytes,
                  count, pieces, worstUs, algbw, busbw, worstErr);
      std::fflush(stdout);
    }
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (bytes > cfg.maxBytes / 2) break;
  }

  CUDACHECK(cudaStreamSynchronize(stream));
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  hostGather.reset();
  ibProxy->stopProxy();
  localProxy->stopProxy();
  if (timingEnd) CUDACHECK(cudaEventDestroy(timingEnd));
  if (timingStart) CUDACHECK(cudaEventDestroy(timingStart));
  CUDACHECK(cudaStreamDestroy(stream));
  CUDACHECK(cudaFree(buffer));
  MPI_Comm_free(&localComm);
  MPI_Finalize();
  return 0;
}
