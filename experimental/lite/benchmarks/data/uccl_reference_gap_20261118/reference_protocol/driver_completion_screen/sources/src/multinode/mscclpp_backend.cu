#include "algorithm_transport.h"
#ifdef UCCL_LITE_HOST_GATHER_EXPERIMENT
#include "async_host_gather.cuh"
#endif
#include "uccl_lite/collectives.h"
#if defined(UCCL_LITE_GRAPH_BINDING_CACHE) ||  \
    defined(UCCL_LITE_SYNC_PRIMARY_CONTEXT) || \
    defined(UCCL_LITE_DRIVER_COMPLETION)
#include <cuda.h>
#endif
#include <linux/mempolicy.h>
#include <mscclpp/algorithm.hpp>
#include <algorithm>
#include <array>
#include <cstdio>
#ifdef UCCL_LITE_COMPLETION_TRACE
#include <chrono>
#ifndef UCCL_LITE_ORDERED_SYNC_DRAIN
#error "Completion tracing expects the ordered-drain implementation"
#endif
#endif
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace uccl_lite {
namespace {
void checked(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
using namespace multinode_kernels;
#ifdef UCCL_LITE_RS_MEDIUM_BYTES
constexpr size_t kRsMediumCapacity = UCCL_LITE_RS_MEDIUM_BYTES;
static_assert(kRsMediumCapacity == (16UL << 20) ||
              kRsMediumCapacity == (32UL << 20) ||
              kRsMediumCapacity == (64UL << 20));
#elif defined(UCCL_LITE_RS_MEDIUM_16M)
constexpr size_t kRsMediumCapacity = 16UL << 20;
#else
constexpr size_t kRsMediumCapacity = 8UL << 20;
#endif

__device__ mscclpp::LL16Packet* localPackets(char* base, Layout layout) {
  return reinterpret_cast<mscclpp::LL16Packet*>(base + layout.immutableInput +
                                                2 * kPacketSlotBytes);
}
__device__ void localPacketGather(LocalDirectChans local, PortChan ib, int rank,
                                  char* buffer, Layout layout,
                                  uint32_t const* input, uint32_t* output,
                                  size_t shardElements, uint32_t flag,
                                  bool drain) {
  int me = rank % 4;
  size_t quads = shardElements / 4, packets = shardElements / 2;
  auto* own = reinterpret_cast<uint4*>(buffer + layout.ibSend);
  for (size_t i = threadIdx.x; i < quads; i += blockDim.x)
    own[i] = loadClientQuad(input, i);
  __syncthreads();
#if defined(UCCL_LITE_EARLY_PACKET_FLUSH) && \
    defined(UCCL_LITE_AG_COALESCED_FLUSH)
  uint64_t flushTicket = packetSend(ib, buffer, layout, layout.ibSend,
                                    shardElements * 4, flag, drain);
#else
  packetSend(ib, buffer, layout, layout.ibSend, shardElements * 4, flag);
#ifdef UCCL_LITE_EARLY_PACKET_FLUSH
  uint64_t flushTicket = 0;
  if (!threadIdx.x && drain) flushTicket = beginPacketFlush(ib);
#endif
#endif
  for (int phase = 0; phase < 2; ++phase) {
    if (phase)
      packetReceive(buffer, layout, layout.ibScratch, shardElements * 4, flag);
    auto const* source = reinterpret_cast<uint4 const*>(
        buffer + (phase ? layout.ibScratch : layout.ibSend));
    for (int peer = 0; peer < 4; ++peer)
      if (peer != me) {
        auto* target =
            localPackets(reinterpret_cast<char*>(local.c[peer].dst_), layout) +
            (phase * 4 + me) * packets;
        for (size_t i = threadIdx.x; i < quads; i += blockDim.x) {
          uint4 v = source[i];
          target[2 * i].write(v.x, v.y, flag);
          target[2 * i + 1].write(v.z, v.w, flag);
        }
      }
    __syncthreads();
  }
  auto const* received = localPackets(buffer, layout);
  for (size_t i = threadIdx.x; i < quads * 8; i += blockDim.x) {
    int owner = int(i / quads), peer = owner % 4;
    int phase = owner / 4 == rank / 4 ? 0 : 1;
    size_t index = i % quads;
    uint4 value;
    if (peer == me)
      value = reinterpret_cast<uint4 const*>(
          buffer + (phase ? layout.ibScratch : layout.ibSend))[index];
    else {
      auto* source = received + (phase * 4 + peer) * packets;
      uint2 x = source[2 * index].read(flag),
            y = source[2 * index + 1].read(flag);
      value = make_uint4(x.x, x.y, y.x, y.y);
    }
    storeClientQuad(output, i, value);
  }
  __syncthreads();
  if (threadIdx.x == 0 && drain) {
#ifdef UCCL_LITE_EARLY_PACKET_FLUSH
    endPacketFlush(ib, flushTicket);
#else
    ib.flush();
#endif
  }
}
__device__ void localPacketScatter(LocalDirectChans local, PortChan ib,
                                   int rank, char* buffer, Layout layout,
                                   uint32_t const* input, uint32_t* output,
                                   size_t shardElements, uint32_t flag,
                                   bool drain) {
  int me = rank % 4;
  size_t quads = shardElements / 4, packets = shardElements / 2;
  for (int peer = 0; peer < 4; ++peer)
    if (peer != me)
      for (int half = 0; half < 2; ++half) {
        auto* target =
            localPackets(reinterpret_cast<char*>(local.c[peer].dst_), layout) +
            (me * 2 + half) * packets;
        for (size_t i = threadIdx.x; i < quads; i += blockDim.x) {
          uint4 v = loadClientQuad(input, size_t(peer + half * 4) * quads + i);
          target[2 * i].write(v.x, v.y, flag);
          target[2 * i + 1].write(v.z, v.w, flag);
        }
      }
  auto const* received = localPackets(buffer, layout);
  auto* partial = reinterpret_cast<float4*>(buffer + layout.ibSend);
  for (int half = 0; half < 2; ++half)
    for (size_t i = threadIdx.x; i < quads; i += blockDim.x) {
      float4 value =
          loadClientFloatQuad(input, size_t(me + half * 4) * quads + i);
      for (int peer = 0; peer < 4; ++peer)
        if (peer != me) {
          auto* source = received + (peer * 2 + half) * packets;
          uint2 x = source[2 * i].read(flag), y = source[2 * i + 1].read(flag);
          value.x += __uint_as_float(x.x);
          value.y += __uint_as_float(x.y);
          value.z += __uint_as_float(y.x);
          value.w += __uint_as_float(y.y);
        }
      partial[size_t(half) * quads + i] = value;
    }
  __syncthreads();
  int ownHalf = rank / 4;
  packetSend(ib, buffer, layout,
             layout.ibSend + size_t(ownHalf ^ 1) * shardElements * 4,
             shardElements * 4, flag);
  packetReceive(buffer, layout, layout.ibScratch, shardElements * 4, flag);
  for (size_t i = threadIdx.x; i < quads; i += blockDim.x) {
    float4 value = partial[size_t(ownHalf) * quads + i];
    float4 peer = reinterpret_cast<float4 const*>(buffer + layout.ibScratch)[i];
    value.x += peer.x;
    value.y += peer.y;
    value.z += peer.z;
    value.w += peer.w;
    storeClientQuad(output, i, floatQuadBits(value));
  }
  __syncthreads();
  if (threadIdx.x == 0 && drain) ib.flush();
}

__device__ void localPacketReduce(LocalDirectChans local, PortChan ib, int rank,
                                  char* buffer, Layout layout,
                                  uint32_t const* input, uint32_t* output,
                                  size_t elements, uint32_t flag, bool drain) {
  int me = rank % 4;
  size_t chunk = elements / 4, quads = chunk / 4, packets = chunk / 2;
  for (int peer = 0; peer < 4; ++peer)
    if (peer != me) {
      auto* target =
          localPackets(reinterpret_cast<char*>(local.c[peer].dst_), layout) +
          me * packets;
      for (size_t i = threadIdx.x; i < quads; i += blockDim.x) {
        uint4 v = loadClientQuad(input, size_t(peer) * quads + i);
        target[2 * i].write(v.x, v.y, flag);
        target[2 * i + 1].write(v.z, v.w, flag);
      }
    }
  auto const* received = localPackets(buffer, layout);
  auto* partial = reinterpret_cast<float4*>(buffer + layout.ibSend);
  for (size_t i = threadIdx.x; i < quads; i += blockDim.x) {
    float4 value = loadClientFloatQuad(input, size_t(me) * quads + i);
    for (int peer = 0; peer < 4; ++peer)
      if (peer != me) {
        uint2 x = received[peer * packets + 2 * i].read(flag);
        uint2 y = received[peer * packets + 2 * i + 1].read(flag);
        value.x += __uint_as_float(x.x);
        value.y += __uint_as_float(x.y);
        value.z += __uint_as_float(y.x);
        value.w += __uint_as_float(y.y);
      }
    partial[i] = value;
  }
  __syncthreads();
  packetSend(ib, buffer, layout, layout.ibSend, chunk * 4, flag);
  packetReceive(buffer, layout, layout.ibScratch, chunk * 4, flag);
  auto* result = reinterpret_cast<float4*>(buffer + layout.data);
  for (size_t i = threadIdx.x; i < quads; i += blockDim.x) {
    float4 value = partial[i],
           other =
               reinterpret_cast<float4 const*>(buffer + layout.ibScratch)[i];
    value.x += other.x;
    value.y += other.y;
    value.z += other.z;
    value.w += other.w;
    result[i] = value;
    uint4 bits = floatQuadBits(value);
    for (int peer = 0; peer < 4; ++peer)
      if (peer != me) {
        auto* target =
            localPackets(reinterpret_cast<char*>(local.c[peer].dst_), layout) +
            (4 + me) * packets;
        target[2 * i].write(bits.x, bits.y, flag);
        target[2 * i + 1].write(bits.z, bits.w, flag);
      }
  }
  __syncthreads();
  for (size_t i = threadIdx.x; i < elements / 4; i += blockDim.x) {
    int peer = int(i / quads);
    size_t index = i % quads;
    uint4 value;
    if (peer == me)
      value = reinterpret_cast<uint4*>(result)[index];
    else {
      auto const* source = received + (4 + peer) * packets;
      uint2 x = source[2 * index].read(flag),
            y = source[2 * index + 1].read(flag);
      value = make_uint4(x.x, x.y, y.x, y.y);
    }
    storeClientQuad(output, i, value);
  }
  __syncthreads();
  if (threadIdx.x == 0 && drain) ib.flush();
}

// Client IO is fused with the retained GPU algorithm. Alternate scratch banks
// protect readers across successive calls without a separate entry handshake.
template <bool Packet, int Operation = 0>
__global__ void stagedCollective(LocalDirectChans local, PortChan ib, int rank,
                                 char* buffer, Layout layout,
                                 NetworkOperation requestedOp,
                                 uint32_t const* input, uint32_t* output,
                                 size_t inElements, size_t outElements,
                                 uint32_t packetFlag = 0,
                                 bool packetDrain = true
#ifdef UCCL_LITE_PACKET_TIMELINE
                                 ,
                                 uint64_t* serialTimeline = nullptr
#endif
) {
  NetworkOperation op =
      Operation ? static_cast<NetworkOperation>(Operation) : requestedOp;
  int localRank = rank % 4;
  // Two banks keep a faster peer's next call away from current readers.
  // Reaching call i+2 requires the local and IB handshakes of call i+1,
  // which cannot occur until every participating rank has completed call i.
  bool ag = op == NetworkOperation::AllGather;
  bool rs = op == NetworkOperation::ReduceScatter;
  size_t elements = inElements > outElements ? inElements : outElements;
  if constexpr (Packet) {
#ifdef UCCL_LITE_AG_LOCAL_PACKET_MAX
    constexpr size_t localPacketMax = UCCL_LITE_AG_LOCAL_PACKET_MAX;
    static_assert(localPacketMax >= (16UL << 10) &&
                  localPacketMax <= (128UL << 10));
    // LL16 doubles the payload; all eight shards share local packet storage.
    static_assert(2 * localPacketMax <= kLocalPacketBytes);
#else
    constexpr size_t localPacketMax = 16UL << 10;
#endif
    if (ag && elements * 4 <= localPacketMax) {
      localPacketGather(local, ib, rank, buffer, layout, input, output,
                        inElements, packetFlag, packetDrain);
      return;
    }
    if (rs) {
      localPacketScatter(local, ib, rank, buffer, layout, input, output,
                         outElements, packetFlag, packetDrain);
      return;
    }
    if (op == NetworkOperation::AllReduce && elements * 4 <= 8192) {
      localPacketReduce(local, ib, rank, buffer, layout, input, output,
                        inElements, packetFlag, packetDrain);
      return;
    }
  }
  if (ag)
    hierarchicalAllGatherSerialBody<true, Packet>(
        local, ib, reinterpret_cast<float*>(buffer), elements, layout, rank,
        localRank, input, output, packetFlag, packetDrain
#ifdef UCCL_LITE_PACKET_TIMELINE
        ,
        serialTimeline
#endif
    );
  else if (rs)
    hierarchicalReduceScatterSerialBody<true, Packet>(
        local, ib, reinterpret_cast<float*>(buffer), elements, layout, rank,
        localRank, input, output, packetFlag, packetDrain);
  else {
    constexpr int position[4] = {0, 1, 3, 2};
    hierarchicalAllReduceSerialBody<true, Packet>(
        local, ib, reinterpret_cast<float*>(buffer), elements, layout,
        localRank, position[localRank], input, output, packetFlag, packetDrain);
  }
}

// Stream both destination halves through the local ring in standard caller
// layout. This removes the packed tensor and intermediate local stores.
template <bool RemoteFirst = false>
__global__ void __launch_bounds__(512)
    mediumScatterStreamed(MemChan send, MemChan recv, PortChan ib, char* buffer,
                          Layout layout, int rank, int pos, size_t shard,
                          uint32_t const* input, uint32_t* output) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nth = gridDim.x * blockDim.x;
  constexpr int owner[4] = {2, 0, 1, 3};
  size_t chunk = 2 * shard;
  auto* peerScratch = reinterpret_cast<float*>(
      reinterpret_cast<char*>(send.dst_) + layout.localScratch);
  auto* scratch = reinterpret_cast<float*>(buffer + layout.localScratch);
  auto* partial = reinterpret_cast<float*>(buffer + layout.ibSend);
  auto const* remote =
      reinterpret_cast<float const*>(buffer + layout.ibScratch);
  int ownHalf = rank / 4;
  if constexpr (RemoteFirst) {
    // Reduce and send the opposite node's half first. Its IB transfer can
    // overlap the own-half ring; the two scratch/partial regions are disjoint.
    for (int phase = 0; phase < 2; ++phase) {
      int half = phase ? ownHalf : (ownHalf ^ 1);
      size_t inputOffset = size_t(owner[pos] + half * 4) * shard;
      auto* target = reinterpret_cast<uint32_t*>(peerScratch + half * shard);
      for (size_t i = tid; i < shard / 4; i += nth)
        reinterpret_cast<uint4*>(target)[i] =
            loadClientQuad(input + inputOffset, i);
      for (size_t i = shard / 4 * 4 + tid; i < shard; i += nth)
        target[i] = input[inputOffset + i];
      publish();
      if (tid == 0) {
        send.signal();
        recv.wait();
      }
      publish();
      for (int step = 0; step < 3; ++step) {
        int receiveChunk = (pos - step - 1 + 4) % 4;
        float* destination =
            step == 2 ? partial + half * shard
                      : peerScratch + size_t(step + 1) * chunk + half * shard;
        addLocalOut(ClientRingChan{}, destination,
                    reinterpret_cast<float const*>(input) +
                        size_t(owner[receiveChunk] + half * 4) * shard,
                    scratch + size_t(step) * chunk + half * shard, shard, tid,
                    nth);
        publish();
        if (step != 2) {
          if (tid == 0) {
            send.signal();
            recv.wait();
          }
          publish();
        }
      }
      if (phase == 0 && tid == 0)
        ib.putWithSignal(layout.ibScratch,
                         layout.ibSend + size_t(half) * shard * 4, shard * 4);
    }
  } else {
    for (int half = 0; half < 2; ++half) {
      size_t inputOffset = size_t(owner[pos] + half * 4) * shard;
      auto* target = reinterpret_cast<uint32_t*>(peerScratch + half * shard);
      for (size_t i = tid; i < shard / 4; i += nth)
        reinterpret_cast<uint4*>(target)[i] =
            loadClientQuad(input + inputOffset, i);
      for (size_t i = shard / 4 * 4 + tid; i < shard; i += nth)
        target[i] = input[inputOffset + i];
    }
    publish();
    if (tid == 0) {
      send.signal();
      recv.wait();
    }
    publish();
    for (int step = 0; step < 3; ++step) {
      int receiveChunk = (pos - step - 1 + 4) % 4;
      for (int half = 0; half < 2; ++half) {
        float* destination =
            step == 2 ? partial + half * shard
                      : peerScratch + size_t(step + 1) * chunk + half * shard;
        addLocalOut(ClientRingChan{}, destination,
                    reinterpret_cast<float const*>(input) +
                        size_t(owner[receiveChunk] + half * 4) * shard,
                    scratch + size_t(step) * chunk + half * shard, shard, tid,
                    nth);
      }
      publish();
      if (step != 2) {
        if (tid == 0) {
          send.signal();
          recv.wait();
        }
        publish();
      }
    }
  }
  if (tid == 0) {
    if constexpr (!RemoteFirst)
      ib.putWithSignal(layout.ibScratch,
                       layout.ibSend + size_t(ownHalf ^ 1) * shard * 4,
                       shard * 4);
    ib.wait();
  }
  publish();
  auto const* own = partial + size_t(ownHalf) * shard;
  for (size_t i = tid; i < shard / 4; i += nth) {
    float4 value = reinterpret_cast<float4 const*>(own)[i];
    float4 other = reinterpret_cast<float4 const*>(remote)[i];
    value.x += other.x;
    value.y += other.y;
    value.z += other.z;
    value.w += other.w;
    storeClientQuad(output, i, floatQuadBits(value));
  }
  for (size_t i = shard / 4 * 4 + tid; i < shard; i += nth)
    output[i] = __float_as_uint(own[i] + remote[i]);
  if (tid == 0) ib.flush();
}

__global__ void drainIb(PortChan ib) {
  if (threadIdx.x == 0) ib.flush();
}

// A large call reuses one registered bank. It must wait for the preceding
// collective's readers and output copies on every rank before overwriting it.
__device__ __forceinline__ void reuseLargeBody(LocalDirectChans local,
                                               PortChan ib, int rank) {
  if (threadIdx.x == 0) {
    for (int p = 0; p < 4; ++p)
      if (p != rank) local.c[p].signal();
    for (int p = 0; p < 4; ++p)
      if (p != rank) local.c[p].wait();
    ib.signal();
    ib.wait();
    ib.flush();
  }
}

__global__ void reuseLarge(LocalDirectChans local, PortChan ib, int rank) {
  reuseLargeBody(local, ib, rank);
}

// Keep client staging on the caller's CUDA stream without CUDA's internal
// peer-copy scheduling. The buffers can have different per-rank alignment.
__global__ void copyClient(uint32_t* output, uint32_t const* input,
                           size_t count) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  if (((reinterpret_cast<uintptr_t>(input) |
        reinterpret_cast<uintptr_t>(output)) &
       15) == 0) {
    for (size_t i = tid; i < count / 4; i += stride)
      reinterpret_cast<uint4*>(output)[i] =
          reinterpret_cast<uint4 const*>(input)[i];
    for (size_t i = count / 4 * 4 + tid; i < count; i += stride)
      output[i] = input[i];
  } else {
    for (size_t i = tid; i < count; i += stride) output[i] = input[i];
  }
}

__global__ void graphGatherStart(PortChan ib, char* buffer, Layout layout,
                                 int rank, size_t shard) {
  auto const* src = reinterpret_cast<uint4 const*>(buffer + layout.ibSend);
  auto* dst = reinterpret_cast<uint4*>(buffer + layout.data + rank * shard);
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = tid; i < shard / sizeof(uint4); i += gridDim.x * blockDim.x)
    dst[i] = src[i];
  // ibSend was populated by the preceding graph node, not by this kernel.
  if (tid == 0)
    ib.putWithSignal(layout.data + rank * shard, layout.ibSend, shard);
}
__global__ void graphPeerReady(MemChan send, MemChan recv) {
  if (threadIdx.x == 0) {
    send.signal();
    recv.wait();
  }
}
__global__ void graphIbReady(PortChan ib, bool flush = true) {
  if (threadIdx.x == 0) {
    ib.wait();
    if (flush) ib.flush();
  }
}
__global__ void graphIbSend(PortChan ib, size_t dst, size_t src, size_t bytes) {
  if (threadIdx.x == 0) ib.putWithSignal(dst, src, bytes);
}
__global__ void graphAdd(float* dst, float const* a, float const* b,
                         size_t count) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  addOut(dst, a, b, count, tid, gridDim.x * blockDim.x);
}

struct GraphInput {
  float const* pointer;
  float* output;
};
__global__ void setGraphInput(GraphInput* slot, void const* pointer,
                              void* output) {
  if (threadIdx.x == 0) {
    slot->pointer = reinterpret_cast<float const*>(pointer);
    slot->output = reinterpret_cast<float*>(output);
  }
}
#ifdef UCCL_LITE_RS_GRAPH_SM_INPUT
__global__ void graphCopyClient(float* destination, GraphInput const* slot,
                                size_t offset, size_t count) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  auto const* source =
      reinterpret_cast<uint32_t const*>(slot->pointer + offset);
  auto* output = reinterpret_cast<uint32_t*>(destination);
  for (size_t i = tid; i < count / 4; i += stride)
    storeClientQuad(output, i, loadClientQuad(source, i));
  for (size_t i = count / 4 * 4 + tid; i < count; i += stride)
    output[i] = source[i];
}
#endif
template <bool ClientOutput = false, bool AddRemote = false>
__device__ __forceinline__ void graphAddClientBody(
    float* dst, GraphInput const* slot, size_t offset, float const* peer,
    size_t count, float const* remote = nullptr) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  auto const* own = slot->pointer + offset;
  for (size_t i = tid; i < count / 4; i += stride) {
    float4 x = loadClientFloatQuad(reinterpret_cast<uint32_t const*>(own), i);
    float4 y = reinterpret_cast<float4 const*>(peer)[i];
    x.x += y.x;
    x.y += y.y;
    x.z += y.z;
    x.w += y.w;
    if constexpr (AddRemote) {
      float4 z = reinterpret_cast<float4 const*>(remote)[i];
      x.x += z.x;
      x.y += z.y;
      x.z += z.z;
      x.w += z.w;
    }
    if constexpr (ClientOutput)
      storeClientQuad(reinterpret_cast<uint32_t*>(dst), i, floatQuadBits(x));
    else
      reinterpret_cast<float4*>(dst)[i] = x;
  }
  for (size_t i = count / 4 * 4 + tid; i < count; i += stride)
    if constexpr (AddRemote)
      dst[i] = (own[i] + peer[i]) + remote[i];
    else
      dst[i] = own[i] + peer[i];
}
__global__ void graphAddClient(float* dst, GraphInput const* slot,
                               size_t offset, float const* peer, size_t count) {
  graphAddClientBody(dst, slot, offset, peer, count);
}
template <bool ClientLocal = false>
__device__ __forceinline__ void graphScatterFinishBody(GraphInput const* slot,
                                                       size_t offset,
                                                       float const* local,
                                                       float const* remote,
                                                       size_t count) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  auto* dst = reinterpret_cast<uint32_t*>(slot->output + offset);
  for (size_t i = tid; i < count / 4; i += stride) {
    float4 x;
    if constexpr (ClientLocal)
      x = loadClientFloatQuad(reinterpret_cast<uint32_t const*>(local), i);
    else
      x = reinterpret_cast<float4 const*>(local)[i];
    float4 y = reinterpret_cast<float4 const*>(remote)[i];
    x.x += y.x;
    x.y += y.y;
    x.z += y.z;
    x.w += y.w;
    storeClientQuad(dst, i, floatQuadBits(x));
  }
  for (size_t i = count / 4 * 4 + tid; i < count; i += stride)
    slot->output[offset + i] = local[i] + remote[i];
}
__global__ void graphScatterFinish(GraphInput const* slot, size_t offset,
                                   float const* local, float const* remote,
                                   size_t count) {
  graphScatterFinishBody(slot, offset, local, remote, count);
}
#ifdef UCCL_LITE_RS_CLIENT_PARTIAL
__global__ void graphAddClientPartial(GraphInput const* slot,
                                      size_t inputOffset, size_t outputOffset,
                                      float const* peer, size_t count) {
  graphAddClientBody<true>(slot->output + outputOffset, slot, inputOffset, peer,
                           count);
}
__global__ void graphScatterFinishPartial(GraphInput const* slot, size_t offset,
                                          float const* remote, size_t count) {
  graphScatterFinishBody<true>(slot, offset, slot->output + offset, remote,
                               count);
}
#endif
#ifdef UCCL_LITE_RS_FUSED_FINAL
__global__ void graphScatterFinal(GraphInput const* slot, size_t inputOffset,
                                  size_t outputOffset, float const* peer,
                                  float const* remote, size_t count) {
  graphAddClientBody<true, true>(slot->output + outputOffset, slot, inputOffset,
                                 peer, count, remote);
}
#endif
struct InputNode {
  cudaGraphNode_t node;
  void* destination;
  size_t offset, bytes;
};
struct ReduceGraph {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  std::vector<InputNode> inputs;
#ifdef UCCL_LITE_GRAPH_BINDING_CACHE
  void const* boundInput = nullptr;
  unsigned long long boundAllocation = 0;
  bool bindingValid = false;
#endif
};

struct Context {
  int rank, device, rail;
  size_t capacity;
  size_t bankBytes = 0;
  uint64_t sequence = 0;
#ifndef UCCL_LITE_PACKET_INITIAL
#define UCCL_LITE_PACKET_INITIAL 0
#endif
  uint32_t packetSequence = UCCL_LITE_PACKET_INITIAL;
#ifdef UCCL_LITE_PACKET_TIMELINE
  static constexpr size_t kSerialTraceWords = 64, kSerialTraceCalls = 128;
  uint64_t* serialTimeline = nullptr;
  size_t serialTimelineBytes = 0, serialTimelineCount = 0;
#endif
  bool ibOutstanding = false;
  uint64_t deferredArCalls = 0;
  char* buffer = nullptr;
  cudaEvent_t done = nullptr;
  cudaStream_t lastStream = nullptr;
  bool pending = false;
#ifdef UCCL_LITE_COMPLETION_TRACE
  using TraceClock = std::chrono::steady_clock;
  struct CompletionSample {
    uint64_t sequence, bytes, enqueue, device, gpu, host, restore;
    int operation;
#ifdef UCCL_LITE_COMPLETION_GPU_EVENTS
    float gpuElapsedUs;
#endif
  };
#ifdef UCCL_LITE_COMPLETION_GPU_EVENTS
  cudaEvent_t completionGpuBegin = nullptr, completionGpuEnd = nullptr;
#endif
  // Value-initialization touches all trace pages before any measured call.
  std::array<CompletionSample, 4096> completionSamples{};
  size_t completionSampleCount = 0;
  uint64_t traceSequence = 0, tracedSequence = 0, traceBytes = 0,
           traceEnqueue = 0;
  int traceOperation = 0;
  static uint64_t traceNs(TraceClock::time_point begin,
                          TraceClock::time_point end) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
        .count();
  }
#endif
#ifdef UCCL_LITE_SYNC_PRIMARY_CONTEXT
  CUcontext syncPrimary = nullptr;
#endif
#ifdef UCCL_LITE_GRAPH_BINDING_CACHE
  uint64_t bindingHits = 0, bindingUpdates = 0, bindingRecycled = 0;
#endif
#ifdef UCCL_LITE_GRAPH_DESCRIPTOR_CACHE
  void const* descriptorInput = nullptr;
  void* descriptorOutput = nullptr;
  bool descriptorKnown = false, descriptorPublished = false;
  uint64_t descriptorReuses = 0, descriptorPending = 0;
#endif
  Layout layout{}, largeLayout{}, mediumLayout{};
  size_t mediumBankBytes = 0;
  uint64_t mediumSequence = 0;
#ifdef UCCL_LITE_RS128_PREFIX_BANKS
  uint64_t prefix128Sequence = 0;
  bool prefix128Active = false;
  uint64_t prefix128Entries = 0, prefix128Reuses = 0, prefix128Leaves = 0;
#endif
  size_t smallBase = 0;
  int blocks = 96;
  std::unique_ptr<GpuTransport> transport;
#ifdef UCCL_LITE_HOST_GATHER_EXPERIMENT
  std::unique_ptr<multinode::AsyncHostGather> hostGather;
#ifdef UCCL_LITE_HOST_HYBRID_EXPERIMENT
  void* hybridShard = nullptr;
#endif
#endif
  cudaStream_t graphBuilder = nullptr;
  using GraphKey = std::pair<size_t, size_t>;
  std::map<GraphKey, cudaGraphExec_t> gatherGraphs;
  std::map<GraphKey, ReduceGraph> reduceGraphs, scatterGraphs;
  GraphInput* graphInput = nullptr;
  std::shared_ptr<mscclpp::Communicator> owner;
  Context(int r, int d, int net, size_t bytes)
      : rank(r), device(d), rail(net), capacity(bytes) {}
  ~Context() {
    int previous = 0;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    if (pending) cudaEventSynchronize(done);
#ifdef UCCL_LITE_COMPLETION_TRACE
    if (auto* directory = std::getenv("UCCL_LITE_COMPLETION_TRACE_DIR")) {
      std::string path =
          std::string(directory) + "/rank_" + std::to_string(rank) + ".csv";
      if (auto* file = std::fopen(path.c_str(), "w")) {
        std::fprintf(file,
                     "sequence,operation,bytes,enqueue_ns,device_ns,gpu_ns,"
                     "host_ns,restore_ns"
#ifdef UCCL_LITE_COMPLETION_GPU_EVENTS
                     ",gpu_elapsed_us"
#endif
                     "\n");
        for (size_t i = 0; i < completionSampleCount; ++i) {
          auto const& v = completionSamples[i];
          std::fprintf(
              file, "%llu,%d,%llu,%llu,%llu,%llu,%llu,%llu",
              (unsigned long long)v.sequence, v.operation,
              (unsigned long long)v.bytes, (unsigned long long)v.enqueue,
              (unsigned long long)v.device, (unsigned long long)v.gpu,
              (unsigned long long)v.host, (unsigned long long)v.restore);
#ifdef UCCL_LITE_COMPLETION_GPU_EVENTS
          std::fprintf(file, ",%.6f", double(v.gpuElapsedUs));
#endif
          std::fputc(10, file);
        }
        std::fclose(file);
      } else {
        std::fprintf(stderr,
                     "UCCL Lite completion trace could not be written\n");
      }
    }
#endif
#ifdef UCCL_LITE_RS128_PREFIX_BANKS
    std::fprintf(
        stderr,
        "UCCL Lite prefix128: rank=%d entries=%llu reuses=%llu leaves=%llu\n",
        rank, (unsigned long long)prefix128Entries,
        (unsigned long long)prefix128Reuses,
        (unsigned long long)prefix128Leaves);
#endif
#ifdef UCCL_LITE_GRAPH_BINDING_CACHE
    std::fprintf(stderr,
                 "UCCL Lite graph bindings: rank=%d hits=%llu updates=%llu "
                 "recycled=%llu\n",
                 rank, (unsigned long long)bindingHits,
                 (unsigned long long)bindingUpdates,
                 (unsigned long long)bindingRecycled);
#endif
#ifdef UCCL_LITE_GRAPH_DESCRIPTOR_CACHE
    std::fprintf(
        stderr,
        "UCCL Lite graph descriptor: rank=%d reused=%llu pending=%llu\n", rank,
        (unsigned long long)descriptorReuses,
        (unsigned long long)descriptorPending);
#endif
#ifdef UCCL_LITE_PACKET_TIMELINE
    if (serialTimeline) {
      std::vector<uint64_t> trace(kSerialTraceWords * serialTimelineCount);
      auto* directory = std::getenv("UCCL_LITE_SERIAL_TIMELINE_DIR");
      if (directory && !trace.empty() &&
          cudaMemcpy(trace.data(), serialTimeline, trace.size() * 8,
                     cudaMemcpyDeviceToHost) == cudaSuccess) {
        std::string prefix =
            std::string(directory) + "/rank_" + std::to_string(rank);
        auto* file = std::fopen((prefix + ".bin").c_str(), "wb");
        if (file) {
          auto written = std::fwrite(trace.data(), 8, trace.size(), file);
          std::fclose(file);
          if (written != trace.size())
            std::fprintf(stderr, "UCCL Lite serial timeline short write\n");
        }
        file = std::fopen((prefix + ".json").c_str(), "w");
        if (file) {
          std::fprintf(file,
                       "{\"rank\":%d,\"bytes\":%zu,\"calls\":%zu,"
                       "\"words_per_call\":%zu}\n",
                       rank, serialTimelineBytes, serialTimelineCount,
                       kSerialTraceWords);
          std::fclose(file);
        }
      }
      cudaFree(serialTimeline);
    }
#endif
#ifdef UCCL_LITE_HOST_GATHER_EXPERIMENT
    hostGather.reset();
#ifdef UCCL_LITE_HOST_HYBRID_EXPERIMENT
    if (hybridShard) cudaFree(hybridShard);
#endif
#endif
    // GPU output completion alone need not retire all outbound sends.
    // Drain through the proxy FIFO before releasing registered storage.
    if (ibOutstanding && transport) {
      std::fprintf(stderr, "UCCL Lite IB drain: rank=%d reason=destroy\n",
                   rank);
      drainIb<<<1, 32, 0, graphBuilder>>>(transport->ib);
      cudaStreamSynchronize(graphBuilder);
    }
    for (auto const& entry : gatherGraphs) cudaGraphExecDestroy(entry.second);
    for (auto const& entry : reduceGraphs) {
      cudaGraphExecDestroy(entry.second.executable);
      cudaGraphDestroy(entry.second.graph);
    }
    for (auto const& entry : scatterGraphs) {
      cudaGraphExecDestroy(entry.second.executable);
      cudaGraphDestroy(entry.second.graph);
    }
    if (graphInput) cudaFree(graphInput);
    if (graphBuilder) cudaStreamDestroy(graphBuilder);
    transport.reset();
    if (buffer) cudaFree(buffer);
#ifdef UCCL_LITE_COMPLETION_GPU_EVENTS
    if (completionGpuBegin) cudaEventDestroy(completionGpuBegin);
    if (completionGpuEnd) cudaEventDestroy(completionGpuEnd);
#endif
    if (done) cudaEventDestroy(done);
    cudaSetDevice(previous);
  }
  void prepare(std::shared_ptr<mscclpp::Communicator> comm) {
    if (transport) return;
    if (buffer) throw std::logic_error("failed preparation cannot be retried");
    checked(cudaDeviceSynchronize());
    // MSCCL++ allocation helpers can bind the caller. Preserve its affinity.
    cpu_set_t affinity;
    if (sched_getaffinity(0, sizeof(affinity), &affinity))
      throw std::runtime_error("cannot read caller affinity");
    int memoryMode = 0;
    std::array<unsigned long, 16> memoryNodes{};
    if (syscall(SYS_get_mempolicy, &memoryMode, memoryNodes.data(),
                sizeof(memoryNodes) * 8, nullptr, 0))
      throw std::runtime_error("cannot read caller NUMA policy");
    struct Restore {
      cpu_set_t const& mask;
      int mode;
      std::array<unsigned long, 16> const& nodes;
      ~Restore() {
        int base = mode & ~MPOL_MODE_FLAGS;
        syscall(
            SYS_set_mempolicy, mode,
            base == MPOL_DEFAULT || base == MPOL_LOCAL ? nullptr : nodes.data(),
            sizeof(nodes) * 8);
        sched_setaffinity(0, sizeof(mask), &mask);
      }
    } restore{affinity, memoryMode, memoryNodes};
    size_t smallCapacity = std::min(capacity, size_t(512) << 10);
    size_t slot = smallCapacity / 4 + 4096;
    layout = {0, smallCapacity, 2 * smallCapacity, 2 * smallCapacity + slot, 0};
    layout.immutableInput = layout.ibScratch + slot;
    bankBytes = layout.immutableInput + kPacketBankBytes;
    if (capacity > smallCapacity) {
      size_t largeSlot = capacity / 4 + 4096;
      largeLayout = {0, capacity, 2 * capacity, 2 * capacity + largeSlot,
                     2 * capacity + 2 * largeSlot};
      smallBase = largeLayout.immutableInput + capacity;
      cudaDeviceProp property{};
      checked(cudaGetDeviceProperties(&property, device));
      int resident = 0;
      auto limit = [&](auto kernel) {
        checked(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel,
                                                              512, 0));
        blocks = std::min(blocks, resident * property.multiProcessorCount);
      };
      limit(hierarchicalAllReduce<DirectRingChan>);
      limit(hierarchicalAllReduce<DirectRingChan, true>);
      limit(hierarchicalAllReduce<DirectRingChan, true, true>);
      limit(mediumScatterStreamed<false>);
#ifdef UCCL_LITE_RS8M_REMOTE_FIRST
      limit(mediumScatterStreamed<true>);
#endif
      limit(hierarchicalAllGatherRing<DirectRingChan>);
      limit(hierarchicalReduceScatterRing<DirectRingChan>);
      limit(hierarchicalReduceScatterDirectRing<DirectRingChan>);
      if (blocks < 1)
        throw std::runtime_error("large collective cannot reside");
    }
    size_t bytes = smallBase + 2 * bankBytes;
    if (capacity > smallCapacity) {
      constexpr size_t page = size_t(2) << 20;
      auto pageAlign = [](size_t value) {
        return (value + page - 1) & ~(page - 1);
      };
      bytes = pageAlign(bytes);
      size_t mediumCapacity = pageAlign(std::min(capacity, kRsMediumCapacity));
      size_t mediumSlot = pageAlign(mediumCapacity / 4 + 4096);
      mediumBankBytes = 3 * mediumCapacity + 2 * mediumSlot;
      mediumLayout = {bytes, bytes + mediumCapacity, bytes + 2 * mediumCapacity,
                      bytes + 2 * mediumCapacity + mediumSlot,
                      bytes + 2 * mediumCapacity + 2 * mediumSlot};
      bytes += 2 * mediumBankBytes;
    }
    if (bytes >= (1ULL << 32))
      throw std::invalid_argument("32-bit channel offset overflow");
    checked(cudaMalloc(&buffer, bytes));
    checked(cudaMemset(buffer, 0, bytes));
    checked(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
#ifdef UCCL_LITE_COMPLETION_GPU_EVENTS
    checked(cudaEventCreate(&completionGpuBegin));
    checked(cudaEventCreate(&completionGpuEnd));
#endif
#ifdef UCCL_LITE_PACKET_TIMELINE
    if (auto* target = std::getenv("UCCL_LITE_SERIAL_TIMELINE_BYTES")) {
      serialTimelineBytes = std::stoull(target);
      if (serialTimelineBytes < (32UL << 10) ||
          serialTimelineBytes > (128UL << 10))
        throw std::invalid_argument("serial timeline requires 32–128 KiB AG");
      checked(cudaMalloc(&serialTimeline,
                         kSerialTraceWords * kSerialTraceCalls * 8));
      checked(cudaMemset(serialTimeline, 0,
                         kSerialTraceWords * kSerialTraceCalls * 8));
    }
#endif
    transport = std::make_unique<GpuTransport>(
        *comm, rank, rank % 4, rail, buffer, bytes, true, false, false, true);
    mscclpp::gpuStreamPool()->clear();
    owner = comm;
    checked(cudaMalloc(&graphInput, sizeof(GraphInput)));
    checked(cudaStreamCreateWithFlags(&graphBuilder, cudaStreamNonBlocking));
#ifdef UCCL_LITE_HOST_GATHER_EXPERIMENT
#ifndef UCCL_LITE_HOST_GATHER_CHUNK
#define UCCL_LITE_HOST_GATHER_CHUNK (256UL << 10)
#endif
    hostGather = std::make_unique<multinode::AsyncHostGather>(
        *comm, rank, UCCL_LITE_HOST_GATHER_CHUNK);
#ifdef UCCL_LITE_HOST_HYBRID_EXPERIMENT
    checked(cudaMalloc(&hybridShard, capacity / 8));
#endif
#endif
  }
  cudaGraphExec_t gatherGraph(size_t bytes, Layout largeLayout) {
    GraphKey key{bytes, largeLayout.data};
    auto found = gatherGraphs.find(key);
    if (found != gatherGraphs.end()) return found->second;
    auto& t = *transport;
    size_t shard = bytes / 8;
    constexpr int positions[4] = {0, 1, 3, 2};
    constexpr int order[4] = {0, 1, 3, 2};
    int localRank = rank % 4, pos = positions[localRank];
    MemChan send = t.localDirect.c[ringSuccessor(localRank)];
    MemChan recv = t.localDirect.c[ringPredecessor(localRank)];
    cudaGraph_t graph = nullptr;
    checked(
        cudaStreamBeginCapture(graphBuilder, cudaStreamCaptureModeThreadLocal));
    graphGatherStart<<<256, 256, 0, graphBuilder>>>(t.ib, buffer, largeLayout,
                                                    rank, shard);
    for (int phase = 0; phase < 2; ++phase) {
      if (phase) graphIbReady<<<1, 32, 0, graphBuilder>>>(t.ib);
      int nodeBase = (rank / 4 * 4) ^ (phase * 4);
      for (int step = 0; step < 3; ++step) {
        size_t offset = largeLayout.data +
                        size_t(nodeBase + order[(pos - step + 4) % 4]) * shard;
        checked(cudaMemcpyAsync(reinterpret_cast<char*>(send.dst_) + offset,
                                buffer + offset, shard,
                                cudaMemcpyDeviceToDevice, graphBuilder));
        graphPeerReady<<<1, 32, 0, graphBuilder>>>(send, recv);
      }
    }
    checked(cudaGetLastError());
    checked(cudaStreamEndCapture(graphBuilder, &graph));
    cudaGraphExec_t executable = nullptr;
    auto result = cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
    checked(result);
    gatherGraphs.emplace(key, executable);
    // Extent-only keys are identical across matching collective sequences.
    // Finish cold graph setup on all ranks before any GPU wait can time out.
    owner->bootstrap()->barrier();
    return executable;
  }
  ReduceGraph& reduceGraph(size_t bytes, Layout largeLayout) {
    GraphKey key{bytes, largeLayout.data};
    auto found = reduceGraphs.find(key);
    if (found != reduceGraphs.end()) return found->second;
    auto& t = *transport;
    constexpr int positions[4] = {0, 1, 3, 2};
    int localRank = rank % 4, pos = positions[localRank];
    MemChan send = t.localDirect.c[ringSuccessor(localRank)];
    MemChan recv = t.localDirect.c[ringPredecessor(localRank)];
    size_t count = bytes / 4;
    size_t chunk = ((count + 3) / 4 + 31) & ~size_t(31);
    size_t ownBase = size_t((pos + 1) % 4) * chunk;
    size_t ownLength = std::min(chunk, count - ownBase);
    int pieces = int(std::max<size_t>(
        2, std::min<size_t>(32, (bytes / 4 + (2UL << 20) - 1) / (2UL << 20))));
    int distance = std::min(2, pieces);
    size_t piece = ((chunk + pieces - 1) / pieces + 31) & ~size_t(31);
    auto* data = reinterpret_cast<float*>(buffer + largeLayout.data);
    auto* scratch = reinterpret_cast<float*>(buffer + largeLayout.localScratch);
    auto* partial = reinterpret_cast<float*>(buffer + largeLayout.ibSend);
    auto* remote = reinterpret_cast<float*>(buffer + largeLayout.ibScratch);
    ReduceGraph resultGraph;
    cudaGraph_t graph = nullptr;
    checked(
        cudaStreamBeginCapture(graphBuilder, cudaStreamCaptureModeThreadLocal));
    for (int p = 0; p < pieces + distance; ++p) {
      if (p < pieces) {
        size_t off = size_t(p) * piece;
        size_t length = off >= chunk ? 0 : std::min(piece, chunk - off);
        for (int step = 0; step < 3; ++step) {
          int cSend = (pos - step + 4) % 4;
          int cRecv = (cSend + 3) % 4;
          size_t sendOff = size_t(cSend) * chunk + off;
          size_t recvOff = size_t(cRecv) * chunk + off;
          size_t sendLength =
              sendOff >= count ? 0 : std::min(length, count - sendOff);
          size_t recvLength =
              recvOff >= count ? 0 : std::min(length, count - recvOff);
          size_t dst =
              largeLayout.localScratch + (size_t(step) * chunk + off) * 4;
          checked(cudaMemcpyAsync(reinterpret_cast<char*>(send.dst_) + dst,
                                  data + sendOff, sendLength * 4,
                                  cudaMemcpyDeviceToDevice, graphBuilder));
          if (step == 0) {
            cudaStreamCaptureStatus status;
            cudaGraphNode_t const* dependencies = nullptr;
            size_t dependencyCount = 0;
            checked(cudaStreamGetCaptureInfo(graphBuilder, &status, nullptr,
                                             nullptr, &dependencies,
                                             &dependencyCount));
            if (status != cudaStreamCaptureStatusActive || dependencyCount != 1)
              throw std::runtime_error(
                  "unexpected first-hop capture dependencies");
            resultGraph.inputs.push_back(
                {dependencies[0], reinterpret_cast<char*>(send.dst_) + dst,
                 sendOff * 4, sendLength * 4});
          }
          graphPeerReady<<<1, 32, 0, graphBuilder>>>(send, recv);
          graphAddClient<<<256, 256, 0, graphBuilder>>>(
              step == 2 ? partial + off : data + recvOff, graphInput, recvOff,
              scratch + size_t(step) * chunk + off, recvLength);
        }
        size_t owned = off >= ownLength ? 0 : std::min(length, ownLength - off);
        graphIbSend<<<1, 32, 0, graphBuilder>>>(
            t.ib, largeLayout.ibScratch + off * 4, largeLayout.ibSend + off * 4,
            owned * 4);
      }
      int completed = p - distance;
      if (completed >= 0 && completed < pieces) {
        size_t off = size_t(completed) * piece;
        size_t length = off >= chunk ? 0 : std::min(piece, chunk - off);
        graphIbReady<<<1, 32, 0, graphBuilder>>>(t.ib, completed == pieces - 1);
        size_t owned = off >= ownLength ? 0 : std::min(length, ownLength - off);
        graphAdd<<<256, 256, 0, graphBuilder>>>(
            data + ownBase + off, partial + off, remote + off, owned);
        for (int step = 0; step < 3; ++step) {
          int cSend = (pos + 1 - step + 4) % 4;
          size_t sendOff = size_t(cSend) * chunk + off;
          size_t sendLength =
              sendOff >= count ? 0 : std::min(length, count - sendOff);
          checked(cudaMemcpyAsync(reinterpret_cast<char*>(send.dst_) +
                                      largeLayout.data + sendOff * 4,
                                  data + sendOff, sendLength * 4,
                                  cudaMemcpyDeviceToDevice, graphBuilder));
          graphPeerReady<<<1, 32, 0, graphBuilder>>>(send, recv);
        }
      }
    }
    checked(cudaGetLastError());
    checked(cudaStreamEndCapture(graphBuilder, &graph));
    cudaGraphExec_t executable = nullptr;
    auto result = cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0);
    if (result != cudaSuccess) {
      cudaGraphDestroy(graph);
      checked(result);
    }
    resultGraph.graph = graph;
    resultGraph.executable = executable;
    auto inserted = reduceGraphs.emplace(key, std::move(resultGraph));
    owner->bootstrap()->barrier();
    return inserted.first->second;
  }
  ReduceGraph& scatterGraph(size_t bytes, Layout largeLayout) {
    GraphKey key{bytes, largeLayout.data};
    auto found = scatterGraphs.find(key);
    if (found != scatterGraphs.end()) return found->second;
    auto& t = *transport;
    constexpr int positions[4] = {0, 1, 3, 2};
    constexpr int ownerChunk[4] = {2, 0, 1, 3};
    int localRank = rank % 4, pos = positions[localRank];
    MemChan send = t.localDirect.c[ringSuccessor(localRank)];
    MemChan recv = t.localDirect.c[ringPredecessor(localRank)];
    size_t shard = bytes / 4 / 8;
#ifdef UCCL_LITE_RS_GRAPH_PIECE_BYTES
    constexpr size_t targetPieceBytes = UCCL_LITE_RS_GRAPH_PIECE_BYTES;
#else
    constexpr size_t targetPieceBytes = 8UL << 20;
#endif
    static_assert(targetPieceBytes >= (1UL << 20) &&
                  targetPieceBytes <= (32UL << 20));
#ifdef UCCL_LITE_RS_GRAPH_SINGLE_MAX_BYTES
    constexpr size_t singlePieceMaximum = UCCL_LITE_RS_GRAPH_SINGLE_MAX_BYTES;
    static_assert(singlePieceMaximum == (16UL << 20) ||
                  singlePieceMaximum == (32UL << 20) ||
                  singlePieceMaximum == (64UL << 20) ||
                  singlePieceMaximum == (128UL << 20));
#else
    constexpr size_t singlePieceMaximum = 16UL << 20;
#endif
    int pieces =
        bytes <= singlePieceMaximum
            ? 1
            : int(std::max<size_t>(
                  2, (bytes / 8 + targetPieceBytes - 1) / targetPieceBytes));
#ifdef UCCL_LITE_RS128_GRAPH_PIECES
    static_assert(UCCL_LITE_RS128_GRAPH_PIECES == 3 ||
                  UCCL_LITE_RS128_GRAPH_PIECES == 4);
    if (bytes == (128UL << 20)) pieces = UCCL_LITE_RS128_GRAPH_PIECES;
#endif
    size_t piece = ((shard + pieces - 1) / pieces + 31) & ~size_t(31);
    int ownHalf = rank / 4;
    auto* data = reinterpret_cast<float*>(buffer + largeLayout.data);
    auto* scratch = reinterpret_cast<float*>(buffer + largeLayout.localScratch);
    auto* partial = reinterpret_cast<float*>(buffer + largeLayout.ibSend);
    auto* remote = reinterpret_cast<float*>(buffer + largeLayout.ibScratch);
    ReduceGraph resultGraph;
    cudaGraph_t graph = nullptr;
    checked(
        cudaStreamBeginCapture(graphBuilder, cudaStreamCaptureModeThreadLocal));
    for (int p = 0; p < pieces; ++p) {
      size_t off = size_t(p) * piece;
      if (off >= shard) break;
      size_t length = std::min(piece, shard - off);
      for (int phase = 0; phase < 2; ++phase) {
        int half = phase == 0 ? ownHalf ^ 1 : ownHalf;
        for (int step = 0; step < 3; ++step) {
          int cSend = (pos - step + 4) % 4;
          int cRecv = (cSend + 3) % 4;
          size_t receiveOffset = size_t(half * 3 + step) * shard + off;
          size_t sourceOffset =
              step == 0 ? size_t(ownerChunk[cSend] + half * 4) * shard + off
                        : size_t(half * 3 + step - 1) * shard + off;
          size_t destination = largeLayout.localScratch + receiveOffset * 4;
#ifdef UCCL_LITE_RS_GRAPH_SM_INPUT
          if (step == 0 && bytes <= (64UL << 20)) {
            // Actual caller pointers come from the per-call GPU descriptor.
            // This input node needs no host-side memcpy parameter rebinding.
            graphCopyClient<<<256, 256, 0, graphBuilder>>>(
                reinterpret_cast<float*>(reinterpret_cast<char*>(send.dst_) +
                                         destination),
                graphInput, sourceOffset, length);
          } else
#endif
          {
            checked(cudaMemcpyAsync(
                reinterpret_cast<char*>(send.dst_) + destination,
                step == 0 ? data + sourceOffset : scratch + sourceOffset,
                length * 4, cudaMemcpyDeviceToDevice, graphBuilder));
            if (step == 0) {
              cudaStreamCaptureStatus status;
              cudaGraphNode_t const* dependencies = nullptr;
              size_t dependencyCount = 0;
              checked(cudaStreamGetCaptureInfo(graphBuilder, &status, nullptr,
                                               nullptr, &dependencies,
                                               &dependencyCount));
              if (status != cudaStreamCaptureStatusActive ||
                  dependencyCount != 1)
                throw std::runtime_error("unexpected RS capture dependencies");
              resultGraph.inputs.push_back(
                  {dependencies[0],
                   reinterpret_cast<char*>(send.dst_) + destination,
                   sourceOffset * 4, length * 4});
            }
          }
          graphPeerReady<<<1, 32, 0, graphBuilder>>>(send, recv);
#ifdef UCCL_LITE_RS_FUSED_FINAL
          if (step == 2 && phase == 1) {
            // All forwarding is complete. Wait for the remote sum, then read
            // the final own input and write the result once, preserving order.
            graphIbReady<<<1, 32, 0, graphBuilder>>>(t.ib,
                                                     off + length == shard);
            graphScatterFinal<<<256, 256, 0, graphBuilder>>>(
                graphInput, size_t(ownerChunk[cRecv] + half * 4) * shard + off,
                off, scratch + receiveOffset, remote + off, length);
          } else
#elif defined(UCCL_LITE_RS_CLIENT_PARTIAL)
          // The own-half final contribution consumes this piece's last input.
          // Subsequent pieces are disjoint, including standard in-place RS.
          if (step == 2 && phase == 1)
            graphAddClientPartial<<<256, 256, 0, graphBuilder>>>(
                graphInput, size_t(ownerChunk[cRecv] + half * 4) * shard + off,
                off, scratch + receiveOffset, length);
          else
#endif
            graphAddClient<<<256, 256, 0, graphBuilder>>>(
                step == 2 ? partial + size_t(half) * shard + off
                          : scratch + receiveOffset,
                graphInput, size_t(ownerChunk[cRecv] + half * 4) * shard + off,
                scratch + receiveOffset, length);
        }
        if (phase == 0)
          graphIbSend<<<1, 32, 0, graphBuilder>>>(
              t.ib, largeLayout.ibScratch + off * 4,
              largeLayout.ibSend + (size_t(half) * shard + off) * 4,
              length * 4);
      }
#ifndef UCCL_LITE_RS_FUSED_FINAL
      graphIbReady<<<1, 32, 0, graphBuilder>>>(t.ib, off + length == shard);
      // All input for this piece has been consumed. Future pieces are disjoint.
#ifdef UCCL_LITE_RS_CLIENT_PARTIAL
      graphScatterFinishPartial<<<256, 256, 0, graphBuilder>>>(
          graphInput, off, remote + off, length);
#else
      graphScatterFinish<<<256, 256, 0, graphBuilder>>>(
          graphInput, off, partial + size_t(ownHalf) * shard + off,
          remote + off, length);
#endif
#endif
    }
    checked(cudaGetLastError());
    checked(cudaStreamEndCapture(graphBuilder, &graph));
    cudaGraphExec_t executable = nullptr;
    auto result = cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0);
    if (result != cudaSuccess) {
      cudaGraphDestroy(graph);
      checked(result);
    }
    resultGraph.graph = graph;
    resultGraph.executable = executable;
    auto inserted = scatterGraphs.emplace(key, std::move(resultGraph));
    owner->bootstrap()->barrier();
    return inserted.first->second;
  }
  mscclpp::CommResult run(NetworkOperation op, void const* input, void* output,
                          size_t inputBytes, size_t outputBytes,
                          cudaStream_t stream, bool hostAllReducePart = false) {
    if (!transport) throw std::logic_error("prepare before enqueue");
    if (pending && stream != lastStream)
      checked(cudaStreamWaitEvent(stream, done, 0));
    auto& t = *transport;
    size_t bytes = std::max(inputBytes, outputBytes);
#ifdef UCCL_LITE_HOST_GATHER_EXPERIMENT
#ifdef UCCL_LITE_HOST_HYBRID_EXPERIMENT
#ifdef UCCL_LITE_HOST_HYBRID_MIN_BYTES
    constexpr size_t hybridMinimum = UCCL_LITE_HOST_HYBRID_MIN_BYTES;
#else
    constexpr size_t hybridMinimum = 8UL << 20;
#endif
    static_assert(hybridMinimum >= (1UL << 20) && hybridMinimum <= (8UL << 20));
    if (op == NetworkOperation::AllReduce && bytes >= hybridMinimum) {
      auto result = run(NetworkOperation::ReduceScatter, input, hybridShard,
                        inputBytes, inputBytes / 8, stream, true);
      if (result != mscclpp::CommResult::CommSuccess) return result;
#ifdef UCCL_LITE_HOST_SHARED_EVENT
      hostGather->enqueue(hybridShard, output, inputBytes / 8, stream, done);
#else
      hostGather->enqueue(hybridShard, output, inputBytes / 8, stream);
      checked(cudaEventRecord(done, stream));
#endif
      lastStream = stream;
      pending = true;
      return mscclpp::CommResult::CommSuccess;
    }
#endif
#if defined(UCCL_LITE_HOST_COOPERATIVE_SM)
#ifdef UCCL_LITE_HOST_GATHER_MIN_BYTES
    constexpr size_t hostGatherMinimum = UCCL_LITE_HOST_GATHER_MIN_BYTES;
    static_assert(hostGatherMinimum == (128UL << 10) ||
                  hostGatherMinimum == (256UL << 10));
#else
    constexpr size_t hostGatherMinimum = 256UL << 10;
#endif
#elif defined(UCCL_LITE_HOST_MEDIUM_SM)
    constexpr size_t hostGatherMinimum = 512UL << 10;
#else
    constexpr size_t hostGatherMinimum = 8UL << 20;
#endif
    if (op == NetworkOperation::AllGather && bytes >= hostGatherMinimum) {
#ifdef UCCL_LITE_HOST_SHARED_EVENT
      hostGather->enqueue(input, output, inputBytes, stream, done);
#else
      hostGather->enqueue(input, output, inputBytes, stream);
      checked(cudaEventRecord(done, stream));
#endif
      lastStream = stream;
      pending = true;
      return mscclpp::CommResult::CommSuccess;
    }
#endif
    bool directRsBoundary = false;
#ifdef UCCL_LITE_RS_STREAM_512K
    // The existing medium banks are allocated only above small capacity.
    directRsBoundary = op == NetworkOperation::ReduceScatter &&
                       bytes == (512UL << 10) && capacity > (512UL << 10);
#endif
    bool arMediumBoundary = false;
#if defined(UCCL_LITE_AR512_MEDIUM_BANKS) || \
    defined(UCCL_LITE_AR256_MEDIUM_BANKS)
#ifdef UCCL_LITE_AR256_MEDIUM_BANKS
    constexpr size_t arBankMinimum = 256UL << 10;
#else
    constexpr size_t arBankMinimum = 512UL << 10;
#endif
    arMediumBoundary = op == NetworkOperation::AllReduce &&
                       bytes >= arBankMinimum && bytes <= (512UL << 10) &&
                       capacity > (512UL << 10);
#endif
    bool medium =
        (bytes > (512UL << 10) || directRsBoundary || arMediumBoundary) &&
        bytes <= (op == NetworkOperation::ReduceScatter ? kRsMediumCapacity
                                                        : (8UL << 20)) &&
        (op != NetworkOperation::AllReduce || bytes >= (8UL << 20) ||
         arMediumBoundary);
    Layout largeLayout = this->largeLayout;
    bool prefix128 = false;
#ifdef UCCL_LITE_RS128_PREFIX_BANKS
    constexpr size_t prefixCapacity = 128UL << 20;
    constexpr size_t prefixPage = 2UL << 20;
    constexpr size_t prefixSlot =
        (prefixCapacity / 4 + 4096 + prefixPage - 1) & ~(prefixPage - 1);
    constexpr size_t prefixStride = 3 * prefixCapacity + 2 * prefixSlot;
    // Two 452-MiB banks fit inside the already allocated large data region.
    // The full region is reused by larger algorithms, so synchronize entry
    // after any ordinary large-layout operation, not on every steady call.
    prefix128 = op == NetworkOperation::ReduceScatter &&
                bytes == prefixCapacity && capacity >= 2 * prefixStride;
    if (prefix128) {
      if (!prefix128Active) {
        ++prefix128Entries;
        reuseLarge<<<1, 32, 0, stream>>>(t.localDirect, t.ib, rank % 4);
        checked(cudaGetLastError());
      } else {
        ++prefix128Reuses;
      }
      prefix128Active = true;
      size_t offset = (prefix128Sequence++ & 1) * prefixStride;
      largeLayout = {offset, offset + prefixCapacity,
                     offset + 2 * prefixCapacity,
                     offset + 2 * prefixCapacity + prefixSlot,
                     offset + 2 * prefixCapacity + 2 * prefixSlot};
      medium = true;
    }
#endif
    if (medium && !prefix128) {
      largeLayout = mediumLayout;
      if (mediumSequence++ & 1) {
        largeLayout.data += mediumBankBytes;
        largeLayout.localScratch += mediumBankBytes;
        largeLayout.ibSend += mediumBankBytes;
        largeLayout.ibScratch += mediumBankBytes;
        largeLayout.immutableInput += mediumBankBytes;
      }
      // Dedicated alternating banks: entering the following collective on
      // every rank protects the prior bank's local and network readers.
    }
#ifdef UCCL_LITE_RS_GRAPH_MIN_BYTES
    constexpr size_t configuredRsGraphMinimum = UCCL_LITE_RS_GRAPH_MIN_BYTES;
#else
    constexpr size_t configuredRsGraphMinimum = 8UL << 20;
#endif
    // Standalone RS and the private shard producer have different measured
    // schedules. Preserve the graph producer before asynchronous host gather;
    // the streamed 8-MiB public RS gain did not transfer to hybrid AllReduce.
    size_t rsGraphMinimum =
        hostAllReducePart ? (8UL << 20) : configuredRsGraphMinimum;
    if (bytes >= (op == NetworkOperation::ReduceScatter ? rsGraphMinimum
                                                        : (16UL << 20))) {
      cudaGraphExec_t executable;
      if (op == NetworkOperation::AllGather)
        executable = gatherGraph(bytes, largeLayout);
      else {
        auto& graph = op == NetworkOperation::AllReduce
                          ? reduceGraph(bytes, largeLayout)
                          : scatterGraph(bytes, largeLayout);
        // Node updates affect only future launches. Rebind every call by
        // default; the optional cache also checks allocation identity so a
        // recycled virtual address cannot retain a stale allocation binding.
#ifdef UCCL_LITE_GRAPH_BINDING_CACHE
        unsigned long long allocation = 0;
        unsigned int legacyIpc = 0;
        bool eligible = false, reuseBindings = graph.inputs.empty();
        if (!graph.inputs.empty()) {
          CUresult status = CUDA_SUCCESS;
          bool identityKnown = false;
#ifdef UCCL_LITE_GRAPH_BINDING_ID_FAST
          if (graph.bindingValid && graph.boundInput == input) {
            status = cuPointerGetAttribute(
                &allocation, CU_POINTER_ATTRIBUTE_BUFFER_ID,
                reinterpret_cast<CUdeviceptr>(input));
            if (status != CUDA_SUCCESS && status != CUDA_ERROR_INVALID_VALUE &&
                status != CUDA_ERROR_NOT_SUPPORTED) {
              char const* message = nullptr;
              cuGetErrorString(status, &message);
              throw std::runtime_error(
                  message ? message : "graph allocation query failed");
            }
            // BUFFER_ID is never reassigned within the process. A matching
            // live allocation retains the legacy eligibility already checked
            // when these bindings were installed. Pointer equality alone is
            // never sufficient; a new ID takes the complete validation below.
            identityKnown = status == CUDA_SUCCESS && allocation &&
                            allocation == graph.boundAllocation;
            if (identityKnown) legacyIpc = 1;
          }
#endif
          if (!identityKnown) {
            CUpointer_attribute attributes[] = {
                CU_POINTER_ATTRIBUTE_BUFFER_ID,
                CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE};
            void* values[] = {&allocation, &legacyIpc};
            status = cuPointerGetAttributes(
                2, attributes, values, reinterpret_cast<CUdeviceptr>(input));
          }
          if (status != CUDA_SUCCESS && status != CUDA_ERROR_INVALID_VALUE &&
              status != CUDA_ERROR_NOT_SUPPORTED) {
            char const* message = nullptr;
            cuGetErrorString(status, &message);
            throw std::runtime_error(message ? message
                                             : "graph allocation query failed");
          }
          // Legacy allocations have lifetime-unique IDs. VMM/unknown kinds
          // retain per-call rebinding; pointer equality alone is insufficient.
          eligible = status == CUDA_SUCCESS && legacyIpc && allocation;
          reuseBindings = eligible && graph.bindingValid &&
                          graph.boundInput == input &&
                          graph.boundAllocation == allocation;
          if (eligible && graph.bindingValid && graph.boundInput == input &&
              graph.boundAllocation != allocation)
            ++bindingRecycled;
          if (reuseBindings) ++bindingHits;
        }
        if (!reuseBindings) {
          graph.bindingValid = false;
#endif
          for (auto const& node : graph.inputs)
            checked(cudaGraphExecMemcpyNodeSetParams1D(
                graph.executable, node.node, node.destination,
                reinterpret_cast<char const*>(input) + node.offset, node.bytes,
                cudaMemcpyDeviceToDevice));
#ifdef UCCL_LITE_GRAPH_BINDING_CACHE
          ++bindingUpdates;
          graph.boundInput = input;
          graph.boundAllocation = allocation;
          graph.bindingValid = eligible;
        }
#endif
        executable = graph.executable;
      }
      if (!medium) {
#ifdef UCCL_LITE_RS128_PREFIX_BANKS
        if (prefix128Active) ++prefix128Leaves;
        prefix128Active = false;
#endif
        reuseLarge<<<1, 32, 0, stream>>>(t.localDirect, t.ib, rank % 4);
        checked(cudaGetLastError());
      }
      if (op == NetworkOperation::AllGather)
        checked(cudaMemcpyAsync(buffer + largeLayout.ibSend, input, inputBytes,
                                cudaMemcpyDeviceToDevice, stream));
      else {
#ifdef UCCL_LITE_GRAPH_DESCRIPTOR_CACHE
        bool reuseDescriptor = false;
        if (descriptorKnown && descriptorInput == input &&
            descriptorOutput == output) {
          if (!descriptorPublished && pending) {
            auto status = cudaEventQuery(done);
            if (status == cudaSuccess)
              descriptorPublished = true;
            else if (status == cudaErrorNotReady)
              ++descriptorPending;
            else
              checked(status);
          }
          reuseDescriptor = descriptorPublished;
        }
        if (reuseDescriptor) ++descriptorReuses;
        // This caches only two raw pointer values in our private GPU
        // descriptor, never input data or a CUDA memcpy allocation binding.
        // Memcpy nodes above are still rebound on every call, including VA
        // recycling. Reuse starts only after the descriptor's writer finished.
        if (!reuseDescriptor) {
#endif
          setGraphInput<<<1, 1, 0, stream>>>(graphInput, input, output);
          checked(cudaGetLastError());
#ifdef UCCL_LITE_GRAPH_DESCRIPTOR_CACHE
          descriptorInput = input;
          descriptorOutput = output;
          descriptorKnown = true;
          descriptorPublished = false;
        }
#endif
      }
      checked(cudaGraphLaunch(executable, stream));
      if (op != NetworkOperation::ReduceScatter)
        checked(cudaMemcpyAsync(output, buffer + largeLayout.data, outputBytes,
                                cudaMemcpyDeviceToDevice, stream));
      checked(cudaEventRecord(done, stream));
      ibOutstanding = false;
      lastStream = stream;
      pending = true;
      return mscclpp::CommResult::CommSuccess;
    }
    bool directArBoundary = false;
#if defined(UCCL_LITE_AR_DIRECT_512K) || defined(UCCL_LITE_AR_DIRECT_MIN_BYTES)
#ifdef UCCL_LITE_AR_DIRECT_MIN_BYTES
    constexpr size_t directArMinimum = UCCL_LITE_AR_DIRECT_MIN_BYTES;
#else
    constexpr size_t directArMinimum = 512UL << 10;
#endif
    static_assert(directArMinimum == (256UL << 10) ||
                  directArMinimum == (512UL << 10));
    // Keep the existing single-bank reuse/drain protocol. The large scratch
    // layout exists only when preparation exceeded the small-buffer capacity.
    directArBoundary = op == NetworkOperation::AllReduce &&
                       bytes >= directArMinimum && bytes <= (512UL << 10) &&
                       capacity > (512UL << 10);
#endif
    if (bytes > (512UL << 10) || directArBoundary || directRsBoundary ||
        arMediumBoundary) {
      if (!medium) {
#ifdef UCCL_LITE_RS128_PREFIX_BANKS
        if (prefix128Active) ++prefix128Leaves;
        prefix128Active = false;
#endif
        reuseLarge<<<1, 32, 0, stream>>>(t.localDirect, t.ib, rank % 4);
        checked(cudaGetLastError());
      }
      bool ag = op == NetworkOperation::AllGather;
      bool rs = op == NetworkOperation::ReduceScatter;
      // The boundary variant changes bank reuse only; keep its in-kernel
      // send retirement rather than introducing a separate per-call drain.
      bool deferAr =
          op == NetworkOperation::AllReduce && medium && !arMediumBoundary;
      // Two pieces emit four WRs per deferred AR call. Together with up to
      // 31 LL calls, bounded draining stays below the 1024-entry send CQ.
      // Alternating banks protect DMA readers through the next collective's
      // all-rank progress; teardown still waits for outstanding completions.
      bool retireSends = !deferAr || (++deferredArCalls % 32) == 0;
      if (ag) {
        copyClient<<<256, 256, 0, stream>>>(
            reinterpret_cast<uint32_t*>(buffer + largeLayout.ibSend),
            reinterpret_cast<uint32_t const*>(input), inputBytes / 4);
        checked(cudaGetLastError());
      }
      constexpr int positions[4] = {0, 1, 3, 2};
      int pos = positions[rank % 4];
      DirectRingChan send{t.localDirect.c[ringSuccessor(rank % 4)]};
      DirectRingChan recv{t.localDirect.c[ringPredecessor(rank % 4)]};
      size_t count = bytes / 4;
      auto* base = reinterpret_cast<float*>(buffer);
      if (ag)
        hierarchicalAllGatherRing<DirectRingChan>
            <<<std::min(blocks, 4), 512, 0, stream>>>(send, recv, t.ib, base,
                                                      count, largeLayout, rank,
                                                      rank % 4, pos);
      else if (rs) {
        int scatterBlocks = std::min(blocks, 8);
#ifdef UCCL_LITE_RS8M_BLOCKS
        static_assert(UCCL_LITE_RS8M_BLOCKS == 16 ||
                      UCCL_LITE_RS8M_BLOCKS == 32);
        if (bytes == (8UL << 20))
          scatterBlocks = std::min(blocks, UCCL_LITE_RS8M_BLOCKS);
#endif
#ifdef UCCL_LITE_RS8M_REMOTE_FIRST
        if (bytes == (8UL << 20))
          mediumScatterStreamed<true><<<scatterBlocks, 512, 0, stream>>>(
              send.channel, recv.channel, t.ib, buffer, largeLayout, rank, pos,
              count / 8, reinterpret_cast<uint32_t const*>(input),
              reinterpret_cast<uint32_t*>(output));
        else
#endif
          mediumScatterStreamed<false><<<scatterBlocks, 512, 0, stream>>>(
              send.channel, recv.channel, t.ib, buffer, largeLayout, rank, pos,
              count / 8, reinterpret_cast<uint32_t const*>(input),
              reinterpret_cast<uint32_t*>(output));
      } else {
        int pieces =
            bytes >= (4UL << 20)
                ? int(std::max<size_t>(
                      2, std::min<size_t>(
                             32, (bytes / 4 + (2UL << 20) - 1) / (2UL << 20))))
                : 1;
        bool aligned = ((reinterpret_cast<uintptr_t>(input) |
                         reinterpret_cast<uintptr_t>(output)) &
                        15) == 0;
        int reduceBlocks = std::min(blocks, 8);
#ifdef UCCL_LITE_AR256_RING_BLOCKS
        static_assert(UCCL_LITE_AR256_RING_BLOCKS == 2 ||
                      UCCL_LITE_AR256_RING_BLOCKS == 4 ||
                      UCCL_LITE_AR256_RING_BLOCKS == 8);
        if (bytes == (256UL << 10))
          reduceBlocks = std::min(blocks, UCCL_LITE_AR256_RING_BLOCKS);
#endif
        if (aligned)
          hierarchicalAllReduce<DirectRingChan, true, true>
              <<<reduceBlocks, 512, 0, stream>>>(
                  send, recv, t.localAg, t.ib, base, count, largeLayout,
                  rank % 4, pos, pieces, std::min(2, pieces), false,
                  reinterpret_cast<uint32_t const*>(input),
                  reinterpret_cast<uint32_t*>(output), retireSends);
        else
          hierarchicalAllReduce<DirectRingChan, true, false>
              <<<reduceBlocks, 512, 0, stream>>>(
                  send, recv, t.localAg, t.ib, base, count, largeLayout,
                  rank % 4, pos, pieces, std::min(2, pieces), false,
                  reinterpret_cast<uint32_t const*>(input),
                  reinterpret_cast<uint32_t*>(output), retireSends);
      }
      checked(cudaGetLastError());
      if (ag) {
        copyClient<<<256, 256, 0, stream>>>(
            reinterpret_cast<uint32_t*>(output),
            reinterpret_cast<uint32_t const*>(buffer + largeLayout.data),
            outputBytes / 4);
        checked(cudaGetLastError());
      }
      checked(cudaEventRecord(done, stream));
      ibOutstanding = !retireSends;
      lastStream = stream;
      pending = true;
      return mscclpp::CommResult::CommSuccess;
    }
    Layout bank = layout;
    bank.data += smallBase;
    bank.localScratch += smallBase;
    bank.ibSend += smallBase;
    bank.ibScratch += smallBase;
    bank.immutableInput += smallBase;
    if (sequence++ & 1) {
      bank.data += bankBytes;
      bank.localScratch += bankBytes;
      bank.ibSend += bankBytes;
      bank.ibScratch += bankBytes;
      bank.immutableInput += bankBytes;
    }
    int serialThreads = bytes >= (64UL << 10) ? 512 : 256;
#ifdef UCCL_LITE_AG_PACKET_THREADS
    static_assert(UCCL_LITE_AG_PACKET_THREADS == 256 ||
                  UCCL_LITE_AG_PACKET_THREADS == 384 ||
                  UCCL_LITE_AG_PACKET_THREADS == 512 ||
                  UCCL_LITE_AG_PACKET_THREADS == 768 ||
                  UCCL_LITE_AG_PACKET_THREADS == 1024);
#ifdef UCCL_LITE_AG_PACKET_THREADS_MIN_BYTES
    constexpr size_t agThreadsMinimum = UCCL_LITE_AG_PACKET_THREADS_MIN_BYTES;
#else
    constexpr size_t agThreadsMinimum = 32UL << 10;
#endif
    static_assert(agThreadsMinimum == (32UL << 10) ||
                  agThreadsMinimum == (64UL << 10) ||
                  agThreadsMinimum == (128UL << 10));
    if (op == NetworkOperation::AllGather && bytes >= agThreadsMinimum &&
        bytes <= (128UL << 10))
      serialThreads = UCCL_LITE_AG_PACKET_THREADS;
#endif
#ifdef UCCL_LITE_AR256_THREADS
    static_assert(
        UCCL_LITE_AR256_THREADS == 256 || UCCL_LITE_AR256_THREADS == 384 ||
        UCCL_LITE_AR256_THREADS == 512 || UCCL_LITE_AR256_THREADS == 1024);
    if (op == NetworkOperation::AllReduce && bytes == (256UL << 10))
      serialThreads = UCCL_LITE_AR256_THREADS;
#endif
    // Compile-time operation selection changes only code generation. Every
    // specialization retains the same body, banks and completion protocol.
    auto launchStaged = [&]<bool Packet>(uint32_t flag = 0, bool drain = true
#ifdef UCCL_LITE_PACKET_TIMELINE
                                         ,
                                         uint64_t* trace = nullptr
#endif
                        ) {
      auto launch = [&]<int Operation>() {
        stagedCollective<Packet, Operation><<<1, serialThreads, 0, stream>>>(
            t.localDirect, t.ib, rank, buffer, bank, op,
            reinterpret_cast<uint32_t const*>(input),
            reinterpret_cast<uint32_t*>(output), inputBytes / 4,
            outputBytes / 4, flag, drain
#ifdef UCCL_LITE_PACKET_TIMELINE
            ,
            trace
#endif
        );
      };
#ifdef UCCL_LITE_STATIC_COLLECTIVE
      if (op == NetworkOperation::AllGather)
        launch.template
        operator()<static_cast<int>(NetworkOperation::AllGather)>();
      else if (op == NetworkOperation::ReduceScatter)
        launch.template
        operator()<static_cast<int>(NetworkOperation::ReduceScatter)>();
      else
        launch.template
        operator()<static_cast<int>(NetworkOperation::AllReduce)>();
#else
      launch.template operator()<0>();
#endif
    };
    bool directGather128 = false;
#ifdef UCCL_LITE_AG_DIRECT128
    directGather128 =
        op == NetworkOperation::AllGather && bytes == (128UL << 10);
#endif
    if (!directGather128 &&
        (bytes <= (32UL << 10) ||
         (op == NetworkOperation::AllReduce && bytes <= (64UL << 10)) ||
         (op == NetworkOperation::AllGather && bytes <= (128UL << 10)))) {
      if (packetSequence == UINT32_MAX) {
        std::fprintf(stderr, "UCCL Lite LL reset: rank=%d\n", rank);
        if (pending) checked(cudaEventSynchronize(done));
        owner->bootstrap()->barrier();
        drainIb<<<1, 32, 0, stream>>>(t.ib);
        checked(cudaGetLastError());
        checked(cudaStreamSynchronize(stream));
        for (size_t b = 0; b < 2; ++b)
          checked(cudaMemset(
              buffer + smallBase + b * bankBytes + layout.immutableInput, 0,
              kPacketBankBytes));
        checked(cudaDeviceSynchronize());
        owner->bootstrap()->barrier();
        packetSequence = 0;
      }
      ++packetSequence;
#ifdef UCCL_LITE_INLINE_PACKET_DRAIN
      bool drain = true;
#else
      bool drain = (packetSequence % 32) == 0;
#endif
      // LL and deferred AR jointly leave at most 31 + 31*4 WRs pending;
      // the next call/drain remains below the 1024-entry send CQ.
      ibOutstanding = !drain;
#ifdef UCCL_LITE_PACKET_TIMELINE
      uint64_t* serialTraceCall = nullptr;
      if (op == NetworkOperation::AllGather && bytes == serialTimelineBytes &&
          serialTimeline && serialTimelineCount < kSerialTraceCalls)
        serialTraceCall =
            serialTimeline + kSerialTraceWords * serialTimelineCount++;
#endif
      launchStaged.template operator()<true>(packetSequence,
#ifdef UCCL_LITE_INLINE_PACKET_DRAIN
                                             true
#else
                                             false
#endif
#ifdef UCCL_LITE_PACKET_TIMELINE
                                             ,
                                             serialTraceCall
#endif
      );
      checked(cudaGetLastError());
#ifndef UCCL_LITE_INLINE_PACKET_DRAIN
      if (drain) drainIb<<<1, 32, 0, stream>>>(t.ib);
#endif
    } else {
      ibOutstanding = false;
      launchStaged.template operator()<false>();
    }
    checked(cudaGetLastError());
    checked(cudaEventRecord(done, stream));
    lastStream = stream;
    pending = true;
    return mscclpp::CommResult::CommSuccess;
  }
};
}  // namespace

class NetworkCollectives::Impl {
 public:
  NetworkId id;
  std::shared_ptr<mscclpp::Communicator> comm;
  std::shared_ptr<Context> context;
  mscclpp::AlgorithmCollection algorithms;
  Impl(NetworkId uid, int rank, int device, int rail, size_t capacity)
      : id(uid),
        context(std::make_shared<Context>(rank, device, rail, capacity)) {
    if (rank < 0 || rank >= 8 || device < 0 || rail < 0 || rail >= 8 ||
        !capacity || capacity > (1UL << 30) || capacity % 128)
      throw std::invalid_argument("unsupported network configuration");
    for (int i = 1; i <= 3; ++i) {
      auto op = static_cast<NetworkOperation>(i);
      std::string name = i == 1   ? "allgather"
                         : i == 2 ? "reducescatter"
                                  : "allreduce";
      auto ctx = context;
      auto algorithm = std::make_shared<mscclpp::NativeAlgorithm>(
          "uccl_lite_gpu_hierarchy", name,
          [ctx](auto comm) { ctx->prepare(comm); },
          [ctx, op](auto, void const* input, void* output, size_t in,
                    size_t out, auto dtype, auto reduction, cudaStream_t stream,
                    int, int, auto const&, auto) {
            if (dtype != mscclpp::DataType::FLOAT32 ||
                (op != NetworkOperation::AllGather &&
                 reduction != mscclpp::ReduceOp::SUM))
              return mscclpp::CommResult::CommInvalidArgument;
            return ctx->run(op, input, output, in, out, stream);
          },
          [ctx](auto, auto, auto, size_t, size_t,
                auto) -> std::shared_ptr<void> { return ctx; },
          [capacity](auto, auto, size_t, size_t, auto, bool) {
            // Internal registered storage is stable even if client pointers,
            // sizes or in-place layouts differ between ranks and calls.
            return mscclpp::AlgorithmCtxKey{nullptr, nullptr, capacity,
                                            capacity, 0};
          });
      algorithms.registerAlgorithm(name, algorithm->name(), algorithm);
    }
    algorithms.setSelectors(
        [](auto const& map, auto const& request) {
          return map.at(request.collective).at("uccl_lite_gpu_hierarchy");
        },
        [](auto const&, auto const&) -> std::shared_ptr<mscclpp::Algorithm> {
          return nullptr;
        });
  }
};

NetworkId NetworkCollectives::createUniqueId() {
  return mscclpp::TcpBootstrap::createUniqueId();
}
NetworkCollectives::NetworkCollectives(NetworkId id, int rank, int device,
                                       int rail, size_t capacity)
    : impl_(std::make_unique<Impl>(id, rank, device, rail, capacity)) {}
NetworkCollectives::~NetworkCollectives() = default;
bool NetworkCollectives::initialized() const {
  return bool(impl_->context->transport);
}
bool NetworkCollectives::idle() const {
  auto const& context = *impl_->context;
  return !context.ibOutstanding &&
         (!context.pending || cudaEventQuery(context.done) == cudaSuccess);
}
void NetworkCollectives::synchronize() {
  auto& c = *impl_->context;
  if (!c.transport) return;
#ifdef UCCL_LITE_COMPLETION_TRACE
  auto traceBegin = Context::TraceClock::now();
  auto traceRestoreBegin = traceBegin;
#ifdef UCCL_LITE_COMPLETION_GPU_EVENTS
  float traceGpuMs = 0;
#endif
  auto traceDeviceEnd = traceBegin, traceGpuEnd = traceBegin,
       traceHostEnd = traceBegin;
#endif
  int previous = c.device;
  bool switchDevice = true;
#ifdef UCCL_LITE_SYNC_PRIMARY_CONTEXT
  auto currentContext = [] {
    CUcontext context = nullptr;
    auto status = cuCtxGetCurrent(&context);
    if (status != CUDA_SUCCESS && status != CUDA_ERROR_NOT_INITIALIZED) {
      char const* message = nullptr;
      cuGetErrorString(status, &message);
      throw std::runtime_error(message ? message : "CUDA context query failed");
    }
    return context;
  };
  // Equal ordinals are insufficient: cudaSetDevice binds the primary context.
  // Skip redundant runtime calls only after observing that exact context.
  switchDevice = !c.syncPrimary || currentContext() != c.syncPrimary;
#endif
  if (switchDevice) {
    checked(cudaGetDevice(&previous));
    checked(cudaSetDevice(c.device));
  }
  auto waitForBoundary = [&] {
#ifdef UCCL_LITE_DRIVER_COMPLETION
    // Runtime and driver event handles are interchangeable. Keep the exact
    // event recorded at the collective boundary and preserve driver errors.
    CUresult status;
    while ((status = cuEventQuery(c.done)) == CUDA_ERROR_NOT_READY) {
#if defined(__x86_64__) || defined(__i386__)
      asm volatile("pause");
#else
      std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
    }
    if (status != CUDA_SUCCESS) {
      char const* message = nullptr;
      cuGetErrorString(status, &message);
      throw std::runtime_error(message ? message : "CUDA event query failed");
    }
#elif defined(UCCL_LITE_QUERY_COMPLETION)
    cudaError_t status;
    while ((status = cudaEventQuery(c.done)) == cudaErrorNotReady) {
#if defined(__x86_64__) || defined(__i386__)
      asm volatile("pause");
#else
      std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
    }
    checked(status);
#else
    checked(cudaEventSynchronize(c.done));
#endif
  };
#ifdef UCCL_LITE_COMPLETION_TRACE
  traceDeviceEnd = Context::TraceClock::now();
#endif
  try {
#ifdef UCCL_LITE_SYNC_PRIMARY_CONTEXT
    if (switchDevice) c.syncPrimary = currentContext();
#endif
#ifdef UCCL_LITE_ORDERED_SYNC_DRAIN
    if (c.ibOutstanding) {
      // Queue retirement after the exact collective boundary before waiting
      // on the host. Do not depend on later application work in lastStream.
      if (c.pending) checked(cudaStreamWaitEvent(c.graphBuilder, c.done, 0));
      drainIb<<<1, 32, 0, c.graphBuilder>>>(c.transport->ib);
      checked(cudaGetLastError());
      checked(cudaStreamSynchronize(c.graphBuilder));
      c.ibOutstanding = false;
    } else if (c.pending) {
      waitForBoundary();
    }
#ifdef UCCL_LITE_COMPLETION_TRACE
    traceGpuEnd = Context::TraceClock::now();
#endif
#ifdef UCCL_LITE_HOST_GATHER_EXPERIMENT
    if (c.hostGather)
      c.hostGather->synchronize(
#ifdef UCCL_LITE_HOST_SHARED_EVENT
          true
#endif
      );
#endif
#else
    if (c.pending) waitForBoundary();
#ifdef UCCL_LITE_HOST_GATHER_EXPERIMENT
    if (c.hostGather)
      c.hostGather->synchronize(
#ifdef UCCL_LITE_HOST_SHARED_EVENT
          true
#endif
      );
#endif
    if (c.ibOutstanding) {
#ifndef UCCL_LITE_QUIET_DRAIN
      std::fprintf(stderr, "UCCL Lite IB drain: rank=%d reason=synchronize\n",
                   c.rank);
#endif
      drainIb<<<1, 32, 0, c.graphBuilder>>>(c.transport->ib);
      checked(cudaGetLastError());
      checked(cudaStreamSynchronize(c.graphBuilder));
      c.ibOutstanding = false;
    }
#endif
#ifdef UCCL_LITE_COMPLETION_TRACE
    traceHostEnd = Context::TraceClock::now();
#ifdef UCCL_LITE_COMPLETION_GPU_EVENTS
    if (c.traceSequence != c.tracedSequence) {
      checked(cudaEventSynchronize(c.completionGpuEnd));
      checked(cudaEventElapsedTime(&traceGpuMs, c.completionGpuBegin,
                                   c.completionGpuEnd));
    }
#endif
#endif
  } catch (...) {
    if (switchDevice) cudaSetDevice(previous);
    throw;
  }
#ifdef UCCL_LITE_COMPLETION_TRACE
  traceRestoreBegin = Context::TraceClock::now();
#endif
  if (switchDevice) checked(cudaSetDevice(previous));
#ifdef UCCL_LITE_COMPLETION_TRACE
  auto traceEnd = Context::TraceClock::now();
  if (c.traceSequence != c.tracedSequence &&
      c.completionSampleCount < c.completionSamples.size()) {
    c.completionSamples[c.completionSampleCount++] = {
        c.traceSequence,
        c.traceBytes,
        c.traceEnqueue,
        Context::traceNs(traceBegin, traceDeviceEnd),
        Context::traceNs(traceDeviceEnd, traceGpuEnd),
        Context::traceNs(traceGpuEnd, traceHostEnd),
        Context::traceNs(traceRestoreBegin, traceEnd),
        c.traceOperation
#ifdef UCCL_LITE_COMPLETION_GPU_EVENTS
        ,
        traceGpuMs * 1000.f
#endif
    };
    c.tracedSequence = c.traceSequence;
  }
#endif
}
void NetworkCollectives::prepare() {
  auto& s = *impl_;
  if (initialized()) return;
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(s.context->rank, 8);
  bootstrap->initialize(s.id, 120);
  struct Metadata {
    size_t capacity;
    int device, rail;
    char host[256];
    char uuid[16];
  };
  std::array<Metadata, 8> metadata{};
  auto& local = metadata[s.context->rank];
  local.capacity = s.context->capacity;
  local.device = s.context->device;
  local.rail = s.context->rail;
  cudaDeviceProp property{};
  checked(cudaGetDeviceProperties(&property, local.device));
  std::memcpy(local.uuid, property.uuid.bytes, sizeof(local.uuid));
  if (gethostname(local.host, sizeof(local.host) - 1))
    throw std::runtime_error("host identity unavailable");
  bootstrap->allGather(metadata.data(), sizeof(Metadata));
  for (int r = 0; r < 8; ++r) {
    if (metadata[r].capacity != local.capacity || metadata[r].device != r % 4 ||
        metadata[r].rail != (r % 4) / 2 ||
        std::strcmp(metadata[r].host, metadata[r / 4 * 4].host) ||
        ((std::strcmp(metadata[r].host, metadata[0].host) == 0) != (r < 4)))
      throw std::invalid_argument("two-host placement/capacity mismatch");
    for (int p = r / 4 * 4; p < r; ++p)
      if (!std::memcmp(metadata[r].uuid, metadata[p].uuid, 16))
        throw std::invalid_argument("duplicate local GPU");
  }
  s.comm = std::make_shared<mscclpp::Communicator>(bootstrap);
  s.context->prepare(s.comm);
}
void NetworkCollectives::enqueue(NetworkOperation op, void const* input,
                                 void* output, size_t count,
                                 cudaStream_t stream) {
#ifdef UCCL_LITE_COMPLETION_TRACE
  auto traceBegin = Context::TraceClock::now();
#endif
  auto& s = *impl_;
  auto& c = *s.context;
  if (op != NetworkOperation::AllGather &&
      op != NetworkOperation::ReduceScatter &&
      op != NetworkOperation::AllReduce)
    throw std::invalid_argument("invalid network operation");
  if (!initialized()) throw std::logic_error("prepare before enqueue");
  int device;
  checked(cudaGetDevice(&device));
  size_t scale = op == NetworkOperation::AllReduce ? 4 : 32;
  if (device != c.device || !count || count > c.capacity / scale ||
      (count * scale) % 128 || !input || !output ||
      (reinterpret_cast<uintptr_t>(input) & 3) ||
      (reinterpret_cast<uintptr_t>(output) & 3))
    throw std::invalid_argument("unsupported network call");
  cudaStreamCaptureStatus capture;
  checked(cudaStreamIsCapturing(stream, &capture));
  if (capture != cudaStreamCaptureStatusNone)
    throw std::invalid_argument("network capture is unsupported");
  size_t bytes = count * scale;
  size_t in = op == NetworkOperation::AllGather ? bytes / 8 : bytes;
  size_t out = op == NetworkOperation::ReduceScatter ? bytes / 8 : bytes;
  std::string name = op == NetworkOperation::AllGather       ? "allgather"
                     : op == NetworkOperation::ReduceScatter ? "reducescatter"
                                                             : "allreduce";
  mscclpp::CollectiveRequest request{
      8,      4,      c.rank,
      input,  output, op == NetworkOperation::AllGather ? in : bytes,
      stream, name,   mscclpp::DataType::FLOAT32,
      {}};
  auto algorithm = s.algorithms.selectAlgorithm(request);
#ifdef UCCL_LITE_COMPLETION_GPU_EVENTS
  checked(cudaEventRecord(c.completionGpuBegin, stream));
#endif
  auto result = algorithm->execute(s.comm, input, output, in, out,
                                   mscclpp::DataType::FLOAT32,
                                   mscclpp::ReduceOp::SUM, stream, nullptr);
  if (result != mscclpp::CommResult::CommSuccess)
    throw std::runtime_error("MSCCL++ algorithm execution failed");
#ifdef UCCL_LITE_COMPLETION_GPU_EVENTS
  checked(cudaEventRecord(c.completionGpuEnd, stream));
#endif
#ifdef UCCL_LITE_COMPLETION_TRACE
  c.traceOperation = static_cast<int>(op);
  c.traceBytes = bytes;
  c.traceEnqueue = Context::traceNs(traceBegin, Context::TraceClock::now());
  ++c.traceSequence;
#endif
}
}  // namespace uccl_lite
