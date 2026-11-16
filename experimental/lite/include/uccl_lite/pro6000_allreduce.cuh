// SPDX-License-Identifier: Apache-2.0
#pragma once
// Topology-aware AllReduce for this host's 8 PCIe-only Blackwell GPUs.
//
// Rank routing follows the physical hierarchy reported by `nvidia-smi topo -m`:
//   xor 1: the other GPU behind the same PCIe switch (PIX)
//   xor 2: the other switch attached to the same CPU socket (NODE)
//   xor 4: the matching GPU on the other CPU socket (SYS)
//
// Recursive halving performs ReduceScatter in that order; recursive doubling
// performs AllGather in reverse.  Each rank moves
//   2 * (N/2 + N/4 + N/8) = 1.75N bytes,
// the bandwidth-optimal volume for eight ranks.  Only N/8 bytes per rank cross
// the inter-socket link in either direction.

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/memory_channel_device.hpp>
#include <mscclpp/packet_device.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <mscclpp/semaphore.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <sched.h>
#include <sys/wait.h>
#include <unistd.h>

namespace uccl_lite {
#ifndef UCCL_LITE_RANKS
#define UCCL_LITE_RANKS 8
#endif
constexpr int kRanks = UCCL_LITE_RANKS;
constexpr int kLevels = kRanks == 8 ? 3 : 2;
constexpr double kBusFactor = 2.0 * (kRanks - 1) / kRanks;

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

// CE uses one immutable send stage and one receive stage per ReduceScatter
// level.  Old send stages are deliberately not reused: proxy copies drain on a
// different CUDA stream and do not acknowledge completion to their sender.
// Receive stages are safely reused as AllGather assembly stages after all three
// matching waits have completed.
struct Layout {
  size_t data;
  size_t send[4];
  size_t recv[3];
  size_t pkt[6];
};

__device__ __forceinline__ uint2 f2u(float2 v) {
  return make_uint2(__float_as_uint(v.x), __float_as_uint(v.y));
}

__device__ __forceinline__ float2 u2f(uint2 v) {
  return make_float2(__uint_as_float(v.x), __uint_as_float(v.y));
}

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

__device__ __forceinline__ void copyOut(float* dst, float const* src, size_t n,
                                        int tid, int nth) {
  size_t n4 = n / 4;
  float4* d4 = reinterpret_cast<float4*>(dst);
  float4 const* s4 = reinterpret_cast<float4 const*>(src);
  for (size_t i = tid; i < n4; i += nth) d4[i] = s4[i];
  for (size_t i = n4 * 4 + tid; i < n; i += nth) dst[i] = src[i];
}

__device__ __forceinline__ Chan levelChan(int level, Chan sw, Chan socket,
                                          Chan system) {
  return level == 0 ? sw : (level == 1 ? socket : system);
}

// All threads publish their own writes before thread 0 hands the source region
// to the proxy.  A fence on thread 0 alone does not publish stores made by the
// rest of the grid.
__device__ __forceinline__ void publishToProxy() {
  __threadfence_system();
  gSyncer.sync(gridDim.x);
}

__global__ void __launch_bounds__(1024)
    ar8Put(Chan chSwitch, Chan chSocket, Chan chSystem, int rank, size_t nelem,
           Layout L) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nth = gridDim.x * blockDim.x;
  char* base = reinterpret_cast<char*>(chSwitch.src_);
  float* data = reinterpret_cast<float*>(base + L.data);

  size_t segBase = 0, segN = nelem;
  for (int level = 0; level < kLevels; ++level) {
    Chan ch = levelChan(level, chSwitch, chSocket, chSystem);
    size_t half = segN / 2;
    size_t keepBase = segBase + ((rank >> level) & 1) * half;
    size_t sendBase = segBase + (((rank >> level) & 1) ^ 1) * half;
    size_t bytes = half * sizeof(float);

    ch.put<16, true>(L.recv[level], L.data + sendBase * sizeof(float), bytes,
                     tid, nth);
    gSyncer.sync(gridDim.x);
    if (tid == 0) {
      ch.signal();
      ch.wait();
    }
    gSyncer.sync(gridDim.x);
    float* scratch = reinterpret_cast<float*>(base + L.recv[level]);
    addInto(data + keepBase, scratch, half, tid, nth);
    gSyncer.sync(gridDim.x);
    segBase = keepBase;
    segN = half;
  }

  for (int level = kLevels - 1; level >= 0; --level) {
    Chan ch = levelChan(level, chSwitch, chSocket, chSystem);
    size_t bytes = segN * sizeof(float);
    ch.put<16, true>(L.data + segBase * sizeof(float),
                     L.data + segBase * sizeof(float), bytes, tid, nth);
    gSyncer.sync(gridDim.x);
    if (tid == 0) {
      ch.signal();
      ch.wait();
    }
    gSyncer.sync(gridDim.x);
    size_t otherBase = segBase ^ segN;
    segBase = min(segBase, otherBase);
    segN *= 2;
  }
}

// LL is used only for small messages and launches one CTA.  Packet flags make
// peer handovers self-synchronising; __syncthreads preserves dependencies among
// the six local phases without paying a grid-wide barrier.
__global__ void __launch_bounds__(1024)
    ar8LL(Chan chSwitch, Chan chSocket, Chan chSystem, int rank, size_t nelem,
          Layout L, uint32_t flag) {
  int tid = threadIdx.x;
  int nth = blockDim.x;
  char* base = reinterpret_cast<char*>(chSwitch.src_);
  float2* data = reinterpret_cast<float2*>(base + L.data);

  size_t segBase = 0, segN = nelem;
  for (int level = 0; level < kLevels; ++level) {
    Chan ch = levelChan(level, chSwitch, chSocket, chSystem);
    size_t half = segN / 2;
    size_t keepBase = segBase + ((rank >> level) & 1) * half;
    size_t sendBase = segBase + (((rank >> level) & 1) ^ 1) * half;
    Pkt* tx =
        reinterpret_cast<Pkt*>(reinterpret_cast<char*>(ch.dst_) + L.pkt[level]);
    Pkt* rx = reinterpret_cast<Pkt*>(base + L.pkt[level]);
    for (size_t i = tid; i < half / 2; i += nth) {
      tx[i].write(f2u(data[sendBase / 2 + i]), flag);
      float2 a = data[keepBase / 2 + i];
      float2 b = u2f(rx[i].read(flag));
      a.x += b.x;
      a.y += b.y;
      data[keepBase / 2 + i] = a;
    }
    __syncthreads();
    segBase = keepBase;
    segN = half;
  }

  for (int level = kLevels - 1; level >= 0; --level) {
    Chan ch = levelChan(level, chSwitch, chSocket, chSystem);
    size_t otherBase = segBase ^ segN;
    Pkt* tx = reinterpret_cast<Pkt*>(reinterpret_cast<char*>(ch.dst_) +
                                     L.pkt[2 * kLevels - 1 - level]);
    Pkt* rx = reinterpret_cast<Pkt*>(base + L.pkt[2 * kLevels - 1 - level]);
    for (size_t i = tid; i < segN / 2; i += nth) {
      tx[i].write(f2u(data[segBase / 2 + i]), flag);
      data[otherBase / 2 + i] = u2f(rx[i].read(flag));
    }
    __syncthreads();
    segBase = min(segBase, otherBase);
    segN *= 2;
  }
}

__device__ __forceinline__ PortChan levelPort(int level, PortChan p0,
                                              PortChan p1, PortChan p2) {
  return level == 0 ? p0 : (level == 1 ? p1 : p2);
}

__global__ void __launch_bounds__(1024)
    arDrainedCE(PortChan rs0, PortChan rs1, PortChan rs2, PortChan ag0,
                PortChan ag1, PortChan ag2, float* rawBase, int rank,
                size_t nelem, Layout L, unsigned long long* profile,
                Chan completion) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nth = gridDim.x * blockDim.x;
  char* base = reinterpret_cast<char*>(rawBase);

  size_t segBase = 0, segN = nelem;
  for (int level = 0; level < kLevels; ++level) {
    unsigned long long start = clock64();
    PortChan p = levelPort(level, rs0, rs1, rs2);
    size_t half = segN / 2;
    size_t keepBase = segBase + ((rank >> level) & 1) * half;
    size_t sendBase = segBase + (((rank >> level) & 1) ^ 1) * half;
    if (level != 0)
      publishToProxy();
    else
      gSyncer.sync(gridDim.x);
    if (tid == 0) {
      p.putWithSignal(sendBase * sizeof(float), sendBase * sizeof(float),
                      half * sizeof(float));
      p.wait();
    }
    gSyncer.sync(gridDim.x);

    unsigned long long received = clock64();
    float* src = reinterpret_cast<float*>(base + L.send[level]);
    float* recv = reinterpret_cast<float*>(base + L.recv[level]);
    float* dst = reinterpret_cast<float*>(base + L.send[level + 1]);
    addOut(dst + keepBase, src + keepBase, recv + keepBase, half, tid, nth);
    gSyncer.sync(gridDim.x);
    if (profile && tid == 0) {
      atomicAdd(profile + 2 * level, received - start);
      atomicAdd(profile + 2 * level + 1, clock64() - received);
    }
    segBase = keepBase;
    segN = half;
  }

  // Reuse recv[2], recv[1], then data as successively larger assembly buffers.
  for (int level = kLevels - 1; level >= 0; --level) {
    unsigned long long start = clock64();
    PortChan p = levelPort(level, ag0, ag1, ag2);
    size_t dstAbs = level == 2 ? L.recv[2] : (level == 1 ? L.recv[1] : L.data);
    size_t srcAbs =
        level == 2 ? L.send[3] : (level == 1 ? L.recv[2] : L.recv[1]);
    float* src = reinterpret_cast<float*>(base + srcAbs);
    float* dst = reinterpret_cast<float*>(base + dstAbs);
    copyOut(dst + segBase, src + segBase, segN, tid, nth);
    publishToProxy();
    unsigned long long copied = clock64();
    if (tid == 0) {
      p.putWithSignal(segBase * sizeof(float), segBase * sizeof(float),
                      segN * sizeof(float));
      p.wait();
    }
    gSyncer.sync(gridDim.x);
    if (profile && tid == 0) {
      int phase = 2 * kLevels - 1 - level;
      atomicAdd(profile + 2 * phase, clock64() - copied);
      atomicAdd(profile + 2 * phase + 1, copied - start);
    }
    segBase = min(segBase, segBase ^ segN);
    segN *= 2;
  }
  // Incoming completion alone does not prove that our outgoing read from
  // recv[1] has drained. The next invocation reuses that staging region.
  if (tid == 0) {
    completion.signal();
    completion.wait();
  }
  gSyncer.sync(gridDim.x);
}

__global__ void __launch_bounds__(1024)
    arStagedCE(PortChan rs0, PortChan rs1, PortChan rs2, PortChan ag0,
               PortChan ag1, PortChan ag2, float* rawBase, int rank,
               size_t nelem, Layout L, unsigned long long* profile,
               Chan completion) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nth = gridDim.x * blockDim.x;
  char* base = reinterpret_cast<char*>(rawBase);

  size_t segBase = 0, segN = nelem;
  for (int level = 0; level < kLevels; ++level) {
    unsigned long long start = clock64();
    PortChan p = levelPort(level, rs0, rs1, rs2);
    size_t half = segN / 2;
    size_t keepBase = segBase + ((rank >> level) & 1) * half;
    size_t sendBase = segBase + (((rank >> level) & 1) ^ 1) * half;
    if (level != 0)
      publishToProxy();
    else
      gSyncer.sync(gridDim.x);
    if (tid == 0) {
      p.putWithSignal(sendBase * sizeof(float), sendBase * sizeof(float),
                      half * sizeof(float));
      p.wait();
    }
    gSyncer.sync(gridDim.x);

    unsigned long long received = clock64();
    float* src = reinterpret_cast<float*>(base + L.send[level]);
    float* recv = reinterpret_cast<float*>(base + L.recv[level]);
    float* dst = reinterpret_cast<float*>(base + L.send[level + 1]);
    addOut(dst + keepBase, src + keepBase, recv + keepBase, half, tid, nth);
    gSyncer.sync(gridDim.x);
    if (profile && tid == 0) {
      atomicAdd(profile + 2 * level, received - start);
      atomicAdd(profile + 2 * level + 1, clock64() - received);
    }
    segBase = keepBase;
    segN = half;
  }

  // Reuse recv[2], recv[1], then data as successively larger assembly buffers.
  for (int level = kLevels - 1; level >= 0; --level) {
    unsigned long long start = clock64();
    PortChan p = levelPort(level, ag0, ag1, ag2);
    size_t dstAbs = L.data;
    size_t srcAbs = level == 2 ? L.send[3] : L.data;
    float* src = reinterpret_cast<float*>(base + srcAbs);
    float* dst = reinterpret_cast<float*>(base + dstAbs);
    if (level == 2) copyOut(dst + segBase, src + segBase, segN, tid, nth);
    publishToProxy();
    unsigned long long copied = clock64();
    if (tid == 0) {
      p.putWithSignal(segBase * sizeof(float), segBase * sizeof(float),
                      segN * sizeof(float));
      p.wait();
    }
    gSyncer.sync(gridDim.x);
    if (profile && tid == 0) {
      int phase = 2 * kLevels - 1 - level;
      atomicAdd(profile + 2 * phase, clock64() - copied);
      atomicAdd(profile + 2 * phase + 1, copied - start);
    }
    segBase = min(segBase, segBase ^ segN);
    segN *= 2;
  }
  // The final direct write must drain before the next invocation reuses data.
  if (tid == 0) {
    completion.signal();
    completion.wait();
  }
  gSyncer.sync(gridDim.x);
}

// A CTA owns one contiguous tensor tile. No grid-wide barrier is needed.
// Double-buffered LL packets preserve the original reuse protocol.
__global__ void arChunkLL(Chan c0, Chan c1, Chan c2, int rank, size_t nelem,
                          Layout layout, uint32_t flag) {
  size_t chunk = nelem / gridDim.x;
  size_t offset = blockIdx.x * chunk * sizeof(float);
  layout.data += offset;
  for (int p = 0; p < 2 * kLevels; ++p) layout.pkt[p] += offset;
  int tid = threadIdx.x, nth = blockDim.x;
  char* base = reinterpret_cast<char*>(c0.src_);
  float2* data = reinterpret_cast<float2*>(base + layout.data);
  size_t segBase = 0, segN = chunk;
  for (int level = 0; level < kLevels; ++level) {
    Chan ch = levelChan(level, c0, c1, c2);
    size_t half = segN / 2;
    size_t keep = segBase + ((rank >> level) & 1) * half;
    size_t send = segBase + (((rank >> level) & 1) ^ 1) * half;
    Pkt* tx = reinterpret_cast<Pkt*>(reinterpret_cast<char*>(ch.dst_) +
                                     layout.pkt[level]);
    Pkt* rx = reinterpret_cast<Pkt*>(base + layout.pkt[level]);
    for (size_t i = tid; i < half / 2; i += nth) {
      tx[i].write(f2u(data[send / 2 + i]), flag);
      float2 a = data[keep / 2 + i], b = u2f(rx[i].read(flag));
      data[keep / 2 + i] = make_float2(a.x + b.x, a.y + b.y);
    }
    __syncthreads();
    segBase = keep;
    segN = half;
  }
  for (int level = kLevels - 1; level >= 0; --level) {
    Chan ch = levelChan(level, c0, c1, c2);
    int p = 2 * kLevels - 1 - level;
    size_t other = segBase ^ segN;
    Pkt* tx = reinterpret_cast<Pkt*>(reinterpret_cast<char*>(ch.dst_) +
                                     layout.pkt[p]);
    Pkt* rx = reinterpret_cast<Pkt*>(base + layout.pkt[p]);
    for (size_t i = tid; i < segN / 2; i += nth) {
      tx[i].write(f2u(data[segBase / 2 + i]), flag);
      data[other / 2 + i] = u2f(rx[i].read(flag));
    }
    __syncthreads();
    segBase = min(segBase, other);
    segN *= 2;
  }
}

struct ChunkChannels {
  PortChan rs[3], ag[3];
  Chan ack[3];
};

// Gray-code ring: every edge is an existing XOR-1/2/4 connection. Only one
// directed edge per socket crosses SYS, spreading inter-socket traffic across
// the operation instead of concentrating it into two collective-wide phases.
__host__ __device__ inline int ringRank(int position) {
  return position ^ (position >> 1);
}
__host__ __device__ inline int ringPosition(int rank) {
  int position = rank;
  for (int shift = 1; shift < kRanks; shift <<= 1)
    position ^= position >> shift;
  return position;
}
__host__ __device__ inline int ringLevel(int rank, int direction) {
  int peer = ringRank((ringPosition(rank) + direction + kRanks) % kRanks);
  int level = 0;
  while ((1 << level) != (rank ^ peer)) ++level;
  return level;
}

struct RingChannels {
  PortChan first, reduced, gather, incoming;
  Chan acknowledge, drained;
};

// Each lane owns disjoint tensor tiles. RS receives each shard once into recv,
// and writes its partial sum into a distinct position of send[1]. Those source
// positions stay immutable until the call completes. AG forwards completed
// shards directly from data. The final acknowledgement proves every outgoing
// copy drained before the next invocation can overwrite data or scratch.
__global__ void arRingCE(RingChannels const* channels, char* base, int rank,
                         size_t nelem, size_t tileElems, Layout layout,
                         mscclpp::DeviceSyncer* syncers, int workers) {
  int lane = blockIdx.x / workers, lanes = gridDim.x / workers;
  int tid = (blockIdx.x % workers) * blockDim.x + threadIdx.x;
  int nth = blockDim.x * workers;
  auto ch = channels[lane];
  int position = ringPosition(rank);
  float* data = reinterpret_cast<float*>(base + layout.data);
  float* partial = reinterpret_cast<float*>(base + layout.send[1]);
  float* recv = reinterpret_cast<float*>(base + layout.recv[0]);
  for (size_t offset = lane * tileElems; offset < nelem;
       offset += lanes * tileElems) {
    size_t shard = min(tileElems, nelem - offset) / kRanks;
    for (int step = 0; step < kRanks - 1; ++step) {
      int sendShard = (position - step + kRanks) % kRanks;
      int recvShard = (sendShard - 1 + kRanks) % kRanks;
      __threadfence_system();
      syncers[lane].sync(workers);
      if (tid == 0) {
        auto port = step == 0 ? ch.first : ch.reduced;
        size_t send = (offset + sendShard * shard) * sizeof(float);
        port.putWithSignal(send, send, shard * sizeof(float));
        ch.incoming.wait();
      }
      syncers[lane].sync(workers);
      size_t keep = offset + recvShard * shard;
      float* dst = step == kRanks - 2 ? data : partial;
      addOut(dst + keep, data + keep, recv + keep, shard, tid, nth);
    }
    // Publish the last reduction once. AG only forwards CE-written bytes, so
    // the leader can advance its seven receives without waking every CTA or
    // flushing SM stores again between steps.
    __threadfence_system();
    syncers[lane].sync(workers);
    if (tid == 0) {
      for (int step = 0; step < kRanks - 1; ++step) {
        int sendShard = (position + 1 - step + kRanks) % kRanks;
        size_t send = (offset + sendShard * shard) * sizeof(float);
        ch.gather.putWithSignal(send, send, shard * sizeof(float));
        ch.incoming.wait();
      }
      ch.acknowledge.signal();
      ch.drained.wait();
    }
    syncers[lane].sync(workers);
  }
}

// Independent lanes pipeline contiguous tiles through the topology. RS sends
// the discarded half and reduces the kept half in place. AG writes directly
// into data. A peer GPU acknowledgement after each receive proves the source
// copy has drained before a later phase or invocation can reuse that storage.
// No NCCL calls or caller staging are involved.
__global__ void arChunkCE(ChunkChannels const* channels, char* base, int rank,
                          size_t nelem, size_t tileElems, Layout layout,
                          unsigned long long* profile,
                          mscclpp::DeviceSyncer* syncers, int workers) {
  int lane = blockIdx.x / workers;
  int lanes = gridDim.x / workers;
  auto ch = channels[lane];
  int tid = (blockIdx.x % workers) * blockDim.x + threadIdx.x;
  int nth = blockDim.x * workers;
  for (size_t offset = lane * tileElems; offset < nelem;
       offset += lanes * tileElems) {
    size_t segBase = offset, segN = min(tileElems, nelem - offset);
    for (int phase = 0; phase < 2 * kLevels; ++phase) {
      int level = phase < kLevels ? phase : 2 * kLevels - 1 - phase;
      bool rs = phase < kLevels;
      size_t half = rs ? segN / 2 : segN;
      size_t keep = rs ? segBase + ((rank >> level) & 1) * half : segBase;
      size_t send = rs ? segBase + (((rank >> level) & 1) ^ 1) * half : segBase;
      unsigned long long start = clock64();
      __threadfence_system();
      syncers[lane].sync(workers);
      if (tid == 0) {
        PortChan port = rs ? ch.rs[level] : ch.ag[level];
        port.putWithSignal(send * sizeof(float), send * sizeof(float),
                           half * sizeof(float));
        port.wait();
        ch.ack[level].signal();
        ch.ack[level].wait();
      }
      syncers[lane].sync(workers);
      unsigned long long received = clock64();
      if (rs) {
        float* data = reinterpret_cast<float*>(base + layout.data);
        float* recv = reinterpret_cast<float*>(base + layout.recv[level]);
        addInto(data + keep, recv + keep, half, tid, nth);
        segBase = keep;
        segN = half;
      } else {
        segBase = min(segBase, segBase ^ segN);
        segN *= 2;
      }
      syncers[lane].sync(workers);
      if (profile && tid == 0) {
        atomicAdd(profile + 2 * phase, received - start);
        atomicAdd(profile + 2 * phase + 1, clock64() - received);
      }
    }
  }
}

// The preceding stream-ordered CE copy has completed locally. Signal the peer,
// wait for its copy, then reduce. Kernel completion orders the next CE copy.
__global__ void arHostPhase(Chan channel, float* data, float const* scratch,
                            size_t keep, size_t count, bool reduce,
                            mscclpp::DeviceSyncer* syncer) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    channel.signal();
    channel.wait();
  }
  syncer->sync(gridDim.x);
  if (reduce)
    addInto(data + keep, scratch + keep, count, tid, gridDim.x * blockDim.x);
}

// Host-queued ring control: the preceding CE transfer is complete on this
// stream. GPU signals replace proxy-issued tiny H2D signal copies. All kernels,
// transfers, and stream joins belong to the completed-call timing interval.
__global__ void arRingHostPhase(Chan next, Chan prev, float* data,
                                float* partial, float const* recv, size_t keep,
                                size_t count, bool reduce, bool lastReduce,
                                mscclpp::DeviceSyncer* syncer) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    next.signal();
    prev.wait();
  }
  if (!reduce) return;
  syncer->sync(gridDim.x);
  if (reduce) {
    float* dst = lastReduce ? data : partial;
    addOut(dst + keep, data + keep, recv + keep, count, tid,
           gridDim.x * blockDim.x);
  }
}

}  // namespace uccl_lite
