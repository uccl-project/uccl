// SPDX-License-Identifier: Apache-2.0
#pragma once
// Experimental FP32 standalone AllGather/ReduceScatter. Reuse the tested
// MSCCL++ primitives and physical Gray-code ring from the AllReduce experiment.
#include "pro6000_allreduce.cuh"

namespace uccl_lite {
struct AgRsLayout {
  size_t input, output, partial, receive, packets;
};
// Every lane/peer direction has exactly one signaling thread, serialized on
// its CUDA stream. Publish a monotonically increasing value instead of a
// remote read-modify-write. MSCCL++ still owns the semaphore and acquire wait.
// A null counter retains the original atomic signal as an ablation control.
struct AgRsPeer : Chan {
  uint64_t* sent = nullptr;
  __device__ void signal() const {
    if (!sent) {
      auto sem = semaphore_;
      sem.signal();
      return;
    }
    uint64_t value = *sent + 1;
    *sent = value;
    asm volatile("st.release.sys.global.u64 [%0], %1;"
                 :
                 : "l"(semaphore_.remoteInboundToken), "l"(value)
                 : "memory");
  }
  __device__ void wait(int64_t maxSpinCount = 10000000) const {
    auto sem = semaphore_;
    sem.wait(maxSpinCount);
  }
};
struct AgRsChannels {
  int order = 0, direction = 1;
  AgRsPeer peers[kRanks];
  PortChan first, partial, gather, incoming;
  AgRsPeer acknowledge, drained;
};

// Two socket-local ring orders; optionally alternate lane directions to use
// both SYS gateways in each direction without changing public shard order.
__host__ __device__ inline int agRsRingRank(int position, int order,
                                            int direction) {
  int rank = ringRank(direction == 1 ? position : (kRanks - position) % kRanks);
  return order ? rank ^ ((rank & 2) >> 1) : rank;
}
__host__ __device__ inline int agRsRingPosition(int rank, int order,
                                                int direction) {
  int position = ringPosition(order ? rank ^ ((rank & 2) >> 1) : rank);
  return direction == 1 ? position : (kRanks - position) % kRanks;
}

// Keep several independent vector loads in flight before issuing the peer
// stores. All harness shard/tile boundaries are 16-byte aligned, including
// the non-power-of-two tail cases. The generic MSCCL++ copy remains a control.
template <int Unroll>
__device__ __forceinline__ void agRsCopyVectors(void* destination, void* source,
                                                size_t bytes, int tid,
                                                int nth) {
  auto dst = static_cast<uint4*>(destination);
  auto src = static_cast<uint4*>(source);
  size_t words = bytes / sizeof(uint4);
  for (size_t base = tid; base < words; base += size_t(nth) * Unroll) {
    uint4 values[Unroll];
#pragma unroll
    for (int u = 0; u < Unroll; ++u) {
      size_t i = base + size_t(u) * nth;
      if (i < words) values[u] = src[i];
    }
#pragma unroll
    for (int u = 0; u < Unroll; ++u) {
      size_t i = base + size_t(u) * nth;
      if (i < words) dst[i] = values[u];
    }
  }
}
__device__ __forceinline__ void agRsPutPayload(AgRsPeer ch, size_t dst,
                                               size_t src, size_t bytes,
                                               int tid, int nth, int unroll) {
  if (unroll == 4)
    agRsCopyVectors<4>(static_cast<char*>(ch.dst_) + dst,
                       static_cast<char*>(ch.src_) + src, bytes, tid, nth);
  else if (unroll == 8)
    agRsCopyVectors<8>(static_cast<char*>(ch.dst_) + dst,
                       static_cast<char*>(ch.src_) + src, bytes, tid, nth);
  else
    ch.template put<16, true>(dst, src, bytes, tid, nth);
}

// Every element exchanges with every rank before its next invocation can
// finish. Two packet generations therefore prevent overwrite of unread data.
// Packet positions encode SOURCE rank; RS input positions encode DESTINATION
// rank. Output always uses public rank order, independent of ring order.
template <bool Gather>
__global__ void agRsLL(AgRsChannels const* channels, char* base, int rank,
                       size_t shard, AgRsLayout layout, uint32_t* epochs,
                       size_t packetSlot) {
  __shared__ uint32_t flag;
  if (threadIdx.x == 0) flag = ++epochs[blockIdx.x];
  __syncthreads();
  layout.packets += (flag & 1) * packetSlot;
  auto const& ch = channels[0];
  float const* input = reinterpret_cast<float*>(base + layout.input);
  float* output = reinterpret_cast<float*>(base + layout.output);
  auto rx = reinterpret_cast<Pkt*>(base + layout.packets);
  size_t pairs = shard / 2;
  for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < pairs;
       j += size_t(gridDim.x) * blockDim.x) {
    float2 local =
        reinterpret_cast<float2 const*>(input)[(Gather ? 0 : rank * pairs) + j];
    for (int delta = 1; delta < kRanks; ++delta) {
      int peer = (rank + delta) % kRanks;
      auto tx = reinterpret_cast<Pkt*>(static_cast<char*>(ch.peers[peer].dst_) +
                                       layout.packets);
      float2 value =
          Gather ? local
                 : reinterpret_cast<float2 const*>(input)[peer * pairs + j];
      tx[rank * pairs + j].write(f2u(value), flag);
    }
    if (Gather) reinterpret_cast<float2*>(output)[rank * pairs + j] = local;
    for (int peer = 0; peer < kRanks; ++peer) {
      if (peer == rank) continue;
      float2 value = u2f(rx[peer * pairs + j].read(flag));
      if (Gather)
        reinterpret_cast<float2*>(output)[peer * pairs + j] = value;
      else {
        local.x += value.x;
        local.y += value.y;
      }
    }
    if (!Gather) reinterpret_cast<float2*>(output)[j] = local;
  }
}

// Recursive doubling/halving uses only log2(ranks) peer exchanges. A CTA owns
// the same slice of every public-rank shard, so no grid barrier is needed.
// AG grows PIX -> NODE -> SYS. RS shrinks SYS -> NODE -> PIX to leave the
// standard rank-contiguous shard at its owner (no hidden rank permutation).
template <bool Gather>
__global__ void agRsHierLL(AgRsChannels const* channels, char* base, int rank,
                           size_t shard, AgRsLayout layout, uint32_t* epochs,
                           size_t packetSlot) {
  __shared__ uint32_t flag;
  if (threadIdx.x == 0) flag = ++epochs[blockIdx.x];
  __syncthreads();
  layout.packets += (flag & 1) * packetSlot;
  size_t phaseSlot = packetSlot / kLevels;
  size_t pairs = shard / 2;
  size_t chunk = (pairs + gridDim.x - 1) / gridDim.x;
  size_t offset = blockIdx.x * chunk;
  if (offset >= pairs) return;
  chunk = min(chunk, pairs - offset);
  auto const& ch = channels[0];
  float2 const* input = reinterpret_cast<float2 const*>(base + layout.input);
  float2* output = reinterpret_cast<float2*>(base + layout.output);
  float2* partial = reinterpret_cast<float2*>(base + layout.partial);
  if (Gather) {
    for (size_t j = threadIdx.x; j < chunk; j += blockDim.x)
      output[rank * pairs + offset + j] = input[offset + j];
    __syncthreads();
  }
  for (int step = 0; step < kLevels; ++step) {
    int level = Gather ? step : kLevels - 1 - step;
    int group = 1 << level;
    int own = (rank / group) * group, other = own ^ group;
    int peer = rank ^ group;
    size_t packetBase = layout.packets + step * phaseSlot;
    auto tx = reinterpret_cast<Pkt*>(static_cast<char*>(ch.peers[peer].dst_) +
                                     packetBase);
    auto rx = reinterpret_cast<Pkt*>(base + packetBase);
    for (size_t j = threadIdx.x; j < group * chunk; j += blockDim.x) {
      int part = j / chunk;
      size_t element = offset + j % chunk;
      size_t keep = (own + part) * pairs + element;
      size_t send = (other + part) * pairs + element;
      if (Gather)
        tx[keep].write(f2u(output[keep]), flag);
      else {
        float2 const* source = step == 0 ? input : partial;
        tx[send].write(f2u(source[send]), flag);
      }
    }
    // Issue the whole slice before polling. Alternating one 8-byte packet
    // write and one peer read serializes each thread on PCIe round trips.
    for (size_t j = threadIdx.x; j < group * chunk; j += blockDim.x) {
      int part = j / chunk;
      size_t element = offset + j % chunk;
      size_t keep = (own + part) * pairs + element;
      size_t send = (other + part) * pairs + element;
      if (Gather)
        output[send] = u2f(rx[send].read(flag));
      else {
        float2 const* source = step == 0 ? input : partial;
        float2 a = source[keep], b = u2f(rx[keep].read(flag));
        float2 result = make_float2(a.x + b.x, a.y + b.y);
        if (step == kLevels - 1)
          output[element] = result;
        else
          partial[keep] = result;
      }
    }
    __syncthreads();
  }
}

// Direct SM writes avoid packet expansion in the medium-size range. Each lane
// owns a disjoint interval and its own peer semaphores. The second handshake
// protects RS readers and the next invocation's input/output reuse.
template <bool Gather>
__global__ void agRsPut(AgRsChannels const* channels, char* base, int rank,
                        size_t shard, size_t tile, AgRsLayout layout,
                        mscclpp::DeviceSyncer* syncers, int workers,
                        bool fullFences, int copyUnroll) {
  int lane = blockIdx.x / workers, lanes = gridDim.x / workers;
  int tid = (blockIdx.x % workers) * blockDim.x + threadIdx.x;
  int nth = blockDim.x * workers;
  auto const& ch = channels[lane];
  float const* input = reinterpret_cast<float*>(base + layout.input);
  float* output = reinterpret_cast<float*>(base + layout.output);
  float const* receive = reinterpret_cast<float*>(base + layout.receive);
  if (Gather) {
    // A previous call's completion does not mean the caller has finished
    // consuming its output. Wait for this call's receiver kernel to start.
    if (tid == 0) {
      for (int p = 0; p < kRanks; ++p)
        if (p != rank) ch.peers[p].signal();
      for (int p = 0; p < kRanks; ++p)
        if (p != rank) ch.peers[p].wait();
    }
    syncers[lane].sync(workers);
  }
  for (size_t offset = lane * tile; offset < shard; offset += lanes * tile) {
    size_t count = min(tile, shard - offset);
    for (int delta = 1; delta < kRanks; ++delta) {
      int p = (rank + delta) % kRanks;
      size_t src = layout.input + (Gather ? offset : p * shard + offset) * 4;
      size_t dst = (Gather ? layout.output : layout.receive) +
                   (rank * shard + offset) * 4;
      agRsPutPayload(ch.peers[p], dst, src, count * 4, tid, nth, copyUnroll);
    }
    if (fullFences || workers != 1) __threadfence_system();
    syncers[lane].sync(workers);
    if (tid == 0) {
      for (int p = 0; p < kRanks; ++p)
        if (p != rank) ch.peers[p].signal();
      for (int p = 0; p < kRanks; ++p)
        if (p != rank) ch.peers[p].wait();
    }
    syncers[lane].sync(workers);
    if (Gather)
      copyOut(output + rank * shard + offset, input + offset, count, tid, nth);
    else {
      copyOut(output + offset, input + rank * shard + offset, count, tid, nth);
      for (int p = 0; p < kRanks; ++p)
        if (p != rank)
          addInto(output + offset, receive + p * shard + offset, count, tid,
                  nth);
    }
    if (fullFences) __threadfence_system();
    syncers[lane].sync(workers);
    if (tid == 0) {
      for (int p = 0; p < kRanks; ++p)
        if (p != rank) ch.peers[p].signal();
      for (int p = 0; p < kRanks; ++p)
        if (p != rank) ch.peers[p].wait();
    }
    syncers[lane].sync(workers);
  }
}

// A single CTA's barrier orders its workers before the leader's system-scope
// release signal. Multi-CTA remote writes retain per-thread system fences.
// Local partial sums need only a device barrier before the next SM copy.
// --full-fences retains the conservative per-thread fences as a control.
// SM-driven Gray-code ring for the range where CE proxy latency dominates.
// Every lane advances independently; signals publish all participating CTAs'
// stores, and the final acknowledgement drains each sender before reuse.
template <bool Gather>
__global__ void agRsRingPut(AgRsChannels const* channels, char* base, int rank,
                            size_t shard, size_t tile, AgRsLayout layout,
                            mscclpp::DeviceSyncer* syncers, int workers,
                            bool fullFences, int copyUnroll, bool fusedCopy) {
  int lane = blockIdx.x / workers, lanes = gridDim.x / workers;
  int tid = (blockIdx.x % workers) * blockDim.x + threadIdx.x;
  int nth = blockDim.x * workers;
  auto const& ch = channels[lane];
  int position = agRsRingPosition(rank, ch.order, ch.direction);
  int next = agRsRingRank((position + 1) % kRanks, ch.order, ch.direction);
  int prev =
      agRsRingRank((position + kRanks - 1) % kRanks, ch.order, ch.direction);
  // Read the immutable table through its device pointer. Copying the whole
  // dynamically indexed table made ptxas allocate an 888-byte stack per thread.
  auto nextCh = ch.peers[next];
  auto prevCh = ch.peers[prev];
  float const* input = reinterpret_cast<float*>(base + layout.input);
  float* output = reinterpret_cast<float*>(base + layout.output);
  float* partial = reinterpret_cast<float*>(base + layout.partial);
  float const* receive = reinterpret_cast<float*>(base + layout.receive);
  if (Gather) {
    if (tid == 0) {
      prevCh.signal();
      nextCh.wait();
    }
    syncers[lane].sync(workers);
  }
  for (size_t offset = lane * tile; offset < shard; offset += lanes * tile) {
    size_t count = min(tile, shard - offset);
    if (!fusedCopy) {
      if (Gather)
        copyOut(output + rank * shard + offset, input + offset, count, tid,
                nth);
      if (fullFences) __threadfence_system();
      syncers[lane].sync(workers);
    }
    for (int step = 0; step < kRanks - 1; ++step) {
      int sendOwner =
          agRsRingRank((position - (Gather ? 0 : 1) - step + kRanks) % kRanks,
                       ch.order, ch.direction);
      size_t send = (sendOwner * shard + offset) * sizeof(float);
      size_t src =
          Gather ? layout.output : (step == 0 ? layout.input : layout.partial);
      size_t dst = Gather ? layout.output : layout.receive;
      if (fusedCopy) {
        // AG's first read supplies both local output and the peer. RS reduces
        // the previously received shard directly into its outgoing store,
        // avoiding an intermediate partial write/read and a second barrier.
        size_t element = send / sizeof(float);
        auto target = reinterpret_cast<float4*>(
            static_cast<char*>(nextCh.dst_) + dst + send);
        auto a = reinterpret_cast<float4 const*>(
            Gather ? (step == 0 ? input + offset : output + element)
                   : input + element);
        auto b = reinterpret_cast<float4 const*>(receive + element);
        auto local = reinterpret_cast<float4*>(output + element);
        for (size_t j = tid; j < count / 4; j += nth) {
          float4 value = a[j];
          if (!Gather && step != 0) {
            float4 other = b[j];
            value.x += other.x;
            value.y += other.y;
            value.z += other.z;
            value.w += other.w;
          }
          target[j] = value;
          if (Gather && step == 0) local[j] = value;
        }
      } else {
        agRsPutPayload(nextCh, dst + send, src + send, count * 4, tid, nth,
                       copyUnroll);
      }
      if (fullFences || workers != 1) __threadfence_system();
      syncers[lane].sync(workers);
      if (tid == 0) {
        nextCh.signal();
        prevCh.wait();
      }
      syncers[lane].sync(workers);
      if (!Gather && (!fusedCopy || step == kRanks - 2)) {
        int recvOwner = agRsRingRank((position - 2 - step + kRanks) % kRanks,
                                     ch.order, ch.direction);
        size_t keep = recvOwner * shard + offset;
        float* dest = step == kRanks - 2 ? output + offset : partial + keep;
        addOut(dest, input + keep, receive + keep, count, tid, nth);
        if (fullFences) __threadfence_system();
        syncers[lane].sync(workers);
      }
    }
    if (tid == 0) {
      prevCh.signal();
      nextCh.wait();
    }
    syncers[lane].sync(workers);
  }
}

// Each lane owns a contiguous interval WITHIN each rank's shard. This retains
// public rank-contiguous output, unlike the tiled AllReduce's internal shard
// assignment. Partial sums have distinct immutable positions until completion.
template <bool Gather>
__global__ void agRsRingCE(AgRsChannels const* channels, char* base, int rank,
                           size_t shard, size_t tile, AgRsLayout layout,
                           mscclpp::DeviceSyncer* syncers, int workers) {
  int lane = blockIdx.x / workers, lanes = gridDim.x / workers;
  int tid = (blockIdx.x % workers) * blockDim.x + threadIdx.x;
  int nth = blockDim.x * workers;
  auto ch = channels[lane];
  int position = ringPosition(rank);
  float const* input = reinterpret_cast<float*>(base + layout.input);
  float* output = reinterpret_cast<float*>(base + layout.output);
  float* partial = reinterpret_cast<float*>(base + layout.partial);
  float const* receive = reinterpret_cast<float*>(base + layout.receive);
  if (Gather) {
    if (tid == 0) {
      ch.acknowledge.signal();
      ch.drained.wait();
    }
    syncers[lane].sync(workers);
  }
  for (size_t offset = lane * tile; offset < shard; offset += lanes * tile) {
    size_t count = min(tile, shard - offset);
    if (Gather) {
      copyOut(output + rank * shard + offset, input + offset, count, tid, nth);
      __threadfence_system();
      syncers[lane].sync(workers);
      if (tid == 0) {
        for (int step = 0; step < kRanks - 1; ++step) {
          int owner = ringRank((position - step + kRanks) % kRanks);
          size_t bytes = (owner * shard + offset) * sizeof(float);
          ch.gather.putWithSignal(bytes, bytes, count * sizeof(float));
          ch.incoming.wait();
        }
      }
    } else {
      for (int step = 0; step < kRanks - 1; ++step) {
        int sendOwner = ringRank((position - 1 - step + kRanks) % kRanks);
        int recvOwner = ringRank((position - 2 - step + kRanks) % kRanks);
        __threadfence_system();
        syncers[lane].sync(workers);
        if (tid == 0) {
          auto port = step == 0 ? ch.first : ch.partial;
          size_t bytes = (sendOwner * shard + offset) * sizeof(float);
          port.putWithSignal(bytes, bytes, count * sizeof(float));
          ch.incoming.wait();
        }
        syncers[lane].sync(workers);
        size_t keep = recvOwner * shard + offset;
        float* dst = step == kRanks - 2 ? output + offset : partial + keep;
        addOut(dst, input + keep, receive + keep, count, tid, nth);
      }
      // The final reduction must finish before the receive region is released.
      __threadfence_system();
      syncers[lane].sync(workers);
    }
    if (tid == 0) {
      ch.acknowledge.signal();
      ch.drained.wait();
    }
    syncers[lane].sync(workers);
  }
}
// Host-queued CE phases share the caller's dependency chain through explicit
// event joins. A phase kernel follows its outgoing copy on the SAME stream.
template <bool Gather>
__global__ void agRsHostPhase(AgRsChannels ch, char* base, int rank,
                              size_t shard, size_t offset, size_t count,
                              AgRsLayout layout, mscclpp::DeviceSyncer* syncer,
                              int step) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nth = gridDim.x * blockDim.x;
  int position = ringPosition(rank);
  int next = ringRank((position + 1) % kRanks);
  int prev = ringRank((position + kRanks - 1) % kRanks);
  float const* input = reinterpret_cast<float*>(base + layout.input);
  float* output = reinterpret_cast<float*>(base + layout.output);
  if (Gather && step == -1) {
    if (tid == 0) {
      ch.acknowledge.signal();
      ch.drained.wait();
    }
    syncer->sync(gridDim.x);
    copyOut(output + rank * shard + offset, input + offset, count, tid, nth);
    __threadfence_system();
    syncer->sync(gridDim.x);
    return;
  }
  if (tid == 0) {
    ch.peers[next].signal();
    ch.peers[prev].wait();
  }
  syncer->sync(gridDim.x);
  if (!Gather) {
    float* partial = reinterpret_cast<float*>(base + layout.partial);
    float const* receive = reinterpret_cast<float*>(base + layout.receive);
    int owner = ringRank((position - 2 - step + kRanks) % kRanks);
    size_t keep = owner * shard + offset;
    float* dest = step == kRanks - 2 ? output + offset : partial + keep;
    addOut(dest, input + keep, receive + keep, count, tid, nth);
    __threadfence_system();
    syncer->sync(gridDim.x);
  }
  if (step == kRanks - 2 && tid == 0) {
    ch.acknowledge.signal();
    ch.drained.wait();
  }
}
}  // namespace uccl_lite
