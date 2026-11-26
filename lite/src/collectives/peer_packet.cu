// Initial four-process FP32 backend for the shared strict NCCL adapter.
// Explicit candidate only; performance qualification is separate.
#include "uccl_lite/collectives/peer_packet.hpp"
#ifdef UCCL_LITE_LOCAL_HOST_REPLICA
#include "local_host_copy.h"
#endif
#if defined(UCCL_LITE_LOCAL_HOST_REPLICA) || defined(UCCL_LITE_LOCAL_AG_PULL)
#include <cooperative_groups.h>
#endif
#if defined(UCCL_LITE_LOCAL_AG_GRAPH_OUTPUT) || \
    defined(UCCL_LITE_LOCAL_SMALL_GRAPH) ||     \
    defined(UCCL_LITE_LOCAL_DRIVER_TREE) ||     \
    defined(UCCL_LITE_LOCAL_RS_MEMOP) || defined(UCCL_LITE_LOCAL_RS_GRAPHS)
#include <map>
#include <cuda.h>
#endif
#ifdef UCCL_LITE_LOCAL_HOST_AG
#include "../multinode/shared_host_packets.h"
#endif
#include <cuda/atomic>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/packet_device.hpp>
#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace uccl_lite {
namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
#ifndef UCCL_LITE_LOCAL_TEST_GENERATION_LIMIT
constexpr unsigned long long kGenerationLimit = 0xffffffffULL;
#else
static_assert(UCCL_LITE_LOCAL_TEST_GENERATION_LIMIT > 0 &&
              UCCL_LITE_LOCAL_TEST_GENERATION_LIMIT <= 0xffffffffULL);
constexpr unsigned long long kGenerationLimit =
    UCCL_LITE_LOCAL_TEST_GENERATION_LIMIT;
#endif
#if defined(UCCL_LITE_LOCAL_AG_HOST_READY) &&    \
    (!defined(UCCL_LITE_LOCAL_AG_EPOCH_GRAPH) || \
     !defined(UCCL_LITE_LOCAL_HOST_AG))
#error "Host readiness requires the epoch graph and shared-host runtime"
#endif
#ifndef UCCL_LITE_LOCAL_AR_RING_MIN_BYTES
constexpr size_t kArRingMinimum = 8UL << 20;
#else
static_assert(UCCL_LITE_LOCAL_AR_RING_MIN_BYTES == (4UL << 20) ||
              UCCL_LITE_LOCAL_AR_RING_MIN_BYTES == (8UL << 20));
constexpr size_t kArRingMinimum = UCCL_LITE_LOCAL_AR_RING_MIN_BYTES;
#endif
struct Peers {
  float* data[4];
  unsigned long long* flags[4];
};
#ifdef UCCL_LITE_LOCAL_AG_PACKET_BYTES
#ifndef UCCL_LITE_LOCAL_PACKET_SPECIALIZE
#error "Extended AG packets require operation specialization"
#endif
#ifdef UCCL_LITE_LOCAL_FIXED_PACKET_BANKS
#error "Extended AG packets use size-dependent banks"
#endif
static_assert(UCCL_LITE_LOCAL_AG_PACKET_BYTES >= 8192 &&
              UCCL_LITE_LOCAL_AG_PACKET_BYTES <= (128UL << 10) &&
              !(UCCL_LITE_LOCAL_AG_PACKET_BYTES &
                (UCCL_LITE_LOCAL_AG_PACKET_BYTES - 1)));
constexpr size_t kPacketEpochOffset =
    std::max(size_t(128UL << 10), size_t(4 * UCCL_LITE_LOCAL_AG_PACKET_BYTES));
#else
constexpr size_t kPacketEpochOffset = 128UL << 10;
#endif
// Flat AG uses four packet rows and two banks: four times its output bytes.
// The separate epoch family keeps a private 128-KiB region above those banks.
constexpr size_t kPacketReserveBytes = kPacketEpochOffset + (128UL << 10);

#ifdef UCCL_LITE_LOCAL_BATCH_SIGNAL
__device__ void signalRelaxedGlobal(void* address, unsigned long long value) {
  asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(address), "l"(value)
               : "memory");
}
#endif
#ifdef UCCL_LITE_LOCAL_BATCH_ACQUIRE
#ifndef UCCL_LITE_LOCAL_BATCH_SIGNAL
#error "Batched acquire requires batched release publication"
#endif
__device__ unsigned long long loadRelaxedGlobal(void const* address) {
  unsigned long long value;
  asm volatile("ld.relaxed.sys.global.u64 %0, [%1];"
               : "=l"(value)
               : "l"(address)
               : "memory");
  return value;
}
#endif
#if defined(UCCL_LITE_LOCAL_GRID_PEERS) || \
    defined(UCCL_LITE_LOCAL_SERIAL_SPECIALIZE)
#define UCCL_LITE_LOCAL_PEER_REF Peers const&
#else
#define UCCL_LITE_LOCAL_PEER_REF Peers
#endif
__device__ void peerBarrier(UCCL_LITE_LOCAL_PEER_REF peers, int rank, int phase,
                            unsigned long long generation) {
#ifdef UCCL_LITE_LOCAL_BATCH_SIGNAL
  // One release fence publishes the preceding data before all four relaxed
  // atomic flag stores. Receivers retain system-acquire loads.
  __threadfence_system();
  for (int r = 0; r < 4; ++r) {
#ifdef UCCL_LITE_LOCAL_BATCH_ACQUIRE
    if (r == rank) continue;
#endif
    signalRelaxedGlobal(peers.flags[r] + phase * 4 + rank, generation);
  }
#else
  for (int r = 0; r < 4; ++r) {
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_system> remote(
        peers.flags[r][phase * 4 + rank]);
    remote.store(generation, cuda::memory_order_release);
  }
#endif
#ifdef UCCL_LITE_LOCAL_BATCH_ACQUIRE
  for (int r = 0; r < 4; ++r)
    if (r != rank)
      while (loadRelaxedGlobal(peers.flags[rank] + phase * 4 + r) <
             generation) {
      }
  // Acquire fence after every observed flag pairs with every sender's
  // release fence before publication. The caller retains its CTA/grid join.
  __threadfence_system();
#else
  for (int r = 0; r < 4; ++r) {
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_system> incoming(
        peers.flags[rank][phase * 4 + r]);
    while (incoming.load(cuda::memory_order_acquire) < generation) {
    }
  }
#endif
}
#if defined(UCCL_LITE_LOCAL_AG_EPOCH_GRAPH) || defined(UCCL_LITE_LOCAL_AG_PULL)
__device__ void peerBarrierConst(Peers const& peers, int rank, int phase,
                                 unsigned long long generation) {
#ifdef UCCL_LITE_LOCAL_BATCH_SIGNAL
  // One release fence publishes the preceding data before all four relaxed
  // atomic flag stores. Receivers retain system-acquire loads.
  __threadfence_system();
  for (int r = 0; r < 4; ++r) {
#ifdef UCCL_LITE_LOCAL_BATCH_ACQUIRE
    if (r == rank) continue;
#endif
    signalRelaxedGlobal(peers.flags[r] + phase * 4 + rank, generation);
  }
#else
  for (int r = 0; r < 4; ++r) {
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_system> remote(
        peers.flags[r][phase * 4 + rank]);
    remote.store(generation, cuda::memory_order_release);
  }
#endif
#ifdef UCCL_LITE_LOCAL_BATCH_ACQUIRE
  for (int r = 0; r < 4; ++r)
    if (r != rank)
      while (loadRelaxedGlobal(peers.flags[rank] + phase * 4 + r) <
             generation) {
      }
  // Acquire fence after every observed flag pairs with every sender's
  // release fence before publication. The caller retains its CTA/grid join.
  __threadfence_system();
#else
  for (int r = 0; r < 4; ++r) {
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_system> incoming(
        peers.flags[rank][phase * 4 + r]);
    while (incoming.load(cuda::memory_order_acquire) < generation) {
    }
  }
#endif
}
#endif
__global__ void barrier(Peers peers, int rank, int phase,
                        unsigned long long generation) {
  peerBarrier(peers, rank, phase, generation);
}
__device__ float sumShard(float const* source, size_t i, size_t shard) {
  float value = source[i];
  for (int r = 1; r < 4; ++r) value += source[r * shard + i];
  return value;
}
__global__ void reduceShard(float const* source, float* output, size_t shard) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < shard;
       i += size_t(gridDim.x) * blockDim.x)
    output[i] = sumShard(source, i, shard);
}
#if defined(UCCL_LITE_LOCAL_HIER_SM) || \
    defined(UCCL_LITE_LOCAL_SERIAL_VECTOR) || defined(UCCL_LITE_LOCAL_HOST_AG)
__device__ float4 loadClient(float const* source, size_t i) {
  if ((reinterpret_cast<uintptr_t>(source) & 15) == 0)
    return reinterpret_cast<float4 const*>(source)[i];
  source += 4 * i;
  return make_float4(source[0], source[1], source[2], source[3]);
}
__device__ void storeClient(float* target, size_t i, float4 value) {
  if ((reinterpret_cast<uintptr_t>(target) & 15) == 0)
    reinterpret_cast<float4*>(target)[i] = value;
  else {
    target += 4 * i;
    target[0] = value.x;
    target[1] = value.y;
    target[2] = value.z;
    target[3] = value.w;
  }
}
__device__ float4 sumQuad(float4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
#endif
#ifdef UCCL_LITE_LOCAL_AG_EPOCH_GRAPH
#if !defined(UCCL_LITE_LOCAL_AG_GRAPH_OUTPUT) || \
    !defined(UCCL_LITE_LOCAL_SERIAL_CREDIT) ||   \
    !defined(UCCL_LITE_LOCAL_SERIAL_VECTOR)
#error "Epoch gather graphs require graph storage, serial credits and vector IO"
#endif
// Graph-private generation and barrier words avoid host parameter updates.
// Output is a kernel argument, so recycled pointers do not cache memcpy
// allocation bindings. Stream order protects the shared generation counter.
__global__ void gatherEpochPublish(const __grid_constant__ Peers peers,
                                   int rank, float const* input, size_t count,
                                   size_t bankElements,
                                   unsigned long long* sequence
#ifdef UCCL_LITE_LOCAL_AG_HOST_READY
                                   ,
                                   unsigned* ready
#endif
) {
  __shared__ unsigned long long epoch;
  if (!threadIdx.x) epoch = ++*sequence;
  __syncthreads();
  size_t bank = (epoch & 1) * bankElements;
  for (int phase = 0; phase < 4; ++phase) {
    int dest = (rank + phase) % 4;
    auto* target = peers.data[dest] + bank + rank * count;
    for (size_t i = threadIdx.x; i < count / 4; i += blockDim.x)
      reinterpret_cast<float4*>(target)[i] = loadClient(input, i);
    __syncthreads();
  }
#ifndef UCCL_LITE_LOCAL_SERIAL_REFERENCE_FENCE
  __threadfence_system();
#endif
  __syncthreads();
#ifdef UCCL_LITE_LOCAL_AG_HOST_READY
  if (!threadIdx.x) {
    unsigned tag = static_cast<unsigned>(epoch);
    cuda::atomic_ref<unsigned, cuda::thread_scope_system>(ready[rank])
        .store(tag, cuda::memory_order_release);
    for (;;) {
      uint4 seen;
      asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(seen.x), "=r"(seen.y), "=r"(seen.z), "=r"(seen.w)
                   : "l"(ready)
                   : "memory");
      if (seen.x >= tag && seen.y >= tag && seen.z >= tag && seen.w >= tag)
        break;
      __nanosleep(64);
    }
    __threadfence_system();
  }
#else
  if (!threadIdx.x) peerBarrierConst(peers, rank, 4, epoch);
#endif
  __syncthreads();
}
#if defined(UCCL_LITE_LOCAL_AG_EPOCH_CACHE64) && \
    !defined(UCCL_LITE_LOCAL_AG_EPOCH_CACHE128)
#error "Cached 64-KiB gather requires the cached producer kernel"
#endif
#ifdef UCCL_LITE_LOCAL_AG_EPOCH_CACHE128
template <int VectorsPerThread>
__global__ void gatherEpochPublishCached(const __grid_constant__ Peers peers,
                                         int rank, float const* input,
                                         size_t count, size_t bankElements,
                                         unsigned long long* sequence
#ifdef UCCL_LITE_LOCAL_AG_HOST_READY
                                         ,
                                         unsigned* ready
#endif
) {
  __shared__ unsigned long long epoch;
  if (!threadIdx.x) epoch = ++*sequence;
  __syncthreads();
  size_t bank = (epoch & 1) * bankElements;
  float4 cached[VectorsPerThread];
#pragma unroll
  for (int j = 0; j < VectorsPerThread; ++j)
    cached[j] = loadClient(input, threadIdx.x + size_t(j) * blockDim.x);
  for (int phase = 0; phase < 4; ++phase) {
    int dest = (rank + phase) % 4;
    auto* target = peers.data[dest] + bank + rank * count;
#pragma unroll
    for (int j = 0; j < VectorsPerThread; ++j)
      reinterpret_cast<float4*>(target)[threadIdx.x + size_t(j) * blockDim.x] =
          cached[j];
    __syncthreads();
  }
#ifndef UCCL_LITE_LOCAL_SERIAL_REFERENCE_FENCE
  __threadfence_system();
#endif
  __syncthreads();
#ifdef UCCL_LITE_LOCAL_AG_HOST_READY
  if (!threadIdx.x) {
    unsigned tag = static_cast<unsigned>(epoch);
    cuda::atomic_ref<unsigned, cuda::thread_scope_system>(ready[rank])
        .store(tag, cuda::memory_order_release);
    for (;;) {
      uint4 seen;
      asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(seen.x), "=r"(seen.y), "=r"(seen.z), "=r"(seen.w)
                   : "l"(ready)
                   : "memory");
      if (seen.x >= tag && seen.y >= tag && seen.z >= tag && seen.w >= tag)
        break;
      __nanosleep(64);
    }
    __threadfence_system();
  }
#else
  if (!threadIdx.x) peerBarrierConst(peers, rank, 4, epoch);
#endif
  __syncthreads();
}
#endif
#if defined(UCCL_LITE_LOCAL_AG_PREFETCH_OUT_ONLY) && \
    !defined(UCCL_LITE_LOCAL_AG_OUTPUT_PREFETCH)
#error "Layout-specific prefetch requires output prefetch"
#endif
__global__ void gatherEpochCopy(float const* staging, float* output,
                                size_t count, size_t bankElements,
                                unsigned long long const* sequence
#ifdef UCCL_LITE_LOCAL_AG_PREFETCH_OUT_ONLY
                                ,
                                bool outOfPlace
#endif
) {
  __shared__ unsigned long long epoch;
  if (!threadIdx.x) {
#ifdef UCCL_LITE_LOCAL_AG_OUTPUT_PREFETCH
    if (count == 4096
#ifdef UCCL_LITE_LOCAL_AG_PREFETCH_OUT_ONLY
        && outOfPlace
#endif
    ) {
      auto* begin = output + size_t(blockIdx.x) * blockDim.x * 4;
      auto* end = begin + blockDim.x * 4 - 1;
      // These addresses belong to the block's output slice. Start their
      // translation while loading the epoch and staged input.
      asm volatile("prefetch.global.L2 [%0];" ::"l"(begin) : "memory");
      asm volatile("prefetch.global.L2 [%0];" ::"l"(end) : "memory");
    }
#endif
    epoch = *reinterpret_cast<unsigned long long const volatile*>(sequence);
  }
  __syncthreads();
  staging += (epoch & 1) * bankElements;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x)
    storeClient(output, i, loadClient(staging, i));
}
#endif
#ifdef UCCL_LITE_LOCAL_AG_SEGMENTED
#if !defined(UCCL_LITE_LOCAL_SERIAL_CREDIT) || \
    !defined(UCCL_LITE_LOCAL_SERIAL_VECTOR) || \
    defined(UCCL_LITE_LOCAL_AG_EPOCH_GRAPH) || \
    defined(UCCL_LITE_LOCAL_AG_PACKET_BYTES)
#error "Segmented gather requires serial credits and its own medium-size path"
#endif
// Each CTA owns a disjoint vector slice and a private ready cache line.
// Producer and consumer complete in one kernel; two banks retain the same
// next-call publication credit as the serial staging protocol.
__global__ void segmentedGather(const __grid_constant__ Peers peers, int rank,
                                float const* input, float* output, size_t count,
                                size_t bankElements,
                                unsigned long long generation) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  bool active = i < count / 4;
  float4 value{};
  if (active) value = loadClient(input, i);
  size_t bank = (generation & 1) * bankElements;
  int control = 160 + blockIdx.x * 16;
  if (active) storeClient(output + rank * count, i, value);
  for (int phase = 1; phase < 4; ++phase) {
    int dest = (rank + phase) % 4;
    if (active)
      reinterpret_cast<float4*>(peers.data[dest] + bank + rank * count)[i] =
          value;
    __syncthreads();
    if (!threadIdx.x)
      cuda::atomic_ref<unsigned long long, cuda::thread_scope_system>(
          peers.flags[dest][control + rank])
          .store(generation, cuda::memory_order_release);
  }
  if (!threadIdx.x)
    for (int source = 0; source < 4; ++source) {
      if (source == rank) continue;
      cuda::atomic_ref<unsigned long long, cuda::thread_scope_system> ready(
          peers.flags[rank][control + source]);
      while (ready.load(cuda::memory_order_acquire) < generation)
        __nanosleep(64);
    }
  __syncthreads();
  if (active)
    for (int source = 0; source < 4; ++source)
      if (source != rank)
        storeClient(output + source * count, i,
                    reinterpret_cast<float4 const*>(peers.data[rank] + bank +
                                                    source * count)[i]);
}
#endif
#ifdef UCCL_LITE_LOCAL_AG_PULL
#if !defined(UCCL_LITE_LOCAL_SERIAL_CREDIT) || \
    !defined(UCCL_LITE_LOCAL_SERIAL_VECTOR) || \
    defined(UCCL_LITE_LOCAL_AG_SEGMENTED) ||   \
    defined(UCCL_LITE_LOCAL_AG_EPOCH_GRAPH) || \
    defined(UCCL_LITE_LOCAL_AG_PACKET_BYTES)
#error \
    "Private pull gather requires serial credits and its own medium-size path"
#endif
__device__ __forceinline__ float4 loadPeerQuad(float const* source, size_t i) {
  uint4 value;
  asm volatile(
#ifdef UCCL_LITE_LOCAL_AG_PULL_CG
      "ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
#else
      "ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
#endif
      : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
      : "l"(source + 4 * i)
      : "memory");
  return make_float4(__uint_as_float(value.x), __uint_as_float(value.y),
                     __uint_as_float(value.z), __uint_as_float(value.w));
}
__global__ void pullGather(const __grid_constant__ Peers peers, int rank,
                           float const* input, float* output, size_t count,
                           size_t bankElements, unsigned long long generation) {
  auto grid = cooperative_groups::this_grid();
  size_t bank = (generation & 1) * bankElements;
  size_t tid = grid.thread_rank();
  auto* local = reinterpret_cast<float4*>(peers.data[rank] + bank);
  for (size_t i = tid; i < count / 4; i += grid.size())
    local[i] = loadClient(input, i);
  if (tid < count / 4) __threadfence_system();
  grid.sync();
  if (!tid) peerBarrierConst(peers, rank, 7, generation);
  grid.sync();
  int source = blockIdx.x % 4;
  size_t lane = (blockIdx.x / 4) * blockDim.x + threadIdx.x;
  size_t stride = (gridDim.x / 4) * blockDim.x;
  for (size_t i = lane; i < count / 4; i += stride)
    storeClient(output + source * count, i,
                loadPeerQuad(peers.data[source] + bank, i));
  // Publication in n+1 follows completion of all reads in n. Receiving all
  // n+1 publications before starting n+2 protects the reused parity bank.
}
#endif
#ifdef UCCL_LITE_LOCAL_AG_HIER_SM
#if !defined(UCCL_LITE_LOCAL_SERIAL_CREDIT) ||  \
    !defined(UCCL_LITE_LOCAL_SERIAL_VECTOR) ||  \
    defined(UCCL_LITE_LOCAL_AG_SEGMENTED) ||    \
    defined(UCCL_LITE_LOCAL_AG_EPOCH_GRAPH) ||  \
    defined(UCCL_LITE_LOCAL_AG_PACKET_BYTES) || \
    defined(UCCL_LITE_LOCAL_AG_PULL)
#error \
    "Hierarchical staged gather requires serial credits and its own medium path"
#endif
__device__ mscclpp::DeviceSyncer gatherPairSync;
__device__ void gatherPairExchange(Peers const& peers, int rank, int peer,
                                   int phase, unsigned long long generation) {
  // Match the established SM hierarchy's grid publication then leader release.
  gatherPairSync.sync(gridDim.x);
  if (!blockIdx.x && !threadIdx.x) {
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_system>(
        peers.flags[peer][96 + phase * 4 + rank])
        .store(generation, cuda::memory_order_release);
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_system> ready(
        peers.flags[rank][96 + phase * 4 + peer]);
    while (ready.load(cuda::memory_order_acquire) < generation) __nanosleep(64);
  }
  gatherPairSync.sync(gridDim.x);
}
__global__ void hierarchicalGather(const __grid_constant__ Peers peers,
                                   int rank, float const* input, float* output,
                                   size_t count, size_t bankElements,
                                   unsigned long long generation) {
  size_t bank = (generation & 1) * bankElements;
  size_t tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t threads = size_t(gridDim.x) * blockDim.x;
  auto* local = peers.data[rank] + bank;
  auto* intra = peers.data[rank ^ 1] + bank;
  auto* cross = peers.data[rank ^ 2] + bank;
  for (size_t i = tid; i < count / 4; i += threads) {
    float4 value = loadClient(input, i);
    reinterpret_cast<float4*>(local + rank * count)[i] = value;
    reinterpret_cast<float4*>(intra + rank * count)[i] = value;
  }
  gatherPairExchange(peers, rank, rank ^ 1, 0, generation);
  for (int source = rank & ~1; source < (rank & ~1) + 2; ++source)
    for (size_t i = tid; i < count / 4; i += threads)
      reinterpret_cast<float4*>(cross + source * count)[i] =
          reinterpret_cast<float4 const*>(local + source * count)[i];
  gatherPairExchange(peers, rank, rank ^ 2, 1, generation);
  for (int source = 0; source < 4; ++source)
    for (size_t i = tid; i < count / 4; i += threads)
      storeClient(output + source * count, i,
                  reinterpret_cast<float4 const*>(local + source * count)[i]);
  // A cross-pair n+1 publication requires both members of that pair to have
  // entered n+1. Together with our local-pair join, this proves all n readers
  // retired before this rank can overwrite their bank in n+2.
}
#endif
// Adapted from peer_ag_rs.cu's pushSerialGather/pushSerialScatter. Each
// source addresses one destination at a time. Only local scratch is read.
// Completion barriers make scratch reuse safe across independently queued
// processes; the standalone harness previously supplied that boundary.
__global__ void serialCollect(Peers peers, int rank, int op, float const* input,
                              float* output, size_t count, size_t bankElements,
                              unsigned long long generation) {
#ifdef UCCL_LITE_LOCAL_SERIAL_CREDIT
  // Publication of n+1 follows consumption of n on every rank. Waiting
  // for every n+1 publication protects the parity bank reused by n+2.
  if (op != 3)
    for (int r = 0; r < 4; ++r)
      peers.data[r] += (generation & 1) * bankElements;
#endif
#ifdef UCCL_LITE_LOCAL_AG_COPY_OUTPUT
#ifndef UCCL_LITE_LOCAL_SERIAL_CREDIT
#error "Split gather output requires protected scratch banks"
#endif
  bool copyOutput = op == 1 && (count == 4096 || count == 16384);
#endif
  size_t shard = op == 3 ? count / 4 : count;
#ifdef UCCL_LITE_LOCAL_AG_SHARED_INPUT
  if (op == 1 && (count == 4096 || count == 16384)) {
    extern __shared__ __align__(16) unsigned char cachedInput[];
    auto* cached = reinterpret_cast<float4*>(cachedInput);
    for (size_t i = threadIdx.x; i < count / 4; i += blockDim.x)
      cached[i] = loadClient(input, i);
    __syncthreads();
    input = reinterpret_cast<float const*>(cachedInput);
  }
#endif
  for (int phase = 0; phase < 4; ++phase) {
    int dest = (rank + phase) % 4;
#ifdef UCCL_LITE_LOCAL_SERIAL_SKIP_SELF
#ifndef UCCL_LITE_LOCAL_SERIAL_VECTOR
#error "Direct self contribution requires vector serial transfers"
#endif
    if (op != 3 && dest == rank
#ifdef UCCL_LITE_LOCAL_AG_COPY_OUTPUT
        && !copyOutput
#endif
    )
      continue;
#endif
    auto* target = peers.data[dest] + rank * shard;
    auto const* source = input + (op == 1 ? 0 : dest * shard);
#ifdef UCCL_LITE_LOCAL_SERIAL_VECTOR
    // Match the retained serial transfer's 16-byte width. Caller pointers
    // can be float-aligned; internal rank rows remain 16-byte aligned.
    for (size_t i = threadIdx.x; i < shard / 4; i += blockDim.x)
      reinterpret_cast<float4*>(target)[i] = loadClient(source, i);
#else
    for (size_t i = threadIdx.x; i < shard; i += blockDim.x)
      target[i] = source[i];
#endif
    __syncthreads();
  }
#ifndef UCCL_LITE_LOCAL_SERIAL_REFERENCE_FENCE
  __threadfence_system();
#endif
  // As in the original arHierPut: block publication precedes the leader's
  // system-release signal; the receiving leader acquires before its barrier.
  __syncthreads();
  if (!threadIdx.x) peerBarrier(peers, rank, 0, generation);
  __syncthreads();
  auto* local = peers.data[rank];
#ifdef UCCL_LITE_LOCAL_AG_COPY_OUTPUT
  if (copyOutput) return;
#endif
  if (op == 3) {
    // Materialize the reduced shard locally before the next peer-write phase.
    for (size_t i = threadIdx.x; i < shard; i += blockDim.x)
      local[bankElements + rank * shard + i] = sumShard(local, i, shard);
    __syncthreads();
    for (int phase = 1; phase < 4; ++phase) {
      int dest = (rank + phase) % 4;
      for (size_t i = threadIdx.x; i < shard; i += blockDim.x)
        peers.data[dest][bankElements + rank * shard + i] =
            local[bankElements + rank * shard + i];
      __syncthreads();
    }
    __threadfence_system();
    __syncthreads();
    if (!threadIdx.x) peerBarrier(peers, rank, 1, generation);
    __syncthreads();
    for (size_t i = threadIdx.x; i < count; i += blockDim.x)
      output[i] = local[bankElements + i];
  } else {
    size_t extent = op == 1 ? 4 * count : count;
#ifdef UCCL_LITE_LOCAL_SERIAL_VECTOR
#ifdef UCCL_LITE_LOCAL_SERIAL_SKIP_SELF
    if (op == 1) {
      // Copy whole source rows: avoid a runtime 64-bit divide/modulo for
      // every vector when selecting the direct self contribution.
      for (int source = 0; source < 4; ++source) {
        auto const* from = source == rank ? input : local + source * shard;
        auto* to = output + source * shard;
        for (size_t i = threadIdx.x; i < shard / 4; i += blockDim.x)
          storeClient(to, i, loadClient(from, i));
      }
    } else
#endif
      for (size_t i = threadIdx.x; i < extent / 4; i += blockDim.x) {
        auto const* quads = reinterpret_cast<float4 const*>(local);
#ifdef UCCL_LITE_LOCAL_SERIAL_SKIP_SELF
        float4 value;
        if (op == 1) {
          value = i / (shard / 4) == rank ? loadClient(input, i % (shard / 4))
                                          : quads[i];
        } else {
          value = rank == 0 ? loadClient(input, i) : quads[i];
          for (int r = 1; r < 4; ++r)
            value =
                sumQuad(value, r == rank ? loadClient(input + rank * shard, i)
                                         : quads[r * (shard / 4) + i]);
        }
#else
      float4 value = quads[i];
      if (op == 2)
        for (int r = 1; r < 4; ++r)
          value = sumQuad(value, quads[r * (shard / 4) + i]);
#endif
        storeClient(output, i, value);
      }
#else
    for (size_t i = threadIdx.x; i < extent; i += blockDim.x)
      output[i] = op == 1 ? local[i] : sumShard(local, i, shard);
#endif
  }
#ifdef UCCL_LITE_LOCAL_SERIAL_CREDIT
  if (op != 3) return;
#endif
  __syncthreads();
  if (!threadIdx.x) peerBarrier(peers, rank, 2, generation);
}
#ifdef UCCL_LITE_LOCAL_SERIAL_SPECIALIZE
#ifndef UCCL_LITE_LOCAL_SERIAL_CREDIT
#error "Specialized serial collectives require parity credits"
#endif
template <int Op>
__global__ void serialCollectSpecialized(const __grid_constant__ Peers peers,
                                         int rank, int unusedOp,
                                         float const* input, float* output,
                                         size_t count, size_t bankElements,
                                         unsigned long long generation) {
#ifdef UCCL_LITE_LOCAL_SERIAL_CREDIT
  // Publication of n+1 follows consumption of n on every rank. Waiting
  // for every n+1 publication protects the parity bank reused by n+2.
  size_t bank = Op != 3 ? (generation & 1) * bankElements : 0;
#endif
#ifdef UCCL_LITE_LOCAL_AG_COPY_OUTPUT
#ifndef UCCL_LITE_LOCAL_SERIAL_CREDIT
#error "Split gather output requires protected scratch banks"
#endif
  bool copyOutput = Op == 1 && (count == 4096 || count == 16384);
#endif
  size_t shard = Op == 3 ? count / 4 : count;
#ifdef UCCL_LITE_LOCAL_AG_SHARED_INPUT
  if (Op == 1 && (count == 4096 || count == 16384)) {
    extern __shared__ __align__(16) unsigned char cachedInput[];
    auto* cached = reinterpret_cast<float4*>(cachedInput);
    for (size_t i = threadIdx.x; i < count / 4; i += blockDim.x)
      cached[i] = loadClient(input, i);
    __syncthreads();
    input = reinterpret_cast<float const*>(cachedInput);
  }
#endif
  for (int phase = 0; phase < 4; ++phase) {
    int dest = (rank + phase) % 4;
#ifdef UCCL_LITE_LOCAL_SERIAL_SKIP_SELF
#ifndef UCCL_LITE_LOCAL_SERIAL_VECTOR
#error "Direct self contribution requires vector serial transfers"
#endif
    if (Op != 3 && dest == rank
#ifdef UCCL_LITE_LOCAL_AG_COPY_OUTPUT
        && !copyOutput
#endif
    )
      continue;
#endif
    auto* target = (peers.data[dest] + bank) + rank * shard;
    auto const* source = input + (Op == 1 ? 0 : dest * shard);
#ifdef UCCL_LITE_LOCAL_SERIAL_VECTOR
    // Match the retained serial transfer's 16-byte width. Caller pointers
    // can be float-aligned; internal rank rows remain 16-byte aligned.
    for (size_t i = threadIdx.x; i < shard / 4; i += blockDim.x)
      reinterpret_cast<float4*>(target)[i] = loadClient(source, i);
#else
    for (size_t i = threadIdx.x; i < shard; i += blockDim.x)
      target[i] = source[i];
#endif
    __syncthreads();
  }
#ifndef UCCL_LITE_LOCAL_SERIAL_REFERENCE_FENCE
  __threadfence_system();
#endif
  // As in the original arHierPut: block publication precedes the leader's
  // system-release signal; the receiving leader acquires before its barrier.
  __syncthreads();
  if (!threadIdx.x) peerBarrier(peers, rank, 0, generation);
  __syncthreads();
  auto* local = (peers.data[rank] + bank);
#ifdef UCCL_LITE_LOCAL_AG_COPY_OUTPUT
  if (copyOutput) return;
#endif
  if (Op == 3) {
    // Materialize the reduced shard locally before the next peer-write phase.
    for (size_t i = threadIdx.x; i < shard; i += blockDim.x)
      local[bankElements + rank * shard + i] = sumShard(local, i, shard);
    __syncthreads();
    for (int phase = 1; phase < 4; ++phase) {
      int dest = (rank + phase) % 4;
      for (size_t i = threadIdx.x; i < shard; i += blockDim.x)
        (peers.data[dest] + bank)[bankElements + rank * shard + i] =
            local[bankElements + rank * shard + i];
      __syncthreads();
    }
    __threadfence_system();
    __syncthreads();
    if (!threadIdx.x) peerBarrier(peers, rank, 1, generation);
    __syncthreads();
    for (size_t i = threadIdx.x; i < count; i += blockDim.x)
      output[i] = local[bankElements + i];
  } else {
    size_t extent = Op == 1 ? 4 * count : count;
#ifdef UCCL_LITE_LOCAL_SERIAL_VECTOR
#ifdef UCCL_LITE_LOCAL_SERIAL_SKIP_SELF
    if (Op == 1) {
      // Copy whole source rows: avoid a runtime 64-bit divide/modulo for
      // every vector when selecting the direct self contribution.
      for (int source = 0; source < 4; ++source) {
        auto const* from = source == rank ? input : local + source * shard;
        auto* to = output + source * shard;
        for (size_t i = threadIdx.x; i < shard / 4; i += blockDim.x)
          storeClient(to, i, loadClient(from, i));
      }
    } else
#endif
      for (size_t i = threadIdx.x; i < extent / 4; i += blockDim.x) {
        auto const* quads = reinterpret_cast<float4 const*>(local);
#ifdef UCCL_LITE_LOCAL_SERIAL_SKIP_SELF
        float4 value;
        if (Op == 1) {
          value = i / (shard / 4) == rank ? loadClient(input, i % (shard / 4))
                                          : quads[i];
        } else {
          value = rank == 0 ? loadClient(input, i) : quads[i];
          for (int r = 1; r < 4; ++r)
            value =
                sumQuad(value, r == rank ? loadClient(input + rank * shard, i)
                                         : quads[r * (shard / 4) + i]);
        }
#else
      float4 value = quads[i];
      if (Op == 2)
        for (int r = 1; r < 4; ++r)
          value = sumQuad(value, quads[r * (shard / 4) + i]);
#endif
        storeClient(output, i, value);
      }
#else
    for (size_t i = threadIdx.x; i < extent; i += blockDim.x)
      output[i] = Op == 1 ? local[i] : sumShard(local, i, shard);
#endif
  }
#ifdef UCCL_LITE_LOCAL_SERIAL_CREDIT
  if (Op != 3) return;
#endif
  __syncthreads();
  if (!threadIdx.x) peerBarrier(peers, rank, 2, generation);
}
#endif
__device__ size_t packetBankOffset(size_t usedPackets,
                                   unsigned long long generation) {
#ifdef UCCL_LITE_LOCAL_FIXED_PACKET_BANKS
  // Each supported packet algorithm uses at most 96 KiB per bank.
  return (generation & 1) * ((128UL << 10) / sizeof(mscclpp::LL16Packet));
#else
  return (generation & 1) * usedPackets;
#endif
}
// LL packets combine data and readiness for small messages, as in the
// retained scatterPacketLL path. Caller pointers may be only float-aligned.
__global__ void packetCollect(Peers peers, int rank, int op, float const* input,
                              float* output, size_t count, size_t capacity,
                              unsigned long long generation) {
  mscclpp::LL16Packet* packets[4];
  for (int r = 0; r < 4; ++r)
    packets[r] = reinterpret_cast<mscclpp::LL16Packet*>(
        reinterpret_cast<char*>(peers.data[r]) + 2 * capacity + 4096);
  size_t units = count / 2;
#ifdef UCCL_LITE_LOCAL_PACKET_CREDIT
  // Publishing n+1 follows consuming n on every source. Receiving n+1
  // before beginning n+2 therefore protects the reused parity bank.
  for (int r = 0; r < 4; ++r)
    packets[r] += packetBankOffset(4 * units, generation);
#endif
  uint32_t flag = static_cast<uint32_t>(generation);
  for (size_t i = threadIdx.x; i < units; i += blockDim.x) {
    for (int phase = 1; phase < 4; ++phase) {
      int dest = (rank + phase) % 4;
      size_t offset = 2 * i + (op == 2 ? dest * count : 0);
      packets[dest][rank * units + i].write(__float_as_uint(input[offset]),
                                            __float_as_uint(input[offset + 1]),
                                            flag);
    }
    size_t own = 2 * i + (op == 2 ? rank * count : 0);
    float x = input[own], y = input[own + 1];
    if (op == 1) {
      output[rank * count + 2 * i] = x;
      output[rank * count + 2 * i + 1] = y;
    }
    // Fixed rank order agrees with the FP32 reference reduction order.
    float sx = 0, sy = 0;
    for (int source = 0; source < 4; ++source) {
      uint2 value = source == rank
                        ? make_uint2(__float_as_uint(x), __float_as_uint(y))
                        : packets[rank][source * units + i].read(flag);
      if (op == 1) {
        if (source != rank) {
          output[source * count + 2 * i] = __uint_as_float(value.x);
          output[source * count + 2 * i + 1] = __uint_as_float(value.y);
        }
      } else {
        sx += __uint_as_float(value.x);
        sy += __uint_as_float(value.y);
      }
    }
    if (op != 1) {
      output[2 * i] = sx;
      output[2 * i + 1] = sy;
    }
  }
#ifndef UCCL_LITE_LOCAL_PACKET_CREDIT
  __syncthreads();
  if (!threadIdx.x) peerBarrier(peers, rank, 2, generation);
#endif
}
#ifdef UCCL_LITE_LOCAL_PACKET_SPECIALIZE
#ifndef UCCL_LITE_LOCAL_PACKET_CREDIT
#error "Specialized packets require parity credits"
#endif
template <int Op>
__global__ void packetCollectSpecialized(const __grid_constant__ Peers peers,
                                         int rank, int unusedOp,
                                         float const* input, float* output,
                                         size_t count, size_t capacity,
                                         unsigned long long generation) {
  using Packet = mscclpp::LL16Packet;
  size_t units = count / 2;
  size_t bank = packetBankOffset(4 * units, generation);
  Packet* ownPackets =
      reinterpret_cast<Packet*>(reinterpret_cast<char*>(peers.data[rank]) +
                                2 * capacity + 4096) +
      bank;
  uint32_t flag = static_cast<uint32_t>(generation);
  size_t tid = threadIdx.x, threads = blockDim.x;
#ifdef UCCL_LITE_LOCAL_AG_PACKET_BYTES
  if constexpr (Op == 1) {
    tid += size_t(blockIdx.x) * blockDim.x;
    threads *= gridDim.x;
  }
#endif
  for (size_t i = tid; i < units; i += threads) {
#pragma unroll
    for (int phase = 1; phase < 4; ++phase) {
      int dest = (rank + phase) % 4;
      Packet* target =
          reinterpret_cast<Packet*>(reinterpret_cast<char*>(peers.data[dest]) +
                                    2 * capacity + 4096) +
          bank;
      size_t offset = 2 * i + (Op == 2 ? dest * count : 0);
      target[rank * units + i].write(__float_as_uint(input[offset]),
                                     __float_as_uint(input[offset + 1]), flag);
    }
    size_t own = 2 * i + (Op == 2 ? rank * count : 0);
    float x = input[own], y = input[own + 1];
    if (Op == 1) {
      output[rank * count + 2 * i] = x;
      output[rank * count + 2 * i + 1] = y;
    }
    // Fixed rank order agrees with the FP32 reference reduction order.
    float sx = 0, sy = 0;
#pragma unroll
    for (int source = 0; source < 4; ++source) {
      uint2 value = source == rank
                        ? make_uint2(__float_as_uint(x), __float_as_uint(y))
                        : ownPackets[source * units + i].read(flag);
      if (Op == 1) {
        if (source != rank) {
          output[source * count + 2 * i] = __uint_as_float(value.x);
          output[source * count + 2 * i + 1] = __uint_as_float(value.y);
        }
      } else {
        sx += __uint_as_float(value.x);
        sy += __uint_as_float(value.y);
      }
    }
    if (Op != 1) {
      output[2 * i] = sx;
      output[2 * i + 1] = sy;
    }
  }
#ifndef UCCL_LITE_LOCAL_PACKET_CREDIT
  __syncthreads();
  if (!threadIdx.x) peerBarrier(peers, rank, 2, generation);
#endif
}
#endif

#ifdef UCCL_LITE_LOCAL_HIER_AG
// Two pairwise packet phases: local-pair gather, then cross-pair gather.
// A CTA boundary prevents concurrent publication to different peers.
__global__ void hierarchyGather(Peers peers, int rank, float const* input,
                                float* output, size_t count, size_t capacity,
                                unsigned long long generation) {
  mscclpp::LL16Packet* packets[4];
  size_t units = count / 2;
  for (int r = 0; r < 4; ++r)
    packets[r] =
        reinterpret_cast<mscclpp::LL16Packet*>(
            reinterpret_cast<char*>(peers.data[r]) + 2 * capacity + 4096) +
        packetBankOffset(3 * units, generation);
  uint32_t flag = static_cast<uint32_t>(generation);
  for (size_t i = threadIdx.x; i < units; i += blockDim.x) {
    float x = input[2 * i], y = input[2 * i + 1];
    packets[rank ^ 1][i].write(__float_as_uint(x), __float_as_uint(y), flag);
    uint2 paired = packets[rank][i].read(flag);
    output[rank * count + 2 * i] = x;
    output[rank * count + 2 * i + 1] = y;
    output[(rank ^ 1) * count + 2 * i] = __uint_as_float(paired.x);
    output[(rank ^ 1) * count + 2 * i + 1] = __uint_as_float(paired.y);
  }
  __syncthreads();
  for (size_t i = threadIdx.x; i < units; i += blockDim.x) {
    for (int local = 0; local < 2; ++local) {
      size_t source = size_t((rank & ~1) + local) * count + 2 * i;
      packets[rank ^ 2][(local + 1) * units + i].write(
          __float_as_uint(output[source]), __float_as_uint(output[source + 1]),
          flag);
    }
    for (int local = 0; local < 2; ++local) {
      uint2 value = packets[rank][(local + 1) * units + i].read(flag);
      size_t target = size_t(((rank ^ 2) & ~1) + local) * count + 2 * i;
      output[target] = __uint_as_float(value.x);
      output[target + 1] = __uint_as_float(value.y);
    }
  }
  // Cross-pair n+1 publication requires consumption of n. Two banks protect
  // the lagging rank; operation/size changes use the common rendezvous.
}
#endif
#if defined(UCCL_LITE_LOCAL_HIER_LL) || defined(UCCL_LITE_LOCAL_TREE_LL)
__device__ __forceinline__ float2 loadArPair(float const* input, size_t i) {
#ifdef UCCL_LITE_LOCAL_AR_PAIR_LOAD
  if ((reinterpret_cast<uintptr_t>(input) & 7) == 0)
    return reinterpret_cast<float2 const*>(input)[i];
#endif
  return make_float2(input[2 * i], input[2 * i + 1]);
}
__device__ __forceinline__ void storeArPair(float* output, size_t i, float x,
                                            float y) {
#ifdef UCCL_LITE_LOCAL_AR_PAIR_STORE
  if ((reinterpret_cast<uintptr_t>(output) & 7) == 0) {
    reinterpret_cast<float2*>(output)[i] = make_float2(x, y);
    return;
  }
#endif
  output[2 * i] = x;
  output[2 * i + 1] = y;
}
#endif
#ifdef UCCL_LITE_LOCAL_TREE_LL
#ifndef UCCL_LITE_LOCAL_TREE_MAX_BYTES
#define UCCL_LITE_LOCAL_TREE_MAX_BYTES (8UL << 10)
#endif
static_assert(UCCL_LITE_LOCAL_TREE_MAX_BYTES <= (8UL << 10));
// Original arTreeLL schedule with direct caller IO and parity packet banks.
#ifdef UCCL_LITE_LOCAL_DIRECT_TREE_POINTERS
__global__ void treePacket(
#ifdef UCCL_LITE_LOCAL_TREE_GRID_PARAM
    const __grid_constant__ Peers peers,
#else
    Peers peers,
#endif
    int rank, float const* input, float* output, size_t count, size_t capacity,
    unsigned long long generation) {
  using Packet = mscclpp::LL16Packet;
  size_t units = count / 2;
  size_t bank = packetBankOffset(3 * units, generation);
  Packet* own =
      reinterpret_cast<Packet*>(reinterpret_cast<char*>(peers.data[rank]) +
                                2 * capacity + 4096) +
      bank;
  Packet* intra =
      reinterpret_cast<Packet*>(reinterpret_cast<char*>(peers.data[rank ^ 1]) +
                                2 * capacity + 4096) +
      bank;
  Packet* crossPeer =
      reinterpret_cast<Packet*>(reinterpret_cast<char*>(peers.data[rank ^ 2]) +
                                2 * capacity + 4096) +
      bank;
  uint32_t flag = static_cast<uint32_t>(generation);
  for (size_t i = threadIdx.x; i < units; i += blockDim.x) {
    float2 ownValue = loadArPair(input, i);
    float x = ownValue.x, y = ownValue.y;
    if (rank & 1) {
      intra[i].write(__float_as_uint(x), __float_as_uint(y), flag);
      uint2 result = own[2 * units + i].read(flag);
      storeArPair(output, i, __uint_as_float(result.x),
                  __uint_as_float(result.y));
    } else {
      uint2 pair = own[i].read(flag);
      x += __uint_as_float(pair.x);
      y += __uint_as_float(pair.y);
      crossPeer[units + i].write(__float_as_uint(x), __float_as_uint(y), flag);
      uint2 cross = own[units + i].read(flag);
      x += __uint_as_float(cross.x);
      y += __uint_as_float(cross.y);
      storeArPair(output, i, x, y);
      intra[2 * units + i].write(__float_as_uint(x), __float_as_uint(y), flag);
    }
  }
}
#else
__global__ void treePacket(
#ifdef UCCL_LITE_LOCAL_TREE_GRID_PARAM
    const __grid_constant__ Peers peers,
#else
    Peers peers,
#endif
    int rank, float const* input, float* output, size_t count, size_t capacity,
    unsigned long long generation) {
  using Packet = mscclpp::LL16Packet;
  size_t units = count / 2;
  Packet* packets[4];
  for (int r = 0; r < 4; ++r)
    packets[r] =
        reinterpret_cast<Packet*>(reinterpret_cast<char*>(peers.data[r]) +
                                  2 * capacity + 4096) +
        packetBankOffset(3 * units, generation);
  uint32_t flag = static_cast<uint32_t>(generation);
  for (size_t i = threadIdx.x; i < units; i += blockDim.x) {
    float2 ownValue = loadArPair(input, i);
    float x = ownValue.x, y = ownValue.y;
    if (rank & 1) {
      packets[rank ^ 1][i].write(__float_as_uint(x), __float_as_uint(y), flag);
      uint2 result = packets[rank][2 * units + i].read(flag);
      storeArPair(output, i, __uint_as_float(result.x),
                  __uint_as_float(result.y));
    } else {
      uint2 pair = packets[rank][i].read(flag);
      x += __uint_as_float(pair.x);
      y += __uint_as_float(pair.y);
      packets[rank ^ 2][units + i].write(__float_as_uint(x), __float_as_uint(y),
                                         flag);
      uint2 cross = packets[rank][units + i].read(flag);
      x += __uint_as_float(cross.x);
      y += __uint_as_float(cross.y);
      storeArPair(output, i, x, y);
      packets[rank ^ 1][2 * units + i].write(__float_as_uint(x),
                                             __float_as_uint(y), flag);
    }
  }
}
#endif
#endif
#ifdef UCCL_LITE_LOCAL_HIER_LL
// Port of allreduce.cu::arHierLL: pair reduction, cross-pair reduction,
// then pair gather. Client accesses are scalar to support float alignment.
#ifdef UCCL_LITE_LOCAL_HIER_GRID_PARAM
__global__ void hierarchyPacket(const __grid_constant__ Peers peers, int rank,
                                float const* input, float* output, size_t count,
                                size_t capacity,
                                unsigned long long generation) {
  using Packet = mscclpp::LL16Packet;
  size_t half = count / 2, units = half / 2;
  size_t mine = (rank & 1) * half, other = ((rank & 1) ^ 1) * half;
  size_t bank = packetBankOffset(3 * units, generation);
  Packet* own =
      reinterpret_cast<Packet*>(reinterpret_cast<char*>(peers.data[rank]) +
                                2 * capacity + 4096) +
      bank;
  Packet* intra =
      reinterpret_cast<Packet*>(reinterpret_cast<char*>(peers.data[rank ^ 1]) +
                                2 * capacity + 4096) +
      bank;
  Packet* crossPeer =
      reinterpret_cast<Packet*>(reinterpret_cast<char*>(peers.data[rank ^ 2]) +
                                2 * capacity + 4096) +
      bank;
  uint32_t flag = static_cast<uint32_t>(generation);
  for (size_t i = threadIdx.x; i < units; i += blockDim.x) {
    float2 ownValue = loadArPair(input + mine, i);
    float2 otherValue = loadArPair(input + other, i);
    float x = ownValue.x, y = ownValue.y;
    intra[i].write(__float_as_uint(otherValue.x), __float_as_uint(otherValue.y),
                   flag);
    uint2 pair = own[i].read(flag);
    x += __uint_as_float(pair.x);
    y += __uint_as_float(pair.y);
    crossPeer[units + i].write(__float_as_uint(x), __float_as_uint(y), flag);
    uint2 cross = own[units + i].read(flag);
    x += __uint_as_float(cross.x);
    y += __uint_as_float(cross.y);
    storeArPair(output + mine, i, x, y);
    intra[2 * units + i].write(__float_as_uint(x), __float_as_uint(y), flag);
    uint2 gathered = own[2 * units + i].read(flag);
    storeArPair(output + other, i, __uint_as_float(gathered.x),
                __uint_as_float(gathered.y));
  }
  // The standalone harness supplies a global completion boundary. The ABI
  // must protect packet reuse even when ranks enqueue the next call early.
#ifndef UCCL_LITE_LOCAL_HIER_CREDIT
  __syncthreads();
  if (!threadIdx.x) peerBarrier(peers, rank, 2, generation);
#endif
}
#else
__global__ void hierarchyPacket(Peers peers, int rank, float const* input,
                                float* output, size_t count, size_t capacity,
                                unsigned long long generation) {
  using Packet = mscclpp::LL16Packet;
  Packet* packets[4];
  for (int r = 0; r < 4; ++r)
    packets[r] = reinterpret_cast<Packet*>(
        reinterpret_cast<char*>(peers.data[r]) + 2 * capacity + 4096);
  size_t half = count / 2, units = half / 2;
  size_t mine = (rank & 1) * half, other = ((rank & 1) ^ 1) * half;
#ifdef UCCL_LITE_LOCAL_HIER_CREDIT
  // A rank cannot finish call n+1 without its cross peer publishing n+1,
  // which requires that peer to have consumed n. Two banks therefore protect
  // n while a faster rank starts n+1. Pair phases have the same dependency.
  for (int r = 0; r < 4; ++r)
    packets[r] += packetBankOffset(3 * units, generation);
#endif
  uint32_t flag = static_cast<uint32_t>(generation);
  for (size_t i = threadIdx.x; i < units; i += blockDim.x) {
    float2 ownValue = loadArPair(input + mine, i);
    float2 otherValue = loadArPair(input + other, i);
    float x = ownValue.x, y = ownValue.y;
    packets[rank ^ 1][i].write(__float_as_uint(otherValue.x),
                               __float_as_uint(otherValue.y), flag);
    uint2 pair = packets[rank][i].read(flag);
    x += __uint_as_float(pair.x);
    y += __uint_as_float(pair.y);
    packets[rank ^ 2][units + i].write(__float_as_uint(x), __float_as_uint(y),
                                       flag);
    uint2 cross = packets[rank][units + i].read(flag);
    x += __uint_as_float(cross.x);
    y += __uint_as_float(cross.y);
    storeArPair(output + mine, i, x, y);
    packets[rank ^ 1][2 * units + i].write(__float_as_uint(x),
                                           __float_as_uint(y), flag);
    uint2 gathered = packets[rank][2 * units + i].read(flag);
    storeArPair(output + other, i, __uint_as_float(gathered.x),
                __uint_as_float(gathered.y));
  }
  // The standalone harness supplies a global completion boundary. The ABI
  // must protect packet reuse even when ranks enqueue the next call early.
#ifndef UCCL_LITE_LOCAL_HIER_CREDIT
  __syncthreads();
  if (!threadIdx.x) peerBarrier(peers, rank, 2, generation);
#endif
}
#endif
#endif
#ifdef UCCL_LITE_LOCAL_SMALL_DEVICE_EPOCH
#if defined(UCCL_LITE_LOCAL_FIXED_PACKET_BANKS) || \
    !defined(UCCL_LITE_LOCAL_HIER_GRID_PARAM) ||   \
    !defined(UCCL_LITE_LOCAL_DIRECT_TREE_POINTERS)
#error "Device epochs require direct, compact packet kernels"
#endif
static_assert(UCCL_LITE_LOCAL_HIER_LL_BYTES <= 16384 &&
              UCCL_LITE_LOCAL_HIER_AG_MAX_BYTES <= 16384);
__global__ void treePacketEpoch(const __grid_constant__ Peers peers, int rank,
                                float const* input, float* output, size_t count,
                                size_t capacity, unsigned long long* sequence) {
  __shared__ unsigned long long generation;
  if (!threadIdx.x) generation = ++*sequence;
  __syncthreads();

  using Packet = mscclpp::LL16Packet;
  size_t units = count / 2;
  size_t bank = packetBankOffset(3 * units, generation);
  Packet* own =
      reinterpret_cast<Packet*>(reinterpret_cast<char*>(peers.data[rank]) +
                                2 * capacity + 4096 + kPacketEpochOffset) +
      bank;
  Packet* intra =
      reinterpret_cast<Packet*>(reinterpret_cast<char*>(peers.data[rank ^ 1]) +
                                2 * capacity + 4096 + kPacketEpochOffset) +
      bank;
  Packet* crossPeer =
      reinterpret_cast<Packet*>(reinterpret_cast<char*>(peers.data[rank ^ 2]) +
                                2 * capacity + 4096 + kPacketEpochOffset) +
      bank;
  uint32_t flag = static_cast<uint32_t>(generation);
  for (size_t i = threadIdx.x; i < units; i += blockDim.x) {
    float2 ownValue = loadArPair(input, i);
    float x = ownValue.x, y = ownValue.y;
    if (rank & 1) {
      intra[i].write(__float_as_uint(x), __float_as_uint(y), flag);
      uint2 result = own[2 * units + i].read(flag);
      storeArPair(output, i, __uint_as_float(result.x),
                  __uint_as_float(result.y));
    } else {
      uint2 pair = own[i].read(flag);
      x += __uint_as_float(pair.x);
      y += __uint_as_float(pair.y);
      crossPeer[units + i].write(__float_as_uint(x), __float_as_uint(y), flag);
      uint2 cross = own[units + i].read(flag);
      x += __uint_as_float(cross.x);
      y += __uint_as_float(cross.y);
      storeArPair(output, i, x, y);
      intra[2 * units + i].write(__float_as_uint(x), __float_as_uint(y), flag);
    }
  }
}
__global__ void hierarchyPacketEpoch(const __grid_constant__ Peers peers,
                                     int rank, float const* input,
                                     float* output, size_t count,
                                     size_t capacity,
                                     unsigned long long* sequence) {
  __shared__ unsigned long long generation;
  if (!threadIdx.x) generation = ++*sequence;
  __syncthreads();

  using Packet = mscclpp::LL16Packet;
  size_t half = count / 2, units = half / 2;
  size_t mine = (rank & 1) * half, other = ((rank & 1) ^ 1) * half;
  size_t bank = packetBankOffset(3 * units, generation);
  Packet* own =
      reinterpret_cast<Packet*>(reinterpret_cast<char*>(peers.data[rank]) +
                                2 * capacity + 4096 + kPacketEpochOffset) +
      bank;
  Packet* intra =
      reinterpret_cast<Packet*>(reinterpret_cast<char*>(peers.data[rank ^ 1]) +
                                2 * capacity + 4096 + kPacketEpochOffset) +
      bank;
  Packet* crossPeer =
      reinterpret_cast<Packet*>(reinterpret_cast<char*>(peers.data[rank ^ 2]) +
                                2 * capacity + 4096 + kPacketEpochOffset) +
      bank;
  uint32_t flag = static_cast<uint32_t>(generation);
  for (size_t i = threadIdx.x; i < units; i += blockDim.x) {
    float2 ownValue = loadArPair(input + mine, i);
    float2 otherValue = loadArPair(input + other, i);
    float x = ownValue.x, y = ownValue.y;
    intra[i].write(__float_as_uint(otherValue.x), __float_as_uint(otherValue.y),
                   flag);
    uint2 pair = own[i].read(flag);
    x += __uint_as_float(pair.x);
    y += __uint_as_float(pair.y);
    crossPeer[units + i].write(__float_as_uint(x), __float_as_uint(y), flag);
    uint2 cross = own[units + i].read(flag);
    x += __uint_as_float(cross.x);
    y += __uint_as_float(cross.y);
    storeArPair(output + mine, i, x, y);
    intra[2 * units + i].write(__float_as_uint(x), __float_as_uint(y), flag);
    uint2 gathered = own[2 * units + i].read(flag);
    storeArPair(output + other, i, __uint_as_float(gathered.x),
                __uint_as_float(gathered.y));
  }
  // The standalone harness supplies a global completion boundary. The ABI
  // must protect packet reuse even when ranks enqueue the next call early.
#ifndef UCCL_LITE_LOCAL_HIER_CREDIT
  __syncthreads();
  if (!threadIdx.x) peerBarrier(peers, rank, 2, generation);
#endif
}
#endif
#ifdef UCCL_LITE_LOCAL_HIER_SM
__device__ mscclpp::DeviceSyncer hierarchySync;
__device__ void hierarchyExchange(UCCL_LITE_LOCAL_PEER_REF peers, int rank,
                                  int peer, int phase,
                                  unsigned long long generation) {
  // The retained arHierPut uses DeviceSyncer's release/acquire grid
  // boundary followed by the leader's system-release signal. Keep the
  // stronger per-writer system fence as the comparison configuration.
#ifndef UCCL_LITE_LOCAL_HIER_REFERENCE_FENCE
  __threadfence_system();
#endif
  hierarchySync.sync(gridDim.x);
  if (!blockIdx.x && !threadIdx.x) {
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_system> remote(
        peers.flags[peer][64 + phase * 4 + rank]);
    remote.store(generation, cuda::memory_order_release);
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_system> incoming(
        peers.flags[rank][64 + phase * 4 + peer]);
    while (incoming.load(cuda::memory_order_acquire) < generation) {
    }
  }
  hierarchySync.sync(gridDim.x);
}
// Adapted from arHierPut: vectorized pair RS, cross-pair AR, pair AG.
// A grid boundary between phases prevents overlapping remote relationships.
__global__ void hierarchySm(
#ifdef UCCL_LITE_LOCAL_GRID_PEERS
    const __grid_constant__ Peers peers,
#else
    Peers peers,
#endif
    int rank, float const* input, float* output, size_t count,
    unsigned long long generation
#ifdef UCCL_LITE_LOCAL_SM_BANKS
    ,
    bool banked
#endif
) {
  size_t half = count / 2, quads = half / 4;
  size_t tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t threads = size_t(gridDim.x) * blockDim.x;
  size_t mine = (rank & 1) * half, other = ((rank & 1) ^ 1) * half;
  size_t bank = 0;
#ifdef UCCL_LITE_LOCAL_SM_BANKS
  // Each call uses 2*count floats. Pair/cross/pair dependencies prevent a
  // rank from reaching call n+2 before its peers consumed call n's bank.
  // Size and algorithm transitions rendezvous before changing this layout.
  if (banked) bank = (generation & 1) * 2 * count;
#endif
  auto* local = reinterpret_cast<float4*>(peers.data[rank] + bank);
  auto* intra = reinterpret_cast<float4*>(peers.data[rank ^ 1] + bank);
  auto* cross = reinterpret_cast<float4*>(peers.data[rank ^ 2] + bank);
  for (size_t i = tid; i < quads; i += threads)
    intra[i] = loadClient(input + other, i);
  hierarchyExchange(peers, rank, rank ^ 1, 0, generation);
  for (size_t i = tid; i < quads; i += threads) {
    float4 partial = sumQuad(loadClient(input + mine, i), local[i]);
    local[3 * quads + i] = partial;
    cross[quads + i] = partial;
  }
  hierarchyExchange(peers, rank, rank ^ 2, 1, generation);
  for (size_t i = tid; i < quads; i += threads) {
    float4 result = sumQuad(local[3 * quads + i], local[quads + i]);
    storeClient(output + mine, i, result);
    intra[2 * quads + i] = result;
  }
  hierarchyExchange(peers, rank, rank ^ 1, 2, generation);
  for (size_t i = tid; i < quads; i += threads)
    storeClient(output + other, i, local[2 * quads + i]);
#ifdef UCCL_LITE_LOCAL_SM_BANKS
  if (banked) return;
#endif
  hierarchySync.sync(gridDim.x);
  if (!tid) peerBarrier(peers, rank, 2, generation);
  hierarchySync.sync(gridDim.x);
}
#endif
#ifdef UCCL_LITE_LOCAL_HOST_AG
struct HostGather {
#ifdef UCCL_LITE_LOCAL_HOST_REPLICA
#ifndef UCCL_LITE_LOCAL_HOST_REPLICA_SLOTS
#define UCCL_LITE_LOCAL_HOST_REPLICA_SLOTS 2
#endif
  static constexpr int slots = UCCL_LITE_LOCAL_HOST_REPLICA_SLOTS;
  static_assert(slots >= 2 && slots <= multinode::SharedHostPackets::slots - 2);
#elif defined(UCCL_LITE_LOCAL_HOST_AG_WIDE)
  static constexpr int slots = 2;
#else
  static constexpr int slots = multinode::SharedHostPackets::slots;
#endif
#ifdef UCCL_LITE_LOCAL_HOST_REPLICA
  static constexpr int rowsPerGroup = 4;
  float* replicas[slots][2];
#else
  static constexpr int rowsPerGroup = 2;
#endif
  float* rows[slots][4];
  uint64_t* flags[slots][2];
  size_t elements;
};
__device__ mscclpp::DeviceSyncer hostGatherSync;
__device__ uint64_t hostFlagLoad(uint64_t* flag) {
  cuda::atomic_ref<uint64_t, cuda::thread_scope_system> value(*flag);
  return value.load(cuda::memory_order_acquire);
}
__device__ void hostFlagStore(uint64_t* flag, uint64_t generation) {
  cuda::atomic_ref<uint64_t, cuda::thread_scope_system> value(*flag);
  value.store(generation, cuda::memory_order_release);
}
// miniHostGather's upload/read/credit protocol, tiled into the shared host
// packet storage already used by the multi-node one-GPU-per-process backend.
__global__ void hostGatherKernel(HostGather host, int rank, float const* input,
                                 float* output, size_t count, uint64_t base) {
  size_t tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t threads = size_t(gridDim.x) * blockDim.x;
  for (size_t offset = 0, tile = 0; offset < count;
       offset += host.elements, ++tile) {
    uint64_t generation = base + tile + 1;
    int slot = (generation - 1) % HostGather::slots;
    size_t length =
        host.elements < count - offset ? host.elements : count - offset;
    if (!tid && generation > HostGather::slots) {
#ifdef UCCL_LITE_LOCAL_BATCH_ACQUIRE
      for (int reader = 0; reader < 4; ++reader)
        while (
            loadRelaxedGlobal(host.flags[slot][rank / 2] + (4 + reader) * 8) <
            generation - HostGather::slots)
          __nanosleep(64);
      __threadfence_system();
#else
      for (int reader = 0; reader < 4; ++reader)
        while (hostFlagLoad(host.flags[slot][rank / 2] + (4 + reader) * 8) <
               generation - HostGather::slots)
          __nanosleep(64);
#endif
    }
    hostGatherSync.sync(gridDim.x);
    for (size_t i = tid; i < length / 4; i += threads)
      reinterpret_cast<float4*>(host.rows[slot][rank])[i] =
          loadClient(input + offset, i);
    __threadfence_system();
    hostGatherSync.sync(gridDim.x);
    if (!tid) hostFlagStore(host.flags[slot][rank / 2] + rank * 8, generation);
    int source = blockIdx.x % 4;
    if (!threadIdx.x && source != rank)
      while (hostFlagLoad(host.flags[slot][source / 2] + source * 8) <
             generation)
        __nanosleep(64);
    __syncthreads();
    size_t lane = (blockIdx.x / 4) * blockDim.x + threadIdx.x;
    size_t stride = (gridDim.x / 4) * blockDim.x;
    auto const* from =
        source == rank ? input + offset : host.rows[slot][source];
    auto* to = output + size_t(source) * count + offset;
    for (size_t i = lane; i < length / 4; i += stride)
      storeClient(to, i, loadClient(from, i));
    hostGatherSync.sync(gridDim.x);
    if (!tid) {
#ifdef UCCL_LITE_LOCAL_BATCH_SIGNAL
      __threadfence_system();
      for (int group = 0; group < 2; ++group)
        signalRelaxedGlobal(host.flags[slot][group] + (4 + rank) * 8,
                            generation);
#else
      for (int group = 0; group < 2; ++group)
        hostFlagStore(host.flags[slot][group] + (4 + rank) * 8, generation);
#endif
    }
    // All next-tile writers wait for these reader credits before reuse.
  }
}
#ifdef UCCL_LITE_LOCAL_HOST_REPLICA
struct ReplicaCounters {
#ifdef UCCL_LITE_LOCAL_HOST_GPU_GATES
  uint64_t uploadGate[HostGather::slots], sourceGate[HostGather::slots][4];
#endif
  unsigned uploaded[HostGather::slots];
#ifdef UCCL_LITE_LOCAL_HOST_PER_SOURCE
  unsigned consumed[HostGather::slots][4];
#else
  unsigned consumed[HostGather::slots];
#endif
};
__global__ void replicaGatherKernel(HostGather host, ReplicaCounters* counters,
                                    int rank, float const* input, float* output,
                                    size_t count, uint64_t base) {
#ifdef UCCL_LITE_LOCAL_HOST_GPU_PROGRESS
  constexpr int uploadBlocks = 32, consumeBlocks = 96;
#else
  constexpr int uploadBlocks = 8, consumeBlocks = 32;
#endif
#ifdef UCCL_LITE_LOCAL_HOST_PER_SOURCE
  constexpr int credits = 4;
#else
  constexpr int credits = 1;
#endif
  if (!blockIdx.x && !threadIdx.x)
    for (int slot = 0; slot < HostGather::slots; ++slot) {
      counters->uploaded[slot] = 0;
#ifdef UCCL_LITE_LOCAL_HOST_GPU_GATES
      counters->uploadGate[slot] = 0;
      for (int source = 0; source < 4; ++source)
        counters->sourceGate[slot][source] = 0;
#endif
#ifdef UCCL_LITE_LOCAL_HOST_PER_SOURCE
      for (int source = 0; source < 4; ++source)
        counters->consumed[slot][source] = 0;
#else
      counters->consumed[slot] = 0;
#endif
    }
  cooperative_groups::this_grid().sync();
  size_t pieces = (count + host.elements - 1) / host.elements;
  if (!blockIdx.x) {
    if (!threadIdx.x) {
      size_t upload[HostGather::slots], consume[HostGather::slots][credits];
      for (int slot = 0; slot < HostGather::slots; ++slot) {
        upload[slot] = (slot + HostGather::slots - base % HostGather::slots) %
                       HostGather::slots;
        for (int source = 0; source < credits; ++source)
          consume[slot][source] = upload[slot];
      }
      size_t complete = 0;
      while (complete < pieces * credits) {
        for (int slot = 0; slot < HostGather::slots; ++slot) {
#ifdef UCCL_LITE_LOCAL_HOST_BATCH_PUBLISH
#if !defined(UCCL_LITE_LOCAL_HOST_PER_SOURCE) || \
    !defined(UCCL_LITE_LOCAL_BATCH_SIGNAL)
#error \
    "Batched host publication requires per-source credits and system-release stores"
#endif
          unsigned pending = 0;
#endif
          cuda::atomic_ref<unsigned, cuda::thread_scope_device> produced(
              counters->uploaded[slot]);
          if (upload[slot] < pieces &&
              produced.load(cuda::memory_order_acquire) == uploadBlocks) {
            produced.store(0, cuda::memory_order_relaxed);
#ifdef UCCL_LITE_LOCAL_HOST_BATCH_PUBLISH
            pending |= 1;
#else
            hostFlagStore(host.flags[slot][rank / 2] + rank * 8,
                          base + upload[slot] + 1);
#endif
            upload[slot] += HostGather::slots;
          }
          for (int source = 0; source < credits; ++source) {
#ifdef UCCL_LITE_LOCAL_HOST_PER_SOURCE
            unsigned& consumed = counters->consumed[slot][source];
#else
            unsigned& consumed = counters->consumed[slot];
#endif
            cuda::atomic_ref<unsigned, cuda::thread_scope_device> received(
                consumed);
            if (consume[slot][source] < pieces &&
                received.load(cuda::memory_order_acquire) ==
                    consumeBlocks / credits) {
              received.store(0, cuda::memory_order_relaxed);
#ifdef UCCL_LITE_LOCAL_HOST_PER_SOURCE
#ifdef UCCL_LITE_LOCAL_HOST_BATCH_PUBLISH
              pending |= 2u << source;
#else
              hostFlagStore(
                  host.flags[slot][source / 2] + (4 + source * 4 + rank) * 8,
                  base + consume[slot][source] + 1);
#endif
#else
              for (int group = 0; group < 2; ++group)
                hostFlagStore(host.flags[slot][group] + (4 + rank) * 8,
                              base + consume[slot][source] + 1);
#endif
              consume[slot][source] += HostGather::slots;
              ++complete;
            }
          }
#ifdef UCCL_LITE_LOCAL_HOST_BATCH_PUBLISH
          if (pending) {
            // Every counter was acquired before this release fence. Publish
            // only those source/reader conditions already known complete.
            __threadfence_system();
            if (pending & 1)
              signalRelaxedGlobal(host.flags[slot][rank / 2] + rank * 8,
                                  base + upload[slot] - HostGather::slots + 1);
            for (int source = 0; source < credits; ++source)
              if (pending & (2u << source))
                signalRelaxedGlobal(
                    host.flags[slot][source / 2] + (4 + source * 4 + rank) * 8,
                    base + consume[slot][source] - HostGather::slots + 1);
          }
#endif
        }
      }
    }
  } else if (blockIdx.x <= uploadBlocks) {
    size_t lane = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
    for (size_t piece = 0; piece < pieces; ++piece) {
      uint64_t epoch = base + piece + 1;
      int slot = (epoch - 1) % HostGather::slots;
      int creditBase = 4;
#ifdef UCCL_LITE_LOCAL_HOST_PER_SOURCE
      creditBase += rank * 4;
#endif
#ifdef UCCL_LITE_LOCAL_HOST_GPU_GATES
      if (blockIdx.x == 1) {
        if (threadIdx.x < 4 && epoch > HostGather::slots)
          while (hostFlagLoad(host.flags[slot][rank / 2] +
                              (creditBase + threadIdx.x) * 8) <
                 epoch - HostGather::slots)
            __nanosleep(64);
        __syncthreads();
        if (!threadIdx.x)
          cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(
              counters->uploadGate[slot])
              .store(epoch, cuda::memory_order_release);
      }
      if (!threadIdx.x)
        while (cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(
                   counters->uploadGate[slot])
                   .load(cuda::memory_order_acquire) < epoch)
          __nanosleep(64);
      __syncthreads();
#else
      if (!threadIdx.x && epoch > HostGather::slots)
        for (int reader = 0; reader < 4; ++reader)
          while (hostFlagLoad(host.flags[slot][rank / 2] +
                              (creditBase + reader) * 8) <
                 epoch - HostGather::slots)
            __nanosleep(64);
      __syncthreads();
#endif
      size_t offset = piece * host.elements;
      size_t length =
          host.elements < count - offset ? host.elements : count - offset;
      for (size_t i = lane; i < length / 4; i += uploadBlocks * blockDim.x)
        reinterpret_cast<float4*>(host.rows[slot][rank])[i] =
            loadClient(input + offset, i);
      __threadfence_system();
      __syncthreads();
      if (!threadIdx.x)
        cuda::atomic_ref<unsigned, cuda::thread_scope_device>(
            counters->uploaded[slot])
            .fetch_add(1, cuda::memory_order_release);
    }
  } else {
    int consumer = blockIdx.x - 1 - uploadBlocks;
    int source = consumer % 4;
    size_t lane = (consumer / 4) * blockDim.x + threadIdx.x;
    for (size_t piece = 0; piece < pieces; ++piece) {
      uint64_t epoch = base + piece + 1;
      int slot = (epoch - 1) % HostGather::slots;
#ifdef UCCL_LITE_LOCAL_HOST_GPU_PROGRESS
      bool remote = false;  // Read the original NUMA-bound source rows.
#else
      bool remote = source / 2 != rank / 2;
#endif
#ifdef UCCL_LITE_LOCAL_HOST_PER_SOURCE
      constexpr int replicaReady = 20;
#else
      constexpr int replicaReady = 10;
#endif
#ifdef UCCL_LITE_LOCAL_HOST_GPU_GATES
#ifndef UCCL_LITE_LOCAL_HOST_GPU_PROGRESS
#error "GPU readiness relay is for direct host-source progress"
#endif
      if (!threadIdx.x) {
        if (consumer < 4) {
          while (hostFlagLoad(host.flags[slot][source / 2] + source * 8) <
                 epoch)
            __nanosleep(64);
          cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(
              counters->sourceGate[slot][source])
              .store(epoch, cuda::memory_order_release);
        } else {
          while (cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(
                     counters->sourceGate[slot][source])
                     .load(cuda::memory_order_acquire) < epoch)
            __nanosleep(64);
        }
      }
#else
      if (!threadIdx.x) {
        if (remote) {
          for (int part = 0; part < LocalHostCopy::parts; ++part)
            while (hostFlagLoad(host.flags[slot][rank / 2] +
                                (replicaReady +
                                 (source % 2) * LocalHostCopy::parts + part) *
                                    8) < epoch)
              __nanosleep(64);
        } else {
          while (hostFlagLoad(host.flags[slot][source / 2] + source * 8) <
                 epoch)
            __nanosleep(64);
        }
      }
#endif
      __syncthreads();
      size_t offset = piece * host.elements;
      size_t length =
          host.elements < count - offset ? host.elements : count - offset;
      auto const* from = source == rank ? input + offset
                         : remote       ? host.replicas[slot][source % 2]
                                        : host.rows[slot][source];
      auto* to = output + size_t(source) * count + offset;
#ifndef UCCL_LITE_LOCAL_HOST_COPY_SELF
      if (to != from)
#endif
        for (size_t i = lane; i < length / 4;
             i += (consumeBlocks / 4) * blockDim.x)
          storeClient(to, i, loadClient(from, i));
      __syncthreads();
#ifdef UCCL_LITE_LOCAL_HOST_PER_SOURCE
      unsigned& consumed = counters->consumed[slot][source];
#else
      unsigned& consumed = counters->consumed[slot];
#endif
      if (!threadIdx.x)
        cuda::atomic_ref<unsigned, cuda::thread_scope_device>(consumed)
            .fetch_add(1, cuda::memory_order_release);
    }
  }
}
#endif
#endif
#ifdef UCCL_LITE_LOCAL_RING
constexpr int kRingTiles = 32;
__global__ void ringSignal(unsigned long long* flag, unsigned long long epoch) {
  cuda::atomic_ref<unsigned long long, cuda::thread_scope_system>(*flag).store(
      epoch, cuda::memory_order_release);
}
__global__ void ringWait(unsigned long long* flag, unsigned long long epoch) {
  cuda::atomic_ref<unsigned long long, cuda::thread_scope_system> value(*flag);
  while (value.load(cuda::memory_order_acquire) < epoch) __nanosleep(64);
}
__global__ void ringSum(float const* input, float const* incoming,
                        float* output, size_t count) {
#ifdef UCCL_LITE_LOCAL_RING_VECTOR
#ifndef UCCL_LITE_LOCAL_SERIAL_VECTOR
#error "Vector ring requires aligned and unaligned client helpers"
#endif
  size_t tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t threads = size_t(gridDim.x) * blockDim.x;
  for (size_t i = tid; i < count / 4; i += threads)
    storeClient(output, i,
                sumQuad(loadClient(input, i), loadClient(incoming, i)));
  for (size_t i = (count / 4) * 4 + tid; i < count; i += threads)
    output[i] = input[i] + incoming[i];
#else
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x)
    output[i] = input[i] + incoming[i];
#endif
}
#ifdef UCCL_LITE_LOCAL_RS_MEMOP
__global__ void ringWaitSum(float const* input, float const* incoming,
                            float* output, size_t count,
                            unsigned long long* flag,
                            unsigned long long generation) {
  if (!threadIdx.x) {
#ifdef UCCL_LITE_LOCAL_RS_MEMOP32
    cuda::atomic_ref<unsigned, cuda::thread_scope_system> ready(
        *reinterpret_cast<unsigned*>(flag));
    while (ready.load(cuda::memory_order_acquire) <
           static_cast<unsigned>(generation))
      __nanosleep(64);
#else
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_system> ready(
        *flag);
    while (ready.load(cuda::memory_order_acquire) < generation) __nanosleep(64);
#endif
  }
  __syncthreads();
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x)
    output[i] = input[i] + incoming[i];
}
#endif
#ifdef UCCL_LITE_LOCAL_RS_HOST_READY
#ifndef UCCL_LITE_LOCAL_RS_NATIVE_CHUNK_BYTES
#define UCCL_LITE_LOCAL_RS_NATIVE_CHUNK_BYTES (8UL << 20)
#endif
static_assert(UCCL_LITE_LOCAL_RS_NATIVE_CHUNK_BYTES >= (8UL << 20) &&
              UCCL_LITE_LOCAL_RS_NATIVE_CHUNK_BYTES <= (16UL << 20) &&
              UCCL_LITE_LOCAL_RS_NATIVE_CHUNK_BYTES % (1UL << 20) == 0);
constexpr size_t kNativeRingChunkBytes = UCCL_LITE_LOCAL_RS_NATIVE_CHUNK_BYTES;
#ifndef UCCL_LITE_LOCAL_RS_NATIVE_FINAL_CHUNK_BYTES
#define UCCL_LITE_LOCAL_RS_NATIVE_FINAL_CHUNK_BYTES \
  UCCL_LITE_LOCAL_RS_NATIVE_CHUNK_BYTES
#endif
static_assert(UCCL_LITE_LOCAL_RS_NATIVE_FINAL_CHUNK_BYTES >= (8UL << 20) &&
              UCCL_LITE_LOCAL_RS_NATIVE_FINAL_CHUNK_BYTES <= (16UL << 20) &&
              UCCL_LITE_LOCAL_RS_NATIVE_FINAL_CHUNK_BYTES % (1UL << 20) == 0);
constexpr size_t kNativeRingFinalChunkBytes =
    UCCL_LITE_LOCAL_RS_NATIVE_FINAL_CHUNK_BYTES;
#ifndef UCCL_LITE_LOCAL_RS_HOST_READY_MIN_BYTES
#define UCCL_LITE_LOCAL_RS_HOST_READY_MIN_BYTES (128UL << 20)
#endif
#if !defined(UCCL_LITE_LOCAL_AG_HOST_READY) ||                               \
    !defined(UCCL_LITE_LOCAL_RING_IPC_EVENTS) ||                             \
    defined(UCCL_LITE_LOCAL_RS_MEMOP) || defined(UCCL_LITE_LOCAL_RS_PULL) || \
    defined(UCCL_LITE_LOCAL_RS_BANKS) || defined(UCCL_LITE_LOCAL_HIER_RS)
#error "Host-ready RS requires the isolated shared-control configuration"
#endif
#ifdef UCCL_LITE_LOCAL_RS_GPU_NATIVE
#ifndef UCCL_LITE_LOCAL_RS_GPU_READY
#error "Native copy publication requires producer-local GPU readiness"
#endif
#endif
__global__ void ringHostWaitSum(float const* input, float const* incoming,
                                float* output, size_t count,
#ifdef UCCL_LITE_LOCAL_RS_GPU_NATIVE
                                unsigned* produced,
#endif
                                unsigned* sourceReady, unsigned long long* gate,
                                unsigned generation, unsigned long long token) {
  if (!blockIdx.x && !threadIdx.x) {
#ifdef UCCL_LITE_LOCAL_RS_GPU_NATIVE
    // The local copy-event wait precedes this kernel. Publish its completed
    // peer payload before waiting on the predecessor, so every rank can
    // advance independently of receiving the same phase's data.
    cuda::atomic_ref<unsigned, cuda::thread_scope_system>(*produced).store(
        generation, cuda::memory_order_release);
#endif
    cuda::atomic_ref<unsigned, cuda::thread_scope_system> ready(*sourceReady);
    while (ready.load(cuda::memory_order_acquire) < generation) __nanosleep(64);
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>(*gate)
        .store(token, cuda::memory_order_release);
  }
  if (!threadIdx.x) {
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> arrived(
        *gate);
    while (arrived.load(cuda::memory_order_acquire) < token) __nanosleep(64);
  }
  __syncthreads();
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x)
    output[i] = input[i] + incoming[i];
}
#ifdef UCCL_LITE_LOCAL_RS_NATIVE_BANKS
#if !defined(UCCL_LITE_LOCAL_RS_NATIVE_GRAPH) || \
    !defined(UCCL_LITE_LOCAL_HIER_CREDIT)
#error "Native receive banks require graph readiness and transition credits"
#endif
#endif
#ifdef UCCL_LITE_LOCAL_RS_NATIVE_GRAPH
#if !defined(UCCL_LITE_LOCAL_RS_GPU_NATIVE_TAIL) || \
    defined(UCCL_LITE_LOCAL_RS_GRAPHS)
#error "Native graph requires the isolated native split-tail readiness path"
#endif
#ifdef UCCL_LITE_LOCAL_RS_DEVICE_EPOCH
#if !defined(UCCL_LITE_LOCAL_RS_GPU_NATIVE) || \
    !defined(UCCL_LITE_LOCAL_RS_GPU_READY) ||  \
    defined(UCCL_LITE_LOCAL_RS_NATIVE_BANKS)
#error "Device RS epochs require the single-bank native readiness protocol"
#endif
// Independent RS epochs must not share retirement words with global epochs.
// Producer flags use 256..447, the local gate 449, and the counter 450.
constexpr int kNativeRetirePhase = 116;
static_assert(4 * kNativeRetirePhase + 3 < 4096 / sizeof(unsigned long long));
__global__ void ringGraphEpoch(unsigned* epoch) { ++*epoch; }
__global__ void ringGraphRetire(Peers peers, int rank, unsigned* epoch) {
  peerBarrier(peers, rank, kNativeRetirePhase, *epoch);
}
#else
__global__ void ringGraphEpoch(unsigned* epoch, unsigned generation) {
  *epoch = generation;
}
__global__ void ringGraphRetire(Peers peers, int rank, unsigned* epoch) {
  peerBarrier(peers, rank, 2, *epoch);
}
#endif
__global__ void ringGraphWaitSum(float const* input, float const* incoming,
                                 float* output, size_t count,
#ifdef UCCL_LITE_LOCAL_RS_GPU_NATIVE
                                 unsigned* produced,
#endif
                                 unsigned* sourceReady,
                                 unsigned long long* gate, unsigned* epoch,
                                 size_t index, size_t stride) {
  unsigned generation = *epoch;
  unsigned long long token =
      static_cast<unsigned long long>(generation) * stride + index;
  if (!blockIdx.x && !threadIdx.x) {
#ifdef UCCL_LITE_LOCAL_RS_GPU_NATIVE
    // The local copy-event wait precedes this kernel. Publish its completed
    // peer payload before waiting on the predecessor, so every rank can
    // advance independently of receiving the same phase's data.
    cuda::atomic_ref<unsigned, cuda::thread_scope_system>(*produced).store(
        generation, cuda::memory_order_release);
#endif
    cuda::atomic_ref<unsigned, cuda::thread_scope_system> ready(*sourceReady);
    while (ready.load(cuda::memory_order_acquire) < generation) __nanosleep(64);
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>(*gate)
        .store(token, cuda::memory_order_release);
  }
  if (!threadIdx.x) {
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> arrived(
        *gate);
    while (arrived.load(cuda::memory_order_acquire) < token) __nanosleep(64);
  }
  __syncthreads();
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x)
    output[i] = input[i] + incoming[i];
}
#endif

#endif
__global__ void ringCopy(float const* input, float* output, size_t count) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x)
    output[i] = input[i];
}
#endif
}  // namespace
#if defined(UCCL_LITE_LOCAL_AG_GRAPH_OUTPUT) || \
    defined(UCCL_LITE_LOCAL_SMALL_GRAPH)
struct LocalCopyGraph {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  cudaGraphNode_t kernel = nullptr, copy = nullptr;
  void* boundOutput = nullptr;
  void const* boundInput = nullptr;
  unsigned long long allocation = 0;
  bool valid = false;
  ~LocalCopyGraph() {
    if (executable) cudaGraphExecDestroy(executable);
    if (graph) cudaGraphDestroy(graph);
  }
};
#endif
#ifdef UCCL_LITE_LOCAL_RS_GRAPHS
#ifndef UCCL_LITE_LOCAL_RS_GRAPH_MIN_BYTES
#define UCCL_LITE_LOCAL_RS_GRAPH_MIN_BYTES (128UL << 20)
#endif
static_assert(UCCL_LITE_LOCAL_RS_GRAPH_MIN_BYTES >= (8UL << 20));
struct LocalRingGraphs {
  struct Phase {
    cudaGraph_t send = nullptr, receive = nullptr;
    cudaGraphExec_t sendExec = nullptr, receiveExec = nullptr;
    std::vector<cudaGraphNode_t> copies, kernels;
    ~Phase() {
      if (sendExec) cudaGraphExecDestroy(sendExec);
      if (receiveExec) cudaGraphExecDestroy(receiveExec);
      if (send) cudaGraphDestroy(send);
      if (receive) cudaGraphDestroy(receive);
    }
  };
  std::array<Phase, 3> phases;
  void const* input = nullptr;
  void* output = nullptr;
  unsigned long long allocation = 0;
};
#endif
#ifdef UCCL_LITE_LOCAL_RS_HOST_READY
struct NativeRingLayout {
  size_t shard;
  std::array<size_t, 3> chunk{}, tiles{};
#ifdef UCCL_LITE_LOCAL_RS_GPU_NATIVE_TAIL
  static constexpr size_t stride = 2 * kRingTiles;
#else
  static constexpr size_t stride = kRingTiles;
#endif
  bool splitFinal = false;
  explicit NativeRingLayout(size_t elements) : shard(elements) {
    if (!shard) throw std::invalid_argument("empty readiness ring");
    for (int step = 0; step < 3; ++step) {
      size_t bytes =
          step == 2 ? kNativeRingFinalChunkBytes : kNativeRingChunkBytes;
#ifdef UCCL_LITE_LOCAL_RS_NATIVE_LARGE_CHUNK_BYTES
      static_assert(
          UCCL_LITE_LOCAL_RS_NATIVE_LARGE_CHUNK_BYTES >= (8UL << 20) &&
          UCCL_LITE_LOCAL_RS_NATIVE_LARGE_CHUNK_BYTES <= (16UL << 20) &&
          UCCL_LITE_LOCAL_RS_NATIVE_LARGE_CHUNK_BYTES % (1UL << 20) == 0);
#ifndef UCCL_LITE_LOCAL_RS_NATIVE_LARGE_MIN_BYTES
#define UCCL_LITE_LOCAL_RS_NATIVE_LARGE_MIN_BYTES (1UL << 30)
#endif
      if (step < 2 && shard * 16 >= UCCL_LITE_LOCAL_RS_NATIVE_LARGE_MIN_BYTES)
        bytes = UCCL_LITE_LOCAL_RS_NATIVE_LARGE_CHUNK_BYTES;
#endif
      chunk[step] = std::min(shard, bytes / 4);
      tiles[step] = (shard + chunk[step] - 1) / chunk[step];
    }
#ifdef UCCL_LITE_LOCAL_RS_GPU_NATIVE_TAIL
    splitFinal = shard - (tiles[2] - 1) * chunk[2] > (1UL << 18);
#endif
    for (int step = 0; step < 3; ++step)
      if (count(step) > stride)
        throw std::invalid_argument("readiness ring extent");
  }
  size_t count(int step) const {
    return tiles[step] + size_t(splitFinal && step == 2);
  }
  size_t offset(int step, size_t tile) const {
    return splitFinal && step == 2 && tile == tiles[step] ? shard - (1UL << 18)
                                                          : tile * chunk[step];
  }
  size_t length(int step, size_t tile) const {
    size_t result = std::min(chunk[step], shard - offset(step, tile));
    if (splitFinal && step == 2 && tile + 1 == tiles[step])
      result -= (1UL << 18);
    return result;
  }
  size_t sourceTile(int step, size_t offset, size_t length) const {
    // Reduction nodes are ordered within a phase. Waiting on the last
    // covered prior tile therefore covers every byte, including crossings.
    return (offset + length - 1) / chunk[step - 1];
  }
};
#endif
#ifdef UCCL_LITE_LOCAL_RS_NATIVE_GRAPH
struct LocalNativeRingGraph {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  cudaGraphNode_t epoch = nullptr;
  std::array<std::vector<cudaGraphNode_t>, 3> copies, kernels;
  void const* input = nullptr;
  void* output = nullptr;
  unsigned long long allocation = 0;
  ~LocalNativeRingGraph() {
    if (executable) cudaGraphExecDestroy(executable);
    if (graph) cudaGraphDestroy(graph);
  }
};
#endif
class PeerPacketCollectives::Impl {
 public:
  NetworkId id;
  int rank, device;
  size_t capacity;
  std::array<int, 2> numaNodes;
  std::shared_ptr<mscclpp::TcpBootstrap> bootstrap;
  Peers peers{};
#ifdef UCCL_LITE_LOCAL_HOST_AG
  std::unique_ptr<mscclpp::Communicator> hostCommunicator;
  std::unique_ptr<multinode::SharedHostPackets> hostPackets;
#ifdef UCCL_LITE_LOCAL_AG_HOST_READY
  std::unique_ptr<multinode::SharedHostPackets> hostReadyPackets;
#endif
  HostGather hostGather{};
#ifdef UCCL_LITE_LOCAL_HOST_REPLICA
  std::unique_ptr<LocalHostCopy> hostCopy;
  HostGather replicaGather{};
#ifdef UCCL_LITE_LOCAL_HOST_PER_SOURCE
  uint64_t replicaGeneration = 0;
#endif
  ReplicaCounters* replicaCounters = nullptr;
#endif
  uint64_t hostGeneration = 0;
#endif
  cudaEvent_t completion = nullptr;
#ifdef UCCL_LITE_LOCAL_DRIVER_TREE
  cudaFunction_t treeFunction = nullptr;
#endif
#ifdef UCCL_LITE_LOCAL_STREAM_ID
  unsigned long long previousStreamId = 0;
#endif
  unsigned long long generation = 0;
#ifdef UCCL_LITE_LOCAL_HIER_CREDIT
  int previousPacketMode = 0;
  size_t previousCount = 0;
#endif
#ifdef UCCL_LITE_LOCAL_AG_GRAPH_OUTPUT
#ifdef UCCL_LITE_LOCAL_AG_EPOCH_GRAPH
  std::map<size_t, std::unique_ptr<LocalCopyGraph>> epochGatherGraphs;
#ifdef UCCL_LITE_LOCAL_AG_EPOCH_CACHE128
  uint64_t cachedGatherCalls = 0;
#endif
  void gatherEpochGraph(void const* input, void* output, size_t count,
                        int threads, cudaStream_t stream) {
#ifdef UCCL_LITE_LOCAL_AG_EPOCH_THREADS
    static_assert(UCCL_LITE_LOCAL_AG_EPOCH_THREADS >= 32 &&
                  UCCL_LITE_LOCAL_AG_EPOCH_THREADS <= 1024 &&
                  !(UCCL_LITE_LOCAL_AG_EPOCH_THREADS &
                    (UCCL_LITE_LOCAL_AG_EPOCH_THREADS - 1)));
    threads = UCCL_LITE_LOCAL_AG_EPOCH_THREADS;
#endif
    size_t bankElements = capacity / 4;
    auto* sequence = peers.flags[rank] + 129;
    auto* staging = peers.data[rank];
#ifdef UCCL_LITE_LOCAL_AG_HOST_READY
    auto* ready = reinterpret_cast<unsigned*>(hostReadyPackets->mapped(0));
#endif
    void* sendArgs[] = {&peers,
                        &rank,
                        &input,
                        &count,
                        &bankElements,
                        &sequence
#ifdef UCCL_LITE_LOCAL_AG_HOST_READY
                        ,
                        &ready
#endif
    };
#ifdef UCCL_LITE_LOCAL_AG_PREFETCH_OUT_ONLY
    bool outOfPlace = input != static_cast<float*>(output) + rank * count;
    void* copyArgs[] = {&staging,      &output,   &count,
                        &bankElements, &sequence, &outOfPlace};
#else
    void* copyArgs[] = {&staging, &output, &count, &bankElements, &sequence};
#endif
    cudaKernelNodeParams send{}, copy{};
    send.func = reinterpret_cast<void*>(gatherEpochPublish);
#ifdef UCCL_LITE_LOCAL_AG_EPOCH_CACHE128
    bool cached = count == 8192 && threads == 256;
#ifdef UCCL_LITE_LOCAL_AG_EPOCH_CACHE64
    cached = cached || (count == 4096 && threads == 128);
#endif
    if (cached)
      send.func = reinterpret_cast<void*>(gatherEpochPublishCached<8>);
#endif
    send.gridDim = dim3(1);
    send.blockDim = dim3(threads);
    send.kernelParams = sendArgs;
    copy.func = reinterpret_cast<void*>(gatherEpochCopy);
    copy.gridDim = dim3(std::min(size_t(128), (count + 255) / 256));
    copy.blockDim = dim3(256);
    copy.kernelParams = copyArgs;
    auto& entry = epochGatherGraphs[count];
    if (!entry) {
      entry = std::make_unique<LocalCopyGraph>();
      check(cudaGraphCreate(&entry->graph, 0));
      check(cudaGraphAddKernelNode(&entry->kernel, entry->graph, nullptr, 0,
                                   &send));
      check(cudaGraphAddKernelNode(&entry->copy, entry->graph, &entry->kernel,
                                   1, &copy));
      cudaGraphNode_t done;
      check(cudaGraphAddEventRecordNode(&done, entry->graph, &entry->copy, 1,
                                        completion));
      check(cudaGraphInstantiate(&entry->executable, entry->graph, nullptr,
                                 nullptr, 0));
    } else if (entry->boundInput != input || entry->boundOutput != output) {
      check(cudaGraphExecKernelNodeSetParams(entry->executable, entry->kernel,
                                             &send));
      check(cudaGraphExecKernelNodeSetParams(entry->executable, entry->copy,
                                             &copy));
    }
    entry->boundInput = input;
    entry->boundOutput = output;
    check(cudaGraphLaunch(entry->executable, stream));
#ifdef UCCL_LITE_LOCAL_AG_EPOCH_CACHE128
    if (cached) ++cachedGatherCalls;
#endif
  }
#endif
  std::map<std::pair<size_t, int>, std::unique_ptr<LocalCopyGraph>> copyGraphs;
  void gatherGraph(void const* input, void* output, size_t count, int threads,
                   size_t sharedBytes, cudaStream_t stream) {
    int operation = 1;
    size_t bankElements = capacity / 4;
    int bank = generation & 1;
    void* args[] = {&peers,  &rank,  &operation,    &input,
                    &output, &count, &bankElements, &generation};
    cudaKernelNodeParams params{};
#ifdef UCCL_LITE_LOCAL_SERIAL_SPECIALIZE
    params.func = reinterpret_cast<void*>(serialCollectSpecialized<1>);
#else
    params.func = reinterpret_cast<void*>(serialCollect);
#endif
    params.gridDim = dim3(1);
    params.blockDim = dim3(threads);
    params.sharedMemBytes = sharedBytes;
    params.kernelParams = args;
    auto& entry = copyGraphs[{count, bank}];
    bool fresh = !entry;
    if (fresh) {
      entry = std::make_unique<LocalCopyGraph>();
      check(cudaGraphCreate(&entry->graph, 0));
      check(cudaGraphAddKernelNode(&entry->kernel, entry->graph, nullptr, 0,
                                   &params));
      check(cudaGraphAddMemcpyNode1D(&entry->copy, entry->graph, &entry->kernel,
                                     1, output,
                                     peers.data[rank] + bank * bankElements,
                                     count * 16, cudaMemcpyDeviceToDevice));
      check(cudaGraphInstantiate(&entry->executable, entry->graph, nullptr,
                                 nullptr, 0));
    } else {
      check(cudaGraphExecKernelNodeSetParams(entry->executable, entry->kernel,
                                             &params));
    }
    // Same allocation-identity rule as mscclpp_backend.cu's binding cache:
    // pointer equality alone cannot validate a recycled allocation.
    unsigned long long allocation = 0;
    unsigned int legacy = 0;
    CUpointer_attribute attributes[] = {
        CU_POINTER_ATTRIBUTE_BUFFER_ID,
        CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE};
    void* values[] = {&allocation, &legacy};
    CUresult status = cuPointerGetAttributes(
        2, attributes, values, reinterpret_cast<CUdeviceptr>(output));
    if (status != CUDA_SUCCESS && status != CUDA_ERROR_INVALID_VALUE &&
        status != CUDA_ERROR_NOT_SUPPORTED)
      throw std::runtime_error("local graph allocation query failed");
    bool eligible = status == CUDA_SUCCESS && legacy && allocation;
    bool reuse = eligible && entry->valid && entry->boundOutput == output &&
                 entry->allocation == allocation;
    if (!fresh && !reuse)
      check(cudaGraphExecMemcpyNodeSetParams1D(
          entry->executable, entry->copy, output,
          peers.data[rank] + bank * bankElements, count * 16,
          cudaMemcpyDeviceToDevice));
    entry->valid = eligible;
    entry->boundOutput = output;
    entry->allocation = allocation;
    check(cudaGraphLaunch(entry->executable, stream));
  }
#endif
#ifdef UCCL_LITE_LOCAL_RING
  cudaStream_t ringReduce = nullptr;
  cudaEvent_t ringStart = nullptr;
  std::array<cudaEvent_t, 6 * kRingTiles> ringReduced{};
#ifdef UCCL_LITE_LOCAL_RS_HOST_READY
  unsigned* ringHostFlags[4]{};
  int ringHostBlocks = 0;
  uint64_t ringReadyCalls = 0;
  unsigned* ringReadyAddress(int source, size_t index) {
#ifdef UCCL_LITE_LOCAL_RS_GPU_READY
    // Low 32 bits of each private producer word; all flags and the relay
    // gate are cleared together at the existing quiescent generation wrap.
    return reinterpret_cast<unsigned*>(peers.flags[source] + 256 + index);
#else
    return ringHostFlags[source] + index;
#endif
  }
#endif
#ifdef UCCL_LITE_LOCAL_RING_IPC_EVENTS
#ifndef UCCL_LITE_LOCAL_HOST_AG_WIDE
#error "IPC event publication reuses the wide shared-host allocation"
#endif
  std::array<cudaEvent_t, 6 * kRingTiles> ringArrived{}, ringPrevious{};
#ifdef UCCL_LITE_LOCAL_HIER_RS
  std::array<cudaEvent_t, 3 * kRingTiles> hierarchyPrevious{};
#endif
  uint64_t* ringPublished[4][6]{};
  uint64_t* ringAcknowledged[4][6]{};
  uint64_t ringEpoch = 0, ringLastPublished[6]{};
#ifdef UCCL_LITE_LOCAL_RS_TILE_PUBLICATION
  uint64_t* tilePublished[4][3 * kRingTiles]{};
  uint64_t* tileAcknowledged[4][3 * kRingTiles]{};
  uint64_t tileLastPublished[3 * kRingTiles]{};
  bool tileProtocol = false;
  uint64_t tileProtocolCalls = 0;
#endif
#if defined(UCCL_LITE_LOCAL_RS_RING4M) || defined(UCCL_LITE_LOCAL_RS_SPLIT_TAIL)
  size_t ringEventStride = kRingTiles;
#endif
  static void cpuWait(uint64_t* value, uint64_t epoch) {
    while (__atomic_load_n(value, __ATOMIC_ACQUIRE) < epoch)
      std::this_thread::yield();
  }
  void publishEvents(int stage, int prev, uint64_t epoch) {
    __atomic_store_n(ringPublished[rank][stage], epoch, __ATOMIC_RELEASE);
    ringLastPublished[stage] = epoch;
    cpuWait(ringPublished[prev][stage], epoch);
  }
  void acknowledgeEvents(int stage, int prev, uint64_t epoch) {
    __atomic_store_n(ringAcknowledged[prev][stage], epoch, __ATOMIC_RELEASE);
  }
#endif
#ifdef UCCL_LITE_LOCAL_HIER_RS
#ifndef UCCL_LITE_LOCAL_RING_IPC_EVENTS
#error "Hierarchical RS requires the IPC event protocol"
#endif
  void scatterHierarchy(float const* input, float* output, size_t shard,
                        cudaStream_t stream) {
    size_t chunk = std::min(shard, size_t(8UL << 20) / 4);
    size_t tiles = (shard + chunk - 1) / chunk;
    if (tiles > kRingTiles || ringEpoch == UINT64_MAX)
      throw std::invalid_argument("hierarchical RS extent or epoch exhausted");
    uint64_t epoch = ++ringEpoch;
    check(cudaEventRecord(ringStart, stream));
    check(cudaStreamWaitEvent(ringReduce, ringStart, 0));
    // Each local pair reduces interleaved destination shards p and p+2.
    // The cross-NUMA exchange then sends only the other group's shard.
    // Distinct receive slots survive until the public-stream retirement fence.
    for (int step = 0; step < 3; ++step) {
      int partner = rank ^ (step < 2 ? 1 : 2);
      cpuWait(ringAcknowledged[rank][step], ringLastPublished[step]);
      for (size_t tile = 0; tile < tiles; ++tile) {
        size_t offset = tile * chunk, length = std::min(chunk, shard - offset);
        size_t index = step * kRingTiles + tile;
        float const* source;
        if (step < 2) {
          int owner = (partner & 1) + 2 * step;
          source = input + owner * shard + offset;
        } else {
          int remoteHalf = 1 - rank / 2;
          check(cudaStreamWaitEvent(
              stream, ringReduced[remoteHalf * kRingTiles + tile], 0));
          source = peers.data[rank] + remoteHalf * shard + offset;
        }
        check(cudaMemcpyAsync(peers.data[partner] + step * shard + offset,
                              source, length * 4, cudaMemcpyDeviceToDevice,
                              stream));
        check(cudaEventRecord(ringArrived[index], stream));
      }
      publishEvents(step, partner, epoch);
      for (size_t tile = 0; tile < tiles; ++tile) {
        size_t offset = tile * chunk, length = std::min(chunk, shard - offset);
        size_t index = step * kRingTiles + tile;
        check(cudaStreamWaitEvent(ringReduce, hierarchyPrevious[index], 0));
        auto* incoming = peers.data[rank] + step * shard + offset;
        auto const* own = step < 2
                              ? input + ((rank & 1) + 2 * step) * shard + offset
                              : peers.data[rank] + (rank / 2) * shard + offset;
        auto* target = step < 2 ? incoming : output + offset;
        int blocks = std::min(size_t(256), (length + 255) / 256);
        ringSum<<<blocks, 256, 0, ringReduce>>>(own, incoming, target, length);
        check(cudaEventRecord(ringReduced[index], ringReduce));
      }
      acknowledgeEvents(step, partner, epoch);
    }
    check(cudaStreamWaitEvent(stream, ringReduced[2 * kRingTiles + tiles - 1],
                              0));
    barrier<<<1, 1, 0, stream>>>(peers, rank, 2, generation);
  }
#endif
#ifdef UCCL_LITE_LOCAL_RS_MEMOP
  void scatterMemop(float const* input, float* output, size_t shard,
                    cudaStream_t stream) {
    constexpr int ring[4] = {0, 1, 3, 2};
    int pos = rank ^ (rank >> 1), next = ring[(pos + 1) % 4];
    size_t chunk = std::min(shard, size_t(8UL << 20) / 4);
    size_t tiles = (shard + chunk - 1) / chunk;
    if (tiles > kRingTiles) throw std::invalid_argument("memop RS extent");
    check(cudaEventRecord(ringStart, stream));
    check(cudaStreamWaitEvent(ringReduce, ringStart, 0));
    // Phase-zero copies have no reduction dependency. Later copies wait on
    // native local events from the preceding phase. The reduction stream
    // completes the whole preceding phase before any next-phase flag wait;
    // its spinning CTAs cannot block the CE work needed for their own flags.
    for (int step = 0; step < 3; ++step) {
      for (size_t tile = 0; tile < tiles; ++tile) {
        size_t offset = tile * chunk, length = std::min(chunk, shard - offset);
        size_t index = step * kRingTiles + tile;
        int sendOwner = ring[(pos - step + 3) % 4];
        if (step)
          check(
              cudaStreamWaitEvent(stream, ringReduced[index - kRingTiles], 0));
        auto const* source =
            step == 0 ? input + sendOwner * shard + offset
                      : peers.data[rank] + (step - 1) * shard + offset;
        check(cudaMemcpyAsync(peers.data[next] + step * shard + offset, source,
                              length * 4, cudaMemcpyDeviceToDevice, stream));
        // Flags=0 retains the driver's system memory barrier after the copy.
        if (
#ifdef UCCL_LITE_LOCAL_RS_MEMOP32
            cuStreamWriteValue32(
#else
            cuStreamWriteValue64(
#endif
                reinterpret_cast<CUstream>(stream),
                reinterpret_cast<CUdeviceptr>(peers.flags[next] + 256 + index),
                generation, 0) != CUDA_SUCCESS)
          throw std::runtime_error("peer stream value publication failed");
        int owner = ring[(pos - step + 2) % 4];
        auto* incoming = peers.data[rank] + step * shard + offset;
        auto* target = step == 2 ? output + offset : incoming;
        int blocks = std::min(size_t(256), (length + 255) / 256);
        ringWaitSum<<<blocks, 256, 0, ringReduce>>>(
            input + owner * shard + offset, incoming, target, length,
            peers.flags[rank] + 256 + index, generation);
        check(cudaEventRecord(ringReduced[index], ringReduce));
      }
    }
    check(cudaStreamWaitEvent(stream, ringReduced[2 * kRingTiles + tiles - 1],
                              0));
    barrier<<<1, 1, 0, stream>>>(peers, rank, 2, generation);
  }
#endif
#ifdef UCCL_LITE_LOCAL_RS_PULL
#if !defined(UCCL_LITE_LOCAL_RING_IPC_EVENTS) ||                             \
    defined(UCCL_LITE_LOCAL_RS_MEMOP) || defined(UCCL_LITE_LOCAL_HIER_RS) || \
    defined(UCCL_LITE_LOCAL_RS_BANKS) ||                                     \
    defined(UCCL_LITE_LOCAL_RS_CHUNK_BYTES)
#error "Pull RS is an isolated IPC-event experiment"
#endif
  void scatterPull(float const* input, float* output, size_t shard,
                   cudaStream_t stream) {
    constexpr int ring[4] = {0, 1, 3, 2};
    int pos = rank ^ (rank >> 1), prev = ring[(pos + 3) % 4];
    size_t chunk = std::min(shard, size_t(8UL << 20) / 4);
    size_t tiles = (shard + chunk - 1) / chunk;
    if (tiles > kRingTiles || ringEpoch == UINT64_MAX)
      throw std::invalid_argument("pull RS extent or epoch exhausted");
    uint64_t epoch = ++ringEpoch;
    check(cudaEventRecord(ringStart, stream));
    check(cudaStreamWaitEvent(ringReduce, ringStart, 0));
    // Only private allocation handles are shared. Stage the initial outgoing
    // shard; subsequent outgoing data are the preceding partial reductions.
    cpuWait(ringAcknowledged[rank][0], ringLastPublished[0]);
    for (size_t tile = 0; tile < tiles; ++tile) {
      size_t offset = tile * chunk, length = std::min(chunk, shard - offset);
      int blocks = std::min(size_t(256), (length + 255) / 256);
      ringCopy<<<blocks, 256, 0, ringReduce>>>(
          input + ring[(pos + 3) % 4] * shard + offset,
          peers.data[rank] + 3 * shard + offset, length);
      check(cudaEventRecord(ringArrived[tile], ringReduce));
    }
    for (int step = 0; step < 3; ++step) {
      publishEvents(step, prev, epoch);
      if (step < 2)
        cpuWait(ringAcknowledged[rank][step + 1], ringLastPublished[step + 1]);
      for (size_t tile = 0; tile < tiles; ++tile) {
        size_t offset = tile * chunk, length = std::min(chunk, shard - offset);
        size_t index = step * kRingTiles + tile;
        check(cudaStreamWaitEvent(stream, ringPrevious[index], 0));
        auto const* source =
            peers.data[prev] + (step == 0 ? 3 : step - 1) * shard + offset;
        auto* incoming = peers.data[rank] + step * shard + offset;
        check(cudaMemcpyAsync(incoming, source, length * 4,
                              cudaMemcpyDeviceToDevice, stream));
        // The CE path records a local event. Cross-process publication is
        // on the producer stream, where its cost can overlap later copies.
        check(cudaEventRecord(ringReduced[index], stream));
        check(cudaStreamWaitEvent(ringReduce, ringReduced[index], 0));
        int owner = ring[(pos - step + 2) % 4];
        int blocks = std::min(size_t(256), (length + 255) / 256);
        ringSum<<<blocks, 256, 0, ringReduce>>>(
            input + owner * shard + offset, incoming,
            step == 2 ? output + offset : incoming, length);
        if (step < 2)
          check(cudaEventRecord(ringArrived[(step + 1) * kRingTiles + tile],
                                ringReduce));
        else
          check(
              cudaEventRecord(ringReduced[3 * kRingTiles + tile], ringReduce));
      }
      acknowledgeEvents(step, prev, epoch);
    }
    check(cudaStreamWaitEvent(stream, ringReduced[3 * kRingTiles + tiles - 1],
                              0));
    // Retire all remote readers before reusing the four private shard slots.
    barrier<<<1, 1, 0, stream>>>(peers, rank, 2, generation);
  }
#endif
#ifdef UCCL_LITE_LOCAL_RS_HOST_READY
#ifdef UCCL_LITE_LOCAL_RS_NATIVE_GRAPH
  std::map<std::pair<size_t, unsigned>, std::unique_ptr<LocalNativeRingGraph>>
      nativeRingGraphs;
  uint64_t nativeGraphCalls = 0, nativeGraphEagerCalls = 0;
  int nativeGraphBlocks = 0;
  bool scatterNativeGraph(float const* input, float* output, size_t shard,
                          cudaStream_t stream) {
    unsigned long long allocation = 0;
    unsigned legacy = 0;
    CUpointer_attribute attributes[] = {
        CU_POINTER_ATTRIBUTE_BUFFER_ID,
        CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE};
    void* values[] = {&allocation, &legacy};
    CUresult status = cuPointerGetAttributes(
        2, attributes, values, reinterpret_cast<CUdeviceptr>(input));
    if (status != CUDA_SUCCESS && status != CUDA_ERROR_INVALID_VALUE &&
        status != CUDA_ERROR_NOT_SUPPORTED)
      throw std::runtime_error("native graph allocation query failed");
    if (status != CUDA_SUCCESS || !legacy || !allocation) {
      ++nativeGraphEagerCalls;
      return false;
    }
    constexpr int ring[4] = {0, 1, 3, 2};
    int pos = rank ^ (rank >> 1), next = ring[(pos + 1) % 4],
        prev = ring[(pos + 3) % 4];
    NativeRingLayout layout(shard);
    size_t stride = layout.stride;
    if (shard < (2UL << 18)) throw std::invalid_argument("native graph extent");
    unsigned bankIndex = 0;
#ifdef UCCL_LITE_LOCAL_RS_NATIVE_BANKS
    bankIndex = generation & 1;
#endif
    // Six shards fit inside the existing twice-capacity allocation.
    size_t bank = bankIndex * 3 * shard;
    auto& entry = nativeRingGraphs[{shard, bankIndex}];
    bool fresh = !entry;
    if (fresh) entry = std::make_unique<LocalNativeRingGraph>();
    bool copyUpdate =
        fresh || entry->input != input || entry->allocation != allocation;
    bool kernelUpdate =
        fresh || entry->input != input || entry->output != output;
    auto* epoch = reinterpret_cast<unsigned*>(peers.flags[rank] + 450);
#ifdef UCCL_LITE_LOCAL_RS_DEVICE_EPOCH
    void* epochArgs[] = {&epoch};
#else
    unsigned tag = static_cast<unsigned>(generation);
    void* epochArgs[] = {&epoch, &tag};
#endif
    cudaKernelNodeParams init{};
    init.func = reinterpret_cast<void*>(ringGraphEpoch);
    init.gridDim = dim3(1);
    init.blockDim = dim3(1);
    init.kernelParams = epochArgs;
    if (fresh) {
      check(cudaGraphCreate(&entry->graph, 0));
      check(cudaGraphAddKernelNode(&entry->epoch, entry->graph, nullptr, 0,
                                   &init));
    }
#ifndef UCCL_LITE_LOCAL_RS_DEVICE_EPOCH
    else
      check(cudaGraphExecKernelNodeSetParams(entry->executable, entry->epoch,
                                             &init));
#else
    if (!fresh && !copyUpdate && !kernelUpdate) {
      check(cudaGraphLaunch(entry->executable, stream));
      ++nativeGraphCalls;
      return true;
    }
#endif
    cudaGraphNode_t lastCopy = entry->epoch, lastKernel = nullptr;
    for (int step = 0; step < 3; ++step) {
      size_t count = layout.count(step);
      if (fresh) {
        entry->copies[step].resize(count);
        entry->kernels[step].resize(count);
      }
      for (size_t tile = 0; tile < count; ++tile) {
        size_t offset = layout.offset(step, tile);
        size_t length = layout.length(step, tile);
        size_t index = step * stride + tile, tokenStride = 3 * stride;
        int sendOwner = ring[(pos - step + 3) % 4];
        auto const* source =
            step == 0 ? input + sendOwner * shard + offset
                      : peers.data[rank] + bank + (step - 1) * shard + offset;
        auto* destination = peers.data[next] + bank + step * shard + offset;
        auto& copy = entry->copies[step][tile];
        if (fresh) {
          cudaGraphNode_t dependencies[] = {
              lastCopy, step ? entry->kernels[step - 1][layout.sourceTile(
                                   step, offset, length)]
                             : nullptr};
          check(cudaGraphAddMemcpyNode1D(&copy, entry->graph, dependencies,
                                         step ? 2 : 1, destination, source,
                                         length * 4, cudaMemcpyDeviceToDevice));
        } else if (copyUpdate && step == 0)
          check(cudaGraphExecMemcpyNodeSetParams1D(
              entry->executable, copy, destination, source, length * 4,
              cudaMemcpyDeviceToDevice));
        lastCopy = copy;
        int owner = ring[(pos - step + 2) % 4];
        auto const* own = input + owner * shard + offset;
        auto* incoming = peers.data[rank] + bank + step * shard + offset;
        auto* target = step == 2 ? output + offset : incoming;
        auto* produced = ringReadyAddress(rank, index);
        auto* ready = ringReadyAddress(prev, index);
        auto* gate = peers.flags[rank] + 449;
        void* args[] = {&own,   &incoming, &target, &length, &produced,
                        &ready, &gate,     &epoch,  &index,  &tokenStride};
        cudaKernelNodeParams params{};
        params.func = reinterpret_cast<void*>(ringGraphWaitSum);
        params.gridDim =
            dim3(std::min(size_t(nativeGraphBlocks), (length + 255) / 256));
        params.blockDim = dim3(256);
        params.kernelParams = args;
        auto& kernel = entry->kernels[step][tile];
        if (fresh) {
          cudaGraphNode_t dependencies[] = {copy, lastKernel};
          check(cudaGraphAddKernelNode(&kernel, entry->graph, dependencies,
                                       lastKernel ? 2 : 1, &params));
          cudaKernelNodeAttrValue cooperative{};
          cooperative.cooperative = 1;
          check(cudaGraphKernelNodeSetAttribute(
              kernel, cudaKernelNodeAttributeCooperative, &cooperative));
        } else if (kernelUpdate)
          check(cudaGraphExecKernelNodeSetParams(entry->executable, kernel,
                                                 &params));
        lastKernel = kernel;
      }
    }
    if (fresh) {
      cudaGraphNode_t retire = lastKernel, completed;
#ifndef UCCL_LITE_LOCAL_RS_NATIVE_BANKS
      void* args[] = {&peers, &rank, &epoch};
      cudaKernelNodeParams params{};
      params.func = reinterpret_cast<void*>(ringGraphRetire);
      params.gridDim = dim3(1);
      params.blockDim = dim3(1);
      params.kernelParams = args;
      check(cudaGraphAddKernelNode(&retire, entry->graph, &lastKernel, 1,
                                   &params));
#endif
      check(cudaGraphAddEventRecordNode(&completed, entry->graph, &retire, 1,
                                        completion));
      check(cudaGraphInstantiate(&entry->executable, entry->graph, nullptr,
                                 nullptr, 0));
    }
    entry->input = input;
    entry->output = output;
    entry->allocation = allocation;
    check(cudaGraphLaunch(entry->executable, stream));
    ++nativeGraphCalls;
    return true;
  }
#endif
  void scatterHostReady(float const* input, float* output, size_t shard,
                        cudaStream_t stream) {
    constexpr int ring[4] = {0, 1, 3, 2};
    int pos = rank ^ (rank >> 1), next = ring[(pos + 1) % 4],
        prev = ring[(pos + 3) % 4];
    NativeRingLayout layout(shard);
    size_t eventStride = layout.stride;
    if (3 * eventStride > ringReduced.size())
      throw std::invalid_argument("readiness RS extent");
    size_t bank = 0;
#ifdef UCCL_LITE_LOCAL_RS_NATIVE_BANKS
    bank = (generation & 1) * 3 * shard;
#endif
#ifdef UCCL_LITE_LOCAL_RS_DEVICE_EPOCH
    // Every eligible call advances the same GPU counter, including eager
    // allocation compatibility paths on only a subset of ranks.
    auto* epoch = reinterpret_cast<unsigned*>(peers.flags[rank] + 450);
    ringGraphEpoch<<<1, 1, 0, stream>>>(epoch);
#endif
    check(cudaEventRecord(ringStart, stream));
    check(cudaStreamWaitEvent(ringReduce, ringStart, 0));
    for (int step = 0; step < 3; ++step) {
      for (size_t tile = 0; tile < layout.count(step); ++tile) {
        size_t offset = layout.offset(step, tile);
        size_t length = layout.length(step, tile);
        size_t index = step * eventStride + tile;
        int sendOwner = ring[(pos - step + 3) % 4];
        if (step)
          check(cudaStreamWaitEvent(
              stream,
              ringReduced[(step - 1) * eventStride +
                          layout.sourceTile(step, offset, length)],
              0));
        auto const* source =
            step == 0 ? input + sendOwner * shard + offset
                      : peers.data[rank] + bank + (step - 1) * shard + offset;
        check(cudaMemcpyAsync(peers.data[next] + bank + step * shard + offset,
                              source, length * 4, cudaMemcpyDeviceToDevice,
                              stream));
#ifndef UCCL_LITE_LOCAL_RS_DEVICE_EPOCH
        unsigned tag = static_cast<unsigned>(generation);
#endif
#ifdef UCCL_LITE_LOCAL_RS_GPU_NATIVE
        // Bind the copy record before re-recording this local event after
        // reduction. No peer binds these native events; readiness uses tags.
        check(cudaEventRecord(ringReduced[index], stream));
        check(cudaStreamWaitEvent(ringReduce, ringReduced[index], 0));
#else
        if (cuStreamWriteValue32(
                reinterpret_cast<CUstream>(stream),
                reinterpret_cast<CUdeviceptr>(ringReadyAddress(rank, index)),
                tag, 0) != CUDA_SUCCESS)
          throw std::runtime_error("host-ready publication failed");
#endif
        int owner = ring[(pos - step + 2) % 4];
        auto const* own = input + owner * shard + offset;
        auto* incoming = peers.data[rank] + bank + step * shard + offset;
        auto* target = step == 2 ? output + offset : incoming;
        auto* ready = ringReadyAddress(prev, index);
        auto* gate = peers.flags[rank] + 449;
#ifndef UCCL_LITE_LOCAL_RS_DEVICE_EPOCH
        unsigned long long token = generation * (3 * eventStride) + index;
#endif
#ifdef UCCL_LITE_LOCAL_RS_GPU_NATIVE
        auto* produced = ringReadyAddress(rank, index);
#ifdef UCCL_LITE_LOCAL_RS_DEVICE_EPOCH
        size_t tokenStride = 3 * eventStride;
        void* args[] = {&own,   &incoming, &target, &length, &produced,
                        &ready, &gate,     &epoch,  &index,  &tokenStride};
#else
        void* args[] = {&own,   &incoming, &target, &length, &produced,
                        &ready, &gate,     &tag,    &token};
#endif
#else
        void* args[] = {&own,   &incoming, &target, &length,
                        &ready, &gate,     &tag,    &token};
#endif
#ifdef UCCL_LITE_LOCAL_RS_DEVICE_EPOCH
        int blocks = std::min(size_t(nativeGraphBlocks), (length + 255) / 256);
#else
        int blocks = std::min(size_t(ringHostBlocks), (length + 255) / 256);
#endif
        // Cooperative residency guarantees the host-polling leader can run
        // while other CTAs wait on its device-local gate. CE work progresses
        // independently; later copies use native prior-phase reduction events.
        check(cudaLaunchCooperativeKernel(
#ifdef UCCL_LITE_LOCAL_RS_DEVICE_EPOCH
            reinterpret_cast<void*>(ringGraphWaitSum),
#else
            reinterpret_cast<void*>(ringHostWaitSum),
#endif
            dim3(blocks), dim3(256), args, 0, ringReduce));
        check(cudaEventRecord(ringReduced[index], ringReduce));
      }
    }
    check(cudaStreamWaitEvent(
        stream, ringReduced[2 * eventStride + layout.count(2) - 1], 0));
#ifndef UCCL_LITE_LOCAL_RS_NATIVE_BANKS
#ifdef UCCL_LITE_LOCAL_RS_DEVICE_EPOCH
    ringGraphRetire<<<1, 1, 0, stream>>>(peers, rank, epoch);
#else
    barrier<<<1, 1, 0, stream>>>(peers, rank, 2, generation);
#endif
#endif
    ++ringReadyCalls;
  }
#endif
#ifdef UCCL_LITE_LOCAL_RS_GRAPHS
#if !defined(UCCL_LITE_LOCAL_RING_IPC_EVENTS) ||                             \
    defined(UCCL_LITE_LOCAL_RS_BANKS) ||                                     \
    defined(UCCL_LITE_LOCAL_RS_HOST_READY) ||                                \
    defined(UCCL_LITE_LOCAL_RS_MEMOP) || defined(UCCL_LITE_LOCAL_RS_PULL) || \
    defined(UCCL_LITE_LOCAL_HIER_RS) ||                                      \
    defined(UCCL_LITE_LOCAL_RS_COPY_GROUP) ||                                \
    defined(UCCL_LITE_LOCAL_RS_RING4M)
#error "Ring graphs require the isolated eight-MiB IPC schedule"
#endif
  std::map<size_t, std::unique_ptr<LocalRingGraphs>> ringGraphs;
  uint64_t ringGraphCalls = 0, ringGraphEagerCalls = 0;
  bool scatterGraphs(float const* input, float* output, size_t shard,
                     cudaStream_t stream) {
    unsigned long long allocation = 0;
    unsigned legacy = 0;
    CUpointer_attribute attributes[] = {
        CU_POINTER_ATTRIBUTE_BUFFER_ID,
        CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE};
    void* values[] = {&allocation, &legacy};
    CUresult status = cuPointerGetAttributes(
        2, attributes, values, reinterpret_cast<CUdeviceptr>(input));
    if (status != CUDA_SUCCESS && status != CUDA_ERROR_INVALID_VALUE &&
        status != CUDA_ERROR_NOT_SUPPORTED)
      throw std::runtime_error("ring graph allocation query failed");
    if (status != CUDA_SUCCESS || !legacy || !allocation) {
      ++ringGraphEagerCalls;
      return false;
    }
    constexpr int ring[4] = {0, 1, 3, 2};
    int pos = rank ^ (rank >> 1), next = ring[(pos + 1) % 4],
        prev = ring[(pos + 3) % 4];
    size_t chunk = std::min(shard, size_t(8UL << 20) / 4);
    size_t tiles = (shard + chunk - 1) / chunk;
    if (tiles > kRingTiles || ringEpoch == UINT64_MAX)
      throw std::invalid_argument("ring graph extent or epoch exhausted");
    auto& entry = ringGraphs[shard];
    bool fresh = !entry;
    if (fresh) entry = std::make_unique<LocalRingGraphs>();
    bool copyUpdate =
        fresh || entry->input != input || entry->allocation != allocation;
    bool kernelUpdate =
        fresh || entry->input != input || entry->output != output;
    for (int step = 0; step < 3; ++step) {
      auto& phase = entry->phases[step];
      if (fresh) {
        check(cudaGraphCreate(&phase.send, 0));
        check(cudaGraphCreate(&phase.receive, 0));
      }
      cudaGraphNode_t sendLast = nullptr, receiveLast = nullptr;
      for (size_t tile = 0; tile < tiles; ++tile) {
        size_t offset = tile * chunk, length = std::min(chunk, shard - offset);
        size_t index = step * kRingTiles + tile;
        int sendOwner = ring[(pos - step + 3) % 4];
        auto const* from = step == 0
                               ? input + sendOwner * shard + offset
                               : peers.data[rank] + (step - 1) * shard + offset;
        auto* to = peers.data[next] + step * shard + offset;
        if (fresh) {
          if (step) {
            cudaGraphNode_t wait;
            check(cudaGraphAddEventWaitNode(
                &wait, phase.send, sendLast ? &sendLast : nullptr,
                sendLast ? 1 : 0, ringReduced[index - kRingTiles]));
            sendLast = wait;
          }
          cudaGraphNode_t copy, recorded;
          check(cudaGraphAddMemcpyNode1D(&copy, phase.send,
                                         sendLast ? &sendLast : nullptr,
                                         sendLast ? 1 : 0, to, from, length * 4,
                                         cudaMemcpyDeviceToDevice));
          phase.copies.push_back(copy);
#ifdef UCCL_LITE_LOCAL_RS_GRAPH_OVERLAP
          sendLast = copy;
#else
          check(cudaGraphAddEventRecordNode(&recorded, phase.send, &copy, 1,
                                            ringArrived[index]));
          sendLast = recorded;
#endif
        } else if (step == 0 && copyUpdate) {
          check(cudaGraphExecMemcpyNodeSetParams1D(
              phase.sendExec, phase.copies[tile], to, from, length * 4,
              cudaMemcpyDeviceToDevice));
        }
        int owner = ring[(pos - step + 2) % 4];
        auto const* own = input + owner * shard + offset;
        auto* incoming = peers.data[rank] + step * shard + offset;
        auto* target = step == 2 ? output + offset : incoming;
        void* args[] = {&own, &incoming, &target, &length};
        cudaKernelNodeParams params{};
        params.func = reinterpret_cast<void*>(ringSum);
        params.gridDim = dim3(std::min(size_t(256), (length + 255) / 256));
        params.blockDim = dim3(256);
        params.kernelParams = args;
        if (fresh) {
          cudaGraphNode_t wait, kernel, recorded;
          check(cudaGraphAddEventWaitNode(
              &wait, phase.receive, receiveLast ? &receiveLast : nullptr,
              receiveLast ? 1 : 0, ringPrevious[index]));
          check(cudaGraphAddKernelNode(&kernel, phase.receive, &wait, 1,
                                       &params));
          phase.kernels.push_back(kernel);
          check(cudaGraphAddEventRecordNode(&recorded, phase.receive, &kernel,
                                            1, ringReduced[index]));
          receiveLast = recorded;
        } else if (kernelUpdate) {
          check(cudaGraphExecKernelNodeSetParams(phase.receiveExec,
                                                 phase.kernels[tile], &params));
        }
      }
      if (fresh) {
#ifdef UCCL_LITE_LOCAL_RS_GRAPH_OVERLAP
        std::vector<cudaGraphNode_t> records;
        for (size_t tile = 0; tile < tiles; ++tile) {
          cudaGraphNode_t recorded;
          check(cudaGraphAddEventRecordNode(
              &recorded, phase.send, &phase.copies[tile], 1,
              ringArrived[step * kRingTiles + tile]));
          records.push_back(recorded);
        }
        // Copy i+1 may overlap record i. Copy i+2 must wait for record i,
        // bounding publication lag without coarsening payload/reduction sizes.
        for (size_t tile = 2; tile < tiles; ++tile)
          check(cudaGraphAddDependencies(phase.send, &records[tile - 2],
                                         &phase.copies[tile], 1));
#endif
        check(cudaGraphInstantiate(&phase.sendExec, phase.send, nullptr,
                                   nullptr, 0));
        check(cudaGraphInstantiate(&phase.receiveExec, phase.receive, nullptr,
                                   nullptr, 0));
      }
    }
    entry->input = input;
    entry->output = output;
    entry->allocation = allocation;
    uint64_t epoch = ++ringEpoch;
    check(cudaEventRecord(ringStart, stream));
    check(cudaStreamWaitEvent(ringReduce, ringStart, 0));
    for (int step = 0; step < 3; ++step) {
      cpuWait(ringAcknowledged[rank][step], ringLastPublished[step]);
      check(cudaGraphLaunch(entry->phases[step].sendExec, stream));
      // Publish only after this launch has registered the current records.
      // A receiver launch must not bind a peer's previous event generation.
      publishEvents(step, prev, epoch);
      check(cudaGraphLaunch(entry->phases[step].receiveExec, ringReduce));
      acknowledgeEvents(step, prev, epoch);
    }
    check(cudaStreamWaitEvent(stream, ringReduced[2 * kRingTiles + tiles - 1],
                              0));
    barrier<<<1, 1, 0, stream>>>(peers, rank, 2, generation);
    ++ringGraphCalls;
    return true;
  }
#endif
#ifdef UCCL_LITE_LOCAL_RS_TILE_PUBLICATION
#if !defined(UCCL_LITE_LOCAL_AG_HOST_READY) ||                               \
    !defined(UCCL_LITE_LOCAL_RING_IPC_EVENTS) ||                             \
    defined(UCCL_LITE_LOCAL_RS_GRAPHS) ||                                    \
    defined(UCCL_LITE_LOCAL_RS_HOST_READY) ||                                \
    defined(UCCL_LITE_LOCAL_RS_MEMOP) || defined(UCCL_LITE_LOCAL_RS_PULL) || \
    defined(UCCL_LITE_LOCAL_RS_BANKS) || defined(UCCL_LITE_LOCAL_HIER_RS) || \
    defined(UCCL_LITE_LOCAL_RS_COPY_GROUP) ||                                \
    defined(UCCL_LITE_LOCAL_RS_RING4M)
#error "Per-tile publication requires the isolated shared-control IPC ring"
#endif
#ifndef UCCL_LITE_LOCAL_RS_TILE_MIN_BYTES
#define UCCL_LITE_LOCAL_RS_TILE_MIN_BYTES (128UL << 20)
#endif
  void scatterTilePublication(float const* input, float* output, size_t shard,
                              cudaStream_t stream) {
    constexpr int ring[4] = {0, 1, 3, 2};
    int pos = rank ^ (rank >> 1), next = ring[(pos + 1) % 4],
        prev = ring[(pos + 3) % 4];
    size_t chunk = std::min(shard, size_t(8UL << 20) / 4);
    size_t tiles = (shard + chunk - 1) / chunk;
    if (tiles > kRingTiles || ringEpoch == UINT64_MAX)
      throw std::invalid_argument("tile publication extent or epoch exhausted");
    if (!tileProtocol) {
      for (int step = 0; step < 6; ++step)
        cpuWait(ringAcknowledged[rank][step], ringLastPublished[step]);
      tileProtocol = true;
    }
    uint64_t epoch = ++ringEpoch;
    check(cudaEventRecord(ringStart, stream));
    check(cudaStreamWaitEvent(ringReduce, ringStart, 0));
    for (int step = 0; step < 3; ++step) {
      for (size_t tile = 0; tile < tiles; ++tile) {
        size_t offset = tile * chunk, length = std::min(chunk, shard - offset);
        size_t index = step * kRingTiles + tile;
        cpuWait(tileAcknowledged[rank][index], tileLastPublished[index]);
        int ownerSend = ring[(pos - step + 3) % 4];
        if (step)
          check(
              cudaStreamWaitEvent(stream, ringReduced[index - kRingTiles], 0));
        auto const* source =
            step == 0 ? input + ownerSend * shard + offset
                      : peers.data[rank] + (step - 1) * shard + offset;
        check(cudaMemcpyAsync(peers.data[next] + step * shard + offset, source,
                              length * 4, cudaMemcpyDeviceToDevice, stream));
        // The local event first denotes copy completion. This wait binds
        // that record before the same event is later recorded after reduction.
        check(cudaEventRecord(ringReduced[index], stream));
        check(cudaStreamWaitEvent(ringReduce, ringReduced[index], 0));
        check(cudaEventRecord(ringArrived[index], ringReduce));
        __atomic_store_n(tilePublished[rank][index], epoch, __ATOMIC_RELEASE);
        tileLastPublished[index] = epoch;
        cpuWait(tilePublished[prev][index], epoch);
        check(cudaStreamWaitEvent(ringReduce, ringPrevious[index], 0));
        int owner = ring[(pos - step + 2) % 4];
        auto* incoming = peers.data[rank] + step * shard + offset;
        int blocks = std::min(size_t(256), (length + 255) / 256);
        ringSum<<<blocks, 256, 0, ringReduce>>>(
            input + owner * shard + offset, incoming,
            step == 2 ? output + offset : incoming, length);
        check(cudaEventRecord(ringReduced[index], ringReduce));
        __atomic_store_n(tileAcknowledged[prev][index], epoch,
                         __ATOMIC_RELEASE);
      }
    }
    check(cudaStreamWaitEvent(stream, ringReduced[2 * kRingTiles + tiles - 1],
                              0));
    barrier<<<1, 1, 0, stream>>>(peers, rank, 2, generation);
    ++tileProtocolCalls;
  }
#endif
  void reduceRing(int op, float const* input, float* output, size_t count,
                  cudaStream_t stream) {
#ifdef UCCL_LITE_LOCAL_RS_TILE_PUBLICATION
    if (tileProtocol) {
      for (size_t index = 0; index < 3 * kRingTiles; ++index)
        cpuWait(tileAcknowledged[rank][index], tileLastPublished[index]);
      tileProtocol = false;
    }
#endif
    constexpr int ring[4] = {0, 1, 3, 2};
    int pos = rank ^ (rank >> 1), next = ring[(pos + 1) % 4];
    size_t shard = op == 3 ? count / 4 : count;
    bool banked = false;
#ifdef UCCL_LITE_LOCAL_RS_BANKS
#if !defined(UCCL_LITE_LOCAL_RING_IPC_EVENTS) || \
    !defined(UCCL_LITE_LOCAL_HIER_CREDIT) ||     \
    defined(UCCL_LITE_LOCAL_RS_MEMOP) || defined(UCCL_LITE_LOCAL_HIER_RS)
#error "Banked RS is an IPC-ring experiment requiring transition rendezvous"
#endif
    banked = op == 2 && shard * 16 >= (128UL << 20);
#endif
    size_t bank = banked ? (generation & 1) * 3 * shard : 0;

    size_t chunkBytes = 8UL << 20;
#ifdef UCCL_LITE_LOCAL_RS_CHUNK_BYTES
    static_assert(UCCL_LITE_LOCAL_RS_CHUNK_BYTES >= (8UL << 20) &&
                  UCCL_LITE_LOCAL_RS_CHUNK_BYTES <= (64UL << 20) &&
                  UCCL_LITE_LOCAL_RS_CHUNK_BYTES % 16 == 0);
#ifndef UCCL_LITE_LOCAL_RS_CHUNK_MIN_BYTES
#define UCCL_LITE_LOCAL_RS_CHUNK_MIN_BYTES 0
#endif
    static_assert(UCCL_LITE_LOCAL_RS_CHUNK_MIN_BYTES <= (1UL << 30));
    if (op == 2 && shard * 16 >= UCCL_LITE_LOCAL_RS_CHUNK_MIN_BYTES)
      chunkBytes = UCCL_LITE_LOCAL_RS_CHUNK_BYTES;
#endif
#ifdef UCCL_LITE_LOCAL_RS_LARGE_CHUNK_BYTES
    static_assert(UCCL_LITE_LOCAL_RS_LARGE_CHUNK_BYTES >= (8UL << 20) &&
                  UCCL_LITE_LOCAL_RS_LARGE_CHUNK_BYTES <= (64UL << 20) &&
                  UCCL_LITE_LOCAL_RS_LARGE_CHUNK_BYTES % 16 == 0);
    if (op == 2 && shard * 16 >= (256UL << 20))
      chunkBytes = UCCL_LITE_LOCAL_RS_LARGE_CHUNK_BYTES;
#endif
    size_t eventStride = kRingTiles;
#if defined(UCCL_LITE_LOCAL_RS_RING4M) || defined(UCCL_LITE_LOCAL_RS_SPLIT_TAIL)
#if !defined(UCCL_LITE_LOCAL_RING_IPC_EVENTS) ||                             \
    defined(UCCL_LITE_LOCAL_RS_HOST_READY) ||                                \
    defined(UCCL_LITE_LOCAL_RS_MEMOP) || defined(UCCL_LITE_LOCAL_RS_PULL) || \
    defined(UCCL_LITE_LOCAL_HIER_RS) ||                                      \
    defined(UCCL_LITE_LOCAL_RS_CHUNK_BYTES) ||                               \
    defined(UCCL_LITE_LOCAL_RS_LARGE_CHUNK_BYTES)
#error "Four-MiB RS chunks require the isolated IPC ring configuration"
#endif
    if (op == 2 && shard * 16 >= (128UL << 20)) {
#ifdef UCCL_LITE_LOCAL_RS_RING4M
      chunkBytes = 4UL << 20;
#endif
      eventStride = 2 * kRingTiles;
    }
    if (eventStride != ringEventStride) {
      // All previous bindings must be installed before an event index is
      // reinterpreted as a different logical stage. GPU stream completion
      // separately keeps record execution and scratch reuse ordered.
      for (int stage = 0; stage < 6; ++stage)
        cpuWait(ringAcknowledged[rank][stage], ringLastPublished[stage]);
      ringEventStride = eventStride;
    }
#endif
    size_t copyChunk = std::min(shard, chunkBytes / 4);
    size_t chunk = copyChunk;
#ifdef UCCL_LITE_LOCAL_RS_COPY_GROUP
#if !defined(UCCL_LITE_LOCAL_RING_IPC_EVENTS) ||                             \
    defined(UCCL_LITE_LOCAL_RS_RING4M) ||                                    \
    defined(UCCL_LITE_LOCAL_RS_HOST_READY) ||                                \
    defined(UCCL_LITE_LOCAL_RS_MEMOP) || defined(UCCL_LITE_LOCAL_RS_PULL) || \
    defined(UCCL_LITE_LOCAL_HIER_RS) ||                                      \
    defined(UCCL_LITE_LOCAL_RS_CHUNK_BYTES)
#error "Grouped RS copies require the isolated eight-MiB IPC ring"
#endif
    static_assert(UCCL_LITE_LOCAL_RS_COPY_GROUP == 2 ||
                  UCCL_LITE_LOCAL_RS_COPY_GROUP == 4);
    if (op == 2 && shard * 16 >= (128UL << 20))
      chunk = std::min(shard, copyChunk * UCCL_LITE_LOCAL_RS_COPY_GROUP);
#endif
    bool separateTail = false;
#ifdef UCCL_LITE_LOCAL_RS_GROUP_TAIL
#ifndef UCCL_LITE_LOCAL_RS_COPY_GROUP
#error "Separate final copy requires grouped copies"
#endif
    separateTail = op == 2 && chunk > copyChunk;
#endif
    size_t lastOffset = ((shard - 1) / copyChunk) * copyChunk;
    size_t tiles = separateTail ? (lastOffset + chunk - 1) / chunk + 1
                                : (shard + chunk - 1) / chunk;
    auto groupOffset = [&](size_t tile) {
      return separateTail && tile + 1 == tiles ? lastOffset : tile * chunk;
    };
    auto groupLength = [&](size_t offset) {
      if (separateTail && offset == lastOffset) return shard - lastOffset;
      return std::min(chunk, (separateTail ? lastOffset : shard) - offset);
    };
    bool splitTail = false;
#ifdef UCCL_LITE_LOCAL_RS_SPLIT_TAIL
#if defined(UCCL_LITE_LOCAL_RS_RING4M) || defined(UCCL_LITE_LOCAL_RS_COPY_GROUP)
#error "Split-tail RS requires the isolated eight-MiB IPC ring"
#endif
    splitTail = op == 2 && shard * 16 >= (128UL << 20);
#endif
    // Only the final phase splits its last full copy. Both pieces depend on
    // the same completed phase-one reduction; they cover disjoint outputs.
    auto phaseTiles = [&](int step) {
      return tiles + size_t(splitTail && step == 2);
    };
    auto phaseOffset = [&](int step, size_t tile) {
      return splitTail && step == 2 && tile == tiles ? shard - (1UL << 18)
                                                     : groupOffset(tile);
    };
    auto phaseLength = [&](int step, size_t tile, size_t offset) {
      if (splitTail && step == 2 && tile + 1 >= tiles)
        return tile == tiles ? size_t(1UL << 18)
                             : groupLength(offset) - (1UL << 18);
      return groupLength(offset);
    };
    if (phaseTiles(2) > eventStride ||
        (op == 3 ? 6 : 3) * eventStride > ringReduced.size())
      throw std::invalid_argument("ring extent exceeds capacity");
#ifdef UCCL_LITE_LOCAL_RING_IPC_EVENTS
    int prev = ring[(pos + 3) % 4];
    if (ringEpoch == UINT64_MAX)
      throw std::runtime_error("ring event epoch exhausted");
    uint64_t epoch = ++ringEpoch;
#endif
    check(cudaEventRecord(ringStart, stream));
    check(cudaStreamWaitEvent(ringReduce, ringStart, 0));
    auto finishReduce = [&](int step, size_t tile) {
      size_t offset = phaseOffset(step, tile);
      size_t length = phaseLength(step, tile, offset);
      size_t index = step * eventStride + tile;
#ifdef UCCL_LITE_LOCAL_RING_IPC_EVENTS
      check(cudaStreamWaitEvent(ringReduce, ringPrevious[index], 0));
#else
      ringWait<<<1, 1, 0, ringReduce>>>(peers.flags[rank] + 256 + index,
                                        generation);
#endif
      int owner = ring[(pos - step + 2) % 4];
      auto* incoming = peers.data[rank] + bank + step * shard + offset;
      auto* target =
          step == 2 ? output + (op == 3 ? rank * shard : 0) + offset : incoming;
      int blocks = std::min(size_t(256), (length + 255) / 256);
#ifdef UCCL_LITE_LOCAL_RS_GROUP_REDUCE_PIECES
#ifndef UCCL_LITE_LOCAL_RS_COPY_GROUP
#error "Piece reductions require grouped copies"
#endif
      if (op == 2 && chunk > copyChunk) {
        for (size_t piece = 0; piece < length; piece += copyChunk) {
          size_t part = std::min(copyChunk, length - piece);
          int partBlocks = std::min(size_t(256), (part + 255) / 256);
          ringSum<<<partBlocks, 256, 0, ringReduce>>>(
              input + owner * shard + offset + piece, incoming + piece,
              target + piece, part);
        }
      } else
#endif
        ringSum<<<blocks, 256, 0, ringReduce>>>(input + owner * shard + offset,
                                                incoming, target, length);
      check(cudaEventRecord(ringReduced[index], ringReduce));
    };
    for (int step = 0; step < 3; ++step) {
#ifdef UCCL_LITE_LOCAL_RING_IPC_EVENTS
      cpuWait(ringAcknowledged[rank][step], ringLastPublished[step]);
#endif
      for (size_t tile = 0; tile < phaseTiles(step); ++tile) {
        size_t offset = phaseOffset(step, tile);
        size_t length = phaseLength(step, tile, offset);
        size_t index = step * eventStride + tile;
        int owner = ring[(pos - step + 3) % 4];
        if (step)
          check(cudaStreamWaitEvent(
              stream,
              ringReduced[(step - 1) * eventStride + std::min(tile, tiles - 1)],
              0));
        auto const* source =
            step == 0 ? input + owner * shard + offset
                      : peers.data[rank] + bank + (step - 1) * shard + offset;
#ifdef UCCL_LITE_LOCAL_RS_COPY_GROUP
        for (size_t piece = 0; piece < length; piece += copyChunk)
          check(cudaMemcpyAsync(
              peers.data[next] + bank + step * shard + offset + piece,
              source + piece, std::min(copyChunk, length - piece) * 4,
              cudaMemcpyDeviceToDevice, stream));
#else
#ifdef UCCL_LITE_LOCAL_RING_PEER_COPY
        if (op == 2 && shard * 16 >= (128UL << 20))
          check(cudaMemcpyPeerAsync(
              peers.data[next] + bank + step * shard + offset, next, source,
              device, length * 4, stream));
        else
#endif
          check(cudaMemcpyAsync(peers.data[next] + bank + step * shard + offset,
                                source, length * 4, cudaMemcpyDeviceToDevice,
                                stream));
#endif
#ifdef UCCL_LITE_LOCAL_RING_IPC_EVENTS
        check(cudaEventRecord(ringArrived[index], stream));
#else
        ringSignal<<<1, 1, 0, stream>>>(peers.flags[next] + 256 + index,
                                        generation);
        finishReduce(step, tile);
#endif
      }
#ifdef UCCL_LITE_LOCAL_RING_IPC_EVENTS
      publishEvents(step, prev, epoch);
      for (size_t tile = 0; tile < phaseTiles(step); ++tile)
        finishReduce(step, tile);
      acknowledgeEvents(step, prev, epoch);
#endif
    }
    check(cudaStreamWaitEvent(
        stream, ringReduced[2 * eventStride + phaseTiles(2) - 1], 0));
    if (op == 3) {
      size_t gathered = capacity / 4;
      auto finishGather = [&](int step, size_t tile) {
        size_t offset = groupOffset(tile), length = groupLength(offset);
        size_t index = (3 + step) * eventStride + tile;
#ifdef UCCL_LITE_LOCAL_RING_IPC_EVENTS
        check(cudaStreamWaitEvent(ringReduce, ringPrevious[index], 0));
#else
        ringWait<<<1, 1, 0, ringReduce>>>(peers.flags[rank] + 256 + index,
                                          generation);
#endif
        int owner = ring[(pos - step + 3) % 4];
        int blocks = std::min(size_t(256), (length + 255) / 256);
        ringCopy<<<blocks, 256, 0, ringReduce>>>(
            peers.data[rank] + gathered + owner * shard + offset,
            output + owner * shard + offset, length);
        check(cudaEventRecord(ringReduced[index], ringReduce));
      };
      for (int step = 0; step < 3; ++step) {
#ifdef UCCL_LITE_LOCAL_RING_IPC_EVENTS
        cpuWait(ringAcknowledged[rank][step + 3], ringLastPublished[step + 3]);
#endif
        for (size_t tile = 0; tile < tiles; ++tile) {
          size_t offset = groupOffset(tile), length = groupLength(offset);
          size_t index = (3 + step) * eventStride + tile;
          int owner = ring[(pos - step + 4) % 4];
          if (step)
            check(cudaStreamWaitEvent(stream, ringReduced[index - eventStride],
                                      0));
          auto const* source =
              step == 0 ? output + rank * shard + offset
                        : peers.data[rank] + gathered + owner * shard + offset;
          check(cudaMemcpyAsync(
              peers.data[next] + gathered + owner * shard + offset, source,
              length * 4, cudaMemcpyDeviceToDevice, stream));
#ifdef UCCL_LITE_LOCAL_RING_IPC_EVENTS
          check(cudaEventRecord(ringArrived[index], stream));
#else
          ringSignal<<<1, 1, 0, stream>>>(peers.flags[next] + 256 + index,
                                          generation);
          finishGather(step, tile);
#endif
        }
#ifdef UCCL_LITE_LOCAL_RING_IPC_EVENTS
        publishEvents(step + 3, prev, epoch);
        for (size_t tile = 0; tile < tiles; ++tile) finishGather(step, tile);
        acknowledgeEvents(step + 3, prev, epoch);
#endif
      }
      check(cudaStreamWaitEvent(stream,
                                ringReduced[5 * eventStride + tiles - 1], 0));
    }
    // Before the predecessor can begin n+2, its completed n+1 reduction
    // must have received this rank's phase-zero n+1 contribution through
    // the three-step ring. That contribution follows our completion of n.
    // Two full receive banks therefore protect n until every reader retires.
    if (!banked) barrier<<<1, 1, 0, stream>>>(peers, rank, 2, generation);
  }
#endif
#ifdef UCCL_LITE_LOCAL_SMALL_GRAPH
  std::map<std::pair<int, size_t>, std::unique_ptr<LocalCopyGraph>> smallGraphs;
  void smallGraph(int op, void const* input, void* output, size_t count,
                  bool hierarchy, bool hierarchyAg, cudaStream_t stream) {
    cudaKernelNodeParams params{};
#ifdef UCCL_LITE_LOCAL_SMALL_DEVICE_EPOCH
    auto* sequence = peers.flags[rank] + 128;
    void* epochArgs[] = {&peers, &rank,     &input,   &output,
                         &count, &capacity, &sequence};
#endif
    void* flatArgs[] = {&peers,  &rank,  &op,       &input,
                        &output, &count, &capacity, &generation};
    void* pairArgs[] = {&peers, &rank,     &input,     &output,
                        &count, &capacity, &generation};
    params.gridDim = dim3(1);
#ifdef UCCL_LITE_LOCAL_AG_PACKET_BYTES
    if (op == 1 && !hierarchyAg)
      params.gridDim = dim3(std::min(size_t(16), (count / 2 + 255) / 256));
#endif
    params.blockDim = dim3(256);
    params.func = reinterpret_cast<void*>(packetCollect);
#ifdef UCCL_LITE_LOCAL_PACKET_SPECIALIZE
    params.func =
        reinterpret_cast<void*>(op == 1   ? packetCollectSpecialized<1>
                                : op == 2 ? packetCollectSpecialized<2>
                                          : packetCollectSpecialized<3>);
#endif
    params.kernelParams = flatArgs;
#ifdef UCCL_LITE_LOCAL_HIER_LL
    if (hierarchy) {
      params.func = reinterpret_cast<void*>(hierarchyPacket);
      params.kernelParams = pairArgs;
      params.blockDim = dim3(512);
#ifdef UCCL_LITE_LOCAL_TREE_LL
      if (count * 4 <= UCCL_LITE_LOCAL_TREE_MAX_BYTES)
        params.func = reinterpret_cast<void*>(treePacket);
#endif
    }
#endif
#ifdef UCCL_LITE_LOCAL_HIER_AG
    if (hierarchyAg) {
      params.func = reinterpret_cast<void*>(hierarchyGather);
      params.kernelParams = pairArgs;
    }
#endif
#ifdef UCCL_LITE_LOCAL_SMALL_DEVICE_EPOCH
    params.func = reinterpret_cast<void*>(
        count * 4 <= UCCL_LITE_LOCAL_TREE_MAX_BYTES ? treePacketEpoch
                                                    : hierarchyPacketEpoch);
    params.kernelParams = epochArgs;
#ifdef UCCL_LITE_LOCAL_PACKET_THREADS
    size_t units =
        count / (count * 4 <= UCCL_LITE_LOCAL_TREE_MAX_BYTES ? 2 : 4);
    unsigned threads = 32;
    while (threads < 1024 && size_t(threads) < units) threads *= 2;
    params.blockDim = dim3(threads);
#endif
#endif
    auto& entry = smallGraphs[{op, count}];
    if (!entry) {
      entry = std::make_unique<LocalCopyGraph>();
      check(cudaGraphCreate(&entry->graph, 0));
      check(cudaGraphAddKernelNode(&entry->kernel, entry->graph, nullptr, 0,
                                   &params));
      check(cudaGraphAddEventRecordNode(&entry->copy, entry->graph,
                                        &entry->kernel, 1, completion));
      check(cudaGraphInstantiate(&entry->executable, entry->graph, nullptr,
                                 nullptr, 0));
    } else
#ifdef UCCL_LITE_LOCAL_SMALL_DEVICE_EPOCH
        if (entry->boundInput != input || entry->boundOutput != output)
#endif
    {
      // Updates affect future launches, including queued pointer changes.
      check(cudaGraphExecKernelNodeSetParams(entry->executable, entry->kernel,
                                             &params));
    }
    entry->boundInput = input;
    entry->boundOutput = output;
    check(cudaGraphLaunch(entry->executable, stream));
  }
#endif
  bool ready = false;
  Impl(NetworkId id, int rank, int device, size_t capacity,
       std::array<int, 2> numaNodes)
      : id(id),
        rank(rank),
        device(device),
        capacity(capacity),
        numaNodes(numaNodes) {}
};
NetworkId PeerPacketCollectives::createUniqueId() {
  return mscclpp::TcpBootstrap::createUniqueId();
}
PeerPacketCollectives::PeerPacketCollectives(NetworkId id, int rank, int device,
                                             int rail, size_t capacity)
    : PeerPacketCollectives(id, rank, device, rail, capacity, {0, 1}) {}
PeerPacketCollectives::PeerPacketCollectives(NetworkId id, int rank, int device,
                                             int rail, size_t capacity,
                                             std::array<int, 2> numaNodes)
    : impl_(std::make_unique<Impl>(id, rank, device, capacity, numaNodes)) {}
void PeerPacketCollectives::prepare() {
  auto& s = *impl_;
  if (s.ready) return;
  check(cudaSetDevice(s.device));
  s.bootstrap = std::make_shared<mscclpp::TcpBootstrap>(s.rank, 4);
  s.bootstrap->initialize(s.id, 120);
#ifdef UCCL_LITE_LOCAL_DRIVER_TREE
  check(cudaGetFuncBySymbol(&s.treeFunction,
                            reinterpret_cast<void const*>(treePacket)));
#endif
#ifdef UCCL_LITE_LOCAL_AG_SHARED_INPUT
  check(cudaFuncSetAttribute(
      serialCollect, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 << 10));
#endif
#ifdef UCCL_LITE_LOCAL_HOST_AG
  s.hostCommunicator = std::make_unique<mscclpp::Communicator>(s.bootstrap);
  size_t packetBytes = std::min(s.capacity / 4, size_t(1) << 20);
  s.hostPackets = std::make_unique<multinode::SharedHostPackets>(
      *s.hostCommunicator, s.rank, packetBytes, s.device, s.numaNodes);
#ifdef UCCL_LITE_LOCAL_AG_HOST_READY
  // Separate control allocation: no collision with host-payload or ring fields.
  s.hostReadyPackets = std::make_unique<multinode::SharedHostPackets>(
      *s.hostCommunicator, s.rank, 16, s.device, s.numaNodes);
#ifdef UCCL_LITE_LOCAL_RS_TILE_PUBLICATION
  if (4096 + 2 * 3 * kRingTiles * 64 > s.hostReadyPackets->bytes())
    throw std::runtime_error(
        "per-tile publication exceeds private control storage");
  for (int rank = 0; rank < 4; ++rank)
    for (size_t index = 0; index < 3 * kRingTiles; ++index) {
      auto* cell = s.hostReadyPackets->host(rank / 2) + 4096 +
                   ((rank % 2) * 3 * kRingTiles + index) * 64;
      s.tilePublished[rank][index] = reinterpret_cast<uint64_t*>(cell);
      s.tileAcknowledged[rank][index] = reinterpret_cast<uint64_t*>(cell + 8);
    }
#endif
#endif
#ifdef UCCL_LITE_LOCAL_HOST_AG_WIDE
  // Four local ranks have no network rows. Reuse that payload space for two
  // banks of two local source rows per NUMA region. The original storage,
  // registration, control layout and capacity validation stay unchanged.
  packetBytes = std::min(s.capacity / 4, size_t(4) << 20);
  if (HostGather::slots * HostGather::rowsPerGroup * packetBytes >
      s.hostPackets->bytes() - s.hostPackets->payloadOffset(0))
    throw std::runtime_error("local host packet layout exceeds shared storage");
#endif
  s.hostGather.elements = packetBytes / 4;
  for (int slot = 0; slot < HostGather::slots; ++slot) {
    for (int rank = 0; rank < 4; ++rank) {
#ifdef UCCL_LITE_LOCAL_HOST_AG_WIDE
      s.hostGather.rows[slot][rank] = reinterpret_cast<float*>(
          s.hostPackets->mapped(rank / 2) + s.hostPackets->payloadOffset(0) +
          (slot * HostGather::rowsPerGroup + rank % 2) * packetBytes);
#else
      s.hostGather.rows[slot][rank] = reinterpret_cast<float*>(
          s.hostPackets->row(rank, slot, packetBytes, true));
#endif
    }
#ifdef UCCL_LITE_LOCAL_HOST_REPLICA
    for (int source = 0; source < 2; ++source)
      s.hostGather.replicas[slot][source] = reinterpret_cast<float*>(
          s.hostPackets->mapped(s.rank / 2) + s.hostPackets->payloadOffset(0) +
          (slot * HostGather::rowsPerGroup + 2 + source) * packetBytes);
#endif
    for (int group = 0; group < 2; ++group) {
      size_t offset =
          reinterpret_cast<char*>(s.hostPackets->flag(group, slot, 0)) -
          s.hostPackets->host(group);
      s.hostGather.flags[slot][group] =
          reinterpret_cast<uint64_t*>(s.hostPackets->mapped(group) + offset);
    }
  }
#endif
  check(cudaMalloc(&s.peers.data[s.rank],
                   2 * s.capacity + 4096 + kPacketReserveBytes));
  s.peers.flags[s.rank] = reinterpret_cast<unsigned long long*>(
      reinterpret_cast<char*>(s.peers.data[s.rank]) + 2 * s.capacity);
  check(cudaMemset(s.peers.flags[s.rank], 0, 4096 + kPacketReserveBytes));
  check(cudaDeviceSynchronize());
  std::array<cudaIpcMemHandle_t, 4> handles{};
  check(cudaIpcGetMemHandle(&handles[s.rank], s.peers.data[s.rank]));
  s.bootstrap->allGather(handles.data(), sizeof(handles[0]));
  for (int r = 0; r < 4; ++r) {
    if (r == s.rank) continue;
    check(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&s.peers.data[r]),
                               handles[r], cudaIpcMemLazyEnablePeerAccess));
    s.peers.flags[r] = reinterpret_cast<unsigned long long*>(
        reinterpret_cast<char*>(s.peers.data[r]) + 2 * s.capacity);
  }
#ifdef UCCL_LITE_LOCAL_RING
  check(cudaStreamCreateWithFlags(&s.ringReduce, cudaStreamNonBlocking));
  check(cudaEventCreateWithFlags(&s.ringStart, cudaEventDisableTiming));
  for (auto& event : s.ringReduced)
    check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
#endif
#ifdef UCCL_LITE_LOCAL_RING_IPC_EVENTS
  if (s.capacity >= kArRingMinimum) {
#ifdef UCCL_LITE_LOCAL_HOST_PER_SOURCE
    size_t control = s.hostPackets->payloadOffset(0) - 2048;
#elif defined(UCCL_LITE_LOCAL_HOST_REPLICA)
    size_t control = reinterpret_cast<char*>(s.hostPackets->flag(
                         0, multinode::SharedHostPackets::slots - 2, 0)) -
                     s.hostPackets->host(0);
#else
    size_t control = s.hostPackets->bytes() - 2048;
#endif
    size_t payloadEnd = s.hostPackets->payloadOffset(0) +
                        HostGather::slots * HostGather::rowsPerGroup *
                            s.hostGather.elements * 4;
    if (control < payloadEnd &&
        control + 2048 > s.hostPackets->payloadOffset(0))
      throw std::runtime_error("ring CPU control overlaps host gather payload");
    for (int rank = 0; rank < 4; ++rank)
      for (int stage = 0; stage < 6; ++stage) {
        auto* base = s.hostPackets->host(rank / 2) + control +
                     ((rank % 2) * 12 + stage * 2) * 64;
        s.ringPublished[rank][stage] = reinterpret_cast<uint64_t*>(base);
        s.ringAcknowledged[rank][stage] =
            reinterpret_cast<uint64_t*>(base + 64);
      }
    std::array<std::array<cudaIpcEventHandle_t, 6 * kRingTiles>, 4> handles{};
    for (size_t i = 0; i < s.ringArrived.size(); ++i) {
      check(cudaEventCreateWithFlags(
          &s.ringArrived[i], cudaEventDisableTiming | cudaEventInterprocess));
      check(cudaIpcGetEventHandle(&handles[s.rank][i], s.ringArrived[i]));
    }
    s.bootstrap->allGather(handles.data(), sizeof(handles[0]));
    constexpr int ring[4] = {0, 1, 3, 2};
    int pos = s.rank ^ (s.rank >> 1), prev = ring[(pos + 3) % 4];
    for (size_t i = 0; i < s.ringPrevious.size(); ++i)
      check(cudaIpcOpenEventHandle(&s.ringPrevious[i], handles[prev][i]));
#ifdef UCCL_LITE_LOCAL_HIER_RS
    for (size_t i = 0; i < s.hierarchyPrevious.size(); ++i) {
      int partner = s.rank ^ (i / kRingTiles < 2 ? 1 : 2);
      if (partner == prev)
        s.hierarchyPrevious[i] = s.ringPrevious[i];
      else
        check(cudaIpcOpenEventHandle(&s.hierarchyPrevious[i],
                                     handles[partner][i]));
    }
#endif
  }
#endif
#ifdef UCCL_LITE_LOCAL_HOST_REPLICA
  s.replicaGather = s.hostGather;
#ifdef UCCL_LITE_LOCAL_HOST_REPLICA_BYTES
  static_assert(UCCL_LITE_LOCAL_HOST_REPLICA_BYTES >= 16 &&
                UCCL_LITE_LOCAL_HOST_REPLICA_BYTES % 16 == 0);
  size_t replicaBytes = std::min(s.hostGather.elements * 4,
                                 size_t(UCCL_LITE_LOCAL_HOST_REPLICA_BYTES));
  s.replicaGather.elements = replicaBytes / 4;
  for (int slot = 0; slot < HostGather::slots; ++slot) {
    for (int rank = 0; rank < 4; ++rank)
      s.replicaGather.rows[slot][rank] = reinterpret_cast<float*>(
          s.hostPackets->mapped(rank / 2) + s.hostPackets->payloadOffset(0) +
          (slot * HostGather::rowsPerGroup + rank % 2) * replicaBytes);
    for (int source = 0; source < 2; ++source)
      s.replicaGather.replicas[slot][source] = reinterpret_cast<float*>(
          s.hostPackets->mapped(s.rank / 2) + s.hostPackets->payloadOffset(0) +
          (slot * HostGather::rowsPerGroup + 2 + source) * replicaBytes);
  }
#endif
#ifdef UCCL_LITE_LOCAL_HOST_PER_SOURCE
#ifndef UCCL_LITE_LOCAL_HOST_REPLICA_BYTES
#error "Per-source credits require host layout transition rendezvous"
#endif
  if (4096 + HostGather::slots * 32 * 64 >
      s.hostPackets->payloadOffset(0) - 2048)
    throw std::runtime_error("per-source controls overlap IPC event metadata");
  for (int slot = 0; slot < HostGather::slots; ++slot)
    for (int group = 0; group < 2; ++group)
      s.replicaGather.flags[slot][group] = reinterpret_cast<uint64_t*>(
          s.hostPackets->mapped(group) + 4096 + slot * 32 * 64);
#endif
#ifdef UCCL_LITE_LOCAL_HOST_GPU_PROGRESS
  int resident = 0;
  cudaDeviceProp property{};
  check(cudaGetDeviceProperties(&property, s.device));
  check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &resident, replicaGatherKernel, 256, 0));
  if (!property.cooperativeLaunch ||
      resident * property.multiProcessorCount < 129)
    throw std::runtime_error(
        "direct host pipeline exceeds cooperative residency");
#endif
  check(cudaMalloc(&s.replicaCounters, sizeof(ReplicaCounters)));
#endif
#ifdef UCCL_LITE_LOCAL_RS_MEMOP
  if (s.capacity >= (128UL << 20)) {
    int supported = 0;
    if (cuDeviceGetAttribute(&supported,
                             CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS,
                             s.device) != CUDA_SUCCESS ||
        !supported)
      throw std::runtime_error("64-bit stream memory operations unsupported");
    constexpr int ring[4] = {0, 1, 3, 2};
    int pos = s.rank ^ (s.rank >> 1), next = ring[(pos + 1) % 4];
    if (
#ifdef UCCL_LITE_LOCAL_RS_MEMOP32
        cuStreamWriteValue32(
#else
        cuStreamWriteValue64(
#endif
            reinterpret_cast<CUstream>(s.ringReduce),
            reinterpret_cast<CUdeviceptr>(s.peers.flags[next] + 448), 1,
            0) != CUDA_SUCCESS)
      throw std::runtime_error("IPC peer stream value probe failed");
    check(cudaStreamSynchronize(s.ringReduce));
    s.bootstrap->barrier();
    unsigned long long observed = 0;
    check(cudaMemcpy(&observed, s.peers.flags[s.rank] + 448, sizeof(observed),
                     cudaMemcpyDeviceToHost));
    if (observed != 1)
      throw std::runtime_error("IPC peer stream value probe mismatch");
    check(cudaMemset(s.peers.flags[s.rank] + 448, 0, sizeof(observed)));
    check(cudaDeviceSynchronize());
  }
#endif
#ifdef UCCL_LITE_LOCAL_AG_SEGMENTED
  {
    int resident = 0;
    cudaDeviceProp property{};
    check(cudaGetDeviceProperties(&property, s.device));
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, segmentedGather, 512, 0));
    if (!property.cooperativeLaunch ||
        resident * property.multiProcessorCount < 4)
      throw std::runtime_error(
          "segmented gather cooperative residency unsupported");
  }
#endif
#ifdef UCCL_LITE_LOCAL_AG_PULL
  {
    int resident = 0;
    cudaDeviceProp property{};
    check(cudaGetDeviceProperties(&property, s.device));
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, pullGather,
                                                        256, 0));
    if (!property.cooperativeLaunch ||
        resident * property.multiProcessorCount < 16)
      throw std::runtime_error(
          "private pull gather cooperative residency unsupported");
  }
#endif
#ifdef UCCL_LITE_LOCAL_AG_HIER_SM
  {
    int resident = 0;
    cudaDeviceProp property{};
    check(cudaGetDeviceProperties(&property, s.device));
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, hierarchicalGather, 256, 0));
    if (!property.cooperativeLaunch ||
        resident * property.multiProcessorCount < 4)
      throw std::runtime_error(
          "hierarchical gather cooperative residency unsupported");
#ifdef UCCL_LITE_LOCAL_AG_HIER_ONE_CTA64
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, hierarchicalGather, 1024, 0));
    if (resident < 1)
      throw std::runtime_error(
          "single-CTA hierarchical gather residency unsupported");
#endif
  }
#endif
#ifdef UCCL_LITE_LOCAL_RS_HOST_READY
  if (s.capacity >= UCCL_LITE_LOCAL_RS_HOST_READY_MIN_BYTES) {
    int resident = 0;
    cudaDeviceProp property{};
    check(cudaGetDeviceProperties(&property, s.device));
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, ringHostWaitSum, 256, 0));
    if (!property.cooperativeLaunch || !resident)
      throw std::runtime_error(
          "host-ready RS cooperative residency unsupported");
    s.ringHostBlocks = std::min(256, resident * property.multiProcessorCount);
#ifdef UCCL_LITE_LOCAL_RS_NATIVE_GRAPH
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, ringGraphWaitSum, 256, 0));
    if (resident < 1)
      throw std::runtime_error("native graph residency unsupported");
    s.nativeGraphBlocks =
        std::min(256, resident * property.multiProcessorCount);
#endif
    for (int rank = 0; rank < 4; ++rank)
      s.ringHostFlags[rank] =
          reinterpret_cast<unsigned*>(s.hostReadyPackets->mapped(rank / 2)) +
          64 + (rank % 2) * 128;
#ifndef UCCL_LITE_LOCAL_RS_GPU_NATIVE
    // Probe the same address class used by the measured path.
    if (cuStreamWriteValue32(
            reinterpret_cast<CUstream>(s.ringReduce),
            reinterpret_cast<CUdeviceptr>(s.ringReadyAddress(s.rank, 96)), 1,
            0) != CUDA_SUCCESS)
      throw std::runtime_error("host-ready stream value probe failed");
    check(cudaStreamSynchronize(s.ringReduce));
#ifdef UCCL_LITE_LOCAL_RS_GPU_READY
    unsigned observed = 0;
    check(cudaMemcpy(&observed, s.ringReadyAddress(s.rank, 96),
                     sizeof(observed), cudaMemcpyDeviceToHost));
    if (observed != 1)
      throw std::runtime_error("GPU-ready stream value probe mismatch");
    check(cudaMemset(s.ringReadyAddress(s.rank, 96), 0, sizeof(observed)));
#else
    auto* probe =
        reinterpret_cast<unsigned*>(s.hostReadyPackets->host(s.rank / 2)) + 64 +
        (s.rank % 2) * 128 + 96;
    if (__atomic_load_n(probe, __ATOMIC_ACQUIRE) != 1)
      throw std::runtime_error("host-ready stream value probe mismatch");
    __atomic_store_n(probe, 0, __ATOMIC_RELEASE);
#endif
#endif
  }
#endif
  check(cudaEventCreateWithFlags(&s.completion, cudaEventDisableTiming));
  s.bootstrap->barrier();
  s.ready = true;
}
void PeerPacketCollectives::enqueue(NetworkOperation operation,
                                    void const* input, void* output,
                                    size_t count, cudaStream_t stream) {
  auto& s = *impl_;
  int op = static_cast<int>(operation);
  size_t bytes = count * (op == 2 ? 16 : 4);
  if (!s.ready || !input || !output || bytes > s.capacity)
    throw std::invalid_argument("invalid local collective submission");
#ifdef UCCL_LITE_LOCAL_STREAM_ID
  unsigned long long streamId;
  check(cudaStreamGetId(stream, &streamId));
  // IDs distinguish recycled stream handles. A reused raw pointer alone is
  // not sufficient evidence that the earlier submission is ordered here.
  if (s.generation && streamId != s.previousStreamId)
    check(cudaStreamWaitEvent(stream, s.completion, 0));
  s.previousStreamId = streamId;
#else
  if (s.generation) check(cudaStreamWaitEvent(stream, s.completion, 0));
#endif
  if (s.generation == kGenerationLimit) {
    synchronize();
    s.bootstrap->barrier();
#ifdef UCCL_LITE_LOCAL_AG_HOST_READY
    // Epoch-kernel counters reset with the GPU flag page. Retire and clear
    // their host publications in the same quiescent wrap boundary.
#ifdef UCCL_LITE_LOCAL_RS_HOST_READY
    if (!(s.rank % 2))
      std::memset(s.hostReadyPackets->host(s.rank / 2), 0, 4096);
#else
    if (!s.rank) std::memset(s.hostReadyPackets->host(0), 0, 64);
#endif
#endif
    check(cudaMemset(s.peers.flags[s.rank], 0, 4096 + kPacketReserveBytes));
    check(cudaDeviceSynchronize());
    s.bootstrap->barrier();
    s.generation = 0;
  }
  ++s.generation;
  size_t extent = count * (op == 3 ? 4 : 16);
  size_t packetLimit = op == 3 ? 4096 : op == 1 ? 8192 : 16384;
#ifdef UCCL_LITE_LOCAL_AG_PACKET_BYTES
  if (op == 1) packetLimit = UCCL_LITE_LOCAL_AG_PACKET_BYTES;
#endif
  bool hierarchyAg = false;
#ifdef UCCL_LITE_LOCAL_HIER_AG
#ifndef UCCL_LITE_LOCAL_HIER_AG_MAX_BYTES
#define UCCL_LITE_LOCAL_HIER_AG_MAX_BYTES (64UL << 10)
#endif
  static_assert(UCCL_LITE_LOCAL_HIER_AG_MAX_BYTES >= (16UL << 10) &&
                UCCL_LITE_LOCAL_HIER_AG_MAX_BYTES <= (64UL << 10));
  hierarchyAg = op == 1 && extent >= (16UL << 10) &&
                extent <= UCCL_LITE_LOCAL_HIER_AG_MAX_BYTES;
#endif
  bool hostGather = false;
#ifdef UCCL_LITE_LOCAL_HOST_AG
#ifndef UCCL_LITE_LOCAL_HOST_AG_MIN_BYTES
#define UCCL_LITE_LOCAL_HOST_AG_MIN_BYTES (1UL << 20)
#endif
  hostGather = op == 1 && extent >= UCCL_LITE_LOCAL_HOST_AG_MIN_BYTES &&
#ifdef UCCL_LITE_LOCAL_HOST_REPLICA
               extent <= (1UL << 30);
#else
               extent <= (16UL << 20);
#endif
#endif
#if (defined(UCCL_LITE_LOCAL_PACKET_CREDIT) || \
     defined(UCCL_LITE_LOCAL_SERIAL_CREDIT) || \
     defined(UCCL_LITE_LOCAL_HIER_AG)) &&      \
    !defined(UCCL_LITE_LOCAL_HIER_CREDIT)
#error "Packet credits require the shared transition rendezvous"
#endif
#ifdef UCCL_LITE_LOCAL_HIER_CREDIT
#ifndef UCCL_LITE_LOCAL_HIER_LL_BYTES
#define UCCL_LITE_LOCAL_HIER_LL_BYTES (32UL << 10)
#endif
  static_assert(UCCL_LITE_LOCAL_HIER_LL_BYTES <= (32UL << 10),
                "hierarchy packet banks exceed reserved storage");
  bool hierarchy = op == 3 && extent <= UCCL_LITE_LOCAL_HIER_LL_BYTES;
  int packetMode = hierarchy ? 1 : 0;
#ifdef UCCL_LITE_LOCAL_RS_NATIVE_BANKS
  // A predecessor cannot start n+2 until its n+1 result has consumed this
  // rank's phase-zero n+1 contribution, which follows local completion of n.
  // Two banks therefore protect asynchronous consecutive calls. Rendezvous
  // before changing operation/count and reinterpreting their layout.
  if (op == 2 && extent >= UCCL_LITE_LOCAL_RS_HOST_READY_MIN_BYTES)
    packetMode = 13;
#endif
#ifdef UCCL_LITE_LOCAL_RS_BANKS
  if (op == 2 && extent >= (128UL << 20)) packetMode = 12;
#endif
#ifdef UCCL_LITE_LOCAL_SM_BANKS
  bool smBanked = op == 3 && !hierarchy && extent <= (2UL << 20) &&
                  extent <= s.capacity / 2;
  if (smBanked) packetMode = 10;
#endif
#ifdef UCCL_LITE_LOCAL_PACKET_CREDIT
  if (!hierarchy && extent <= packetLimit) packetMode = op + 1;
#endif
#ifdef UCCL_LITE_LOCAL_SERIAL_CREDIT
  if (op != 3 && extent > packetLimit && extent <= (512UL << 10) && !hostGather)
    packetMode = op + 4;
#endif
  if (hierarchyAg) packetMode = 7;
#ifdef UCCL_LITE_LOCAL_HOST_REPLICA_BYTES
  // Layout changes reinterpret source rows; join all old readers first.
  if (hostGather) packetMode = extent > (16UL << 20) ? 9 : 8;
#endif
  if (s.generation > 1 && (packetMode != s.previousPacketMode ||
                           (packetMode && count != s.previousCount)))
    barrier<<<1, 1, 0, stream>>>(s.peers, s.rank, 3, s.generation);
  s.previousPacketMode = packetMode;
  s.previousCount = count;
#endif
  int treeThreads = 512, hierarchyThreads = 512;
#ifdef UCCL_LITE_LOCAL_PACKET_THREADS
  treeThreads = 32;
  while (treeThreads < 1024 && size_t(treeThreads) < count / 2)
    treeThreads *= 2;
  hierarchyThreads = 32;
  while (hierarchyThreads < 1024 && size_t(hierarchyThreads) < count / 4)
    hierarchyThreads *= 2;
#endif
  bool graphCompletion = false;
#ifdef UCCL_LITE_LOCAL_AG_HIER_SM
  if (op == 1 && (extent == (64UL << 10) || extent == (128UL << 10))) {
    size_t bankElements = s.capacity / 4;
    int blocks = 4, threads = 256;
#ifdef UCCL_LITE_LOCAL_AG_HIER_ONE_CTA64
    if (extent == (64UL << 10)) {
      blocks = 1;
      threads = 1024;
    }
#endif
    void* args[] = {&s.peers, &s.rank,       &input,       &output,
                    &count,   &bankElements, &s.generation};
    check(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(hierarchicalGather), dim3(blocks),
        dim3(threads), args, 0, stream));
  } else
#endif
#ifdef UCCL_LITE_LOCAL_AG_PULL
      if (op == 1 && (extent == (64UL << 10) || extent == (128UL << 10))) {
    size_t bankElements = s.capacity / 4;
    void* args[] = {&s.peers, &s.rank,       &input,       &output,
                    &count,   &bankElements, &s.generation};
    check(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(pullGather),
                                      dim3(16), dim3(256), args, 0, stream));
  } else
#endif
#ifdef UCCL_LITE_LOCAL_AG_SEGMENTED
      if (op == 1 && extent >= (32UL << 10) && extent <= (128UL << 10)) {
    size_t bankElements = s.capacity / 4;
    void* args[] = {&s.peers, &s.rank,       &input,       &output,
                    &count,   &bankElements, &s.generation};
    int blocks = (count / 4 + 511) / 512;
    check(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(segmentedGather),
                                      dim3(blocks), dim3(512), args, 0,
                                      stream));
  } else
#endif
#ifdef UCCL_LITE_LOCAL_AG_EPOCH_GRAPH
      if (op == 1 &&
#ifdef UCCL_LITE_LOCAL_AG_EPOCH_RANGE
          extent >= (32UL << 10) && extent <= (256UL << 10)
#else
          (extent == (64UL << 10) || extent == (128UL << 10))
#endif
      ) {
    s.gatherEpochGraph(input, output, count, extent == (64UL << 10) ? 128 : 256,
                       stream);
    graphCompletion = true;
  } else
#endif
#ifdef UCCL_LITE_LOCAL_SMALL_GRAPH
#ifndef UCCL_LITE_LOCAL_HIER_CREDIT
#error "Small graphs require the shared transition protocol"
#endif
#ifdef UCCL_LITE_LOCAL_SMALL_DEVICE_EPOCH
      if (op == 3 && hierarchy) {
#else
      if (extent <= packetLimit || hierarchy || hierarchyAg) {
#endif
    s.smallGraph(op, input, output, count, hierarchy, hierarchyAg, stream);
    graphCompletion = true;
  } else
#endif
#ifdef UCCL_LITE_LOCAL_HIER_LL
#ifdef UCCL_LITE_LOCAL_HIER_CREDIT
      if (hierarchy) {
#else
      if (op == 3 && extent <= (64UL << 10)) {
#endif
#ifdef UCCL_LITE_LOCAL_TREE_LL
#ifndef UCCL_LITE_LOCAL_HIER_CREDIT
#error "Tree packets require the shared transition rendezvous"
#endif
    if (extent <= UCCL_LITE_LOCAL_TREE_MAX_BYTES) {
#ifdef UCCL_LITE_LOCAL_DRIVER_TREE
      void* args[] = {&s.peers, &s.rank,     &input,       &output,
                      &count,   &s.capacity, &s.generation};
      CUresult result =
          cuLaunchKernel(s.treeFunction, 1, 1, 1, treeThreads, 1, 1, 0,
                         reinterpret_cast<CUstream>(stream), args, nullptr);
      if (result != CUDA_SUCCESS)
        throw std::runtime_error("driver tree launch failed");
#else
      treePacket<<<1, treeThreads, 0, stream>>>(
          s.peers, s.rank, static_cast<float const*>(input),
          static_cast<float*>(output), count, s.capacity, s.generation);
#endif
    } else
#endif
      hierarchyPacket<<<1, hierarchyThreads, 0, stream>>>(
          s.peers, s.rank, static_cast<float const*>(input),
          static_cast<float*>(output), count, s.capacity, s.generation);
  } else
#endif
#ifdef UCCL_LITE_LOCAL_HIER_AG
      if (hierarchyAg) {
    hierarchyGather<<<1, 256, 0, stream>>>(
        s.peers, s.rank, static_cast<float const*>(input),
        static_cast<float*>(output), count, s.capacity, s.generation);
  } else
#endif
      if (extent <= packetLimit) {
#ifdef UCCL_LITE_LOCAL_PACKET_SPECIALIZE
    int flatBlocks = 1;
#ifdef UCCL_LITE_LOCAL_AG_PACKET_BYTES
    if (op == 1) flatBlocks = std::min(size_t(16), (count / 2 + 255) / 256);
#endif
    if (op == 1)
      packetCollectSpecialized<1><<<flatBlocks, 256, 0, stream>>>(
          s.peers, s.rank, op, static_cast<float const*>(input),
          static_cast<float*>(output), count, s.capacity, s.generation);
    else if (op == 2)
      packetCollectSpecialized<2><<<1, 256, 0, stream>>>(
          s.peers, s.rank, op, static_cast<float const*>(input),
          static_cast<float*>(output), count, s.capacity, s.generation);
    else
      packetCollectSpecialized<3><<<1, 256, 0, stream>>>(
          s.peers, s.rank, op, static_cast<float const*>(input),
          static_cast<float*>(output), count, s.capacity, s.generation);
#else
    packetCollect<<<1, 256, 0, stream>>>(
        s.peers, s.rank, op, static_cast<float const*>(input),
        static_cast<float*>(output), count, s.capacity, s.generation);
#endif
  }
#ifdef UCCL_LITE_LOCAL_HIER_SM
  else if (op == 3 && extent <= (2UL << 20)) {
    int blocks = extent <= (128UL << 10) ? 1 : 8;
    hierarchySm<<<blocks, 512, 0, stream>>>(
        s.peers, s.rank, static_cast<float const*>(input),
        static_cast<float*>(output), count, s.generation
#ifdef UCCL_LITE_LOCAL_SM_BANKS
        ,
        smBanked
#endif
    );
  }
#endif
#ifdef UCCL_LITE_LOCAL_HOST_AG
  else if (hostGather) {
    HostGather* activeHost = &s.hostGather;
#ifdef UCCL_LITE_LOCAL_HOST_REPLICA
    if (extent > (16UL << 20)) activeHost = &s.replicaGather;
#endif
    uint64_t& activeGeneration =
#ifdef UCCL_LITE_LOCAL_HOST_PER_SOURCE
        extent > (16UL << 20) ? s.replicaGeneration :
#endif
                              s.hostGeneration;
    size_t tiles = (count + activeHost->elements - 1) / activeHost->elements;
    if (activeGeneration > UINT64_MAX - tiles)
      throw std::runtime_error("host packet generation exhausted");
    int hostBlocks = 16;
#ifdef UCCL_LITE_LOCAL_HOST_AG_SMALL_BLOCKS
    static_assert(UCCL_LITE_LOCAL_HOST_AG_SMALL_BLOCKS >= 4 &&
                  UCCL_LITE_LOCAL_HOST_AG_SMALL_BLOCKS <= 32 &&
                  UCCL_LITE_LOCAL_HOST_AG_SMALL_BLOCKS % 4 == 0);
    if (extent <= (512UL << 10))
      hostBlocks = UCCL_LITE_LOCAL_HOST_AG_SMALL_BLOCKS;
#endif
#ifdef UCCL_LITE_LOCAL_HOST_REPLICA
    if (extent > (16UL << 20)) {
#ifndef UCCL_LITE_LOCAL_HOST_GPU_PROGRESS
      if (s.rank % 2 == 0) {
        if (!s.hostCopy)
          s.hostCopy = std::make_unique<LocalHostCopy>(
              *s.hostPackets, s.rank / 2, activeHost->elements * 4,
              HostGather::slots, s.numaNodes[s.rank / 2]);
        s.hostCopy->enqueue(activeGeneration, count);
      }
#endif
      void* args[] = {activeHost, &s.replicaCounters, &s.rank, &input, &output,
                      &count,     &activeGeneration};
      check(cudaLaunchCooperativeKernel(
          reinterpret_cast<void*>(replicaGatherKernel),
#ifdef UCCL_LITE_LOCAL_HOST_GPU_PROGRESS
          dim3(129),
#else
          dim3(41),
#endif
          dim3(256), args, 0, stream));
    } else
#endif
      hostGatherKernel<<<hostBlocks, 256, 0, stream>>>(
          s.hostGather, s.rank, static_cast<float const*>(input),
          static_cast<float*>(output), count, activeGeneration);
    activeGeneration += tiles;
  }
#endif
  else if (extent <= (512UL << 10)) {
    int serialThreads = 512;
#ifdef UCCL_LITE_LOCAL_SERIAL_REFERENCE_THREADS
    if (op != 3) serialThreads = op == 1 && extent == (64UL << 10) ? 128 : 256;
#endif
    size_t sharedInputBytes = 0;
#ifdef UCCL_LITE_LOCAL_AG_SHARED_INPUT
    if (op == 1 && (count == 4096 || count == 16384))
      sharedInputBytes = count * 4;
#endif
#ifdef UCCL_LITE_LOCAL_AG_GRAPH_OUTPUT
#ifndef UCCL_LITE_LOCAL_AG_COPY_OUTPUT
#error "Graph output requires split gather output"
#endif
    if (op == 1 && (count == 4096 || count == 16384)) {
      s.gatherGraph(input, output, count, serialThreads, sharedInputBytes,
                    stream);
    } else
#endif
    {
#ifdef UCCL_LITE_LOCAL_SERIAL_SPECIALIZE
      if (op == 1)
        serialCollectSpecialized<1>
            <<<1, serialThreads, sharedInputBytes, stream>>>(
                s.peers, s.rank, op, static_cast<float const*>(input),
                static_cast<float*>(output), count, s.capacity / 4,
                s.generation);
      else if (op == 2)
        serialCollectSpecialized<2>
            <<<1, serialThreads, sharedInputBytes, stream>>>(
                s.peers, s.rank, op, static_cast<float const*>(input),
                static_cast<float*>(output), count, s.capacity / 4,
                s.generation);
      else
#endif
        serialCollect<<<1, serialThreads, sharedInputBytes, stream>>>(
            s.peers, s.rank, op, static_cast<float const*>(input),
            static_cast<float*>(output), count, s.capacity / 4, s.generation);
#ifdef UCCL_LITE_LOCAL_AG_COPY_OUTPUT
      if (op == 1 && (count == 4096 || count == 16384))
        check(cudaMemcpyAsync(
            output,
            s.peers.data[s.rank] + (s.generation & 1) * (s.capacity / 4),
            count * 16, cudaMemcpyDeviceToDevice, stream));
#endif
    }
  }
#ifdef UCCL_LITE_LOCAL_RING
  else if ((op == 3 && extent >= kArRingMinimum) ||
           (op == 2 && extent >= (8UL << 20))) {
#ifdef UCCL_LITE_LOCAL_RS_TILE_PUBLICATION
    if (op == 2 && extent >= UCCL_LITE_LOCAL_RS_TILE_MIN_BYTES)
      s.scatterTilePublication(static_cast<float const*>(input),
                               static_cast<float*>(output), count, stream);
    else
#endif
#ifdef UCCL_LITE_LOCAL_RS_GRAPHS
        if (op == 2 && extent >= UCCL_LITE_LOCAL_RS_GRAPH_MIN_BYTES &&
            s.scatterGraphs(static_cast<float const*>(input),
                            static_cast<float*>(output), count, stream)) {
    } else
#endif
#ifdef UCCL_LITE_LOCAL_RS_NATIVE_GRAPH
        if (op == 2 && extent >= UCCL_LITE_LOCAL_RS_HOST_READY_MIN_BYTES &&
            s.scatterNativeGraph(static_cast<float const*>(input),
                                 static_cast<float*>(output), count, stream)) {
      graphCompletion = true;
    } else
#endif
#ifdef UCCL_LITE_LOCAL_RS_HOST_READY
        if (op == 2 && extent >= UCCL_LITE_LOCAL_RS_HOST_READY_MIN_BYTES)
      s.scatterHostReady(static_cast<float const*>(input),
                         static_cast<float*>(output), count, stream);
    else
#endif
#ifdef UCCL_LITE_LOCAL_RS_PULL
        if (op == 2 && extent >= (128UL << 20))
      s.scatterPull(static_cast<float const*>(input),
                    static_cast<float*>(output), count, stream);
    else
#endif
#ifdef UCCL_LITE_LOCAL_RS_MEMOP
        if (op == 2 && extent >= (128UL << 20))
      s.scatterMemop(static_cast<float const*>(input),
                     static_cast<float*>(output), count, stream);
    else
#endif
#ifdef UCCL_LITE_LOCAL_HIER_RS
        if (op == 2 && extent >= (128UL << 20))
      s.scatterHierarchy(static_cast<float const*>(input),
                         static_cast<float*>(output), count, stream);
    else
#endif
      s.reduceRing(op, static_cast<float const*>(input),
                   static_cast<float*>(output), count, stream);
  }
#endif
  else {
    size_t shard = op == 3 ? count / 4 : count;
    auto const* source = static_cast<float const*>(input);
    auto* local = s.peers.data[s.rank];
    for (int phase = 0; phase < 4; ++phase) {
      int dest = (s.rank + phase) % 4;
      check(cudaMemcpyAsync(s.peers.data[dest] + s.rank * shard,
                            source + (op == 1 ? 0 : dest * shard), shard * 4,
                            cudaMemcpyDeviceToDevice, stream));
    }
    barrier<<<1, 1, 0, stream>>>(s.peers, s.rank, 0, s.generation);
    if (op == 3) {
      auto* gathered = local + s.capacity / 4;
      reduceShard<<<128, 256, 0, stream>>>(local, gathered + s.rank * shard,
                                           shard);
      for (int phase = 1; phase < 4; ++phase) {
        int dest = (s.rank + phase) % 4;
        check(cudaMemcpyAsync(
            s.peers.data[dest] + s.capacity / 4 + s.rank * shard,
            gathered + s.rank * shard, shard * 4, cudaMemcpyDeviceToDevice,
            stream));
      }
      barrier<<<1, 1, 0, stream>>>(s.peers, s.rank, 1, s.generation);
      check(cudaMemcpyAsync(output, gathered, count * 4,
                            cudaMemcpyDeviceToDevice, stream));
    } else if (op == 2) {
      reduceShard<<<128, 256, 0, stream>>>(local, static_cast<float*>(output),
                                           shard);
    } else {
      check(cudaMemcpyAsync(output, local, count * 16, cudaMemcpyDeviceToDevice,
                            stream));
    }
    barrier<<<1, 1, 0, stream>>>(s.peers, s.rank, 2, s.generation);
  }
  check(cudaGetLastError());
  if (!graphCompletion) check(cudaEventRecord(s.completion, stream));
}
void PeerPacketCollectives::synchronize() {
  if (impl_->generation) check(cudaEventSynchronize(impl_->completion));
#ifdef UCCL_LITE_LOCAL_RS_GRAPHS
  if (!impl_->ringGraphs.empty())
    check(cudaStreamSynchronize(impl_->ringReduce));
#endif
}
bool PeerPacketCollectives::initialized() const { return impl_->ready; }
bool PeerPacketCollectives::idle() const {
  return !impl_->generation || cudaEventQuery(impl_->completion) == cudaSuccess;
}
PeerPacketCollectives::~PeerPacketCollectives() {
  auto& s = *impl_;
  if (!s.ready) return;
  synchronize();
  s.bootstrap->barrier();
#ifdef UCCL_LITE_LOCAL_RS_NATIVE_GRAPH
  std::fprintf(stderr,
               "UCCL Lite native graph: rank=%d calls=%llu eager=%llu\n",
               s.rank, static_cast<unsigned long long>(s.nativeGraphCalls),
               static_cast<unsigned long long>(s.nativeGraphEagerCalls));
  s.nativeRingGraphs.clear();
#endif
#ifdef UCCL_LITE_LOCAL_RS_GPU_READY
  std::fprintf(stderr, "UCCL Lite GPU readiness: rank=%d calls=%llu\n", s.rank,
               static_cast<unsigned long long>(s.ringReadyCalls));
#endif
#ifdef UCCL_LITE_LOCAL_RS_TILE_PUBLICATION
  std::fprintf(stderr, "UCCL Lite tile publication: rank=%d calls=%llu\n",
               s.rank, static_cast<unsigned long long>(s.tileProtocolCalls));
#endif
#ifdef UCCL_LITE_LOCAL_AG_GRAPH_OUTPUT
  s.copyGraphs.clear();
#ifdef UCCL_LITE_LOCAL_AG_EPOCH_GRAPH
#ifdef UCCL_LITE_LOCAL_AG_EPOCH_CACHE128
  std::fprintf(stderr, "UCCL Lite cached gather: rank=%d calls=%llu\n", s.rank,
               static_cast<unsigned long long>(s.cachedGatherCalls));
#endif
  s.epochGatherGraphs.clear();
#endif
#endif
#ifdef UCCL_LITE_LOCAL_SMALL_GRAPH
  s.smallGraphs.clear();
#endif
#ifdef UCCL_LITE_LOCAL_RS_GRAPHS
  std::fprintf(stderr, "UCCL Lite ring graphs: rank=%d calls=%llu eager=%llu\n",
               s.rank, static_cast<unsigned long long>(s.ringGraphCalls),
               static_cast<unsigned long long>(s.ringGraphEagerCalls));
  s.ringGraphs.clear();
#endif
  for (int r = 0; r < 4; ++r)
    if (r != s.rank) cudaIpcCloseMemHandle(s.peers.data[r]);
  s.bootstrap->barrier();
#ifdef UCCL_LITE_LOCAL_RING_IPC_EVENTS
#ifdef UCCL_LITE_LOCAL_HIER_RS
  for (size_t i = 0; i < s.hierarchyPrevious.size(); ++i)
    if (s.hierarchyPrevious[i] && s.hierarchyPrevious[i] != s.ringPrevious[i])
      cudaEventDestroy(s.hierarchyPrevious[i]);
#endif
  for (auto event : s.ringPrevious)
    if (event) cudaEventDestroy(event);
  s.bootstrap->barrier();
  for (auto event : s.ringArrived)
    if (event) cudaEventDestroy(event);
#endif
#ifdef UCCL_LITE_LOCAL_RING
  for (auto event : s.ringReduced) cudaEventDestroy(event);
  cudaEventDestroy(s.ringStart);
  cudaStreamDestroy(s.ringReduce);
#endif
#ifdef UCCL_LITE_LOCAL_HOST_REPLICA
  s.hostCopy.reset();
  cudaFree(s.replicaCounters);
#endif
  cudaFree(s.peers.data[s.rank]);
  cudaEventDestroy(s.completion);
}
}  // namespace uccl_lite
