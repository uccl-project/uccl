#include "channel_device.cuh"
#include "cpu_executor.hpp"
#include "cutlass_gated_grouped.cuh"
#include "packet_pipeline.hpp"
#include "timeline.hpp"
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <linux/mempolicy.h>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <immintrin.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace uccl_lite::cpu_router {
namespace {

constexpr int kSlots = 2;
constexpr int kRingBlocks = 128;

constexpr size_t kQueueDepth = 128;
void check(cudaError_t error, char const* expression) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(expression) + ": " +
                             cudaGetErrorString(error));
}
#define CUDA(expr) check((expr), #expr)
size_t rounded(size_t n, size_t alignment) {
  return (n + alignment - 1) / alignment * alignment;
}
constexpr size_t kShortPrefillElements = 12UL << 20;
size_t packetCapacity(size_t elements, PacketLayout layout, int ranks,
                      PacketPipelineOptions const& options) {
  size_t expansion = layout == PacketLayout::Concatenate ? ranks : 1;
  // Scatter has one consumer per shard. Keep two packets per destination when
  // possible so upload, reduction and download overlap before its shard ends.
  size_t extent = layout == PacketLayout::Scatter ? elements / ranks : elements;
  return std::min(rounded(elements, 8),
                  std::min(options.packetBytes / (2 * expansion),
                           std::max(options.minPacketBytes / (2 * expansion),
                                    rounded((extent + 1) / 2, 8))));
}
int residentTransportCrew(int maxBlocks, size_t work, size_t totalElements,
                          PacketLayout layout, size_t packetBytes,
                          int overrideCrew) {
  int crew = std::max(1, maxBlocks / 2);
  if (overrideCrew) return std::min(crew, overrideCrew);
  // Full-size ReduceScatter packets otherwise occupy nearly every CTA in a
  // cooperative GEMM grid. Prefill needs more transport bandwidth than decode.
  if (layout == PacketLayout::Scatter && work * 2 == packetBytes) {
    int limit = 12;
    if (totalElements < (1UL << 20))
      limit = 4;
    else if (totalElements < kShortPrefillElements)
      limit = 16;
    crew = std::min(crew, limit);
  }
  return crew;
}
using Word = GenerationSignal;
struct Flags {
  Word uploaded[kSlots][kMaxRanks];
  Word released[kSlots][kMaxRanks];
  Word routed[kSlots][kMaxRanks];
  Word consumed[kSlots][kMaxRanks];
  Word readers[kSlots][kMaxRanks];
};
using CpuWord = GenerationSignal;
struct alignas(128) GridState {
  uint32_t arrivals;
  uint32_t generation;
  uint32_t ready;
  uint32_t sourceReady[kMaxRanks];
  uint64_t times[6];
};
struct alignas(16) Transport {
  uint16_t const* input;
  uint16_t* output;
  uint16_t* uploads[kSlots];
  uint16_t const* const* sources[kSlots];
  uint16_t const* results[kSlots];
  Flags* flags;
  Flags* readerFlags;
  union {
    Flags* const* peerFlags;
    Word* const* ringProgress;
  };
  GridState* grid;
  PacketProgram plan;
  size_t packetElements;
  uint32_t previous[2];
  uint32_t firstSequence;
  int rank;
  bool profile;
  bool gpuReduce;
  bool interleaveScatter;
  bool gpuProgress;
  bool agPerSourceProgress;
  int vectorBytes;
  int producerBlocks;
  int consumerBlocks;
  uint32_t initialPhase[2];
  uint32_t const* completed;
  int columns, tileRows, tileColumns;
  uint32_t* gathered;
  bool gatherLocalReady;
};
struct alignas(128) Completion {
  uint64_t value;
};
struct Command {
  Transport payload;
  Word ready;
};
struct Dispatch {
  uint64_t ticket;
  uint32_t finishedBlocks;
  Completion* completion;
};
__device__ uint64_t timer() {
  uint64_t v;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v));
  return v;
}
__device__ uint32_t acquireSys(uint32_t const* p) {
  uint32_t value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(value)
               : "l"(p)
               : "memory");
  return value;
}
__device__ void releaseSys(uint32_t* p, uint32_t value) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(value)
               : "memory");
}
__device__ uint32_t acquireGpu(uint32_t const* p) {
  uint32_t value;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
               : "=r"(value)
               : "l"(p)
               : "memory");
  return value;
}
__device__ void releaseGpu(uint32_t* p, uint32_t value) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(value)
               : "memory");
}
__device__ void barrier(GridState* state, uint32_t& phase, int blocks) {
  __syncthreads();
  if (blocks == 1) return;
  ++phase;
  if (threadIdx.x == 0) {
    // Each CTA leader acquires its block's writes through __syncthreads.
    // The acq_rel RMW chain collects every leader; generation acquire then
    // carries those writes to the single system-release publication below.
    uint32_t arrivals;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                 : "=r"(arrivals)
                 : "l"(&state->arrivals), "r"(1)
                 : "memory");
    if (arrivals == blocks - 1) {
      atomicExch(&state->arrivals, 0);
      asm volatile(
          "st.release.gpu.global.u32 [%0], %1;" ::"l"(&state->generation),
          "r"(phase)
          : "memory");
    }
    while (acquireGpu(&state->generation) < phase) __nanosleep(64);
  }
  __syncthreads();
}
__device__ uint16_t finishScatter(uint16_t routed, uint16_t own) {
  float value = __uint_as_float(uint32_t(routed) << 16) +
                __uint_as_float(uint32_t(own) << 16);
  uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7fffffffU) > 0x7f800000U)
    bits |= 0x00400000U;
  else
    bits += 0x7fffU + ((bits >> 16) & 1U);
  return bits >> 16;
}
__device__ uint32_t finishPair(uint32_t routed, uint32_t own) {
  return uint32_t(finishScatter(routed, own)) |
         (uint32_t(finishScatter(routed >> 16, own >> 16)) << 16);
}
__device__ uint4 finishVector(uint4 routed, uint4 own) {
  return make_uint4(finishPair(routed.x, own.x), finishPair(routed.y, own.y),
                    finishPair(routed.z, own.z), finishPair(routed.w, own.w));
}
// Sum immutable uploaded packets on the destination GPU. The rank's local
// contribution stays on-device, including ReduceScatter's retained shard.
// Accumulate in FP32 and round once; unlike CPU hierarchical BF16 reduction,
// this does not round intermediate NUMA partials.
__device__ uint4 sumPacket8(Transport const& p, int slot, size_t offset,
                            size_t global, bool vector) {
  float sum[8] = {};
  for (int rank = 0; rank < p.plan.ranks; ++rank) {
    if (!(p.plan.sources & (1U << rank))) continue;
    auto from =
        rank == p.rank ? p.input + global : p.sources[slot][rank] + offset;
    if (vector) {
      uint4 v = DeviceChannel::get16(from);
      uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        sum[j * 2] += __uint_as_float(words[j] << 16);
        sum[j * 2 + 1] += __uint_as_float(words[j] & 0xffff0000U);
      }
    } else
      sum[0] += __uint_as_float(uint32_t(DeviceChannel::get2(from)) << 16);
  }
  uint32_t words[4] = {};
#pragma unroll
  for (int j = 0; j < 4; ++j)
    words[j] =
        uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(sum[j * 2]))) |
        (uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(sum[j * 2 + 1])))
         << 16);
  return make_uint4(words[0], words[1], words[2], words[3]);
}
__device__ DeviceChannel::Vector32 sumPacket16(Transport const& p, int slot,
                                               size_t offset, size_t global) {
  float sum[16] = {};
  for (int rank = 0; rank < p.plan.ranks; ++rank) {
    if (!(p.plan.sources & (1U << rank))) continue;
    auto from =
        rank == p.rank ? p.input + global : p.sources[slot][rank] + offset;
    auto v = DeviceChannel::get32(from);
    uint32_t words[8] = {v.lo.x, v.lo.y, v.lo.z, v.lo.w,
                         v.hi.x, v.hi.y, v.hi.z, v.hi.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      sum[j * 2] += __uint_as_float(words[j] << 16);
      sum[j * 2 + 1] += __uint_as_float(words[j] & 0xffff0000U);
    }
  }
  uint32_t words[8];
#pragma unroll
  for (int j = 0; j < 8; ++j)
    words[j] =
        uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(sum[j * 2]))) |
        (uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(sum[j * 2 + 1])))
         << 16);
  return {make_uint4(words[0], words[1], words[2], words[3]),
          make_uint4(words[4], words[5], words[6], words[7])};
}
__device__ void upload(Transport const& p, size_t tile, bool vector, bool wide,
                       size_t lane, size_t stride) {
  if (p.interleaveScatter) {
    if (!(p.plan.sources & (1U << p.rank))) return;
    size_t shard = p.plan.inputElements / p.plan.ranks;
    size_t stripe = p.packetElements / p.plan.ranks;
    size_t n = min(stripe, shard - tile * stripe);
    for (int dest = 0; dest < p.plan.ranks; ++dest) {
      if (dest == p.rank)
        continue;  // Retained locally, just like flat scatter.
      auto from = p.input + size_t(dest) * shard + tile * stripe;
      auto to = p.uploads[tile & 1] + size_t(dest) * stripe;
      if (wide)
        for (size_t i = lane; i < n / 16; i += stride)
          DeviceChannel::put32(to + i * 16,
                               DeviceChannel::get32(from + i * 16));
      else if (vector)
        for (size_t i = lane; i < n / 8; i += stride)
          DeviceChannel::put16(to + i * 8, DeviceChannel::get16(from + i * 8));
      else
        for (size_t i = lane; i < n; i += stride)
          DeviceChannel::put2(to + i, DeviceChannel::get2(from + i));
    }
    return;
  }
  auto f = p.plan.uploadRanges(p.rank, tile, p.packetElements);
  for (int range = 0; range < f.count; ++range) {
    auto r = f.ranges[range];
    if (wide)
      for (size_t i = lane; i < r.elements / 16; i += stride)
        DeviceChannel::put32(
            p.uploads[tile & 1] + r.packetOffset + i * 16,
            DeviceChannel::get32(p.input + r.inputOffset + i * 16));
    else if (vector)
      for (size_t i = lane; i < r.elements / 8; i += stride)
        DeviceChannel::put16(
            p.uploads[tile & 1] + r.packetOffset + i * 8,
            DeviceChannel::get16(p.input + r.inputOffset + i * 8));
    else
      for (size_t i = lane; i < r.elements; i += stride)
        DeviceChannel::put2(p.uploads[tile & 1] + r.packetOffset + i,
                            DeviceChannel::get2(p.input + r.inputOffset + i));
  }
}
__device__ PacketRoute reductionRoute(Transport const& p, size_t tile) {
  if (!p.interleaveScatter)
    return p.plan.outputRoute(p.rank, tile, p.packetElements);
  if (!(p.plan.destinations & (1U << p.rank))) return {};
  size_t shard = p.plan.inputElements / p.plan.ranks;
  size_t stripe = p.packetElements / p.plan.ranks;
  size_t begin = tile * stripe;
  return {size_t(p.rank) * stripe,
          begin,
          min(stripe, shard - begin),
          size_t(p.rank) * shard + begin,
          -1,
          false,
          (p.plan.sources & (1U << p.rank)) != 0};
}
// Flags are published by resident GEMM worker CTAs in this same cooperative
// grid, never by a future kernel. Every output tile intersecting a packet must
// finish its epilogue before that packet can leave the GPU.
__device__ void waitGemm(Transport const& p, size_t tile) {
  if (!p.completed) return;
  size_t begin = tile * p.packetElements;
  size_t end = min((tile + 1) * p.packetElements, p.plan.inputElements);
  size_t first = begin / p.columns / p.tileRows;
  size_t last = (end - 1) / p.columns / p.tileRows;
  int nTiles = (p.columns + p.tileColumns - 1) / p.tileColumns;
  for (size_t rowTile = first; rowTile <= last; ++rowTile) {
    size_t rowBegin = rowTile * size_t(p.tileRows) * p.columns;
    size_t lo = max(begin, rowBegin),
           hi = min(end, rowBegin + size_t(p.tileRows) * p.columns);
    // A flat interval shorter than one matrix row can cover a prefix, suffix,
    // or both after wrapping. Every other N tile is independent of this packet.
    int firstCol = int((lo % p.columns) / p.tileColumns);
    int lastCol = int(((hi - 1) % p.columns) / p.tileColumns);
    bool fullWidth = hi - lo >= size_t(p.columns);
    bool wraps = lo / p.columns != (hi - 1) / p.columns;
    for (int col = 0; col < nTiles; ++col) {
      bool intersects =
          fullWidth || (wraps ? col >= firstCol || col <= lastCol
                              : col >= firstCol && col <= lastCol);
      if (intersects)
        while (acquireGpu(p.completed + rowTile * nTiles + col) == 0)
          __nanosleep(64);
    }
  }
}
// Input credits and output acknowledgements share one synchronization protocol
// across layouts. Prefetch and consume operate on disjoint memory ranges.
template <int GpuReduction = -1, int VectorBytes = 0>
__device__ void runTransport(Transport const& p, size_t firstTile,
                             size_t endTile, int blockIndex = -1) {
  bool gpuReduce = GpuReduction < 0 ? p.gpuReduce : bool(GpuReduction);
  int index = blockIndex < 0 ? int(blockIdx.x) : blockIndex;
  bool producer = index < p.producerBlocks;
  int block = producer ? index : index - p.producerBlocks;
  int blocks = producer ? p.producerBlocks : p.consumerBlocks;
  GridState* state = p.grid + (producer ? 0 : 1);
  if (p.profile && block == 0 && threadIdx.x == 0) state->times[0] = timer();
  // Every CTA uses the same command generation, including after changes in
  // packet count and crew geometry across captured replays.
  uint32_t phase =
      p.initialPhase[producer ? 0 : 1] + (blocks > 1 ? uint32_t(firstTile) : 0);
  bool vector = ((reinterpret_cast<uintptr_t>(p.input) |
                  reinterpret_cast<uintptr_t>(p.output)) &
                 15) == 0 &&
                p.plan.inputElements % 8 == 0 && p.packetElements % 8 == 0 &&
                (p.plan.layout != PacketLayout::Scatter ||
                 (p.plan.inputElements / p.plan.ranks) % 8 == 0);
  bool wide =
      vector && (VectorBytes ? VectorBytes : p.vectorBytes) == 32 &&
      ((reinterpret_cast<uintptr_t>(p.input) |
        reinterpret_cast<uintptr_t>(p.output)) &
       31) == 0 &&
      p.plan.inputElements % 16 == 0 && p.packetElements % 16 == 0 &&
      (p.plan.layout != PacketLayout::Scatter ||
       (p.plan.inputElements / p.plan.ranks) % 16 == 0) &&
      (!p.interleaveScatter || (p.packetElements / p.plan.ranks) % 16 == 0);
  size_t vectorElements = wide ? 16 : 8;
  size_t thread = block * blockDim.x + threadIdx.x,
         stride = blocks * blockDim.x;
  if (producer) {
    for (size_t tile = firstTile; tile < endTile; ++tile) {
      int slot = tile & 1;
      uint32_t old = tile >= 2 ? p.firstSequence + tile - 2 : p.previous[slot];
      if (p.gpuProgress && block == 0) {
        // All ranks acknowledge every generation, including masked/non-reading
        // ranks. This makes slot reuse independent of the preceding operation's
        // destination mask. One lane polls each rank; the rest never poll PCIe.
        if (old && threadIdx.x < p.plan.ranks)
          while (acquireSys(&p.readerFlags->readers[slot][threadIdx.x].value) <
                 old)
            __nanosleep(64);
        __syncthreads();
      }
      if (threadIdx.x == 0) {
        if (block == 0) {
          if (!p.gpuProgress)
            DevicePacketOwnership{{&p.flags->uploaded[slot][p.rank]},
                                  {&p.flags->released[slot][p.rank]},
                                  p.readerFlags->readers[slot]}
                .acquire(old);
          waitGemm(p, tile);
          releaseGpu(&state->ready, p.firstSequence + tile);
        } else
          while (acquireGpu(&state->ready) < p.firstSequence + tile)
            __nanosleep(64);
      }
      __syncthreads();
      upload(p, tile, vector, wide, thread, stride);
      barrier(state, phase, blocks);
      if (block == 0 && threadIdx.x == 0)
        DevicePacketOwnership{{&p.flags->uploaded[slot][p.rank]},
                              {&p.flags->released[slot][p.rank]},
                              p.readerFlags->readers[slot]}
            .publish(p.firstSequence + tile);
    }
    if (p.profile && block == 0 && threadIdx.x == 0) state->times[5] = timer();
    return;
  }
  uint32_t fullMask =
      p.plan.ranks == 32 ? 0xffffffffU : (1U << p.plan.ranks) - 1U;
  bool perSource = p.agPerSourceProgress && p.gpuProgress &&
                   p.plan.transform == PacketTransform::Copy &&
                   p.plan.layout == PacketLayout::Concatenate &&
                   p.plan.sources == fullMask &&
                   p.plan.destinations == fullMask && blocks >= p.plan.ranks;
  for (size_t tile = firstTile; tile < endTile; ++tile) {
    int slot = tile & 1;
    uint32_t seq = p.firstSequence + tile;
    if (perSource) {
      int owner = block % p.plan.ranks;
      if (threadIdx.x == 0) {
        if (block == owner) {
          // The local contribution stays on this GPU; remote contributions
          // become readable after their producer's system-release upload.
          if (owner != p.rank)
            while (acquireSys(
                       &p.peerFlags[owner]->uploaded[slot][owner].value) < seq)
              __nanosleep(64);
          releaseGpu(&state->sourceReady[owner], seq);
        } else
          while (acquireGpu(&state->sourceReady[owner]) < seq) __nanosleep(64);
      }
      __syncthreads();
      if (p.profile && tile == 0 && block == 0 && threadIdx.x == 0)
        state->times[3] = timer();
      size_t packetUnits =
          min(p.packetElements, p.plan.inputElements - tile * p.packetElements);
      if (vector) packetUnits /= vectorElements;
      auto route = p.plan.outputRoute(p.rank, tile, p.packetElements, owner);
      int groupBlock = block / p.plan.ranks;
      int groupBlocks = (blocks - 1 - owner) / p.plan.ranks + 1;
      if (!(p.gatherLocalReady && route.local))
        for (size_t i = size_t(groupBlock) * blockDim.x + threadIdx.x;
             i < packetUnits; i += size_t(groupBlocks) * blockDim.x) {
          size_t index = vector ? i * vectorElements : i;
          uint16_t const* from =
              route.local ? p.input + route.sourceOffset + index
                          : p.sources[slot][owner] + route.sourceOffset + index;
          uint16_t* to = p.output + route.outputOffset + index;
          if (wide)
            DeviceChannel::put32(to, DeviceChannel::get32(from));
          else if (vector)
            DeviceChannel::put16(to, DeviceChannel::get16(from));
          else
            DeviceChannel::put2(to, DeviceChannel::get2(from));
        }
      barrier(state, phase, blocks);
      if (p.profile && tile == 0 && block == 0 && threadIdx.x == 0)
        state->times[4] = timer();
      if (block == 0 && threadIdx.x == 0) {
        if (p.gathered) releaseGpu(p.gathered, uint32_t(tile + 1));
        releaseSys(&p.flags->consumed[slot][p.rank].value, seq);
        releaseSys(&p.readerFlags->readers[slot][p.rank].value, seq);
      }
      continue;
    }
    if (p.gpuProgress && block == 0) {
      if (threadIdx.x < p.plan.ranks && (p.plan.sources & (1U << threadIdx.x)))
        while (
            acquireSys(
                &p.peerFlags[threadIdx.x]->uploaded[slot][threadIdx.x].value) <
            seq)
          __nanosleep(64);
      // Collect every polling lane's system acquire before the CTA leader
      // releases GPU-local readiness to the other consumer CTAs.
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      if (block == 0) {
        bool hasResult = p.plan.layout != PacketLayout::Scatter ||
                         reductionRoute(p, tile).elements;
        if (hasResult && !p.gpuProgress)
          while (acquireSys(&p.flags->routed[slot][p.rank].value) < seq)
            __nanosleep(64);
        releaseGpu(&state->ready, seq);
      } else
        while (acquireGpu(&state->ready) < seq) __nanosleep(64);
    }
    __syncthreads();
    if (p.profile && tile == 0 && block == 0 && threadIdx.x == 0)
      state->times[3] = timer();
    bool concatenate = p.plan.layout == PacketLayout::Concatenate;
    size_t packetUnits =
        vector ? min(p.packetElements,
                     p.plan.inputElements - tile * p.packetElements) /
                     vectorElements
               : min(p.packetElements,
                     p.plan.inputElements - tile * p.packetElements);
    size_t totalUnits = concatenate ? packetUnits * p.plan.ranks : 0;
    auto single = reductionRoute(p, tile);
    if (!concatenate)
      totalUnits = vector ? single.elements / vectorElements : single.elements;
    for (size_t flat = thread; flat < totalUnits; flat += stride) {
      int owner = concatenate ? int(flat / packetUnits) : 0;
      auto route = concatenate ? p.plan.outputRoute(p.rank, tile,
                                                    p.packetElements, owner)
                               : single;
      // Non-destination ranks still participate in packet synchronization.
      if (!route.elements) continue;
      if (p.gatherLocalReady && route.local) continue;
      size_t i = concatenate ? flat % packetUnits : flat;
      size_t index = vector ? i * vectorElements : i;
      size_t target = route.outputOffset + index;
      if (gpuReduce) {
        size_t offset = route.sourceOffset + index;
        size_t global = p.interleaveScatter ? route.retainedOffset + index
                                            : tile * p.packetElements + offset;
        if (wide)
          DeviceChannel::put32(p.output + target,
                               sumPacket16(p, slot, offset, global));
        else {
          auto value = sumPacket8(p, slot, offset, global, vector);
          if (vector)
            DeviceChannel::put16(p.output + target, value);
          else
            DeviceChannel::put2(p.output + target, uint16_t(value.x));
        }
        continue;
      }
      uint16_t const* from =
          route.sourceRank < 0 ? p.results[slot] + route.sourceOffset + index
          : route.local
              ? p.input + route.sourceOffset + index
              : p.sources[slot][route.sourceRank] + route.sourceOffset + index;
      if (wide) {
        auto routed = DeviceChannel::get32(from);
        if (route.addRetained) {
          auto own =
              DeviceChannel::get32(p.input + route.retainedOffset + index);
          routed.lo = finishVector(routed.lo, own.lo);
          routed.hi = finishVector(routed.hi, own.hi);
        }
        DeviceChannel::put32(p.output + target, routed);
      } else if (vector) {
        uint4 routed = DeviceChannel::get16(from);
        if (route.addRetained) {
          uint4 own =
              DeviceChannel::get16(p.input + route.retainedOffset + index);
          routed.x = finishPair(routed.x, own.x);
          routed.y = finishPair(routed.y, own.y);
          routed.z = finishPair(routed.z, own.z);
          routed.w = finishPair(routed.w, own.w);
        }
        DeviceChannel::put16(p.output + target, routed);
      } else {
        uint16_t routed = DeviceChannel::get2(from);
        if (route.addRetained)
          routed = finishScatter(routed, p.input[route.retainedOffset + index]);
        DeviceChannel::put2(p.output + target, routed);
      }
    }
    barrier(state, phase, blocks);
    if (p.profile && tile == 0 && block == 0 && threadIdx.x == 0)
      state->times[4] = timer();
    if (block == 0 && threadIdx.x == 0) {
      if (p.gathered) releaseGpu(p.gathered, uint32_t(tile + 1));
      releaseSys(&p.flags->consumed[slot][p.rank].value, seq);
      if (p.gpuProgress)
        releaseSys(&p.readerFlags->readers[slot][p.rank].value, seq);
      else if ((p.plan.transform == PacketTransform::Copy || gpuReduce) &&
               (p.plan.destinations & (1U << p.rank)))
        DevicePacketOwnership{{&p.flags->routed[slot][p.rank]},
                              {&p.flags->released[slot][p.rank]},
                              p.readerFlags->readers[slot]}
            .consume(p.rank, seq);
    }
  }
  if (p.profile && block == 0 && threadIdx.x == 0) state->times[5] = timer();
}
// CTA-local credits keep each 16-byte vector in the same lane across
// changing extents. Every peer uses the same fixed block/thread geometry.
// CTA barriers carry all writers' data to the leader's system release; a
// per-thread system fence would duplicate this ordering and serialize PCIe.
__device__ uint4 ringLoad(uint16_t const* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
__device__ uint32_t ringAdd(uint32_t a, uint32_t b) {
  __nv_bfloat162_raw x{uint16_t(a), uint16_t(a >> 16)};
  __nv_bfloat162_raw y{uint16_t(b), uint16_t(b >> 16)};
  __nv_bfloat162_raw out = __hadd2(__nv_bfloat162(x), __nv_bfloat162(y));
  return uint32_t(out.x) | (uint32_t(out.y) << 16);
}
__device__ void runRingScatter(Transport const& p) {
  int ranks = p.plan.ranks, prev = (p.rank + ranks - 1) % ranks,
      next = (p.rank + 1) % ranks;
  size_t shard = p.plan.inputElements / ranks;
  size_t tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x,
         stride = size_t(gridDim.x) * blockDim.x;
  auto* own = &p.ringProgress[p.rank][blockIdx.x].value;
  auto* reader = &p.ringProgress[next][blockIdx.x].value;
  auto* writer = &p.ringProgress[prev][blockIdx.x].value;
  uint32_t base = p.firstSequence - 1;
  if (threadIdx.x == 0 && base)
    while (acquireSys(reader) < base) __nanosleep(64);
  __syncthreads();
  for (size_t i = tid; i < shard / 8; i += stride)
    DeviceChannel::put16(p.uploads[0] + i * 8,
                         ringLoad(p.input + size_t(prev) * shard + i * 8));
  __syncthreads();
  if (threadIdx.x == 0) releaseSys(own, base + 1);
  for (int step = 0; step < ranks - 1; ++step) {
    uint32_t seq = base + step + 1;
    bool final = step == ranks - 2;
    if (threadIdx.x == 0) {
      while (acquireSys(writer) < seq) __nanosleep(64);
      // The next rank's publication acknowledges its preceding input read.
      if (!final && step)
        while (acquireSys(reader) < seq) __nanosleep(64);
    }
    __syncthreads();
    int chunk = (p.rank + ranks - step - 2) % ranks, slot = step & 1;
    auto* output = final ? p.output : p.uploads[slot ^ 1];
    for (size_t i = tid; i < shard / 8; i += stride) {
      auto a = ringLoad(p.sources[slot][prev] + i * 8);
      auto b = ringLoad(p.input + size_t(chunk) * shard + i * 8);
      DeviceChannel::put16(output + i * 8,
                           make_uint4(ringAdd(a.x, b.x), ringAdd(a.y, b.y),
                                      ringAdd(a.z, b.z), ringAdd(a.w, b.w)));
    }
    __syncthreads();
    if (threadIdx.x == 0) releaseSys(own, seq + 1);
  }
}
template <int GpuReduction = -1, int VectorBytes = 0>
__global__ void dispatchTransport(Command const* commands, Dispatch* dispatch,
                                  size_t firstTile = 0, size_t endTile = 0,
                                  bool finalEpoch = true) {
  __shared__ uint4 storage[sizeof(Transport) / 16];
  __shared__ uint64_t ticket;
  if (threadIdx.x == 0) {
    ticket = dispatch->ticket;
    while (acquireSys(&commands[ticket % kQueueDepth].ready.value) <
           uint32_t(ticket + 1))
      __nanosleep(64);
  }
  __syncthreads();
  uint4 const* source =
      reinterpret_cast<uint4 const*>(&commands[ticket % kQueueDepth].payload);
  for (size_t i = threadIdx.x; i < sizeof(Transport) / 16; i += blockDim.x)
    storage[i] = DeviceChannel::get16(source + i);
  __syncthreads();
  auto const& p = *reinterpret_cast<Transport*>(storage);
  if constexpr (GpuReduction == 2) {
    runRingScatter(p);
  } else {
    if (!endTile)
      endTile =
          (p.plan.inputElements + p.packetElements - 1) / p.packetElements;
    if (gridDim.x == 1) {
      // Both logical crews have one CTA. Completing each read before beginning
      // the next upload preserves slot credits without a software grid barrier.
      for (size_t tile = firstTile; tile < endTile; ++tile) {
        runTransport<GpuReduction, VectorBytes>(p, tile, tile + 1, 0);
        runTransport<GpuReduction, VectorBytes>(p, tile, tile + 1,
                                                p.producerBlocks);
      }
    } else
      runTransport<GpuReduction, VectorBytes>(p, firstTile, endTile);
  }
  __syncthreads();
  if (threadIdx.x == 0 &&
      atomicAdd(&dispatch->finishedBlocks, 1) == gridDim.x - 1) {
    atomicExch(&dispatch->finishedBlocks, 0);
    if (finalEpoch) {
      dispatch->ticket = ticket + 1;
      asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(
                       &dispatch->completion->value),
                   "l"(ticket + 1)
                   : "memory");
    }
  }
}
void* transportKernel(bool gpuReduction, int vectorBytes, bool ring = false) {
  if (ring) return reinterpret_cast<void*>(dispatchTransport<2, 16>);
  if (vectorBytes == 32)
    return gpuReduction ? reinterpret_cast<void*>(dispatchTransport<1, 32>)
                        : reinterpret_cast<void*>(dispatchTransport<0, 32>);
  return gpuReduction ? reinterpret_cast<void*>(dispatchTransport<1, 16>)
                      : reinterpret_cast<void*>(dispatchTransport<0, 16>);
}
struct ResidentRouterDispatch {
  static constexpr bool kEnabled = true;
  static constexpr int kBlocks = 0;
  Command const* commands = nullptr;
  Dispatch* dispatch = nullptr;
  uint32_t const* completed = nullptr;
  int columns = 0, tileRows = 0, tileColumns = 0;
  uint32_t* gathered = nullptr;
  bool gatherLocalReady = false;
  CUTLASS_DEVICE void operator()() {
    __shared__ uint4 storage[sizeof(Transport) / 16];
    __shared__ uint64_t ticket;
    if (threadIdx.x == 0) {
      ticket = dispatch->ticket;
      while (acquireSys(&commands[ticket % kQueueDepth].ready.value) <
             uint32_t(ticket + 1))
        __nanosleep(64);
    }
    __syncthreads();
    uint4 const* source =
        reinterpret_cast<uint4 const*>(&commands[ticket % kQueueDepth].payload);
    for (size_t i = threadIdx.x; i < sizeof(Transport) / 16; i += blockDim.x)
      storage[i] = DeviceChannel::get16(source + i);
    __syncthreads();
    auto& p = *reinterpret_cast<Transport*>(storage);
    if (threadIdx.x == 0) {
      p.completed = completed;
      p.columns = columns;
      p.tileRows = tileRows;
      p.tileColumns = tileColumns;
      p.gathered = gathered;
      p.gatherLocalReady = gatherLocalReady;
    }
    __syncthreads();
    size_t tiles =
        (p.plan.inputElements + p.packetElements - 1) / p.packetElements;
    runTransport(p, 0, tiles);
    __syncthreads();
    if (threadIdx.x == 0 && atomicAdd(&dispatch->finishedBlocks, 1) ==
                                p.producerBlocks + p.consumerBlocks - 1) {
      atomicExch(&dispatch->finishedBlocks, 0);
      dispatch->ticket = ticket + 1;
      asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(
                       &dispatch->completion->value),
                   "l"(ticket + 1)
                   : "memory");
    }
  }
};
struct ResidentProducer {
  virtual ~ResidentProducer() = default;
  virtual bool matches(PacketProgram const& plan, int rank, void const* input,
                       void const* output) const = 0;
  virtual void configureGather(int localM,
                               std::vector<MemorySpan> const& localA) = 0;
  virtual int gridLimit(int rank) const = 0;
  virtual bool decodeGather() const = 0;
  virtual void launch(int rank, cudaStream_t stream, Command const* commands,
                      Dispatch* dispatch, int transportBlocks,
                      bool producerOnly, size_t packetElements = 0) = 0;
};
template <int TileM, int TileN, int TileK = 32, int Stages = 4>
struct ResidentGemm : ResidentProducer {
  using Elem = cutlass::bfloat16_t;
  using Kernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
      Elem, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,
      Elem, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,
      Elem, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<TileM, TileN, TileK>,
      cutlass::gemm::GemmShape<TileM / 2, TileN / 4, TileK>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<Elem, 8, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      Stages>::GemmKernel;
  using Op = GatedGrouped<Kernel, ResidentRouterDispatch>;
  struct Meta {
    cutlass::gemm::GemmCoord problem;
    Elem *a, *b, *c, *d;
    int64_t lda, ldb, ldc, ldd;
  };
  std::array<RegisteredMemory, kMaxRanks> metadata, completionFlags, tileTrace;
  std::array<cudaGraphExec_t, kMaxRanks> decodeGraphs{};
  std::array<typename Op::Params, kMaxRanks> params;
  std::array<int, kMaxRanks> gridLimits;
  std::vector<MemorySpan> inputsA, inputsB, outputs, localInputs;
  std::array<RegisteredMemory, kMaxRanks> arrivalFlags;
  size_t contributionElements = 0;
  size_t flagBytes;
  int m, n, k, ranks;
  ResidentGemm(int mm, int nn, int kk, std::vector<MemorySpan> const& a,
               std::vector<MemorySpan> const& b,
               std::vector<MemorySpan> const& c,
               std::array<int, kMaxRanks> const& devices, int mGroup)
      : inputsA(a),
        inputsB(b),
        outputs(c),
        m(mm),
        n(nn),
        k(kk),
        ranks(int(a.size())) {
    static_assert(Op::kThreadCount == 256,
                  "router and GEMM CTA threads must agree");
    flagBytes = size_t((m + TileM - 1) / TileM) * ((n + TileN - 1) / TileN) *
                sizeof(uint32_t);
    for (int r = 0; r < ranks; ++r) {
      CUDA(cudaSetDevice(devices[r]));
      metadata[r] = RegisteredMemory::gpu(sizeof(Meta), devices[r]);
      completionFlags[r] = RegisteredMemory::gpu(flagBytes, devices[r]);
      Meta host{{m, n, k},
                static_cast<Elem*>(a[r].address),
                static_cast<Elem*>(b[r].address),
                static_cast<Elem*>(c[r].address),
                static_cast<Elem*>(c[r].address),
                k,
                k,
                n,
                n};
      CUDA(cudaMemcpy(metadata[r].data(), &host, sizeof(host),
                      cudaMemcpyHostToDevice));
      auto* gpu = static_cast<Meta*>(metadata[r].data());
      typename Kernel::Arguments args(
          &gpu->problem, 1, 1,
          typename Kernel::EpilogueOutputOp::Params(1.f, 0.f), &gpu->a, &gpu->b,
          &gpu->c, &gpu->d, &gpu->lda, &gpu->ldb, &gpu->ldc, &gpu->ldd,
          &host.problem);
      params[r] = typename Op::Params(typename Kernel::Params(args, nullptr, 0),
                                      nullptr, nullptr);
      int l2Bytes;
      CUDA(
          cudaDeviceGetAttribute(&l2Bytes, cudaDevAttrL2CacheSize, devices[r]));
      // Group rows when B cannot stay in L2; otherwise publish the first row
      // tile sooner, which matters when CPU routing dominates the latency.
      params[r].tile_m_group =
          mGroup ? mGroup
                 : (size_t(n) * k * sizeof(Elem) > size_t(l2Bytes) * 3 / 4 ? 4
                                                                           : 1);
      if (std::getenv("UCCL_LITE_GPU_TIMELINE_PATH")) {
        tileTrace[r] =
            RegisteredMemory::gpu(sizeof(uint64_t) + flagBytes * 4, devices[r]);
        params[r].kernel_time = static_cast<uint64_t*>(tileTrace[r].data());
        params[r].tile_times = params[r].kernel_time + 1;
      }
      params[r].completed = static_cast<uint32_t*>(completionFlags[r].data());
      params[r].dispatcher = {nullptr, nullptr, params[r].completed,
                              n,       TileM,   TileN};
      CUDA(cudaFuncSetAttribute(gatedGroupedKernel<Op>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                sizeof(typename Op::SharedStorage)));
      int active, sms;
      CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active, gatedGroupedKernel<Op>, Op::kThreadCount,
          sizeof(typename Op::SharedStorage)));
      CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                                  devices[r]));
      gridLimits[r] = active * sms;
      if (r == 0) {
        cudaFuncAttributes attrs;
        CUDA(cudaFuncGetAttributes(&attrs, gatedGroupedKernel<Op>));
        std::printf(
            "# Resident cooperative GEMM BF16 tile=%dx%d Ktile=%d stages=%d "
            "Mgroup=%d shared=%zu "
            "registers=%d local=%zu grid_limit=%d; same-grid tile "
            "publication\n",
            TileM, TileN, TileK, Stages, params[r].tile_m_group,
            sizeof(typename Op::SharedStorage), attrs.numRegs,
            attrs.localSizeBytes, gridLimits[r]);
      }
      if (!active)
        throw std::runtime_error(
            "resident GEMM has no legal cooperative occupancy");
    }
  }
  int gridLimit(int rank) const override { return gridLimits[rank]; }
  bool decodeGather() const override { return contributionElements && m <= 64; }
  bool matches(PacketProgram const& plan, int rank, void const* input,
               void const* output) const override {
    if (contributionElements)
      return plan.transform == PacketTransform::Copy &&
             plan.layout == PacketLayout::Concatenate &&
             plan.inputElements == contributionElements &&
             input == localInputs[rank].address &&
             output == inputsA[rank].address;
    return plan.transform == PacketTransform::Sum &&
           (plan.layout == PacketLayout::Replicate ||
            plan.layout == PacketLayout::Scatter) &&
           plan.inputElements == size_t(m) * n &&
           input == outputs[rank].address;
  }
  void configureGather(int localM,
                       std::vector<MemorySpan> const& localA) override {
    contributionElements = size_t(localM) * k;
    localInputs = localA;
    for (int r = 0; r < ranks; ++r) {
      CUDA(cudaSetDevice(outputs[r].device));
      arrivalFlags[r] =
          RegisteredMemory::gpu(sizeof(uint32_t), outputs[r].device);
      auto* flag = static_cast<uint32_t*>(arrivalFlags[r].data());
      params[r].completed = nullptr;
      params[r].dispatcher.completed = nullptr;
      params[r].dispatcher.gathered = flag;
      // Small decode row groups have no useful local-row overlap. Let the
      // resident consumer copy the local shard with its remote peers, avoiding
      // a separate device copy before each cooperative launch.
      bool decode = m <= 64;
      params[r].dispatcher.gatherLocalReady = !decode;
      params[r].dispatcher_join = decode;
      params[r].input_completed = flag;
      params[r].input_contribution_elements = contributionElements;
      params[r].input_local_begin = size_t(r) * contributionElements;
      params[r].input_local_end =
          decode ? 0 : size_t(r + 1) * contributionElements;
      params[r].tile_m_offset = (r * localM) / TileM;
      if (localM % TileM == 0) {
        params[r].tile_m_offset = 0;
        params[r].input_tiles_per_rank = localM / TileM;
        params[r].input_ranks = ranks;
        params[r].input_rank = r;
      }
    }
  }
  ~ResidentGemm() override {
    for (int r = 0; r < ranks; ++r)
      if (decodeGraphs[r]) {
        cudaSetDevice(outputs[r].device);
        cudaGraphExecDestroy(decodeGraphs[r]);
      }
    char const* path = std::getenv("UCCL_LITE_GPU_TIMELINE_PATH");
    if (!path) return;
    FILE* file = std::fopen(path, "a");
    if (!file) return;
    std::fprintf(file,
                 "# M=%d,N=%d,K=%d,tileM=%d,tileN=%d; GPU globaltimer tile "
                 "MMA+epilogue intervals\n",
                 m, n, k, TileM, TileN);
    for (int r = 0; r < ranks; ++r) {
      cudaSetDevice(outputs[r].device);
      std::vector<uint64_t> times(1 + flagBytes / sizeof(uint32_t) * 2);
      if (cudaMemcpy(times.data(), tileTrace[r].data(),
                     times.size() * sizeof(uint64_t),
                     cudaMemcpyDeviceToHost) != cudaSuccess)
        continue;
      for (size_t tile = 0; tile < (times.size() - 1) / 2; ++tile)
        std::fprintf(file, "%d,%zu,%llu,%llu,%llu\n", r, tile,
                     (unsigned long long)times[1 + tile * 2],
                     (unsigned long long)times[2 + tile * 2],
                     (unsigned long long)times[0]);
    }
    std::fclose(file);
  }
  void launch(int rank, cudaStream_t stream, Command const* commands,
              Dispatch* dispatch, int transportBlocks, bool producerOnly,
              size_t packetElements) override {
    auto p = params[rank];
    if (contributionElements) {
      if (producerOnly) {
        // Diagnostic producer: all gathered rows are already valid on this
        // stream.
        p.input_completed = nullptr;
      } else {
        if (!packetElements)
          throw std::invalid_argument("AG GEMM requires transport");
        p.input_packet_elements = packetElements;
        if (!p.dispatcher_join)
          CUDA(cudaMemsetAsync(arrivalFlags[rank].data(), 0, sizeof(uint32_t),
                               stream));
        // Make this rank's complete rows valid before any worker can read them.
        // The dispatcher omits its local slice, so it cannot race these reads.
        if (p.dispatcher.gatherLocalReady)
          CUDA(cudaMemcpyAsync(static_cast<Elem*>(inputsA[rank].address) +
                                   rank * contributionElements,
                               localInputs[rank].address,
                               contributionElements * sizeof(Elem),
                               cudaMemcpyDeviceToDevice, stream));
      }
    }
    int workers = gridLimits[rank] - transportBlocks;
    if (workers < 1)
      throw std::invalid_argument("transport leaves no resident GEMM worker");
    p.base.threadblock_count =
        workers + (p.dispatcher_join && !producerOnly ? transportBlocks : 0);
    p.dispatcher_blocks = producerOnly ? 0 : transportBlocks;
    p.dispatcher.commands = commands;
    p.dispatcher.dispatch = dispatch;
    if (!contributionElements)
      CUDA(cudaMemsetAsync(completionFlags[rank].data(), 0, flagBytes, stream));
    if (p.tile_times)
      CUDA(cudaMemsetAsync(tileTrace[rank].data(), 0,
                           sizeof(uint64_t) + flagBytes * 4, stream));
    void* args[] = {&p};
    if (p.dispatcher_join && !producerOnly) {
      // The decode launch shape is fixed by prepareAllGatherGemm. Package the
      // readiness reset and cooperative kernel into one host graph launch.
      if (!decodeGraphs[rank]) {
        cudaGraph_t graph;
        cudaGraphNode_t clear, kernel;
        CUDA(cudaGraphCreate(&graph, 0));
        cudaMemsetParams memset{};
        memset.dst = arrivalFlags[rank].data();
        memset.elementSize = sizeof(uint32_t);
        memset.width = 1;
        memset.height = 1;
        CUDA(cudaGraphAddMemsetNode(&clear, graph, nullptr, 0, &memset));
        cudaKernelNodeParams node{};
        node.func = reinterpret_cast<void*>(gatedGroupedKernel<Op>);
        node.gridDim = dim3(workers + p.dispatcher_blocks);
        node.blockDim = dim3(Op::kThreadCount);
        node.sharedMemBytes = sizeof(typename Op::SharedStorage);
        node.kernelParams = args;
        CUDA(cudaGraphAddKernelNode(&kernel, graph, &clear, 1, &node));
        cudaKernelNodeAttrValue cooperative{};
        cooperative.cooperative = true;
        CUDA(cudaGraphKernelNodeSetAttribute(
            kernel, cudaKernelNodeAttributeCooperative, &cooperative));
        CUDA(cudaGraphInstantiate(&decodeGraphs[rank], graph, 0));
        CUDA(cudaGraphDestroy(graph));
      }
      CUDA(cudaGraphLaunch(decodeGraphs[rank], stream));
      return;
    }
    // Cooperative occupancy admits the WHOLE grid before any CTA can wait.
    // Every flag writer is a resident worker of this same grid.
    CUDA(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(gatedGroupedKernel<Op>),
        dim3(workers + p.dispatcher_blocks), dim3(Op::kThreadCount), args,
        sizeof(typename Op::SharedStorage), stream));
  }
};
enum class DmaAction { Acquire, Publish, Wait, Consume };
__global__ void dmaBegin(Command const* commands, Dispatch* dispatch,
                         Transport* state) {
  uint64_t ticket = dispatch->ticket;
  while (
      DeviceSignalRef{const_cast<Word*>(&commands[ticket % kQueueDepth].ready)}
          .acquire() < uint32_t(ticket + 1))
    __nanosleep(64);
  *state = commands[ticket % kQueueDepth].payload;
}
__global__ void dmaControl(Transport const* state, size_t tile,
                           DmaAction action) {
  auto const& p = *state;
  int slot = tile & 1;
  uint32_t seq = p.firstSequence + tile;
  if (action == DmaAction::Acquire) {
    uint32_t old = tile >= 2 ? seq - 2 : p.previous[slot];
    DevicePacketOwnership{{&p.flags->uploaded[slot][p.rank]},
                          {&p.flags->released[slot][p.rank]},
                          p.readerFlags->readers[slot]}
        .acquire(old);
  } else if (action == DmaAction::Publish)
    DeviceSignalRef{&p.flags->uploaded[slot][p.rank]}.publish(seq);
  else if (action == DmaAction::Wait) {
    if (p.plan.layout != PacketLayout::Scatter ||
        p.plan.outputRoute(p.rank, tile, p.packetElements).elements)
      DeviceSignalRef{&p.flags->routed[slot][p.rank]}.wait(seq);
  } else {
    DeviceSignalRef{&p.flags->consumed[slot][p.rank]}.publish(seq);
    if (p.plan.transform == PacketTransform::Copy &&
        (p.plan.destinations & (1U << p.rank)))
      DevicePacketOwnership{{&p.flags->routed[slot][p.rank]},
                            {&p.flags->released[slot][p.rank]},
                            p.readerFlags->readers[slot]}
          .consume(p.rank, seq);
  }
}
__global__ void dmaAdd(uint16_t* output, uint16_t const* retained, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x)
    output[i] = finishScatter(output[i], retained[i]);
}
__global__ void dmaFinish(Dispatch* dispatch) {
  uint64_t ticket = ++dispatch->ticket;
  asm volatile(
      "st.release.sys.global.u64 [%0], %1;" ::"l"(&dispatch->completion->value),
      "l"(ticket)
      : "memory");
}
uint32_t cpuAcquire(Word const& word) {
  return __atomic_load_n(&word.value, __ATOMIC_ACQUIRE);
}
void cpuRelease(Word& word, uint32_t value) {
  __atomic_store_n(&word.value, value, __ATOMIC_RELEASE);
}
struct Job {
  PacketProgram plan{};
  uint32_t sequence = 0;
  size_t tiles = 0;
  int active = 0;
  size_t packetElements = 0;
  std::array<uint32_t, 2> previous{};
  std::array<std::array<uint32_t, 2>, kMaxRanks> previousResults{},
      previousConsumers{};
};
struct JobSlot {
  void* context = nullptr;
  Job job;
  std::atomic<int> readers{0};
};
}  // namespace

namespace {
uint32_t rankMask(int ranks) {
  if (ranks < 1 || ranks > kMaxRanks)
    throw std::invalid_argument("rank count must be 1..32");
  return ranks == 32 ? UINT32_MAX : (1U << ranks) - 1;
}
std::vector<int> cpuList(std::string const& path) {
  std::ifstream file(path);
  std::string text;
  std::getline(file, text);
  std::vector<int> cpus;
  std::stringstream tokens(text);
  std::string token;
  while (std::getline(tokens, token, ',')) {
    size_t dash = token.find('-');
    int first = std::stoi(token);
    int last =
        dash == std::string::npos ? first : std::stoi(token.substr(dash + 1));
    for (int cpu = first; cpu <= last && cpu < CPU_SETSIZE; ++cpu)
      cpus.push_back(cpu);
  }
  return cpus;
}
int deviceNuma(int device) {
  int node = -1;
  auto status = cudaDeviceGetAttribute(&node, cudaDevAttrHostNumaId, device);
  if (status != cudaSuccess) cudaGetLastError();
  if (status != cudaSuccess || node < 0) {
    char bus[32];
    CUDA(cudaDeviceGetPCIBusId(bus, sizeof(bus), device));
    std::string id(bus);
    std::transform(id.begin(), id.end(), id.begin(),
                   [](unsigned char ch) { return std::tolower(ch); });
    // CUDA formats the domain with eight digits, Linux PCI sysfs with four.
    if (id.size() > 12) id = id.substr(id.size() - 12);
    std::ifstream file("/sys/bus/pci/devices/" + id + "/numa_node");
    file >> node;
  }
  // A device without a NUMA locality hint uses the first online host node.
  if (node < 0) {
    auto nodes = cpuList("/sys/devices/system/node/online");
    node = nodes.empty() ? 0 : nodes.front();
  }
  return node;
}
}  // namespace
int gpuNumaNode(int device) { return deviceNuma(device); }
PacketProgram PacketProgram::allReduce(size_t n, int ranks) {
  auto mask = rankMask(ranks);
  return {PacketTransform::Sum, PacketLayout::Replicate, n, mask, mask, ranks};
}
PacketProgram PacketProgram::allGather(size_t n, int ranks) {
  auto mask = rankMask(ranks);
  return {
      PacketTransform::Copy, PacketLayout::Concatenate, n, mask, mask, ranks};
}
PacketProgram PacketProgram::reduceScatter(size_t n, int ranks) {
  auto mask = rankMask(ranks);
  if (n > SIZE_MAX / size_t(ranks))
    throw std::invalid_argument("shard count overflow");
  return {PacketTransform::Sum,
          PacketLayout::Scatter,
          n * ranks,
          mask,
          mask,
          ranks};
}
PacketProgram PacketProgram::broadcast(size_t n, int root, int ranks) {
  auto mask = rankMask(ranks);
  if (root < 0 || root >= ranks)
    throw std::invalid_argument("invalid broadcast root");
  return {PacketTransform::Copy,
          PacketLayout::Replicate,
          n,
          1U << root,
          mask,
          ranks};
}
PacketProgram PacketProgram::reduce(size_t n, int root, int ranks) {
  auto mask = rankMask(ranks);
  if (root < 0 || root >= ranks)
    throw std::invalid_argument("invalid reduce root");
  return {PacketTransform::Sum,
          PacketLayout::Replicate,
          n,
          mask,
          1U << root,
          ranks};
}

struct PacketPipeline::Impl {
  PacketPipelineOptions options;
  std::unique_ptr<ResidentProducer> gemm;
  std::array<int, kMaxRanks> gemmTransportBlocks{};
  int ranks = 0, nodes = 0;
  std::vector<int> nodeIds;
  std::vector<std::vector<int>> groupRanks;
  std::array<int, kMaxRanks> rankGroup{};
  std::vector<RegisteredMemory> flagMemory;
  std::vector<Flags*> flags;
  RegisteredMemory inputs[kSlots][kMaxRanks];
  std::vector<RegisteredMemory> outputs[kSlots], partials[kSlots],
      temporary[kSlots];
  std::vector<RegisteredMemory> scatterPartials[kSlots],
      scatterTemporary[kSlots];
  size_t scatterStride = 0;
  std::unique_ptr<CpuWord[]> localReady[kSlots], readDone[kSlots],
      complete[kSlots];
  std::array<GridState*, kMaxRanks> grids{};
  std::array<std::array<uint32_t, 2>, kMaxRanks> barrierPhases{};
  std::array<RegisteredMemory, kMaxRanks> deviceGrids, deviceDispatch;
  std::array<RegisteredMemory, kMaxRanks> deviceSources, devicePeerFlags;
  std::array<RegisteredMemory, kMaxRanks> ringSignals, deviceRingPointers;
  std::array<RegisteredMemory, kMaxRanks> dmaState, dmaRetained;
  std::array<std::unique_ptr<Channel>, kMaxRanks> channels;
  std::array<cudaStream_t, kMaxRanks> uploadStreams{}, downloadStreams{};
  std::array<cudaEvent_t, kMaxRanks> forkEvents{}, uploadEvents{},
      downloadEvents{};
  std::array<PacketPipeline::Binding, kQueueDepth> bindings;
  std::vector<PacketPipeline::Binding> singleBindings;
  std::array<cudaEvent_t, kQueueDepth> singleDone{};
  std::array<Dispatch*, kMaxRanks> dispatches{};
  std::array<Command*, kMaxRanks> mappedCommands{};
  std::array<RegisteredMemory, kMaxRanks> commandMemory, completionMemory;
  using TransportGraphs = std::array<cudaGraphExec_t, kMaxTransportBlocks + 1>;
  std::array<std::array<TransportGraphs, 2>, kMaxRanks> graphs{};
  std::array<cudaStream_t, kMaxRanks> streams{};
  std::array<Flags*, kMaxRanks> mappedFlags{};
  std::array<Flags*, kMaxRanks> mappedReaderFlags{};
  std::array<std::array<Flags*, kMaxRanks>, kMaxRanks> mappedPeerFlags{};
  std::array<std::array<uint16_t*, kMaxRanks>, 2> mappedInputs{},
      mappedOutputs{};
  uint16_t* mappedSources[kMaxRanks][kSlots][kMaxRanks]{};
  std::array<int, kMaxRanks> devices{}, maxBlocks{};
  std::array<std::array<int, 2>, kMaxRanks> maxBlocksByKind{};
  std::unique_ptr<CpuExecutor> executor;
  std::unique_ptr<CpuExecutor> launchExecutor;
  std::vector<std::vector<int>> launchRanks;
  std::vector<size_t> launchLanes;
  std::array<cudaGraphExec_t, kMaxRanks> launchGraphs{};
  std::array<std::atomic<bool>, kMaxRanks> launchDone{};
  std::array<std::exception_ptr, kMaxRanks> launchErrors{};

  std::array<JobSlot, kQueueDepth> queue;
  std::mutex queueMutex;

  std::condition_variable credit;
  uint64_t published = 0;
  uint64_t singlePublished = 0;
  uint32_t sequence = 1;
  std::array<uint32_t, 2> lastSequence{};
  // Last actual write/readers of each NUMA result bank, not the last packet.
  // Empty scatter packets and Copy jobs do not replace this ownership.
  std::array<std::array<uint32_t, 2>, kMaxRanks> lastResults{}, lastConsumers{};
  size_t partialStride = 0;
  size_t temporaryStride = 0;
  std::atomic<bool> stopping{false};
  bool initialized = false;
  struct CapturedCall {
    PacketProgram plan;
    std::vector<PacketPort> ports;
    PacketPipeline::Binding binding;
    bool fusedGemm = false;
  };
  std::vector<CapturedCall> captured;
  bool recording = false;
  std::thread scheduler;
  std::mutex replayMutex;
  std::condition_variable replayCondition;
  bool replayPending = false, schedulerStop = false;
  std::exception_ptr replayError;
  bool profile = std::getenv("UCCL_LITE_ROUTER_PROFILE") != nullptr;

  explicit Impl(PacketPipelineOptions o) : options(o) {
    if (options.hostRingReduceScatter &&
        (options.backend != TransportBackend::GpuKernel ||
         options.reduction != ReductionPlacement::Gpu || !options.gpuProgress ||
         options.interleaveScatter || options.concurrentProducer ||
         options.transportBlocks > kRingBlocks || options.singleCtaBytes ||
         options.transportVectorBytes != 16))
      throw std::invalid_argument("invalid standalone host-ring options");
    if (options.reduction != ReductionPlacement::Cpu &&
        options.reduction != ReductionPlacement::Gpu)
      throw std::invalid_argument("invalid reduction placement");
    if (options.reduction == ReductionPlacement::Gpu &&
        options.backend != TransportBackend::GpuKernel)
      throw std::invalid_argument("GPU reduction requires the SM transport");
    if (options.interleaveScatter &&
        (options.backend != TransportBackend::GpuKernel ||
         (options.reduction == ReductionPlacement::Cpu &&
          options.concurrentProducer)))
      throw std::invalid_argument(
          "interleaved scatter requires standalone SM transport");
    if (options.gpuProgress && options.reduction != ReductionPlacement::Gpu)
      throw std::invalid_argument("GPU progress requires GPU reduction");
    if (options.agPerSourceProgress && !options.gpuProgress)
      throw std::invalid_argument("per-source AllGather requires GPU progress");
    if (!options.maxTensorBytes || options.packetBytes < 16 ||
        options.packetBytes % 16 || options.minPacketBytes < 16 ||
        options.minPacketBytes > options.packetBytes ||
        options.minPacketBytes % 16 || options.transportBlocks < 2 ||
        options.transportBlocks > kMaxTransportBlocks ||
        options.transportThreads < 32 || options.transportThreads > 1024 ||
        options.transportThreads % 32 ||
        (options.transportVectorBytes != 16 &&
         options.transportVectorBytes != 32) ||
        options.fusedTransportCrew < 0 || options.fusedTransportCrew > 127 ||
        options.workersPerNode < 1 || options.workersPerNode > 63 ||
        options.scatterWorkersPerNode < 0 ||
        options.scatterWorkersPerNode > 63 || !options.grainElements ||
        options.smallReduceScatterWorkersPerNode < 0 ||
        options.smallReduceScatterWorkersPerNode > 63 ||
        options.gpuLaunchWorkersPerNode < 0 ||
        options.gpuLaunchWorkersPerNode > kMaxRanks ||
        options.transportGrainBytes < 16)
      throw std::invalid_argument("invalid CPU router options");
    if (!__builtin_cpu_supports("avx2"))
      throw std::runtime_error("CPU router requires AVX2");
  }
  void init(std::vector<PacketPort> const& ports) {
    ranks = int(ports.size());
    rankMask(ranks);
    nodeIds.clear();
    groupRanks.clear();
    if (!options.numaNodes.empty() && options.numaNodes.size() != ports.size())
      throw std::invalid_argument("NUMA override count mismatch");
    for (int rank = 0; rank < ranks; ++rank) {
      if (options.gpuLaunchWorkersPerNode &&
          (!ports[rank].stream || ports[rank].stream == cudaStreamPerThread))
        throw std::invalid_argument(
            "parallel GPU launch requires explicit streams or "
            "cudaStreamLegacy");
      for (int peer = 0; peer < rank; ++peer)
        if (ports[rank].device == ports[peer].device)
          throw std::invalid_argument("duplicate router device");
      CUDA(cudaSetDevice(ports[rank].device));
      int node = options.numaNodes.empty() ? gpuNumaNode(ports[rank].device)
                                           : options.numaNodes[rank];
      if (node < 0 || node >= 64 ||
          cpuList("/sys/devices/system/node/node" + std::to_string(node) +
                  "/cpulist")
              .empty())
        throw std::invalid_argument("invalid or unavailable host NUMA node");
      auto found = std::find(nodeIds.begin(), nodeIds.end(), node);
      if (found == nodeIds.end()) {
        nodeIds.push_back(node);
        groupRanks.emplace_back();
      }
      rankGroup[rank] = int(std::find(nodeIds.begin(), nodeIds.end(), node) -
                            nodeIds.begin());
      groupRanks[rankGroup[rank]].push_back(rank);
    }
    nodes = int(nodeIds.size());
    std::vector<std::vector<int>> nodeCpus;
    auto allowed = cpuList("/sys/fs/cgroup/cpuset.cpus.effective");
    for (int node : nodeIds) {
      auto cpus = cpuList("/sys/devices/system/node/node" +
                          std::to_string(node) + "/cpulist");
      if (!allowed.empty())
        cpus.erase(std::remove_if(cpus.begin(), cpus.end(),
                                  [&](int cpu) {
                                    return std::find(allowed.begin(),
                                                     allowed.end(),
                                                     cpu) == allowed.end();
                                  }),
                   cpus.end());
      if (cpus.empty())
        throw std::invalid_argument("no permitted CPU on selected NUMA node");
      nodeCpus.push_back(std::move(cpus));
    }
    if (options.workerCpus.empty()) {
      for (auto const& cpus : nodeCpus)
        options.workersPerNode =
            std::min(options.workersPerNode, std::max(1, int(cpus.size()) - 1));
      for (auto const& cpus : nodeCpus)
        options.workerCpus.insert(options.workerCpus.end(), cpus.begin(),
                                  cpus.begin() + options.workersPerNode);
    }
    if (options.workerCpus.size() != size_t(options.workersPerNode * nodes))
      throw std::invalid_argument("worker affinity count mismatch");
    for (size_t i = 0; i < options.workerCpus.size(); ++i) {
      int cpu = options.workerCpus[i];
      auto const& cpus = nodeCpus[i / options.workersPerNode];
      if (std::find(cpus.begin(), cpus.end(), cpu) == cpus.end() ||
          std::find(options.workerCpus.begin(), options.workerCpus.begin() + i,
                    cpu) != options.workerCpus.begin() + i)
        throw std::invalid_argument(
            "worker CPUs must be unique, permitted and NUMA local");
    }
    if (options.replayCpu < 0) {
      for (auto group = nodeCpus.rbegin();
           group != nodeCpus.rend() && options.replayCpu < 0; ++group)
        for (int cpu : *group)
          if (std::find(options.workerCpus.begin(), options.workerCpus.end(),
                        cpu) == options.workerCpus.end()) {
            options.replayCpu = cpu;
            break;
          }
    } else if (options.replayCpu >= CPU_SETSIZE ||
               (!allowed.empty() &&
                std::find(allowed.begin(), allowed.end(), options.replayCpu) ==
                    allowed.end()))
      throw std::invalid_argument("invalid replay CPU");
    std::vector<int> launchCpus;
    if (options.gpuLaunchWorkersPerNode && ranks > 1) {
      for (int node = 0; node < nodes; ++node) {
        size_t count = std::min<size_t>(options.gpuLaunchWorkersPerNode,
                                        groupRanks[node].size());
        size_t begin = launchCpus.size();
        for (auto cpu = nodeCpus[node].rbegin();
             cpu != nodeCpus[node].rend() && launchCpus.size() < begin + count;
             ++cpu)
          if (*cpu != options.replayCpu &&
              (options.gpuLaunchAllowedCpus.empty() ||
               std::find(options.gpuLaunchAllowedCpus.begin(),
                         options.gpuLaunchAllowedCpus.end(),
                         *cpu) != options.gpuLaunchAllowedCpus.end()) &&
              std::find(options.workerCpus.begin(), options.workerCpus.end(),
                        *cpu) == options.workerCpus.end())
            launchCpus.push_back(*cpu);
        if (launchCpus.size() != begin + count)
          throw std::invalid_argument(
              "not enough spare NUMA-local CPUs for GPU launch workers");
        launchRanks.resize(begin + count);
        for (size_t r = 0; r < groupRanks[node].size(); ++r)
          launchRanks[begin + r % count].push_back(groupRanks[node][r]);
      }
      for (size_t i = 0; i < launchCpus.size(); ++i) launchLanes.push_back(i);
    }
    flagMemory.resize(nodes);
    flags.resize(nodes);
    CUDA(cudaSetDevice(ports[0].device));
    for (int node = 0; node < nodes; ++node) {
      flagMemory[node].allocate(sizeof(Flags), nodeIds[node]);
      flags[node] = static_cast<Flags*>(flagMemory[node].data());
    }
    for (int slot = 0; slot < kSlots; ++slot) {
      outputs[slot].resize(nodes);
      partials[slot].resize(nodes);
      temporary[slot].resize(nodes);
      localReady[slot] =
          std::make_unique<CpuWord[]>(nodes * options.workersPerNode);
      readDone[slot] =
          std::make_unique<CpuWord[]>(nodes * options.workersPerNode);
      complete[slot] =
          std::make_unique<CpuWord[]>(nodes * options.workersPerNode);
    }
    lastPartial.resize(nodes * options.workersPerNode);
    lastReaders.resize(nodes * options.workersPerNode);
    size_t capacity = options.packetBytes / 2;
    size_t rowElements =
        std::min(capacity, std::max((capacity + options.workersPerNode - 1) /
                                        options.workersPerNode,
                                    options.grainElements));
    partialStride = rounded(rowElements * 2, 128) / 2;
    temporaryStride = rounded(rowElements * 2, 128) / 2;
    for (int slot = 0; slot < 2; ++slot) {
      for (int rank = 0; rank < ranks; ++rank)
        inputs[slot][rank].allocate(options.packetBytes,
                                    nodeIds[rankGroup[rank]]);
      for (int node = 0; node < nodes; ++node) {
        outputs[slot][node].allocate(options.packetBytes, nodeIds[node]);
        partials[slot][node].allocate(
            partialStride * 2 * options.workersPerNode, nodeIds[node], false);
        temporary[slot][node].allocate(
            temporaryStride * 2 * options.workersPerNode, nodeIds[node], false);
      }
    }
    if (options.hostRingReduceScatter)
      for (int r = 0; r < ranks; ++r)
        ringSignals[r].allocate(sizeof(Word) * kRingBlocks,
                                nodeIds[rankGroup[r]]);
    for (int rank = 0; rank < ranks; ++rank) {
      devices[rank] = ports[rank].device;
      CUDA(cudaSetDevice(devices[rank]));
      int cooperative;
      CUDA(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch,
                                  devices[rank]));
      if (options.backend == TransportBackend::GpuKernel && !cooperative)
        throw std::runtime_error("kernel router requires cooperative GPUs");
      int sms;
      CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                                  devices[rank]));
      for (int kind = 0; kind < 2; ++kind) {
        int active;
        CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active,
            options.concurrentProducer
                ? reinterpret_cast<void*>(dispatchTransport<>)
                : transportKernel(kind, options.transportVectorBytes,
                                  options.hostRingReduceScatter),
            options.transportThreads, 0));
        int limit = std::min(options.transportBlocks, active * sms);
        if (options.concurrentProducer) {
          // Leave two SMs for an independent producer, as in the generic
          // transport. Concurrent launches still use its shared dispatcher.
          if (sms < 4)
            throw std::runtime_error(
                "concurrent transport requires at least four SMs");
          limit = std::min(limit, sms - 2);
        }
        maxBlocksByKind[rank][kind] = limit;
      }
      maxBlocks[rank] =
          maxBlocksByKind[rank][options.reduction == ReductionPlacement::Gpu];
      if (!maxBlocksByKind[rank][0] && !maxBlocksByKind[rank][1])
        throw std::runtime_error("no resident transport blocks");
      deviceGrids[rank] =
          RegisteredMemory::gpu(sizeof(GridState) * 2, devices[rank]);
      grids[rank] = static_cast<GridState*>(deviceGrids[rank].data());
      CUDA(cudaMemset(grids[rank], 0, sizeof(GridState) * 2));
      streams[rank] = ports[rank].stream;
      if (options.backend == TransportBackend::Dma) {
        ChannelOptions channel;
        channel.backend = options.backend;
        channel.device = devices[rank];
        channel.completionSlots = 0;
        channels[rank] = std::make_unique<Channel>(channel);
        dmaState[rank] =
            RegisteredMemory::gpu(sizeof(Transport), devices[rank]);
        dmaRetained[rank] =
            RegisteredMemory::gpu(options.packetBytes, devices[rank]);
        CUDA(cudaStreamCreateWithFlags(&uploadStreams[rank],
                                       cudaStreamNonBlocking));
        CUDA(cudaStreamCreateWithFlags(&downloadStreams[rank],
                                       cudaStreamNonBlocking));
        CUDA(cudaEventCreateWithFlags(&forkEvents[rank],
                                      cudaEventDisableTiming));
        CUDA(cudaEventCreateWithFlags(&uploadEvents[rank],
                                      cudaEventDisableTiming));
        CUDA(cudaEventCreateWithFlags(&downloadEvents[rank],
                                      cudaEventDisableTiming));
      }
      commandMemory[rank].allocate(sizeof(Command) * kQueueDepth,
                                   nodeIds[rankGroup[rank]]);
      completionMemory[rank].allocate(sizeof(Completion),
                                      nodeIds[rankGroup[rank]]);
      CUDA(cudaHostGetDevicePointer(
          reinterpret_cast<void**>(&mappedCommands[rank]),
          commandMemory[rank].data(), 0));
      Completion* mappedCompletion;
      CUDA(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mappedCompletion),
                                    completionMemory[rank].data(), 0));
      Dispatch initial{};
      initial.completion = mappedCompletion;
      deviceDispatch[rank] =
          RegisteredMemory::gpu(sizeof(Dispatch), devices[rank]);
      dispatches[rank] = static_cast<Dispatch*>(deviceDispatch[rank].data());
      CUDA(cudaMemcpy(dispatches[rank], &initial, sizeof(initial),
                      cudaMemcpyHostToDevice));
      CUDA(
          cudaHostGetDevicePointer(reinterpret_cast<void**>(&mappedFlags[rank]),
                                   flagMemory[rankGroup[rank]].data(), 0));
      CUDA(cudaHostGetDevicePointer(
          reinterpret_cast<void**>(&mappedReaderFlags[rank]),
          flagMemory[0].data(), 0));
      if (options.gpuProgress)
        for (int owner = 0; owner < ranks; ++owner)
          CUDA(cudaHostGetDevicePointer(
              reinterpret_cast<void**>(&mappedPeerFlags[rank][owner]),
              flagMemory[rankGroup[owner]].data(), 0));
      for (int slot = 0; slot < 2; ++slot) {
        CUDA(cudaHostGetDevicePointer(
            reinterpret_cast<void**>(&mappedInputs[slot][rank]),
            inputs[slot][rank].data(), 0));
        for (int owner = 0; owner < ranks; ++owner)
          CUDA(cudaHostGetDevicePointer(
              reinterpret_cast<void**>(&mappedSources[rank][slot][owner]),
              inputs[slot][owner].data(), 0));
        CUDA(cudaHostGetDevicePointer(
            reinterpret_cast<void**>(&mappedOutputs[slot][rank]),
            outputs[slot][rankGroup[rank]].data(), 0));
      }
      if (options.hostRingReduceScatter) {
        std::array<Word*, kMaxRanks> pointers{};
        for (int r = 0; r < ranks; ++r)
          pointers[r] = static_cast<Word*>(
              ringSignals[r].span().gpuAddress(devices[rank]));
        deviceRingPointers[rank] =
            RegisteredMemory::gpu(sizeof(pointers), devices[rank]);
        CUDA(cudaMemcpy(deviceRingPointers[rank].data(), pointers.data(),
                        sizeof(pointers), cudaMemcpyHostToDevice));
      }
      // These port mappings never change after prepare. Keep them on-device
      // instead of transferring all 32-rank pointer tables with every command.
      std::array<std::array<uint16_t const*, kMaxRanks>, kSlots> sourceTable{};
      for (int slot = 0; slot < kSlots; ++slot)
        for (int owner = 0; owner < ranks; ++owner)
          sourceTable[slot][owner] = mappedSources[rank][slot][owner];
      deviceSources[rank] =
          RegisteredMemory::gpu(sizeof(sourceTable), devices[rank]);
      CUDA(cudaMemcpy(deviceSources[rank].data(), sourceTable.data(),
                      sizeof(sourceTable), cudaMemcpyHostToDevice));
      if (options.gpuProgress) {
        devicePeerFlags[rank] =
            RegisteredMemory::gpu(sizeof(mappedPeerFlags[rank]), devices[rank]);
        CUDA(cudaMemcpy(devicePeerFlags[rank].data(),
                        mappedPeerFlags[rank].data(),
                        sizeof(mappedPeerFlags[rank]), cudaMemcpyHostToDevice));
      }
    }
    if (options.hostRingReduceScatter) {
      int common = options.transportBlocks;
      for (int r = 0; r < ranks; ++r)
        common = std::min(common, maxBlocksByKind[r][1]);
      for (int r = 0; r < ranks; ++r) maxBlocksByKind[r][1] = common;
      std::fprintf(
          stderr,
          "# ReduceScatter host ring; BF16 per-hop rounding; fixed CTAs=%d\n",
          common);
    }
    for (int group = 0; group < nodes; ++group) {
      std::fprintf(stderr, "# NUMA group%d host_node=%d ranks=", group,
                   nodeIds[group]);
      for (int rank : groupRanks[group])
        std::fprintf(stderr, "%d(device%d),", rank, devices[rank]);
      std::fprintf(stderr, "\n");
    }
    initialized = true;
    std::fprintf(stderr,
                 "# PacketPipeline transport=%s; registered spans; shared BF16 "
                 "executor; "
                 "generational ownership; ranks=%d NUMA_groups=%d\n",
                 options.backend == TransportBackend::Dma ? "dma" : "kernel",
                 ranks, nodes);
    if (!options.gpuProgress)
      executor = std::make_unique<CpuExecutor>(options.workerCpus);
    if (!launchCpus.empty()) {
      launchExecutor =
          std::make_unique<CpuExecutor>(launchCpus, kQueueDepth, 2000);
      std::fprintf(stderr, "# GPU launch workers=%zu CPUs=", launchCpus.size());
      for (int cpu : launchCpus) std::fprintf(stderr, "%d,", cpu);
      std::fprintf(stderr, "\n");
    }
    std::fprintf(stderr, "# packet_progress=%s routing_cpu_workers=%zu\n",
                 options.gpuProgress ? "gpu" : "cpu",
                 options.gpuProgress ? size_t(0) : options.workerCpus.size());
    std::fprintf(stderr,
                 "# CPU-router BF16: packet=%zu slots=2 blocks<=%d threads=%d "
                 "CPU<=%d/node "
                 "grain=%zu; no NCCL collective fallback\n",
                 options.packetBytes, options.transportBlocks,
                 options.transportThreads, options.workersPerNode,
                 options.grainElements);
    std::fprintf(stderr, "# reduction_placement=%s\n",
                 options.reduction == ReductionPlacement::Cpu ? "cpu" : "gpu");
    std::fprintf(
        stderr, "# scatter_packet_order=%s\n",
        options.interleaveScatter ? "interleaved-destinations" : "flat");
  }
  ~Impl() {
    if (initialized) {
      for (int rank = 0; rank < ranks; ++rank) {
        cudaSetDevice(devices[rank]);
        cudaDeviceSynchronize();
      }
      if (profile)
        for (int rank = 0; rank < ranks; ++rank) {
          cudaSetDevice(devices[rank]);
          GridState state{};
          cudaMemcpy(&state, grids[rank] + 1, sizeof(state),
                     cudaMemcpyDeviceToHost);
          auto& t = state.times;
          std::fprintf(
              stderr,
              "profile GPU%d consumer ready0 %.1f copy0 %.1f total %.1f us\n",
              rank, (t[3] - t[0]) / 1000.0, (t[4] - t[3]) / 1000.0,
              (t[5] - t[0]) / 1000.0);
        }
    }
    {
      std::lock_guard<std::mutex> lock(replayMutex);
      schedulerStop = true;
      replayCondition.notify_all();
    }
    if (scheduler.joinable()) scheduler.join();
    {
      std::lock_guard<std::mutex> lock(queueMutex);
      stopping.store(true, std::memory_order_release);
    }
    if (executor) executor->stop();
    if (launchExecutor) launchExecutor->stop();
    if (ranks == 1) {
      cudaSetDevice(devices[0]);
      for (auto event : singleDone)
        if (event) cudaEventDestroy(event);
    }
    for (int rank = 0; rank < ranks; ++rank)
      if (uploadStreams[rank]) {
        cudaSetDevice(devices[rank]);
        cudaStreamDestroy(uploadStreams[rank]);
        cudaStreamDestroy(downloadStreams[rank]);
        cudaEventDestroy(forkEvents[rank]);
        cudaEventDestroy(uploadEvents[rank]);
        cudaEventDestroy(downloadEvents[rank]);
      }
    for (int rank = 0; rank < ranks; ++rank) {
      cudaSetDevice(devices[rank]);
      for (auto const& kind : graphs[rank])
        for (auto graph : kind)
          if (graph) cudaGraphExecDestroy(graph);
    }
  }
  void waitGpu(Word const& word, uint32_t seq) {
    while (cpuAcquire(word) < seq && !stopping.load(std::memory_order_relaxed))
      _mm_pause();
  }
  void waitCpu(CpuWord const& word, uint32_t seq) {
    while (SignalRef{const_cast<CpuWord*>(&word)}.acquire() < seq &&
           !stopping.load(std::memory_order_relaxed))
      _mm_pause();
  }
  void prepareScatterScratch() {
    if (scatterStride) return;
    size_t capacity = options.packetBytes / 2;
    int workers =
        options.scatterWorkersPerNode
            ? std::min(options.workersPerNode, options.scatterWorkersPerNode)
            : options.workersPerNode;
    // Fused short prefill may reduce CPU lanes to eight after a prior
    // unfused call. Size scratch for that wider range in either order.
    int configuredWorkers = workers;
    workers = std::min(workers, 8);
    if (options.smallReduceScatterWorkersPerNode)
      workers = std::min(workers, options.smallReduceScatterWorkersPerNode);
    size_t rowElements = std::min(
        capacity,
        std::max((capacity + workers - 1) / workers, options.grainElements));
    scatterStride = rounded(rowElements * 2, 128) / 2;
    // Separate banks keep AllReduce's layout/cache footprint unchanged. The
    // scatter cap requires wider per-lane ranges than the full executor pool.
    for (int slot = 0; slot < kSlots; ++slot) {
      scatterPartials[slot].resize(nodes);
      scatterTemporary[slot].resize(nodes);
      for (int node = 0; node < nodes; ++node) {
        scatterPartials[slot][node].allocate(
            scatterStride * 2 * options.workersPerNode, nodeIds[node], false);
        if (!options.directScatterResults)
          scatterTemporary[slot][node].allocate(
              scatterStride * 2 * options.workersPerNode, nodeIds[node], false);
      }
    }
    std::fprintf(stderr,
                 "# ReduceScatter owner-only merge; active CPU<=%d/node; "
                 "scratch_stride=%zu\n",
                 configuredWorkers, scatterStride);
  }
  uint32_t destinations(PacketProgram const& plan, size_t begin,
                        size_t end) const {
    if (plan.layout != PacketLayout::Scatter) return plan.destinations;
    uint32_t mask = 0;
    size_t shard = plan.inputElements / ranks;
    for (int r = 0; r < ranks; ++r)
      if ((plan.destinations & (1U << r)) &&
          std::max(begin, size_t(r) * shard) <
              std::min(end, size_t(r + 1) * shard))
        mask |= 1U << r;
    return mask;
  }
  uint32_t nodeDestinations(uint32_t mask, int node) const {
    uint32_t local = 0;
    for (int r : groupRanks[node]) local |= mask & (1U << r);
    return local;
  }
  bool striped(PacketProgram const& plan) const {
    return options.interleaveScatter && plan.layout == PacketLayout::Scatter;
  }
  size_t packetExtent(Job const& job, size_t tile) const {
    return striped(job.plan)
               ? job.packetElements
               : std::min(job.packetElements,
                          job.plan.inputElements - tile * job.packetElements);
  }
  // Split a CPU lane at destination boundaries and at the valid tail of each
  // stripe. Packet offsets are shared with GPU upload/reductionRoute.
  size_t rangeEnd(Job const& job, size_t tile, size_t i, size_t hi,
                  int& destination, bool& valid) const {
    valid = true;
    destination = -1;
    if (job.plan.layout != PacketLayout::Scatter) return hi;
    size_t shard = job.plan.inputElements / ranks;
    if (!striped(job.plan)) {
      size_t begin = tile * job.packetElements;
      destination = int((begin + i) / shard);
      return std::min(hi, size_t(destination + 1) * shard - begin);
    }
    size_t stripe = job.packetElements / ranks;
    size_t length = std::min(stripe, shard - tile * stripe);
    destination = int(i / stripe);
    size_t end = size_t(destination) * stripe + length;
    if (i >= end) {
      valid = false;
      end = size_t(destination + 1) * stripe;
    }
    return std::min(hi, end);
  }
  uint32_t packetDestinations(Job const& job, size_t tile, size_t lo,
                              size_t hi) const {
    if (!striped(job.plan)) {
      size_t begin = tile * job.packetElements;
      return destinations(job.plan, begin + lo, begin + hi);
    }
    uint32_t mask = 0;
    for (size_t i = lo; i < hi;) {
      int destination;
      bool valid;
      i = rangeEnd(job, tile, i, hi, destination, valid);
      if (valid) mask |= job.plan.destinations & (1U << destination);
    }
    return mask;
  }
  // One-NUMA CPU reduction does not need an intermediate partial or merge.
  // Wait for the previous result's readers before overwriting its slot, then
  // publish upload credit only after every CPU lane has finished reading.
  void reduceSingleNuma(Job const& job, int lane) {
    auto previousResults = job.previousResults[0];
    auto previousConsumers = job.previousConsumers[0];
    for (size_t tile = 0; tile < job.tiles; ++tile) {
      int slot = tile & 1;
      uint32_t seq = job.sequence + tile;
      for (int peer : groupRanks[0])
        waitGpu(flags[0]->uploaded[slot][peer], seq);
      size_t n = packetExtent(job, tile);
      size_t width = (n + job.active - 1) / job.active;
      size_t lo = std::min(n, size_t(lane) * width);
      size_t hi = std::min(n, lo + width);
      uint32_t consumers = packetDestinations(job, tile, 0, n);
      if (consumers && lo < hi) {
        TimelineRange wait("packet.wait_consumers", seq, lane == 0);
        if (previousResults[slot])
          for (int peer : groupRanks[0])
            if (previousConsumers[slot] & (1U << peer))
              waitGpu(flags[0]->consumed[slot][peer], previousResults[slot]);
      }
      {
        TimelineRange math("cpu.direct_bf16", seq, lane == 0);
        size_t i = lo;
        while (i < hi) {
          bool scatter = job.plan.layout == PacketLayout::Scatter;
          int excluded;
          bool valid;
          size_t end = rangeEnd(job, tile, i, hi, excluded, valid);
          if (valid && (!scatter || (consumers & (1U << excluded)))) {
            std::array<MemorySpan, kMaxRanks> sources;
            size_t count = 0;
            for (int peer : groupRanks[0])
              if ((job.plan.sources & (1U << peer)) && peer != excluded)
                sources[count++] = MemorySpan::borrowedPinned(
                    static_cast<uint16_t*>(inputs[slot][peer].data()) + i,
                    (end - i) * 2, nodeIds[0]);
            RouterCore::process(
                sources.data(), count,
                MemorySpan::borrowedPinned(
                    static_cast<uint16_t*>(outputs[slot][0].data()) + i,
                    (end - i) * 2, nodeIds[0]));
          }
          i = end;
        }
      }
      SignalRef{&localReady[slot][lane]}.publish(seq);
      if (lane == 0) {
        for (int peer : groupRanks[0])
          PacketOwnershipView{{&flags[0]->uploaded[slot][peer]},
                              {&flags[0]->released[slot][peer]},
                              localReady[slot].get(),
                              (1ULL << job.active) - 1}
              .release(seq);
        for (int peer : groupRanks[0])
          if (job.plan.layout != PacketLayout::Scatter ||
              (consumers & (1U << peer)))
            cpuRelease(flags[0]->routed[slot][peer], seq);
      }
      if (consumers) {
        previousResults[slot] = seq;
        previousConsumers[slot] = consumers;
      }
    }
  }
  // forward uploads -> process NUMA-local partials and merge -> multicast the
  // immutable result. Both transports share this CPU schedule and its credits.
  void routePackets(Job const& job, int worker) {
    if (nodes == 1 && options.singleNumaReduction &&
        job.plan.transform == PacketTransform::Sum &&
        options.reduction == ReductionPlacement::Cpu) {
      reduceSingleNuma(job, worker);
      return;
    }
    int node = worker / options.workersPerNode,
        lane = worker % options.workersPerNode;
    auto previousResults = job.previousResults[node];
    auto previousConsumers = job.previousConsumers[node];
    for (size_t tile = 0; tile < job.tiles; ++tile) {
      int slot = tile & 1;
      uint32_t seq = job.sequence + tile;
      if (job.plan.transform == PacketTransform::Copy ||
          options.reduction == ReductionPlacement::Gpu) {
        for (int peer = 0; peer < ranks; ++peer)
          if (job.plan.sources & (1U << peer))
            waitGpu(flags[rankGroup[peer]]->uploaded[slot][peer], seq);
        RouterCore::multicast(
            seq, groupRanks[node], [&](int peer, uint32_t generation) {
              cpuRelease(flags[node]->routed[slot][peer], generation);
            });
        for (int peer : groupRanks[node]) {
          if (job.plan.sources & (1U << peer))
            PacketOwnershipView{{&flags[node]->uploaded[slot][peer]},
                                {&flags[node]->released[slot][peer]},
                                flags[0]->readers[slot],
                                job.plan.destinations}
                .release(seq);
          else
            SignalRef{&flags[node]->released[slot][peer]}.publish(seq);
        }
        continue;
      }
      // Wait only for groups that read this lane's previous partial.
      if (lastPartial[worker][slot])
        for (int owner = 0; owner < nodes; ++owner)
          if (owner != node && (lastReaders[worker][slot] & (1U << owner)))
            waitCpu(readDone[slot][owner * options.workersPerNode + lane],
                    lastPartial[worker][slot]);
      size_t n = packetExtent(job, tile);
      {
        TimelineRange trace("packet.wait_upload", seq, lane == 0);
        for (int peer : groupRanks[node])
          waitGpu(flags[node]->uploaded[slot][peer], seq);
      }
      size_t width = (n + job.active - 1) / job.active;
      size_t lo = std::min(n, size_t(lane) * width),
             hi = std::min(n, lo + width);
      bool scatter = job.plan.layout == PacketLayout::Scatter;
      auto& partialBanks = scatter ? scatterPartials[slot] : partials[slot];
      auto& temporaryBanks = scatter ? scatterTemporary[slot] : temporary[slot];
      size_t rowStart = lane * (scatter ? scatterStride : partialStride);
      uint16_t* local =
          static_cast<uint16_t*>(partialBanks[node].data()) + rowStart;
      {
        TimelineRange trace("cpu.local_bf16", seq, lane == 0);
        size_t i = lo;
        while (i < hi) {
          int excluded;
          bool valid;
          size_t end = rangeEnd(job, tile, i, hi, excluded, valid);
          if (!valid) {
            i = end;
            continue;
          }
          std::array<MemorySpan, kMaxRanks> sourceSpans;
          size_t count = 0;
          for (int peer : groupRanks[node])
            if ((job.plan.sources & (1U << peer)) && peer != excluded)
              sourceSpans[count++] = MemorySpan::borrowedPinned(
                  static_cast<uint16_t*>(inputs[slot][peer].data()) + i,
                  (end - i) * 2, nodeIds[node]);
          MemorySpan dst{local + i - lo,       (end - i) * 2,
                         MemoryKind::NumaHost, -1,
                         nodeIds[node],        {}};
          RouterCore::process(sourceSpans.data(), count, dst);
          i = end;
        }
      }
      SignalRef{&localReady[slot][worker]}.publish(seq);
      if (lane == 0)
        for (int port : groupRanks[node])
          PacketOwnershipView{
              {&flags[node]->uploaded[slot][port]},
              {&flags[node]->released[slot][port]},
              localReady[slot].get() + node * options.workersPerNode,
              (1ULL << job.active) - 1}
              .release(seq);
      uint32_t consumers = packetDestinations(job, tile, 0, n);
      uint32_t localConsumers = nodeDestinations(consumers, node);
      uint32_t readers = 0;
      if (job.plan.layout == PacketLayout::Scatter) {
        uint32_t laneConsumers = packetDestinations(job, tile, lo, hi);
        for (int r = 0; r < ranks; ++r)
          if (laneConsumers & (1U << r)) readers |= 1U << rankGroup[r];
      } else {
        readers = rankMask(nodes);
      }
      bool writesResult =
          job.plan.layout != PacketLayout::Scatter || localConsumers;
      bool direct = scatter && options.directScatterResults;
      uint16_t* tmp =
          direct ? static_cast<uint16_t*>(outputs[slot][node].data()) + lo
                 : static_cast<uint16_t*>(temporaryBanks[node].data()) +
                       lane * (scatter ? scatterStride : temporaryStride);
      if (direct && writesResult && lo < hi && previousResults[slot]) {
        TimelineRange trace("packet.wait_consumers", seq, lane == 0);
        for (int peer : groupRanks[node])
          if (previousConsumers[slot] & (1U << peer))
            waitGpu(flags[node]->consumed[slot][peer], previousResults[slot]);
      }
      // Every source group produces its local partial. Only the destination
      // group reads those partials and performs the final BF16 merge.
      if (readers & (1U << node)) {
        TimelineRange trace("cpu.wait_remote_partial", seq, lane == 0);
        for (int owner = 0; owner < nodes; ++owner)
          if (owner != node)
            waitCpu(localReady[slot][owner * options.workersPerNode + lane],
                    seq);
        TimelineRange math("cpu.merge_bf16", seq, lane == 0);
        size_t i = lo;
        while (i < hi) {
          int destination;
          bool valid;
          size_t end = rangeEnd(job, tile, i, hi, destination, valid);
          bool owned =
              valid &&
              (!scatter || ((job.plan.destinations & (1U << destination)) &&
                            rankGroup[destination] == node));
          if (owned) {
            if (nodes == 1)
              RouterCore::forward({tmp + i - lo,
                                   (end - i) * 2,
                                   MemoryKind::NumaHost,
                                   -1,
                                   nodeIds[node],
                                   {}},
                                  {local + i - lo,
                                   (end - i) * 2,
                                   MemoryKind::NumaHost,
                                   -1,
                                   nodeIds[node],
                                   {}});
            else {
              std::array<MemorySpan, kMaxRanks> sourceSpans;
              for (int owner = 0; owner < nodes; ++owner)
                sourceSpans[owner] = {
                    static_cast<uint16_t*>(partialBanks[owner].data()) +
                        rowStart + i - lo,
                    (end - i) * 2,
                    MemoryKind::NumaHost,
                    -1,
                    nodeIds[owner],
                    {}};
              RouterCore::process(sourceSpans.data(), nodes,
                                  {tmp + i - lo,
                                   (end - i) * 2,
                                   MemoryKind::NumaHost,
                                   -1,
                                   nodeIds[node],
                                   {}});
            }
          }
          i = end;
        }
      }
      // Reader credits cover precisely the destination groups that accessed
      // this lane's partial. Publish even on empty lanes to keep generations
      // monotonic; their source reader mask is empty.
      SignalRef{&readDone[slot][worker]}.publish(seq);
      if (!direct && writesResult && lo < hi) {
        TimelineRange trace("packet.wait_consumers", seq, lane == 0);
        if (previousResults[slot])
          for (int peer : groupRanks[node])
            if (previousConsumers[slot] & (1U << peer))
              waitGpu(flags[node]->consumed[slot][peer], previousResults[slot]);
        size_t i = lo;
        while (i < hi) {
          int destination;
          bool valid;
          size_t end = rangeEnd(job, tile, i, hi, destination, valid);
          bool owned =
              valid &&
              (!scatter || ((job.plan.destinations & (1U << destination)) &&
                            rankGroup[destination] == node));
          if (owned)
            RouterCore::forward(
                {static_cast<uint16_t*>(outputs[slot][node].data()) + i,
                 (end - i) * 2,
                 MemoryKind::NumaPinned,
                 -1,
                 nodeIds[node],
                 {}},
                {tmp + i - lo,
                 (end - i) * 2,
                 MemoryKind::NumaHost,
                 -1,
                 nodeIds[node],
                 {}});
          i = end;
        }
      }
      lastPartial[worker][slot] = seq;
      lastReaders[worker][slot] = readers;
      if (writesResult) {
        previousResults[slot] = seq;
        previousConsumers[slot] = localConsumers;
      }
      SignalRef{&complete[slot][worker]}.publish(seq);
      if (lane == 0 && writesResult) {
        for (int peer = 0; peer < job.active; ++peer)
          waitCpu(complete[slot][node * options.workersPerNode + peer], seq);
        // GPU consumers pull one shared NUMA-local packet. Multicast publishes
        // its generation to each reader, avoiding per-GPU CPU payload copies.
        RouterCore::multicast(
            seq, groupRanks[node], [&](int port, uint32_t generation) {
              if (localConsumers & (1U << port))
                cpuRelease(flags[node]->routed[slot][port], generation);
            });
        // Replicated Sum keeps the historical no-output-rank notifications.
        if (job.plan.layout != PacketLayout::Scatter)
          for (int port : groupRanks[node])
            if (!(localConsumers & (1U << port)))
              cpuRelease(flags[node]->routed[slot][port], seq);
      }
    }
  }
  std::vector<std::array<uint32_t, 2>> lastPartial, lastReaders;
  static void executeJob(void* context, int rank) {
    auto* slot = static_cast<JobSlot*>(context);
    auto* owner = static_cast<Impl*>(slot->context);
    owner->routePackets(slot->job, rank);
    {
      std::lock_guard<std::mutex> lock(owner->queueMutex);
      slot->readers.fetch_sub(1, std::memory_order_release);
    }
    owner->credit.notify_one();
  }
  static void launchGpuGraphs(void* context, int lane) {
    auto& owner = *static_cast<Impl*>(context);
    TimelineRange trace("packet.gpu_launch_lane", lane);
    try {
      for (int rank : owner.launchRanks[lane]) {
        CUDA(cudaSetDevice(owner.devices[rank]));
        CUDA(cudaGraphLaunch(owner.launchGraphs[rank], owner.streams[rank]));
      }
    } catch (...) {
      owner.launchErrors[lane] = std::current_exception();
    }
    owner.launchDone[lane].store(true, std::memory_order_release);
  }
  void launchGpuBatch() {
    TimelineRange trace("packet.gpu_launch_batch", published);
    for (size_t lane : launchLanes) {
      launchErrors[lane] = nullptr;
      launchDone[lane].store(false, std::memory_order_relaxed);
    }
    launchExecutor->enqueueBatch(launchLanes.data(), launchLanes.size(),
                                 launchGpuGraphs, this);
    // Return only after every CUDA launch API has returned, preserving caller
    // stream ordering for events or kernels immediately following enqueue.
    for (size_t lane : launchLanes)
      while (!launchDone[lane].load(std::memory_order_acquire)) _mm_pause();
    for (size_t lane : launchLanes)
      if (launchErrors[lane]) std::rethrow_exception(launchErrors[lane]);
  }
  void submit(PacketProgram const& plan, std::vector<PacketPort> const& ports,
              bool publishOnly = false,
              PacketPipeline::Binding const* binding = nullptr,
              bool fusedGemm = false) {
    if (ports.size() != size_t(plan.ranks) ||
        (initialized && plan.ranks != ranks))
      throw std::invalid_argument("program/port rank count mismatch");
    rankMask(plan.ranks);
    if (!initialized) ranks = plan.ranks;
    if (!plan.inputElements) return;
    if (fusedGemm && (!gemm || options.backend != TransportBackend::GpuKernel ||
                      plan.sources != rankMask(ranks) ||
                      plan.destinations != rankMask(ranks)))
      throw std::invalid_argument(
          "resident GEMM requires a prepared reduction producer");
    if ((plan.transform != PacketTransform::Sum &&
         plan.transform != PacketTransform::Copy) ||
        (plan.layout != PacketLayout::Replicate &&
         plan.layout != PacketLayout::Concatenate &&
         plan.layout != PacketLayout::Scatter))
      throw std::invalid_argument("invalid router transform/layout");
    if (!plan.sources || (plan.sources & ~rankMask(ranks)) ||
        !plan.destinations || (plan.destinations & ~rankMask(ranks)))
      throw std::invalid_argument("invalid router masks");
    if (plan.layout == PacketLayout::Scatter &&
        (plan.inputElements % ranks || plan.transform != PacketTransform::Sum))
      throw std::invalid_argument("invalid scatter plan");
    if (plan.layout == PacketLayout::Concatenate &&
        (plan.transform != PacketTransform::Copy ||
         plan.sources != rankMask(ranks)))
      throw std::invalid_argument("invalid concatenate plan");
    if (plan.transform == PacketTransform::Copy &&
        plan.layout == PacketLayout::Replicate &&
        __builtin_popcount(plan.sources) != 1)
      throw std::invalid_argument("replicated copy requires one source");
    size_t multiplier =
        plan.layout == PacketLayout::Concatenate ? size_t(ranks) * 2 : 2;
    if (plan.inputElements > options.maxTensorBytes / multiplier)
      throw std::invalid_argument("router tensor too large");
    if (options.hostRingReduceScatter &&
        (fusedGemm || (binding && binding->producerElements) || ranks < 2 ||
         plan.layout != PacketLayout::Scatter ||
         plan.transform != PacketTransform::Sum ||
         plan.sources != rankMask(ranks) ||
         plan.destinations != rankMask(ranks) ||
         plan.inputElements % (size_t(ranks) * 8) ||
         plan.inputElements / ranks > options.packetBytes / 2))
      throw std::invalid_argument(
          "host ring requires complete aligned RS shards within packet "
          "capacity");
    for (int rank = 0; rank < ranks; ++rank) {
      if (fusedGemm &&
          !gemm->matches(plan, rank, ports[rank].input, ports[rank].output))
        throw std::invalid_argument(
            "resident GEMM shape or partial binding changed");
      for (int peer = 0; peer < rank; ++peer)
        if (ports[rank].device == ports[peer].device)
          throw std::invalid_argument("duplicate router device");
      if (initialized && ports[rank].device != devices[rank])
        throw std::invalid_argument("router device order changed");
      if (initialized && ports[rank].stream != streams[rank])
        throw std::invalid_argument("router port stream changed");
      if ((plan.sources & (1U << rank)) && !ports[rank].input)
        throw std::invalid_argument("null router input");
      if ((plan.destinations & (1U << rank)) && !ports[rank].output)
        throw std::invalid_argument("null router output");
      if (options.hostRingReduceScatter) {
        uintptr_t in = reinterpret_cast<uintptr_t>(ports[rank].input),
                  out = reinterpret_cast<uintptr_t>(ports[rank].output);
        size_t inBytes = plan.inputElements * 2, outBytes = inBytes / ranks;
        if (((in | out) & 15) || in > UINTPTR_MAX - inBytes ||
            out > UINTPTR_MAX - outBytes ||
            (in < out + outBytes && out < in + inBytes))
          throw std::invalid_argument(
              "host ring requires aligned non-overlapping buffers");
      }
      if (!publishOnly) {
        // Default/legacy streams refer to the current device. Keep capture
        // validation here, before any command or CPU callback is published.
        CUDA(cudaSetDevice(ports[rank].device));
        cudaStreamCaptureStatus status;
        CUDA(cudaStreamIsCapturing(ports[rank].stream, &status));
        if (status != cudaStreamCaptureStatusNone && !recording)
          throw std::invalid_argument(
              "call beginCapture before external capture");
      }
    }
    if (!initialized) init(ports);
    if (ranks == 1 && !fusedGemm) {
      // All supported one-rank collectives are identity operations. Keep
      // explicit producer dependencies for bindings using a separate stream.
      CUDA(cudaSetDevice(devices[0]));
      size_t index = singlePublished % kQueueDepth;
      if (recording)
        captured.push_back({plan, ports,
                            binding ? *binding : PacketPipeline::Binding(1),
                            false});
      else {
        if (singleDone[index])
          CUDA(cudaEventSynchronize(singleDone[index]));
        else
          CUDA(cudaEventCreateWithFlags(&singleDone[index],
                                        cudaEventDisableTiming));
        if (singleBindings.empty())
          singleBindings.resize(kQueueDepth, PacketPipeline::Binding(1));
        singleBindings[index] = binding ? *binding : PacketPipeline::Binding(1);
        ++singlePublished;
      }
      if (binding && binding->producerElements)
        for (auto event : binding->producerEvents[0])
          CUDA(cudaStreamWaitEvent(ports[0].stream, event, 0));
      if (ports[0].input != ports[0].output)
        CUDA(cudaMemcpyAsync(ports[0].output, ports[0].input,
                             plan.inputElements * 2, cudaMemcpyDeviceToDevice,
                             ports[0].stream));
      if (!recording) CUDA(cudaEventRecord(singleDone[index], ports[0].stream));
      return;
    }
    if (fusedGemm && options.reduction != ReductionPlacement::Cpu)
      throw std::invalid_argument(
          "GPU reduction has not been validated with resident GEMM");
    bool gpuReduce = plan.transform == PacketTransform::Sum &&
                     options.reduction == ReductionPlacement::Gpu;
    bool interleaveScatter =
        options.interleaveScatter && plan.layout == PacketLayout::Scatter;
    if (interleaveScatter && fusedGemm)
      throw std::invalid_argument(
          "interleaved scatter does not support resident GEMM");
    if (interleaveScatter && binding && binding->producerElements)
      throw std::invalid_argument(
          "interleaved scatter cannot use flat producer-event stripes");
    if (plan.layout == PacketLayout::Scatter && !gpuReduce)
      prepareScatterScratch();
    size_t capacity = packetCapacity(
        plan.inputElements,
        interleaveScatter ? PacketLayout::Replicate : plan.layout, ranks,
        options);
    if (interleaveScatter) {
      size_t alignment = size_t(ranks) * 8;
      capacity = std::max(alignment, capacity / alignment * alignment);
      if (capacity > options.packetBytes / 2)
        throw std::invalid_argument("packet too small for interleaved scatter");
    }
    if (!capacity)
      throw std::invalid_argument("packet too small for rank count");
    size_t tiles = (plan.inputElements + capacity - 1) / capacity;
    if (interleaveScatter) {
      size_t stripe = capacity / ranks, shard = plan.inputElements / ranks;
      tiles = (shard + stripe - 1) / stripe;
    }
    if (options.hostRingReduceScatter) {
      capacity = plan.inputElements / ranks;
      tiles =
          size_t(ranks);  // Reserve all progress states, including completion.
    }
    if (tiles > UINT32_MAX - sequence)
      throw std::runtime_error("router sequence exhausted");
    uint32_t first = sequence;
    size_t work = std::min(capacity, plan.inputElements);
    size_t tensorBytes = plan.inputElements * multiplier;
    bool singleCta = options.singleCtaBytes &&
                     tensorBytes <= options.singleCtaBytes && !fusedGemm &&
                     !options.concurrentProducer &&
                     options.backend == TransportBackend::GpuKernel;
    int kernelKind = int(gpuReduce);
    void* kernel = transportKernel(gpuReduce, options.transportVectorBytes,
                                   options.hostRingReduceScatter);
    std::array<int, kMaxRanks> producerBlocks{}, consumerBlocks{};
    for (int rank = 0; rank < ranks; ++rank) {
      int limit =
          fusedGemm ? maxBlocks[rank] : maxBlocksByKind[rank][kernelKind];
      if (limit < (singleCta ? 1 : 2))
        throw std::runtime_error("no resident grid for this transport variant");
      int crew = fusedGemm
                     ? residentTransportCrew(limit, work, plan.inputElements,
                                             plan.layout, options.packetBytes,
                                             options.fusedTransportCrew)
                     : std::max(1, limit / 2);
      if (fusedGemm && gemm->decodeGather() &&
          plan.layout == PacketLayout::Concatenate &&
          !options.fusedTransportCrew)
        crew = limit;
      size_t egress = plan.layout == PacketLayout::Concatenate ? work * ranks
                      : plan.layout == PacketLayout::Scatter
                          ? std::min(work, plan.inputElements / ranks)
                          : work;
      if (interleaveScatter)
        egress = std::min(capacity / ranks, plan.inputElements / ranks);
      producerBlocks[rank] = std::min<size_t>(
          crew,
          std::max<size_t>(1, (work * 2 + options.transportGrainBytes - 1) /
                                  options.transportGrainBytes));
      if (fusedGemm && gemm->decodeGather() &&
          plan.layout == PacketLayout::Concatenate)
        producerBlocks[rank] = std::min(producerBlocks[rank], limit - 1);
      consumerBlocks[rank] = std::min<size_t>(
          crew,
          std::max<size_t>(1, (egress * 2 + options.transportGrainBytes - 1) /
                                  options.transportGrainBytes));
      if (fusedGemm && gemm->decodeGather() &&
          plan.layout == PacketLayout::Concatenate)
        consumerBlocks[rank] =
            std::min(consumerBlocks[rank], limit - producerBlocks[rank]);
      if (singleCta) producerBlocks[rank] = consumerBlocks[rank] = 1;
      if (options.hostRingReduceScatter) {
        producerBlocks[rank] = limit;
        consumerBlocks[rank] = 0;
      }
      int blocks = singleCta ? 1 : producerBlocks[rank] + consumerBlocks[rank];
      if (blocks > kMaxTransportBlocks)
        throw std::invalid_argument("router graph grid limit exceeded");
      if (options.backend == TransportBackend::GpuKernel &&
          !graphs[rank][kernelKind][blocks]) {
        if (recording || publishOnly)
          throw std::invalid_argument("warm up this geometry before capture");
        CUDA(cudaSetDevice(devices[rank]));
        cudaGraph_t graph;
        cudaGraphNode_t node;
        CUDA(cudaGraphCreate(&graph, 0));
        size_t beginTile = 0, endTile = 0;
        bool finalEpoch = true;
        void* args[] = {&mappedCommands[rank], &dispatches[rank], &beginTile,
                        &endTile, &finalEpoch};
        cudaKernelNodeParams params{};
        params.func = kernel;
        params.gridDim = dim3(blocks);
        params.blockDim = dim3(options.transportThreads);
        params.kernelParams = args;
        CUDA(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &params));
        cudaKernelNodeAttrValue attr{};
        attr.cooperative = !options.concurrentProducer;
        CUDA(cudaGraphKernelNodeSetAttribute(
            node, cudaKernelNodeAttributeCooperative, &attr));
        CUDA(cudaGraphInstantiate(&graphs[rank][kernelKind][blocks], graph, 0));
        CUDA(cudaGraphDestroy(graph));
      }
    }
    if (recording) {
      captured.push_back({plan, ports,
                          binding ? *binding : PacketPipeline::Binding{},
                          fusedGemm});
      if (fusedGemm) {
        emitGemm(ports, producerBlocks, consumerBlocks, capacity);
        return;
      }
      if (options.backend == TransportBackend::Dma) {
        emitDma(plan, ports, capacity, tiles);
        return;
      }
      if (options.concurrentProducer) {
        emitConcurrent(plan, ports, binding, capacity, tiles, producerBlocks,
                       consumerBlocks);
        return;
      }
      for (int rank = 0; rank < ranks; ++rank) {
        CUDA(cudaSetDevice(devices[rank]));
        size_t beginTile = 0, endTile = 0;
        bool finalEpoch = true;
        void* args[] = {&mappedCommands[rank], &dispatches[rank], &beginTile,
                        &endTile, &finalEpoch};
        if (options.concurrentProducer)
          dispatchTransport<>
              <<<producerBlocks[rank] + consumerBlocks[rank],
                 options.transportThreads, 0, ports[rank].stream>>>(
                  mappedCommands[rank], dispatches[rank]);
        else
          CUDA(cudaLaunchCooperativeKernel(
              kernel,
              dim3(singleCta ? 1 : producerBlocks[rank] + consumerBlocks[rank]),
              dim3(options.transportThreads), args, 0, ports[rank].stream));
      }
      return;
    }
    sequence += tiles;
    if (published >= kQueueDepth)
      for (int rank = 0; rank < ranks; ++rank)
        while (
            __atomic_load_n(
                &static_cast<Completion*>(completionMemory[rank].data())->value,
                __ATOMIC_ACQUIRE) <= published - kQueueDepth)
          std::this_thread::yield();
    uint64_t ticket = published;
    auto previous = lastSequence;
    auto& slot = queue[published % kQueueDepth];
    size_t cpuWork =
        work * (plan.layout == PacketLayout::Concatenate ? ranks : 1);
    int active =
        (plan.transform == PacketTransform::Copy || gpuReduce)
            ? 1
            : std::min<size_t>(
                  options.workersPerNode,
                  std::max<size_t>(1, (cpuWork + options.grainElements - 1) /
                                          options.grainElements));
    if (plan.layout == PacketLayout::Scatter && options.scatterWorkersPerNode)
      active = std::min(active, options.scatterWorkersPerNode);
    if (plan.layout == PacketLayout::Scatter && !gpuReduce &&
        options.smallReduceScatterWorkersPerNode &&
        plan.inputElements * 2 >= (256UL << 10) &&
        plan.inputElements * 2 <= (1UL << 20))
      active = std::min(active, options.smallReduceScatterWorkersPerNode);
    if (fusedGemm && plan.layout == PacketLayout::Scatter &&
        plan.inputElements >= (1UL << 20) &&
        plan.inputElements < kShortPrefillElements)
      active = std::min(active, 8);
    {
      std::unique_lock<std::mutex> lock(queueMutex);
      credit.wait(lock, [&] {
        return slot.readers.load(std::memory_order_acquire) == 0;
      });
      slot.context = this;
      bindings[ticket % kQueueDepth] =
          binding ? *binding : PacketPipeline::Binding{};
      slot.job = {plan,     first,        tiles,       active,
                  capacity, lastSequence, lastResults, lastConsumers};
      if (plan.transform == PacketTransform::Sum && !gpuReduce) {
        for (size_t tile = 0; tile < tiles; ++tile) {
          uint32_t mask = packetDestinations(slot.job, tile, 0,
                                             packetExtent(slot.job, tile));
          for (int node = 0; node < nodes; ++node) {
            uint32_t local = nodeDestinations(mask, node);
            if (plan.layout != PacketLayout::Scatter || local) {
              lastResults[node][tile & 1] = first + uint32_t(tile);
              lastConsumers[node][tile & 1] = local;
            }
          }
        }
      }
      slot.readers.store(options.gpuProgress ? 0 : active * nodes,
                         std::memory_order_relaxed);
      lastSequence[0] = first + ((tiles - 1) & ~size_t(1));
      if (tiles > 1) lastSequence[1] = first + ((tiles - 2) | size_t(1));
      ++published;
    }
    if (!options.gpuProgress) {
      if (options.batchCpuJobs) {
        size_t lanes[kMaxRanks * 63];
        size_t count = 0;
        for (int node = 0; node < nodes; ++node)
          for (int i = 0; i < active; ++i)
            lanes[count++] = size_t(node * options.workersPerNode + i);
        executor->enqueueBatch(lanes, count, executeJob, &slot);
      } else {
        for (int node = 0; node < nodes; ++node)
          for (int i = 0; i < active; ++i)
            RouterCore::process(*executor, node * options.workersPerNode + i,
                                executeJob, &slot);
      }
    }
    bool parallelLaunch = launchExecutor && !publishOnly &&
                          options.backend == TransportBackend::GpuKernel &&
                          !options.concurrentProducer && !fusedGemm;
    for (int rank = 0; rank < ranks; ++rank) {
      if (!publishOnly && !parallelLaunch) CUDA(cudaSetDevice(devices[rank]));
      Transport p{};
      p.input = static_cast<uint16_t const*>(ports[rank].input);
      p.output = static_cast<uint16_t*>(ports[rank].output);
      for (int slot = 0; slot < 2; ++slot) {
        p.uploads[slot] = mappedInputs[slot][rank];
        p.sources[slot] =
            static_cast<uint16_t const* const*>(deviceSources[rank].data()) +
            slot * kMaxRanks;
        p.results[slot] = mappedOutputs[slot][rank];
      }
      p.flags = mappedFlags[rank];
      p.readerFlags = mappedReaderFlags[rank];
      if (options.gpuProgress)
        p.peerFlags = static_cast<Flags* const*>(devicePeerFlags[rank].data());
      if (options.hostRingReduceScatter)
        p.ringProgress =
            static_cast<Word* const*>(deviceRingPointers[rank].data());
      p.grid = grids[rank];
      p.plan = plan;
      p.packetElements = capacity;
      p.firstSequence = first;
      p.rank = rank;
      p.profile = profile;
      p.gpuReduce = gpuReduce;
      p.interleaveScatter = interleaveScatter;
      p.gpuProgress = options.gpuProgress;
      p.agPerSourceProgress = options.agPerSourceProgress;
      p.vectorBytes = options.transportVectorBytes;
      for (int slot = 0; slot < 2; ++slot) p.previous[slot] = previous[slot];
      p.producerBlocks = producerBlocks[rank];
      p.consumerBlocks = consumerBlocks[rank];
      for (int crew = 0; crew < 2; ++crew) {
        p.initialPhase[crew] = barrierPhases[rank][crew];
        if ((crew ? p.consumerBlocks : p.producerBlocks) > 1 &&
            options.backend == TransportBackend::GpuKernel)
          barrierPhases[rank][crew] += uint32_t(tiles);
      }
      int blocks = singleCta ? 1 : p.producerBlocks + p.consumerBlocks;
      auto& command = static_cast<Command*>(
          commandMemory[rank].data())[ticket % kQueueDepth];
      command.payload = p;
      cpuRelease(command.ready, uint32_t(ticket + 1));
      if (parallelLaunch)
        launchGraphs[rank] = graphs[rank][kernelKind][blocks];
      else if (!publishOnly && options.backend == TransportBackend::GpuKernel &&
               !options.concurrentProducer && !fusedGemm)
        CUDA(cudaGraphLaunch(graphs[rank][kernelKind][blocks],
                             ports[rank].stream));
    }
    if (parallelLaunch) launchGpuBatch();
    if (!publishOnly && fusedGemm)
      emitGemm(ports, producerBlocks, consumerBlocks, capacity);
    else if (!publishOnly && options.backend == TransportBackend::Dma)
      emitDma(plan, ports, capacity, tiles);
    else if (!publishOnly && options.backend == TransportBackend::GpuKernel &&
             options.concurrentProducer)
      emitConcurrent(plan, ports, binding, capacity, tiles, producerBlocks,
                     consumerBlocks);
  }
  void emitGemm(std::vector<PacketPort> const& ports,
                std::array<int, kMaxRanks> const& producers,
                std::array<int, kMaxRanks> const& consumers, size_t capacity) {
    for (int r = 0; r < ranks; ++r) {
      CUDA(cudaSetDevice(devices[r]));
      gemm->launch(r, ports[r].stream, mappedCommands[r], dispatches[r],
                   producers[r] + consumers[r], false, capacity);
    }
  }
  // One CPU command and ownership sequence, multiple GPU epochs. Every epoch
  // has CUDA-visible producer dependencies; no resident CTA waits on a kernel
  // that the graph scheduler may place behind it on the same work queue.
  void emitConcurrent(PacketProgram const& plan,
                      std::vector<PacketPort> const& ports,
                      PacketPipeline::Binding const* binding, size_t capacity,
                      size_t tiles, std::array<int, kMaxRanks> const& producers,
                      std::array<int, kMaxRanks> const& consumers) {
    size_t stripes =
        binding && binding->producerElements
            ? 1 + (plan.inputElements - 1) / binding->producerElements
            : 1;
    size_t begin = 0;
    for (size_t stripe = 0; stripe < stripes; ++stripe) {
      size_t end =
          stripe + 1 == stripes
              ? tiles
              : std::min(tiles,
                         ((stripe + 1) * binding->producerElements) / capacity);
      if (end <= begin) continue;
      for (int r = 0; r < ranks; ++r) {
        CUDA(cudaSetDevice(devices[r]));
        if (binding && binding->producerElements &&
            !binding->producerEvents[r].empty() && (plan.sources & (1U << r)))
          CUDA(cudaStreamWaitEvent(ports[r].stream,
                                   binding->producerEvents[r][stripe], 0));
        dispatchTransport<><<<producers[r] + consumers[r],
                              options.transportThreads, 0, ports[r].stream>>>(
            mappedCommands[r], dispatches[r], begin, end, end == tiles);
        CUDA(cudaGetLastError());
      }
      begin = end;
    }
  }
  void emitDma(PacketProgram const& plan, std::vector<PacketPort> const& ports,
               size_t capacity, size_t tiles) {
    for (int r = 0; r < ranks; ++r) {
      CUDA(cudaSetDevice(devices[r]));
      dmaBegin<<<1, 1, 0, ports[r].stream>>>(
          mappedCommands[r], dispatches[r],
          static_cast<Transport*>(dmaState[r].data()));
      CUDA(cudaEventRecord(forkEvents[r], ports[r].stream));
      CUDA(cudaStreamWaitEvent(uploadStreams[r], forkEvents[r], 0));
      CUDA(cudaStreamWaitEvent(downloadStreams[r], forkEvents[r], 0));
    }
    for (size_t tile = 0; tile < tiles; ++tile)
      for (int r = 0; r < ranks; ++r) {
        CUDA(cudaSetDevice(devices[r]));
        auto* state = static_cast<Transport*>(dmaState[r].data());
        auto frame = plan.frame(r, tile, capacity);
        int slot = tile & 1;
        dmaControl<<<1, 1, 0, uploadStreams[r]>>>(state, tile,
                                                  DmaAction::Acquire);
        for (int i = 0; i < frame.uploads; ++i) {
          auto move = frame.upload[i];
          auto src = MemorySpan::borrowedGpu(
              const_cast<uint16_t*>(
                  static_cast<uint16_t const*>(ports[r].input)) +
                  move.inputOffset,
              move.elements * 2, devices[r]);
          auto dst = inputs[slot][r].span().subspan(move.packetOffset * 2,
                                                    move.elements * 2);
          RouterCore::forwardAsync(*channels[r], dst, src, uploadStreams[r]);
        }
        dmaControl<<<1, 1, 0, uploadStreams[r]>>>(state, tile,
                                                  DmaAction::Publish);
        dmaControl<<<1, 1, 0, downloadStreams[r]>>>(state, tile,
                                                    DmaAction::Wait);
        for (int i = 0; i < frame.routes; ++i) {
          auto route = frame.route[i];
          auto dst = MemorySpan::borrowedGpu(
              static_cast<uint16_t*>(ports[r].output) + route.outputOffset,
              route.elements * 2, devices[r]);
          MemorySpan src;
          if (route.sourceRank < 0)
            src = outputs[slot][rankGroup[r]].span().subspan(
                route.sourceOffset * 2, route.elements * 2);
          else if (route.local)
            src = MemorySpan::borrowedGpu(
                const_cast<uint16_t*>(
                    static_cast<uint16_t const*>(ports[r].input)) +
                    route.sourceOffset,
                route.elements * 2, devices[r]);
          else
            src = inputs[slot][route.sourceRank].span().subspan(
                route.sourceOffset * 2, route.elements * 2);
          if (route.addRetained) {
            auto own = MemorySpan::borrowedGpu(
                const_cast<uint16_t*>(
                    static_cast<uint16_t const*>(ports[r].input)) +
                    route.retainedOffset,
                route.elements * 2, devices[r]);
            RouterCore::forwardAsync(
                *channels[r],
                dmaRetained[r].span().subspan(0, route.elements * 2), own,
                downloadStreams[r]);
          }
          if (src.address != dst.address)
            RouterCore::forwardAsync(*channels[r], dst, src,
                                     downloadStreams[r]);
          if (route.addRetained)
            dmaAdd<<<std::min(options.transportBlocks, 32),
                     options.transportThreads, 0, downloadStreams[r]>>>(
                static_cast<uint16_t*>(dst.address),
                static_cast<uint16_t const*>(dmaRetained[r].data()),
                route.elements);
        }
        dmaControl<<<1, 1, 0, downloadStreams[r]>>>(state, tile,
                                                    DmaAction::Consume);
      }
    for (int r = 0; r < ranks; ++r) {
      CUDA(cudaSetDevice(devices[r]));
      CUDA(cudaEventRecord(uploadEvents[r], uploadStreams[r]));
      CUDA(cudaEventRecord(downloadEvents[r], downloadStreams[r]));
      CUDA(cudaStreamWaitEvent(ports[r].stream, uploadEvents[r], 0));
      CUDA(cudaStreamWaitEvent(ports[r].stream, downloadEvents[r], 0));
      dmaFinish<<<1, 1, 0, ports[r].stream>>>(dispatches[r]);
      CUDA(cudaGetLastError());
    }
  }
  void beginCapture() {
    if (!initialized)
      throw std::invalid_argument("warm up router before capture");
    endReplay();
    captured.clear();
    recording = true;
  }
  void endCapture() {
    if (!recording) throw std::invalid_argument("router is not recording");
    recording = false;
  }
  void beginReplay() {
    std::lock_guard<std::mutex> lock(replayMutex);
    if (ranks == 1 && !recording && !replayPending &&
        std::none_of(captured.begin(), captured.end(),
                     [](CapturedCall const& call) { return call.fusedGemm; }))
      return;
    if (recording || captured.empty() || replayPending)
      throw std::invalid_argument(
          "invalid captured program or overlapping replay");
    if (options.inlineReplay && captured.size() <= kQueueDepth) {
      for (auto const& call : captured)
        if (ranks != 1 || call.fusedGemm)
          submit(call.plan, call.ports, true, &call.binding, call.fusedGemm);
      return;
    }
    if (!scheduler.joinable()) {
      scheduler = std::thread([this] {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        if (options.replayCpu >= 0) CPU_SET(options.replayCpu, &mask);
        if (options.replayCpu >= 0) sched_setaffinity(0, sizeof(mask), &mask);
        std::unique_lock<std::mutex> lock(replayMutex);
        while (true) {
          replayCondition.wait(
              lock, [this] { return schedulerStop || replayPending; });
          if (schedulerStop) return;
          lock.unlock();
          std::exception_ptr error;
          try {
            for (const auto& call : captured)
              if (ranks != 1 || call.fusedGemm)
                submit(call.plan, call.ports, true, &call.binding,
                       call.fusedGemm);
          } catch (...) {
            error = std::current_exception();
          }
          lock.lock();
          replayError = error;
          replayPending = false;
          replayCondition.notify_all();
        }
      });
    }
    replayError = nullptr;
    replayPending = true;
    replayCondition.notify_all();
  }
  void endReplay() {
    std::unique_lock<std::mutex> lock(replayMutex);
    replayCondition.wait(lock, [this] { return !replayPending; });
    if (replayError) std::rethrow_exception(replayError);
  }
};
PacketPipeline::PacketPipeline(PacketPipelineOptions options)
    : impl_(std::make_unique<Impl>(options)) {}
PacketPipeline::~PacketPipeline() = default;
void PacketPipeline::enqueue(PacketProgram const& plan,
                             std::vector<PacketPort> const& ports) {
  impl_->submit(plan, ports);
}
void PacketPipeline::prepare(std::vector<PacketPort> const& ports) {
  if (!impl_->initialized) {
    impl_->init(ports);
    return;
  }
  if (ports.size() != size_t(impl_->ranks))
    throw std::invalid_argument("prepared rank count changed");
  for (int r = 0; r < impl_->ranks; ++r)
    if (ports[r].device != impl_->devices[r] ||
        ports[r].stream != impl_->streams[r])
      throw std::invalid_argument("prepared device/stream order changed");
}
void PacketPipeline::enqueue(PacketProgram const& plan, Binding const& b) {
  if (!plan.inputElements) return;
  int ranks = plan.ranks;
  if (b.devices.size() != size_t(ranks) || b.inputs.size() != size_t(ranks) ||
      b.outputs.size() != size_t(ranks) || b.streams.size() != size_t(ranks) ||
      b.producerEvents.size() != size_t(ranks))
    throw std::invalid_argument("binding rank count mismatch");
  rankMask(ranks);
  std::vector<PacketPort> ports(ranks);
  size_t input = plan.inputElements * 2;
  size_t output = plan.layout == PacketLayout::Concatenate ? input * ranks
                  : plan.layout == PacketLayout::Scatter   ? input / ranks
                                                           : input;
  for (int r = 0; r < ranks; ++r) {
    if ((plan.sources & (1U << r)) &&
        (b.inputs[r].kind != MemoryKind::Gpu ||
         b.inputs[r].device != b.devices[r] || b.inputs[r].bytes < input))
      throw std::invalid_argument("invalid program input binding");
    if ((plan.destinations & (1U << r)) &&
        (b.outputs[r].kind != MemoryKind::Gpu ||
         b.outputs[r].device != b.devices[r] || b.outputs[r].bytes < output))
      throw std::invalid_argument("invalid program output binding");
    if (b.producerElements && (plan.sources & (1U << r))) {
      size_t stripes = 1 + (plan.inputElements - 1) / b.producerElements;
      if (!impl_->options.concurrentProducer ||
          impl_->options.backend != TransportBackend::GpuKernel ||
          (!b.producerEvents[r].empty() &&
           b.producerEvents[r].size() < stripes))
        throw std::invalid_argument(
            "invalid concurrent producer event binding");
      for (size_t stripe = 0; stripe < b.producerEvents[r].size(); ++stripe)
        if (!b.producerEvents[r][stripe])
          throw std::invalid_argument("null producer stripe event");
    }
    ports[r] = {b.inputs[r].address, b.outputs[r].address, b.streams[r],
                b.devices[r]};
  }
  impl_->submit(plan, ports, false, &b);
}
void PacketPipeline::prepareGemm(int m, int n, int k,
                                 std::vector<MemorySpan> const& a,
                                 std::vector<MemorySpan> const& b,
                                 std::vector<MemorySpan> const& partial,
                                 PacketLayout layout) {
  if (impl_->options.hostRingReduceScatter || !impl_->initialized ||
      impl_->gemm || m <= 0 || n <= 0 || k <= 0 || n % 8 || k % 8 ||
      impl_->options.backend != TransportBackend::GpuKernel ||
      impl_->options.transportThreads != 256)
    throw std::invalid_argument(
        "invalid resident GEMM shape, transport or preparation order");
  if (layout != PacketLayout::Replicate && layout != PacketLayout::Scatter)
    throw std::invalid_argument(
        "resident producer needs replicate or scatter output");
  int ranks = impl_->ranks;
  if (a.size() != size_t(ranks) || b.size() != size_t(ranks) ||
      partial.size() != size_t(ranks))
    throw std::invalid_argument("resident operand rank count mismatch");
  for (int r = 0; r < ranks; ++r) {
    auto valid = [r, this](MemorySpan const& span, size_t bytes) {
      return span.address && span.kind == MemoryKind::Gpu &&
             span.device == impl_->devices[r] && span.bytes >= bytes &&
             !(reinterpret_cast<uintptr_t>(span.address) & 15);
    };
    if (!valid(a[r], size_t(m) * k * 2) || !valid(b[r], size_t(n) * k * 2) ||
        !valid(partial[r], size_t(m) * n * 2))
      throw std::invalid_argument("invalid resident GEMM operands");
  }
  int tileM = impl_->options.gemmTileM, tileN = impl_->options.gemmTileN;
  if (!tileM && !tileN) {
    bool wide = (layout == PacketLayout::Scatter && m >= 2048) ||
                (m > 256 && k >= 8192);
    tileM = wide ? 128 : 64;
    tileN = tileM == 128 ? 256 : 128;
  }
  if (impl_->options.gemmMGroup < 0 || impl_->options.gemmMGroup > 64)
    throw std::invalid_argument("GEMM M grouping must be 0..64");
  int tileK = impl_->options.gemmTileK, stages = impl_->options.gemmStages;
  if (!tileK && !stages) {
    tileK = 32;
    stages = 4;
  }
  if (tileM == 128 && tileN == 128 && tileK == 32 && stages == 3)
    impl_->gemm = std::make_unique<ResidentGemm<128, 128, 32, 3>>(
        m, n, k, a, b, partial, impl_->devices, impl_->options.gemmMGroup);
  else if (tileM == 256 && tileN == 128 && tileK == 32 && stages == 4)
    impl_->gemm = std::make_unique<ResidentGemm<256, 128>>(
        m, n, k, a, b, partial, impl_->devices, impl_->options.gemmMGroup);
  else if (tileM == 256 && tileN == 128 && tileK == 64 && stages == 2)
    impl_->gemm = std::make_unique<ResidentGemm<256, 128, 64, 2>>(
        m, n, k, a, b, partial, impl_->devices, impl_->options.gemmMGroup);
  else if (tileM == 32 && tileN == 128 && tileK == 64 && stages == 2)
    impl_->gemm = std::make_unique<ResidentGemm<32, 128, 64, 2>>(
        m, n, k, a, b, partial, impl_->devices, impl_->options.gemmMGroup);
  else if (tileM == 128 && tileN == 256 && tileK == 64 && stages == 2)
    impl_->gemm = std::make_unique<ResidentGemm<128, 256, 64, 2>>(
        m, n, k, a, b, partial, impl_->devices, impl_->options.gemmMGroup);
  else if (tileM == 64 && tileN == 128 && tileK == 64 && stages == 2)
    impl_->gemm = std::make_unique<ResidentGemm<64, 128, 64, 2>>(
        m, n, k, a, b, partial, impl_->devices, impl_->options.gemmMGroup);
  else if (tileM == 128 && tileN == 256 && tileK == 32 && stages == 4)
    impl_->gemm = std::make_unique<ResidentGemm<128, 256>>(
        m, n, k, a, b, partial, impl_->devices, impl_->options.gemmMGroup);
  else if (tileM == 64 && tileN == 128 && tileK == 32 && stages == 4)
    impl_->gemm = std::make_unique<ResidentGemm<64, 128>>(
        m, n, k, a, b, partial, impl_->devices, impl_->options.gemmMGroup);
  else
    throw std::invalid_argument(
        "unsupported GEMM tile/pipeline; M/N=32x128, 64x128, 128x256, or "
        "256x128, K/stages=32/4 or "
        "64/2");
  for (int rank = 0; rank < ranks; ++rank) {
    impl_->maxBlocks[rank] =
        std::min(impl_->maxBlocks[rank], impl_->gemm->gridLimit(rank) - 1);
    if (impl_->maxBlocks[rank] < 2)
      throw std::invalid_argument("no co-resident transport and GEMM capacity");
  }
  size_t elements = size_t(m) * n;
  size_t capacity = packetCapacity(elements, layout, ranks, impl_->options);
  for (int rank = 0; rank < ranks; ++rank) {
    size_t work = std::min(capacity, elements);
    size_t egress = layout == PacketLayout::Scatter
                        ? std::min(work, elements / ranks)
                        : work;
    size_t grain = impl_->options.transportGrainBytes;
    int crew = residentTransportCrew(impl_->maxBlocks[rank], work, elements,
                                     layout, impl_->options.packetBytes,
                                     impl_->options.fusedTransportCrew);
    int upload = std::min<size_t>(
        crew, std::max<size_t>(1, (work * 2 + grain - 1) / grain));
    int download = std::min<size_t>(
        crew, std::max<size_t>(1, (egress * 2 + grain - 1) / grain));
    impl_->gemmTransportBlocks[rank] = upload + download;
  }
}
void PacketPipeline::prepareAllGatherGemm(
    int localM, int n, int k, std::vector<MemorySpan> const& localA,
    std::vector<MemorySpan> const& gatheredA, std::vector<MemorySpan> const& b,
    std::vector<MemorySpan> const& output) {
  int ranks = impl_->ranks;
  if (impl_->options.hostRingReduceScatter || !impl_->initialized ||
      localM <= 0 || n <= 0 || k <= 0 || localM > INT_MAX / ranks ||
      localA.size() != size_t(ranks))
    throw std::invalid_argument("invalid AllGather GEMM shape");
  for (int r = 0; r < ranks; ++r)
    if (!localA[r].address || localA[r].kind != MemoryKind::Gpu ||
        localA[r].device != impl_->devices[r] ||
        localA[r].bytes < size_t(localM) * k * 2 ||
        (reinterpret_cast<uintptr_t>(localA[r].address) & 15))
      throw std::invalid_argument("invalid local AllGather GEMM operand");
  prepareGemm(localM * ranks, n, k, gatheredA, b, output);
  impl_->gemm->configureGather(localM, localA);
  size_t elements = size_t(localM) * k;
  size_t capacity = packetCapacity(elements, PacketLayout::Concatenate, ranks,
                                   impl_->options);
  if (!capacity)
    throw std::invalid_argument("AllGather packet too small for rank count");
  size_t work = std::min(capacity, elements),
         grain = impl_->options.transportGrainBytes;
  for (int rank = 0; rank < ranks; ++rank) {
    int crew = std::max(1, impl_->maxBlocks[rank] / 2);
    int upload = std::min<size_t>(
        crew, std::max<size_t>(1, (work * 2 + grain - 1) / grain));
    int download = std::min<size_t>(
        crew, std::max<size_t>(1, (work * ranks * 2 + grain - 1) / grain));
    impl_->gemmTransportBlocks[rank] = upload + download;
  }
}
void PacketPipeline::enqueueGemm(PacketProgram const& plan,
                                 Binding const& binding) {
  int ranks = plan.ranks;
  rankMask(ranks);
  if (binding.inputs.size() != size_t(ranks) ||
      binding.outputs.size() != size_t(ranks) ||
      binding.devices.size() != size_t(ranks) ||
      binding.streams.size() != size_t(ranks))
    throw std::invalid_argument("resident binding rank count mismatch");
  std::vector<PacketPort> ports(ranks);
  for (int r = 0; r < ranks; ++r) {
    if (!binding.inputs[r].address || !binding.outputs[r].address ||
        binding.inputs[r].kind != MemoryKind::Gpu ||
        binding.outputs[r].kind != MemoryKind::Gpu ||
        binding.inputs[r].device != binding.devices[r] ||
        binding.outputs[r].device != binding.devices[r] ||
        binding.inputs[r].bytes < plan.inputElements * 2 ||
        binding.outputs[r].bytes <
            (plan.layout == PacketLayout::Concatenate
                 ? plan.inputElements * 2 * ranks
                 : plan.inputElements * 2 /
                       (plan.layout == PacketLayout::Scatter ? ranks : 1)))
      throw std::invalid_argument("invalid resident GEMM output binding");
    ports[r] = {binding.inputs[r].address, binding.outputs[r].address,
                binding.streams[r], binding.devices[r]};
  }
  impl_->submit(plan, ports, false, &binding, true);
}
void PacketPipeline::enqueueGemmProducerOnly(
    std::vector<cudaStream_t> const& streams) {
  int ranks = impl_->ranks;
  if (!impl_->gemm || streams.size() != size_t(ranks))
    throw std::invalid_argument(
        "resident GEMM is not prepared or stream count changed");
  for (int r = 0; r < ranks; ++r) {
    CUDA(cudaSetDevice(impl_->devices[r]));
    impl_->gemm->launch(r, streams[r], nullptr, nullptr,
                        impl_->gemmTransportBlocks[r], true);
  }
}
void PacketPipeline::beginCapture() { impl_->beginCapture(); }
void PacketPipeline::endCapture() { impl_->endCapture(); }
void PacketPipeline::beginReplay() { impl_->beginReplay(); }
void PacketPipeline::endReplay() { impl_->endReplay(); }
}  // namespace uccl_lite::cpu_router
