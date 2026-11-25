#include "uccl_lite/multiprocess_packet_ag.h"
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdlib>

// Build-time launch geometry for explicit strict-local candidate experiments.
// Legacy integrations keep their original 32-block configuration.
#ifndef UCCL_LITE_MP_AG_BLOCKS
#define UCCL_LITE_MP_AG_BLOCKS 32
#endif
#ifndef UCCL_LITE_MP_WAIT_LANES
#define UCCL_LITE_MP_WAIT_LANES 1
#endif
#ifndef UCCL_LITE_MP_AG_SINGLE_PRODUCERS
#define UCCL_LITE_MP_AG_SINGLE_PRODUCERS 8
#endif
#ifndef UCCL_LITE_MP_AG_SMALL_BLOCKS
#define UCCL_LITE_MP_AG_SMALL_BLOCKS 32
#endif
#ifndef UCCL_LITE_MP_RS_BLOCKS
#define UCCL_LITE_MP_RS_BLOCKS 32
#endif

namespace uccl_lite::multiprocess {
namespace {
namespace cg = cooperative_groups;

__device__ uint64_t acquire(PacketGeneration const* signal) {
  uint64_t value;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
               : "=l"(value)
               : "l"(&signal->value)
               : "memory");
  return value;
}

__device__ void publish(PacketGeneration* signal, uint64_t value) {
  asm volatile("st.release.sys.global.u64 [%0], %1;"
               :
               : "l"(&signal->value), "l"(value)
               : "memory");
}

__device__ uint64_t acquireGpu(uint64_t const* address) {
  uint64_t value;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];"
               : "=l"(value)
               : "l"(address)
               : "memory");
  return value;
}

__device__ uint32_t acquireGpu(uint32_t const* address) {
  uint32_t value;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
               : "=r"(value)
               : "l"(address)
               : "memory");
  return value;
}

__device__ uint32_t acquireReady(uint32_t const* address) {
  uint32_t value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(value)
               : "l"(address)
               : "memory");
  return value;
}

__device__ void publishGpu(uint64_t* address, uint64_t value) {
  asm volatile("st.release.gpu.global.u64 [%0], %1;"
               :
               : "l"(address), "l"(value)
               : "memory");
}

__device__ void crewBarrier(CrewState* state, uint32_t& phase, int blocks) {
  __syncthreads();
  ++phase;
  if (threadIdx.x == 0) {
    uint32_t arrivals;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                 : "=r"(arrivals)
                 : "l"(&state->arrivals), "r"(1)
                 : "memory");
    if (arrivals == uint32_t(blocks - 1)) {
      atomicExch(&state->arrivals, 0);
      asm volatile("st.release.gpu.global.u32 [%0], %1;"
                   :
                   : "l"(&state->generation), "r"(phase)
                   : "memory");
    }
    while (acquireGpu(&state->generation) < phase) __nanosleep(64);
  }
  __syncthreads();
}

__device__ uint16_t addRetained(uint16_t routed, uint16_t own) {
  float value = __uint_as_float(uint32_t(routed) << 16) +
                __uint_as_float(uint32_t(own) << 16);
  uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7fffffffU) > 0x7f800000U)
    bits |= 0x00400000U;
  else
    bits += 0x7fffU + ((bits >> 16) & 1U);
  return uint16_t(bits >> 16);
}

__device__ uint32_t addRetainedPair(uint32_t routed, uint32_t own) {
  return uint32_t(addRetained(uint16_t(routed), uint16_t(own))) |
         (uint32_t(addRetained(uint16_t(routed >> 16), uint16_t(own >> 16)))
          << 16);
}

__device__ void uploadRsStripe(uint16_t* packet, uint16_t const* input,
                               size_t shardCount, size_t begin, size_t length,
                               int rank, size_t lane, size_t stride,
                               bool skipOwn) {
  bool vector = ((reinterpret_cast<uintptr_t>(input) |
                  reinterpret_cast<uintptr_t>(packet)) &
                 15) == 0 &&
                shardCount % 8 == 0 && length % 8 == 0;
  if (vector) {
    size_t vectorsPerStripe = length / 8;
    for (size_t flat = lane; flat < size_t(kPacketRanks) * vectorsPerStripe;
         flat += stride) {
      int destination = int(flat / vectorsPerStripe);
      if (skipOwn && destination == rank) continue;
      size_t index = flat % vectorsPerStripe;
      uint16_t const* source =
          input + size_t(destination) * shardCount + begin + index * 8;
      uint16_t* target =
          packet + size_t(destination) * kRsStripeElements + index * 8;
      *reinterpret_cast<uint4*>(target) =
          *reinterpret_cast<uint4 const*>(source);
    }
    return;
  }
  for (int destination = 0; destination < kPacketRanks; ++destination) {
    if (skipOwn && destination == rank) continue;
    uint16_t const* source = input + size_t(destination) * shardCount + begin;
    uint16_t* target = packet + size_t(destination) * kRsStripeElements;
    for (size_t i = lane; i < length; i += stride) target[i] = source[i];
  }
}

__device__ void uploadRsSum2Stripe(uint16_t* packet, uint16_t const* input,
                                   uint16_t const* auxiliary,
                                   size_t shardCount, size_t begin,
                                   size_t length, int rank, size_t lane,
                                   size_t stride) {
  bool vector = ((reinterpret_cast<uintptr_t>(input) |
                  reinterpret_cast<uintptr_t>(auxiliary) |
                  reinterpret_cast<uintptr_t>(packet)) &
                 15) == 0 &&
                shardCount % 8 == 0 && length % 8 == 0;
  if (vector) {
    size_t vectorsPerStripe = length / 8;
    for (size_t flat = lane; flat < size_t(kPacketRanks) * vectorsPerStripe;
         flat += stride) {
      size_t destination = flat / vectorsPerStripe;
      if (destination == size_t(rank)) continue;
      size_t index = flat % vectorsPerStripe;
      size_t offset = destination * shardCount + begin + index * 8;
      size_t packetOffset = destination * kRsStripeElements + index * 8;
      uint4 first = *reinterpret_cast<uint4 const*>(input + offset);
      uint4 second = *reinterpret_cast<uint4 const*>(auxiliary + offset);
      *reinterpret_cast<uint4*>(packet + packetOffset) = make_uint4(
          addRetainedPair(first.x, second.x),
          addRetainedPair(first.y, second.y),
          addRetainedPair(first.z, second.z),
          addRetainedPair(first.w, second.w));
    }
    return;
  }
  for (size_t flat = lane; flat < size_t(kPacketRanks) * length;
       flat += stride) {
    size_t destination = flat / length;
    if (destination == size_t(rank)) continue;
    size_t index = flat % length;
    size_t sourceOffset = destination * shardCount + begin + index;
    packet[destination * kRsStripeElements + index] =
        addRetained(input[sourceOffset], auxiliary[sourceOffset]);
  }
}

__device__ uint4 sumRsPacked(PacketShared const* shared, int slot,
                             uint16_t const* own, size_t offset, int rank,
                             uint16_t const* ownAux = nullptr) {
  float sum[8] = {};
#pragma unroll
  for (int peer = 0; peer < kPacketRanks; ++peer) {
    uint16_t const* source =
        peer == rank ? own + offset
                     : shared->input[slot][peer] +
                           size_t(rank) * kRsStripeElements + offset;
    uint4 words = *reinterpret_cast<uint4 const*>(source);
    if (peer == rank && ownAux) {
      uint4 second = *reinterpret_cast<uint4 const*>(ownAux + offset);
      words = make_uint4(addRetainedPair(words.x, second.x),
                         addRetainedPair(words.y, second.y),
                         addRetainedPair(words.z, second.z),
                         addRetainedPair(words.w, second.w));
    }
    uint32_t packed[4] = {words.x, words.y, words.z, words.w};
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      sum[part * 2] += __uint_as_float(packed[part] << 16);
      sum[part * 2 + 1] += __uint_as_float(packed[part] & 0xffff0000U);
    }
  }
  uint32_t packed[4];
#pragma unroll
  for (int part = 0; part < 4; ++part)
    packed[part] =
        uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(sum[part * 2]))) |
        (uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(sum[part * 2 + 1])))
         << 16);
  return make_uint4(packed[0], packed[1], packed[2], packed[3]);
}

__device__ void reduceRsStripe(PacketShared const* shared, int slot,
                               uint16_t const* input, uint16_t* output,
                               size_t shardCount, size_t begin, size_t length,
                               int rank, size_t lane, size_t stride) {
  uint16_t const* own = input + size_t(rank) * shardCount + begin;
  uint16_t* target = output + begin;
  bool vector = ((reinterpret_cast<uintptr_t>(own) |
                  reinterpret_cast<uintptr_t>(target)) &
                 15) == 0 &&
                shardCount % 8 == 0;
  if (vector) {
    for (size_t i = lane; i < length / 8; i += stride)
      reinterpret_cast<uint4*>(target)[i] =
          sumRsPacked(shared, slot, own, i * 8, rank);
  }
  size_t scalarBegin = vector ? length / 8 * 8 : 0;
  for (size_t i = scalarBegin + lane; i < length; i += stride) {
    float sum = 0.0f;
#pragma unroll
    for (int peer = 0; peer < kPacketRanks; ++peer) {
      uint16_t bits =
          peer == rank
              ? own[i]
              : shared->input[slot][peer][size_t(rank) * kRsStripeElements + i];
      sum += __uint_as_float(uint32_t(bits) << 16);
    }
    target[i] = __bfloat16_as_ushort(__float2bfloat16_rn(sum));
  }
}

__global__ void allGatherPacket(PacketShared* shared, uint16_t* agInput,
                                uint16_t const* input, uint16_t* output,
                                size_t count, int rank, uint64_t base,
                                uint64_t prior0, uint64_t prior1) {
  auto grid = cg::this_grid();
  size_t lane = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = size_t(gridDim.x) * blockDim.x;
  size_t tiles = (count + kAgPacketElements - 1) / kAgPacketElements;
  for (size_t tile = 0; tile < tiles; ++tile) {
    int slot = int(tile & 1);
    uint64_t previous = tile >= 2 ? base + tile - 1 : slot ? prior1 : prior0;
    uint64_t generation = base + tile + 1;
    if (lane < UCCL_LITE_MP_WAIT_LANES && previous)
      for (int peer = int(lane); peer < kPacketRanks;
           peer += UCCL_LITE_MP_WAIT_LANES)
        while (acquire(&shared->consumed[slot][peer]) < previous)
          __nanosleep(64);
    grid.sync();

    size_t begin = tile * kAgPacketElements;
    size_t length = min(kAgPacketElements, count - begin);
    uint16_t* packet =
        agInput + (size_t(slot) * kPacketRanks + rank) * kAgPacketElements;
    bool alignedInput = (reinterpret_cast<uintptr_t>(input + begin) & 15) == 0;
    if (alignedInput) {
      for (size_t i = lane; i < length / 8; i += stride)
        reinterpret_cast<uint4*>(packet)[i] =
            reinterpret_cast<uint4 const*>(input + begin)[i];
      for (size_t i = length / 8 * 8 + lane; i < length; i += stride)
        packet[i] = input[begin + i];
    } else {
      for (size_t i = lane; i < length; i += stride)
        packet[i] = input[begin + i];
    }
    grid.sync();
    if (lane == 0) publish(&shared->uploaded[slot][rank], generation);
    if (lane < UCCL_LITE_MP_WAIT_LANES)
      for (int peer = int(lane); peer < kPacketRanks;
           peer += UCCL_LITE_MP_WAIT_LANES)
        while (acquire(&shared->uploaded[slot][peer]) < generation)
          __nanosleep(64);
    grid.sync();

    for (int peer = 0; peer < kPacketRanks; ++peer) {
      uint16_t const* source =
          peer == rank ? input + begin
                       : agInput + (size_t(slot) * kPacketRanks + peer) *
                                       kAgPacketElements;
      uint16_t* destination = output + size_t(peer) * count + begin;
      bool aligned = ((reinterpret_cast<uintptr_t>(destination) |
                       reinterpret_cast<uintptr_t>(source)) &
                      15) == 0;
      if (aligned) {
        for (size_t i = lane; i < length / 8; i += stride)
          reinterpret_cast<uint4*>(destination)[i] =
              reinterpret_cast<uint4 const*>(source)[i];
        for (size_t i = length / 8 * 8 + lane; i < length; i += stride)
          destination[i] = source[i];
      } else {
        for (size_t i = lane; i < length; i += stride)
          destination[i] = source[i];
      }
    }
    grid.sync();
    if (lane == 0) publish(&shared->consumed[slot][rank], generation);
    grid.sync();
  }
}

__global__ void allGatherGemm128x4096Packet(
    PacketShared* shared, uint16_t* agInput, uint16_t const* input,
    uint16_t* gathered, uint16_t const* weight, uint16_t* logits, int rank,
    uint64_t base, uint64_t prior0) {
  auto grid = cg::this_grid();
  size_t lane = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = size_t(gridDim.x) * blockDim.x;
  uint64_t generation = base + 1;
  if (lane == 0 && prior0)
    for (int peer = 0; peer < kPacketRanks; ++peer)
      while (acquire(&shared->consumed[0][peer]) < prior0) __nanosleep(64);
  grid.sync();

  uint16_t* packet = agInput + size_t(rank) * kAgPacketElements;
  for (size_t i = lane; i < 4096 / 8; i += stride)
    reinterpret_cast<uint4*>(packet)[i] =
        reinterpret_cast<uint4 const*>(input)[i];
  grid.sync();
  if (lane == 0) publish(&shared->uploaded[0][rank], generation);
  if (lane == 0)
    for (int peer = 0; peer < kPacketRanks; ++peer)
      while (acquire(&shared->uploaded[0][peer]) < generation) __nanosleep(64);
  grid.sync();

  for (size_t flat = lane; flat < size_t(kPacketRanks) * 4096 / 8;
       flat += stride) {
    int peer = int(flat / (4096 / 8));
    size_t index = flat % (4096 / 8);
    uint16_t const* source =
        peer == rank ? input : agInput + size_t(peer) * kAgPacketElements;
    reinterpret_cast<uint4*>(gathered + size_t(peer) * 4096)[index] =
        reinterpret_cast<uint4 const*>(source)[index];
  }
  grid.sync();
  if (lane == 0) publish(&shared->consumed[0][rank], generation);
  grid.sync();

  // Eight warps per CTA. Each warp computes four BF16 gate logits with FP32
  // accumulation; the packet kernel and GEMM share one cooperative launch.
  int warp = int(blockIdx.x) * 8 + int(threadIdx.x) / 32;
  int warpLane = int(threadIdx.x) % 32;
  for (int output = warp; output < 8 * 128; output += 32 * 8) {
    int row = output / 128;
    int column = output % 128;
    float sum = 0.f;
    for (int k = warpLane; k < 4096; k += 32) {
      float a = __uint_as_float(uint32_t(gathered[size_t(row) * 4096 + k])
                                << 16);
      float b = __uint_as_float(uint32_t(weight[size_t(column) * 4096 + k])
                                << 16);
      sum = fmaf(a, b, sum);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (warpLane == 0)
      logits[output] = __bfloat16_as_ushort(__float2bfloat16_rn(sum));
  }
}

__global__ void allGatherPacketPipelined(
    PacketShared* shared, uint16_t* agInput, CrewState* crews,
    uint16_t const* input, uint16_t* output, size_t count, int rank,
    uint64_t base, uint64_t prior0, uint64_t prior1, uint32_t initialPhase) {
  int producerBlocks =
      count <= kAgPacketElements ? UCCL_LITE_MP_AG_SINGLE_PRODUCERS : 8;
#if UCCL_LITE_MP_WAIT_LANES > 1
  bool perSource = count <= kAgPacketElements;
#endif
  int consumerBlocks = int(gridDim.x) - producerBlocks;
  bool producer = blockIdx.x < producerBlocks;
  int block = producer ? int(blockIdx.x) : int(blockIdx.x) - producerBlocks;
  size_t lane = size_t(block) * blockDim.x + threadIdx.x;
  size_t stride =
      size_t(producer ? producerBlocks : consumerBlocks) * blockDim.x;
  CrewState* state = crews + (producer ? 0 : 1);
  uint32_t phase = initialPhase;
  size_t tiles = (count + kAgPacketElements - 1) / kAgPacketElements;
  for (size_t tile = 0; tile < tiles; ++tile) {
    int slot = int(tile & 1);
    uint64_t previous = tile >= 2 ? base + tile - 1 : slot ? prior1 : prior0;
    uint64_t generation = base + tile + 1;
    size_t begin = tile * kAgPacketElements;
    size_t length = min(kAgPacketElements, count - begin);
#if UCCL_LITE_MP_WAIT_LANES > 1
    // Consumers follow individual sources. A slow upload no longer prevents
    // every other source from making progress into the final output.
    if (!producer && perSource) {
      int peer = block % kPacketRanks;
      if (threadIdx.x == 0) {
        if (block == peer) {
          if (peer != rank)
            while (acquire(&shared->uploaded[slot][peer]) < generation)
              __nanosleep(64);
          publishGpu(&state->sourceReady[peer], generation);
        } else {
          while (acquireGpu(&state->sourceReady[peer]) < generation)
            __nanosleep(64);
        }
      }
    } else if (block == 0) {
      if (threadIdx.x < UCCL_LITE_MP_WAIT_LANES)
        for (int peer = int(threadIdx.x); peer < kPacketRanks;
             peer += UCCL_LITE_MP_WAIT_LANES) {
          if (producer) {
            if (previous)
              while (acquire(&shared->consumed[slot][peer]) < previous)
                __nanosleep(64);
          } else {
            while (acquire(&shared->uploaded[slot][peer]) < generation)
              __nanosleep(64);
          }
        }
      __syncthreads();
      if (threadIdx.x == 0) publishGpu(&state->ready, generation);
    } else if (threadIdx.x == 0) {
      while (acquireGpu(&state->ready) < generation) __nanosleep(64);
    }
#else
    if (block == 0 && threadIdx.x == 0) {
      if (producer) {
        if (previous)
          for (int peer = 0; peer < kPacketRanks; ++peer)
            while (acquire(&shared->consumed[slot][peer]) < previous)
              __nanosleep(64);
      } else {
        for (int peer = 0; peer < kPacketRanks; ++peer)
          while (acquire(&shared->uploaded[slot][peer]) < generation)
            __nanosleep(64);
      }
      publishGpu(&state->ready, generation);
    } else if (threadIdx.x == 0) {
      while (acquireGpu(&state->ready) < generation) __nanosleep(64);
    }
#endif
    __syncthreads();
    bool vector = (count & 7) == 0 && (length & 7) == 0 &&
                  (reinterpret_cast<uintptr_t>(input) & 15) == 0 &&
                  (reinterpret_cast<uintptr_t>(output) & 15) == 0;
    if (producer) {
      uint16_t* packet =
          agInput + (size_t(slot) * kPacketRanks + rank) * kAgPacketElements;
      if (vector) {
        for (size_t i = lane; i < length / 8; i += stride)
          reinterpret_cast<uint4*>(packet)[i] =
              reinterpret_cast<uint4 const*>(input + begin)[i];
      } else {
        for (size_t i = lane; i < length; i += stride)
          packet[i] = input[begin + i];
      }
      crewBarrier(state, phase, producerBlocks);
      if (block == 0 && threadIdx.x == 0)
        publish(&shared->uploaded[slot][rank], generation);
    } else {
#if UCCL_LITE_MP_WAIT_LANES > 1
      if (perSource) {
        int peer = block % kPacketRanks;
        size_t groupLane =
            size_t(block / kPacketRanks) * blockDim.x + threadIdx.x;
        size_t groupStride = size_t(consumerBlocks / kPacketRanks) * blockDim.x;
        uint16_t const* source =
            peer == rank ? input + begin
                         : agInput + (size_t(slot) * kPacketRanks + peer) *
                                         kAgPacketElements;
        uint16_t* target = output + size_t(peer) * count + begin;
        if (vector) {
          for (size_t i = groupLane; i < length / 8; i += groupStride)
            reinterpret_cast<uint4*>(target)[i] =
                reinterpret_cast<uint4 const*>(source)[i];
        } else {
          for (size_t i = groupLane; i < length; i += groupStride)
            target[i] = source[i];
        }
      } else
#endif
      {
        if (vector) {
          size_t vectorsPerPeer = length / 8;
          for (size_t flat = lane; flat < size_t(kPacketRanks) * vectorsPerPeer;
               flat += stride) {
            int peer = int(flat / vectorsPerPeer);
            size_t i = flat % vectorsPerPeer;
            uint16_t const* source =
                peer == rank ? input + begin
                             : agInput + (size_t(slot) * kPacketRanks + peer) *
                                             kAgPacketElements;
            uint16_t* target = output + size_t(peer) * count + begin;
            reinterpret_cast<uint4*>(target)[i] =
                reinterpret_cast<uint4 const*>(source)[i];
          }
        } else {
          for (int peer = 0; peer < kPacketRanks; ++peer) {
            uint16_t const* source =
                peer == rank ? input + begin
                             : agInput + (size_t(slot) * kPacketRanks + peer) *
                                             kAgPacketElements;
            uint16_t* target = output + size_t(peer) * count + begin;
            for (size_t i = lane; i < length; i += stride)
              target[i] = source[i];
          }
        }
      }
      crewBarrier(state, phase, consumerBlocks);
      if (block == 0 && threadIdx.x == 0)
        publish(&shared->consumed[slot][rank], generation);
    }
  }
}

__global__ void allReduceSmallPacket(PacketShared* shared,
                                     uint16_t const* input, uint16_t* output,
                                     size_t count, int rank,
                                     uint64_t generation, uint64_t previous) {
  auto grid = cg::this_grid();
  size_t lane = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = size_t(gridDim.x) * blockDim.x;
  if (lane < kPacketRanks && previous)
    while (acquire(&shared->consumed[0][lane]) < previous) __nanosleep(64);
  grid.sync();
  for (size_t i = lane; i < count; i += stride)
    shared->input[0][rank][i] = input[i];
  grid.sync();
  if (lane == 0) publish(&shared->uploaded[0][rank], generation);
  if (lane < kPacketRanks)
    while (acquire(&shared->uploaded[0][lane]) < generation) __nanosleep(64);
  grid.sync();
  for (size_t i = lane; i < count; i += stride) {
    float sum = 0.0f;
#pragma unroll
    for (int peer = 0; peer < kPacketRanks; ++peer) {
      uint16_t bits = peer == rank ? input[i] : shared->input[0][peer][i];
      sum += __uint_as_float(uint32_t(bits) << 16);
    }
    output[i] = __bfloat16_as_ushort(__float2bfloat16_rn(sum));
  }
  grid.sync();
  if (lane == 0) publish(&shared->consumed[0][rank], generation);
}

__global__ void reduceScatterPacket(PacketShared* shared, uint16_t const* input,
                                    uint16_t* output, size_t shardCount,
                                    int rank, uint64_t base, uint64_t prior0,
                                    uint64_t prior1) {
  auto grid = cg::this_grid();
  size_t lane = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = size_t(gridDim.x) * blockDim.x;
  size_t tiles = (shardCount + kRsStripeElements - 1) / kRsStripeElements;
  for (size_t tile = 0; tile < tiles; ++tile) {
    int slot = int(tile & 1);
    uint64_t previous = tile >= 2 ? base + tile - 1 : slot ? prior1 : prior0;
    uint64_t generation = base + tile + 1;
    if (lane == 0 && previous)
      for (int peer = 0; peer < kPacketRanks; ++peer)
        while (acquire(&shared->consumed[slot][peer]) < previous)
          __nanosleep(64);
    grid.sync();

    size_t begin = tile * kRsStripeElements;
    size_t length = min(kRsStripeElements, shardCount - begin);
    uploadRsStripe(shared->input[slot][rank], input, shardCount, begin, length,
                   rank, lane, stride, true);
    grid.sync();
    if (lane == 0) publish(&shared->uploaded[slot][rank], generation);
    if (lane == 0)
      for (int peer = 0; peer < kPacketRanks; ++peer)
        while (acquire(&shared->uploaded[slot][peer]) < generation)
          __nanosleep(64);
    grid.sync();

    reduceRsStripe(shared, slot, input, output, shardCount, begin, length, rank,
                   lane, stride);
    grid.sync();
    if (lane == 0) publish(&shared->consumed[slot][rank], generation);
    grid.sync();
  }
}

__global__ void reduceScatterSum2Packet(
    PacketShared* shared, uint16_t const* input, uint16_t const* auxiliary,
    uint16_t* output, size_t shardCount, int rank, uint64_t base,
    uint64_t prior0, uint64_t prior1) {
  auto grid = cg::this_grid();
  size_t lane = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = size_t(gridDim.x) * blockDim.x;
  size_t tiles = (shardCount + kRsStripeElements - 1) / kRsStripeElements;
  for (size_t tile = 0; tile < tiles; ++tile) {
    int slot = int(tile & 1);
    uint64_t previous = tile >= 2 ? base + tile - 1 : slot ? prior1 : prior0;
    uint64_t generation = base + tile + 1;
    if (lane == 0 && previous)
      for (int peer = 0; peer < kPacketRanks; ++peer)
        while (acquire(&shared->consumed[slot][peer]) < previous)
          __nanosleep(64);
    grid.sync();

    size_t begin = tile * kRsStripeElements;
    size_t length = min(kRsStripeElements, shardCount - begin);
    uploadRsSum2Stripe(shared->input[slot][rank], input, auxiliary, shardCount,
                       begin, length, rank, lane, stride);
    grid.sync();
    if (lane == 0) publish(&shared->uploaded[slot][rank], generation);
    if (lane == 0)
      for (int peer = 0; peer < kPacketRanks; ++peer)
        while (acquire(&shared->uploaded[slot][peer]) < generation)
          __nanosleep(64);
    grid.sync();

    uint16_t const* own = input + size_t(rank) * shardCount + begin;
    uint16_t const* ownAux = auxiliary + size_t(rank) * shardCount + begin;
    uint16_t* target = output + begin;
    for (size_t i = lane; i < length / 8; i += stride)
      reinterpret_cast<uint4*>(target)[i] =
          sumRsPacked(shared, slot, own, i * 8, rank, ownAux);
    for (size_t i = length / 8 * 8 + lane; i < length; i += stride) {
      float sum = 0.0f;
#pragma unroll
      for (int peer = 0; peer < kPacketRanks; ++peer)
        sum += __uint_as_float(uint32_t(
                                   peer == rank
                                       ? addRetained(own[i], ownAux[i])
                                       : shared->input[slot][peer]
                                                      [size_t(rank) *
                                                           kRsStripeElements +
                                                       i])
                               << 16);
      target[i] = __bfloat16_as_ushort(__float2bfloat16_rn(sum));
    }
    grid.sync();
    if (lane == 0) publish(&shared->consumed[slot][rank], generation);
    grid.sync();
  }
}

__global__ void reduceScatterReadySum2Packet(
    PacketShared* shared, uint16_t const* input, uint16_t const* auxiliary,
    uint16_t* output, uint32_t const* ready, int rank, uint64_t base,
    uint64_t prior0) {
  int destination = int(blockIdx.x);
  int lane = int(threadIdx.x);
  uint64_t generation = base + 1;
  if (lane == 0 && prior0)
    for (int peer = 0; peer < kPacketRanks; ++peer)
      while (acquire(&shared->consumed[0][peer]) < prior0) __nanosleep(64);
  __syncthreads();

  if (lane == 0)
    while (acquireReady(ready + destination) < 1) __nanosleep(64);
  __syncthreads();
  if (destination != rank) {
    uint16_t const* first = input + size_t(destination) * 4096;
    uint16_t const* second = auxiliary + size_t(destination) * 4096;
    uint16_t* packet =
        shared->input[0][rank] + size_t(destination) * kRsStripeElements;
    for (int i = lane; i < 4096 / 8; i += blockDim.x) {
      uint4 a = reinterpret_cast<uint4 const*>(first)[i];
      uint4 b = reinterpret_cast<uint4 const*>(second)[i];
      reinterpret_cast<uint4*>(packet)[i] =
          make_uint4(addRetainedPair(a.x, b.x), addRetainedPair(a.y, b.y),
                     addRetainedPair(a.z, b.z), addRetainedPair(a.w, b.w));
    }
  }
  __syncthreads();
  if (lane == 0)
    publish(&shared->uploadedDest[0][rank][destination], generation);

  if (destination == rank) {
    if (lane == 0)
      for (int peer = 0; peer < kPacketRanks; ++peer)
        while (acquire(&shared->uploadedDest[0][peer][rank]) < generation)
          __nanosleep(64);
    __syncthreads();
    uint16_t const* own = input + size_t(rank) * 4096;
    uint16_t const* ownAux = auxiliary + size_t(rank) * 4096;
    for (int i = lane; i < 4096 / 8; i += blockDim.x)
      reinterpret_cast<uint4*>(output)[i] =
          sumRsPacked(shared, 0, own, size_t(i) * 8, rank, ownAux);
    __syncthreads();
    if (lane == 0) publish(&shared->consumed[0][rank], generation);
  }
}

__global__ void reduceScatterPipeline(PacketShared* shared, CrewState* crews,
                                      uint16_t const* input, uint16_t* output,
                                      size_t shardCount, int rank,
                                      uint64_t base, uint64_t prior0,
                                      uint64_t prior1, uint32_t initialPhase) {
  constexpr int crewBlocks = 16;
  bool producer = blockIdx.x < crewBlocks;
  int block = producer ? int(blockIdx.x) : int(blockIdx.x) - crewBlocks;
  size_t lane = size_t(block) * blockDim.x + threadIdx.x;
  size_t stride = size_t(crewBlocks) * blockDim.x;
  CrewState* state = crews + (producer ? 0 : 1);
  uint32_t phase = initialPhase;
  size_t tiles = (shardCount + kRsStripeElements - 1) / kRsStripeElements;
  for (size_t tile = 0; tile < tiles; ++tile) {
    int slot = int(tile & 1);
    uint64_t previous = tile >= 2 ? base + tile - 1 : slot ? prior1 : prior0;
    uint64_t generation = base + tile + 1;
    size_t begin = tile * kRsStripeElements;
    size_t length = min(kRsStripeElements, shardCount - begin);
    if (block == 0 && threadIdx.x == 0) {
      if (producer) {
        if (previous)
          for (int peer = 0; peer < kPacketRanks; ++peer)
            while (acquire(&shared->consumed[slot][peer]) < previous)
              __nanosleep(64);
      } else {
        for (int peer = 0; peer < kPacketRanks; ++peer)
          while (acquire(&shared->uploaded[slot][peer]) < generation)
            __nanosleep(64);
      }
      publishGpu(&state->ready, generation);
    } else {
      while (acquireGpu(&state->ready) < generation) __nanosleep(64);
    }
    __syncthreads();

    if (producer) {
      uploadRsStripe(shared->input[slot][rank], input, shardCount, begin,
                     length, rank, lane, stride, true);
      crewBarrier(state, phase, crewBlocks);
      if (block == 0 && threadIdx.x == 0)
        publish(&shared->uploaded[slot][rank], generation);
    } else {
      reduceRsStripe(shared, slot, input, output, shardCount, begin, length,
                     rank, lane, stride);
      crewBarrier(state, phase, crewBlocks);
      if (block == 0 && threadIdx.x == 0)
        publish(&shared->consumed[slot][rank], generation);
    }
  }
}

__global__ void reduceScatterCpuPacket(PacketShared* shared, CrewState* crews,
                                       uint16_t const* input, uint16_t* output,
                                       size_t shardCount, int rank,
                                       uint64_t base, uint64_t prior0,
                                       uint64_t prior1, uint32_t initialPhase,
                                       uint32_t previousCpuMask,
                                       bool splitCredits, bool retainOwn) {
  int producerBlocks = shardCount >= (1UL << 20) ? 8 : int(gridDim.x) / 2;
#ifdef UCCL_LITE_MP_RS_SINGLE_TILE_PRODUCERS
  if (shardCount >= (1UL << 16) && shardCount <= kRsStripeElements)
    producerBlocks = UCCL_LITE_MP_RS_SINGLE_TILE_PRODUCERS;
#endif
  int consumerBlocks = int(gridDim.x) - producerBlocks;
  bool producer = blockIdx.x < producerBlocks;
  int block = producer ? int(blockIdx.x) : int(blockIdx.x) - producerBlocks;
  size_t lane = size_t(block) * blockDim.x + threadIdx.x;
  size_t stride =
      size_t(producer ? producerBlocks : consumerBlocks) * blockDim.x;
  CrewState* state = crews + (producer ? 0 : 1);
  uint32_t phase = initialPhase;
  size_t tiles = (shardCount + kRsStripeElements - 1) / kRsStripeElements;
  for (size_t tile = 0; tile < tiles; ++tile) {
    int slot = int(tile & 1);
    uint64_t previous = tile >= 2 ? base + tile - 1 : slot ? prior1 : prior0;
    uint64_t generation = base + tile + 1;
    size_t begin = tile * kRsStripeElements;
    size_t length = min(kRsStripeElements, shardCount - begin);
    if (block == 0 && threadIdx.x == 0) {
      if (producer) {
        bool priorCpu =
            splitCredits && (tile >= 2 || (previousCpuMask & (1U << slot)));
        if (previous)
          for (int peer = 0; peer < kPacketRanks; ++peer)
            while (acquire(priorCpu ? &shared->inputReadDone[slot][peer]
                                    : &shared->consumed[slot][peer]) < previous)
              __nanosleep(64);
      } else {
        while (acquire(&shared->routed[slot][rank]) < generation)
          __nanosleep(64);
      }
      publishGpu(&state->ready, generation);
    } else {
      while (acquireGpu(&state->ready) < generation) __nanosleep(64);
    }
    __syncthreads();

    if (producer) {
      uploadRsStripe(shared->input[slot][rank], input, shardCount, begin,
                     length, rank, lane, stride, retainOwn);
      crewBarrier(state, phase, producerBlocks);
      if (block == 0 && threadIdx.x == 0)
        publish(&shared->uploaded[slot][rank], generation);
    } else {
      uint16_t const* source = shared->result[slot][rank / 4][rank];
      uint16_t* target = output + begin;
      bool vector =
          (reinterpret_cast<uintptr_t>(target) & 15) == 0 &&
          (!retainOwn || (reinterpret_cast<uintptr_t>(
                              input + size_t(rank) * shardCount + begin) &
                          15) == 0);
      if (vector) {
        for (size_t i = lane; i < length / 8; i += stride) {
          uint4 result = reinterpret_cast<uint4 const*>(source)[i];
          if (retainOwn) {
            uint4 own = reinterpret_cast<uint4 const*>(
                input + size_t(rank) * shardCount + begin)[i];
            result.x = addRetainedPair(result.x, own.x);
            result.y = addRetainedPair(result.y, own.y);
            result.z = addRetainedPair(result.z, own.z);
            result.w = addRetainedPair(result.w, own.w);
          }
          reinterpret_cast<uint4*>(target)[i] = result;
        }
        for (size_t i = length / 8 * 8 + lane; i < length; i += stride)
          target[i] =
              retainOwn
                  ? addRetained(source[i],
                                input[size_t(rank) * shardCount + begin + i])
                  : source[i];
      } else {
        for (size_t i = lane; i < length; i += stride)
          target[i] =
              retainOwn
                  ? addRetained(source[i],
                                input[size_t(rank) * shardCount + begin + i])
                  : source[i];
      }
      crewBarrier(state, phase, consumerBlocks);
      if (block == 0 && threadIdx.x == 0)
        publish(&shared->consumed[slot][rank], generation);
    }
  }
}

__global__ void reduceScatterDestReady(PacketShared* shared, CrewState* crews,
                                       uint16_t const* input, uint16_t* output,
                                       size_t shardCount, int rank,
                                       uint64_t base, uint64_t prior0,
                                       uint64_t prior1, uint32_t initialPhase) {
  bool producer = blockIdx.x < kPacketRanks;
  int block = producer ? int(blockIdx.x) : int(blockIdx.x) - kPacketRanks;
  constexpr int consumerBlocks = 16;
  size_t lane = size_t(block) * blockDim.x + threadIdx.x;
  size_t stride = size_t(consumerBlocks) * blockDim.x;
  CrewState* consumerState = crews + 1;
  uint32_t phase = initialPhase;
  size_t tiles = (shardCount + kRsStripeElements - 1) / kRsStripeElements;
  for (size_t tile = 0; tile < tiles; ++tile) {
    int slot = int(tile & 1);
    uint64_t previous = tile >= 2 ? base + tile - 1 : slot ? prior1 : prior0;
    uint64_t generation = base + tile + 1;
    size_t begin = tile * kRsStripeElements;
    size_t length = min(kRsStripeElements, shardCount - begin);
    if (producer) {
      int destination = block;
      if (threadIdx.x == 0 && previous)
        for (int peer = 0; peer < kPacketRanks; ++peer)
          while (acquire(&shared->consumed[slot][peer]) < previous)
            __nanosleep(64);
      __syncthreads();
      uint16_t const* source = input + size_t(destination) * shardCount + begin;
      uint16_t* target =
          shared->input[slot][rank] + size_t(destination) * kRsStripeElements;
      bool vector = ((reinterpret_cast<uintptr_t>(source) |
                      reinterpret_cast<uintptr_t>(target)) &
                     15) == 0;
      if (vector) {
        for (size_t i = threadIdx.x; i < length / 8; i += blockDim.x)
          reinterpret_cast<uint4*>(target)[i] =
              reinterpret_cast<uint4 const*>(source)[i];
        for (size_t i = length / 8 * 8 + threadIdx.x; i < length;
             i += blockDim.x)
          target[i] = source[i];
      } else {
        for (size_t i = threadIdx.x; i < length; i += blockDim.x)
          target[i] = source[i];
      }
      __syncthreads();
      if (threadIdx.x == 0)
        publish(&shared->uploadedDest[slot][rank][destination], generation);
    } else {
      if (block == 0 && threadIdx.x == 0) {
        while (acquire(&shared->routed[slot][rank]) < generation)
          __nanosleep(64);
        publishGpu(&consumerState->ready, generation);
      } else {
        while (acquireGpu(&consumerState->ready) < generation) __nanosleep(64);
      }
      __syncthreads();
      uint16_t const* source = shared->result[slot][rank / 4][rank];
      uint16_t* target = output + begin;
      bool vector = (reinterpret_cast<uintptr_t>(target) & 15) == 0;
      if (vector) {
        for (size_t i = lane; i < length / 8; i += stride)
          reinterpret_cast<uint4*>(target)[i] =
              reinterpret_cast<uint4 const*>(source)[i];
        for (size_t i = length / 8 * 8 + lane; i < length; i += stride)
          target[i] = source[i];
      } else {
        for (size_t i = lane; i < length; i += stride) target[i] = source[i];
      }
      crewBarrier(consumerState, phase, consumerBlocks);
      if (block == 0 && threadIdx.x == 0)
        publish(&shared->consumed[slot][rank], generation);
    }
  }
}

__global__ void allReduceCpuPacket(PacketShared* shared, CrewState* crews,
                                   uint16_t const* input, uint16_t* output,
                                   size_t count, int rank, uint64_t base,
                                   uint64_t prior0, uint64_t prior1,
                                   uint32_t initialPhase, bool replicateResults,
                                   uint32_t previousCpuMask,
                                   bool splitCredits) {
  constexpr int producerBlocks = 8;
  int consumerBlocks = int(gridDim.x) - producerBlocks;
  bool producer = blockIdx.x < producerBlocks;
  int block = producer ? int(blockIdx.x) : int(blockIdx.x) - producerBlocks;
  size_t lane = size_t(block) * blockDim.x + threadIdx.x;
  size_t stride =
      size_t(producer ? producerBlocks : consumerBlocks) * blockDim.x;
  CrewState* state = crews + (producer ? 0 : 1);
  uint32_t phase = initialPhase;
  size_t maxShard = (count + kPacketRanks - 1) / kPacketRanks;
  size_t stripeElements =
      count <= (1UL << 19) ? kRsStripeElements / 2 : kRsStripeElements;
  size_t tiles = (maxShard + stripeElements - 1) / stripeElements;
  for (size_t tile = 0; tile < tiles; ++tile) {
    int slot = int(tile & 1);
    uint64_t previous = tile >= 2 ? base + tile - 1 : slot ? prior1 : prior0;
    uint64_t generation = base + tile + 1;
    size_t begin = tile * stripeElements;
    if (producer) {
#if UCCL_LITE_MP_WAIT_LANES > 1
      if (block == 0) {
        bool priorCpu =
            splitCredits && (tile >= 2 || (previousCpuMask & (1U << slot)));
        if (previous && threadIdx.x < UCCL_LITE_MP_WAIT_LANES)
          for (int peer = int(threadIdx.x); peer < kPacketRanks;
               peer += UCCL_LITE_MP_WAIT_LANES)
            while (acquire(priorCpu ? &shared->inputReadDone[slot][peer]
                                    : &shared->consumed[slot][peer]) < previous)
              __nanosleep(64);
        __syncthreads();
        if (threadIdx.x == 0) publishGpu(&state->ready, generation);
      } else if (threadIdx.x == 0) {
        while (acquireGpu(&state->ready) < generation) __nanosleep(64);
      }
#else
      if (block == 0 && threadIdx.x == 0) {
        bool priorCpu =
            splitCredits && (tile >= 2 || (previousCpuMask & (1U << slot)));
        if (previous)
          for (int peer = 0; peer < kPacketRanks; ++peer)
            while (acquire(priorCpu ? &shared->inputReadDone[slot][peer]
                                    : &shared->consumed[slot][peer]) < previous)
              __nanosleep(64);
        publishGpu(&state->ready, generation);
      } else if (block != 0 || threadIdx.x != 0) {
        while (acquireGpu(&state->ready) < generation) __nanosleep(64);
      }
#endif
    } else if (threadIdx.x == 0) {
      int sourceRank = block % kPacketRanks;
      while (acquire(&shared->routed[slot][sourceRank]) < generation)
        __nanosleep(64);
    }
    __syncthreads();

    if (producer) {
      uint16_t* packet = shared->input[slot][rank];
      if (count % kPacketRanks == 0 && count / kPacketRanks % 8 == 0 &&
          begin < count / kPacketRanks) {
        size_t shard = count / kPacketRanks;
        size_t length = min(stripeElements, shard - begin);
        size_t vectorsPerStripe = length / 8;
        for (size_t flat = lane; flat < size_t(kPacketRanks) * vectorsPerStripe;
             flat += stride) {
          int destination = int(flat / vectorsPerStripe);
          size_t index = flat % vectorsPerStripe;
          uint16_t const* source =
              input + size_t(destination) * shard + begin + index * 8;
          uint16_t* target =
              packet + size_t(destination) * kRsStripeElements + index * 8;
          *reinterpret_cast<uint4*>(target) =
              *reinterpret_cast<uint4 const*>(source);
        }
      } else
        for (int destination = 0; destination < kPacketRanks; ++destination) {
          size_t partBegin = count * size_t(destination) / kPacketRanks;
          size_t partEnd = count * size_t(destination + 1) / kPacketRanks;
          size_t partLength = partEnd - partBegin;
          if (begin >= partLength) continue;
          size_t length = min(stripeElements, partLength - begin);
          uint16_t const* source = input + partBegin + begin;
          uint16_t* target = packet + size_t(destination) * kRsStripeElements;
          bool vector = ((reinterpret_cast<uintptr_t>(source) |
                          reinterpret_cast<uintptr_t>(target)) &
                         15) == 0;
          if (vector) {
            for (size_t i = lane; i < length / 8; i += stride)
              reinterpret_cast<uint4*>(target)[i] =
                  reinterpret_cast<uint4 const*>(source)[i];
            for (size_t i = length / 8 * 8 + lane; i < length; i += stride)
              target[i] = source[i];
          } else {
            for (size_t i = lane; i < length; i += stride)
              target[i] = source[i];
          }
        }
      crewBarrier(state, phase, producerBlocks);
      if (block == 0 && threadIdx.x == 0)
        publish(&shared->uploaded[slot][rank], generation);
    } else {
      int sourceRank = block % kPacketRanks;
      size_t groupLane =
          size_t(block / kPacketRanks) * blockDim.x + threadIdx.x;
      size_t groupStride = size_t(consumerBlocks / kPacketRanks) * blockDim.x;
      size_t partBegin = count * size_t(sourceRank) / kPacketRanks;
      size_t partEnd = count * size_t(sourceRank + 1) / kPacketRanks;
      size_t partLength = partEnd - partBegin;
      if (begin < partLength) {
        size_t length = min(stripeElements, partLength - begin);
        int bank = replicateResults ? rank / 4 : sourceRank / 4;
        uint16_t const* source = shared->result[slot][bank][sourceRank];
        uint16_t* target = output + partBegin + begin;
        bool vector = (reinterpret_cast<uintptr_t>(target) & 15) == 0;
        if (vector) {
          for (size_t i = groupLane; i < length / 8; i += groupStride)
            reinterpret_cast<uint4*>(target)[i] =
                reinterpret_cast<uint4 const*>(source)[i];
          for (size_t i = length / 8 * 8 + groupLane; i < length;
               i += groupStride)
            target[i] = source[i];
        } else {
          for (size_t i = groupLane; i < length; i += groupStride)
            target[i] = source[i];
        }
      }
      crewBarrier(state, phase, consumerBlocks);
      if (block == 0 && threadIdx.x == 0)
        publish(&shared->consumed[slot][rank], generation);
    }
  }
}
}  // namespace

cudaError_t launchAgPacket(PacketShared* mapped, uint16_t* agInput,
                           void const* input, void* output, size_t count,
                           int rank, uint64_t baseGeneration,
                           uint64_t previousSlot0, uint64_t previousSlot1,
                           cudaStream_t stream) {
  uint16_t const* source = static_cast<uint16_t const*>(input);
  uint16_t* destination = static_cast<uint16_t*>(output);
  void* args[] = {&mapped,         &agInput,       &source,
                  &destination,    &count,         &rank,
                  &baseGeneration, &previousSlot0, &previousSlot1};
  int blocks =
      count <= 1024 ? UCCL_LITE_MP_AG_SMALL_BLOCKS : UCCL_LITE_MP_AG_BLOCKS;
  return cudaLaunchCooperativeKernel(reinterpret_cast<void*>(allGatherPacket),
                                     dim3(blocks), dim3(256), args, 0, stream);
}

cudaError_t launchAgGemm128x4096Packet(
    PacketShared* mapped, uint16_t* agInput, void const* input, void* gathered,
    void const* weight, void* logits, int rank, uint64_t baseGeneration,
    uint64_t previousSlot0, cudaStream_t stream) {
  uint16_t const* source = static_cast<uint16_t const*>(input);
  uint16_t* gatheredRows = static_cast<uint16_t*>(gathered);
  uint16_t const* gateWeight = static_cast<uint16_t const*>(weight);
  uint16_t* gateLogits = static_cast<uint16_t*>(logits);
  void* args[] = {&mapped,      &agInput,       &source, &gatheredRows,
                  &gateWeight,  &gateLogits,    &rank,   &baseGeneration,
                  &previousSlot0};
  return cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(allGatherGemm128x4096Packet), dim3(32),
      dim3(256), args, 0, stream);
}

cudaError_t launchAgPacketPipelined(
    PacketShared* mapped, uint16_t* agInput, CrewState* crews,
    void const* input, void* output, size_t count, int rank,
    uint64_t baseGeneration, uint64_t previousSlot0, uint64_t previousSlot1,
    uint32_t initialPhase, cudaStream_t stream) {
  uint16_t const* source = static_cast<uint16_t const*>(input);
  uint16_t* destination = static_cast<uint16_t*>(output);
  void* args[] = {&mapped,        &agInput,       &crews,       &source,
                  &destination,   &count,         &rank,        &baseGeneration,
                  &previousSlot0, &previousSlot1, &initialPhase};
  return cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(allGatherPacketPipelined),
      dim3(UCCL_LITE_MP_AG_BLOCKS), dim3(256), args, 0, stream);
}

cudaError_t launchRsPacket(PacketShared* mapped, void const* input,
                           void* output, size_t shardCount, int rank,
                           uint64_t baseGeneration, uint64_t previousSlot0,
                           uint64_t previousSlot1, cudaStream_t stream) {
  uint16_t const* source = static_cast<uint16_t const*>(input);
  uint16_t* destination = static_cast<uint16_t*>(output);
  void* args[] = {&mapped, &source,         &destination,   &shardCount,
                  &rank,   &baseGeneration, &previousSlot0, &previousSlot1};
  return cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(reduceScatterPacket), dim3(32), dim3(256), args,
      0, stream);
}

cudaError_t launchRsSum2Packet(PacketShared* mapped, void const* input,
                               void const* auxiliary, void* output,
                               size_t shardCount, int rank,
                               uint64_t baseGeneration, uint64_t previousSlot0,
                               uint64_t previousSlot1, cudaStream_t stream) {
  uint16_t const* source = static_cast<uint16_t const*>(input);
  uint16_t const* second = static_cast<uint16_t const*>(auxiliary);
  uint16_t* destination = static_cast<uint16_t*>(output);
  void* args[] = {&mapped, &source,         &second,        &destination,
                  &shardCount, &rank,       &baseGeneration, &previousSlot0,
                  &previousSlot1};
  static int configuredBlocks = [] {
    char const* value = std::getenv("UCCL_LITE_RS_SUM2_BLOCKS");
    if (!value) return 32;
    int blocks = std::atoi(value);
    return blocks == 8 || blocks == 16 || blocks == 24 || blocks == 32
               ? blocks
               : 32;
  }();
  int blocks = shardCount == 4096 ? configuredBlocks : 32;
  return cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(reduceScatterSum2Packet), dim3(blocks), dim3(256),
      args, 0, stream);
}

cudaError_t launchRsReadySum2Packet(
    PacketShared* mapped, void const* input, void const* auxiliary,
    void* output, uint32_t const* ready, int rank, uint64_t baseGeneration,
    uint64_t previousSlot0, cudaStream_t stream) {
  uint16_t const* source = static_cast<uint16_t const*>(input);
  uint16_t const* second = static_cast<uint16_t const*>(auxiliary);
  uint16_t* destination = static_cast<uint16_t*>(output);
  void* args[] = {&mapped,      &source,         &second,      &destination,
                  &ready,       &rank,           &baseGeneration,
                  &previousSlot0};
  return cudaLaunchKernel(
      reinterpret_cast<void*>(reduceScatterReadySum2Packet), dim3(8),
      dim3(256), args, 0, stream);
}

cudaError_t launchRsPacketPipelined(
    PacketShared* mapped, CrewState* crews, void const* input, void* output,
    size_t shardCount, int rank, uint64_t baseGeneration,
    uint64_t previousSlot0, uint64_t previousSlot1, uint32_t initialPhase,
    cudaStream_t stream) {
  uint16_t const* source = static_cast<uint16_t const*>(input);
  uint16_t* destination = static_cast<uint16_t*>(output);
  void* args[] = {&mapped,        &crews,       &source,         &destination,
                  &shardCount,    &rank,        &baseGeneration, &previousSlot0,
                  &previousSlot1, &initialPhase};
  return cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(reduceScatterPipeline), dim3(32), dim3(256), args,
      0, stream);
}

cudaError_t launchRsPacketCpu(PacketShared* mapped, CrewState* crews,
                              void const* input, void* output,
                              size_t shardCount, int rank,
                              uint64_t baseGeneration, uint64_t previousSlot0,
                              uint64_t previousSlot1, uint32_t initialPhase,
                              uint32_t previousCpuMask, bool splitCredits,
                              bool retainOwn, cudaStream_t stream) {
  uint16_t const* source = static_cast<uint16_t const*>(input);
  uint16_t* destination = static_cast<uint16_t*>(output);
  void* args[] = {&mapped,         &crews,           &source,
                  &destination,    &shardCount,      &rank,
                  &baseGeneration, &previousSlot0,   &previousSlot1,
                  &initialPhase,   &previousCpuMask, &splitCredits,
                  &retainOwn};
  return cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(reduceScatterCpuPacket),
      dim3(shardCount <= kRsStripeElements ? UCCL_LITE_MP_RS_BLOCKS : 32),
      dim3(256), args, 0, stream);
}

cudaError_t launchRsPacketDestReady(
    PacketShared* mapped, CrewState* crews, void const* input, void* output,
    size_t shardCount, int rank, uint64_t baseGeneration,
    uint64_t previousSlot0, uint64_t previousSlot1, uint32_t initialPhase,
    cudaStream_t stream) {
  uint16_t const* source = static_cast<uint16_t const*>(input);
  uint16_t* destination = static_cast<uint16_t*>(output);
  void* args[] = {&mapped,        &crews,       &source,         &destination,
                  &shardCount,    &rank,        &baseGeneration, &previousSlot0,
                  &previousSlot1, &initialPhase};
  return cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(reduceScatterDestReady), dim3(24), dim3(256),
      args, 0, stream);
}

cudaError_t launchArPacketCpu(PacketShared* mapped, CrewState* crews,
                              void const* input, void* output, size_t count,
                              int rank, uint64_t baseGeneration,
                              uint64_t previousSlot0, uint64_t previousSlot1,
                              uint32_t initialPhase, bool replicateResults,
                              uint32_t previousCpuMask, bool splitCredits,
                              cudaStream_t stream) {
  uint16_t const* source = static_cast<uint16_t const*>(input);
  uint16_t* destination = static_cast<uint16_t*>(output);
  void* args[] = {
      &mapped,        &crews,        &source,           &destination,
      &count,         &rank,         &baseGeneration,   &previousSlot0,
      &previousSlot1, &initialPhase, &replicateResults, &previousCpuMask,
      &splitCredits};
  return cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(allReduceCpuPacket),
      dim3(count <= (1UL << 20) ? 16 : 48), dim3(256), args, 0, stream);
}

cudaError_t launchArPacketSmall(PacketShared* mapped, void const* input,
                                void* output, size_t count, int rank,
                                uint64_t generation, uint64_t previousSlot0,
                                cudaStream_t stream) {
  uint16_t const* source = static_cast<uint16_t const*>(input);
  uint16_t* destination = static_cast<uint16_t*>(output);
  void* args[] = {&mapped, &source,     &destination,  &count,
                  &rank,   &generation, &previousSlot0};
  return cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(allReduceSmallPacket),
      dim3(count >= 1024 && count <= 4096 ? 1 : 8), dim3(256), args, 0, stream);
}
}  // namespace uccl_lite::multiprocess
