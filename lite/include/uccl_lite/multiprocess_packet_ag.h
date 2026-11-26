#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

#ifndef UCCL_LITE_MP_PACKET_BYTES
#define UCCL_LITE_MP_PACKET_BYTES (1UL << 20)
#endif

namespace uccl_lite::multiprocess {
inline constexpr int kPacketRanks = 8;
inline constexpr size_t kAgPacketBytes = UCCL_LITE_MP_PACKET_BYTES;
inline constexpr size_t kAgPacketElements = kAgPacketBytes / sizeof(uint16_t);
inline constexpr size_t kTransportPacketBytes = UCCL_LITE_MP_PACKET_BYTES;
inline constexpr size_t kRsStripeElements =
    kTransportPacketBytes / sizeof(uint16_t) / kPacketRanks;
static_assert(kAgPacketBytes == kTransportPacketBytes);
static_assert(kTransportPacketBytes == (1UL << 20) ||
              kTransportPacketBytes == (2UL << 20));

struct alignas(128) PacketGeneration {
  uint64_t value = 0;
};

struct alignas(128) CrewState {
  uint32_t arrivals = 0;
  uint32_t generation = 0;
  uint64_t ready = 0;
  uint64_t sourceReady[kPacketRanks]{};
};
static_assert(sizeof(CrewState) == 128);

struct alignas(4096) PacketShared {
  PacketGeneration uploaded[2][kPacketRanks];
  PacketGeneration uploadedDest[2][kPacketRanks][kPacketRanks];
  PacketGeneration inputReadDone[2][kPacketRanks];
  PacketGeneration localReady[2][kPacketRanks];
  PacketGeneration partialReadDone[2][kPacketRanks];
  PacketGeneration routed[2][kPacketRanks];
  PacketGeneration consumed[2][kPacketRanks];
  alignas(4096)
      uint16_t input[2][kPacketRanks][kTransportPacketBytes / sizeof(uint16_t)];
  alignas(4096) uint16_t result[2][2][kPacketRanks][kRsStripeElements];
  alignas(4096) uint16_t partial[2][kPacketRanks][2][kRsStripeElements];
};

cudaError_t launchAgPacket(PacketShared* mapped, uint16_t* agInput,
                           void const* input, void* output, size_t count,
                           int rank, uint64_t baseGeneration,
                           uint64_t previousSlot0, uint64_t previousSlot1,
                           cudaStream_t stream);
// Small BF16 AG+GEMM: each rank contributes one 4096-element row; every rank
// receives eight rows and computes A[8,4096] x B[128,4096]^T -> C[8,128].
cudaError_t launchAgGemm128x4096Packet(
    PacketShared* mapped, uint16_t* agInput, void const* input, void* gathered,
    void const* weight, void* logits, int rank, uint64_t baseGeneration,
    uint64_t previousSlot0, cudaStream_t stream);
cudaError_t launchAgPacketPipelined(PacketShared* mapped, uint16_t* agInput,
                                    CrewState* crews, void const* input,
                                    void* output, size_t count, int rank,
                                    uint64_t baseGeneration,
                                    uint64_t previousSlot0,
                                    uint64_t previousSlot1,
                                    uint32_t initialPhase, cudaStream_t stream);
cudaError_t launchRsPacket(PacketShared* mapped, void const* input,
                           void* output, size_t shardCount, int rank,
                           uint64_t baseGeneration, uint64_t previousSlot0,
                           uint64_t previousSlot1, cudaStream_t stream);
// Add two BF16 GEMM/epilogue outputs with BF16 rounding while uploading the
// ReduceScatter packet. The local shard is retained on its GPU and added with
// the same BF16 rounding during reduction.
cudaError_t launchRsSum2Packet(PacketShared* mapped, void const* input,
                               void const* auxiliary, void* output,
                               size_t shardCount, int rank,
                               uint64_t baseGeneration, uint64_t previousSlot0,
                               uint64_t previousSlot1, cudaStream_t stream);
// Dedicated-stream, one-row-per-rank RS. A local producer publishes one
// system-scope readiness flag per 4096-element destination row.
cudaError_t launchRsReadySum2Packet(
    PacketShared* mapped, void const* input, void const* auxiliary,
    void* output, uint32_t const* ready, int rank, uint64_t baseGeneration,
    uint64_t previousSlot0, cudaStream_t stream);
cudaError_t launchRsPacketPipelined(PacketShared* mapped, CrewState* crews,
                                    void const* input, void* output,
                                    size_t shardCount, int rank,
                                    uint64_t baseGeneration,
                                    uint64_t previousSlot0,
                                    uint64_t previousSlot1,
                                    uint32_t initialPhase, cudaStream_t stream);
cudaError_t launchRsPacketCpu(PacketShared* mapped, CrewState* crews,
                              void const* input, void* output,
                              size_t shardCount, int rank,
                              uint64_t baseGeneration, uint64_t previousSlot0,
                              uint64_t previousSlot1, uint32_t initialPhase,
                              uint32_t previousCpuMask, bool splitCredits,
                              bool retainOwn, cudaStream_t stream);
cudaError_t launchRsPacketDestReady(PacketShared* mapped, CrewState* crews,
                                    void const* input, void* output,
                                    size_t shardCount, int rank,
                                    uint64_t baseGeneration,
                                    uint64_t previousSlot0,
                                    uint64_t previousSlot1,
                                    uint32_t initialPhase, cudaStream_t stream);
cudaError_t launchArPacketCpu(PacketShared* mapped, CrewState* crews,
                              void const* input, void* output, size_t count,
                              int rank, uint64_t baseGeneration,
                              uint64_t previousSlot0, uint64_t previousSlot1,
                              uint32_t initialPhase, bool replicateResults,
                              uint32_t previousCpuMask, bool splitCredits,
                              cudaStream_t stream);
cudaError_t launchArPacketSmall(PacketShared* mapped, void const* input,
                                void* output, size_t count, int rank,
                                uint64_t generation, uint64_t previousSlot0,
                                cudaStream_t stream);
}  // namespace uccl_lite::multiprocess
