#pragma once
// Included inside uccl_lite::multinode by async_host_gather.cuh, after its
// packet flag primitives. A cooperative launch keeps every producer and
// consumer resident; no CUDA operation is issued by the network worker.
#ifdef UCCL_LITE_HOST_UPLOAD_BLOCKS
constexpr int kHostPipelineUploadBlocks = UCCL_LITE_HOST_UPLOAD_BLOCKS;
#else
constexpr int kHostPipelineUploadBlocks = 4;
#endif
static_assert(kHostPipelineUploadBlocks > 0 && kHostPipelineUploadBlocks <= 32);
#ifdef UCCL_LITE_HOST_CONSUMER_BLOCKS
constexpr int kHostPipelineConsumerBlocks = UCCL_LITE_HOST_CONSUMER_BLOCKS;
#else
constexpr int kHostPipelineConsumerBlocks = 28;
#endif
static_assert(kHostPipelineConsumerBlocks > 0 &&
              kHostPipelineConsumerBlocks <= 64);
#ifdef UCCL_LITE_HOST_PUBLICATION_BLOCK
constexpr int kHostPipelineControlBlocks = 1;
#else
constexpr int kHostPipelineControlBlocks = 0;
#endif
struct alignas(128) HostPipelineSlot {
  unsigned uploaded, consumed;
#ifdef UCCL_LITE_HOST_GPU_GATES
  uint64_t ready[3];
#endif
};
struct HostPipelineView {
  HostPacketView flags;
  size_t capacity, payloadOffset, shardBytes, pieceBytes;
  uint64_t firstEpoch[SharedHostPackets::slots];
  int rank;
#ifdef FLASHINFER_ENABLE_PROFILER
  uint64_t* profiler_buffer;
#endif
};
#ifdef FLASHINFER_ENABLE_PROFILER
#define UCCL_LITE_PHASE_START(event) PROFILER_EVENT_START(trace, (event))
#define UCCL_LITE_PHASE_END(event) PROFILER_EVENT_END(trace, (event))
#else
#define UCCL_LITE_PHASE_START(event)
#define UCCL_LITE_PHASE_END(event)
#endif
__device__ inline char* pipelineRow(UCCL_LITE_VIEW_REF(HostPipelineView) v,
                                    int source, int slot, size_t length) {
  bool local = source / 4 == v.rank / 4;
  int group = local ? source % 4 / 2 : v.rank % 4 / 2;
  size_t offset = v.payloadOffset + size_t(slot) * 6 * v.capacity +
                  (local ? 0 : 2 * v.capacity) +
                  size_t(local ? source % 2 : source % 4) * length;
  return v.flags.control[group] + offset;
}
#ifdef UCCL_LITE_HOST_GPU_GATES
#if !defined(UCCL_LITE_HOST_WARP_PUBLISHER) || \
    !defined(UCCL_LITE_HOST_PUBLICATION_BLOCK)
#error GPU gates require the warp publisher
#endif
__device__ inline uint64_t pipelineGateLoad(uint64_t const* p) {
  uint64_t value;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];"
               : "=l"(value)
               : "l"(p)
               : "memory");
  return value;
}
__device__ inline void pipelineGateStore(uint64_t* p, uint64_t value) {
  asm volatile("st.release.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(value)
               : "memory");
}
__device__ inline bool pipelineHostReady(HostPacketView v, int rank, int slot,
                                         uint64_t epoch, int phase) {
  unsigned lane = threadIdx.x;
  int group = rank % 4 / 2, field = 13;
  bool active = lane == 31;
  if (phase == 0 && lane < 7) {
    active = true;
    field = lane < 6 ? 2 + lane : 12;
  } else if (phase == 1 && lane < 4) {
    active = true;
    group = lane / 2;
    field = lane % 2;
  } else if (phase == 2 && lane < 2) {
    active = true;
    field = 10 + lane;
  }
  uint64_t value = active ? packetLoad(packetFlag(v, group, slot, field)) : 0;
  if (__any_sync(0xffffffff, lane == 31 && value)) asm volatile("trap;");
  bool ready = !active || lane == 31 ||
               (phase == 0 ? value >= epoch - 1 : value == epoch);
  bool all = __all_sync(0xffffffff, ready);
  // Transfer all system acquires to the lane that publishes the GPU token.
  __syncwarp();
  return all;
}
#endif
__device__ inline void pipelineWait(UCCL_LITE_VIEW_REF(HostPipelineView) v,
                                    HostPipelineSlot* counters, int slot,
                                    uint64_t epoch, int phase) {
#ifdef UCCL_LITE_HOST_GPU_GATES
  if (!threadIdx.x)
    while (pipelineGateLoad(&counters[slot].ready[phase]) != epoch)
      __nanosleep(64);
#else
#ifdef UCCL_LITE_HOST_WARP_FLAGS
  if (threadIdx.x < 32)
#else
  if (!threadIdx.x)
#endif
    hostPacketWaitBody(v.flags, v.rank, slot, epoch, phase);
#endif
  __syncthreads();
}
__device__ inline void pipelineCopy(char* dst, char const* src, size_t bytes,
                                    size_t index, size_t stride) {
  if (dst == src) return;
  if (!((reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src) |
         bytes) &
        15)) {
    for (size_t i = index; i < bytes / 16; i += stride)
      reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<uint4 const*>(src)[i];
  } else {
    for (size_t i = index; i < bytes; i += stride) dst[i] = src[i];
  }
}
__device__ inline unsigned pipelineControlCount(unsigned* counter) {
  unsigned value = threadIdx.x == 0 ? atomicAdd(counter, 0) : 0;
#ifdef UCCL_LITE_HOST_WARP_PUBLISHER
  value = __shfl_sync(0xffffffff, value, 0);
#endif
  return value;
}
__global__ void hostPacketPersistent(UCCL_LITE_GRID_PARAM(HostPipelineView) v,
                                     HostPipelineSlot* counters,
                                     char const* input, char* output) {
#ifdef FLASHINFER_ENABLE_PROFILER
  using namespace flashinfer;
  struct {
    PROFILER_CLOSURE_PARAMS_DECL
  } trace{};
  if (v.profiler_buffer) {
    PROFILER_INIT(v, unused, trace, 0, 1, threadIdx.x == 0);
  }
#endif
  if (!blockIdx.x && threadIdx.x < SharedHostPackets::slots) {
    counters[threadIdx.x].uploaded = 0;
    counters[threadIdx.x].consumed = 0;
#ifdef UCCL_LITE_HOST_GPU_GATES
    for (int phase = 0; phase < 3; ++phase)
      counters[threadIdx.x].ready[phase] = 0;
#endif
  }
  cooperative_groups::this_grid().sync();
  size_t pieces = (v.shardBytes + v.pieceBytes - 1) / v.pieceBytes;
#ifdef UCCL_LITE_HOST_PUBLICATION_BLOCK
  if (!blockIdx.x) {
#ifdef UCCL_LITE_HOST_WARP_PUBLISHER
    if (threadIdx.x < 32) {
#else
    if (!threadIdx.x) {
#endif
      size_t next[SharedHostPackets::slots];
      bool published[SharedHostPackets::slots]{};
#ifdef UCCL_LITE_HOST_GPU_GATES
      bool ready[SharedHostPackets::slots][3]{};
#endif
      for (int slot = 0; slot < SharedHostPackets::slots; ++slot)
        next[slot] = slot;
      size_t complete = 0;
      unsigned poll = 0;
      while (complete < pieces) {
        if (!(++poll & 1023) &&
            packetLoad(packetFlag(v.flags, v.rank % 4 / 2, 0, 13)))
          asm volatile("trap;");
        for (int slot = 0; slot < SharedHostPackets::slots; ++slot) {
          if (next[slot] >= pieces) continue;
          uint64_t epoch =
              v.firstEpoch[slot] + next[slot] / SharedHostPackets::slots;
#ifdef UCCL_LITE_HOST_GPU_GATES
          if (!ready[slot][0] &&
              pipelineHostReady(v.flags, v.rank, slot, epoch, 0)) {
            if (!threadIdx.x)
              pipelineGateStore(&counters[slot].ready[0], epoch);
            __syncwarp();
            ready[slot][0] = true;
          }
#endif
          if (!published[slot] &&
              pipelineControlCount(&counters[slot].uploaded) ==
                  kHostPipelineUploadBlocks) {
            UCCL_LITE_PHASE_START(8);
            if (!threadIdx.x) {
              atomicExch(&counters[slot].uploaded, 0);
              packetStore(packetFlag(v.flags, v.rank % 4 / 2, slot, v.rank % 2),
                          epoch);
            }
#ifdef UCCL_LITE_HOST_WARP_PUBLISHER
            __syncwarp();
#endif
            published[slot] = true;
            UCCL_LITE_PHASE_END(8);
          }
#ifdef UCCL_LITE_HOST_GPU_GATES
          if (published[slot])
            for (int phase = 1; phase <= 2; ++phase)
              if (!ready[slot][phase] &&
                  pipelineHostReady(v.flags, v.rank, slot, epoch, phase)) {
                if (!threadIdx.x)
                  pipelineGateStore(&counters[slot].ready[phase], epoch);
                __syncwarp();
                ready[slot][phase] = true;
              }
#endif
          if (published[slot] &&
              pipelineControlCount(&counters[slot].consumed) ==
                  gridDim.x - kHostPipelineControlBlocks -
                      kHostPipelineUploadBlocks) {
            // Reset precedes the system-release ACK. Producers cannot reuse
            // this slot until the same ACK and network retirement are visible.
            UCCL_LITE_PHASE_START(9);
#ifdef UCCL_LITE_HOST_WARP_PUBLISHER
            if (!threadIdx.x) atomicExch(&counters[slot].consumed, 0);
            __syncwarp();
            // Every consumer has finished. The independent region ACKs need
            // no ordering among themselves; reset still precedes every ACK.
            if (threadIdx.x < 3) {
              int group = threadIdx.x < 2 ? int(threadIdx.x) : v.rank % 4 / 2;
              int field = threadIdx.x < 2 ? 2 + v.rank % 4 : 6 + v.rank % 2;
              packetStore(packetFlag(v.flags, group, slot, field), epoch);
            }
            __syncwarp();
#else
            atomicExch(&counters[slot].consumed, 0);
            for (int group = 0; group < 2; ++group)
              packetStore(packetFlag(v.flags, group, slot, 2 + v.rank % 4),
                          epoch);
            packetStore(
                packetFlag(v.flags, v.rank % 4 / 2, slot, 6 + v.rank % 2),
                epoch);
#endif
            next[slot] += SharedHostPackets::slots;
            published[slot] = false;
#ifdef UCCL_LITE_HOST_GPU_GATES
            for (int phase = 0; phase < 3; ++phase) ready[slot][phase] = false;
#endif
            ++complete;
            UCCL_LITE_PHASE_END(9);
          }
        }
        __nanosleep(64);
      }
    }
    return;
  }
#endif
  unsigned roleBlock = blockIdx.x - kHostPipelineControlBlocks;
  bool producer = roleBlock < kHostPipelineUploadBlocks;
  unsigned participants = producer ? kHostPipelineUploadBlocks
                                   : gridDim.x - kHostPipelineControlBlocks -
                                         kHostPipelineUploadBlocks;
  size_t index =
      size_t(producer ? roleBlock : roleBlock - kHostPipelineUploadBlocks) *
          blockDim.x +
      threadIdx.x;
  size_t stride = size_t(participants) * blockDim.x;
  for (size_t p = 0; p < pieces; ++p) {
    int slot = p % SharedHostPackets::slots;
    uint64_t epoch = v.firstEpoch[slot] + p / SharedHostPackets::slots;
    size_t offset = p * v.pieceBytes;
    size_t length = min(v.pieceBytes, v.shardBytes - offset);
    if (producer) {
      // Reuse waits for all local/remote readers AND outgoing retirement.
      UCCL_LITE_PHASE_START(0);
      pipelineWait(v, counters, slot, epoch, 0);
      UCCL_LITE_PHASE_END(0);
      UCCL_LITE_PHASE_START(1);
      pipelineCopy(pipelineRow(v, v.rank, slot, length), input + offset, length,
                   index, stride);
      UCCL_LITE_PHASE_END(1);
      UCCL_LITE_PHASE_START(2);
      // Every producer's host stores finish before its completion count.
      __threadfence_system();
      __syncthreads();
#ifdef UCCL_LITE_HOST_PUBLICATION_BLOCK
      if (!threadIdx.x) atomicAdd(&counters[slot].uploaded, 1);
#else
      if (!threadIdx.x && atomicInc(&counters[slot].uploaded,
                                    participants - 1) == participants - 1)
        packetStore(packetFlag(v.flags, v.rank % 4 / 2, slot, v.rank % 2),
                    epoch);
#endif
      UCCL_LITE_PHASE_END(2);
    } else {
      // Local and remote consumption can overlap other slots' uploads and
      // the CPU RDMA worker. Each block owns disjoint output elements.
      for (int phase = 1; phase <= 2; ++phase) {
        UCCL_LITE_PHASE_START(phase == 1 ? 3 : 5);
        pipelineWait(v, counters, slot, epoch, phase);
        UCCL_LITE_PHASE_END(phase == 1 ? 3 : 5);
        UCCL_LITE_PHASE_START(phase == 1 ? 4 : 6);
        int first = (phase == 1 ? v.rank / 4 : (v.rank / 4) ^ 1) * 4;
        for (int source = first; source < first + 4; ++source) {
          auto const* src = source == v.rank
                                ? input + offset
                                : pipelineRow(v, source, slot, length);
          pipelineCopy(output + size_t(source) * v.shardBytes + offset, src,
                       length, index, stride);
        }
        UCCL_LITE_PHASE_END(phase == 1 ? 4 : 6);
      }
      UCCL_LITE_PHASE_START(7);
#ifdef UCCL_LITE_HOST_DEVICE_ACK_FENCE
      // Consumers write only GPU output. All their host reads have completed
      // before this fence/barrier and the per-slot completion count. The ACK
      // still uses a system-release store, and producer payload stores still
      // require system fences. Caller completion is the CUDA kernel boundary.
      __threadfence();
#else
      __threadfence_system();
#endif
      __syncthreads();
#ifdef UCCL_LITE_HOST_PUBLICATION_BLOCK
      if (!threadIdx.x) atomicAdd(&counters[slot].consumed, 1);
#else
      if (!threadIdx.x && atomicInc(&counters[slot].consumed,
                                    participants - 1) == participants - 1) {
        for (int group = 0; group < 2; ++group)
          packetStore(packetFlag(v.flags, group, slot, 2 + v.rank % 4), epoch);
        packetStore(packetFlag(v.flags, v.rank % 4 / 2, slot, 6 + v.rank % 2),
                    epoch);
      }
#endif
      UCCL_LITE_PHASE_END(7);
    }
    // Completion counters reset before publication. A subsequent
    // generation cannot use the slot until these ACKs and network retirement.
    // No grid-wide barrier is used between packets: producer lookahead stays
    // bounded by the existing slot credits rather than by consumer scheduling.
  }
}

#undef UCCL_LITE_PHASE_START
#undef UCCL_LITE_PHASE_END
