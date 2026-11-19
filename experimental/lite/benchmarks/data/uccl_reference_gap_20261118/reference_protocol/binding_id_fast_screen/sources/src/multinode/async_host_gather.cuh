#pragma once
#include "host_packet_transport.h"
#ifdef FLASHINFER_ENABLE_PROFILER
#include <flashinfer/profiler.cuh>
#endif
#include <cstdlib>
#if defined(UCCL_LITE_HOST_COOPERATIVE_SM) || \
    defined(UCCL_LITE_HOST_PERSISTENT_SM)
#include <cooperative_groups.h>
#endif
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>
#include <sched.h>

namespace uccl_lite::multinode {
#ifdef UCCL_LITE_GRID_CONSTANT_PARAMS
#define UCCL_LITE_VIEW_REF(T) T const&
#define UCCL_LITE_GRID_PARAM(T) const __grid_constant__ T
#else
#define UCCL_LITE_VIEW_REF(T) T
#define UCCL_LITE_GRID_PARAM(T) T
#endif

#ifdef UCCL_LITE_HOST_COOPERATIVE_8M
constexpr size_t kHostCooperativePacket = 1UL << 20;
constexpr size_t kConfiguredCooperativeMaxShard = 1UL << 20;
#elif defined(UCCL_LITE_HOST_COOPERATIVE_16M)
constexpr size_t kHostCooperativePacket = 1UL << 20;
constexpr size_t kConfiguredCooperativeMaxShard = 2UL << 20;
#else
constexpr size_t kHostCooperativePacket = 512UL << 10;
constexpr size_t kConfiguredCooperativeMaxShard = 512UL << 10;
#endif
#ifdef UCCL_LITE_HOST_COOPERATIVE_MAX_SHARD
constexpr size_t kHostCooperativeMaxShard =
    UCCL_LITE_HOST_COOPERATIVE_MAX_SHARD;
#else
constexpr size_t kHostCooperativeMaxShard = kConfiguredCooperativeMaxShard;
#endif
static_assert(kHostCooperativeMaxShard >= (512UL << 10) &&
              kHostCooperativeMaxShard <= (2UL << 20));
struct HostPacketView {
  char* control[2];
};
__device__ inline uint64_t* packetFlag(UCCL_LITE_VIEW_REF(HostPacketView) v,
                                       int group, int slot, int field) {
  return reinterpret_cast<uint64_t*>(
      v.control[group] + (slot * SharedHostPackets::flagCount + field) * 64);
}
__device__ inline uint64_t packetLoad(uint64_t const* p) {
  uint64_t value;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
               : "=l"(value)
               : "l"(p)
               : "memory");
  return value;
}
__device__ inline void packetStore(uint64_t* p, uint64_t value) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(value)
               : "memory");
}
__device__ inline void hostPacketWaitBody(UCCL_LITE_VIEW_REF(HostPacketView) v,
                                          int rank, int slot, uint64_t epoch,
                                          int phase) {
  int group = rank % 4 / 2;
#ifdef UCCL_LITE_HOST_AGGREGATED_FLAGS
  int field = phase == 0   ? SharedHostPackets::reuseReady
              : phase == 1 ? SharedHostPackets::localReady
              : phase == 2 ? SharedHostPackets::remoteReady
                           : SharedHostPackets::allReady;
  for (;;) {
    if (packetLoad(packetFlag(v, group, slot, 13))) asm volatile("trap;");
    if (packetLoad(packetFlag(v, group, slot, field)) == epoch) return;
    __nanosleep(64);
  }
#elif defined(UCCL_LITE_HOST_COMBINED_FLAG_LOAD) && \
    defined(UCCL_LITE_HOST_WARP_FLAGS)
  unsigned lane = threadIdx.x;
  bool active = lane == 31;
  int flagGroup = group, field = 13;
  if (phase == 0 && lane < 7) {
    active = true;
    field = lane < 6 ? 2 + lane : 12;
  } else if (phase != 0 && phase != 2 && lane < 4) {
    active = true;
    flagGroup = lane / 2;
    field = lane % 2;
  } else if (phase != 0 && phase != 1 && lane >= 4 && lane < 6) {
    active = true;
    field = 10 + lane - 4;
  }
  for (;;) {
    // One predicated load instruction covers the independent counters and
    // abort flag. Every lane still participates in both warp votes.
    uint64_t value =
        active ? packetLoad(packetFlag(v, flagGroup, slot, field)) : 0;
    if (__any_sync(0xffffffff, lane == 31 && value)) asm volatile("trap;");
    bool ready = !active || lane == 31 ||
                 (phase == 0 ? value >= epoch - 1 : value == epoch);
    if (__all_sync(0xffffffff, ready)) return;
    __nanosleep(64);
  }
#elif defined(UCCL_LITE_HOST_WARP_FLAGS)
  // Independent system-memory counters are read concurrently by one warp.
  // Every lane participates in the vote, including lanes with no counter.
  unsigned lane = threadIdx.x;
  for (;;) {
    if (lane == 31 && packetLoad(packetFlag(v, group, slot, 13)))
      asm volatile("trap;");
    bool ready = true;
    if (phase == 0) {
      if (lane < 6)
        ready = packetLoad(packetFlag(v, group, slot, 2 + lane)) >= epoch - 1;
      if (lane == 6)
        ready = packetLoad(packetFlag(v, group, slot, 12)) >= epoch - 1;
    } else {
      if (phase != 2 && lane < 4)
        ready = packetLoad(packetFlag(v, lane / 2, slot, lane % 2)) == epoch;
      if (phase != 1 && lane >= 4 && lane < 6)
        ready = packetLoad(packetFlag(v, group, slot, 10 + lane - 4)) == epoch;
    }
    if (__all_sync(0xffffffff, ready)) return;
    __nanosleep(64);
  }
#else
  for (;;) {
    if (packetLoad(packetFlag(v, group, slot, 13))) asm volatile("trap;");
    bool ready = true;
    if (phase != 0) {
      if (phase != 2)
        for (int g = 0; g < 2; ++g)
          for (int r = 0; r < 2; ++r)
            ready &= packetLoad(packetFlag(v, g, slot, r)) == epoch;
      if (phase != 1)
        for (int r = 0; r < 2; ++r)
          ready &= packetLoad(packetFlag(v, group, slot, 10 + r)) == epoch;
    } else {
      for (int r = 0; r < 4; ++r)
        ready &= packetLoad(packetFlag(v, group, slot, 2 + r)) >= epoch - 1;
      for (int r = 0; r < 2; ++r)
        ready &= packetLoad(packetFlag(v, group, slot, 6 + r)) >= epoch - 1;
      ready &= packetLoad(packetFlag(v, group, slot, 12)) >= epoch - 1;
    }
    if (ready) return;
    __nanosleep(64);
  }
#endif
}
__global__ void hostPacketWait(UCCL_LITE_GRID_PARAM(HostPacketView) v, int rank,
                               int slot, uint64_t epoch, int phase) {
#ifndef UCCL_LITE_HOST_WARP_FLAGS
  if (threadIdx.x) return;
#endif
  hostPacketWaitBody(v, rank, slot, epoch, phase);
}
__global__ void hostPacketPublish(UCCL_LITE_GRID_PARAM(HostPacketView) v,
                                  int rank, int slot, uint64_t epoch,
                                  bool consume, int waitPhase = -1,
                                  uint64_t waitEpoch = 0) {
#ifdef UCCL_LITE_HOST_WARP_FLAGS
  int group = rank % 4 / 2;
  if (consume) {
    if (threadIdx.x < 2)
      packetStore(packetFlag(v, threadIdx.x, slot, 2 + rank % 4), epoch);
    if (threadIdx.x == 2)
      packetStore(packetFlag(v, group, slot, 6 + rank % 2), epoch);
  } else if (threadIdx.x == 0) {
    packetStore(packetFlag(v, group, slot, rank % 2), epoch);
  }
  __syncwarp();
#else
  if (threadIdx.x) return;
  int group = rank % 4 / 2;
  if (consume) {
    for (int g = 0; g < 2; ++g)
      packetStore(packetFlag(v, g, slot, 2 + rank % 4), epoch);
    packetStore(packetFlag(v, group, slot, 6 + rank % 2), epoch);
  } else
    packetStore(packetFlag(v, group, slot, rank % 2), epoch);
#endif
  if (waitPhase >= 0) hostPacketWaitBody(v, rank, slot, waitEpoch, waitPhase);
}

#ifdef UCCL_LITE_HOST_MEDIUM_SM
struct HostPacketSources {
  char const* row[8];
};
// Same rank-row multicast as the README SM medium path. Readiness is enforced
// by the preceding stream operation; the final ACK follows the whole grid.
__global__ void hostPacketMulticast(HostPacketSources sources, char* output,
                                    size_t shardBytes, size_t offset,
                                    size_t length, int first, int own) {
  int source = first + blockIdx.y;
  if (own >= first && source >= own) ++source;
  auto* dst = output + size_t(source) * shardBytes + offset;
  auto const* src = sources.row[source];
  size_t stride = size_t(gridDim.x) * blockDim.x;
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (!((reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src) |
         length) &
        15)) {
    for (size_t i = index; i < length / 16; i += stride)
      reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<uint4 const*>(src)[i];
  } else {
    // NCCL accepts offset float buffers: preserve the unaligned ABI cases.
    for (size_t i = index; i < length; i += stride) dst[i] = src[i];
  }
}
#endif

#ifdef UCCL_LITE_HOST_COOPERATIVE_SM
struct CooperativeHostPacket {
  char* row[8];
  size_t offset, length;
  uint64_t epoch;
  int slot;
};
struct CooperativeHostPackets {
  CooperativeHostPacket packet[2];
  int count;
};
__device__ inline void cooperativePacketWait(UCCL_LITE_VIEW_REF(HostPacketView)
                                                 view,
                                             int rank,
                                             CooperativeHostPacket const& p,
                                             int phase) {
#ifdef UCCL_LITE_HOST_WARP_FLAGS
  if (!blockIdx.x && threadIdx.x < 32)
#else
  if (!blockIdx.x && !threadIdx.x)
#endif
    hostPacketWaitBody(view, rank, p.slot, p.epoch, phase);
  cooperative_groups::this_grid().sync();
}
__device__ inline void cooperativePacketCopy(char* dst, char const* src,
                                             size_t bytes) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = size_t(gridDim.x) * blockDim.x;
  if (!((reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src) |
         bytes) &
        15)) {
    for (; i < bytes / 16; i += stride)
      reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<uint4 const*>(src)[i];
  } else {
    for (; i < bytes; i += stride) dst[i] = src[i];
  }
}
// Only the medium path (one or two packets) uses this resident cooperative
// grid. It uploads all input packets before consuming them. Every rank runs
// exactly one grid in its caller stream, avoiding cross-slot scheduling cycles.
__global__ void hostPacketCooperative(
    UCCL_LITE_GRID_PARAM(HostPacketView) view,
    UCCL_LITE_GRID_PARAM(CooperativeHostPackets) packets, int rank,
    char const* input, char* output, size_t shardBytes) {
  auto grid = cooperative_groups::this_grid();
  for (int i = 0; i < packets.count; ++i)
    cooperativePacketWait(view, rank, packets.packet[i], 0);
  for (int i = 0; i < packets.count; ++i) {
    auto const& p = packets.packet[i];
    cooperativePacketCopy(p.row[rank], input + p.offset, p.length);
  }
  // Each writer completes its host stores before the leader publishes upload.
  __threadfence_system();
  grid.sync();
  if (!blockIdx.x && !threadIdx.x)
    for (int i = 0; i < packets.count; ++i) {
      auto const& p = packets.packet[i];
      packetStore(packetFlag(view, rank % 4 / 2, p.slot, rank % 2), p.epoch);
    }
#ifdef UCCL_LITE_HOST_COOPERATIVE_LOCAL_FIRST
  for (int step = 0; step < 2 * packets.count; ++step) {
#ifdef UCCL_LITE_HOST_COOPERATIVE_LOCAL_FIRST
    int i = step % packets.count, phase = step / packets.count + 1;
#else
    int i = step / 2, phase = step % 2 + 1;
#endif
    auto const& p = packets.packet[i];
    cooperativePacketWait(view, rank, p, phase);
    int first = (phase == 1 ? rank / 4 : (rank / 4) ^ 1) * 4;
    for (int source = first; source < first + 4; ++source) {
      char const* src = source == rank ? input + p.offset : p.row[source];
      char* dst = output + size_t(source) * shardBytes + p.offset;
      if (dst != src) cooperativePacketCopy(dst, src, p.length);
    }
    if (phase == 2) {
      // Both phases for this packet have finished before publishing its ACK.
      // Local-first scheduling never releases a packet after local reads alone.
      grid.sync();
#ifdef UCCL_LITE_HOST_PARALLEL_ACK
      // All readers have passed the preceding grid barrier. These flags guard
      // disjoint regions, so their releases need no inter-flag ordering.
      if (!blockIdx.x && threadIdx.x < 3) {
        int group = threadIdx.x < 2 ? int(threadIdx.x) : rank % 4 / 2;
        int field = threadIdx.x < 2 ? 2 + rank % 4 : 6 + rank % 2;
        packetStore(packetFlag(view, group, p.slot, field), p.epoch);
      }
#else
      if (!blockIdx.x && !threadIdx.x) {
        for (int group = 0; group < 2; ++group)
          packetStore(packetFlag(view, group, p.slot, 2 + rank % 4), p.epoch);
        packetStore(packetFlag(view, rank % 4 / 2, p.slot, 6 + rank % 2),
                    p.epoch);
      }
#endif
#ifdef UCCL_LITE_HOST_COOPERATIVE_NO_FINAL_BARRIER
      if (i + 1 < packets.count)
#endif
        grid.sync();
    }
  }
#else
  for (int i = 0; i < packets.count; ++i) {
    auto const& p = packets.packet[i];
    for (int phase = 1; phase <= 2; ++phase) {
      cooperativePacketWait(view, rank, p, phase);
      int first = (phase == 1 ? rank / 4 : (rank / 4) ^ 1) * 4;
      for (int source = first; source < first + 4; ++source) {
        char const* src = source == rank ? input + p.offset : p.row[source];
        char* dst = output + size_t(source) * shardBytes + p.offset;
        if (dst != src) cooperativePacketCopy(dst, src, p.length);
      }
    }
    // All readers must finish before the leader releases either local region
    // or the remote rows. No successor can overwrite a packet before this ACK.
    grid.sync();
#ifdef UCCL_LITE_HOST_PARALLEL_ACK
    // All readers have passed the preceding grid barrier. These flags guard
    // disjoint regions, so their releases need no inter-flag ordering.
    if (!blockIdx.x && threadIdx.x < 3) {
      int group = threadIdx.x < 2 ? int(threadIdx.x) : rank % 4 / 2;
      int field = threadIdx.x < 2 ? 2 + rank % 4 : 6 + rank % 2;
      packetStore(packetFlag(view, group, p.slot, field), p.epoch);
    }
#else
    if (!blockIdx.x && !threadIdx.x) {
      for (int group = 0; group < 2; ++group)
        packetStore(packetFlag(view, group, p.slot, 2 + rank % 4), p.epoch);
      packetStore(packetFlag(view, rank % 4 / 2, p.slot, 6 + rank % 2),
                  p.epoch);
    }
#endif
#ifdef UCCL_LITE_HOST_COOPERATIVE_NO_FINAL_BARRIER
    if (i + 1 < packets.count)
#endif
      grid.sync();
  }
#endif
}
#endif

#ifdef UCCL_LITE_HOST_PERSISTENT_SM
#include "persistent_host_gather.cuh"
#endif

// Explicit preparation; one GPU per process. The worker performs only host
// atomics and IB operations. All CUDA submissions stay on the caller thread.
class AsyncHostGather {
  static constexpr int kSlots = SharedHostPackets::slots;
  struct Piece {
    int slot;
    size_t length;
    uint64_t epoch;
  };
  struct Work {
    std::vector<Piece> pieces;
  };
  int rank_;
  size_t chunk_;
  SharedHostPackets packets_;
  HostPacketTransport transport_;
  HostPacketView view_{};
  std::array<cudaStream_t, kSlots> lanes_{};
  std::array<cudaEvent_t, kSlots> joined_{};
  cudaEvent_t start_{}, done_{};
#ifdef UCCL_LITE_HOST_SHARED_EVENT
  // Borrowed from the owning network context; it outlives this runtime.
  cudaEvent_t completion_{};
#endif
  std::array<uint64_t, kSlots> epochs_{};
  bool pending_ = false, stop_ = false, active_ = false;
#ifdef UCCL_LITE_HOST_SPIN_PROGRESS
  std::atomic<size_t> queuedWork_{0};
  std::atomic<bool> stopRequested_{false};
  static void relaxCpu() {
#if defined(__x86_64__) || defined(__i386__)
    asm volatile("pause");
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
  }
#endif
#ifdef UCCL_LITE_HOST_STREAM_ID_ORDER
  unsigned long long lastStreamId_ = 0;
#endif
  std::mutex mutex_;
  std::condition_variable changed_;
  std::deque<Work> queue_;
  std::exception_ptr failure_;
  std::thread worker_;
#ifdef UCCL_LITE_HOST_COOPERATIVE_SM
  int cooperativeBlocks_ = 0;
#endif
#ifdef UCCL_LITE_HOST_PERSISTENT_SM
  HostPipelineSlot* pipelineSlots_ = nullptr;
  int pipelineBlocks_ = 0;
#ifdef UCCL_LITE_HOST_MEDIUM_CONSUMERS
  int pipelineMediumBlocks_ = 0;
#endif
#ifdef FLASHINFER_ENABLE_PROFILER
  uint64_t* pipelineTimeline_ = nullptr;
  size_t timelineEntries_ = 0, timelineShard_ = 0, timelinePiece_ = 0;
  bool timelineInplace_ = false;
  int timelineBlocks_ = 0;
  size_t timelineTarget_ = 0;
#endif
#endif
  static void check(cudaError_t s) {
    if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
  }
  cudaEvent_t completionEvent() const {
#ifdef UCCL_LITE_HOST_SHARED_EVENT
    return completion_ ? completion_ : done_;
#else
    return done_;
#endif
  }
  void recordCompletion(cudaStream_t stream
#ifdef UCCL_LITE_HOST_SHARED_EVENT
                        ,
                        cudaEvent_t external = nullptr
#endif
  ) {
#ifdef UCCL_LITE_HOST_SHARED_EVENT
    auto event = external ? external : done_;
#else
    auto event = done_;
#endif
    check(cudaEventRecord(event, stream));
#ifdef UCCL_LITE_HOST_SHARED_EVENT
    completion_ = event;
#endif
  }
  void progress(Work const& work) {
    std::array<size_t, kSlots> next{};
    for (int i = 0; i < kSlots; ++i) next[i] = i;
    std::array<int, kSlots> phase{};
#ifdef UCCL_LITE_HOST_AGGREGATED_FLAGS
    std::array<bool, kSlots> localReady{}, remoteReady{}, allReady{};
    auto publish = [&](int slot, int field, uint64_t epoch) {
      __atomic_store_n(packets_.flag(rank_ % 4 / 2, slot, field), epoch,
                       __ATOMIC_RELEASE);
    };
#endif
    size_t complete = 0;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (complete < work.pieces.size()) {
      bool advanced = false;
      for (int slot = 0; slot < kSlots; ++slot) {
        if (next[slot] >= work.pieces.size()) continue;
        auto const& p = work.pieces[next[slot]];
        if (phase[slot] == 0 && transport_.canReuse(slot, p.epoch)) {
          transport_.grant(slot, p.epoch);
#ifdef UCCL_LITE_HOST_AGGREGATED_FLAGS
          // Aggregate only after both reader completion and send retirement.
          publish(slot, SharedHostPackets::reuseReady, p.epoch);
#endif
          phase[slot] = 1;
          advanced = true;
        }
        if (phase[slot] == 1 && transport_.canSend(slot, p.epoch)) {
          transport_.send(slot, p.length, p.epoch);
#ifndef UCCL_LITE_HOST_LATE_FLUSH
          transport_.flush();
#endif
          phase[slot] = 2;
          advanced = true;
        }
#ifdef UCCL_LITE_HOST_AGGREGATED_FLAGS
        if (phase[slot] > 0) {
          if (!localReady[slot] && transport_.localReady(slot, p.epoch)) {
            publish(slot, SharedHostPackets::localReady, p.epoch);
            localReady[slot] = advanced = true;
          }
          if (!remoteReady[slot] && transport_.received(slot, p.epoch)) {
            publish(slot, SharedHostPackets::remoteReady, p.epoch);
            remoteReady[slot] = advanced = true;
          }
          if (!allReady[slot] && localReady[slot] && remoteReady[slot]) {
            publish(slot, SharedHostPackets::allReady, p.epoch);
            allReady[slot] = advanced = true;
          }
        }
#endif
#ifdef UCCL_LITE_HOST_LATE_FLUSH
        // At most kSlots packet slots can be outstanding. Reader ACKs let
        // networking overlap consumption; a slot is never reused until flush.
        if (phase[slot] == 2 && transport_.readersConsumed(slot, p.epoch) &&
            !transport_.canReuse(slot, p.epoch + 1))
          transport_.flush();
#endif
        if (phase[slot] == 2 && transport_.canReuse(slot, p.epoch + 1)) {
          next[slot] += kSlots;
          phase[slot] = 0;
#ifdef UCCL_LITE_HOST_AGGREGATED_FLAGS
          localReady[slot] = remoteReady[slot] = allReady[slot] = false;
#endif
          ++complete;
          advanced = true;
        }
      }
      if (advanced)
        deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
      else {
        if (std::chrono::steady_clock::now() > deadline)
          throw std::runtime_error("host gather progress timeout");
#ifdef UCCL_LITE_HOST_SPIN_PROGRESS
        relaxCpu();
#else
        std::this_thread::yield();
#endif
      }
    }
  }
  void worker() {
    try {
#ifdef UCCL_LITE_HOST_PIN_PROGRESS
      // Explicit testbed placement, matching the reference's routing CPUs.
      // This changes only the private network thread, never the caller mask.
      cpu_set_t affinity;
      CPU_ZERO(&affinity);
      int cpu = rank_ % 4 / 2;
      CPU_SET(cpu, &affinity);
      if (sched_setaffinity(0, sizeof(affinity), &affinity))
        throw std::runtime_error("host progress CPU affinity failed");
      std::fprintf(stderr, "UCCL Lite host progress: rank=%d cpu=%d\n", rank_,
                   cpu);
#endif
      for (;;) {
#ifdef UCCL_LITE_HOST_SPIN_PROGRESS
        // A queue doorbell avoids per-call OS wakeup while leaving queue,
        // failure and active-work state protected by the original mutex.
        while (!queuedWork_.load(std::memory_order_acquire) &&
               !stopRequested_.load(std::memory_order_acquire))
          relaxCpu();
#endif
        Work work;
        {
          std::unique_lock lock(mutex_);
#ifndef UCCL_LITE_HOST_SPIN_PROGRESS
          changed_.wait(lock, [&] { return stop_ || !queue_.empty(); });
#endif
          if (stop_ && queue_.empty()) return;
          work = std::move(queue_.front());
          queue_.pop_front();
#ifdef UCCL_LITE_HOST_SPIN_PROGRESS
          queuedWork_.fetch_sub(1, std::memory_order_release);
#endif
          active_ = true;
        }
        progress(work);
        {
          std::lock_guard lock(mutex_);
          active_ = false;
        }
        changed_.notify_all();
      }
    } catch (...) {
      for (int slot = 0; slot < kSlots; ++slot)
        __atomic_store_n(packets_.flag(rank_ % 4 / 2, slot, 13), uint64_t(1),
                         __ATOMIC_RELEASE);
      {
        std::lock_guard lock(mutex_);
        failure_ = std::current_exception();
        active_ = false;
      }
      changed_.notify_all();
    }
  }

 public:
  AsyncHostGather(mscclpp::Communicator& comm, int rank,
                  size_t chunk = 256UL << 10)
      : rank_(rank),
        chunk_(chunk),
#ifdef UCCL_LITE_HOST_COOPERATIVE_ONE_PACKET
        packets_(comm, rank, std::max(chunk, kHostCooperativePacket)),
#else
        packets_(comm, rank, chunk),
#endif
        transport_(comm, packets_, rank) {
    for (int g = 0; g < 2; ++g) view_.control[g] = packets_.mapped(g);
#ifdef UCCL_LITE_HOST_COOPERATIVE_SM
    int device, blocks;
    cudaDeviceProp property{};
    check(cudaGetDevice(&device));
    check(cudaGetDeviceProperties(&property, device));
    if (!property.cooperativeLaunch)
      throw std::runtime_error("cooperative host gather unsupported");
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, hostPacketCooperative, 256, 0));
#ifdef UCCL_LITE_HOST_COOPERATIVE_BLOCKS
    constexpr int cooperativeTarget = UCCL_LITE_HOST_COOPERATIVE_BLOCKS;
#else
    constexpr int cooperativeTarget = 32;
#endif
    static_assert(cooperativeTarget > 0 && cooperativeTarget <= 128);
    cooperativeBlocks_ =
        std::min(cooperativeTarget, blocks * property.multiProcessorCount);
    if (!cooperativeBlocks_)
      throw std::runtime_error("cooperative host gather occupancy");
#endif
#ifdef UCCL_LITE_HOST_PERSISTENT_SM
    int pipelineDevice, resident;
    cudaDeviceProp pipelineProperty{};
    check(cudaGetDevice(&pipelineDevice));
    check(cudaGetDeviceProperties(&pipelineProperty, pipelineDevice));
    if (!pipelineProperty.cooperativeLaunch)
      throw std::runtime_error(
          "persistent host gather requires cooperative launch");
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, hostPacketPersistent, 256, 0));
    pipelineBlocks_ =
        std::min(kHostPipelineUploadBlocks + kHostPipelineConsumerBlocks +
                     kHostPipelineControlBlocks,
                 resident * pipelineProperty.multiProcessorCount);
#ifdef UCCL_LITE_HOST_MEDIUM_CONSUMERS
    static_assert(UCCL_LITE_HOST_MEDIUM_CONSUMERS > 0 &&
                  UCCL_LITE_HOST_MEDIUM_CONSUMERS <= 64);
    pipelineMediumBlocks_ =
        std::min(kHostPipelineUploadBlocks + UCCL_LITE_HOST_MEDIUM_CONSUMERS +
                     kHostPipelineControlBlocks,
                 resident * pipelineProperty.multiProcessorCount);
#endif
    if (pipelineBlocks_ <=
        kHostPipelineControlBlocks + kHostPipelineUploadBlocks)
      throw std::runtime_error("persistent host gather occupancy");
    check(cudaMalloc(&pipelineSlots_, kSlots * sizeof(HostPipelineSlot)));
#ifdef FLASHINFER_ENABLE_PROFILER
    if (auto* target = std::getenv("UCCL_LITE_HOST_TIMELINE_BYTES"))
      timelineTarget_ = std::stoull(target);
    int timelineBlocks = pipelineBlocks_;
#ifdef UCCL_LITE_HOST_MEDIUM_CONSUMERS
    timelineBlocks = std::max(timelineBlocks, pipelineMediumBlocks_);
#endif
    timelineEntries_ =
        1 + timelineBlocks * 12 *
                ((size_t(128) << 20) / std::min(chunk_, size_t(256) << 10) + 1);
    check(cudaMalloc(&pipelineTimeline_, timelineEntries_ * sizeof(uint64_t)));
#endif
#endif
    check(cudaEventCreateWithFlags(&start_, cudaEventDisableTiming));
    check(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming));
    for (int i = 0; i < kSlots; ++i) {
      check(cudaStreamCreateWithFlags(&lanes_[i], cudaStreamNonBlocking));
      check(cudaEventCreateWithFlags(&joined_[i], cudaEventDisableTiming));
    }
    if (!(rank % 2)) worker_ = std::thread([this] { worker(); });
  }
  ~AsyncHostGather() {
    synchronize();
    {
      std::lock_guard lock(mutex_);
      stop_ = true;
#ifdef UCCL_LITE_HOST_SPIN_PROGRESS
      stopRequested_.store(true, std::memory_order_release);
#endif
    }
    changed_.notify_all();
    if (worker_.joinable()) worker_.join();
#ifdef UCCL_LITE_HOST_PERSISTENT_SM
#ifdef FLASHINFER_ENABLE_PROFILER
    if (timelineShard_ && std::getenv("UCCL_LITE_HOST_TIMELINE_DIR")) {
      std::vector<uint64_t> trace(timelineEntries_);
      check(cudaMemcpy(trace.data(), pipelineTimeline_, trace.size() * 8,
                       cudaMemcpyDeviceToHost));
      std::string prefix =
          std::string(std::getenv("UCCL_LITE_HOST_TIMELINE_DIR")) + "/rank_" +
          std::to_string(rank_);
      auto* file = std::fopen((prefix + ".bin").c_str(), "wb");
      if (!file ||
          std::fwrite(trace.data(), 8, trace.size(), file) != trace.size())
        throw std::runtime_error("host timeline write failed");
      std::fclose(file);
      file = std::fopen((prefix + ".json").c_str(), "w");
      if (!file) throw std::runtime_error("host timeline metadata failed");
      std::fprintf(file,
                   "{\"rank\":%d,\"shard_bytes\":%zu,\"piece_bytes\":%zu,"
                   "\"blocks\":%d,\"upload_blocks\":%d,\"control_blocks\":%d,"
                   "\"inplace\":%s}\n",
                   rank_, timelineShard_, timelinePiece_, timelineBlocks_,
                   kHostPipelineUploadBlocks, kHostPipelineControlBlocks,
                   timelineInplace_ ? "true" : "false");
      std::fclose(file);
    }
    cudaFree(pipelineTimeline_);
#endif
    cudaFree(pipelineSlots_);
#endif
    for (int i = 0; i < kSlots; ++i) {
      cudaEventDestroy(joined_[i]);
      cudaStreamDestroy(lanes_[i]);
    }
    cudaEventDestroy(done_);
    cudaEventDestroy(start_);
  }
  AsyncHostGather(AsyncHostGather const&) = delete;
  AsyncHostGather& operator=(AsyncHostGather const&) = delete;
  void synchronize(
#ifdef UCCL_LITE_HOST_SHARED_EVENT
      bool gpuComplete = false
#endif
  ) {
#ifdef UCCL_LITE_HOST_SHARED_EVENT
    // Only the owner may supply true after waiting its latest, ordered GPU
    // boundary. CPU transport work still drains below without any shortcut.
    if (pending_ && !gpuComplete)
      check(cudaEventSynchronize(completionEvent()));
#else
    if (pending_) check(cudaEventSynchronize(done_));
#endif
    if (worker_.joinable()) {
      std::unique_lock lock(mutex_);
      changed_.wait(lock,
                    [&] { return failure_ || (queue_.empty() && !active_); });
      if (failure_) std::rethrow_exception(failure_);
    }
  }
  void enqueue(void const* input, void* output, size_t shardBytes,
               cudaStream_t stream
#ifdef UCCL_LITE_HOST_SHARED_EVENT
               ,
               cudaEvent_t completion = nullptr
#endif
  ) {
    if (!input || !output || !shardBytes || shardBytes % 16 ||
        shardBytes > (1UL << 27))
      throw std::invalid_argument("host gather extent/pointers");
    cudaStreamCaptureStatus capture;
    check(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone)
      throw std::invalid_argument("host gather caller capture unsupported");
#ifdef UCCL_LITE_HOST_STREAM_ID_ORDER
    unsigned long long streamId;
    check(cudaStreamGetId(stream, &streamId));
    // Stream IDs distinguish recycled handles. Same-stream CUDA ordering is
    // sufficient; a different stream still waits at the exact prior boundary.
    if (pending_ && streamId != lastStreamId_)
      check(cudaStreamWaitEvent(stream, completionEvent(), 0));
#else
    if (pending_) check(cudaStreamWaitEvent(stream, completionEvent(), 0));
#endif
    size_t pieceBytes = shardBytes <= (4UL << 20)
                            ? std::min(chunk_, size_t(256) << 10)
                            : chunk_;
#ifdef UCCL_LITE_HOST_MEDIUM_PACKET_BYTES
    static_assert(UCCL_LITE_HOST_MEDIUM_PACKET_BYTES >= 16384 &&
                  UCCL_LITE_HOST_MEDIUM_PACKET_BYTES <= 262144);
    if (shardBytes <= (4UL << 20))
      pieceBytes = std::min(chunk_, size_t(UCCL_LITE_HOST_MEDIUM_PACKET_BYTES));
#endif
#ifdef UCCL_LITE_HOST_COOPERATIVE_ONE_PACKET
    // Only the cooperative medium path changes packet geometry. Bulk CE still
    // uses the configured chunk_ and the same three-slot runtime policy.
    if (shardBytes <= kHostCooperativeMaxShard)
      pieceBytes = std::min(shardBytes, kHostCooperativePacket);
#endif
    bool early = shardBytes <= (4UL << 20);
    Work work;
    for (size_t offset = 0, p = 0; offset < shardBytes;
         offset += pieceBytes, ++p) {
      int slot = p % kSlots;
      if (epochs_[slot] == UINT64_MAX)
        throw std::overflow_error("host packet epoch exhausted");
      work.pieces.push_back(
          {slot, std::min(pieceBytes, shardBytes - offset), ++epochs_[slot]});
    }
    if (!(rank_ % 2)) {
      std::lock_guard lock(mutex_);
      if (failure_) std::rethrow_exception(failure_);
      queue_.push_back(work);
#ifdef UCCL_LITE_HOST_SPIN_PROGRESS
      queuedWork_.fetch_add(1, std::memory_order_release);
#endif
      changed_.notify_all();
    }
    auto* out = static_cast<char*>(output);
    auto const* in = static_cast<char const*>(input);
#ifdef UCCL_LITE_HOST_COOPERATIVE_SM
    if (shardBytes <= kHostCooperativeMaxShard && work.pieces.size() <= 2) {
      CooperativeHostPackets packets{};
      packets.count = work.pieces.size();
      for (int i = 0; i < packets.count; ++i) {
        auto const& piece = work.pieces[i];
        auto& p = packets.packet[i];
        p.offset = size_t(i) * pieceBytes;
        p.length = piece.length;
        p.epoch = piece.epoch;
        p.slot = piece.slot;
        for (int r = 0; r < 8; ++r)
          p.row[r] = packets_.row(r, piece.slot, piece.length, true);
      }
      void* args[] = {&view_, &packets, &rank_, &in, &out, &shardBytes};
      check(cudaLaunchCooperativeKernel(
          reinterpret_cast<void*>(hostPacketCooperative), cooperativeBlocks_,
          256, args, 0, stream));
      recordCompletion(stream
#ifdef UCCL_LITE_HOST_SHARED_EVENT
                       ,
                       completion
#endif
      );
#ifdef UCCL_LITE_HOST_STREAM_ID_ORDER
      lastStreamId_ = streamId;
#endif
      pending_ = true;
      return;
    }
#endif
#ifdef UCCL_LITE_HOST_PERSISTENT_SM
    if (shardBytes > (512UL << 10)) {
      HostPipelineView view{};
      view.flags = view_;
      view.capacity = packets_.capacity();
      view.payloadOffset = packets_.payloadOffset(0);
      view.shardBytes = shardBytes;
      view.pieceBytes = pieceBytes;
      view.rank = rank_;
      int launchBlocks = pipelineBlocks_;
#ifdef UCCL_LITE_HOST_MEDIUM_CONSUMERS
      if (shardBytes <= (4UL << 20)) launchBlocks = pipelineMediumBlocks_;
#endif
#ifdef FLASHINFER_ENABLE_PROFILER
      if (!timelineTarget_ || shardBytes * 8 == timelineTarget_) {
        view.profiler_buffer = pipelineTimeline_;
        timelineBlocks_ = launchBlocks;
        timelineShard_ = shardBytes;
        timelinePiece_ = pieceBytes;
        timelineInplace_ = out + rank_ * shardBytes == in;
        check(cudaMemsetAsync(pipelineTimeline_, 0, timelineEntries_ * 8,
                              stream));
      }

#endif
      for (size_t i = 0; i < std::min(work.pieces.size(), size_t(kSlots)); ++i)
        view.firstEpoch[i] = work.pieces[i].epoch;
      void* args[] = {&view, &pipelineSlots_, &in, &out};
      check(cudaLaunchCooperativeKernel(
          reinterpret_cast<void*>(hostPacketPersistent), launchBlocks, 256,
          args, 0, stream));
      recordCompletion(stream
#ifdef UCCL_LITE_HOST_SHARED_EVENT
                       ,
                       completion
#endif
      );
#ifdef UCCL_LITE_HOST_STREAM_ID_ORDER
      lastStreamId_ = streamId;
#endif
      pending_ = true;
      return;
    }
#endif
    if (out + rank_ * shardBytes != in)
      check(cudaMemcpyAsync(out + rank_ * shardBytes, in, shardBytes,
                            cudaMemcpyDeviceToDevice, stream));
    check(cudaEventRecord(start_, stream));
    for (auto lane : lanes_) check(cudaStreamWaitEvent(lane, start_, 0));
    for (size_t p = 0, offset = 0; p < work.pieces.size();
         offset += pieceBytes, ++p) {
      auto const& piece = work.pieces[p];
      auto lane = lanes_[piece.slot];
#ifdef UCCL_LITE_HOST_FUSE_FLAGS
      if (p < kSlots)
#endif
        hostPacketWait<<<1, 32, 0, lane>>>(view_, rank_, piece.slot,
                                           piece.epoch, false);
      check(cudaGetLastError());
      check(cudaMemcpyAsync(packets_.row(rank_, piece.slot, piece.length),
                            in + offset, piece.length, cudaMemcpyDeviceToHost,
                            lane));
#ifdef UCCL_LITE_HOST_FUSE_FLAGS
      hostPacketPublish<<<1, 32, 0, lane>>>(view_, rank_, piece.slot,
                                            piece.epoch, false, early ? 1 : 3,
                                            piece.epoch);
#else
      hostPacketPublish<<<1, 32, 0, lane>>>(view_, rank_, piece.slot,
                                            piece.epoch, false);
#endif
      for (int phase = early ? 1 : 3; phase <= (early ? 2 : 3); ++phase) {
#ifdef UCCL_LITE_HOST_FUSE_FLAGS
        if (phase != (early ? 1 : 3))
#endif
          hostPacketWait<<<1, 32, 0, lane>>>(view_, rank_, piece.slot,
                                             piece.epoch, phase);
        check(cudaGetLastError());
        int first =
            phase == 3 ? 0 : (phase == 1 ? rank_ / 4 : (rank_ / 4) ^ 1) * 4;
        int last = phase == 3 ? 8 : first + 4;
#ifdef UCCL_LITE_HOST_MEDIUM_SM
        if (shardBytes <= (512UL << 10)) {
          HostPacketSources sources{};
          for (int r = first; r < last; ++r)
            if (r != rank_)
              sources.row[r] = packets_.row(r, piece.slot, piece.length, true);
          bool skipOwn = rank_ >= first && rank_ < last;
          int blocks = std::min<size_t>(128, (piece.length / 16 + 255) / 256);
          hostPacketMulticast<<<dim3(blocks, last - first - int(skipOwn)), 256,
                                0, lane>>>(sources, out, shardBytes, offset,
                                           piece.length, first,
                                           skipOwn ? rank_ : -1);
          check(cudaGetLastError());
          continue;
        }
#endif
#ifdef UCCL_LITE_HOST_BATCH_CONSUME
        // The readiness kernel covers every source in this phase. Its copies
        // write disjoint output rows, so no ordering within the batch is
        // needed.
        std::array<cudaMemcpy3DBatchOp, 4> copies{};
        size_t copyCount = 0;
#endif
        while (first < last) {
          if (first == rank_) {
            ++first;
            continue;
          }
          int end = first / 4 == rank_ / 4 ? (first / 2 + 1) * 2
                                           : (first / 4 + 1) * 4;
          if (rank_ > first && rank_ < end) end = rank_;
#ifdef UCCL_LITE_HOST_BATCH_CONSUME
          auto& copy = copies.at(copyCount++);
          copy.src.type = copy.dst.type = cudaMemcpyOperandTypePointer;
          copy.src.op.ptr.ptr = packets_.row(first, piece.slot, piece.length);
          copy.src.op.ptr.rowLength = piece.length;
          copy.dst.op.ptr.ptr = out + size_t(first) * shardBytes + offset;
          copy.dst.op.ptr.rowLength = shardBytes;
          copy.extent = make_cudaExtent(piece.length, end - first, 1);
          // Host rows are produced asynchronously. Never allow the batch to
          // access them during submission or ahead of the readiness kernel.
          copy.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
#else
          check(cudaMemcpy2DAsync(
              out + size_t(first) * shardBytes + offset, shardBytes,
              packets_.row(first, piece.slot, piece.length), piece.length,
              piece.length, end - first, cudaMemcpyHostToDevice, lane));
#endif
          first = end;
        }
#ifdef UCCL_LITE_HOST_BATCH_CONSUME
        size_t failed = SIZE_MAX;
        check(
            cudaMemcpy3DBatchAsync(copyCount, copies.data(), &failed, 0, lane));
#endif
      }
#ifdef UCCL_LITE_HOST_FUSE_FLAGS
      // Publish this packet's ACKs before waiting for this same lane's next
      // reuse. Only within-call successors are fused; the last ACK never waits
      // for an invocation which the caller has not submitted yet.
      hostPacketPublish<<<1, 32, 0, lane>>>(
          view_, rank_, piece.slot, piece.epoch, true,
          p + kSlots < work.pieces.size() ? 0 : -1, piece.epoch + 1);
#else
      hostPacketPublish<<<1, 32, 0, lane>>>(view_, rank_, piece.slot,
                                            piece.epoch, true);
#endif
      check(cudaGetLastError());
    }
    for (int i = 0; i < kSlots; ++i) {
      check(cudaEventRecord(joined_[i], lanes_[i]));
      check(cudaStreamWaitEvent(stream, joined_[i], 0));
    }
    recordCompletion(stream
#ifdef UCCL_LITE_HOST_SHARED_EVENT
                     ,
                     completion
#endif
    );
#ifdef UCCL_LITE_HOST_STREAM_ID_ORDER
    lastStreamId_ = streamId;
#endif
    pending_ = true;
  }
};
}  // namespace uccl_lite::multinode

#undef UCCL_LITE_VIEW_REF
#undef UCCL_LITE_GRID_PARAM
