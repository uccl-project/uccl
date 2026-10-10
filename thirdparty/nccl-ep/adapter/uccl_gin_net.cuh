#pragma once

// NCCL-EP HT subset: one registered window, rail team peers, indexed additive
// signals and thread/full-warp cooperation. Host setup must register gin_base_ptr
// and provide the same UCCL signal storage to senders and receivers.
#include <cuda_runtime.h>
#include <climits>
#include <cstdint>
#include <nccl_device.h>
#include <type_traits>

#include "uccl_gin/uccl_gin.cuh"

namespace nccl_ep_adapter {

struct UcclGinNet {
  uccl_gin::UCCLGin gin;
  int lane_hint;

  __device__ __forceinline__
  UcclGinNet(uccl_gin::UCCLGinResources const& res, int lane)
      : gin(res), lane_hint(lane) {}

  __device__ __forceinline__ void put(
      ncclTeam, int peer, ncclWindow_t, size_t dst_off, ncclWindow_t,
      size_t src_off, size_t bytes, ncclGin_None = {}, ncclGin_None = {},
      ncclCoopThread = {}, ncclGin_None = {},
      cuda::thread_scope given_release = cuda::thread_scope_thread,
      cuda::thread_scope required_release = cuda::thread_scope_device,
      uint32_t flags = ncclGinOptFlagsDefault,
      ncclGin_SegmentDevice = {}) const {
    if (bytes > INT_MAX || given_release != cuda::thread_scope_thread ||
        required_release != cuda::thread_scope_device ||
        (flags != ncclGinOptFlagsDefault &&
         flags != ncclGinOptFlagsAggregateRequests)) UCCL_GIN_TRAP();
    void* recv = reinterpret_cast<void*>(gin.res.window_base + dst_off);
    void* send = reinterpret_cast<void*>(gin.res.window_base + src_off);
    gin.put<ncclTeamTagRail>(recv, send, static_cast<int>(bytes),
                             rail_peer(peer), lane_hint);
  }

  template <typename Coop = ncclCoopThread>
  __device__ __forceinline__ void signal(
      ncclTeam, int peer, ncclGin_SignalAdd action, Coop coop = {},
      ncclGin_None = {},
      cuda::thread_scope given_release = cuda::thread_scope_thread,
      cuda::thread_scope required_release = cuda::thread_scope_device,
      uint32_t flags = ncclGinOptFlagsDefault) const {
    validate_coop<Coop>();
    if (action.value >= kMaxSendAtomicValue ||
        given_release != cuda::thread_scope_thread ||
        (required_release != cuda::thread_scope_thread &&
         required_release != cuda::thread_scope_device) ||
        flags != ncclGinOptFlagsDefault) UCCL_GIN_TRAP();
    coop.sync();
    if (coop.thread_rank() == 0) {
      gin.red_add_rel<ncclTeamTagRail>(
          signal_slot(action.signal), static_cast<int>(action.value),
          rail_peer(peer), lane_hint);
    }
    coop.sync();
  }

  template <typename Coop>
  __device__ __forceinline__ void waitSignal(
      Coop coop, ncclGinSignal_t id, uint64_t expected, int bits = 64,
      cuda::memory_order order = cuda::memory_order_acquire) const {
    validate_coop<Coop>();
    if (bits != 64 || order != cuda::memory_order_acquire) UCCL_GIN_TRAP();
    coop.sync();
    if (coop.thread_rank() == 0) {
      while (!nccl::utility::rollingLessEq(expected, readSignal(id), bits))
        __nanosleep(64);
    }
    coop.sync();
  }

  __device__ __forceinline__ uint64_t readSignal(
      ncclGinSignal_t id, int bits = 64,
      cuda::memory_order order = cuda::memory_order_acquire) const {
    if (bits != 64 || order != cuda::memory_order_acquire) UCCL_GIN_TRAP();
    return static_cast<uint64_t>(
        mscclpp::atomicLoad<int64_t, mscclpp::scopeSystem>(
            signal_slot(id), mscclpp::memoryOrderAcquire));
  }

  template <typename Coop>
  __device__ __forceinline__ void flush(
      Coop coop, cuda::memory_order order = cuda::memory_order_acquire) const {
    if (order != cuda::memory_order_acquire) UCCL_GIN_TRAP();
    gin.flush(coop);
  }

 private:
  template <typename Coop>
  __device__ static constexpr void validate_coop() {
    static_assert(std::is_same_v<Coop, ncclCoopThread> ||
                      std::is_same_v<Coop, ncclCoopWarp>,
                  "UCCL-GIN: signal cooperation supports Thread/Warp only");
  }

  __device__ __forceinline__ int rail_peer(int node) const {
    if (node < 0 || node >= gin.res.num_scaleout_ranks) UCCL_GIN_TRAP();
    return node * gin.res.num_scaleup_ranks + gin.res.scaleup_rank;
  }

  __device__ __forceinline__ int64_t* signal_slot(ncclGinSignal_t id) const {
    if (id > uccl_gin::kAtomicOffMask / sizeof(int64_t)) UCCL_GIN_TRAP();
    // Ordered ATOMIC uses atomic_offset=1 as its opcode flag, so byte offset 0
    // is valid here. Only WRITE piggyback reserves counter slot 0.
    return reinterpret_cast<int64_t*>(gin.res.atomic_tail_base +
                                      uint64_t{id} * sizeof(int64_t));
  }
};

}  // namespace nccl_ep_adapter
