#pragma once
//
// UcclGinNet — drop-in replacement for ncclGin at NCCL-EP HT call sites.
// Accepts the same extra parameters (ncclTeam world, ncclWindow_t win,
// remote/local action templates, coop, scopes, optFlags) but ignores
// everything except dst, offsets, bytes, signal id, and delta.
//
// The ONLY code change needed in hybrid_ep.cuh:
//   #ifdef NCCL_EP_USE_UCCL_GIN
//     UcclGinNet net(*uccl_resources, global_channel);
//   #else
//     ncclGin net(dcomms[comm_idx], ctx_idx, NCCL_GIN_RESOURCE_SHARING_CTA);
//     ncclTeam world = ncclTeamWorld(dcomms[comm_idx]);
//   #endif
//
// All net.put(world, ...) / net.signal(world, ...) / net.waitSignal(...)
// call sites remain unchanged under the macro guard that provides `world`
// as a dummy for the UCCL path.

#include <cuda_runtime.h>
#include <cstdint>
#include <nccl_device.h>

#include "uccl_gin/uccl_gin.cuh"

namespace nccl_ep_adapter {

struct UcclGinNet {
  uccl_gin::UCCLGin gin;
  int lane_hint;

  __device__ __forceinline__
  UcclGinNet(const uccl_gin::UCCLGinResources& res, int lane)
    : gin(res), lane_hint(lane) {}

  // -- put: NCCL-EP passes (world, dst, win, roff, win, soff, bytes,
  //    ncclGin_None, ncclGin_None, ncclCoopThread, ncclGin_None,
  //    cuda::thread_scope_thread, cuda::thread_scope_device,
  //    ncclGinOptFlagsAggregateRequests)
  // We ignore world/win/remote_action/local_action/coop/descriptor/scopes/flags.
  // Symmetric window: window_base + offset = same VA everywhere.
  template <typename... Args>
  __device__ __forceinline__
  void put(ncclTeam /*world*/, int dst,
           ncclWindow_t /*w1*/, size_t dst_off,
           ncclWindow_t /*w2*/, size_t src_off,
           size_t bytes, Args&&... /*ignored*/) {
    void* recv = reinterpret_cast<void*>(gin.res.window_base + dst_off);
    void* send = reinterpret_cast<void*>(gin.res.window_base + src_off);
    gin.put<ncclTeamTagRail>(recv, send, static_cast<int>(bytes), dst, lane_hint);
  }

  // -- signal: NCCL-EP passes (world, dst, signal_descriptor)
  // signal_descriptor is a struct with .indexedSignal.signalId and .opArg
  // For SignalAdd: .type = NCCL_GIN_SIGNAL_TYPE_INDEXED, .op = NCCL_GIN_SIGNAL_OP_ADD
  // We accept the raw descriptor but extract id/delta ourselves.
  struct SignalDescriptor {
    int type;
    struct { int signalId; } indexedSignal;
    int op;
    uint64_t opArg;
  };

  template <typename... Args>
  __device__ __forceinline__
  void signal(ncclTeam /*world*/, int dst, const SignalDescriptor& sd) {
    const uint64_t slot_addr = gin.res.atomic_tail_base +
        static_cast<uint64_t>(sd.indexedSignal.signalId) * sizeof(int64_t);
    gin.red_add_rel<ncclTeamTagRail>(
        reinterpret_cast<void*>(slot_addr),
        static_cast<int>(sd.opArg), dst, lane_hint);
  }

  // Accept the NCCL-EP signal call format: signal(world, dst, ncclGin_SignalAdd{id, delta})
  // which constructs an ncclGinSignalDescriptor with type=INDEXED, op=ADD, opArg=delta.
  // The NCCL GIN types aren't directly usable here, so we accept via template + reinterpret.
  // NCCL-EP calls: net.signal(world, dst, ncclGin_SignalAdd{id, delta})
  // where ncclGin_SignalAdd is an aggregate with .signal and .value fields.
  struct SignalAdd {
    ncclGinSignal_t signal;
    uint64_t value;
  };

  // Overload for the common NCCL-EP pattern: signal(world, dst, SignalAdd{id, delta})
  __device__ __forceinline__
  void signal(ncclTeam /*world*/, int dst, const SignalAdd& sa) {
    const uint64_t slot_addr = gin.res.atomic_tail_base +
        static_cast<uint64_t>(sa.signal) * sizeof(int64_t);
    gin.red_add_rel<ncclTeamTagRail>(
        reinterpret_cast<void*>(slot_addr),
        static_cast<int>(sa.value), dst, lane_hint);
  }

  // -- waitSignal: NCCL-EP passes (coop, id, expected)
  template <typename Coop>
  __device__ __forceinline__
  void waitSignal(Coop /*coop*/, int signal_id, uint64_t expected) const {
    volatile int64_t* slot = reinterpret_cast<volatile int64_t*>(
        gin.res.atomic_tail_base +
        static_cast<uint64_t>(signal_id) * sizeof(int64_t));
    while (__ldg(reinterpret_cast<const int64_t*>(slot)) <
           static_cast<int64_t>(expected)) {
      __nanosleep(64);
    }
  }

  // -- readSignal
  __device__ __forceinline__
  uint64_t readSignal(int signal_id) const {
    volatile int64_t* slot = reinterpret_cast<volatile int64_t*>(
        gin.res.atomic_tail_base +
        static_cast<uint64_t>(signal_id) * sizeof(int64_t));
    return static_cast<uint64_t>(__ldg(reinterpret_cast<const int64_t*>(slot)));
  }

  // -- flush: NCCL-EP passes (ncclCoopWarp(), cuda::memory_order_acquire)
  template <typename Coop>
  __device__ __forceinline__
  void flush(Coop /*coop*/, cuda::memory_order /*ord*/ =
             cuda::memory_order_acquire) const {
    gin.flush();
  }
};

}  // namespace nccl_ep_adapter
