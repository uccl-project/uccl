#pragma once
#include "ring_buffer.cuh"
#include <cstddef>
#include <cstdint>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
// Strip CUDA attrs when not compiling with nvcc
#ifndef __device__
#define __device__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#endif

namespace uccl {

// template <bool kAlwaysDoPostSend = false>
// Note(MaoZiming): the qp_id here is actually the sm_id, which is used to tell
// which ring buffer to use. However, we still have an issue: the total SMs can
// be say 64 (= number of experts), but the number of ring buffers is small (say
// 6).
__device__ __forceinline__ void nvshmemi_ibgda_put_nbi_warp(
    uint64_t req_rptr, uint64_t req_lptr, size_t bytes, int dst_rank, int sm_id,
    int lane_id, int message_idx, uint64_t const* ring_addrs,
    int num_ring_addrs) {
  // NOTE(MaoZiming): different from the nvshmemi_ibgda_put_nbi_warp in
  // ibgda_device.cuh, we don't do warp-cooperation.
  if (lane_id != 0) return;
  int safe_n = num_ring_addrs > 0 ? num_ring_addrs : 1;
  int ring_idx = (sm_id >= 0 ? sm_id : 0) % safe_n;

  unsigned long long rptr_val = static_cast<unsigned long long>(req_rptr);
  unsigned long long lptr_val = static_cast<unsigned long long>(req_lptr);
  unsigned long long bytes_val = static_cast<unsigned long long>(bytes);

  auto* rb = reinterpret_cast<DeviceToHostCmdBuffer*>(
      static_cast<uintptr_t>(ring_addrs[ring_idx]));

  uint64_t cur_head = rb->head;
  uint64_t cur_tail = rb->volatile_tail();
  uint64_t inflight = cur_head - cur_tail;

  // NOTE(MaoZiming): Spins until there is a free slot in the ring buffer.
  auto last_print = clock64();
  while (true) {
    // NOTE(MaoZiming): update the view.
    cur_head = rb->head;
    cur_tail = rb->volatile_tail();
    inflight = cur_head - cur_tail;
    if (inflight < kMaxInflight) {
      uint64_t slot = cur_head;
      TransferCmd cmd{};
      // TODO(MaoZiming): Check fields here.
      // NOTE(MaoZiming): cmd is needed for proxy to process the command.
      cmd.cmd = (static_cast<uint64_t>(sm_id + 1) << 32) |
                (message_idx & 0xFFFFFFFF);  // NOTE(MaoZiming): Use sm_id + 1
                                             // to avoid 0 as a valid command.
      cmd.req_rptr = rptr_val;
      cmd.req_lptr = lptr_val;
      cmd.bytes = bytes_val;
      cmd.dst_rank = dst_rank;
      cmd.sm_id = sm_id;
      cmd.lane_id = lane_id;
      cmd.message_idx = message_idx;
      rb->atomic_set_and_commit(cmd, &slot);
      break;
    }
  }
}

// NOTE(MaoZiming): Remove this. We don't need nvshmem and ibgda.
#ifdef false
__device__ __forceinline__ nvshmemi_ibgda_device_state_t* ibgda_get_state() {
  return &nvshmemi_ibgda_device_state_d;
}

__device__ nvshmemi_ibgda_device_state_t nvshmemi_ibgda_device_state_d;

struct nvshmemi_ibgda_device_state_t {
  int _unused{0};
  uint32_t num_rc_per_pe{0};
};
#endif

// TODO(MaoZiming): Fix. This should be a non-fetch add operation. This could be
// implemented with CPU proxy.
__device__ __forceinline__ void nvshmemi_ibgda_amo_nonfetch_add(
    void* rptr, int const& value, int pe, int qp_id, int sm_id,
    bool is_local_copy = false, uint64_t const* ring_addrs = nullptr,
    int num_ring_addrs = 0) {
  (void)rptr;
  (void)value;
  (void)is_local_copy;
  if (is_local_copy) {
    atomicAdd(reinterpret_cast<int*>(rptr), value);
  } else {
    int safe_n = num_ring_addrs > 0 ? num_ring_addrs : 1;
    int ring_idx = (sm_id >= 0 ? sm_id : 0) % safe_n;

    auto* rb = reinterpret_cast<DeviceToHostCmdBuffer*>(
        static_cast<uintptr_t>(ring_addrs[ring_idx]));
    uint64_t cur_head = rb->head;
    uint64_t cur_tail = rb->volatile_tail();
    uint64_t inflight = cur_head - cur_tail;
    auto last_print = clock64();
    while (true) {
      // NOTE(MaoZiming): update the view.
      cur_head = rb->head;
      cur_tail = rb->volatile_tail();
      inflight = cur_head - cur_tail;
      if (inflight < kMaxInflight) {
        uint64_t slot = cur_head;
        TransferCmd cmd{};
        // TODO(MaoZiming): Check fields here.
        // NOTE(MaoZiming): cmd is needed for proxy to process the command.
        cmd.cmd = 1;  // to avoid 0 as a valid command.
        cmd.sm_id = sm_id;
        cmd.value = value;
        cmd.req_rptr = reinterpret_cast<uint64_t>(rptr);
        rb->atomic_set_and_commit(cmd, &slot);
        break;
      } else {
        auto now = clock64();
        if (now - last_print > 10 * 1e9) {
          uint64_t tail_cmd = rb->buf[cur_tail & rb->mask()].cmd;
          printf(
              "[nvshmemi_ibgda_amo_nonfetch_add] %p waiting sm_id: %d, "
              "cur_head: "
              "%llu, cur_tail: %llu, inflight: %llu, tail_cmd: %llu\n",
              rb, sm_id, cur_head, cur_tail, inflight, tail_cmd);
          last_print = now;
        }
      }
    }
  }
}

#ifdef false
__device__ __forceinline__ uint64_t nvshmemi_get_p2p_ptr(uint64_t const& ptr,
                                                         int const& rank,
                                                         int const& dst_rank) {
  return ptr;
}
#endif
}  // namespace uccl