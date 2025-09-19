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
// Note(MaoZiming, Yang): the warp_id here is used to tell which ring buffer to
// use. The total concurrent warps can be say 64 (= number of experts), while
// the number of ring buffers is small (say 6).
__device__ __forceinline__ void nvshmemi_ibgda_put_nbi_warp(
    uint64_t req_rptr, uint64_t req_lptr, size_t bytes, int dst_rank,
    int warp_id, int lane_id, int message_idx, uint64_t const* ring_addrs,
    int num_ring_addrs, bool is_combine) {
  // NOTE(MaoZiming): different from the nvshmemi_ibgda_put_nbi_warp in
  // ibgda_device.cuh, we don't do warp-cooperation.
  if (lane_id != 0) return;
  int safe_n = num_ring_addrs > 0 ? num_ring_addrs : 1;
  int ring_idx = (warp_id >= 0 ? warp_id : 0) % safe_n;

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
      cmd.cmd = (static_cast<uint64_t>(warp_id + 1) << 32) |
                (message_idx & 0xFFFFFFFF);  // NOTE(MaoZiming): Use warp_id + 1
                                             // to avoid 0 as a valid command.
      cmd.req_rptr = rptr_val;
      cmd.req_lptr = lptr_val;
      cmd.bytes = bytes_val;
      cmd.dst_rank = dst_rank;
      cmd.warp_id = warp_id;
      cmd.lane_id = lane_id;
      cmd.message_idx = message_idx;
      cmd.is_combine = is_combine;
      rb->atomic_set_and_commit(cmd, &slot);
      break;
    }
    if ((clock64() - last_print) > kPrintCycleInterval) {
      if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf(
            "[dispatch] stuck waiting, inflight=%ld (cur_head=%lu "
            "cur_tail=%lu)\n",
            (long)inflight, (unsigned long)cur_head, (unsigned long)cur_tail);
      }
      last_print = clock64();
    }
  }
}

// TODO(MaoZiming): Fix. This should be a non-fetch add operation. This could be
// implemented with CPU proxy.
__device__ __forceinline__ void nvshmemi_ibgda_amo_nonfetch_add(
    uint64_t rptr, int const& value, int dst_rank, int qp_id, int warp_id,
    bool is_local_copy = false, uint64_t const* ring_addrs = nullptr,
    int num_ring_addrs = 0, bool is_combine = true) {
  if (is_local_copy) {
    atomicAdd(reinterpret_cast<int*>(rptr), value);
  } else {
    int safe_n = num_ring_addrs > 0 ? num_ring_addrs : 1;
    int ring_idx = (warp_id >= 0 ? warp_id : 0) % safe_n;

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
        cmd.warp_id = warp_id;
        cmd.value = value;
        cmd.dst_rank = dst_rank;
        cmd.is_atomic = true;
        cmd.is_combine = is_combine;
        cmd.req_rptr = rptr;
        rb->atomic_set_and_commit(cmd, &slot);
        break;
      } else {
        auto now = clock64();
        if (now - last_print > kPrintCycleInterval) {
          uint64_t tail_cmd = rb->buf[cur_tail & rb->mask()].cmd;
          printf(
              "[nvshmemi_ibgda_amo_nonfetch_add] %p waiting ring_idx: %d, "
              "cur_head: "
              "%llu, cur_tail: %llu, inflight: %llu, tail_cmd: %llu\n",
              rb, ring_idx, cur_head, cur_tail, inflight, tail_cmd);
          last_print = now;
        }
      }
    }
  }
}

// GPU IPC handle support - replacement for nvshmemi_get_p2p_ptr
// This function will be used to get P2P pointers for intra-node communication
// The actual IPC handles will be managed by the Buffer class in uccl_ep.cc
__device__ __forceinline__ uint64_t get_ipc_p2p_ptr(uint64_t const& local_ptr,
                                                    void** ipc_base_ptrs,
                                                    int src_rank, int dst_rank,
                                                    int ranks_per_node,
                                                    size_t buffer_size) {
  // If same rank, return local pointer directly
  if (src_rank == dst_rank) {
    return local_ptr;
  }

  // Check if both ranks are on the same node
  int src_node = src_rank / ranks_per_node;
  int dst_node = dst_rank / ranks_per_node;
  if (src_node != dst_node) {
    return 0;
  }

  int src_local_rank = src_rank % ranks_per_node;
  int dst_local_rank = dst_rank % ranks_per_node;

  if (ipc_base_ptrs == nullptr || ipc_base_ptrs[src_local_rank] == nullptr ||
      ipc_base_ptrs[dst_local_rank] == nullptr) {
    return 0;
  }

  size_t offset =
      reinterpret_cast<uintptr_t>(reinterpret_cast<void*>(local_ptr)) -
      reinterpret_cast<uintptr_t>(ipc_base_ptrs[src_local_rank]);

  // printf("src_local_rank: %d, dst_local_rank: %d, offset: %zu, local_ptr: %p,
  // base_ptr: %p\n", src_local_rank, dst_local_rank, offset, (void*)local_ptr,
  // ipc_base_ptrs[src_local_rank]);

  // Return the remote pointer as uint64_t
  return reinterpret_cast<uint64_t>(ipc_base_ptrs[dst_local_rank]) + offset;
}

}  // namespace uccl