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
// Note(MaoZiming, Yang): the expert_idx here is used to tell which ring buffer
// to use. The total concurrent warps can be say 64 (= number of experts), while
// the number of ring buffers is small (say 6).
__device__ __forceinline__ void nvshmemi_ibgda_put_nbi_warp(
    uint64_t req_rptr, uint64_t req_lptr, size_t bytes, int dst_rank,
    int expert_idx, int lane_id, int message_idx, uint64_t const* ring_addrs,
    int num_ring_addrs, bool is_combine, int low_latency_buffer_idx = 0,
    int barrier_id = -1, uint64_t atomic_offset = 0, uint64_t atomic_val = 0) {
  // NOTE(MaoZiming): different from the nvshmemi_ibgda_put_nbi_warp in
  // ibgda_device.cuh, we don't do warp-cooperation.
  if (lane_id != 0) return;
  int safe_n = num_ring_addrs > 0 ? num_ring_addrs : 1;
  int ring_idx = (expert_idx >= 0 ? expert_idx : 0) % safe_n;

  unsigned long long rptr_val = static_cast<unsigned long long>(req_rptr);
  unsigned long long lptr_val = static_cast<unsigned long long>(req_lptr);
  unsigned long long bytes_val = static_cast<unsigned long long>(bytes);

  auto* rb = reinterpret_cast<DeviceToHostCmdBuffer*>(
      static_cast<uintptr_t>(ring_addrs[ring_idx]));

  uint64_t cur_head = rb->head;
  uint64_t cur_tail = rb->volatile_tail();
  uint64_t inflight = cur_head - cur_tail;
  // printf("nvshmemi_ibgda_put_nbi_warp. dst_rank: %d\n", dst_rank);

  assert((atomic_offset == 0 && atomic_val == 0) || barrier_id != -1);

  if (low_latency_buffer_idx == -1) {
    /* Normal mode */
    expert_idx = 0;
    low_latency_buffer_idx = 0;
  }

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
      cmd.cmd_type = CmdType::WRITE;
      cmd.cmd =
          (static_cast<uint64_t>(expert_idx + 1) << 32) |
          (message_idx & 0xFFFFFFFF);  // NOTE(MaoZiming): Use expert_idx + 1
                                       // to avoid 0 as a valid command.
      cmd.req_rptr = rptr_val;
      cmd.req_lptr = lptr_val;
      cmd.bytes = bytes_val;
      cmd.dst_rank = dst_rank;
      cmd.expert_idx = expert_idx;
      cmd.lane_id = lane_id;
      cmd.message_idx = message_idx;
      cmd.is_combine = is_combine;
      cmd.low_latency_buffer_idx = low_latency_buffer_idx;
      cmd.barrier_id = barrier_id;
      cmd.atomic_offset = atomic_offset;
      cmd.atomic_val = atomic_val;
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
    uint64_t rptr, uint64_t atomic_base_addr, int const& value, int dst_rank,
    int warp_id, bool is_local_copy = false,
    uint64_t const* ring_addrs = nullptr, int num_ring_addrs = 0,
    bool is_combine = true, int low_latency_buffer_idx = 0,
    int barrier_id = -1) {
  if (is_local_copy) {
    atomicAdd(reinterpret_cast<unsigned long long*>(rptr),
              static_cast<unsigned long long>(value));
  } else {
    if (barrier_id != -1) {
      return;
    }
    rptr -= atomic_base_addr;
    int safe_n = num_ring_addrs > 0 ? num_ring_addrs : 1;
    int ring_idx = (warp_id >= 0 ? warp_id : 0) % safe_n;

    auto* rb = reinterpret_cast<DeviceToHostCmdBuffer*>(
        static_cast<uintptr_t>(ring_addrs[ring_idx]));
    uint64_t cur_head = rb->head;
    uint64_t cur_tail = rb->volatile_tail();
    uint64_t inflight = cur_head - cur_tail;
    auto last_print = clock64();

    if (low_latency_buffer_idx == -1) {
      /* Normal mode */
      low_latency_buffer_idx = 0;
    }

    while (true) {
      // NOTE(MaoZiming): update the view.
      cur_head = rb->head;
      cur_tail = rb->volatile_tail();
      inflight = cur_head - cur_tail;
      if (inflight < kMaxInflight) {
        uint64_t slot = cur_head;
        TransferCmd cmd{};
        cmd.cmd_type = CmdType::ATOMIC;
        // TODO(MaoZiming): Check fields here.
        // NOTE(MaoZiming): cmd is needed for proxy to process the command.
        cmd.cmd = 1;  // to avoid 0 as a valid command.
        cmd.warp_id = warp_id;
        cmd.value = value;
        cmd.dst_rank = dst_rank;
        cmd.is_atomic = true;
        cmd.is_combine = is_combine;
        cmd.req_rptr = rptr;
        cmd.low_latency_buffer_idx = low_latency_buffer_idx;
        cmd.barrier_id = barrier_id;
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

  // Return the remote pointer as uint64_t
  return reinterpret_cast<uint64_t>(ipc_base_ptrs[dst_local_rank]) + offset;
}

__device__ static __forceinline__ void wait_until_cmd_consumed(
    DeviceToHostCmdBuffer* rb, uint64_t slot, int nvl_rank = -1,
    CmdType cmd_type = CmdType::EMPTY) {
  auto last_print = clock64();
  while (true) {
    uint64_t cur_tail = rb->volatile_tail();
    if (cur_tail > slot) {
      break;
    }
    if ((clock64() - last_print) > kPrintCycleInterval) {
      printf(
          "[wait_until_cmd_consumed, nvl_rank: %d, cmd: %d] still waiting, "
          "head=%lu tail=%lu "
          "slot=%lu\n",
          nvl_rank, static_cast<int>(cmd_type), (unsigned long)rb->head,
          (unsigned long)cur_tail, (unsigned long)slot);
      last_print = clock64();
    }
  }
}

__device__ static __forceinline__ void nvshmemi_ibgda_quiet(
    uint64_t const* ring_addrs, int num_ring_addrs, int nvl_rank = -1) {
  assert(num_ring_addrs % kRingsPerProxy == 0 &&
         "num_ring_addrs must be multiple of kRingsPerProxy");
  /* NOTE(MaoZiming): This should be sent to all proxy threads. Since each proxy
   * thread manages kRingsPerProxy ring buffers, we just need to post a quiet
   * command to one out of the kRingsPerProxy ring buffer. */
  int ring_idx = 0;
  auto* rb = reinterpret_cast<DeviceToHostCmdBuffer*>(
      static_cast<uintptr_t>(ring_addrs[ring_idx]));
  uint64_t cur_head = rb->head;
  uint64_t cur_tail = rb->volatile_tail();
  uint64_t inflight = cur_head - cur_tail;
  auto last_print = clock64();
  while (true) {
    cur_head = rb->head;
    cur_tail = rb->volatile_tail();
    inflight = cur_head - cur_tail;
    if (inflight < kMaxInflight) {
      uint64_t slot = cur_head;
      TransferCmd cmd{};
      cmd.cmd = 1;  // dummy valid cmd.
      cmd.cmd_type = CmdType::QUIET;
      rb->atomic_set_and_commit(cmd, &slot);
      wait_until_cmd_consumed(rb, slot, nvl_rank, CmdType::QUIET);
      break;
    }
    if ((clock64() - last_print) > kPrintCycleInterval) {
      printf(
          "[quiet] stuck waiting, inflight=%ld (cur_head=%lu "
          "cur_tail=%lu)\n",
          (long)inflight, (unsigned long)cur_head, (unsigned long)cur_tail);
      last_print = clock64();
    }
  }
}

__forceinline__ __device__ void nvshmem_sync_with_same_gpu_idx(
    uint64_t const* ring_addrs, int num_ring_addrs, int nvl_rank = -1) {
  int ring_idx = 0;
  auto* rb = reinterpret_cast<DeviceToHostCmdBuffer*>(
      static_cast<uintptr_t>(ring_addrs[ring_idx]));
  uint64_t cur_head = rb->head;
  uint64_t cur_tail = rb->volatile_tail();
  uint64_t inflight = cur_head - cur_tail;
  auto last_print = clock64();
  while (true) {
    cur_head = rb->head;
    cur_tail = rb->volatile_tail();
    inflight = cur_head - cur_tail;
    if (inflight < kMaxInflight) {
      uint64_t slot = cur_head;
      TransferCmd cmd{};
      cmd.cmd = 1;  // dummy valid cmd.
      cmd.cmd_type = CmdType::BARRIER;
      rb->atomic_set_and_commit(cmd, &slot);
      printf("[nvl_rank: %d] barrier posted slot %lu\n", nvl_rank,
             (unsigned long)slot);
      wait_until_cmd_consumed(rb, slot, nvl_rank, CmdType::BARRIER);
      printf("[nvl_rank: %d] barrier completed slot %lu\n", nvl_rank,
             (unsigned long)slot);
      break;
    }
    if ((clock64() - last_print) > kPrintCycleInterval) {
      printf(
          "[barrier] stuck waiting, inflight=%ld (cur_head=%lu "
          "cur_tail=%lu)\n",
          (long)inflight, (unsigned long)cur_head, (unsigned long)cur_tail);
      last_print = clock64();
    }
  }
}

}  // namespace uccl