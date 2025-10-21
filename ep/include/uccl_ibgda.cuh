#pragma once
#include "common.hpp"
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
    uint64_t atomic_offset = 0, uint64_t atomic_val = 0) {
  // NOTE(MaoZiming): different from the nvshmemi_ibgda_put_nbi_warp in
  // ibgda_device.cuh, we don't do warp-cooperation.
  if (lane_id != 0) return;
  int thread_idx = (expert_idx % num_ring_addrs) % kNumThBlocks;
  int ring_buffer_idx = (expert_idx % num_ring_addrs) / kNumThBlocks;
  assert(ring_buffer_idx < kRingsPerProxy);
  int ring_idx = thread_idx * kRingsPerProxy + ring_buffer_idx;
  assert(ring_idx < num_ring_addrs);
  unsigned long long rptr_val = static_cast<unsigned long long>(req_rptr);
  unsigned long long lptr_val = static_cast<unsigned long long>(req_lptr);
  unsigned long long bytes_val = static_cast<unsigned long long>(bytes);

  auto* rb = reinterpret_cast<DeviceToHostCmdBuffer*>(
      static_cast<uintptr_t>(ring_addrs[ring_idx]));

  uint64_t cur_head;
  uint64_t cur_tail;
  uint64_t inflight;
#ifdef USE_NORMAL_MODE
  if (low_latency_buffer_idx == -1) {
    /* Normal mode */
    expert_idx = 0;
    low_latency_buffer_idx = 0;
  }
#endif
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
      cmd.cmd_type =
          make_cmd_type(CmdType::WRITE, is_combine, low_latency_buffer_idx);
      cmd.req_rptr = rptr_val;
      cmd.req_lptr = lptr_val;
      cmd.bytes = bytes_val;
      cmd.dst_rank = dst_rank;
      cmd.expert_idx = expert_idx;
#ifdef USE_NORMAL_MODE
      cmd.atomic_offset = atomic_offset;
      cmd.atomic_val = atomic_val;
#endif
      rb->atomic_set_and_commit(cmd, &slot);
      break;
    }
    if (clock64() - last_print > kPrintCycleInterval) {
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
    bool skip_remote = false) {
  if (is_local_copy) {
    atomicAdd(reinterpret_cast<unsigned long long*>(rptr),
              static_cast<unsigned long long>(value));
  } else {
#ifdef USE_NORMAL_MODE
    if (skip_remote) {
      return;
    }
#endif
    rptr -= atomic_base_addr;
    int thread_idx = (warp_id % num_ring_addrs) % kNumThBlocks;
    int ring_buffer_idx = (warp_id % num_ring_addrs) / kNumThBlocks;
    assert(ring_buffer_idx < kRingsPerProxy);
    int ring_idx = thread_idx * kRingsPerProxy + ring_buffer_idx;
    assert(ring_idx < num_ring_addrs);
    auto* rb = reinterpret_cast<DeviceToHostCmdBuffer*>(
        static_cast<uintptr_t>(ring_addrs[ring_idx]));
    uint64_t cur_head;
    uint64_t cur_tail;
    uint64_t inflight;
    auto last_print = clock64();
#ifdef USE_NORMAL_MODE
    if (low_latency_buffer_idx == -1) {
      /* Normal mode */
      low_latency_buffer_idx = 0;
    }
#endif
    while (true) {
      // NOTE(MaoZiming): update the view.
      cur_head = rb->head;
      cur_tail = rb->volatile_tail();
      inflight = cur_head - cur_tail;
      if (inflight < kMaxInflight) {
        uint64_t slot = cur_head;
        TransferCmd cmd{};
        cmd.cmd_type =
            make_cmd_type(CmdType::ATOMIC, is_combine, low_latency_buffer_idx);
        // TODO(MaoZiming): Check fields here.
        // NOTE(MaoZiming): cmd is needed for proxy to process the command.
        cmd.value = value;
        cmd.dst_rank = dst_rank;
        cmd.req_rptr = rptr;
        rb->atomic_set_and_commit(cmd, &slot);
        break;
      } else {
        auto now = clock64();
        if (now - last_print > kPrintCycleInterval) {
          printf(
              "[nvshmemi_ibgda_amo_nonfetch_add] %p waiting ring_idx: %d, "
              "cur_head: "
              "%llu, cur_tail: %llu, inflight: %llu\n",
              rb, ring_idx, cur_head, cur_tail, inflight);
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
    CmdType cmd_type = CmdType::EMPTY, int label = -1) {
  auto last_print = clock64();
  while (true) {
    uint64_t cur_tail = rb->volatile_tail();
    if (cur_tail > slot) {
      break;
    }
    if ((clock64() - last_print) > kPrintCycleInterval) {
      printf(
          "[wait_until_cmd_consumed, nvl_rank: %d, cmd: %d, label: %d] still "
          "waiting, "
          "head=%lu tail=%lu "
          "slot=%lu\n",
          nvl_rank, static_cast<int>(cmd_type), label, (unsigned long)rb->head,
          (unsigned long)cur_tail, (unsigned long)slot);
      last_print = clock64();
    }
  }
}

__device__ static __forceinline__ void nvshmemi_ibgda_quiet(
    uint64_t const* ring_addrs, int num_ring_addrs, int nvl_rank = -1,
    int label = -1) {
  assert(num_ring_addrs % kRingsPerProxy == 0 &&
         "num_ring_addrs must be multiple of kRingsPerProxy");
  /* NOTE(MaoZiming): This is sent to all proxy threads. Since each proxy
   * thread manages kRingsPerProxy ring buffers, we just need to post a quiet
   * command to one out of the kRingsPerProxy ring buffer per cpu thread. */
  assert(num_ring_addrs % kRingsPerProxy == 0);
  assert(num_ring_addrs / kRingsPerProxy == kNumThBlocks);
  // First, atomically commit QUIET to one ring per proxy
  uint64_t slots[kNumThBlocks];
  int num_posted = 0;
  for (int ring_idx = 0; ring_idx < num_ring_addrs;
       ring_idx += kRingsPerProxy) {
    auto* rb = reinterpret_cast<DeviceToHostCmdBuffer*>(
        static_cast<uintptr_t>(ring_addrs[ring_idx]));

    while (true) {
      uint64_t cur_head = rb->head;
      uint64_t cur_tail = rb->volatile_tail();
      uint64_t inflight = cur_head - cur_tail;
      if (inflight < 1) {
        uint64_t slot = cur_head;
        TransferCmd cmd{};
        cmd.cmd_type = CmdType::QUIET;
        rb->atomic_set_and_commit(cmd, &slot);
        slots[num_posted] = slot;
        ++num_posted;
        break;
      }
    }
    break;
  }

  // Then wait for all QUIET commands to complete
  for (int i = 0; i < num_posted; ++i) {
    auto* rb = reinterpret_cast<DeviceToHostCmdBuffer*>(
        static_cast<uintptr_t>(ring_addrs[i * kRingsPerProxy]));
    wait_until_cmd_consumed(rb, slots[i], nvl_rank, CmdType::QUIET);
  }
}

__forceinline__ __device__ void nvshmem_sync_with_same_gpu_idx(
    uint64_t const* ring_addrs, int num_ring_addrs, int nvl_rank = -1,
    int label = -1) {
  assert(num_ring_addrs % kRingsPerProxy == 0 &&
         "num_ring_addrs must be multiple of kRingsPerProxy");
  assert(num_ring_addrs / kRingsPerProxy == kNumThBlocks);
  uint64_t slots[kNumThBlocks];
  int num_posted = 0;

  // First, post one BARRIER command per proxy
  for (int ring_idx = 0; ring_idx < num_ring_addrs;
       ring_idx += kRingsPerProxy) {
    auto* rb = reinterpret_cast<DeviceToHostCmdBuffer*>(
        static_cast<uintptr_t>(ring_addrs[ring_idx]));

    while (true) {
      uint64_t cur_head = rb->head;
      uint64_t cur_tail = rb->volatile_tail();
      uint64_t inflight = cur_head - cur_tail;
      if (inflight < 1) {
        uint64_t slot = cur_head;
        TransferCmd cmd{};
        cmd.cmd_type = CmdType::BARRIER;
        rb->atomic_set_and_commit(cmd, &slot);
        slots[num_posted++] = slot;
        break;
      }
    }
    break;
  }

  // Then wait for each proxy’s barrier to complete
  for (int i = 0; i < num_posted; ++i) {
    auto* rb = reinterpret_cast<DeviceToHostCmdBuffer*>(
        static_cast<uintptr_t>(ring_addrs[i * kRingsPerProxy]));
    wait_until_cmd_consumed(rb, slots[i], nvl_rank, CmdType::BARRIER, label);
  }
}

}  // namespace uccl