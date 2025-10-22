#pragma once

#include "common.hpp"
#include "fifo_device.hpp"
#include "ring_buffer.cuh"
#include <cstdint>

namespace d2hq {

static_assert(sizeof(TransferCmd) == 16, "TransferCmd must be 128 bits");

__host__ __device__ inline void pack_transfer_cmd(TransferCmd const& c,
                                                  uint64_t& fst,
                                                  uint64_t& snd) {
  // Layout (explicit, endian-agnostic):
  // fst[  7:0 ]   = cmd_type (8)
  // fst[ 15:8 ]   = dst_rank (8)
  // fst[ 47:16 ]  = bytes_and_val (32)
  // fst[ 63:48 ]  = expert_idx / atomic_offset (16)
  // snd[ 31:0 ]   = req_rptr (32)
  // snd[ 63:32 ]  = req_lptr / value (32)
  fst = 0;
  snd = 0;

  uint64_t cmd_type_u8 = static_cast<uint8_t>(c.cmd_type);
  uint64_t dst_rank_u8 = static_cast<uint8_t>(c.dst_rank);
  uint64_t bytes_and_val_u32 = static_cast<uint32_t>(c.bytes_and_val);
  uint64_t idx_or_off_u16 = static_cast<uint16_t>(c.expert_idx);  // union
  uint64_t req_rptr_u32 = static_cast<uint32_t>(c.req_rptr);
  uint64_t req_lptr_u32 = static_cast<uint32_t>(c.req_lptr);  // union

  fst |= (cmd_type_u8 & 0xFFull);
  fst |= ((dst_rank_u8 & 0xFFull) << 8);
  fst |= ((bytes_and_val_u32 & 0xFFFFFFFFull) << 16);
  fst |= ((idx_or_off_u16 & 0xFFFFull) << 48);

  snd |= ((req_rptr_u32 & 0xFFFFFFFFull) << 0);
  snd |= ((req_lptr_u32 & 0xFFFFFFFFull) << 32);
}

struct D2HHandle {
#ifdef USE_MSCCLPP_FIFO_BACKEND
  mscclpp::FifoDeviceHandle fifo;
#else
  DeviceToHostCmdBuffer* ring;
#endif

  // Backend-aware accessors (now member functions)
  __device__ __forceinline__ uint64_t head() const {
#ifdef USE_MSCCLPP_FIFO_BACKEND
    // FIFO backend does not have a volatile head/tail; just return dummy.
    return kMaxInflight - 1;
#else
    return ring->head;
#endif
  }

  __host__ inline void init_from_dev_ptr(void* dev_ptr) noexcept {
#ifndef USE_MSCCLPP_FIFO_BACKEND
    ring = reinterpret_cast<DeviceToHostCmdBuffer*>(dev_ptr);
#else
    fifo = *reinterpret_cast<mscclpp::FifoDeviceHandle*>(dev_ptr);
#endif
  }

  __device__ __forceinline__ uint64_t tail() const {
#ifdef USE_MSCCLPP_FIFO_BACKEND
    return 0;
#else
    return ring->volatile_tail();
#endif
  }

  __device__ __forceinline__ bool atomic_set_and_commit(
      TransferCmd const& item, uint64_t* out_slot = nullptr) {
#ifdef USE_MSCCLPP_FIFO_BACKEND
#if defined(MSCCLPP_DEVICE_COMPILE)
    mscclpp::ProxyTrigger trig;
    uint64_t fst, snd;
    pack_transfer_cmd(item, fst, snd);
    trig.fst = fst;
    trig.snd = snd;
    // Only available inside device compilation
    uint64_t slot = fifo.push(trig, /*maxSpinCount=*/-1);
    if (out_slot) *out_slot = slot;
#else
    // Host stub (no push)
    if (out_slot) *out_slot = 0;
#endif

    return true;
#else
    // Ring buffer path
    return ring->atomic_set_and_commit(item, out_slot);
#endif
  }
};

}  // namespace d2hq