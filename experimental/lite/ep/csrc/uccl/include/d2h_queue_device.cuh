#pragma once

#include "common.hpp"
#include "fifo_device.hpp"
#include "ring_buffer.cuh"
#include <cstdint>

namespace d2hq {

static_assert(sizeof(TransferCmd) == 16, "TransferCmd must be 128 bits");

__device__ inline void pack_transfer_cmd(TransferCmd const& c, uint64_t& fst,
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
  uint64_t req_lptr_u32;
  {
    CmdType base = get_base_cmd(c.cmd_type);
    if (base == CmdType::ATOMIC) {
      req_lptr_u32 = static_cast<uint32_t>(static_cast<int32_t>(c.value));
    } else {
      req_lptr_u32 = static_cast<uint32_t>(c.req_lptr);
    }
  }
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

#ifndef USE_MSCCLPP_FIFO_BACKEND
  __device__ __forceinline__ uint64_t head() const { return ring->head; }

  __device__ __forceinline__ uint64_t tail() const {
    return ring->volatile_tail();
  }
#endif

#ifdef USE_MSCCLPP_FIFO_BACKEND
  __host__ inline void init_from_host_value(
      mscclpp::FifoDeviceHandle const& v) noexcept {
    fifo = v;
  }
#else
  __host__ inline void init_from_dev_ptr(void* dev_ptr) noexcept {
    ring = reinterpret_cast<DeviceToHostCmdBuffer*>(dev_ptr);
  }
#endif

  __device__ __forceinline__ bool atomic_set_and_commit(
      TransferCmd const& item, uint64_t* out_slot = nullptr) {
#ifdef USE_MSCCLPP_FIFO_BACKEND
#ifdef MSCCLPP_DEVICE_COMPILE
    mscclpp::ProxyTrigger trig;
    uint64_t fst, snd;
    pack_transfer_cmd(item, fst, snd);
    trig.fst = fst;
    trig.snd = snd;
    uint64_t slot = fifo.push(trig, /*maxSpinCount=*/-1);
    if (out_slot) *out_slot = slot;
#else
    if (out_slot) *out_slot = 0;
#endif
    return true;
#else
    return ring->atomic_set_and_commit(item, out_slot);
#endif
  }
};

}  // namespace d2hq