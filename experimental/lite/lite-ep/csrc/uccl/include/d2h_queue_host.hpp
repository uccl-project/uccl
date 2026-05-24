#pragma once
#include "common.hpp"
#include "ring_buffer.cuh"
#include <cstdint>
#ifdef USE_MSCCLPP_FIFO_BACKEND
#include "fifo.hpp"
#endif

namespace d2hq {
// Layout (must match pack_transfer_cmd):
// fst[  7:0 ]   = cmd_type (8)
// fst[ 15:8 ]   = dst_rank (8)
// fst[ 47:16 ]  = bytes_and_val (32)
// fst[ 63:48 ]  = expert_idx / atomic_offset (16)
// snd[ 31:0 ]   = req_rptr (32)
// snd[ 63:32 ]  = req_lptr / value (32)
inline void decode_packed_transfer_cmd(uint64_t fst, uint64_t snd,
                                       TransferCmd& out) noexcept {
  // This is needed because of
  // trigger.snd ^= flipMask in fifo.push.
  snd ^= (1ULL << 63);
  out = TransferCmd{};
  const uint8_t cmd_type_u8 = static_cast<uint8_t>(fst & 0xFFull);
  const uint8_t dst_rank_u8 = static_cast<uint8_t>((fst >> 8) & 0xFFull);
  const uint32_t bytes_and_val_u32 =
      static_cast<uint32_t>((fst >> 16) & 0xFFFFFFFFull);
  const uint16_t idx_or_off_u16 =
      static_cast<uint16_t>((fst >> 48) & 0xFFFFull);
  const uint32_t req_rptr_u32 = static_cast<uint32_t>(snd & 0xFFFFFFFFull);
  const uint32_t req_lptr_u32 =
      static_cast<uint32_t>((snd >> 32) & 0xFFFFFFFFull);

  out.cmd_type = static_cast<CmdType>(cmd_type_u8);
  out.dst_rank = static_cast<int>(dst_rank_u8);
  out.bytes_and_val = bytes_and_val_u32;
  out.expert_idx = static_cast<int>(idx_or_off_u16);
  out.req_rptr = static_cast<uint64_t>(req_rptr_u32);
  auto base = get_base_cmd(static_cast<CmdType>(cmd_type_u8));
  if (base == CmdType::ATOMIC) {
    out.value = static_cast<int32_t>(req_lptr_u32);
  } else {
    out.req_lptr = static_cast<uint64_t>(req_lptr_u32);
  }
}

#ifdef USE_MSCCLPP_FIFO_BACKEND
inline TransferCmd decode_packed_transfer_cmd(uint64_t fst,
                                              uint64_t snd) noexcept {
  TransferCmd c{};
  decode_packed_transfer_cmd(fst, snd, c);
  return c;
}

inline TransferCmd decode_from_trigger(
    mscclpp::ProxyTrigger const& trig) noexcept {
  return decode_packed_transfer_cmd(trig.fst, trig.snd);
}
#endif

struct HostD2HHandle {
#ifdef USE_MSCCLPP_FIFO_BACKEND
  mscclpp::Fifo* fifo = nullptr;
#else
  DeviceToHostCmdBuffer* ring = nullptr;
#endif

#ifndef USE_MSCCLPP_FIFO_BACKEND
  inline uint64_t volatile_head() const noexcept {
    return ring->volatile_head();
  }
  inline uint64_t volatile_tail() const noexcept {
    return ring->volatile_tail();
  }

  inline CmdType volatile_load_cmd_type(size_t idx) const noexcept {
    return ring->volatile_load_cmd_type(static_cast<int>(idx));
  }

  inline TransferCmd& load_cmd_entry(size_t idx) noexcept {
    return ring->load_cmd_entry(static_cast<int>(idx));
  }

  inline void volatile_clear_cmd_type(size_t idx) noexcept {
    ring->volatile_clear_cmd_type(static_cast<int>(idx));
  }

  inline void cpu_volatile_store_tail(uint64_t new_tail) noexcept {
    ring->cpu_volatile_store_tail(new_tail);
  }

  inline void mark_acked(size_t idx) noexcept { ring->mark_acked(idx); }

  inline void clear_acked(size_t idx) noexcept { ring->clear_acked(idx); }

  inline bool is_acked(size_t idx) const noexcept {
    return ring->is_acked(idx);
  }

  inline uint64_t advance_tail_from_mask() noexcept {
    return ring->advance_tail_from_mask();
  }

  inline size_t capacity() const noexcept { return ring->capacity; }
#endif
};

#ifdef USE_MSCCLPP_FIFO_BACKEND
inline void init_handle(HostD2HHandle& h, mscclpp::Fifo* q) { h.fifo = q; }

inline HostD2HHandle make_handle(mscclpp::Fifo* q) {
  HostD2HHandle h{};
  init_handle(h, q);
  return h;
}
#else
inline void init_handle(HostD2HHandle& h, DeviceToHostCmdBuffer* rb) {
  h.ring = rb;
}

inline HostD2HHandle make_handle(DeviceToHostCmdBuffer* rb) {
  HostD2HHandle h{};
  init_handle(h, rb);
  return h;
}
#endif

#ifdef USE_MSCCLPP_FIFO_BACKEND

inline void init_d2h_from_fifo(mscclpp::Fifo* const* fifos, size_t count,
                               std::vector<HostD2HHandle>& storage,
                               std::vector<HostD2HHandle*>& out) {
  storage.resize(count);
  out.clear();
  out.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    storage[i].fifo = fifos[i];
    out.push_back(&storage[i]);
  }
}

inline void init_d2h_from_fifo(
    std::vector<std::unique_ptr<mscclpp::Fifo>> const& fifo_uptrs,
    std::vector<HostD2HHandle>& storage, std::vector<HostD2HHandle*>& out) {
  std::vector<mscclpp::Fifo*> ptrs;
  ptrs.reserve(fifo_uptrs.size());
  for (auto const& u : fifo_uptrs) ptrs.push_back(u.get());
  init_d2h_from_fifo(ptrs.data(), ptrs.size(), storage, out);
}

inline void init_d2h_from_fifo(
    std::vector<std::unique_ptr<mscclpp::Fifo>> const& fifo_uptrs,
    std::vector<HostD2HHandle>& storage) {
  std::vector<mscclpp::Fifo*> ptrs;
  ptrs.reserve(fifo_uptrs.size());
  for (auto const& u : fifo_uptrs) ptrs.push_back(u.get());
  std::vector<HostD2HHandle*> dummy;
  init_d2h_from_fifo(ptrs.data(), ptrs.size(), storage, dummy);
}
#else

inline void init_d2h_from_ring(DeviceToHostCmdBuffer* rbs, size_t count,
                               std::vector<HostD2HHandle>& storage) {
  storage.resize(count);
  for (size_t i = 0; i < count; ++i) {
    storage[i] = make_handle(&rbs[i]);
  }
}
#endif  // USE_MSCCLPP_FIFO_BACKEND

inline void init_from_addr(HostD2HHandle& h, uintptr_t addr) {
#ifdef USE_MSCCLPP_FIFO_BACKEND
  h.fifo = reinterpret_cast<mscclpp::Fifo*>(addr);
#else
  h.ring = reinterpret_cast<DeviceToHostCmdBuffer*>(addr);
#endif
}

}  // namespace d2hq