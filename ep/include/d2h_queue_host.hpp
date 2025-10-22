#pragma once
#include "common.hpp"
#include "ring_buffer.cuh"
#include <cstdint>
#ifdef USE_MSCCLPP_FIFO_BACKEND
#include "fifo.hpp"
#endif

#define NOT_IMPLEMENTED()                                                    \
  do {                                                                       \
    std::fprintf(stderr, "ERROR: Not implemented (%s:%d in %s)\n", __FILE__, \
                 __LINE__, __func__);                                        \
    assert(false && "NOT IMPLEMENTED");                                      \
    std::abort();                                                            \
  } while (0)

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

  // Populate TransferCmd
  out.cmd_type = static_cast<CmdType>(cmd_type_u8);
  out.dst_rank = static_cast<int>(dst_rank_u8);
  out.bytes_and_val = bytes_and_val_u32;
  out.expert_idx = static_cast<int>(idx_or_off_u16);
  out.req_rptr = static_cast<uint64_t>(req_rptr_u32);
  auto base = get_base_cmd(static_cast<CmdType>(cmd_type_u8));
  if (base == CmdType::ATOMIC) {
    // upper 32 bits carry the atomic immediate (signed)
    out.value = static_cast<int32_t>(req_lptr_u32);
  } else {
    // non-atomic: upper 32 bits carry req_lptr
    out.req_lptr = static_cast<uint64_t>(req_lptr_u32);
  }

  // printf("Decoded cmd: type=%d, dst_rank=%d, bytes_and_val=%u, idx_or_off=%u,
  // req_rptr=%u, req_lptr=%u\n",
  //        static_cast<int>(out.cmd_type), out.dst_rank, out.bytes_and_val,
  //        out.expert_idx, static_cast<uint32_t>(out.req_rptr),
  //        static_cast<uint32_t>(out.req_lptr));
}

#ifdef USE_MSCCLPP_FIFO_BACKEND
// Convenience return-by-value variant
inline TransferCmd decode_packed_transfer_cmd(uint64_t fst,
                                              uint64_t snd) noexcept {
  TransferCmd c{};
  decode_packed_transfer_cmd(fst, snd, c);
  return c;
}

// Direct decode from a ProxyTrigger (FIFO poll result)
inline TransferCmd decode_from_trigger(
    mscclpp::ProxyTrigger const& trig) noexcept {
  return decode_packed_transfer_cmd(trig.fst, trig.snd);
}
#endif

struct HostD2HHandle {
#ifdef USE_MSCCLPP_FIFO_BACKEND
  mscclpp::Fifo* fifo = nullptr;  // queue wrapper (pop-based)
#else
  DeviceToHostCmdBuffer* ring = nullptr;  // indexed ring buffer
#endif

  inline uint64_t volatile_head() const noexcept {
#ifdef USE_MSCCLPP_FIFO_BACKEND
    // No random-access head in FIFO; provide a conservative value (no new work)
    // so callers using ring-style scans do nothing. Real consumption should
    // use try_pop() via host_try_pop_next().
    NOT_IMPLEMENTED();
    return 0;
#else
    return ring->volatile_head();
#endif
  }

  inline uint64_t volatile_tail() const noexcept {
#ifdef USE_MSCCLPP_FIFO_BACKEND
    // Same note as above; return 0 to keep scans inert under FIFO.
    NOT_IMPLEMENTED();
    return 0;
#else
    return ring->volatile_tail();
#endif
  }

  inline CmdType volatile_load_cmd_type(size_t idx) const noexcept {
#ifndef USE_MSCCLPP_FIFO_BACKEND
    return ring->volatile_load_cmd_type(static_cast<int>(idx));
#else
    // FIFO has no indexed view; return EMPTY so scan-based callers will skip.
    (void)idx;
    NOT_IMPLEMENTED();
    return CmdType::EMPTY;
#endif
  }

  inline TransferCmd& load_cmd_entry(size_t idx) noexcept {
#ifndef USE_MSCCLPP_FIFO_BACKEND
    return ring->load_cmd_entry(static_cast<int>(idx));
#else
    // FIFO is pop-driven. Provide a thread-local slot to return by reference.
    static thread_local TransferCmd tmp_cmd{};
    (void)idx;  // FIFO backend ignores idx; it is sequential, not random-access
    tmp_cmd.cmd_type = CmdType::EMPTY;
    tmp_cmd.bytes_and_val = 0;
    tmp_cmd.dst_rank = 0;
    tmp_cmd.expert_idx = 0;
    tmp_cmd.req_lptr = 0;
    tmp_cmd.req_rptr = 0;
    return tmp_cmd;
#endif
  }

  inline void volatile_clear_cmd_type(size_t idx) noexcept {
#ifndef USE_MSCCLPP_FIFO_BACKEND
    ring->volatile_clear_cmd_type(static_cast<int>(idx));
#else
    (void)idx;
    NOT_IMPLEMENTED();
#endif
  }

  inline void cpu_volatile_store_tail(uint64_t new_tail) noexcept {
#ifndef USE_MSCCLPP_FIFO_BACKEND
    ring->cpu_volatile_store_tail(new_tail);
#else
    (void)new_tail;
    NOT_IMPLEMENTED();
#endif
  }

  inline void mark_acked(size_t idx) noexcept {
#ifndef USE_MSCCLPP_FIFO_BACKEND
    ring->mark_acked(idx);
#else
    (void)idx;
    NOT_IMPLEMENTED();
#endif
  }

  inline void clear_acked(size_t idx) noexcept {
#ifndef USE_MSCCLPP_FIFO_BACKEND
    ring->clear_acked(idx);
#else
    (void)idx;
#endif
  }

  inline bool is_acked(size_t idx) const noexcept {
#ifndef USE_MSCCLPP_FIFO_BACKEND
    return ring->is_acked(idx);
#else
    (void)idx;
    NOT_IMPLEMENTED();
    return false;
#endif
  }

  inline uint64_t advance_tail_from_mask() noexcept {
#ifndef USE_MSCCLPP_FIFO_BACKEND
    return ring->advance_tail_from_mask();
#else
    NOT_IMPLEMENTED();
    return 0;
#endif
  }

  inline size_t capacity() const noexcept {
#ifndef USE_MSCCLPP_FIFO_BACKEND
    return ring->capacity;
#else
    NOT_IMPLEMENTED();
    return fifo ? static_cast<size_t>(fifo->size()) : 0;
#endif
  }
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

// Simple FIFO initialization helpers (no wrapper layer needed anymore)
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