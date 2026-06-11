#pragma once
//
// UCCL-GIN Rail device backend (P0 seed).
//
// Minimal device-side ops for the EFA `Rail` (scale-out / inter-node) path of
// the planned `handle::UCCLGin`. They push the old 16B `TransferCmd` into a
// host-pinned D2H ring (`d2hq::D2HHandle`); the UCCL CPU proxy drains the ring
// and posts EFA verbs. This header is intentionally lean (only D2H ring + cstdint)
// so it can later be #included by the JIT-compiled DeepEP kernels too.
//
// Covered now (per the standalone microbench scope): put + red_add_rel + the
// piggyback tail add. Not yet: signal / wait / coalescing (see uccl_gin_plan.md).
//
// Offset conventions (must match ep/src/rdma.cpp):
//   * put (WRITE): req_lptr/req_rptr are window offsets shifted right by
//     kWriteAddrShiftNormal (=2, 4-byte granularity), relative to the single
//     registered window base.
//   * red_add (ordered ATOMIC): value = signed delta (must fit 15 bits),
//     req_rptr = RAW byte offset of the counter inside the receiver's atomic
//     buffer (<= AtomicsImm::kOFF_MASK), atomic_offset = 1 (non-zero => the
//     proxy takes the PackAtomicWithSeq ordered path).
//   * put_tail_add (WRITE + piggyback): atomic_offset = RAW byte offset of the
//     receiver tail word, atomic_val = count delta. The proxy emits the count
//     as an ordered WRITE_WITH_IMM only when atomic_offset > 0, so tail slots
//     are 1-based (slot 0 reserved) to stay compatible with the V1 trigger.

#include "../ring_buffer.cuh"        // TransferCmd, CmdType, make_cmd_type, kWriteAddrShiftNormal
#include "../d2h_queue_device.cuh"   // d2hq::D2HHandle
#include <cstdint>

namespace uccl_gin {

static constexpr uint32_t kAtomicOffMask = 0x1FFFu;
static constexpr int kAtomicValueMin = -(1 << 14);
static constexpr int kAtomicValueMax = (1 << 14) - 1;

// Window offset (4-byte shifted) for a payload pointer relative to window base.
__device__ __forceinline__ uint32_t window_off(uint64_t addr, uint64_t window_base) {
  if (addr < window_base || ((addr - window_base) & ((1u << kWriteAddrShiftNormal) - 1u))) {
    __trap();
  }
  const uint64_t shifted = (addr - window_base) >> kWriteAddrShiftNormal;
  if (shifted > 0xFFFFFFFFull) {
    __trap();
  }
  return static_cast<uint32_t>(shifted);
}

// Rail put: one-sided WRITE of `bytes` from local window offset -> remote window
// offset on global rank `dst_rank`. Both offsets are already 4-byte shifted
// (use window_off()). Returns the D2H ring slot it landed in.
__device__ __forceinline__ uint64_t rail_put(d2hq::D2HHandle* q, int dst_rank,
                                             uint32_t bytes,
                                             uint32_t local_off_shifted,
                                             uint32_t remote_off_shifted) {
  TransferCmd cmd{};
  cmd.cmd_type = make_cmd_type(CmdType::WRITE, /*is_combine=*/false,
                               /*low_latency=*/false);
  cmd.dst_rank = static_cast<uint8_t>(dst_rank);
  cmd.bytes = bytes;
  cmd.req_lptr = local_off_shifted;
  cmd.req_rptr = remote_off_shifted;
  uint64_t slot = 0;
  q->atomic_set_and_commit(cmd, &slot, kUCCLGinMaxInflightNormal);
  return slot;
}

// Rail put + EFA piggyback tail add: one payload WRITE carries a receiver-side
// software-atomic immediate.  This mirrors the original UCCL/EP EFA path where
// a chunk payload WR also advances the channel tail, avoiding a separate tiny
// WRITE_WITH_IMM for the count update.
//
// The 16B TransferCmd stores the piggyback delta in `atomic_val`, an 8-bit field
// sharing the bytes word, so this helper is intentionally chunk-count only
// (1..255).  Larger finish/control deltas still use rail_red_add.
__device__ __forceinline__ uint64_t rail_put_tail_add(
    d2hq::D2HHandle* q, int dst_rank, uint32_t bytes,
    uint32_t local_off_shifted, uint32_t remote_off_shifted, uint32_t count_delta,
    uint32_t atomic_byte_off) {
  if (count_delta == 0 || count_delta > 0xFFu || atomic_byte_off > kAtomicOffMask ||
      (atomic_byte_off & 0x7u)) {
    __trap();
  }
  TransferCmd cmd{};
  cmd.cmd_type = make_cmd_type(CmdType::WRITE, /*is_combine=*/false,
                               /*low_latency=*/false);
  cmd.dst_rank = static_cast<uint8_t>(dst_rank);
  cmd.bytes = bytes;
  cmd.atomic_val = static_cast<uint8_t>(count_delta);
  cmd.req_lptr = local_off_shifted;
  cmd.req_rptr = remote_off_shifted;
  cmd.atomic_offset = static_cast<uint16_t>(atomic_byte_off);
  uint64_t slot = 0;
  q->atomic_set_and_commit(cmd, &slot, kUCCLGinMaxInflightNormal);
  return slot;
}

// Rail red_add_rel: ordered remote atomic add of `delta` to the int64 counter at
// `atomic_byte_off` inside the receiver's atomic buffer on global rank `dst_rank`.
// The proxy applies it in seq order (PackAtomicWithSeq) so a stream of adds to
// the same counter cannot be reordered. `delta` must fit 15 bits.
__device__ __forceinline__ uint64_t rail_red_add(d2hq::D2HHandle* q, int dst_rank,
                                                 int delta,
                                                 uint32_t atomic_byte_off) {
  if (delta < kAtomicValueMin || delta > kAtomicValueMax ||
      atomic_byte_off > kAtomicOffMask || (atomic_byte_off & 0x7u)) {
    __trap();
  }
  TransferCmd cmd{};
  // UCCL-GIN Rail ordered atomics run only in normal mode. PackAtomicWithSeq
  // consumes the legacy is_combine bit as seq[3], so phase labeling here would
  // be inert and unsafe if this command were accidentally routed to fast mode.
  cmd.cmd_type = make_cmd_type(CmdType::ATOMIC, /*is_combine=*/false,
                               /*low_latency=*/false);
  cmd.dst_rank = static_cast<uint8_t>(dst_rank);
  cmd.value = delta;               // unions with req_lptr; proxy reads cmd.value
  cmd.req_rptr = atomic_byte_off;  // RAW byte offset into receiver atomic buffer
  cmd.atomic_offset = 1;           // non-zero => ordered (PackAtomicWithSeq) path
  uint64_t slot = 0;
  q->atomic_set_and_commit(cmd, &slot, kUCCLGinMaxInflightNormal);
  return slot;
}

}  // namespace uccl_gin
