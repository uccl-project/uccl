#pragma once
#include "common.hpp"
#include "d2h_queue_adapter.cuh"  // pack/unpack helpers & TransferCmd
#include "d2h_queue_host.hpp"
#include "fifo.hpp"         // MSCCLPP host-side FIFO
#include "fifo_device.hpp"  // ProxyTrigger
#include <cstdint>

namespace d2hq {

inline void unpack_transfer_cmd(uint64_t fst, uint64_t snd, TransferCmd& c) {
  uint8_t cmd_type_u8 = static_cast<uint8_t>(fst & 0xFFull);
  uint8_t dst_rank_u8 = static_cast<uint8_t>((fst >> 8) & 0xFFull);
  uint32_t bytes_and_val_u32 =
      static_cast<uint32_t>((fst >> 16) & 0xFFFFFFFFull);
  uint16_t idx_or_off_u16 = static_cast<uint16_t>((fst >> 48) & 0xFFFFull);

  uint32_t req_rptr_u32 = static_cast<uint32_t>(snd & 0xFFFFFFFFull);
  uint32_t req_lptr_u32 = static_cast<uint32_t>((snd >> 32) & 0xFFFFFFFFull);

  c.cmd_type = static_cast<CmdType>(cmd_type_u8);
  c.dst_rank = dst_rank_u8;
  c.bytes_and_val = bytes_and_val_u32;
  c.req_rptr = req_rptr_u32;
  c.req_lptr = req_lptr_u32;
  c.expert_idx = idx_or_off_u16;  // union with atomic_offset
}

class FifoCmdQueue {
 public:
  explicit FifoCmdQueue(mscclpp::Fifo* f = nullptr) : fifo_(f) {}

  // Obtain device-visible handle to pass into kernels
  mscclpp::FifoDeviceHandle device_handle() const {
    return fifo_ ? fifo_->deviceHandle() : mscclpp::FifoDeviceHandle{};
  }

  // Non-blocking try-pop: returns true if a command was popped into 'out'
  bool try_pop(TransferCmd& out) {
    if (!fifo_) return false;

    mscclpp::ProxyTrigger t = fifo_->poll();
    // Empty slot if fst == 0
    if (t.fst == 0) return false;

    // FIFO flips the MSB of 'snd' on push; undo here.
    constexpr uint64_t kFlipMask = (uint64_t{1} << 63);
    uint64_t fst = t.fst;
    uint64_t snd = (t.snd ^ kFlipMask);

    unpack_transfer_cmd(fst, snd, out);
    fifo_->pop();
    return true;
  }

  int size() const { return fifo_ ? fifo_->size() : 0; }

  void reset_underlying(mscclpp::Fifo* f) { fifo_ = f; }

 private:
  mscclpp::Fifo* fifo_ = nullptr;  // store pointer, not object
};

}  // namespace d2hq