#pragma once
//
// handle::UCCLGin (standalone) — the UCCL-GIN device abstraction. Mirrors the
// method surface of DeepEP's `deep_ep::elastic::handle::NCCLGin`
// (`deep_ep/common/handle.cuh`) so the SAME kernel call sites
// (`gin.put<Team>(...)`, `gin.red_add_rel<Team>(...)`) work by just swapping the
// gin type:
//
//   Team == ncclTeamTagRail (scale-out / inter-node) -> UCCL D2H + proxy + EFA
//   Team == ncclTeamTagLsa  (scale-up / NVLink)       -> forward to NCCL/NVLink
//
// This is the lean, DeepEP-free handle: it depends only on the UCCL transport
// substrate (D2H rings + CPU proxy + EFA verbs) and the NCCL team tags, so it
// can be exercised by the standalone microbench. The DeepEP-coupled sibling
// (which composes an NCCLGin for the Lsa/World branches) is a separate header.
//
// SCOPE: the Rail branch of put / red_add_rel / put_tail_add (piggyback) /
// quiet is implemented. The Lsa branch and the remaining NCCLGin surface trap so
// gaps are loud, not silent.

#include "resources.cuh"
#include "uccl_gin_rail.cuh"
#include <nccl_device.h>   // ncclTeamTagRail, ncclTeamTagLsa
#include <type_traits>

namespace uccl_gin {

// Spin-wait diagnostic cadence for quiet() (~10s at 2GHz).
static constexpr unsigned long long kUCCLGinQuietPrintCycles = 20000000000ull;

struct UCCLGin {
  UCCLGinResources res;

  __device__ __forceinline__ explicit UCCLGin(const UCCLGinResources& r) : res(r) {}

  // Choose a D2H lane. NCCLGin hides lane behind qp/context; here the caller may
  // pass a hint (e.g. channel idx); default round-robins on the hint.
  __device__ __forceinline__ d2hq::D2HHandle* lane(int hint) const {
    if (res.d2h_queues == nullptr || res.num_queues == 0) {
      __trap();
    }
    return res.d2h_queues[uccl_gin::queue_index_from_hint(res, hint)];
  }

  __device__ __forceinline__ void quiet_on_queue(d2hq::D2HHandle* q,
                                                 int label) const {
    uint64_t slot = 0;
    TransferCmd cmd{};
    cmd.cmd_type = make_cmd_type(CmdType::QUIET, /*is_combine=*/false,
                                 /*low_latency=*/false);
    q->atomic_set_and_commit(cmd, &slot, kUCCLGinMaxInflightNormal);

    auto last_print = clock64();
    while (true) {
#ifdef USE_MSCCLPP_FIFO_BACKEND
      if (q->fifo.poll(slot)) break;
#else
      if (q->tail() > slot) break;
#endif
      if (clock64() - last_print > kUCCLGinQuietPrintCycles) {
        printf("[UCCL-GIN quiet] waiting queue=%d slot=%llu\n", label,
               static_cast<unsigned long long>(slot));
        last_print = clock64();
      }
      __nanosleep(64);
    }
  }

  // ---- put -------------------------------------------------------------
  // Signature mirrors handle::NCCLGin::put: symmetric pointers in, internally
  // converted to window offsets. `dst_rank` is the global proxy peer rank.
  // Rail currently means "same local rank on a different node"; other peers
  // are rejected by validate_rail_dst().
  template <typename team_t>
  __device__ __forceinline__ void put(void* recv_sym_ptr, void* send_sym_ptr,
                                      int num_bytes, int dst_rank,
                                      int lane_hint = 0) const {
    if constexpr (std::is_same_v<team_t, ncclTeamTagRail>) {
      validate_rail_dst(res, dst_rank);
      if (num_bytes < 0) {
        __trap();
      }
      if (num_bytes == 0) {
        return;
      }
      const uint32_t loff = window_off(reinterpret_cast<uint64_t>(send_sym_ptr),
                                       res.window_base, res.window_bytes,
                                       static_cast<uint32_t>(num_bytes));
      const uint32_t roff = window_off(reinterpret_cast<uint64_t>(recv_sym_ptr),
                                       res.window_base, res.window_bytes,
                                       static_cast<uint32_t>(num_bytes));
      auto* q = lane(lane_hint);
      uint32_t remaining = static_cast<uint32_t>(num_bytes);
      uint32_t byte_offset = 0;
      while (remaining != 0) {
        const uint32_t chunk =
            remaining > kTransferCmdMaxBytes ? kTransferCmdMaxAlignedBytes
                                             : remaining;
        rail_put(q, dst_rank, chunk, add_window_off(loff, byte_offset),
                 add_window_off(roff, byte_offset));
        remaining -= chunk;
        byte_offset += chunk;
      }
    } else {
      // Lsa (NVLink) — forward to NCCLGin / NVLink ptx. Not in standalone.
      __trap();
    }
  }

  // ---- put_tail_add (WRITE + piggyback count) --------------------------
  // One payload WRITE that also advances a receiver tail counter (1..255). The
  // tail offset is a RAW byte offset into the receiver atomic buffer and must be
  // non-zero (slot 0 reserved) so the proxy's piggyback trigger fires under the
  // V1-compatible (atomic_offset>0 && atomic_val>0) rule.
  template <typename team_t>
  __device__ __forceinline__ void put_tail_add(void* recv_sym_ptr,
                                               void* send_sym_ptr, int num_bytes,
                                               int dst_rank, int count_delta,
                                               uint32_t atomic_byte_off,
                                               int lane_hint = 0) const {
    if constexpr (std::is_same_v<team_t, ncclTeamTagRail>) {
      validate_rail_dst(res, dst_rank);
      if (num_bytes < 0 || count_delta <= 0 || count_delta > 0xFF ||
          atomic_byte_off == 0) {
        __trap();  // tail slots are 1-based; slot 0 is reserved (see header).
      }
      const uint32_t loff = window_off(reinterpret_cast<uint64_t>(send_sym_ptr),
                                       res.window_base, res.window_bytes,
                                       static_cast<uint32_t>(num_bytes));
      const uint32_t roff = window_off(reinterpret_cast<uint64_t>(recv_sym_ptr),
                                       res.window_base, res.window_bytes,
                                       static_cast<uint32_t>(num_bytes));
      auto* q = lane(lane_hint);
      const uint32_t bytes = static_cast<uint32_t>(num_bytes);
      if (bytes <= kTransferCmdMaxBytes) {
        rail_put_tail_add(q, dst_rank, bytes, loff, roff,
                          static_cast<uint32_t>(count_delta), atomic_byte_off);
        return;
      }

      // A final-chunk piggyback is not sufficient here: EFA SRD does not
      // guarantee arrival order across the payload WRs. Reuse the proxy's
      // existing plain-WRITE dependency tracking plus ordered ATOMIC so the
      // logical tail becomes visible only after every chunk completes. This
      // multi-command path is needed only above the 24-bit TransferCmd limit.
      uint32_t remaining = bytes;
      uint32_t byte_offset = 0;
      while (remaining != 0) {
        const uint32_t chunk =
            remaining > kTransferCmdMaxBytes ? kTransferCmdMaxAlignedBytes
                                             : remaining;
        rail_put(q, dst_rank, chunk, add_window_off(loff, byte_offset),
                 add_window_off(roff, byte_offset));
        remaining -= chunk;
        byte_offset += chunk;
      }
      rail_red_add(q, dst_rank, count_delta, atomic_byte_off);
    } else {
      __trap();
    }
  }

  // ---- red_add_rel -----------------------------------------------------
  // Mirrors handle::NCCLGin::red_add_rel. `sym_ptr` is a counter inside the
  // atomic buffer; offset is taken relative to atomic_tail_base. Ordered
  // (PackAtomicWithSeq) so a stream of adds to one counter is not reordered.
  //
  // Payload-before-tail ordering is guaranteed only for commands issued to the
  // same lane_hint / D2H queue. Cross-lane ordering requires an explicit quiet
  // or a higher-level protocol.
  template <typename team_t>
  __device__ __forceinline__ void red_add_rel(void* sym_ptr, int value,
                                              int dst_rank, int lane_hint = 0) const {
    if constexpr (std::is_same_v<team_t, ncclTeamTagRail>) {
      validate_rail_dst(res, dst_rank);
      const uint32_t off = static_cast<uint32_t>(
          reinterpret_cast<uint64_t>(sym_ptr) - res.atomic_tail_base);
      rail_red_add(lane(lane_hint), dst_rank, value, off);
    } else {
      __trap();  // Lsa: ptx::red_add_rel_sys
    }
  }

  // ---- quiet -----------------------------------------------------------
  // Drain one lane: enqueue a QUIET marker and wait for proxy acknowledgement
  // after the proxy has observed CQEs for prior WRITE/WRITE_VALUE commands on
  // that proxy thread. This makes prior source buffers safe to reuse.
  __device__ __forceinline__ void quiet(int lane_hint = 0) const {
    quiet_on_queue(lane(lane_hint), lane_hint);
  }

  // ---- put_value --------------------------------------------------------
  // Mirrors handle::NCCLGin::putValue: writes a single word to a remote
  // window slot. Used for notify counts and small metadata. The value travels
  // inline in TransferCmd::value; the CPU proxy copies it to a per-WR host
  // bounce slot before posting the EFA WRITE, so there is no GPU staging-slot
  // lifetime race.
  template <typename team_t>
  __device__ __forceinline__ void put_value(void* sym_ptr, int value,
                                            int dst_rank, int lane_hint = 0) const {
    if constexpr (std::is_same_v<team_t, ncclTeamTagRail>) {
      validate_rail_dst(res, dst_rank);
      const uint32_t roff = window_off(reinterpret_cast<uint64_t>(sym_ptr),
                                       res.window_base, res.window_bytes,
                                       static_cast<uint32_t>(sizeof(int)));
      rail_write_value(lane(lane_hint), dst_rank, value, roff);
    } else {
      __trap();  // Lsa
    }
  }
  __device__ __forceinline__ void flush() const {
    for (uint32_t i = 0; i < res.num_queues; ++i) {
      quiet_on_queue(res.d2h_queues[i], static_cast<int>(i));
    }
  }
  // NCCL-GIN's flush(coop) is one cooperative flush per group (elected thread
  // + group sync). Without the coop API here, forwarding to flush() would
  // silently run a full per-thread drain (CTA of 512 => 512 x num_queues
  // QUIET round trips). Keep the gap loud until the cooperative version is
  // implemented at DeepEP-integration time.
  template <typename coop_t>
  __device__ __forceinline__ void flush(coop_t) const {
    static_assert(sizeof(coop_t) == 0,
                  "UCCL-GIN: cooperative flush(coop) is not implemented; call "
                  "flush() from a single thread");
  }
};

}  // namespace uccl_gin
