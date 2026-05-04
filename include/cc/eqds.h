/**
 * @file eqds.h
 * @brief EQDS congestion control [NSDI'22]
 */

#pragma once

#include "util/list.h"
#include <atomic>
#include <cstdint>

namespace uccl {
namespace eqds {

struct PacerCreditQPWrapper;

struct EQDSCC;

struct active_item {
  struct EQDSCC* eqds_cc;
  struct list_head active_link;
};

struct idle_item {
  struct EQDSCC* eqds_cc;
  struct list_head idle_link;
};

typedef uint8_t PullQuanta;

constexpr bool pullno_lt(PullQuanta a, PullQuanta b) {
  return static_cast<int8_t>(a - b) < 0;
}
constexpr bool pullno_le(PullQuanta a, PullQuanta b) {
  return static_cast<int8_t>(a - b) <= 0;
}
constexpr bool pullno_eq(PullQuanta a, PullQuanta b) {
  return static_cast<int8_t>(a - b) == 0;
}
constexpr bool pullno_ge(PullQuanta a, PullQuanta b) {
  return static_cast<int8_t>(a - b) >= 0;
}
constexpr bool pullno_gt(PullQuanta a, PullQuanta b) {
  return static_cast<int8_t>(a - b) > 0;
}

#define PULL_QUANTUM 16384
#define PULL_SHIFT 14

static inline uint32_t unquantize(uint8_t pull_quanta) {
  return (uint32_t)pull_quanta << PULL_SHIFT;
}

static inline PullQuanta quantize_floor(uint32_t bytes) {
  return bytes >> PULL_SHIFT;
}

static inline PullQuanta quantize_ceil(uint32_t bytes) {
  return (bytes + PULL_QUANTUM - 1) >> PULL_SHIFT;
}

// Per-QP congestion control state for EQDS.
struct EQDSCC {
  static constexpr PullQuanta INIT_PULL_QUANTA = 50;
  // static constexpr uint32_t kEQDSMaxCwnd = 1000000; // Bytes
  static constexpr uint32_t kEQDSMaxCwnd = 500000;  // Bytes

  /********************************************************************/
  /************************ Sender-side states ************************/
  /********************************************************************/

  // Last received highest credit in PullQuanta.
  PullQuanta pull_ = INIT_PULL_QUANTA;
  PullQuanta last_sent_pull_target_ = INIT_PULL_QUANTA;
  // Receive request credit in PullQuanta, but consume it in bytes
  uint32_t credit_pull_ = 0;
  uint32_t credit_spec_ = kEQDSMaxCwnd;
  bool in_speculating_ = true;
  /********************************************************************/
  /*********************** Receiver-side states ***********************/
  /********************************************************************/

  /***************** Shared between engine and pacer ******************/
  std::atomic<PullQuanta> highest_pull_target_;

  /*************************** Pacer only *****************************/
  PullQuanta latest_pull_;
  struct active_item active_item;
  struct idle_item idle_item;
  /************************* No modification **************************/
  uint32_t fid_;
  struct PacerCreditQPWrapper* pc_qpw_;

  inline uint32_t credit() { return credit_pull_ + credit_spec_; }

  // Called when transmitting a chunk.
  // Return true if we can transmit the chunk. Otherwise,
  // sender should pause sending this message until credit is received.
  inline bool spend_credit(uint32_t chunk_size) {
    if (credit_pull_ > 0) {
      if (credit_pull_ > chunk_size)
        credit_pull_ -= chunk_size;
      else
        credit_pull_ = 0;
      return true;
    } else if (in_speculating_ && credit_spec_ > 0) {
      if (credit_spec_ > chunk_size)
        credit_spec_ -= chunk_size;
      else
        credit_spec_ = 0;
      return true;
    }

    // let pull target can advance
    if (credit_spec_ > chunk_size)
      credit_spec_ -= chunk_size;
    else
      credit_spec_ = 0;

    return false;
  }

  // Called when we receiving ACK or pull packet.
  inline void stop_speculating() { in_speculating_ = false; }

  PullQuanta compute_pull_target(void* context, uint32_t chunk_size);

  inline bool handle_pull_target(PullQuanta pull_target) {
    PullQuanta hpt = highest_pull_target_.load();
    if (pullno_gt(pull_target, hpt)) {
      // Only we can increase the pull target.
      highest_pull_target_.store(pull_target);
      return true;
    }
    return false;
  }

  inline bool handle_pull(PullQuanta pullno) {
    if (pullno_gt(pullno, pull_)) {
      PullQuanta extra_credit = pullno - pull_;
      credit_pull_ += unquantize(extra_credit);
      if (credit_pull_ > kEQDSMaxCwnd) {
        credit_pull_ = kEQDSMaxCwnd;
      }
      pull_ = pullno;
      return true;
    }
    return false;
  }

  /// Helper functions called by pacer ///

  inline void set_fid(uint32_t fid) { fid_ = fid; }

  inline void set_pacer_credit_qpw(struct PacerCreditQPWrapper* pc_qpw) {
    pc_qpw_ = pc_qpw;
  }

  inline void init_active_item(void) {
    INIT_LIST_HEAD(&active_item.active_link);
    active_item.eqds_cc = this;
  }

  inline void init_idle_item(void) {
    INIT_LIST_HEAD(&idle_item.idle_link);
    idle_item.eqds_cc = this;
  }

  inline PullQuanta backlog() {
    auto hpt = highest_pull_target_.load();
    if (pullno_gt(hpt, latest_pull_)) {
      return hpt - latest_pull_;
    } else {
      return 0;
    }
  }

  inline bool idle_credit_enough() {
    PullQuanta idle_cumulate_credit;
    auto hpt = highest_pull_target_.load();

    if (pullno_ge(hpt, latest_pull_)) {
      idle_cumulate_credit = 0;
    } else {
      idle_cumulate_credit = latest_pull_ - hpt;
    }

    return idle_cumulate_credit >= quantize_floor(kEQDSMaxCwnd);
  }
};

}  // namespace eqds
}  // namespace uccl
