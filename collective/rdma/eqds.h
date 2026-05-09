/**
 * @file eqds.h
 * @brief EQDS congestion control [NSDI'22]
 */

#pragma once

#include "transport_config.h"
#include "util/jring.h"
#include "util/latency.h"
#include "util/list.h"
#include "util/util.h"
#include "util_buffpool.h"
#include "util_timer.h"
#include <cc/eqds.h>
#include <infiniband/verbs.h>
#include <atomic>
#include <iomanip>
#include <list>
#include <optional>

namespace uccl {

namespace eqds {

struct EQDS;
class CreditChunkBuffPool;
struct PacerCreditQPWrapper;

struct pacer_credit_cq_item {
  PacerCreditQPWrapper* pc_qpw;
  struct list_head poll_link;
};
struct PacerCreditQPWrapper {
  struct ibv_qp* credit_qp_;
  struct ibv_cq_ex* pacer_credit_cq_;
  CreditChunkBuffPool* pacer_credit_chunk_pool_;

  uint32_t poll_cq_cnt_ = 0;
  pacer_credit_cq_item poll_item;
};
/**
 * @brief Buffer pool for pull packets.
 */
class CreditChunkBuffPool : public BuffPool {
 public:
  static constexpr uint32_t kPktSize = 4;
  static constexpr uint32_t kChunkSize = kPktSize * 1;
  static constexpr uint32_t kNumChunk = kMaxBatchCQ << 6;
  static constexpr uint32_t kCreditMRSize = kNumChunk * kChunkSize;
  static_assert((kNumChunk & (kNumChunk - 1)) == 0,
                "kNumChunk must be power of 2");

  CreditChunkBuffPool(struct ibv_mr* mr)
      : BuffPool(kNumChunk, kChunkSize, mr) {}

  ~CreditChunkBuffPool() = default;
};

class EQDSChannel {
  static constexpr uint32_t kChannelSize = 2048;

 public:
  struct Msg {
    enum Op : uint8_t {
      kRequestPull,
    };
    Op opcode;
    EQDSCC* eqds_cc;
  };
  static_assert(sizeof(Msg) % 4 == 0, "channelMsg must be 32-bit aligned");

  EQDSChannel() { cmdq_ = create_ring(sizeof(Msg), kChannelSize); }

  ~EQDSChannel() { free(cmdq_); }

  jring_t* cmdq_;
};

class EQDS {
 public:
  // How many credits to grant per pull.
  static constexpr PullQuanta kCreditPerPull = 4;
  // How many senders to grant credit per iteration.
  static constexpr uint32_t kSendersPerPull = 1;

  // Reference: for PULL_QUANTUM = 16384, LINK_BANDWIDTH = 400 * 1e9 / 8,
  // kCreditPerPull = 4, kSendersPerPull = 4, kPacingIntervalUs ~= 5.3 us.

  EQDSChannel channel_;

  // Make progress on the pacer.
  void run_pacer(void);

  void handle_poll_cq(void);

  void handle_grant_credit(void);

  bool poll_cq(struct PacerCreditQPWrapper* pc_qpw);

  // Handle registration requests.
  void handle_pull_request(void);

  // Grant credit to the sender of this flow.
  bool grant_credit(EQDSCC* eqds_cc, bool idle, PullQuanta* ret_increment);

  bool send_pull_packet(EQDSCC* eqds_cc);

  // For original EQDS, it stalls the pacer when ECN ratio reaches a threshold
  // (i.e., 10%). Here we use resort to RTT-based stall.
  void update_cc_state(void);

  // [Thread-safe] Request pacer to grant credit to this flow.
  inline void request_pull(EQDSCC* eqds_cc) {
    EQDSChannel::Msg msg = {
        .opcode = EQDSChannel::Msg::Op::kRequestPull,
        .eqds_cc = eqds_cc,
    };
    while (jring_mp_enqueue_bulk(channel_.cmdq_, &msg, 1, nullptr) != 1) {
    }
  }

  EQDS(int dev, double link_bandwidth);

  ~EQDS() {}

  // Shutdown the EQDS pacer thread.
  inline void shutdown(void) {
    shutdown_.store(true, std::memory_order_release);
    pacer_th_.join();
  }

 private:
  std::thread pacer_th_;
  int dev_;

  LIST_HEAD(active_senders_);
  LIST_HEAD(idle_senders_);
  LIST_HEAD(poll_cq_list_);

  uint64_t last_pacing_tsc_;

  uint64_t pacing_interval_tsc_;

  std::atomic<bool> shutdown_{false};
};

}  // namespace eqds
};  // namespace uccl