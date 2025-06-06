/**
 * @file eqds.h
 * @brief EQDS congestion control [NSDI'22]
 */

#pragma once

#include "transport_config.h"
#include "transport_header.h"
#include "util.h"
#include "util_buffpool.h"
#include "util_efa.h"
#include "util_jring.h"
#include "util_list.h"
#include "util_timer.h"
#include <infiniband/verbs.h>
#include <atomic>
#include <iomanip>
#include <list>
#include <optional>

namespace uccl {

typedef uint64_t FlowID;

class UcclFlow;

namespace eqds {

constexpr bool pullno_lt(PullQuanta a, PullQuanta b) {
  return static_cast<int16_t>(a - b) < 0;
}
constexpr bool pullno_le(PullQuanta a, PullQuanta b) {
  return static_cast<int16_t>(a - b) <= 0;
}
constexpr bool pullno_eq(PullQuanta a, PullQuanta b) {
  return static_cast<int16_t>(a - b) == 0;
}
constexpr bool pullno_ge(PullQuanta a, PullQuanta b) {
  return static_cast<int16_t>(a - b) >= 0;
}
constexpr bool pullno_gt(PullQuanta a, PullQuanta b) {
  return static_cast<int16_t>(a - b) > 0;
}

#define PULL_QUANTUM 8192
#define PULL_SHIFT 13

static inline uint32_t unquantize(uint32_t pull_quanta) {
  return (uint32_t)pull_quanta << PULL_SHIFT;
}

static inline PullQuanta quantize_floor(uint32_t bytes) {
  return bytes >> PULL_SHIFT;
}

static inline PullQuanta quantize_ceil(uint32_t bytes) {
  return (bytes + PULL_QUANTUM - 1) >> PULL_SHIFT;
}

struct active_item {
  struct EQDSCC* eqds_cc;
  struct list_head active_link;
};

struct idle_item {
  struct EQDSCC* eqds_cc;
  struct list_head idle_link;
};

class CreditQPContext;

// Congestion control state for EQDS.
class EQDSCC {
 public:
  EQDSCC() {
    INIT_LIST_HEAD(&active_item.active_link);
    active_item.eqds_cc = this;

    INIT_LIST_HEAD(&idle_item.idle_link);
    idle_item.eqds_cc = this;
  }

  ~EQDSCC() {}

  std::function<bool(PullQuanta const&)> send_pullpacket;

  inline PullQuanta get_latest_pull() { return latest_pull_; }

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

  PullQuanta compute_pull_target();

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
  inline bool in_active_list() { return !list_empty(&active_item.active_link); }
  inline bool in_idle_list() { return !list_empty(&idle_item.idle_link); }

  inline void add_to_active_list(struct list_head* active_senders) {
    DCHECK(!in_active_list());
    list_add_tail(&active_item.active_link, active_senders);
  }

  inline void add_to_idle_list(struct list_head* idle_senders) {
    DCHECK(!in_idle_list());
    list_add_tail(&idle_item.idle_link, idle_senders);
  }

  inline void remove_from_active_list() {
    DCHECK(in_active_list());
    list_del(&active_item.active_link);
  }

  inline void remove_from_idle_list() {
    DCHECK(in_idle_list());
    list_del(&idle_item.idle_link);
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

  inline void inc_lastest_pull(uint32_t inc) {
    // Handle wraparound correctly.
    latest_pull_ += inc;
  }

  inline void dec_latest_pull(uint32_t dec) {
    // Handle wraparound correctly.
    latest_pull_ -= dec;
  }

  inline void inc_backlog(uint32_t inc) { backlog_bytes_ += inc; }

  inline void dec_backlog(uint32_t dec) {
    DCHECK(backlog_bytes_ >= dec);
    backlog_bytes_ -= dec;
  }

 private:
  static constexpr PullQuanta INIT_PULL_QUANTA = 100;
  static constexpr uint32_t kEQDSMaxCwnd =
      625000;  // BDPBytes = 100Gbps * 50us RTT
  // static constexpr uint32_t kEQDSMaxCwnd = 700000;

  /********************************************************************/
  /************************ Sender-side states ************************/
  /********************************************************************/
  uint32_t backlog_bytes_ = 0;
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

// Per engine Credit QP context.
class CreditQPContext {
 public:
  CreditQPContext(struct ibv_context* context, struct ibv_pd* pd, ibv_gid gid) {
    context_ = context;
    pd_ = pd;
    gid_ = gid;

    pacer_credit_cq_ =
        ibv_create_cq(context_, kMaxCqeTotal, nullptr, nullptr, 0);
    engine_credit_cq_ =
        ibv_create_cq(context_, kMaxCqeTotal, nullptr, nullptr, 0);
    DCHECK(pacer_credit_cq_ && engine_credit_cq_)
        << "Failed to create pacer/engine_credit_cq.";

    for (int i = 0; i < kMaxSrcDstQPCredit; i++) {
      credit_qp_list_[i] =
          __create_credit_qp(pacer_credit_cq_, engine_credit_cq_);
    }

    // Allocate memory for credit headers on engine/pacer.
    void* pacer_hdr_addr = mmap(
        nullptr, PullHdrBuffPool::kNumPktHdr * PullHdrBuffPool::kPullHdrSize,
        PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    DCHECK(pacer_hdr_addr != MAP_FAILED);

    pacer_hdr_mr_ =
        ibv_reg_mr(pd_, pacer_hdr_addr,
                   PullHdrBuffPool::kNumPktHdr * PullHdrBuffPool::kPullHdrSize,
                   IBV_ACCESS_LOCAL_WRITE);
    DCHECK(pacer_hdr_mr_ != nullptr);

    pacer_hdr_pool_ = new PullHdrBuffPool(pacer_hdr_mr_);

    void* engine_hdr_addr = mmap(
        nullptr, PullHdrBuffPool::kNumPktHdr * PullHdrBuffPool::kPullHdrSize,
        PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    DCHECK(engine_hdr_addr != MAP_FAILED);

    engine_hdr_mr_ =
        ibv_reg_mr(pd_, engine_hdr_addr,
                   PullHdrBuffPool::kNumPktHdr * PullHdrBuffPool::kPullHdrSize,
                   IBV_ACCESS_LOCAL_WRITE);
    DCHECK(engine_hdr_mr_ != nullptr);

    engine_hdr_pool_ = new PullHdrBuffPool(engine_hdr_mr_);

    // Allocate memory for frame desc on engine/pacer.
    pacer_frame_desc_pool_ = new FrameDescBuffPool();

    engine_frame_desc_pool_ = new FrameDescBuffPool();

    for (int i = 0; i < kMaxChainedWr; i++) {
      rq_wrs_[i].num_sge = 1;
      rq_wrs_[i].sg_list = &rq_sges_[i];
      rq_wrs_[i].next = (i == kMaxChainedWr - 1) ? nullptr : &rq_wrs_[i + 1];
    }

    for (int i = 0; i < kMaxSrcDstQPCredit; i++) {
      __post_recv_wrs_for_credit(kMaxSendRecvWrForCredit, i);
    }

    init_ = true;
  }

  ~CreditQPContext() {
    if (!init_) return;

    delete engine_frame_desc_pool_;

    delete pacer_frame_desc_pool_;

    delete engine_hdr_pool_;

    munmap(engine_hdr_mr_->addr, engine_hdr_mr_->length);
    ibv_dereg_mr(engine_hdr_mr_);

    munmap(pacer_hdr_mr_->addr, pacer_hdr_mr_->length);
    ibv_dereg_mr(pacer_hdr_mr_);

    for (int i = 0; i < kMaxSrcDstQPCredit; i++)
      ibv_destroy_qp(credit_qp_list_[i]);

    ibv_destroy_cq(pacer_credit_cq_);
    ibv_destroy_cq(engine_credit_cq_);

    init_ = false;
  }

  std::vector<FrameDesc*> engine_poll_credit_cq(void);

  int pacer_poll_credit_cq(void);

  inline uint32_t get_qpn(uint32_t idx) { return credit_qp_list_[idx]->qp_num; }

  inline void engine_push_frame_desc(uint64_t pkt_frame_desc) {
    engine_frame_desc_pool_->free_buff(pkt_frame_desc);
  }

  inline void pacer_push_frame_desc(uint64_t pkt_frame_desc) {
    pacer_frame_desc_pool_->free_buff(pkt_frame_desc);
  }

  inline uint64_t engine_pop_frame_desc() {
    uint64_t frame_desc;
    DCHECK(engine_frame_desc_pool_->alloc_buff(&frame_desc) == 0);
    return frame_desc;
  }
  inline uint64_t pacer_pop_frame_desc() {
    uint64_t frame_desc;
    DCHECK(pacer_frame_desc_pool_->alloc_buff(&frame_desc) == 0);
    return frame_desc;
  }

  inline void engine_push_pkt_hdr(uint64_t pkt_hdr_addr) {
    engine_hdr_pool_->free_buff(pkt_hdr_addr);
  }
  inline void pacer_push_pkt_hdr(uint64_t pkt_hdr_addr) {
    pacer_hdr_pool_->free_buff(pkt_hdr_addr);
  }

  inline uint64_t engine_pop_pkt_hdr() {
    uint64_t pkt_hdr;
    DCHECK(engine_hdr_pool_->alloc_buff(&pkt_hdr) == 0);
    return pkt_hdr;
  }
  inline uint64_t pacer_pop_pkt_hdr() {
    uint64_t pkt_hdr;
    DCHECK(pacer_hdr_pool_->alloc_buff(&pkt_hdr) == 0);
    return pkt_hdr;
  }

  inline struct ibv_qp* get_qp_by_idx(uint32_t idx) {
    DCHECK(idx < kMaxSrcDstQPCredit);
    return credit_qp_list_[idx];
  }

  inline uint32_t get_engine_hdr_pool_lkey() {
    return engine_hdr_pool_->get_lkey();
  }

  inline uint32_t get_pacer_hdr_pool_lkey() {
    return pacer_hdr_pool_->get_lkey();
  }

 private:
  void __post_recv_wrs_for_credit(int nb, uint32_t src_qp_idx);

  inline struct ibv_qp* __create_credit_qp(struct ibv_cq* scq,
                                           struct ibv_cq* rcq) {
    struct ibv_qp_init_attr qp_attr = {};
    qp_attr.send_cq = scq;
    qp_attr.recv_cq = rcq;
    qp_attr.cap.max_send_wr = kMaxSendRecvWrForCredit;
    qp_attr.cap.max_recv_wr = kMaxSendRecvWrForCredit;
    qp_attr.cap.max_send_sge = 1;
    qp_attr.cap.max_recv_sge = 1;
    qp_attr.cap.max_inline_data = EFA_MAX_INLINE_SIZE;

    qp_attr.qp_type = IBV_QPT_UD;
    struct ibv_qp* qp = ibv_create_qp(pd_, &qp_attr);
    DCHECK(qp) << "Failed to create credit QP.";

    struct ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = EFA_PORT_NUM;
    attr.qkey = QKEY;
    DCHECK(ibv_modify_qp(qp, &attr,
                         IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                             IBV_QP_QKEY) == 0)
        << "Failed to modify Credit QP INIT.";

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    DCHECK(ibv_modify_qp(qp, &attr, IBV_QP_STATE) == 0)
        << "Failed to modify Credit QP RTR.";

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = SQ_PSN;
    DCHECK(ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN) == 0)
        << "Failed to modify Credit QP RTS.";

    return qp;
  }

  struct ibv_qp* credit_qp_list_[kMaxSrcDstQPCredit];

  /////////// Data touched by UCCL Pacer ///////////
  // Used by pacer to poll TX events.
  struct ibv_cq* pacer_credit_cq_;

  struct ibv_mr* pacer_hdr_mr_;
  PullHdrBuffPool* pacer_hdr_pool_;
  FrameDescBuffPool* pacer_frame_desc_pool_;

  /////////// Data touched by UCCL Engine ///////////
  // Used by engine to poll RX events.
  struct ibv_cq* engine_credit_cq_;

  struct ibv_mr* engine_hdr_mr_;
  PullHdrBuffPool* engine_hdr_pool_;
  FrameDescBuffPool* engine_frame_desc_pool_;
  uint32_t post_rq_cnt_[kMaxSrcDstQPCredit] = {};

  struct ibv_sge rq_sges_[kMaxChainedWr];
  struct ibv_recv_wr rq_wrs_[kMaxChainedWr];

  /////////// Read only values ///////////
  bool init_ = false;
  struct ibv_context* context_;
  struct ibv_pd* pd_;
  ibv_gid gid_;
};

class EQDS {
 public:
  // How many credits to grant per pull.
  static const PullQuanta kCreditPerPull = 6;
  // How many senders to grant credit per iteration.
  // static const uint32_t kSendersPerPull = 2;
  static const uint32_t kSendersPerPull = 1;  // Good for allreduce

  // Reference: for PULL_QUANTUM = 16384, kLinkBandwidth = 400 * 1e9 / 8,
  // kCreditPerPull = 4, kSendersPerPull = 4, kPacingIntervalUs ~= 5.3 us.
  static const uint64_t kPacingIntervalUs =
      1.01 /* slower than line rate */ * (PULL_QUANTUM)*kCreditPerPull * 1e6 *
      kSendersPerPull / (kLinkBandwidth);

  CreditQPContext* credit_qp_ctx_[kNumEnginesPerVdev * 2];

  EQDSChannel channel_;

  // Make progress on the pacer.
  void run_pacer(void);

  // Handle registration requests.
  void handle_pull_request(void);

  // Handle Credit CQ TX events.
  void handle_poll_cq(void);

  // Grant credits to senders.
  void handle_grant_credit(void);

  bool grant_credit(EQDSCC* eqds_cc, bool idle, uint32_t* total_inc);

  bool send_pull_packet(EQDSCC* eqds_cc);

  EQDS(int pdev_idx) : pdev_idx_(pdev_idx), channel_() {
    if constexpr (kReceiverCCType != ReceiverCCType::kEQDS) {
      return;
    }

    pacing_interval_tsc_ = us_to_cycles(kPacingIntervalUs, freq_ghz);

    auto* factory_dev = EFAFactory::GetEFADevice(pdev_idx_);
    for (int i = 0; i < kNumEnginesPerVdev * 2; i++)
      credit_qp_ctx_[i] = new CreditQPContext(
          factory_dev->context, factory_dev->pd, factory_dev->gid);

    auto numa_node = pdev_idx_ / 2;
    auto pacer_cpu = PACER_CPU_START[numa_node] + pdev_idx_ % 2;

    // Initialize the pacer thread.
    pacer_th_ = std::thread([this, pacer_cpu] {
      // Pin the pacer thread to a specific CPU.
      pin_thread_to_cpu(pacer_cpu);
      LOG(INFO) << "[Pacer] thread " << pdev_idx_ << " running on CPU "
                << pacer_cpu;
      while (!shutdown_) {
        run_pacer();
      }
    });
  }

  ~EQDS() { shutdown(); }

  // Shutdown the EQDS pacer thread.
  inline void shutdown(void) {
    shutdown_ = true;

    if constexpr (kReceiverCCType != ReceiverCCType::kEQDS) {
      return;
    }

    pacer_th_.join();

    for (int i = 0; i < kNumEnginesPerVdev * 2; i++) delete credit_qp_ctx_[i];
  }

 private:
  std::thread pacer_th_;
  int pdev_idx_;

  LIST_HEAD(active_senders_);
  LIST_HEAD(idle_senders_);

  uint64_t last_pacing_tsc_;

  uint64_t pacing_interval_tsc_;

  bool shutdown_ = false;
};

};  // namespace eqds

};  // namespace uccl