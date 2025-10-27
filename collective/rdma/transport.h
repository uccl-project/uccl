#pragma once

#include "eqds.h"
#include "rdma_io.h"
#include "transport_config.h"
#include "util/latency.h"
#include "util/shared_pool.h"
#include "util/util.h"
#include "util_rdma.h"
#include "util_timer.h"
#include <glog/logging.h>
#include <infiniband/verbs.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <netdb.h>

namespace uccl {

struct ConnID {
  void* context;
  int sock_fd;     // Used for further coordination if necessary.
  FlowID flow_id;  // Used for RDMAEndpoint to look up UcclFlow.
  PeerID peer_id;  // Used for UcclEngine to look up RDMAContext.
  int dev;
};

struct Mhandle {
  struct ibv_mr* mr;
};

/**
 * @class Channel
 * @brief A channel is a command queue for application threads to submit rx and
 * tx requests to the UcclFlow. A channel is only served by one UcclFlow, but
 * could be shared by multiple app threads if needed.
 */
class Channel {
  constexpr static uint32_t kChannelSize = 1024;

 public:
  struct Msg {
    enum Op : uint8_t { kTx, kRx, kRead, kWrite };
    Op opcode;
    PeerID peer_id;
    struct ucclRequest* ureq;
    PollCtx* poll_ctx;
  };
  static_assert(sizeof(Msg) % 4 == 0, "channelMsg must be 32-bit aligned");

  struct CtrlMsg {
    enum Op : uint8_t {
      // Endpoint --> Engine
      kInstallCtx = 0,
      kInstallFlow,
    };
    Op opcode;
    PeerID peer_id;

    union CtrlMeta meta;
    // Wakeup handler
    PollCtx* poll_ctx;
  };
  static_assert(sizeof(CtrlMsg) % 4 == 0, "channelMsg must be 32-bit aligned");

  Channel() {
    tx_cmdq_ = create_ring(sizeof(Msg), kChannelSize);
    rx_cmdq_ = create_ring(sizeof(Msg), kChannelSize);
    ctrl_cmdq_ = create_ring(sizeof(CtrlMsg), kChannelSize);
  }

  ~Channel() {
    free(tx_cmdq_);
    free(rx_cmdq_);
    free(ctrl_cmdq_);
  }

  jring_t* tx_cmdq_;
  jring_t* rx_cmdq_;
  jring_t* ctrl_cmdq_;
};

class UcclFlow;
class UcclRDMAEngine;
class RDMAEndpoint;

/**
 * @brief Buffer pool for work request extension.
 */
class WrExBuffPool : public BuffPool {
  static constexpr size_t kWrSize = sizeof(struct wr_ex);
  static constexpr uint32_t kNumWr = 4096;
  static_assert((kNumWr & (kNumWr - 1)) == 0, "kNumWr must be power of 2");

 public:
  WrExBuffPool()
      : BuffPool(kNumWr, kWrSize, nullptr, [](uint64_t buff) {
          struct wr_ex* wr_ex = reinterpret_cast<struct wr_ex*>(buff);
          auto wr = &wr_ex->wr;
          wr->sg_list = &wr_ex->sge;
          wr->num_sge = 1;
          wr->next = nullptr;
          wr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
          wr->wr_id = 0;
        }) {}

  ~WrExBuffPool() = default;
};

class IMMData {
 public:
  // HINT: Indicates whether the last chunk of a message.
  // CSN:  Chunk Sequence Number.
  // RID:  Request ID (4 bits).
  // FID:  Flow Index (11 bits).
  // High-----------------32bit------------------Low
  //  | HINT |  RESERVED  |  CSN  |  RID  |   FID   |
  //    1bit      8bit       8bit    4bit    11bit
  // Bit layout:
  // [31]      HINT (1 bit)
  // [30:23]   RESERVED (8 bits)
  // [22:15]   CSN (8 bits, or UINT_CSN_BIT)
  // [14:11]   RID (4 bits)
  // [10:0]    FID (11 bits)
  constexpr static int kFID = 0;
  constexpr static int kRID = 11;
  constexpr static int kCSN = 15;
  constexpr static int kRESERVED = kCSN + UINT_CSN_BIT;
  constexpr static int kHINT = kRESERVED + 8;

  IMMData(uint32_t imm_data) : imm_data_(imm_data) {}

  inline uint32_t GetHINT(void) { return (imm_data_ >> kHINT) & 0x1; }

  inline uint32_t GetRESERVED(void) { return (imm_data_ >> kRESERVED) & 0xFF; }

  inline uint32_t GetCSN(void) { return (imm_data_ >> kCSN) & UINT_CSN_MASK; }

  inline uint32_t GetRID(void) { return (imm_data_ >> kRID) & 0xF; }

  inline uint32_t GetFID(void) { return (imm_data_ >> kFID) & 0x7FF; }

  inline void SetHINT(uint32_t hint) { imm_data_ |= (hint & 0x1) << kHINT; }

  inline void SetRESERVED(uint32_t reserved) {
    imm_data_ |= (reserved & 0xFF) << kRESERVED;
  }

  inline void SetCSN(uint32_t csn) {
    imm_data_ |= (csn & UINT_CSN_MASK) << kCSN;
  }

  inline void SetRID(uint32_t rid) { imm_data_ |= (rid & 0xF) << kRID; }

  inline void SetFID(uint32_t fid) { imm_data_ |= (fid & 0x7FF) << kFID; }

  inline uint32_t GetImmData(void) { return imm_data_; }

 protected:
  uint32_t imm_data_;
};

class IMMDataEQDS : public IMMData {
 public:
  // PULL_TARGET: Target for pulling data.
  // High-----------------32bit------------------Low
  //  | HINT | PULL_TARGET |  CSN  |  RID  |  FID  |
  //    1bit      8bit        8bit    7bit    8bit
  constexpr static int kPULL_TARGET = kRESERVED;

  IMMDataEQDS(uint32_t imm_data) : IMMData(imm_data) {}

  inline uint32_t GetTarget(void) { return (imm_data_ >> kPULL_TARGET) & 0xFF; }

  inline void SetTarget(uint32_t pull_target) {
    imm_data_ |= (pull_target & 0xFF) << kPULL_TARGET;
  }
};

/**
 * @brief RDMA context for a remote peer on an engine, which is produced by
 * RDMAFactory. It contains:
 *   - (Data path QP): Multiple UC/RC QPs and a shared CQ. All data path QPs
 * share the same SRQ.
 *   - (Ctrl QP): A high-priority QP for control messages and a dedicated CQ,
 * PD, and MR.
 */
class RDMAContext {
 private:
  // Pointer to the IO context of each engine.
  SharedIOContext* io_ctx_;

  // Offset of the engine this context belongs to. 0, 1, ... kNumEngine - 1.
  uint32_t engine_offset_;

  // Number of flows installed on this context.
  uint32_t nr_flows_ = 0;

  // Remote RDMA context.
  struct RemoteRDMAContext remote_ctx_;

  // Timer manager for RTO.
  TimerManager* rto_;

  // Track installed flows.
  // When a flow is installed, it should be added to this table.
  // Flows are split into two tables: sender and receiver.
  void* sender_flow_tbl_[MAX_FLOW] = {};
  void* receiver_flow_tbl_[MAX_FLOW] = {};

  // Protection domain for all RDMA resources.
  struct ibv_pd* pd_ = nullptr;

  // QPs for data transfer based on UC or RC.
  std::vector<QPWrapper> dp_qps_;

  // # of QPs used per engine for data transfer.
  uint32_t port_entropy_;

  // Chunk size for data transfer.
  uint32_t chunk_size_;

  // Data path QPN to index mapping.
  std::unordered_map<uint32_t, int> qpn2idx_;

  // (high-priority) QP for credit messages (e.g., pull of EQDS).
  struct ibv_qp* credit_qp_;

  // (Engine only) Dedicated CQ for credit messages.
  struct ibv_cq_ex* engine_credit_cq_ex_;
  // (Engine only) Memory region for credit messages.
  struct ibv_mr* engine_credit_mr_;
  // (Pacer only) Dedicated CQ for credit messages.
  struct ibv_cq_ex* pacer_credit_cq_ex_;
  // (Pacer only) Memory region for credit messages.
  struct ibv_mr* pacer_credit_mr_;

  // Global timing wheel for all data path QPs.
  TimingWheel wheel_;

  // RDMA device context per device.
  struct ibv_context* context_;

  // MTU of this device in bytes.
  uint32_t mtu_bytes_;

  // GID Index of the device
  int gid_idx_;

  // Whether RoCE or IB.
  bool is_roce_;

  // Link Speed of the device
  double link_speed_ = 0;

  // (Engine) Buffer pool for credit chunks.
  std::optional<eqds::CreditChunkBuffPool> engine_credit_chunk_pool_;

  // (Pacer) Buffer pool for credit chunks.
  std::optional<eqds::CreditChunkBuffPool> pacer_credit_chunk_pool_;

  // Buffer pool for work request extension items.
  std::optional<WrExBuffPool> wr_ex_pool_;

  // Pre-allocated WQEs/SGEs for receiving credits.
  struct RecvWRs credit_recv_wrs_;

  double nic_ts_ratio_;
  double nic_ts_offset_;

  uint32_t consecutive_same_choice_bytes_ = 0;
  uint32_t last_qp_choice_ = 0;

  uint32_t* engine_unacked_bytes_;

  LIST_HEAD(ack_list_);

  inline bool is_roce() { return is_roce_; }

  inline bool rc_mode() { return io_ctx_->rc_mode_; }

 public:
  // 256-bit SACK bitmask => we can track up to 256 packets
  static constexpr std::size_t kReassemblyMaxSeqnoDistance = kSackBitmapSize;

  // EQDS.
  eqds::EQDS* eqds_;

  inline void add_sender_flow(void* flow, uint32_t flow_id) {
    DCHECK(sender_flow_tbl_[flow_id] == nullptr) << flow_id;
    sender_flow_tbl_[flow_id] = flow;
    nr_flows_++;
  }

  inline void add_receiver_flow(void* flow, uint32_t flow_id) {
    DCHECK(receiver_flow_tbl_[flow_id] == nullptr) << flow_id;
    receiver_flow_tbl_[flow_id] = flow;
    nr_flows_++;
  }

  eqds::PacerCreditQPWrapper pc_qpw_;

  // Try to arm a timer for the given flow. If the timer is already armed, do
  // nothing.
  void arm_timer_for_flow(void* context);

  // Try to rearm a timer for the given flow. If the timer is not armed, arm
  // it. If the timer is already armed, rearm it.
  void rearm_timer_for_flow(void* context);

  // Mark the flow as timeout.
  void mark_flow_timeout(void* context);

  // Disarm the timer for the given flow.
  void disarm_timer_for_flow(void* context);

  inline void update_clock(double ratio, double offset) {
    nic_ts_ratio_ = ratio;
    nic_ts_offset_ = offset;
  }

  // Convert NIC clock to host clock (TSC).
  inline uint64_t convert_nic_to_host(uint64_t nic_clock) {
    return nic_ts_ratio_ * nic_clock + nic_ts_offset_;
  }

  inline bool can_use_last_choice(uint32_t msize) {
    bool cond1 = msize <= kMAXUseCacheQPSize;
    bool cond2 = consecutive_same_choice_bytes_ + msize <=
                 kMAXConsecutiveSameChoiceBytes;
    if (cond1 && cond2) {
      consecutive_same_choice_bytes_ += msize;
      return true;
    }
    consecutive_same_choice_bytes_ = 0;
    return false;
  }

  // Select a QP index in a round-robin manner.
  inline uint32_t select_qpidx_rr(void) {
    static uint32_t next_qp_idx = 0;
    return next_qp_idx++ % port_entropy_;
  }

  // Select a QP index randomly.
  inline uint32_t select_qpidx_rand() {
    static thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<uint32_t> distribution(0, port_entropy_ - 1);
    return distribution(generator);
  }

  // Select a QP index in a power-of-two manner.
  uint32_t select_qpidx_pot(uint32_t msize, void* subflow_context);

  /**
   * @brief Poll the completion queue for the Credit QP.
   * SQ and RQ use different completion queues.
   */
  int poll_credit_cq(void);

  /**
   * @brief Check if we need to post enough recv WQEs to the Credit QP.
   * @param force Force to post WQEs.
   */
  void check_credit_rq(bool force = false);

  /**
   * @brief Retransmit a chunk for the given subUcclFlow.
   *
   * @param subflow
   * @param wr_ex
   * @return true Retransmission is successful.
   * @return false Retransmission is failed due to maximum outstanding
   * retransmission chunks or lack of credits (Receiver-driven CC).
   */
  bool try_retransmit_chunk(SubUcclFlow* subflow, struct wr_ex* wr_ex);

  /**
   * @brief Try to drain the retransmission queue for the given subUcclFlow.
   * @param subflow
   */
  void drain_rtx_queue(SubUcclFlow* subflow);

  void uc_post_acks();

  template <typename T>
  void uc_rx_chunk(T* wc_or_cq_ex);

  template <typename T>
  void uc_rx_ack(T* wc_or_cq_ex, UcclSackHdr* ucclsackh);

  template <typename T>
  void uc_rx_rtx_chunk(T* wc_or_cq_ex, uint64_t chunk_addr);

  /**
   * @brief Receive a credit.
   * @param pkt_addr The position of the Credit packet in the Credit chunk.
   */
  void rx_credit(uint64_t pkt_addr);

  /**
   * @brief Supply buffers for receiving data.
   * @param ureq
   * @return 0 success
   * @return -1 fail
   */
  int supply_rx_buff(struct ucclRequest* ureq);

  /**
   * @brief Transmit a message.
   * @param ureq
   * @return true message is transmitted successfully.
   * @return false message is not transmitted successfully.
   */
  bool tx_message(struct ucclRequest* ureq) {
    if constexpr (kReceiverCCA != RECEIVER_CCA_NONE) {
      return receiverCC_tx_message(ureq);
    } else {
      if (ureq->type == ReqRead)
        return senderCC_tx_read(ureq);
      else if (ureq->type == ReqWrite)
        return senderCC_tx_write(ureq);
      return senderCC_tx_message(ureq);
    }
  }
  bool receiverCC_tx_message(struct ucclRequest* ureq);
  bool senderCC_tx_message(struct ucclRequest* ureq);
  bool senderCC_tx_read(struct ucclRequest* ureq);
  bool senderCC_tx_write(struct ucclRequest* ureq);

  virtual uint32_t EventOnSelectPath(SubUcclFlow* subflow,
                                     uint32_t chunk_size) = 0;

  virtual uint32_t EventOnChunkSize(SubUcclFlow* subflow,
                                    uint32_t remaining_bytes) = 0;

  virtual bool EventOnQueueData(SubUcclFlow* subflow, struct wr_ex* wr_ex,
                                uint32_t full_chunk_size, uint64_t now) = 0;

  virtual bool EventOnTxRTXData(SubUcclFlow* subflow, struct wr_ex* wr_ex) = 0;

  virtual void EventOnRxRTXData(SubUcclFlow* subflow, IMMData* imm_data) = 0;

  virtual void EventOnRxData(SubUcclFlow* subflow, IMMData* imm_data) = 0;

  virtual void EventOnRxNACK(SubUcclFlow* subflow, UcclSackHdr* sack_hdr) = 0;

  virtual void EventOnRxACK(SubUcclFlow* subflow, UcclSackHdr* sack_hdr) = 0;

  virtual void EventOnRxCredit(SubUcclFlow* subflow,
                               eqds::PullQuanta pullno) = 0;

  /**
   * @brief Craft an ACK for a subUcclFlow using the given WR index.
   *
   * @param chunk_addr
   * @param idx_in_chunk
   */
  void craft_ack(SubUcclFlow* subflow, uint64_t chunk_addr, int idx_in_chunk);

  /**
   * @brief Flush all ACKs in the batch.
   *
   * @param num_ack
   * @param chunk_addr
   */
  void try_post_acks(int num_ack, uint64_t chunk_addr, bool force);

  /**
   * @brief Transmit a batch of chunks queued in the timing wheel.
   */
  void burst_timing_wheel(void);

  /**
   * @brief Try to update the CSN for the given data path QP.
   * @param qpw
   */
  void try_update_csn(SubUcclFlow* subflow);

  /**
   * @brief Retransmit chunks for the given subUcclFlow.
   * @param rto triggered by RTO or not.
   */
  void __retransmit_for_flow(void* context, bool rto);
  inline void fast_retransmit_for_flow(void* context) {
    if (is_roce() || kTestLoss) {
      __retransmit_for_flow(context, false);
    }
  }
  inline void rto_retransmit_for_flow(void* context) {
    if (is_roce() || kTestLoss) {
      __retransmit_for_flow(context, true);
    }
  }

  template <typename T>
  void rc_rx_ack(T* wc_or_cq_ex);

  template <typename T>
  void rc_rx_chunk(T* wc_or_cq_ex);

  std::string to_string();

  RDMAContext(TimerManager* rto, uint32_t* ob, eqds::EQDS* eqds, int dev,
              uint32_t engine_offset, union CtrlMeta meta,
              SharedIOContext* io_ctx);

  ~RDMAContext(void);

  friend class RDMAFactory;

  friend class UcclRDMAEngine;

  friend class TXTracking;

  friend class TimelyRDMAContext;

  friend class SwiftRDMAContext;

  friend class EQDSRDMAContext;
};

class EQDSRDMAContext : public RDMAContext {
 public:
  using RDMAContext::RDMAContext;

  uint32_t EventOnSelectPath(SubUcclFlow* subflow,
                             uint32_t chunk_size) override {
    return select_qpidx_pot(chunk_size, subflow);
  }

  uint32_t EventOnChunkSize(SubUcclFlow* subflow,
                            uint32_t remaining_bytes) override {
    uint32_t chunk_size;

    if constexpr (kSenderCCA != SENDER_CCA_NONE) {
      if (subflow->unacked_bytes_ >= subflow->pcb.timely_cc.get_wnd()) return 0;
    }

    chunk_size = std::min(chunk_size_, subflow->pcb.eqds_cc.credit());
    chunk_size = std::min(chunk_size, remaining_bytes);
    if (!subflow->pcb.eqds_cc.spend_credit(chunk_size)) chunk_size = 0;

    return chunk_size;
  }

  bool EventOnQueueData(SubUcclFlow* subflow, struct wr_ex* wr_ex,
                        uint32_t full_chunk_size, uint64_t now) override {
    return false;
  }

  void EventOnRxData(SubUcclFlow* subflow, IMMData* imm_data) override {
    auto* imm = reinterpret_cast<IMMDataEQDS*>(imm_data);
    if (subflow->pcb.eqds_cc.handle_pull_target(imm->GetTarget())) {
      VLOG(5) << "Request pull for new target: " << (uint32_t)imm->GetTarget();
      eqds_->request_pull(&subflow->pcb.eqds_cc);
    }
  }

  bool EventOnTxRTXData(SubUcclFlow* subflow, struct wr_ex* wr_ex) override {
    // We can't retransmit the chunk unless we have enough credits.
    auto permitted_bytes = subflow->pcb.eqds_cc.credit();
    if (permitted_bytes < wr_ex->sge.length ||
        !subflow->pcb.eqds_cc.spend_credit(wr_ex->sge.length)) {
      subflow->in_rtx = true;
      VLOG(5) << "Cannot retransmit chunk due to insufficient credits";
      return false;
    }
    // Re-compute pull target.
    auto pull_target =
        subflow->pcb.eqds_cc.compute_pull_target(subflow, wr_ex->sge.length);
    auto imm_data = IMMDataEQDS(ntohl(wr_ex->wr.imm_data));
    imm_data.SetTarget(pull_target);
    wr_ex->wr.imm_data = htonl(imm_data.GetImmData());
    return true;
  }

  void EventOnRxRTXData(SubUcclFlow* subflow, IMMData* imm_data) override {
    EventOnRxData(subflow, imm_data);
  }

  void EventOnRxACK(SubUcclFlow* subflow, UcclSackHdr* sack_hdr) override {
    // After receiving a valid ACK, we can exit the pending retransmission
    // state.
    subflow->in_rtx = false;
    subflow->pcb.eqds_cc.stop_speculating();
  }

  void EventOnRxNACK(SubUcclFlow* subflow, UcclSackHdr* sack_hdr) override {}

  void EventOnRxCredit(SubUcclFlow* subflow, eqds::PullQuanta pullno) override {
    subflow->pcb.eqds_cc.stop_speculating();

    VLOG(5) << "Received credit: " << (uint32_t)pullno;
    if (subflow->pcb.eqds_cc.handle_pull(pullno)) {
      // TODO: trigger transmission for this subflow immediately.
    }

    if (subflow->in_rtx) {
      // We have pending retransmission chunks due to lack of credits.
      // Try to drain the retransmission queue.
      drain_rtx_queue(subflow);
    }
  }
};

class SwiftRDMAContext : public RDMAContext {
 public:
  using RDMAContext::RDMAContext;

  uint32_t EventOnSelectPath(SubUcclFlow* subflow,
                             uint32_t chunk_size) override {
    return select_qpidx_pot(chunk_size, subflow);
  }

  uint32_t EventOnChunkSize(SubUcclFlow* subflow,
                            uint32_t remaining_bytes) override {
    if (remaining_bytes <= chunk_size_) return remaining_bytes;

    auto hard_budget = (is_roce() ? kMaxUnAckedBytesPerEngineHighForRoCE
                                  : kMaxUnAckedBytesPerEngineHighForIB) -
                       *engine_unacked_bytes_;
    auto soft_budget = (is_roce() ? kMaxUnAckedBytesPerEngineLowForRoCE
                                  : kMaxUnAckedBytesPerEngineLowForIB) -
                       *engine_unacked_bytes_;
    auto flow_budget = kMaxUnAckedBytesPerFlow - subflow->unacked_bytes_;

    auto cc_budget = subflow->pcb.swift_cc.get_wnd() - subflow->unacked_bytes_;

    // Enforce swift congestion control window.
    auto ready_bytes = std::min(remaining_bytes, cc_budget);

    // Chunking to CHUNK_SIZE.
    ready_bytes = std::min(ready_bytes, chunk_size_);

    // First, check if we have touched the hard budget.
    if (ready_bytes > hard_budget) return 0;

    // Second, check if we have touched the soft budget.
    // If we havent touched our per-flow budget, we can ignore the soft budget.
    if (ready_bytes <= soft_budget || ready_bytes <= flow_budget)
      return ready_bytes;

    return 0;
  }

  bool EventOnQueueData(SubUcclFlow* subflow, struct wr_ex* wr_ex,
                        uint32_t full_chunk_size, uint64_t now) override {
    return false;
  }

  void EventOnRxData(SubUcclFlow* subflow, IMMData* imm_data) override {}

  bool EventOnTxRTXData(SubUcclFlow* subflow, struct wr_ex* wr_ex) override {
    return true;
  }

  void EventOnRxRTXData(SubUcclFlow* subflow, IMMData* imm_data) override {}

  void EventOnRxACK(SubUcclFlow* subflow, UcclSackHdr* sack_hdr) override {}

  void EventOnRxNACK(SubUcclFlow* subflow, UcclSackHdr* sack_hdr) override {}

  void EventOnRxCredit(SubUcclFlow* subflow, eqds::PullQuanta pullno) override {
  }
};

class TimelyRDMAContext : public RDMAContext {
 public:
  using RDMAContext::RDMAContext;

  uint32_t EventOnSelectPath(SubUcclFlow* subflow,
                             uint32_t chunk_size) override {
    return select_qpidx_pot(chunk_size, subflow);
  }

  uint32_t EventOnChunkSize(SubUcclFlow* subflow,
                            uint32_t remaining_bytes) override {
    auto ready_bytes = std::min(remaining_bytes, chunk_size_);

    if (*engine_unacked_bytes_ + ready_bytes >
        (is_roce() ? kMaxUnAckedBytesPerEngineHighForRoCE
                   : kMaxUnAckedBytesPerEngineHighForIB))
      return 0;

    if (*engine_unacked_bytes_ + ready_bytes <=
            (is_roce() ? kMaxUnAckedBytesPerEngineLowForRoCE
                       : kMaxUnAckedBytesPerEngineLowForIB) ||
        subflow->unacked_bytes_ + ready_bytes <= kMaxUnAckedBytesPerFlow) {
      return ready_bytes;
    }
    return 0;
  }

  bool EventOnQueueData(SubUcclFlow* subflow, struct wr_ex* wr_ex,
                        uint32_t full_chunk_size, uint64_t now) override {
    return wheel_.queue_on_timing_wheel(
        subflow->pcb.timely_cc.rate_,
        &subflow->pcb.timely_cc.prev_desired_tx_tsc_, now, wr_ex,
        full_chunk_size, subflow->in_wheel_cnt_ == 0);
  }

  void EventOnRxData(SubUcclFlow* subflow, IMMData* imm_data) override {}

  bool EventOnTxRTXData(SubUcclFlow* subflow, struct wr_ex* wr_ex) override {
    return true;
  }

  void EventOnRxRTXData(SubUcclFlow* subflow, IMMData* imm_data) override {}

  void EventOnRxACK(SubUcclFlow* subflow, UcclSackHdr* sack_hdr) override {}

  void EventOnRxNACK(SubUcclFlow* subflow, UcclSackHdr* sack_hdr) override {}

  void EventOnRxCredit(SubUcclFlow* subflow, eqds::PullQuanta pullno) override {
  }
};

/**
 * @brief Class `UcclRDMAEngine' abstracts the main Uccl engine which supports
 * RDMA. This engine contains all the functionality need to be run by the
 * stack's threads.
 */
class UcclRDMAEngine {
 public:
  TimerManager rto_tm_;
  uint32_t engine_outstanding_bytes_ = 0;

  UcclRDMAEngine() = delete;
  UcclRDMAEngine(UcclRDMAEngine const&) = delete;

  /**
   * @brief Construct a new UcclRDMAEngine object.
   * @param dev           Device index.
   * @param engine_id     Engine index.
   * @param channel       Uccl channel the engine will be responsible for.
   * For now, we assume an engine is responsible for a single channel, but
   * future it may be responsible for multiple channels.
   */
  UcclRDMAEngine(int dev, int engine_id, Channel* channel, eqds::EQDS* eqds)
      : engine_idx_(engine_id),
        dev_(dev),
        channel_(channel),
        eqds_(eqds),
        last_periodic_tsc_(rdtsc()),
        last_sync_clock_tsc_(rdtsc()),
        rto_tm_(kRTOUSec),
        io_ctx_(dev),
        kSlowTimerIntervalTsc_(us_to_cycles(kSlowTimerIntervalUs, freq_ghz)) {
    auto context = RDMAFactory::get_factory_dev(dev_)->context;
    is_no_rto_ =
        (RDMAFactory::is_roce(dev_) || kTestLoss)
            ? false
            : true;  // Infiniband is lossless, disable RTO even for UC.;
    struct ibv_values_ex values;
    values.comp_mask = IBV_VALUES_MASK_RAW_CLOCK;
    ibv_query_rt_values_ex(context, &values);
    auto nic_clock = values.raw_clock.tv_sec * 1e9 + values.raw_clock.tv_nsec;
    last_nic_clock_ = nic_clock;
    last_host_clock_ = rdtsc();
  }

  /**
   * @brief Handling async send requests from Endpoint for all flows.
   */
  void handle_tx_work(void);

  /**
   * @brief Handling async recv requests from Endpoint for all flows.
   */
  void handle_rx_work(void);

  /**
   * @brief Handling all completion events for all RDMAContexts, including:
   * High-priority completion events from all Ctrl QPs.
   * Datapath completion events from all data path QPs.
   */
  void uc_handle_completion(void);

  void rc_handle_completion(void);

  inline void handle_completion(void) {
    if (!io_ctx_.is_rc_mode())
      uc_handle_completion();
    else
      rc_handle_completion();
  }

  /**
   * @brief Handle all timing wheel events for all RDMAContexts.
   *
   */
  void handle_timing_wheel(void);

  /**
   * @brief This is the main event cycle of the Uccl engine.
   * It is called by a separate thread running the Uccl engine.
   * On each iteration, the engine processes incoming packets in the RX
   * queue and enqueued messages in all channels that it is responsible
   * for. This method is not thread-safe.
   */
  void run();

  /**
   * @brief Method to perform periodic processing. This is called by the
   * main engine cycle (see method `Run`).
   */
  void periodic_process();

  /**
   * @brief Install a new RDMA context on the engine.
   * @param ctrl_work
   */
  void handle_install_ctx_on_engine(Channel::CtrlMsg& ctrl_work);

  /**
   * @brief Install a new UcclFlow on the engine.
   *
   * @param ctrl_work
   */
  void handle_install_flow_on_engine(Channel::CtrlMsg& ctrl_work);

  inline bool need_sync(uint64_t now) {
    return now - last_sync_clock_tsc_ >
           ns_to_cycles(kSyncClockIntervalNS, freq_ghz);
  }

  /**
   * @brief Synchronize the clock between host and NIC every
   * kSyncClockIntervalNS according to Flor[OSDI'23]
   */
  inline void handle_clock_synchronization(void) {
    auto host_clock = rdtsc();
    if (need_sync(host_clock)) {
      auto context = RDMAFactory::get_factory_dev(dev_)->context;
      struct ibv_values_ex values;
      values.comp_mask = IBV_VALUES_MASK_RAW_CLOCK;
      ibv_query_rt_values_ex(context, &values);

      auto nic_clock = values.raw_clock.tv_sec * 1e9 + values.raw_clock.tv_nsec;

      // Update ratio and offset
      nic_ts_ratio_ = (1.0 * (int64_t)host_clock - (int64_t)last_host_clock_) /
                      ((int64_t)nic_clock - (int64_t)last_nic_clock_);
      nic_ts_offset_ = host_clock - nic_ts_ratio_ * nic_clock;

      last_sync_clock_tsc_ = host_clock;
    }
  }

  inline bool is_no_rto() { return is_no_rto_; }

  // Called by application to shutdown the engine. App will need to join
  // the engine thread.
  inline void shutdown() { shutdown_ = true; }

  void release();

  std::string status_to_string();

 protected:
  /**
   * @brief Iterate throught the list of flows, check and handle RTOs.
   */
  void handle_rto();

  /**
   * @brief This method polls active channels for all control plane
   * requests and processes them. It is called periodically.
   */
  void process_ctl_reqs();

 private:
  // Device index
  int dev_;
  // Engine index
  uint32_t engine_idx_;
  // RDMAContext map
  std::unordered_map<PeerID, RDMAContext*> rdma_ctx_map_;
  // Control plane channel with RDMAEndpoint.
  Channel* channel_;

  eqds::EQDS* eqds_;

  // Pending rx work due to no available request.
  std::deque<std::pair<RDMAContext*, struct ucclRequest*>> pending_rx_works_;
  // Pending tx work due to reaching the max outstanding bytes.
  std::deque<std::pair<RDMAContext*, struct ucclRequest*>> pending_tx_works_;

  std::deque<Channel::CtrlMsg> pending_install_flow_works_;

  SharedIOContext io_ctx_;

  // Timestamp of last periodic process execution.
  uint64_t last_periodic_tsc_;
  // Slow timer interval in TSC.
  uint64_t kSlowTimerIntervalTsc_;

  // Timestamp of last clock synchronization.
  uint64_t last_sync_clock_tsc_;
  uint64_t last_host_clock_;
  uint64_t last_nic_clock_;
  // RTO disabled
  bool is_no_rto_;

  double nic_ts_ratio_ = 0;
  double nic_ts_offset_ = 0;

  // Whether shutdown is requested.
  std::atomic<bool> shutdown_{false};

  friend class UcclFlow;
};

/**
 * @brief A peer is identified by its IP address and device index and GPU index.
 */
struct UcclPeer {
  std::string remote_ip;
  int remote_dev;
  int remote_gpuidx;
};

static bool operator==(UcclPeer const& lhs, UcclPeer const& rhs) {
  return lhs.remote_ip == rhs.remote_ip && lhs.remote_dev == rhs.remote_dev &&
         lhs.remote_gpuidx == rhs.remote_gpuidx;
}

struct UcclPeerHash {
  std::size_t operator()(UcclPeer const& peer) const {
    return std::hash<std::string>()(peer.remote_ip) ^
           std::hash<int>()(peer.remote_dev) ^
           std::hash<int>()(peer.remote_gpuidx);
  }
};

struct PeerInfo {
  PeerID peer_id;
  ibv_gid remote_gid;
  struct ibv_port_attr remote_port_attr;
  // -1: peer id is allocated by accept()
  // 0:  peer id is allocated by connect()
  // 1:  rdma context is ready.
  int ready;
};

/**
 * @class RDMAEndpoint
 * @brief application-facing interface, communicating with `UcclRDMAEngine'
 * through `Channel'. Each connection is identified by a unique flow_id, and
 * uses multiple src+dst port combinations to leverage multiple paths. Under the
 * hood, we leverage TCP to boostrap our connections. We do not consider
 * multi-tenancy for now, assuming this endpoint exclusively uses the NIC and
 * its all queues. Note that all IB devices are managed by a single
 * RDMAEndpoint.
 */
class RDMAEndpoint {
  constexpr static uint32_t kMaxInflightMsg = 1024 * 256;
  constexpr static uint32_t kStatsTimerIntervalSec = 2;
  constexpr static uint32_t RC_MAGIC = 0x12345678;

  std::shared_ptr<RDMAFactory> rdma_ctl_;

  // RDMA devices.
  int num_devices_;

  int num_engines_per_dev_;
  // Per-engine communication channel
  std::vector<Channel*> channel_vec_;
  std::vector<std::unique_ptr<UcclRDMAEngine>> engine_vec_;
  std::unordered_map<int, std::unique_ptr<UcclRDMAEngine>>
      engine_id_to_engine_map_;
  std::mutex engine_map_mu_;
  std::vector<std::unique_ptr<std::thread>> engine_th_vec_;
  std::mutex engine_th_mu_;

  // Number of outstanding messages for each engine.
  std::vector<std::unique_ptr<std::atomic<uint32_t>>> engine_load_vec_;

  // Receiver-driven congestion control.
  std::vector<eqds::EQDS*> eqds_;

  SharedPool<PollCtx*, true>* ctx_pool_;
  uint8_t* ctx_pool_buf_;

  std::vector<uint16_t> p2p_listen_ports_;
  std::vector<int> p2p_listen_fds_;

  std::mutex fd_vec_mu_;
  // Mapping from unique (within this engine) flow_id to the boostrap fd.
  std::vector<int> fd_vec_;

  // Peer map for connecting/accepting
  std::vector<std::unordered_map<UcclPeer, PeerInfo, UcclPeerHash>> peer_map_;
  std::vector<std::unique_ptr<std::mutex>> peer_map_mu_;

  std::atomic<PeerID> next_peer_id_ = 1;

  std::vector<std::vector<Spin>> flow_id_spin_;
  std::vector<std::vector<FlowID>> next_flow_id_;

  std::vector<std::vector<UcclFlow*>> active_flows_vec_;
  std::vector<Spin> active_flows_spin_;

  int ib_relaxed_ordering_enabled_;

 public:
  RDMAEndpoint(int num_engines_per_dev);

  ~RDMAEndpoint();

  uint32_t get_num_devices() { return num_devices_; }

  int get_best_dev_idx(int gpu_idx) {
    return RDMAFactory::get_best_dev_idx(gpu_idx);
  }

  inline uint16_t get_p2p_listen_port(int dev) {
    CHECK(p2p_listen_ports_[dev] != 0)
        << "Error: p2p_listen_ports_[" << dev << "] is not set.";
    return p2p_listen_ports_[dev];
  }

  inline int get_p2p_listen_fd(int dev) {
    CHECK(p2p_listen_fds_[dev] >= 0)
        << "Error: p2p_listen_fds_[" << dev << "] is not set.";
    return p2p_listen_fds_[dev];
  }

  inline std::string get_p2p_listen_ip(int dev) {
    auto factory_dev = RDMAFactory::get_factory_dev(dev);
    CHECK(factory_dev) << "get_p2p_listen_ip: get_factory_dev()";
    return factory_dev->local_ip_str;
  }

  void initialize_resources(int total_num_engines);

  void cleanup_resources();

  bool initialize_engine_by_dev(int dev, bool enable_p2p_listen);

  /// For testing easily.
  ConnID test_uccl_connect(int dev, int gpu, int remote_dev, int remote_gpu,
                           std::string remote_ip, int remote_port) {
    return uccl_connect(dev, gpu, remote_dev, remote_gpu, remote_ip,
                        remote_port);
  }
  ConnID test_uccl_accept(int dev, int gpu, std::string& remote_ip,
                          int* remote_dev, int* remote_gpuidx) {
    return uccl_accept(dev, p2p_listen_fds_[dev], gpu, remote_ip, remote_dev,
                       remote_gpuidx);
  }
  /// For testing easily.

  // Connect to a remote peer <remote_ip, remote_dev> with the given dev, who
  // is listening on the given listen_port. This function is thread-safe.
  ConnID uccl_connect(int dev, int local_gpuidx, int remote_dev,
                      int remote_gpuidx, std::string remote_ip,
                      uint16_t remote_port);

  // Accept a connection using the given listen_fd. <remote_ip, remote_dev> is
  // returned. This function is thread-safe.
  ConnID uccl_accept(int dev, int listen_fd, int local_gpuidx,
                     std::string& remote_ip, int* remote_dev,
                     int* remote_gpuidx);

  bool is_local_leader(int ldev, int lgpu, std::string lip, uint16_t lport,
                       int rdev, int rgpu, std::string rip, uint16_t rport) {
    auto lip_int = str_to_ip(lip.c_str());
    auto rip_int = str_to_ip(rip.c_str());

    if (lip_int < rip_int) return true;
    if (lip_int > rip_int) return false;

    if (ldev < rdev) return true;
    if (ldev > rdev) return false;

    if (lgpu < rgpu) return true;
    if (lgpu > rgpu) return false;

    // This is to handle an edge case where both p2p connections run on the same
    // GPU (debug.)
    if (lport < rport) return true;
    if (lport > rport) return false;

    // If all else is equal (very rare), default to false (non-leader)
    return false;
  }

  // Register a memory region.
  int uccl_regmr(UcclFlow* flow, void* data, size_t len, int type,
                 struct Mhandle** mhandle);
  int uccl_regmr(int dev, void* data, size_t len, int type,
                 struct Mhandle** mhandle);
  // Register a DMA-BUF memory region.
  int uccl_regmr_dmabuf(UcclFlow* flow, void* data, size_t len, int type,
                        int offset, int fd, struct Mhandle** mhandle);
  int uccl_regmr_dmabuf(int dev, void* data, size_t len, int type, int offset,
                        int fd, struct Mhandle** mhandle);
  // Deregister a memory region.
  void uccl_deregmr(struct Mhandle* mhandle);

  // Post a buffer to engine for sending data asynchronously.
  int uccl_send_async(UcclFlow* flow, struct Mhandle* mhandle, void const* data,
                      size_t const size, struct ucclRequest* ureq);

  int prepare_fifo_metadata(UcclFlow* flow, struct Mhandle** mhandles,
                            void const* data, size_t size, char* out_buf);

  int uccl_read_async(UcclFlow* flow, Mhandle* local_mh, void* dst, size_t size,
                      FifoItem const& slot_item, ucclRequest* ureq);

  int uccl_write_async(UcclFlow* flow, Mhandle* local_mh, void* src,
                       size_t size, FifoItem const& slot_item,
                       ucclRequest* ureq);

  // Post n buffers to engine for receiving data asynchronously.
  int uccl_recv_async(UcclFlow* flow, struct Mhandle** mhandles, void** data,
                      int* size, int n, struct ucclRequest* ureq);

  // Ensure that all received data is visible to GPU.
  int uccl_flush(UcclFlow* flow, struct Mhandle** mhandles, void** data,
                 int* size, int n, struct ucclRequest* ureq);

  bool uccl_poll_ureq_once(struct ucclRequest* ureq);

  inline bool uccl_poll_ureq(struct ucclRequest* ureq) {
    while (!uccl_poll_ureq_once(ureq)) {
    }
    return true;
  }

  inline bool uccl_wait(PollCtx* ctx) {
    std::unique_lock<std::mutex> lock(ctx->mu);
    ctx->cv.wait(lock, [&ctx] { return ctx->done.load(); });
    fence_and_clean_ctx(ctx);
    return true;
  }

  inline bool uccl_poll_once(PollCtx* ctx) {
    if (!ctx->done.load()) return false;
    fence_and_clean_ctx(ctx);
    return true;
  }

  inline bool uccl_check_once(PollCtx* ctx) { return ctx->done.load(); }

  inline bool uccl_poll(PollCtx* ctx) {
    while (!uccl_poll_once(ctx)) {
    }
    return true;
  }

  inline PeerID alloc_peer_id() { return next_peer_id_.fetch_add(1); }

 private:
  PollCtx* install_ctx_on_engine(uint32_t engine_idx, PeerID peer_id,
                                 union CtrlMeta meta);

  PollCtx* install_flow_on_engine(uint32_t engine_idx, PeerID peer_id,
                                  union CtrlMeta meta);

  void install_ctx_on_engines(int fd, int dev, PeerID peer_id,
                              std::string remote_ip, int remote_dev,
                              int remote_gpuidx);

  std::vector<PollCtx*> install_flow_on_engines(int dev, PeerID peer_id,
                                                FlowID flow_id, UcclFlow* flow,
                                                bool is_send);

  inline void inc_load_on_engine(int engine_id) {
    engine_load_vec_[engine_id]->fetch_add(1);
  }

  inline void dec_load_on_engine(int engine_id) {
    engine_load_vec_[engine_id]->fetch_sub(1);
  }

  // Find a least loaded engine and update the load for the given device.
  inline uint32_t find_least_loaded_engine_idx(int dev);

  inline uint32_t find_pot_load_engine_idx(int dev);

  // Find an engine in a round-robin manner.
  inline uint32_t find_rr_engine_idx(int dev, uint32_t* next_candidate);

  // Find an engine in an oblivious manner.
  inline uint32_t find_oblivious_engine_idx(int dev);

  inline int find_first_engine_idx_on_dev(int dev) {
    return dev * num_engines_per_dev_;
  }

  inline void fence_and_clean_ctx(PollCtx* ctx) {
    // Make the data written by the engine thread visible to the app thread.
    std::ignore =
        std::atomic_load_explicit(&ctx->fence, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);

    ctx_pool_->push(ctx);
  }

  std::thread stats_thread_;
  void stats_thread_fn();
  std::mutex stats_mu_;
  std::condition_variable stats_cv_;
  std::atomic<bool> shutdown_{false};

  friend class UcclFlow;
};

/**
 * @class UcclFlow, a connection between a local and a remote NIC.
 * @brief Class to abstract the components and functionality of a single flow.
 * A flow is a **unidirectional** connection between two NICs, uniquely
 * identified by a TCP-negotiated `FlowID'.
 */
class UcclFlow {
  static constexpr int kFifoMRSize = sizeof(struct RemFifo);
  static constexpr int kFifoCQSize = 4096;

 public:
  std::vector<SubUcclFlow*> sub_flows_;

  UcclFlow(RDMAEndpoint* ep, int dev, PeerID peer_id, FlowID flow_id,
           std::string remote_ip, int remote_dev, bool is_send)
      : ep_(ep),
        dev_(dev),
        peer_id_(peer_id),
        flow_id_(flow_id),
        remote_ip_(remote_ip),
        remote_dev_(remote_dev),
        is_send_(is_send) {
    auto factory_dev = RDMAFactory::get_factory_dev(dev);
    for (int i = 0; i < ucclParamNUM_ENGINES(); i++) {
      sub_flows_.push_back(new SubUcclFlow(flow_id, factory_dev->link_bw));
    }

    memset(&send_comm_, 0, sizeof(send_comm_));
    memset(&recv_comm_, 0, sizeof(recv_comm_));
    int num_devices = ep->get_num_devices();
    // Avoid all flows using the same initial engine offset.

#ifdef DISABLE_CALL_ONCE_STATIC
    std::vector<std::atomic<uint32_t>>* off =
        new std::vector<std::atomic<uint32_t>>(num_devices);
#else
    static std::vector<std::atomic<uint32_t>>* off =
        new std::vector<std::atomic<uint32_t>>(num_devices);
#endif
    next_engine_offset_ = (*off)[dev].fetch_add(1) % ucclParamNUM_ENGINES();
  }

  ~UcclFlow() {
    auto comm_base = is_send_ ? &send_comm_.base : &recv_comm_.base;

    munmap(comm_base->fifo_mr->addr, comm_base->fifo_mr->length);
    ibv_dereg_mr(comm_base->fifo_mr);
    ibv_destroy_qp(comm_base->fifo_qp);

    ibv_destroy_cq(comm_base->flow_cq);

    if (!is_send_) {
      munmap(recv_comm_.gpu_flush_mr->addr, recv_comm_.gpu_flush_mr->length);
      ibv_dereg_mr(recv_comm_.gpu_flush_mr);
      ibv_destroy_qp(recv_comm_.gpu_flush_qp);
    }

    for (int i = 0; i < ucclParamNUM_ENGINES(); i++) {
      delete sub_flows_[i];
    }
  }

  uint32_t create_fifo_and_gpuflush(int bootstrap_fd, int dev) {
    auto comm_base = is_send_ ? &send_comm_.base : &recv_comm_.base;

    auto factory_dev = RDMAFactory::get_factory_dev(dev);

    // Fifo QP.
    util_rdma_create_qp(factory_dev->context, &comm_base->fifo_qp, IBV_QPT_RC,
                        false, false, &comm_base->flow_cq, false, kFifoCQSize,
                        factory_dev->pd, factory_dev->ib_port_num,
                        &comm_base->fifo_mr, nullptr, kFifoMRSize,
                        kMaxSendRecvWR, kMaxSendRecvWR, 1, 1);
    comm_base->fifo =
        reinterpret_cast<struct RemFifo*>(comm_base->fifo_mr->addr);

    // Exchange local QPN for Fifo QP with remote peer.
    char buf[sizeof(uint32_t)];
    auto fifo_lqpn = comm_base->fifo_qp->qp_num;
    memcpy(buf, &fifo_lqpn, sizeof(uint32_t));

    UCCL_INIT_CHECK(
        send_message(bootstrap_fd, buf, sizeof(uint32_t)) == sizeof(uint32_t),
        "uccl_connect: send_message()");

    UCCL_INIT_CHECK(receive_message(bootstrap_fd, buf, sizeof(uint32_t)) ==
                        sizeof(uint32_t),
                    "uccl_connect: receive_message()");

    // Exchange addr and rkey for Fifo MR with remote peer.
    char buf2[sizeof(uint64_t) + sizeof(uint32_t)];
    auto fifo_laddr = reinterpret_cast<uint64_t>(comm_base->fifo_mr->addr);
    auto fifo_lrkey = comm_base->fifo_mr->rkey;
    memcpy(buf2, &fifo_laddr, sizeof(uint64_t));
    memcpy(buf2 + sizeof(uint64_t), &fifo_lrkey, sizeof(uint32_t));

    UCCL_INIT_CHECK(
        send_message(bootstrap_fd, buf2, sizeof(uint64_t) + sizeof(uint32_t)) ==
            sizeof(uint64_t) + sizeof(uint32_t),
        "uccl_connect: send_message()");

    UCCL_INIT_CHECK(receive_message(bootstrap_fd, buf2,
                                    sizeof(uint64_t) + sizeof(uint32_t)) ==
                        sizeof(uint64_t) + sizeof(uint32_t),
                    "uccl_connect: receive_message()");

    comm_base->remote_fifo_addr = *reinterpret_cast<uint64_t*>(buf2);
    comm_base->remote_fifo_rkey =
        *reinterpret_cast<uint32_t*>(buf2 + sizeof(uint64_t));

    // GPU flush QP for receiver.
    if (!is_send_) {
      util_rdma_create_qp(factory_dev->context, &recv_comm_.gpu_flush_qp,
                          IBV_QPT_RC, false, false, &comm_base->flow_cq, true,
                          0, factory_dev->pd, factory_dev->ib_port_num,
                          &recv_comm_.gpu_flush_mr, &recv_comm_.gpu_flush,
                          sizeof(int), kMaxSendRecvWR, kMaxSendRecvWR, kMaxSge,
                          kMaxSge);

      recv_comm_.gpu_flush_sge.addr = (uint64_t)&recv_comm_.gpu_flush;
      recv_comm_.gpu_flush_sge.length = 1;
      recv_comm_.gpu_flush_sge.lkey = recv_comm_.gpu_flush_mr->lkey;

      UCCL_INIT_CHECK(modify_qp_rtr_gpuflush(recv_comm_.gpu_flush_qp, dev) == 0,
                      "Failed to modify GPU flush QP to RTR");
      UCCL_INIT_CHECK(modify_qp_rts(recv_comm_.gpu_flush_qp, true) == 0,
                      "Failed to modify GPU flush QP to RTS");
    }

    return *reinterpret_cast<uint32_t*>(buf);
  }

  void modify_fifo(int bootstrap_fd, int dev,
                   struct RemoteRDMAContext remote_ctx,
                   uint32_t remote_fifo_qpn) {
    auto comm_base = is_send_ ? &send_comm_.base : &recv_comm_.base;

    auto factory_dev = RDMAFactory::get_factory_dev(dev);

    UCCL_INIT_CHECK(modify_qp_rtr(comm_base->fifo_qp, dev, &remote_ctx,
                                  remote_fifo_qpn) == 0,
                    "Failed to modify Fifo QP to RTR");
    UCCL_INIT_CHECK(modify_qp_rts(comm_base->fifo_qp, true) == 0,
                    "Failed to modify Fifo QP to RTS");

    // RC QP
    if constexpr (kRCSize > 0) {
      struct ibv_qp_init_attr qp_init_attr;
      memset(&qp_init_attr, 0, sizeof(qp_init_attr));
      qp_init_attr.qp_context = this;
      qp_init_attr.send_cq = comm_base->flow_cq;
      qp_init_attr.recv_cq = comm_base->flow_cq;
      qp_init_attr.qp_type = IBV_QPT_RC;
      qp_init_attr.cap.max_send_wr = kMaxSendRecvWR;
      qp_init_attr.cap.max_recv_wr = kMaxSendRecvWR;
      qp_init_attr.cap.max_send_sge = kMaxSge;
      qp_init_attr.cap.max_recv_sge = kMaxSge;
      qp_init_attr.cap.max_inline_data = 0;

      struct ibv_qp_attr qpAttr;
      memset(&qpAttr, 0, sizeof(qpAttr));
      qpAttr.qp_state = IBV_QPS_INIT;
      qpAttr.pkey_index = 0;
      qpAttr.port_num = factory_dev->ib_port_num;
      qpAttr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

      comm_base->rc_qp = ibv_create_qp(factory_dev->pd, &qp_init_attr);
      UCCL_INIT_CHECK(comm_base->rc_qp != nullptr, "Failed to create RC QP");

      UCCL_INIT_CHECK(ibv_modify_qp(comm_base->rc_qp, &qpAttr,
                                    IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                                        IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) == 0,
                      "Failed to modify RC QP to INIT");

      char buf[sizeof(uint32_t)];
      // Send QPN to remote peer.
      memcpy(buf, &comm_base->rc_qp->qp_num, sizeof(uint32_t));
      int ret = send_message(bootstrap_fd, buf, sizeof(uint32_t));
      DCHECK(ret == sizeof(uint32_t));

      // Receive QPN from remote peer.
      ret = receive_message(bootstrap_fd, buf, sizeof(uint32_t));
      DCHECK(ret == sizeof(uint32_t));

      auto rc_rqpn = *reinterpret_cast<uint32_t*>(buf);

      UCCL_INIT_CHECK(
          modify_qp_rtr(comm_base->rc_qp, dev, &remote_ctx, rc_rqpn) == 0,
          "Failed to modify RC QP to RTR");

      UCCL_INIT_CHECK(modify_qp_rts(comm_base->rc_qp, true) == 0,
                      "Failed to modify RC QP to RTS");
    }
  }

  inline FlowID flowid() const { return flow_id_; }

  friend class UcclRDMAEngine;

  void release() {}

 private:
  inline int check_need_flush(int* size, int n) {
    // Only flush once using the last non-zero receive
    int last = -1;
    for (int i = 0; i < n; i++)
      if (size[i]) last = i;
    return last;
  }
  /**
   * @brief Post a RDMA READ operation to GPU flush QP. This operation
   * bypasses the UcclEngine.
   */
  void post_flush(struct Mhandle** mhandles, void** data, int* size, int n,
                  uint64_t* flush_done, int last);

  /**
   * @brief Post multiple recv requests to a FIFO queue for remote peer to use
   * RDMA WRITE. These requests are transmitted through the underlyding fifo
   * QP (RC).
   * @param data Array of data buffers.
   * @param size Array of buffer sizes.
   * @param n Number of buffers.
   */
  struct FifoItem* post_fifo(uint32_t engine_idx, void** data, int* size, int n,
                             struct Mhandle** mhandle, struct ibv_send_wr* wr,
                             struct ibv_sge* sge);

  void rc_recv(void* data, int size, struct Mhandle* mhandle,
               struct ibv_send_wr* wr, struct ibv_sge* sge,
               struct ucclRequest* ureq);

  /**
   * @brief Poll the completion queue for the Fifo/GPU flush QP.
   */
  void poll_flow_cq(void);

  /**
   * @brief This function is called by uccl_send_async to check if the
   * receiver has posted buffers.
   */
  bool check_fifo_ready(int* ret_slot, int* ret_nmsgs);

  /**
   * @brief This function is called by uccl_send_async to post multiple send
   * requests to UcclEngine.
   * @param engine_offset The engine offset to use, which is determined by the
   * receiver. 0 <= engine_offset < num_engines_per_dev_.
   */
  void post_multi_send(struct ucclRequest** ureqs, uint32_t engine_offset);

  void post_multi_read(struct ucclRequest** ureqs, uint32_t engine_offset);

  void post_multi_write(struct ucclRequest** ureqs, uint32_t engine_offset);

  void rc_send(struct ucclRequest* ureq);
  void rc_read(struct ucclRequest* ureq);

  inline bool check_room(void) { return outstanding_reqs_ < kMaxReq; }

  inline void inc_outstanding_reqs(void) { outstanding_reqs_++; }

  inline void dec_outstanding_reqs(void) { outstanding_reqs_--; }

  inline uint32_t get_last_rc_size(void) { return last_rc_recv_; }

  inline void set_last_rc_size(uint32_t size) { last_rc_recv_ = size; }

  RDMAEndpoint* ep_;

  PeerID peer_id_;
  FlowID flow_id_;

  int dev_;
  int remote_dev_;
  std::string remote_ip_;

  /**
   * @brief Next engine offset to use for receving.
   */
  uint32_t next_engine_offset_ = 0;

  /**
   * @brief # of CQEs need to be polled for Fifo/GPU flush QP.
   */
  uint32_t flow_cq_cnt_ = 0;

  /**
   * @brief Communication abstraction for sending and receiving.
   * Since data flow is unidirectional in NCCL, we use two different
   * structures.
   */
  union {
    // For connection setup by connect().
    struct SendComm send_comm_;
    // For connection setup by accept().
    struct RecvComm recv_comm_;
  };

  uint32_t outstanding_reqs_ = 0;

  uint32_t last_rc_recv_ = 0;

  // Whether this context is for sending or receiving.
  bool is_send_;

  // Measure the distribution of probed RTT.
  Latency rtt_stats_;
  uint64_t rtt_probe_count_ = 0;

  friend class UcclRDMAEngine;
  friend class RDMAEndpoint;
};

}  // namespace uccl
