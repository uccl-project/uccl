#pragma once

#include "cuda_runtime.h"
#include "driver_types.h"
#include "eqds.h"
#include "scattered_memcpy.cuh"
#include "transport_cc.h"
#include "transport_config.h"
#include "transport_header.h"
#include "util.h"
#include "util_efa.h"
#include "util_endian.h"
#include "util_latency.h"
#include "util_shared_pool.h"
#include "util_timer.h"
#include <glog/logging.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <bitset>
#include <chrono>
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <netdb.h>

namespace uccl {

typedef uint64_t FlowID;

struct ConnID {
  FlowID flow_id;       // Used for UcclEngine to look up UcclFlow.
  uint32_t engine_idx;  // Used for Endpoint to locate the right engine.
  int boostrap_id;      // Used for bootstrap connection with the peer.
};

struct Mhandle {
  struct ibv_mr* mr;
};

struct alignas(64) PollCtx {
  std::mutex mu;
  std::condition_variable cv;
  std::atomic<bool> fence;               // Sync rx/tx memcpy visibility.
  std::atomic<bool> done;                // Sync cv wake-up.
  std::atomic<uint16_t> num_unfinished;  // Number of unfinished requests.
  uint64_t timestamp;                    // Timestamp for request issuing.
  uint32_t engine_idx;                   // Engine index for request issuing.
#ifdef POLLCTX_DEBUG
  FlowID flow_id;   // Flow ID for request issuing.
  uint64_t req_id;  // Tx ID for request issuing.
#endif
  PollCtx() : fence(false), done(false), num_unfinished(0), timestamp(0){};
  ~PollCtx() { clear(); }

  inline void clear() {
    mu.~mutex();
    cv.~condition_variable();
    fence = false;
    done = false;
    num_unfinished = 0;
    timestamp = 0;
  }

  inline void write_barrier() {
    std::atomic_store_explicit(&fence, true, std::memory_order_release);
  }

  inline void read_barrier() {
    std::ignore = std::atomic_load_explicit(&fence, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);
  }
};

class PollCtxPool : public BuffPool {
 public:
  static constexpr uint32_t kPollCtxSize = sizeof(PollCtx);
  static constexpr uint32_t kNumPollCtx = NUM_FRAMES;
  static_assert((kNumPollCtx & (kNumPollCtx - 1)) == 0,
                "kNumPollCtx must be power of 2");

  PollCtxPool() : BuffPool(kNumPollCtx, kPollCtxSize, nullptr) {}

  ~PollCtxPool() = default;
};

int const kMaxIovs = 64;

enum ReqType {
  ReqTx,
  ReqRx,
  ReqRxScattered,
  ReqRxFreePtrs,
  ReqFlush,
};

struct UcclRequest {
  ReqType type;
  int n;
  int send_len = 0;
  int recv_len[kMaxMultiRecv] = {};
  PollCtx* poll_ctx = nullptr;
  void* req_pool = nullptr;
  /* Do not change the order */
  void* iov_addrs[kMaxIovs];
  int iov_lens[kMaxIovs];
  int dst_offsets[kMaxIovs];
  int iov_n;
  int pad;
  void* ptrs[32];
  /***********************/
};
static const uint32_t kIovStart = offsetof(struct UcclRequest, iov_addrs);
static const uint32_t kPtrsStart = offsetof(struct UcclRequest, ptrs);

/**
 * @class Channel
 * @brief A channel is a command queue for application threads to submit rx and
 * tx requests to the UcclFlow. A channel is only served by one UcclFlow, but
 * could be shared by multiple app threads if needed.
 */
class Channel {
  constexpr static uint32_t kChannelSize = 8192;

 public:
  struct Msg {
    enum Op : uint8_t {
      kTx = 0,
      kRx = 1,
      kRxScattered = 2,
      kRxFreePtrs = 3,
    };
    Op opcode;
    uint8_t unused_bytes[3];
    int len;
    int* len_p;
    union {
      void* data;
      UcclRequest* req;
    };
    Mhandle* mhandle;
    FlowID flow_id;
    // A list of FrameDesc bw deser_th and engine_th.
    FrameDesc* deser_msgs;
    // Wakeup handler
    PollCtx* poll_ctx;
    uint64_t reserved;
  };
  const static uint32_t kMsgSize = sizeof(Msg);
  static_assert(kMsgSize % 4 == 0, "Msg must be 32-bit aligned");

  struct CtrlMsg {
    enum Op : uint8_t {
      kInstallFlow = 0,
    };
    Op opcode;
    FlowID flow_id;
    uint32_t remote_ip;
    uint32_t remote_engine_idx;
    ConnMeta* local_meta;
    ConnMeta* remote_meta;
    bool is_sender;
    // Wakeup handler
    PollCtx* poll_ctx;
  };
  const static uint32_t kCtrlMsgSize = sizeof(CtrlMsg);
  static_assert(sizeof(kCtrlMsgSize) % 4 == 0,
                "CtrlMsg must be 32-bit aligned");

  Channel() {
    tx_task_q_ = create_ring(sizeof(Msg), kChannelSize);
    rx_task_q_ = create_ring(sizeof(Msg), kChannelSize);
    rx_copy_q_ = create_ring(sizeof(Msg), kChannelSize);
    rx_copy_done_q_ = create_ring(sizeof(Msg), kChannelSize);
    ctrl_task_q_ = create_ring(sizeof(CtrlMsg), kChannelSize);
  }

  ~Channel() {
    free(tx_task_q_);
    free(rx_task_q_);
    free(rx_copy_q_);
    free(rx_copy_done_q_);
    free(ctrl_task_q_);
  }

  // Communicating rx/tx cmds between app thread and engine thread.
  jring_t* tx_task_q_;
  jring_t* rx_task_q_;
  // Communicating copy requests between engine thread to copy thread.
  jring_t* rx_copy_q_;
  jring_t* rx_copy_done_q_;
  // Communicating ctrl cmds between app thread and engine thread.
  jring_t* ctrl_task_q_;

  // A set of helper functions to enqueue/dequeue messages.
  static inline void enqueue_sp(jring_t* ring, void const* data) {
    while (jring_sp_enqueue_bulk(ring, data, 1, nullptr) != 1) {
    }
  }
  static inline void enqueue_mp(jring_t* ring, void const* data) {
    while (jring_mp_enqueue_bulk(ring, data, 1, nullptr) != 1) {
    }
  }
  static inline void enqueue_sp_multi(jring_t* ring, void const* data, int n) {
    while (jring_sp_enqueue_bulk(ring, data, n, nullptr) != n) {
    }
  }
  static inline void enqueue_mp_multi(jring_t* ring, void const* data, int n) {
    while (jring_mp_enqueue_bulk(ring, data, n, nullptr) != n) {
    }
  }
  static inline bool dequeue_sc(jring_t* ring, void* data) {
    return jring_sc_dequeue_bulk(ring, data, 1, nullptr) == 1;
  }
};

class UcclFlow;
class UcclEngine;
class Endpoint;

class TXTracking {
 public:
  TXTracking() = delete;
  TXTracking(EFASocket* socket, Channel* channel)
      : socket_(socket),
        channel_(channel),
        oldest_unacked_msgbuf_(nullptr),
        oldest_unsent_msgbuf_(nullptr),
        last_msgbuf_(nullptr),
        num_unacked_msgbufs_(0),
        num_unsent_msgbufs_(0),
        num_tracked_msgbufs_(0) {
    static double const kMinTxIntervalUs = EFA_MTU * 1.0 / kMaxBwPP * 1e6;
    kMinTxIntervalTsc = us_to_cycles(kMinTxIntervalUs, freq_ghz);
  }

  void receive_acks(uint32_t num_acked_pkts);
  void append(FrameDesc* msgbuf_head, FrameDesc* msgbuf_tail,
              uint32_t num_frames);
  std::optional<FrameDesc*> get_and_update_oldest_unsent();

  uint32_t convert_permitted_bytes_to_packets(uint32_t permitted_bytes);

  uint32_t convert_permitted_packets_to_bytes(uint32_t permitted_packets);

  inline const uint32_t num_unacked_msgbufs() const {
    return num_unacked_msgbufs_;
  }
  inline const uint32_t num_unsent_msgbufs() const {
    return num_unsent_msgbufs_;
  }
  inline FrameDesc* get_oldest_unacked_msgbuf() const {
    return oldest_unacked_msgbuf_;
  }

  friend class UcclFlow;
  friend class UcclEngine;

 private:
  EFASocket* socket_;
  Channel* channel_;

  /**
   * For the linked list of FrameDescs in the channel (chain going
   * downwards), we track 3 pointers
   *
   * B   -> oldest sent but unacknowledged MsgBuf
   * ...
   * B   -> oldest unsent MsgBuf
   * ...
   * B   -> last MsgBuf, among all active messages in this flow
   */

  FrameDesc* oldest_unacked_msgbuf_;
  FrameDesc* oldest_unsent_msgbuf_;
  FrameDesc* last_msgbuf_;

  uint32_t num_unacked_msgbufs_;
  uint32_t num_unsent_msgbufs_;
  uint32_t num_tracked_msgbufs_;

  uint16_t unacked_pkts_pp_[kMaxPath] = {0};
  inline void inc_unacked_pkts_pp(uint32_t path_id) {
    unacked_pkts_pp_[path_id]++;
  }
  inline void dec_unacked_pkts_pp(uint32_t path_id) {
    DCHECK_GT(unacked_pkts_pp_[path_id], 0) << "path_id " << path_id;
    unacked_pkts_pp_[path_id]--;
  }
  inline uint32_t get_unacked_pkts_pp(uint32_t path_id) {
    return unacked_pkts_pp_[path_id];
  }
  inline std::string unacked_pkts_pp_to_string() {
    std::stringstream ss;
    ss << "unacked_pkts_pp_: ";
    for (uint32_t i = 0; i < kMaxPath; i++) ss << unacked_pkts_pp_[i] << " ";
    return ss.str();
  }

  uint64_t kMinTxIntervalTsc = 0;
  uint64_t last_tx_tsc_pp_[kMaxPath] = {0};
  inline void set_last_tx_tsc_pp(uint32_t path_id, uint64_t tx_tsc) {
    last_tx_tsc_pp_[path_id] = tx_tsc;
  }
  inline bool is_available_for_tx(uint32_t path_id, uint64_t now_tsc) {
    return now_tsc - last_tx_tsc_pp_[path_id] >= kMinTxIntervalTsc;
  }
};

/**
 * @class RXTracking
 * @brief Tracking for message buffers that are received from the network. This
 * class is handling out-of-order reception of packets, and delivers complete
 * messages to the application.
 */
class RXTracking {
 public:
  // 256-bit SACK bitmask => we can track up to 256 packets
  static constexpr std::size_t kReassemblyMaxSeqnoDistance = kSackBitmapSize;

  static_assert((kReassemblyMaxSeqnoDistance &
                 (kReassemblyMaxSeqnoDistance - 1)) == 0,
                "kReassemblyMaxSeqnoDistance must be a power of two");

  RXTracking(RXTracking const&) = delete;
  RXTracking(EFASocket* socket, Channel* channel,
             std::unordered_map<FlowID, UcclFlow*>& active_flows_map)
      : socket_(socket),
        channel_(channel),
        active_flows_map_(active_flows_map) {}

  friend class UcclFlow;
  friend class UcclEngine;

  enum ConsumeRet : int {
    kOldPkt = 0,
    kOOOUntrackable = 1,
    kOOOTrackableDup = 2,
    kOOOTrackableExpectedOrInOrder = 3,
  };

  ConsumeRet consume(UcclFlow* flow, FrameDesc* msgbuf);

 private:
  std::unordered_map<FlowID, UcclFlow*>& active_flows_map_;
  void push_inorder_msgbuf_to_app(swift::Pcb* pcb);

 public:
  /**
   * Either the app supplies the app buffer or the engine receives a full msg.
   * It returns true if successfully copying the msgbuf to the app buffer;
   * otherwise false. Using rx_work as a pointer to diffirentiate null case.
   */
  void try_copy_msgbuf_to_appbuf(Channel::Msg* rx_work);

  // Two parts: messages that are out-of-order but trackable, and messages
  // that are ready but have not been delivered to app (eg, because of no app
  // buffer supplied by users).
  uint32_t num_unconsumed_msgbufs_ = 0;
  inline uint32_t num_unconsumed_msgbufs() const {
    return num_unconsumed_msgbufs_;
  }

 private:
  static void copy_thread_func(uint32_t engine_idx, UcclEngine* engine);

  EFASocket* socket_;
  Channel* channel_;

  struct seqno_cmp {
    bool operator()(uint32_t const& a, uint32_t const& b) const {
      return swift::seqno_lt(a, b);  // assending order
    }
  };
  // Using seqno_cmp to handle integer wrapping.
  std::map<uint32_t, FrameDesc*, seqno_cmp> reass_q_;

  // FIFO queue for ready messages that wait for app to claim.
  std::deque<FrameDesc*> ready_msg_queue_;
  struct app_buf_t {
    Channel::Msg rx_work;
  };
  std::deque<app_buf_t> app_buf_queue_;
  FrameDesc* deser_msgs_head_ = nullptr;
  FrameDesc* deser_msgs_tail_ = nullptr;
  size_t deser_msg_len_ = 0;
  int iov_n_ = 0;

  friend class Endpoint;
};

static inline FlowID get_peer_flow_id(FlowID flow_id) {
  return flow_id >= MAX_FLOW_ID ? flow_id - MAX_FLOW_ID : flow_id + MAX_FLOW_ID;
}

/**
 * @class UcclFlow, a connection between a local and a remote endpoint.
 * @brief Class to abstract the components and functionality of a single flow.
 * A flow is a bidirectional connection between two hosts, uniquely identified
 * by a TCP-negotiated `FlowID', Protocol is always UDP.
 *
 * A flow is always associated with a single `Channel' object which serves as
 * the communication interface with the application to which the flow belongs.
 *
 * On normal operation, a flow is:
 *    - Receiving network packets from the NIC, which then converts to messages
 *      and enqueues to the `Channel', so that they reach the application.
 *    - Receiving messages from the application (via the `Channel'), which then
 *      converts to network packets and sends them out to the remote recipient.
 */
class UcclFlow {
 public:
  /**
   * @brief Construct a new flow.
   *
   * @param local_addr Local IP address.
   * @param remote_addr Remote IP address.
   * @param EFASocket object for packet IOs.
   * @param FlowID Connection ID for the flow.
   */
  UcclFlow(std::string local_ip_str, std::string remote_ip_str,
           ConnMeta* local_meta, ConnMeta* remote_meta,
           uint32_t local_engine_idx, uint32_t remote_engine_idx,
           EFASocket* socket, eqds::CreditQPContext* credit_qp_ctx,
           eqds::EQDSChannel* eqds_channel, Channel* channel, FlowID flow_id,
           std::unordered_map<FlowID, UcclFlow*>& active_flows_map_,
           bool is_sender)
      : remote_ip_str_(remote_ip_str),
        local_ip_str_(local_ip_str),
        local_meta_(local_meta),
        remote_meta_(remote_meta),
        local_engine_idx_(local_engine_idx),
        remote_engine_idx_(remote_engine_idx),
        socket_(CHECK_NOTNULL(socket)),
        credit_qp_ctx_(credit_qp_ctx),
        eqds_channel_(eqds_channel),
        channel_(channel),
        flow_id_(flow_id),
        pcb_(),
        cubic_g_(),
        timely_g_(),
        tx_tracking_(socket, channel),
        rx_tracking_(socket, channel, active_flows_map_),
        is_sender_(is_sender),
        eqds_cc_() {
    // Initing per-flow CC states.
    timely_g_.init(&pcb_);
    if constexpr (kSenderCCType == SenderCCType::kTimelyPP) {
      timely_pp_ = new swift::TimelyCtl[kMaxPath];
      for (uint32_t i = 0; i < kMaxPath; i++) timely_pp_[i].init(&pcb_);
    }

    cubic_g_.init(&pcb_, kMaxUnackedPktsPerEngine);
    if constexpr (kSenderCCType == SenderCCType::kCubicPP) {
      cubic_pp_ = new swift::CubicCtl[kMaxPath];
      for (uint32_t i = 0; i < kMaxPath; i++)
        cubic_pp_[i].init(&pcb_, kMaxUnackedPktsPP);
    }

    peer_flow_id_ = get_peer_flow_id(flow_id);

    if constexpr (kReceiverCCType == ReceiverCCType::kEQDS) {
      eqds_cc_.send_pullpacket = [this](PullQuanta const& pullno) -> bool {
        return this->send_pullpacket(pullno);
      };
    }
  }
  ~UcclFlow() {
    delete local_meta_;
    delete remote_meta_;
    if constexpr (kSenderCCType == SenderCCType::kTimelyPP) delete[] timely_pp_;
    if constexpr (kSenderCCType == SenderCCType::kCubicPP) delete[] cubic_pp_;
  }

  friend class UcclEngine;

  std::string to_string() const;
  inline void shutdown() {}

  /**
   * @brief Push the received packet onto the ingress queue of the flow.
   * Decrypts packet if required, stores the payload in the relevant channel
   * shared memory space, and if the message is ready for delivery notifies
   * the application.
   *
   * If this is a transport control packet (e.g., ACK) it only updates
   * transport-related parameters for the flow.
   */
  void rx_messages();

  inline void rx_supply_app_buf(Channel::Msg& rx_work) {
    rx_tracking_.try_copy_msgbuf_to_appbuf(&rx_work);
  }

  /**
   * @brief Push a Message from the application onto the egress queue of
   * the flow. Segments the message, and encrypts the packets, and adds
   * all packets onto the egress queue. Caller is responsible for freeing
   * the MsgBuf object.
   *
   * @param msg Pointer to the first message buffer on a train of buffers,
   * aggregating to a partial or a full Message.
   */
  void tx_prepare_messages(Channel::Msg& tx_deser_work);

  void process_rttprobe_rsp(uint64_t ts1, uint64_t ts2, uint64_t ts3,
                            uint64_t ts4, uint32_t path_id);

  /**
   * @brief Periodically checks the state of the flow and performs
   * necessary actions.
   *
   * This method is called periodically to check the state of the flow,
   * update the RTO timer, retransmit unacknowledged messages, and
   * potentially remove the flow or notify the application about the
   * connection state.
   *
   * @return Returns true if the flow should continue to be checked
   * periodically, false if the flow should be removed or closed.
   */
  bool periodic_check();

  inline swift::Pcb* get_pcb() { return &pcb_; }

  inline eqds::EQDSCC* get_eqds_cc() { return &eqds_cc_; }

  // [Thread-safe] Request pacer to grant credit to this flow.
  inline void request_pull() {
    eqds::EQDSChannel::Msg msg = {
        .opcode = eqds::EQDSChannel::Msg::Op::kRequestPull,
        .eqds_cc = &eqds_cc_,
    };
    while (jring_mp_enqueue_bulk(eqds_channel_->cmdq_, &msg, 1, nullptr) != 1) {
    }
  }

 private:
  void process_ack(UcclPktHdr const* ucclh);

  void process_credit(UcclPullHdr const* ucclh);

  void fast_retransmit();
  bool rto_retransmit(FrameDesc* msgbuf, uint32_t seqno);

  inline bool can_rtx(FrameDesc* msgbuf) {
    // Avoid too many inflight WQEs.
    if (socket_->send_queue_wrs() >= kMaxUnackedPktsPerEngine) return false;

    // // The following code may have BUG.
    // if constexpr (kReceiverCCType == ReceiverCCType::kEQDS) {
    //     if (eqds_cc_.credit() < msgbuf->get_pkt_data_len()) return false;
    //     if (!eqds_cc_.spend_credit(msgbuf->get_pkt_data_len())) return
    //     false;
    // }

    return true;
  }

  /**
   * @brief Helper function to transmit a number of packets from the queue
   * of pending TX data.
   */
  uint32_t transmit_pending_packets(uint32_t budget);

  /**
   * @brief This function is a variant of transmit_pending_packets, which
   * work in a Deficit Round Robin manner.
   */
  inline void transmit_pending_packets_drr(bool bypass) {
    if (bypass) {
      transmit_pending_packets(SEND_BATCH_SIZE);
      return;
    }

    if (deficit_ <= 0) deficit_ += quantum_;

    if (deficit_ > 0) deficit_ -= transmit_pending_packets(deficit_);

    if (!has_pending_packets()) deficit_ = 0;
  }

  inline bool has_pending_packets() {
    return tx_tracking_.num_unsent_msgbufs();
  }

  struct pending_tx_msg_t {
    Channel::Msg tx_work;
    size_t cur_offset = 0;
  };

  std::deque<pending_tx_msg_t> pending_tx_msgs_;

  /**
   * @brief Deserialize a chunk of data from the application buffer and append
   * to the tx tracking.
   */
  void deserialize_and_append_to_txtracking();

  void prepare_datapacket(FrameDesc* msgbuf, uint32_t path_id, uint32_t seqno,
                          const UcclPktHdr::UcclFlags net_flags);
  FrameDesc* craft_ackpacket(uint32_t path_id, uint32_t seqno, uint32_t ackno,
                             const UcclPktHdr::UcclFlags net_flags,
                             uint64_t ts1, uint64_t ts2, uint32_t rwnd);

  bool send_pullpacket(PullQuanta const& pullno);

  std::string local_ip_str_;
  std::string remote_ip_str_;

  // The following is used to fill packet headers.
  ConnMeta* local_meta_;
  ConnMeta* remote_meta_;
  struct ibv_ah* remote_ah_;

  // Which engine (also NIC queue and xsk) this flow belongs to.
  uint32_t local_engine_idx_;
  uint32_t remote_engine_idx_;

  // The underlying EFASocket.
  EFASocket* socket_;

  eqds::CreditQPContext* credit_qp_ctx_;
  eqds::EQDSChannel* eqds_channel_;
  uint32_t credit_qpidx_rr_ = 0;

  // The channel this flow belongs to.
  Channel* channel_;
  // FlowID of this flow.
  FlowID flow_id_;
  // Accumulated data frames to be sent.
  std::vector<FrameDesc*> pending_tx_frames_;
  // Missing data frames to be sent.
  std::vector<FrameDesc*> missing_frames_;
  // Frames that are pending rx processing in a batch.
  std::deque<FrameDesc*> pending_rx_msgbufs_;
  // Whether this is a sender or receiver flow in NCCL.
  bool is_sender_;

  TXTracking tx_tracking_;
  RXTracking rx_tracking_;

  // Swift reliable transmission control block.
  swift::Pcb pcb_;
  swift::TimelyCtl timely_g_;
  swift::CubicCtl cubic_g_;
  // Each path has its own PCB for CC.
  swift::TimelyCtl* timely_pp_;
  swift::CubicCtl* cubic_pp_;
  // Peer flow_id used for communication.
  FlowID peer_flow_id_ = 0;

  // EQDS
  eqds::EQDSCC eqds_cc_;

  uint32_t last_received_rwnd_ = kMaxUnconsumedRxMsgbufs;

  // Deficit Round Robin
  int32_t deficit_ = 0;
  int32_t quantum_ = SEND_BATCH_SIZE;

  inline std::tuple<uint16_t, uint16_t> path_id_to_src_dst_qp(
      uint32_t path_id) {
    return {path_id / kMaxDstQP, path_id % kMaxDstQP};
  }
  inline uint32_t src_dst_qp_to_path_id(uint16_t src_qp, uint16_t dst_qp) {
    DCHECK(src_qp < kMaxSrcQP && dst_qp < kMaxDstQP);
    return src_qp * kMaxDstQP + dst_qp;
  }
  inline uint32_t data_path_id_to_ctrl_path_id(uint32_t data_path_id) {
    return data_path_id % kMaxPathCtrl;
  }
  inline std::tuple<uint16_t, uint16_t> path_id_to_src_dst_qp_for_ctrl(
      uint32_t path_id) {
    return {path_id / kMaxDstQPCtrl, path_id % kMaxDstQPCtrl};
  }
  inline uint32_t src_dst_qp_to_path_id_for_ctrl(uint16_t src_qp,
                                                 uint16_t dst_qp) {
    DCHECK(src_qp < kMaxSrcQPCtrl && dst_qp < kMaxDstQPCtrl);
    return src_qp * kMaxDstQPCtrl + dst_qp;
  }

  // Path ID for each packet indexed by seqno.
  uint16_t hist_path_id_[kMaxPathHistoryPerEngine] = {0};
  inline void set_path_id(uint32_t seqno, uint32_t path_id) {
    hist_path_id_[seqno & (kMaxPathHistoryPerEngine - 1)] = path_id;
  }
  inline uint32_t get_path_id(uint32_t seqno) {
    return hist_path_id_[seqno & (kMaxPathHistoryPerEngine - 1)];
  }

  // Measure the distribution of probed RTT.
  Latency rtt_stats_;
  uint64_t rtt_probe_count_ = 0;

  // RTT in tsc, indexed by path_id.
  size_t port_path_rtt_[kMaxPath] = {0};

  // For ctrl, its path is derived from data path_id.
  uint16_t next_src_qp = 0;
  inline uint16_t get_src_qp_rr() { return (next_src_qp++) % kMaxSrcQP; }

  uint16_t next_dst_qp = 0;
  inline uint16_t get_dst_qp_pow2(uint16_t src_qp_idx) {
#ifdef PATH_SELECTION
    auto idx_u32 = U32Rand(0, UINT32_MAX);
    auto idx1 = idx_u32 % kMaxDstQP;
    auto idx2 = (idx_u32 >> 16) % kMaxDstQP;
    auto path_id1 = src_dst_qp_to_path_id(src_qp_idx, idx1);
    auto path_id2 = src_dst_qp_to_path_id(src_qp_idx, idx2);
    VLOG(3) << "rtt: idx1 " << port_path_rtt_[path_id1] << " idx2 "
            << port_path_rtt_[path_id2];
    return (port_path_rtt_[path_id1] < port_path_rtt_[path_id2]) ? idx1 : idx2;
#else
    return (next_dst_qp++) % kMaxDstQP;
#endif
  }

  uint32_t next_path_id = 0;
  inline uint32_t get_path_id_with_lowest_rtt() {
#ifdef PATH_SELECTION
    auto idx_u32 = U32Rand(0, UINT32_MAX);
    auto idx1 = idx_u32 % kMaxPath;
    auto idx2 = (idx_u32 >> 16) % kMaxPath;
    VLOG(3) << "rtt: idx1 " << port_path_rtt_[idx1] << " idx2 "
            << port_path_rtt_[idx2];
    return (port_path_rtt_[idx1] < port_path_rtt_[idx2]) ? idx1 : idx2;
#else
    return (next_path_id++) % kMaxPath;
#endif
  }

  friend class UcclEngine;
  friend class Endpoint;
};

/**
 * @brief Class `UcclEngine' abstracts the main Uccl engine. This engine
 * contains all the functionality need to be run by the stack's threads.
 */
class UcclEngine {
 public:
  // Slow timer (periodic processing) interval in microseconds.
  // const size_t kSlowTimerIntervalUs = 1000;
  const size_t kSlowTimerIntervalUs = 2000;
  UcclEngine() = delete;
  UcclEngine(UcclEngine const&) = delete;

  /**
   * @brief Construct a new UcclEngine object.
   *
   * @param socket_idx      Global socket idx or engine idx.
   * @param channel       Uccl channel the engine will be responsible for.
   * For now, we assume an engine is responsible for a single channel, but
   * future it may be responsible for multiple channels.
   */
  UcclEngine(std::string local_ip_str, int gpu_idx, int dev_idx, int socket_idx,
             Channel* channel, eqds::CreditQPContext* credit_qp_ctx,
             eqds::EQDSChannel* eqds_channel)
      : local_ip_str_(local_ip_str),
        local_engine_idx_(socket_idx),
        socket_(EFAFactory::CreateSocket(gpu_idx, dev_idx, socket_idx)),
        channel_(channel),
        credit_qp_ctx_(credit_qp_ctx),
        eqds_channel_(eqds_channel),
        last_periodic_tsc_(rdtsc()),
        periodic_ticks_(0),
        kSlowTimerIntervalTsc_(us_to_cycles(kSlowTimerIntervalUs, freq_ghz)) {}

  /**
   * @brief This is the main event cycle of the Uccl engine.
   * It is called by a separate thread running the Uccl engine.
   * On each iteration, the engine processes incoming packets in the RX
   * queue and enqueued messages in all channels that it is responsible
   * for. This method is not thread-safe.
   */
  void run();

  void sender_only_run();

  void receiver_only_run();

  /**
   * @brief Method to perform periodic processing. This is called by the
   * main engine cycle (see method `Run`).
   */
  void periodic_process();

  void handle_install_flow_on_engine(Channel::CtrlMsg& ctrl_work);

  // Called by application to shutdown the engine. App will need to join
  // the engine thread.
  inline void shutdown() { shutdown_ = true; }

  std::string status_to_string(bool abbrev);

 protected:
  /**
   * @brief Process incoming packets.
   *
   * @param pkt_msgs Pointer to a list of packets.
   */
  void process_rx_msg(std::vector<FrameDesc*>& pkt_msgs);

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
  // Local IP address.
  std::string local_ip_str_;
  // Engine index, also NIC queue ID and xsk index.
  uint32_t local_engine_idx_;
  // AFXDP socket used for send/recv packets.
  EFASocket* socket_;

  // Receiver-driven CC.
  eqds::CreditQPContext* credit_qp_ctx_;
  eqds::EQDSChannel* eqds_channel_;

  // UcclFlow map
  std::unordered_map<FlowID, UcclFlow*> active_flows_map_;
  // Control plane channel with Endpoint.
  Channel* channel_;
  // Timestamp of last periodic process execution.
  uint64_t last_periodic_tsc_;
  // Clock ticks for the slow timer.
  uint64_t periodic_ticks_;
  // Slow timer interval in TSC.
  uint64_t kSlowTimerIntervalTsc_;
  // Whether shutdown is requested.
  std::atomic<bool> shutdown_{false};

  friend class RXTracking;
  friend class Endpoint;
};

/**
 * @class Endpoint
 * @brief application-facing interface, communicating with `UcclEngine' through
 * `Channel'. Each connection is identified by a unique flow_id, and uses
 * multiple src+dst port combinations to leverage multiple paths. Under the
 * hood, we leverage TCP to boostrap our connections. We do not consider
 * multi-tenancy for now, assuming this endpoint exclusively uses the NIC and
 * its all queues.
 */
class Endpoint {
  constexpr static uint16_t kBootstrapPort = 5000;
  constexpr static uint32_t kStatsTimerIntervalSec = 2;

  Channel* channel_vec_[kNumEngines];
  std::vector<std::unique_ptr<UcclEngine>> engine_vec_;
  std::unordered_map<int, std::unique_ptr<UcclEngine>> engine_id_to_engine_map_;
  std::mutex engine_map_mutex_;
  std::vector<std::unique_ptr<std::thread>> engine_th_vec_;
  std::vector<std::unique_ptr<std::thread>> copy_th_vec_;

  std::atomic<uint16_t> listen_port_cur_ = {kBootstrapPort};

  // Number of flows on each engine, indexed by engine_idx.
  std::mutex engine_load_vec_mu_;
  std::array<int, kNumEngines> engine_load_vec_ = {0};
  std::array<int, kNumEngines> engine_tx_load_vec_ = {0};
  std::array<int, kNumEngines> engine_rx_load_vec_ = {0};

  // We must use a thread-safe pool but not per-engine poll, as different
  // threads would issue send/recv even for the same engine.
  SharedPool<PollCtx*, true>* ctx_pool_;
  uint8_t* ctx_pool_buf_;

  std::mutex fd_map_mu_;
  // Mapping from unique (within this engine) flow_id to the boostrap fd.
  std::unordered_map<FlowID, int> fd_map_;

  // Each physical device has its own EQDS pacer.
  eqds::EQDS* eqds_[NUM_DEVICES] = {};

 public:
  int gpu_;
  Endpoint(int gpu);
  Endpoint();
  ~Endpoint();

  std::mutex listen_mu_;
  std::vector<uint16_t> listen_port_vec_;
  std::vector<int> listen_fd_vec_;

  // Using TCP socket to listen on, and return the listen port and fd.
  std::tuple<uint16_t, int> uccl_listen();
  // Connecting to a remote address; thread-safe
  ConnID uccl_connect(int local_vdev, int remote_vdev, std::string remote_ip,
                      uint16_t listen_port);
  // Accepting a connection from a remote address; thread-safe
  ConnID uccl_accept(int local_vdev, int* remote_vdev, std::string& remote_ip,
                     int listen_fd);

  bool create_engine_and_add_to_engine_future(
      int engine_idx, int gpu_idx,
      std::vector<std::future<std::unique_ptr<UcclEngine>>>& engine_futures);
  bool initialize_engine_by_gpu_idx(int gpu_idx);

  // Sending the data by leveraging multiple port combinations.
  bool uccl_send(ConnID conn_id, void const* data, int const len,
                 Mhandle* mhandle, bool busypoll = false);
  // Receiving the data by leveraging multiple port combinations.
  bool uccl_recv(ConnID conn_id, void* data, int* len_p, Mhandle* mhandle,
                 bool busypoll = false);
  bool uccl_recv_multi(ConnID conn_id, void** data, int* len_p,
                       Mhandle** mhandle, int n, bool busypoll = false);

  // Sending the data by leveraging multiple port combinations.
  PollCtx* uccl_send_async(ConnID conn_id, void const* data, int const len,
                           Mhandle* mhandle);
  // Receiving the data by leveraging multiple port combinations.
  PollCtx* uccl_recv_async(ConnID conn_id, void* data, int* len_p,
                           Mhandle* mhandle);
  PollCtx* uccl_recv_scattered_async(ConnID conn_id, UcclRequest* req,
                                     Mhandle* mhandle);
  void uccl_recv_free_ptrs(ConnID conn_id, int iov_n, void** iov_addrs);
  PollCtx* uccl_recv_multi_async(ConnID conn_id, void** data, int* len_p,
                                 Mhandle** mhandle, int n);
  PollCtx* uccl_flush_async(ConnID conn_id, void** data, int* len_p,
                            Mhandle** mhandle, int n);

  bool uccl_wait(PollCtx* ctx);
  bool uccl_poll(PollCtx* ctx);
  bool uccl_poll_once(PollCtx* ctx);

  int uccl_regmr_dmabuf(int dev, void* addr, size_t len, int type, int offset,
                        int fd, struct Mhandle** mhandle);
  int uccl_regmr(int dev, void* addr, size_t len, int type /*unsed for now*/,
                 struct Mhandle** mhandle);
  void uccl_deregmr(struct Mhandle* mhandle);

 private:
  void install_flow_on_engine(FlowID flow_id, std::string const& remote_ip,
                              uint32_t local_engine_idx, int bootstrap_fd,
                              bool is_sender);
  inline int find_least_loaded_engine_idx_and_update(int vdev_idx,
                                                     FlowID flow_id,
                                                     bool is_sender);
  inline int find_dedicated_engine_idx(int vdev_idx, bool is_sender);
  inline void fence_and_clean_ctx(PollCtx* ctx);

  std::mutex stats_mu_;
  std::condition_variable stats_cv_;
  std::atomic<bool> shutdown_{false};
  std::thread stats_thread_;
  void stats_thread_fn();

  friend class UcclFlow;
};

static inline int receive_message(int sockfd, void* buffer, size_t n_bytes) {
  int bytes_read = 0;
  int r;
  while (bytes_read < n_bytes) {
    // Make sure we read exactly n_bytes
    r = read(sockfd, buffer + bytes_read, n_bytes - bytes_read);
    if (r < 0 && !(errno == EAGAIN || errno == EWOULDBLOCK)) {
      CHECK(false) << "ERROR reading from socket";
    }
    if (r > 0) {
      bytes_read += r;
    }
  }
  return bytes_read;
}

static inline int send_message(int sockfd, void const* buffer, size_t n_bytes) {
  int bytes_sent = 0;
  int r;
  while (bytes_sent < n_bytes) {
    // Make sure we write exactly n_bytes
    r = write(sockfd, buffer + bytes_sent, n_bytes - bytes_sent);
    if (r < 0 && !(errno == EAGAIN || errno == EWOULDBLOCK)) {
      CHECK(false) << "ERROR writing to socket";
    }
    if (r > 0) {
      bytes_sent += r;
    }
  }
  return bytes_sent;
}

static inline void net_barrier(int bootstrap_fd) {
  bool sync = true;
  int ret = send_message(bootstrap_fd, &sync, sizeof(bool));
  ret = receive_message(bootstrap_fd, &sync, sizeof(bool));
  DCHECK(ret == sizeof(bool) && sync);
}

static inline uint32_t get_gpu_idx_by_engine_idx(uint32_t engine_idx) {
  return engine_idx / kNumEnginesPerVdev;
}

static inline uint32_t get_dev_idx_by_engine_idx(uint32_t engine_idx) {
  return engine_idx / (kNumEnginesPerVdev * 2);
}

static inline uint32_t get_engine_off_by_engine_idx(uint32_t engine_idx) {
  return engine_idx % (kNumEnginesPerVdev * 2);
}

static inline uint32_t get_dev_idx_by_gpu_idx(uint32_t gpu_idx) {
  // Fixed on P4D.
  return gpu_idx / 2;
}

static inline int get_pdev(int vdev) { return vdev / 2; }
static inline int get_vdev(int pdev) { return pdev * 2; }

}  // namespace uccl
