#ifndef UTIL_RDMA_H
#define UTIL_RDMA_H

#include <cstdint>
#include <cstring>
#include <set>
#include <unordered_map>
#include <vector>

#include <infiniband/verbs.h>
#include <sys/mman.h>

#include <glog/logging.h>

#include "eqds.h"
#include "pcb.h"
#include "transport_config.h"

#include "util.h"
#include "util_buffpool.h"
#include "util_endian.h"
#include "util_list.h"

namespace uccl {

typedef uint64_t FlowID;
typedef uint64_t PeerID;

class RDMAContext;
class RDMAFactory;
extern std::shared_ptr<RDMAFactory> rdma_ctl;

// LRH (Local Routing Header) + GRH (Global Routing Header) + BTH (Base
// Transport Header)
static constexpr uint32_t IB_HDR_OVERHEAD = (8 + 40 + 12);
// Ethernet + IPv4 + UDP + BTH
static constexpr uint32_t ROCE_IPV4_HDR_OVERHEAD = (14 + 20 + 8 + 12);
// Ethernet + IPv6 + UDP + BTH
static constexpr uint32_t ROCE_IPV6_HDR_OVERHEAD = (14 + 40 + 8 + 12);

static constexpr uint32_t BASE_PSN = 0;

// For quick computation at MTU 4096
static constexpr uint32_t MAX_CHUNK_ROCE_IPV4_4096_HDR_OVERHEAD =
    ((kChunkSize + 4096) / 4096) * ROCE_IPV4_HDR_OVERHEAD;
static constexpr uint32_t MAX_CHUNK_ROCE_IPV6_4096_HDR_OVERHEAD =
    ((kChunkSize + 4096) / 4096) * ROCE_IPV6_HDR_OVERHEAD;
static constexpr uint32_t MAX_CHUNK_IB_4096_HDR_OVERHEAD =
    ((kChunkSize + 4096) / 4096) * IB_HDR_OVERHEAD;

/**
 * @brief Buffer pool for work request extension.
 */
class WrExBuffPool : public BuffPool {
    static constexpr uint32_t kWrSize = sizeof(struct wr_ex);
    static constexpr uint32_t kNumWr = 4096;
    static_assert((kNumWr & (kNumWr - 1)) == 0, "kNumWr must be power of 2");

  public:
    WrExBuffPool()
        : BuffPool(kNumWr, kWrSize, nullptr, [](uint64_t buff) {
              struct wr_ex *wr_ex = reinterpret_cast<struct wr_ex *>(buff);
              auto wr = &wr_ex->wr;
              wr->sg_list = &wr_ex->sge;
              wr->num_sge = 1;
              wr->next = nullptr;
              wr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
          }) {}

    ~WrExBuffPool() = default;
};

class IMMData {
    public:
      // HINT: Indicates whether the last chunk of a message.
      // CSN:  Chunk Sequence Number.
      // RID:  Request ID.
      // FID:  Flow Index.
      // High-----------------32bit------------------Low
      //  | HINT |  RESERVED  |  CSN  |  RID  |  FID  |
      //    1bit      8bit       8bit    7bit    8bit
      constexpr static int kFID = 0;
      constexpr static int kRID = 8;
      constexpr static int kCSN = 15;
      constexpr static int kRESERVED = kCSN + UINT_CSN_BIT;
      constexpr static int kHINT = kRESERVED + 8;
  
      IMMData(uint32_t imm_data) : imm_data_(imm_data) {}
  
      inline uint32_t GetHINT(void) { return (imm_data_ >> kHINT) & 0x1; }
  
      inline uint32_t GetRESERVED(void) {
          return (imm_data_ >> kRESERVED) & 0xFF;
      }
  
      inline uint32_t GetCSN(void) { return (imm_data_ >> kCSN) & UINT_CSN_MASK; }
  
      inline uint32_t GetRID(void) { return (imm_data_ >> kRID) & 0x7F; }
  
      inline uint32_t GetFID(void) { return (imm_data_ >> kFID) & 0xFF; }
  
      inline void SetHINT(uint32_t hint) { imm_data_ |= (hint & 0x1) << kHINT; }
  
      inline void SetRESERVED(uint32_t reserved) {
          imm_data_ |= (reserved & 0xFF) << kRESERVED;
      }
  
      inline void SetCSN(uint32_t csn) {
          imm_data_ |= (csn & UINT_CSN_MASK) << kCSN;
      }
  
      inline void SetRID(uint32_t rid) { imm_data_ |= (rid & 0x7F) << kRID; }
  
      inline void SetFID(uint32_t fid) { imm_data_ |= (fid & 0xFF) << kFID; }
  
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
  
      inline uint32_t GetTarget(void) {
          return (imm_data_ >> kPULL_TARGET) & 0xFF;
      }
  
      inline void SetTarget(uint32_t pull_target) {
          imm_data_ |= (pull_target & 0xFF) << kPULL_TARGET;
      }
  };

struct __attribute__((packed)) retr_chunk_hdr {
    // Target address for the lost chunk.
    uint64_t remote_addr;
    uint32_t imm_data;
};

/**
 * @brief Buffer pool for retransmission chunks (original chunk + retransmission
 * header). Original chunk and retransmission header are transmitted through
 * scatter-gather list.
 */
class RetrChunkBuffPool : public BuffPool {
  public:
    static constexpr uint32_t kRetrChunkSize =
        kChunkSize + sizeof(retr_chunk_hdr);
    static constexpr uint32_t kNumChunk = 4096;
    static_assert((kNumChunk & (kNumChunk - 1)) == 0,
                  "kNumChunk must be power of 2");

    RetrChunkBuffPool(struct ibv_mr *mr)
        : BuffPool(kNumChunk, kRetrChunkSize, mr) {}

    ~RetrChunkBuffPool() = default;
};

/**
 * @brief Buffer pool for retransmission headers.
 */
class RetrHdrBuffPool : public BuffPool {
  public:
    static constexpr uint32_t kHdrSize = sizeof(struct retr_chunk_hdr);
    static constexpr uint32_t kNumHdr = 1024;
    static_assert((kNumHdr & (kNumHdr - 1)) == 0, "kNumHdr must be power of 2");

    RetrHdrBuffPool(struct ibv_mr *mr) : BuffPool(kNumHdr, kHdrSize, mr) {}

    ~RetrHdrBuffPool() = default;
};

/**
 * @brief Buffer pool for control packets.
 */
class CtrlChunkBuffPool : public BuffPool {
  public:
    static constexpr uint32_t kPktSize = 32;
    static constexpr uint32_t kChunkSize = kPktSize * kMaxBatchCQ;
    static constexpr uint32_t kNumChunk = kMaxBatchCQ << 6;
    static_assert((kNumChunk & (kNumChunk - 1)) == 0,
                  "kNumChunk must be power of 2");

    CtrlChunkBuffPool(struct ibv_mr *mr)
        : BuffPool(kNumChunk, kChunkSize, mr) {}

    ~CtrlChunkBuffPool() = default;
};

/**
 * @brief Metadata for control messages.
 */
union CtrlMeta {
    // kInstallCtx
    struct {
        union ibv_gid remote_gid;
        struct ibv_port_attr remote_port_attr;
        bool is_send;
        int bootstrap_fd;
    } install_ctx;

    // kInstallFlow
    struct {
        FlowID flow_id;
        void *context;
        bool is_send;
    } install_flow;
};

struct FifoItem {
    uint64_t addr;
    uint32_t size;
    uint32_t rkey;
    uint32_t nmsgs;
    uint32_t rid;
    uint64_t idx;
    uint32_t engine_offset;
    char padding[28];
};
static_assert(sizeof(struct FifoItem) == 64, "FifoItem size is not 64 bytes");

/**
 * @brief A FIFO queue for flow control.
 * Receiver posts a buffer to the FIFO queue for the sender to use RDMA WRITE.
 */
struct RemFifo {
    // FIFO elements prepared for sending to remote peer.
    struct FifoItem elems[kMaxReq][kMaxRecv];
    // Tail pointer of the FIFO.
    uint64_t fifo_tail;
    // Only used for testing RC.
    uint32_t sizes[kMaxReq][kMaxRecv];
};

struct RemoteRDMAContext {
    union ibv_gid remote_gid;
    struct ibv_port_attr remote_port_attr;
};

enum ReqType {
    ReqTx,
    ReqRx,
    ReqFlush,
    ReqTxRC,
    ReqRxRC,
};

/**
 * @brief ucclRequest is a handle provided by the user to post a request to UCCL
 * RDMAEndpoint. It is the responsibility of the user to manage the memory of
 * ucclRequest. UCCL RDMAEndpoint will not free the memory of ucclRequest. UCCL
 * fills the ucclRequest with the result of the request. The user can use the
 * ucclRequest to check the status of the request.
 */
struct ucclRequest {
    enum ReqType type;
    union {
        int n;
        int mid; // used for multi-send
    };
    union {
        PollCtx *poll_ctx;
        // For reducing overhead of PollCtx for RC and Flush operation.
        uint64_t rc_or_flush_done;
    };
    void *context;
    void *req_pool;
    uint32_t engine_idx;
    union {
        struct {
            int data_len[kMaxRecv];
            uint64_t data[kMaxRecv];
            struct FifoItem *elems;
            struct ibv_send_wr wr;
            struct ibv_sge sge;
            struct ibv_qp *qp;
        } recv;
        struct {
            int data_len;
            int inc_backlog;
            uint64_t laddr;
            uint64_t raddr;
            uint32_t lkey;
            uint32_t rkey;
            uint32_t rid;
            uint32_t sent_offset;
            uint32_t acked_bytes; // RC only.
        } send;
    };
    uint64_t rtt_tsc;
};

/**
 * @brief Each RDMAContext has a pool of RecvRequest.
 * After the recevier posting an async recv ucclRequest to an engine, the engine
 * will allocate a RecvRequest from its RDMAContext. Then, when receiving the
 * data, the engine will locate the RecvRequest and then further find the
 * ucclRequest.
 */
struct RecvRequest {
    enum type {
        UNUSED = 0,
        RECV,
    };
    enum type type;
    struct ucclRequest *ureq;
    uint32_t received_bytes[kMaxRecv];
    uint32_t fin_msg;
};

/// @ref ncclIbNetCommBase
struct alignas(32) NetCommBase {
    // Pointing to rdma_ctx_->fifo_mr_->addr.
    struct RemFifo *fifo;

    // CQ for Fifo QP and GPU flush QP and RC QP.
    struct ibv_cq *flow_cq;

    // Fifo QP based on Reliable Connection (RC).
    struct ibv_qp *fifo_qp;
    // Local PSN for Fifo.
    uint32_t fifo_local_psn;
    // Memory region for Fifo.
    struct ibv_mr *fifo_mr;

    // RC UP for small messages bypassing UcclEngine.
    struct ibv_qp *rc_qp;
    uint32_t rc_local_psn;

    uint64_t remote_fifo_addr;
    uint32_t remote_fifo_rkey;
};

/// @ref ncclIbSendComm
struct SendComm {
    struct NetCommBase base;
    // Track outstanding FIFO requests.
    struct ucclRequest *fifo_ureqs[kMaxReq][kMaxRecv];
    uint64_t fifo_head;
};

/// @ref ncclIbRecvComm
struct RecvComm {
    struct NetCommBase base;

    // QP for GPU flush.
    struct ibv_qp *gpu_flush_qp;
    // Memory region for GPU flush.
    struct ibv_mr *gpu_flush_mr;
    struct ibv_sge gpu_flush_sge;
    // GPU flush buffer
    int gpu_flush;
};

class RXTracking {
  public:
    std::set<std::pair<UINT_CSN, void *>> ready_csn_;

    RXTracking() = default;
    ~RXTracking() = default;

    // Immediate Acknowledgement.
    inline void cumulate_wqe(void) { cumulative_wqe_++; }
    inline void cumulate_bytes(uint32_t bytes) { cumulative_bytes_ += bytes; }
    inline void encounter_ooo(void) {
        if (++consectutive_ooo_ >= kMAXRXOOO) {
            ooo_ = true;
            consectutive_ooo_ = 0;
        }
    }

    inline bool real_ooo(void) { return ooo_; }

    /**
     * @brief Send ack immediately if the following conditions are met:
     * 1. Out-of-order packets are received.
     * 2. The number of received WQE reaches kMAXCumWQE.
     * 3. The number of received bytes reaches kMAXCumBytes.
     */
    inline bool need_imm_ack(void) {
        return ooo_ || cumulative_wqe_ == kMAXCumWQE ||
               cumulative_bytes_ >= kMAXCumBytes;
    }
    /**
     * @brief After sending immediate ack, clear the states.
     */
    inline void clear_imm_ack(void) {
        ooo_ = false;
        cumulative_wqe_ = 0;
        cumulative_bytes_ = 0;
        consectutive_ooo_ = 0;
    }

  private:
    bool ooo_ = false;
    uint32_t cumulative_wqe_ = 0;
    uint32_t cumulative_bytes_ = 0;
    uint32_t consectutive_ooo_ = 0;
};

class TXTracking {
  public:
    struct ChunkTrack {
        struct ucclRequest *ureq;
        struct wr_ex *wr_ex;
        uint64_t timestamp;
        uint32_t csn;
        bool last_chunk;
    };

    TXTracking() = default;
    ~TXTracking() = default;

    inline bool empty(void) { return unacked_chunks_.empty(); }

    inline TXTracking::ChunkTrack get_unacked_chunk_from_idx(uint32_t idx) {
        return unacked_chunks_[idx];
    }

    inline TXTracking::ChunkTrack get_oldest_unacked_chunk(void) {
        return unacked_chunks_.front();
    }

    std::pair<uint64_t, uint32_t>
    ack_rc_transmitted_chunks(void *subflow_context, RDMAContext *rdma_ctx,
                              UINT_CSN csn, uint64_t now,
                              uint32_t *flow_unacked_bytes,
                              uint32_t *engine_outstanding_bytes);

    uint64_t ack_transmitted_chunks(void *subflow_context,
                                    RDMAContext *rdma_ctx,
                                    uint32_t num_acked_chunks, uint64_t t5,
                                    uint64_t t6, uint64_t remote_queueing_tsc,
                                    uint32_t *flow_unacked_bytes);

    inline void track_chunk(struct ucclRequest *ureq, struct wr_ex *wr_ex,
                            uint64_t timestamp, uint32_t csn, bool last_chunk) {
        unacked_chunks_.push_back({ureq, wr_ex, timestamp, csn, last_chunk});
    }

    inline size_t track_size(void) { return unacked_chunks_.size(); }

    inline uint64_t track_lookup_ts(uint32_t track_idx) {
        return unacked_chunks_[track_idx].timestamp;
    }

  private:
    std::vector<TXTracking::ChunkTrack> unacked_chunks_;
};

class SubUcclFlow;

struct ack_item {
    SubUcclFlow *subflow;
    struct list_head ack_link;
};

class SubUcclFlow {
  public:
    SubUcclFlow() {}

    SubUcclFlow(uint32_t fid)
        : fid_(fid), in_wheel_cnt_(0), txtracking(), rxtracking(), pcb() {
        INIT_LIST_HEAD(&ack.ack_link);
        ack.subflow = this;
    }

    ~SubUcclFlow() = default;

    // FlowID.
    uint32_t fid_;

    // Next path used in the ACK.
    uint16_t next_ack_path_;

    uint32_t unacked_bytes_ = 0;

    uint32_t backlog_bytes_ = 0;

    // # of chunks in the timing wheel.
    uint32_t in_wheel_cnt_;

    // Protocol Control Block.
    PCB pcb;

    // Whether RTO is armed for the flow.
    bool rto_armed = false;

    // Whether this flow has pending retransmission chunks for no credits.
    bool in_rtx = false;

    // We use list_empty(&flow->ack.ack_link) to check if it has pending ACK to
    // send.
    struct ack_item ack;

    // States for tracking sent chunks.
    TXTracking txtracking;
    // States for tracking received chunks.
    RXTracking rxtracking;

    // RTT scoreboard for each path.
    double scoreboard_rtt_[kPortEntropy];

    inline void update_scoreboard_rtt(uint64_t newrtt_tsc, uint32_t qpidx) {
        scoreboard_rtt_[qpidx] = (1 - kPPEwmaAlpha) * scoreboard_rtt_[qpidx] +
                                 kPPEwmaAlpha * to_usec(newrtt_tsc, freq_ghz);
    }
};

/**
 * @brief UCQPWrapper is a wrapper for ibv_qp with additional information for
 * implementing reliable data transfer.
 */
struct UCQPWrapper {
    struct ibv_qp *qp;
    uint32_t local_psn;
    // A counter for occasionally posting IBV_SEND_SIGNALED flag.
    uint32_t signal_cnt_ = 0;
};

/**
 * @brief UCCL SACK Packet Header for each QP.
 * Multiple SACKs are packed in a single packet transmitted through the Ctrl QP.
 */
struct __attribute__((packed)) UcclSackHdr {
    be16_t fid; // Flow ID
    be16_t path;
    be16_t ackno; // Sequence number to denote the packet counter in the flow.
    be16_t sack_bitmap_count; // Length of the SACK bitmap [0-256].
    be64_t remote_queueing;   // t_ack_sent (SW) - t_remote_nic_rx (HW)
    be64_t sack_bitmap[kSackBitmapSize /
                       PCB::kSackBitmapBucketSize]; // Bitmap of the
                                                    // SACKs received.
};

/**
 * @brief UCCL Pull Packet Header for each QP.
 */
struct __attribute__((packed)) UcclPullHdr {
    be16_t fid;
    be16_t pullno;
};

static const size_t kUcclSackHdrLen = sizeof(UcclSackHdr);
static_assert(kUcclSackHdrLen == 32, "UcclSackHdr size mismatch");
static_assert(CtrlChunkBuffPool::kPktSize >= kUcclSackHdrLen,
              "CtrlChunkBuffPool::PktSize must be larger than UcclSackHdr");

class UcclEngine;

/**
 * @brief RDMA context for a remote peer on an engine, which is produced by
 * RDMAFactory. It contains:
 *   - (Data path QP): Multiple UC/RC QPs and a shared CQ. All data path QPs
 * share the same SRQ.
 *   - (Ctrl QP): A high-priority QP for control messages and a dedicated CQ,
 * PD, and MR.
 */
class RDMAContext {
  public:
    constexpr static int kCtrlMRSize =
        CtrlChunkBuffPool::kChunkSize * CtrlChunkBuffPool::kNumChunk;
    /// TODO: How to determine the size of retransmission MR?
    constexpr static int kRetrMRSize =
        RetrChunkBuffPool::kRetrChunkSize * RetrChunkBuffPool::kNumChunk;
    // 256-bit SACK bitmask => we can track up to 256 packets
    static constexpr std::size_t kReassemblyMaxSeqnoDistance = kSackBitmapSize;

    uint32_t engine_offset_;

    uint32_t flow_cnt_ = 0;

    void *sender_flow_tbl_[MAX_FLOW] = {};
    void *receiver_flow_tbl_[MAX_FLOW] = {};

    // Track outstanding RECV requests.
    struct RecvRequest reqs_[kMaxReq];

    inline uint64_t get_recvreq_id(struct RecvRequest *req) {
        return req - reqs_;
    }

    inline struct RecvRequest *get_recvreq_by_id(int id) { return &reqs_[id]; }

    inline void free_recvreq(struct RecvRequest *req) {
        VLOG(4) << "free_recvreq: " << req;
        memset(req, 0, sizeof(struct RecvRequest));
    }

    /**
     * @brief Get an unused request, if no request is available, return nullptr.
     * @return struct RecvRequest*
     */
    inline struct RecvRequest *alloc_recvreq(void) {
        for (int i = 0; i < kMaxReq; i++) {
            auto *req = &reqs_[i];
            if (req->type == RecvRequest::UNUSED) {
                VLOG(4) << "alloc_recvreq: " << req;
                return req;
            }
        }
        VLOG(4) << "alloc_recvreq: nullptr";
        return nullptr;
    }

    PeerID peer_id_;

    TimerManager *rto_;

    eqds::EQDS *eqds_;

    // Try to arm a timer for the given flow. If the timer is already armed, do
    // nothing.
    void arm_timer_for_flow(void *context);

    // Try to rearm a timer for the given flow. If the timer is not armed, arm
    // it. If the timer is already armed, rearm it.
    void rearm_timer_for_flow(void *context);

    void mark_flow_timeout(void *context);

    void disarm_timer_for_flow(void *context);

    // Remote RDMA context.
    struct RemoteRDMAContext remote_ctx_;

    // Protection domain for all RDMA resources.
    struct ibv_pd *pd_ = nullptr;

    // QPs for data transfer based on UC or RC.
    struct UCQPWrapper dp_qps_[kPortEntropy];

    // Data path QPN to index mapping.
    std::unordered_map<uint32_t, int> qpn2idx_;

    // Shared CQ for all data path QPs.
    struct ibv_cq_ex *send_cq_ex_;
    struct ibv_cq_ex *recv_cq_ex_;
    struct ibv_srq *srq_;

    // (high-priority) QP for credit messages (e.g., pull of EQDS).
    struct ibv_qp *credit_qp_;
    // Local PSN for credit messages.
    uint32_t credit_local_psn_;
    // Remote PSN for credit messages.
    uint32_t credit_remote_psn_;
    // (Engine only) Dedicated CQ for credit messages.
    struct ibv_cq_ex *engine_credit_cq_ex_;
    // (Engine only) Memory region for credit messages.
    struct ibv_mr *engine_credit_mr_;
    // (Pacer only) Dedicated CQ for credit messages.
    struct ibv_cq_ex *pacer_credit_cq_ex_;
    // (Pacer only) Memory region for credit messages.
    struct ibv_mr *pacer_credit_mr_;

    eqds::PacerCreditQPWrapper pc_qpw_;

    // (high-priority) QP for control messages (e.g., ACK).
    struct ibv_qp *ctrl_qp_;
    // Local PSN for control messages.
    uint32_t ctrl_local_psn_;
    // Remote PSN for control messages.
    uint32_t ctrl_remote_psn_;
    // Dedicated CQ for control messages.
    struct ibv_cq_ex *ctrl_cq_ex_;
    // Memory region for control messages.
    struct ibv_mr *ctrl_mr_;

    // Memory region for retransmission.
    struct ibv_mr *retr_mr_;
    struct ibv_mr *retr_hdr_mr_;

    // Global timing wheel for all data path QPs.
    TimingWheel wheel_;

    // The device index that this context belongs to.
    int dev_;

    // RDMA device context per device.
    struct ibv_context *context_;
    // MTU of this device.
    ibv_mtu mtu_;
    uint32_t mtu_bytes_;
    // GID index of this device.
    uint8_t sgid_index_;

    // (Engine) Buffer pool for credit chunks.
    std::optional<eqds::CreditChunkBuffPool> engine_credit_chunk_pool_;

    // (Pacer) Buffer pool for credit chunks.
    std::optional<eqds::CreditChunkBuffPool> pacer_credit_chunk_pool_;

    // Buffer pool for control chunks.
    std::optional<CtrlChunkBuffPool> ctrl_chunk_pool_;

    // Buffer pool for retransmission headers.
    std::optional<RetrHdrBuffPool> retr_hdr_pool_;

    // Buffer pool for retransmission chunks.
    std::optional<RetrChunkBuffPool> retr_chunk_pool_;

    // Buffer pool for work request extension items.
    std::optional<WrExBuffPool> wr_ex_pool_;

    // Pre-allocated WQEs for consuming retransmission chunks.
    struct ibv_recv_wr retr_wrs_[kMaxBatchCQ];

    // WQE for sending ACKs.
    struct ibv_send_wr tx_ack_wr_;

    // Pre-allocated WQEs/SGEs for receiving credits.
    struct ibv_recv_wr rx_credit_wrs_[kPostRQThreshold];
    struct ibv_sge rx_credit_sges_[kPostRQThreshold];
    uint32_t post_credit_rq_cnt_ = 0;

    // Pre-allocated WQEs/SGEs for receiving ACKs.
    struct ibv_recv_wr rx_ack_wrs_[kPostRQThreshold];
    struct ibv_sge rx_ack_sges_[kPostRQThreshold];
    uint32_t post_ctrl_rq_cnt_ = 0;

    // Pre-allocated WQEs for consuming immediate data.
    struct ibv_recv_wr imm_wrs_[kPostRQThreshold];
    uint32_t post_srq_cnt_ = 0;

    double ratio_;
    double offset_;

    // Pending signals need to be polled.
    uint32_t pending_signal_poll_ = 0;

    uint32_t consecutive_same_choice_bytes_ = 0;
    uint32_t last_qp_choice_ = 0;

    uint32_t *engine_unacked_bytes_;

    inline void update_clock(double ratio, double offset) {
        ratio_ = ratio;
        offset_ = offset;
    }

    // Convert NIC clock to host clock (TSC).
    inline uint64_t convert_nic_to_host(uint64_t nic_clock) {
        return ratio_ * nic_clock + offset_;
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
        return next_qp_idx++ % kPortEntropy;
    }

    // Select a QP index randomly.
    inline uint32_t select_qpidx_rand() {
        static thread_local std::mt19937 generator(std::random_device{}());
        std::uniform_int_distribution<uint32_t> distribution(0,
                                                             kPortEntropy - 1);
        return distribution(generator);
    }

    // Select a QP index in a power-of-two manner.
    uint32_t select_qpidx_pot(uint32_t msize, void *subflow_context);

    /**
     * @brief Poll the completion queues for all UC QPs.
     * SQ and RQ use separate completion queues.
     */
    inline int poll_uc_cq(void) {
        int work = 0;
        work += sender_poll_uc_cq();
        work += receiver_poll_uc_cq();
        return work;
    }
    int sender_poll_uc_cq(void);
    int receiver_poll_uc_cq(void);

    /**
     * @brief Poll the completion queues for all RC QPs.
     * SQ and RQ use separate completion queues.
     */
    inline int poll_rc_cq(void) {
        int work = 0;
        work += sender_poll_rc_cq();
        work += receiver_poll_rc_cq();
        return work;
    }
    int sender_poll_rc_cq(void);
    int receiver_poll_rc_cq(void);

    /**
     * @brief Poll the completion queue for the Ctrl QP.
     * SQ and RQ use the same completion queue.
     */
    int poll_ctrl_cq(void);

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
     * @brief Check if we need to post enough recv WQEs to the Ctrl QP.
     * @param force Force to post WQEs.
     */
    void check_ctrl_rq(bool force = false);

    /**
     * @brief Check if we need to post enough recv WQEs to the SRQ.
     * @param force Force to post WQEs.
     */
    void check_srq(bool force = false);

    /**
     * @brief Retransmit a chunk for the given subUcclFlow.
     *
     * @param subflow
     * @param wr_ex
     * @return true Retransmission is successful.
     * @return false Retransmission is failed due to maximum outstanding
     * retransmission chunks or lack of credits (Receiver-driven CC).
     */
    bool try_retransmit_chunk(SubUcclFlow *subflow, struct wr_ex *wr_ex);

    /**
     * @brief Try to drain the retransmission queue for the given subUcclFlow.
     * @param subflow
     */
    void drain_rtx_queue(SubUcclFlow *subflow);

    /**
     * @brief Receive a chunk. Flow infromation is embedded in the immediate
     * data.
     * @param ack_list If this QP needs ACK, add it to the list.
     * @return true If the chunk is received successfully.
     */
    void rx_data(struct list_head *ack_list);

    /**
     * @brief Rceive an ACK from the Ctrl QP.
     * @param pkt_addr The position of the ACK packet in the ACK chunk.
     */
    void rx_ack(uint64_t pkt_addr);

    /**
     * @brief Receive a retransmitted chunk. Flow infromation is embedded in the
     * immediate data.
     * @param ack_list If this QP needs ACK, add it to the list.
     * @return true If the chunk is received successfully.
     */
    void rx_rtx_data(struct list_head *ack_list);

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
    int supply_rx_buff(struct ucclRequest *ureq);

    /**
     * @brief Transmit a message.
     * @param ureq
     * @return true message is transmitted successfully.
     * @return false message is not transmitted successfully.
     */
    bool tx_message(struct ucclRequest *ureq) {
        if constexpr (kReceiverCCA != RECEIVER_CCA_NONE) {
            return receiverCC_tx_message(ureq);
        } else {
            return senderCC_tx_message(ureq);
        }
    }
    bool receiverCC_tx_message(struct ucclRequest *ureq);
    bool senderCC_tx_message(struct ucclRequest *ureq);

    virtual uint32_t EventOnSelectPath(SubUcclFlow *subflow,
                                       uint32_t chunk_size) = 0;

    virtual uint32_t EventOnChunkSize(SubUcclFlow *subflow,
                                      uint32_t remaining_bytes) = 0;

    virtual bool EventOnQueueData(SubUcclFlow *subflow, struct wr_ex *wr_ex,
                                  uint32_t full_chunk_size, uint64_t now) = 0;

    virtual bool EventOnTxRTXData(SubUcclFlow *subflow,
                                  struct wr_ex *wr_ex) = 0;

    virtual void EventOnRxRTXData(SubUcclFlow *subflow, IMMData *imm_data) = 0;

    virtual void EventOnRxData(SubUcclFlow *subflow, IMMData *imm_data) = 0;

    virtual void EventOnRxNACK(SubUcclFlow *subflow, UcclSackHdr *sack_hdr) = 0;

    virtual void EventOnRxACK(SubUcclFlow *subflow, UcclSackHdr *sack_hdr) = 0;

    virtual void EventOnRxCredit(SubUcclFlow *subflow,
                                 eqds::PullQuanta pullno) = 0;

    /**
     * @brief Craft an ACK for a subUcclFlow using the given WR index.
     *
     * @param chunk_addr
     * @param num_sge
     */
    void craft_ack(SubUcclFlow *subflow, uint64_t chunk_addr, int num_sge);

    /**
     * @brief Flush all ACKs in the batch.
     *
     * @param num_ack
     * @param chunk_addr
     */
    void flush_acks(int num_ack, uint64_t chunk_addr);

    /**
     * @brief Transmit a batch of chunks queued in the timing wheel.
     */
    void burst_timing_wheel(void);

    /**
     * @brief Try to update the CSN for the given data path QP.
     * @param qpw
     */
    void try_update_csn(SubUcclFlow *subflow);

    /**
     * @brief Retransmit chunks for the given subUcclFlow.
     * @param rto triggered by RTO or not.
     */
    void __retransmit_for_flow(void *context, bool rto);
    inline void fast_retransmit_for_flow(void *context) {
        if constexpr (ROCE_NET || kTestLoss) {
            __retransmit_for_flow(context, false);
        }
    }
    inline void rto_retransmit_for_flow(void *context) {
        if constexpr (ROCE_NET || kTestLoss) {
            __retransmit_for_flow(context, true);
        }
    }

    void rc_rx_ack(void);

    void rc_rx_data(void);

    std::string to_string();

    RDMAContext(PeerID peer_id, TimerManager *rto, uint32_t *ob,
                eqds::EQDS *eqds, int dev, uint32_t engine_offset,
                union CtrlMeta meta);

    ~RDMAContext(void);

    friend class RDMAFactory;
};

class EQDSRDMAContext : public RDMAContext {
public:
    using RDMAContext::RDMAContext;

    uint32_t EventOnSelectPath(SubUcclFlow *subflow,
                                uint32_t chunk_size) override {
        return select_qpidx_pot(chunk_size, subflow);
    }

    uint32_t EventOnChunkSize(SubUcclFlow *subflow,
                                uint32_t remaining_bytes) override {
        uint32_t chunk_size;

        if constexpr (kSenderCCA != SENDER_CCA_NONE) {
            if (subflow->unacked_bytes_ >= subflow->pcb.timely_cc.get_wnd())
                return 0;
        }

        chunk_size = std::min(kChunkSize, subflow->pcb.eqds_cc.credit());
        chunk_size = std::min(chunk_size, remaining_bytes);
        if (!subflow->pcb.eqds_cc.spend_credit(chunk_size))
            chunk_size = 0;

        return chunk_size;
    }

    bool EventOnQueueData(SubUcclFlow *subflow, struct wr_ex *wr_ex,
                            uint32_t full_chunk_size, uint64_t now) override {
        return false;
    }

    void EventOnRxData(SubUcclFlow *subflow, IMMData *imm_data) override {
        auto *imm = reinterpret_cast<IMMDataEQDS *>(imm_data);
        if (subflow->pcb.eqds_cc.handle_pull_target(imm->GetTarget())) {
            VLOG(5) << "Request pull for new target: " << (uint32_t)imm->GetTarget();
            eqds_->request_pull(&subflow->pcb.eqds_cc);
        }
    }

    bool EventOnTxRTXData(SubUcclFlow *subflow, struct wr_ex *wr_ex) override {
        // We can't retransmit the chunk unless we have enough credits.
        auto permitted_bytes = subflow->pcb.eqds_cc.credit();
        if (permitted_bytes < wr_ex->sge.length ||
            !subflow->pcb.eqds_cc.spend_credit(wr_ex->sge.length)) {
            subflow->in_rtx = true;
            VLOG(5) << "Cannot retransmit chunk due to insufficient credits";
            return false;
        }
        // Re-compute pull target.
        auto pull_target = subflow->pcb.eqds_cc.compute_pull_target(
            subflow, wr_ex->sge.length);
        auto imm_data = IMMDataEQDS(ntohl(wr_ex->wr.imm_data));
        imm_data.SetTarget(pull_target);
        wr_ex->wr.imm_data = htonl(imm_data.GetImmData());
        return true;
    }

    void EventOnRxRTXData(SubUcclFlow *subflow, IMMData *imm_data) override {
        EventOnRxData(subflow, imm_data);
    }

    void EventOnRxACK(SubUcclFlow *subflow, UcclSackHdr *sack_hdr) override {
        // After receiving a valid ACK, we can exit the pending retransmission
        // state.
        subflow->in_rtx = false;
        subflow->pcb.eqds_cc.stop_speculating();
    }

    void EventOnRxNACK(SubUcclFlow *subflow, UcclSackHdr *sack_hdr) override {}

    void EventOnRxCredit(SubUcclFlow *subflow,
                            eqds::PullQuanta pullno) override {
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

    uint32_t EventOnSelectPath(SubUcclFlow *subflow,
                                uint32_t chunk_size) override {
        return select_qpidx_pot(chunk_size, subflow);
    }
  
    uint32_t EventOnChunkSize(SubUcclFlow *subflow, uint32_t remaining_bytes) override {
        
        auto hard_budget = kMaxUnAckedBytesPerEngineHigh - *engine_unacked_bytes_;
        auto soft_budget = kMaxUnAckedBytesPerEngineLow - *engine_unacked_bytes_;
        auto flow_budget = kMaxUnAckedBytesPerFlow - subflow->unacked_bytes_;
        
        auto cc_budget = subflow->pcb.swift_cc.get_wnd() - subflow->unacked_bytes_;
        
        // Enforce swift congestion control window.
        auto ready_bytes = std::min(remaining_bytes, cc_budget);

        // Chunking to kChunkSize.
        ready_bytes = std::min(ready_bytes, kChunkSize);

        // First, check if we have touched the hard budget.
        if (ready_bytes > hard_budget)
            return 0;

        // Second, check if we have touched the soft budget.
        // If we havent touched our per-flow budget, we can ignore the soft budget.
        if (ready_bytes <= soft_budget || ready_bytes <= flow_budget)
            return ready_bytes;
        
        return 0;
    }
  
    bool EventOnQueueData(SubUcclFlow *subflow, struct wr_ex *wr_ex,
                        uint32_t full_chunk_size, uint64_t now) override {
        return false;
    }

    void EventOnRxData(SubUcclFlow *subflow, IMMData *imm_data) override {}

    bool EventOnTxRTXData(SubUcclFlow *subflow, struct wr_ex *wr_ex) override {
        return true;
    }

    void EventOnRxRTXData(SubUcclFlow *subflow, IMMData *imm_data) override {}

    void EventOnRxACK(SubUcclFlow *subflow, UcclSackHdr *sack_hdr) override {}

    void EventOnRxNACK(SubUcclFlow *subflow, UcclSackHdr *sack_hdr) override {}

    void EventOnRxCredit(SubUcclFlow *subflow,
                        eqds::PullQuanta pullno) override {}
  };

class TimelyRDMAContext : public RDMAContext {
public:
    using RDMAContext::RDMAContext;

    uint32_t EventOnSelectPath(SubUcclFlow *subflow,
                                uint32_t chunk_size) override {
        return select_qpidx_pot(chunk_size, subflow);
    }

    uint32_t EventOnChunkSize(SubUcclFlow *subflow,
                                uint32_t remaining_bytes) override {
        auto ready_bytes = std::min(remaining_bytes, kChunkSize);

        if (*engine_unacked_bytes_ + ready_bytes > kMaxUnAckedBytesPerEngineHigh)
            return 0;

        if (*engine_unacked_bytes_ + ready_bytes <= kMaxUnAckedBytesPerEngineLow ||
            subflow->unacked_bytes_ + ready_bytes <= kMaxUnAckedBytesPerFlow) {
            return ready_bytes;
        }
        return 0;
    }

    bool EventOnQueueData(SubUcclFlow *subflow, struct wr_ex *wr_ex,
                            uint32_t full_chunk_size, uint64_t now) override {
        return wheel_.queue_on_timing_wheel(
            subflow->pcb.timely_cc.rate_,
            &subflow->pcb.timely_cc.prev_desired_tx_tsc_, now, wr_ex,
            full_chunk_size, subflow->in_wheel_cnt_ == 0);
    }

    void EventOnRxData(SubUcclFlow *subflow, IMMData *imm_data) override {}

    bool EventOnTxRTXData(SubUcclFlow *subflow, struct wr_ex *wr_ex) override {
        return true;
    }

    void EventOnRxRTXData(SubUcclFlow *subflow, IMMData *imm_data) override {}

    void EventOnRxACK(SubUcclFlow *subflow, UcclSackHdr *sack_hdr) override {}

    void EventOnRxNACK(SubUcclFlow *subflow, UcclSackHdr *sack_hdr) override {}

    void EventOnRxCredit(SubUcclFlow *subflow,
                            eqds::PullQuanta pullno) override {}
};

struct FactoryDevice {
    char ib_name[64];
    std::string local_ip_str;

    struct ibv_context *context;
    struct ibv_device_attr dev_attr;
    struct ibv_port_attr port_attr;

    uint8_t ib_port_num;
    uint8_t gid_idx;
    union ibv_gid gid;

    struct ibv_pd *pd;

    // DMA-BUF support
    bool dma_buf_support;
};

/**
 * @brief Global RDMA factory, which is responsible for
 *  - Initializing the RDMA NIC.
 *  - Creating RDMA contexts for one UcclFlow
 */
class RDMAFactory {

    std::vector<struct FactoryDevice> devices_;

  public:
    ~RDMAFactory() { devices_.clear(); }

    /**
     * @brief Initialize RDMA device.
     */
    static void init_dev(int devname_suffix);
    /**
     * @brief Create a Context object for the given device using the given meta.
     * @param dev
     * @param meta
     * @return RDMAContext*
     */
    static RDMAContext *CreateContext(PeerID peer_id, TimerManager *rto,
                                      uint32_t *engine_unacked_bytes, eqds::EQDS *eqds,
                                      int dev, uint32_t engine_offset_,
                                      union CtrlMeta meta);
    static inline struct FactoryDevice *get_factory_dev(int dev) {
        DCHECK(dev >= 0 && dev < rdma_ctl->devices_.size());
        return &rdma_ctl->devices_[dev];
    }

    std::string to_string(void) const;
};

static inline uint16_t
util_rdma_extract_local_subnet_prefix(uint64_t subnet_prefix) {
    return (be64toh(subnet_prefix) & 0xffff);
}

static inline int modify_qp_rtr_gpuflush(struct ibv_qp *qp, int dev) {
    struct ibv_qp_attr attr;
    int attr_mask = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV |
                    IBV_QP_DEST_QPN | IBV_QP_RQ_PSN;

    memset(&attr, 0, sizeof(attr));

    auto factory_dev = RDMAFactory::get_factory_dev(dev);

    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = factory_dev->port_attr.active_mtu;
    if (ROCE_NET) {
        attr.ah_attr.is_global = 1;
        attr.ah_attr.grh.dgid = factory_dev->gid;
        attr.ah_attr.grh.sgid_index = factory_dev->gid_idx;
        attr.ah_attr.grh.hop_limit = 0xff;
        attr.ah_attr.grh.traffic_class = kTrafficClass;
    } else {
        attr.ah_attr.is_global = 0;
        attr.ah_attr.dlid = factory_dev->port_attr.lid;
    }

    attr.ah_attr.sl = kServiceLevel;

    attr.ah_attr.port_num = IB_PORT_NUM;
    attr.dest_qp_num = qp->qp_num;
    attr.rq_psn = 0;

    attr.min_rnr_timer = 12;
    attr.max_dest_rd_atomic = 1;
    attr_mask |= IBV_QP_MIN_RNR_TIMER | IBV_QP_MAX_DEST_RD_ATOMIC;

    if (FLAGS_v >= 1) {
        std::ostringstream oss;
        oss << "QP#";
        oss << qp->qp_num;
        oss << " RTR(mtu, port_num, sgidx_idx, dest_qp_num, rq_psn):"
            << (uint32_t)attr.path_mtu << "," << (uint32_t)attr.ah_attr.port_num
            << "," << (uint32_t)attr.ah_attr.grh.sgid_index << ","
            << attr.dest_qp_num << "," << attr.rq_psn;
        VLOG(6) << oss.str();
    }

    return ibv_modify_qp(qp, &attr, attr_mask);
}

static inline int modify_qp_rtr(struct ibv_qp *qp, int dev,
                                struct RemoteRDMAContext *remote_ctx,
                                uint32_t remote_qpn, uint32_t remote_psn) {
    struct ibv_qp_attr attr;
    int attr_mask = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV |
                    IBV_QP_DEST_QPN | IBV_QP_RQ_PSN;

    auto factory_dev = RDMAFactory::get_factory_dev(dev);

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = factory_dev->port_attr.active_mtu;
    attr.ah_attr.port_num = IB_PORT_NUM;
    if (ROCE_NET) {
        attr.ah_attr.is_global = 1;
        attr.ah_attr.grh.dgid = remote_ctx->remote_gid;
        attr.ah_attr.grh.sgid_index = factory_dev->gid_idx;
        attr.ah_attr.grh.hop_limit = 0xff;
        attr.ah_attr.grh.traffic_class = kTrafficClass;
    } else {
        if (util_rdma_extract_local_subnet_prefix(
                factory_dev->gid.global.subnet_prefix) !=
            util_rdma_extract_local_subnet_prefix(
                remote_ctx->remote_gid.global.subnet_prefix)) {
            LOG(ERROR) << "Only support same subnet communication for now.";
        }
        attr.ah_attr.is_global = 0;
        attr.ah_attr.dlid = remote_ctx->remote_port_attr.lid;
    }
    attr.ah_attr.sl = kServiceLevel;
    attr.dest_qp_num = remote_qpn;
    attr.rq_psn = remote_psn;

    if (qp->qp_type == IBV_QPT_RC) {
        attr.min_rnr_timer = 12;
        attr.max_dest_rd_atomic = 1;
        attr_mask |= IBV_QP_MIN_RNR_TIMER | IBV_QP_MAX_DEST_RD_ATOMIC;
    }

    if (FLAGS_v >= 1) {
        std::ostringstream oss;
        oss << "QP#";
        oss << qp->qp_num;
        oss << " RTR(mtu, port_num, sgidx_idx, dest_qp_num, rq_psn):"
            << (uint32_t)attr.path_mtu << "," << (uint32_t)attr.ah_attr.port_num
            << "," << (uint32_t)attr.ah_attr.grh.sgid_index << ","
            << attr.dest_qp_num << "," << attr.rq_psn;
        VLOG(6) << oss.str();
    }

    return ibv_modify_qp(qp, &attr, attr_mask);
}

static inline int modify_qp_rts(struct ibv_qp *qp, uint32_t local_psn,
                                bool rc) {
    struct ibv_qp_attr attr;
    int attr_mask = IBV_QP_STATE | IBV_QP_SQ_PSN;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = local_psn;

    if (rc) {
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.max_rd_atomic = 1;
        attr_mask |= IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                     IBV_QP_MAX_QP_RD_ATOMIC;
    }

    if (FLAGS_v >= 1) {
        std::ostringstream oss;
        oss << "QP#";
        oss << qp->qp_num;
        oss << " RTS(sq_psn):" << attr.sq_psn;
        VLOG(6) << oss.str();
    }

    return ibv_modify_qp(qp, &attr, attr_mask);
}

static inline void util_rdma_create_qp_seperate_cq(
    struct ibv_context *context, struct ibv_qp **qp, enum ibv_qp_type qp_type,
    bool cq_ex, bool ts, struct ibv_cq **scq, struct ibv_cq **rcq,
    bool share_cq, uint32_t cqsize, struct ibv_pd *pd, uint32_t max_send_wr,
    uint32_t max_recv_wr, uint32_t max_send_sge, uint32_t max_recv_sge) {
    // Creating SCQ and RCQ
    if (!share_cq) {
        if (cq_ex) {
            struct ibv_cq_init_attr_ex cq_ex_attr;
            cq_ex_attr.cqe = cqsize;
            cq_ex_attr.cq_context = nullptr;
            cq_ex_attr.channel = nullptr;
            cq_ex_attr.comp_vector = 0;
            cq_ex_attr.wc_flags =
                IBV_WC_EX_WITH_BYTE_LEN | IBV_WC_EX_WITH_IMM |
                IBV_WC_EX_WITH_QP_NUM | IBV_WC_EX_WITH_SRC_QP |
                IBV_WC_EX_WITH_COMPLETION_TIMESTAMP; // Timestamp support.
            if constexpr (kTestNoHWTimestamp)
                cq_ex_attr.wc_flags &= ~IBV_WC_EX_WITH_COMPLETION_TIMESTAMP;
            cq_ex_attr.comp_mask = IBV_CQ_INIT_ATTR_MASK_FLAGS;
            cq_ex_attr.flags = IBV_CREATE_CQ_ATTR_SINGLE_THREADED |
                               IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN;
            auto scq_ex = (struct ibv_cq_ex **)scq;
            *scq_ex = ibv_create_cq_ex(context, &cq_ex_attr);
            UCCL_INIT_CHECK(*scq_ex != nullptr, "ibv_create_cq_ex failed");

            auto rcq_ex = (struct ibv_cq_ex **)rcq;
            *rcq_ex = ibv_create_cq_ex(context, &cq_ex_attr);
            UCCL_INIT_CHECK(*rcq_ex != nullptr, "ibv_create_cq_ex failed");
        } else {
            *scq = ibv_create_cq(context, cqsize, nullptr, nullptr, 0);
            UCCL_INIT_CHECK(*scq != nullptr, "ibv_create_cq failed");

            *rcq = ibv_create_cq(context, cqsize, nullptr, nullptr, 0);
            UCCL_INIT_CHECK(*rcq != nullptr, "ibv_create_cq failed");
        }
    }

    struct ibv_qp_init_attr qp_init_attr;
    memset(&qp_init_attr, 0, sizeof(qp_init_attr));

    qp_init_attr.send_cq = *scq;
    qp_init_attr.recv_cq = *rcq;
    qp_init_attr.qp_type = qp_type;

    qp_init_attr.cap.max_send_wr = max_send_wr;
    qp_init_attr.cap.max_recv_wr = max_recv_wr;
    qp_init_attr.cap.max_send_sge = max_send_sge;
    qp_init_attr.cap.max_recv_sge = max_recv_sge;
    // kMaxRecv * sizeof(struct FifoItem)
    qp_init_attr.cap.max_inline_data = kMaxInline;

    // Creating QP
    *qp = ibv_create_qp(pd, &qp_init_attr);
    UCCL_INIT_CHECK(*qp != nullptr, "ibv_create_qp failed");

    // Modifying QP state to INIT
    struct ibv_qp_attr qp_attr;
    int attr_mask =
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = IB_PORT_NUM;
    qp_attr.qp_access_flags =
        IBV_ACCESS_REMOTE_WRITE |
        ((qp_type == IBV_QPT_RC) ? IBV_ACCESS_REMOTE_READ : 0);

    UCCL_INIT_CHECK(ibv_modify_qp(*qp, &qp_attr, attr_mask) == 0,
                    "ibv_modify_qp failed");
}

static inline void
util_rdma_create_qp(struct ibv_context *context, struct ibv_qp **qp,
                    enum ibv_qp_type qp_type, bool cq_ex, bool ts,
                    struct ibv_cq **cq, bool share_cq, uint32_t cqsize,
                    struct ibv_pd *pd, struct ibv_mr **mr, void *addr,
                    size_t mr_size, uint32_t max_send_wr, uint32_t max_recv_wr,
                    uint32_t max_send_sge, uint32_t max_recv_sge) {
    // Creating CQ
    if (!share_cq) {
        if (cq_ex) {
            struct ibv_cq_init_attr_ex cq_ex_attr;
            cq_ex_attr.cqe = cqsize;
            cq_ex_attr.cq_context = nullptr;
            cq_ex_attr.channel = nullptr;
            cq_ex_attr.comp_vector = 0;
            cq_ex_attr.wc_flags =
                IBV_WC_EX_WITH_BYTE_LEN | IBV_WC_EX_WITH_IMM |
                IBV_WC_EX_WITH_QP_NUM | IBV_WC_EX_WITH_SRC_QP |
                IBV_WC_EX_WITH_COMPLETION_TIMESTAMP; // Timestamp support.
            if constexpr (kTestNoHWTimestamp)
                cq_ex_attr.wc_flags &= ~IBV_WC_EX_WITH_COMPLETION_TIMESTAMP;
            cq_ex_attr.comp_mask = IBV_CQ_INIT_ATTR_MASK_FLAGS;
            cq_ex_attr.flags = IBV_CREATE_CQ_ATTR_SINGLE_THREADED |
                               IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN;
            auto cq_ex = (struct ibv_cq_ex **)cq;
            *cq_ex = ibv_create_cq_ex(context, &cq_ex_attr);
            UCCL_INIT_CHECK(*cq_ex != nullptr, "ibv_create_cq_ex failed");
        } else {
            *cq = ibv_create_cq(context, cqsize, nullptr, nullptr, 0);
            UCCL_INIT_CHECK(*cq != nullptr, "ibv_create_cq failed");
        }
    }

    // Creating MR
    if (addr == nullptr) {
        addr = mmap(nullptr, mr_size, PROT_READ | PROT_WRITE,
                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        UCCL_INIT_CHECK(addr != MAP_FAILED, "mmap failed");
    }
    memset(addr, 0, mr_size);

    *mr =
        ibv_reg_mr(pd, addr, mr_size,
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                       ((qp_type == IBV_QPT_RC) ? IBV_ACCESS_REMOTE_READ : 0));
    UCCL_INIT_CHECK(*mr != nullptr, "ibv_reg_mr failed");

    struct ibv_qp_init_attr qp_init_attr;
    memset(&qp_init_attr, 0, sizeof(qp_init_attr));

    qp_init_attr.send_cq = *cq;
    qp_init_attr.recv_cq = *cq;
    qp_init_attr.qp_type = qp_type;

    qp_init_attr.cap.max_send_wr = max_send_wr;
    qp_init_attr.cap.max_recv_wr = max_recv_wr;
    qp_init_attr.cap.max_send_sge = max_send_sge;
    qp_init_attr.cap.max_recv_sge = max_recv_sge;
    // kMaxRecv * sizeof(struct FifoItem)
    qp_init_attr.cap.max_inline_data = kMaxInline;

    // Creating QP
    *qp = ibv_create_qp(pd, &qp_init_attr);
    UCCL_INIT_CHECK(*qp != nullptr, "ibv_create_qp failed");

    // Modifying QP state to INIT
    struct ibv_qp_attr qp_attr;
    int attr_mask =
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = IB_PORT_NUM;
    qp_attr.qp_access_flags =
        IBV_ACCESS_REMOTE_WRITE |
        ((qp_type == IBV_QPT_RC) ? IBV_ACCESS_REMOTE_READ : 0);

    UCCL_INIT_CHECK(ibv_modify_qp(*qp, &qp_attr, attr_mask) == 0,
                    "ibv_modify_qp failed");
}

/**
 * @brief This helper function converts an Infiniband name (e.g., mlx5_0) to an
 * Ethernet name (e.g., eth0)
 * @return int -1 on error, 0 on success
 */
static inline int util_rdma_ib2eth_name(const char *ib_name,
                                        char *ethernet_name) {
    char command[512];
    snprintf(command, sizeof(command),
             "ls -l /sys/class/infiniband/%s/device/net | sed -n '2p' | sed "
             "'s/.* //'",
             ib_name);
    FILE *fp = popen(command, "r");
    if (fp == nullptr) {
        perror("popen");
        return -1;
    }
    if (fgets(ethernet_name, 64, fp) == NULL) {
        pclose(fp);
        return -1;
    }
    pclose(fp);
    // Remove newline character if present
    ethernet_name[strcspn(ethernet_name, "\n")] = '\0';
    return 0;
}

/**
 * @brief This helper function gets the Infiniband name from the suffix.
 *
 * @param suffix
 * @param ib_name
 * @return int
 */
static inline int util_rdma_get_ib_name_from_suffix(int suffix, char *ib_name) {
    sprintf(ib_name, "%s%d", IB_DEVICE_NAME_PREFIX, suffix);
    return 0;
}

/**
 * @brief This helper function gets the IP address of the device from Infiniband
 * name.
 *
 * @param ib_name
 * @param ip
 * @return int
 */
static inline int util_rdma_get_ip_from_ib_name(const char *ib_name,
                                                std::string *ip) {
    char ethernet_name[64];
    if (util_rdma_ib2eth_name(ib_name, ethernet_name)) {
        return -1;
    }

    *ip = get_dev_ip(ethernet_name);

    return *ip == "" ? -1 : 0;
}

static inline int util_rdma_get_mtu_from_ibv_mtu(ibv_mtu mtu) {
    switch (mtu) {
    case IBV_MTU_256:
        return 256;
    case IBV_MTU_512:
        return 512;
    case IBV_MTU_1024:
        return 1024;
    case IBV_MTU_2048:
        return 2048;
    case IBV_MTU_4096:
        return 4096;
    default:
        return 0;
    }
}

} // namespace uccl

#endif