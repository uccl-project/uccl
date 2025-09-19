#ifndef RDMA_IO_H
#define RDMA_IO_H

#include "eqds.h"
#include "pcb.h"
#include "transport_config.h"
#include "util/endian.h"
#include "util/list.h"
#include "util/util.h"
#include "util_buffpool.h"
#include "util_rdma.h"
#include <glog/logging.h>
#include <infiniband/verbs.h>
#include <cstdint>
#include <cstring>
#include <set>
#include <unordered_map>
#include <vector>
#include <sys/mman.h>
#ifndef __HIP_PLATFORM_AMD__
#include <cuda_runtime.h>
#else
#include <hip/hip_runtime.h>
#endif

namespace uccl {
typedef uint64_t FlowID;
typedef uint64_t PeerID;

class RDMAContext;
class RDMAFactory;
class TXTracking;
extern std::shared_ptr<RDMAFactory> rdma_ctl;

struct CQEDesc {
  uint64_t data;
  uint64_t ts;
};

class CQEDescPool : public BuffPool {
 public:
  static constexpr size_t kDescSize = sizeof(struct CQEDesc);
  static constexpr uint32_t kNumDesc = 4 * 65536;
  static_assert((kNumDesc & (kNumDesc - 1)) == 0,
                "kNumDesc must be power of 2");
  CQEDescPool(struct ibv_mr* mr) : BuffPool(kNumDesc, kDescSize, mr) {}

  ~CQEDescPool() = default;
};

struct __attribute__((packed)) retr_chunk_hdr {
  // Target address for the lost chunk.
  uint64_t remote_addr;
  uint32_t imm_data;
};
static_assert(sizeof(struct retr_chunk_hdr) == 12,
              "retr_chunk_hdr size is not 12 bytes");

/**
 * @brief Buffer pool for retransmission chunks (original chunk + retransmission
 * header). Original chunk and retransmission header are transmitted through
 * scatter-gather list.
 */

class RetrChunkBuffPool : public BuffPool {
 public:
  static constexpr uint32_t kNumChunk = 4096;
  static_assert((kNumChunk & (kNumChunk - 1)) == 0,
                "kNumChunk must be power of 2");

  RetrChunkBuffPool(struct ibv_mr* mr)
      : BuffPool(kNumChunk, kRetrChunkSize, mr) {}

  ~RetrChunkBuffPool() = default;
};

/**
 * @brief Buffer pool for retransmission headers.
 */
class RetrHdrBuffPool : public BuffPool {
 public:
  static constexpr size_t kHdrSize = sizeof(struct retr_chunk_hdr);
  static constexpr uint32_t kNumHdr = 4096;
  static_assert((kNumHdr & (kNumHdr - 1)) == 0, "kNumHdr must be power of 2");

  RetrHdrBuffPool(struct ibv_mr* mr) : BuffPool(kNumHdr, kHdrSize, mr) {}

  ~RetrHdrBuffPool() = default;
};

/**
 * @brief Buffer pool for control packets.
 */
class CtrlChunkBuffPool : public BuffPool {
 public:
  static constexpr size_t kPktSize = 36;
  static constexpr size_t kChunkSize = kPktSize * kMaxBatchCQ;
  static constexpr uint32_t kNumChunk = 65536;
  static_assert((kNumChunk & (kNumChunk - 1)) == 0,
                "kNumChunk must be power of 2");

  CtrlChunkBuffPool(struct ibv_mr* mr) : BuffPool(kNumChunk, kChunkSize, mr) {}

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
    std::atomic<int>* next_install_engine;
  } install_ctx;

  // kInstallFlow
  struct {
    FlowID flow_id;
    void* context;
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

void serialize_fifo_item(FifoItem const& item, char* buf);
void deserialize_fifo_item(char const* buf, FifoItem* item);

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
  uint32_t remote_ctrl_qpn;
  uint32_t remote_peer_id;
  struct ibv_ah* dest_ah;
};

enum ReqType { ReqTx, ReqRx, ReqFlush, ReqTxRC, ReqRxRC, ReqRead, ReqWrite };

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
    int mid;  // used for multi-send
  };
  union {
    PollCtx* poll_ctx;
    // For reducing overhead of PollCtx for RC and Flush operation.
    uint64_t rc_or_flush_done;
  };
  void* context;
  void* req_pool;
  uint32_t engine_idx;
  union {
    struct {
      int data_len[kMaxRecv];
      uint64_t data[kMaxRecv];
      struct FifoItem* elems;
      struct ibv_send_wr wr;
      struct ibv_sge sge;
      struct ibv_qp* qp;
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
      uint32_t acked_bytes;  // RC only.
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
  struct ucclRequest* ureq;
  uint32_t received_bytes[kMaxRecv];
  uint32_t fin_msg;
};

/// @ref ncclIbNetCommBase
struct alignas(32) NetCommBase {
  // Pointing to rdma_ctx_->fifo_mr_->addr.
  struct RemFifo* fifo;

  // CQ for Fifo QP and GPU flush QP and RC QP.
  struct ibv_cq* flow_cq;

  // Fifo QP based on Reliable Connection (RC).
  struct ibv_qp* fifo_qp;
  // Memory region for Fifo.
  struct ibv_mr* fifo_mr;

  // RC UP for small messages bypassing UcclEngine.
  struct ibv_qp* rc_qp;

  uint64_t remote_fifo_addr;
  uint32_t remote_fifo_rkey;
};

/// @ref ncclIbSendComm
struct SendComm {
  struct NetCommBase base;
  // Track outstanding FIFO requests.
  struct ucclRequest* fifo_ureqs[kMaxReq][kMaxRecv];
  uint64_t fifo_head;
};

/// @ref ncclIbRecvComm
struct RecvComm {
  struct NetCommBase base;

  // QP for GPU flush.
  struct ibv_qp* gpu_flush_qp;
  // Memory region for GPU flush.
  struct ibv_mr* gpu_flush_mr;
  struct ibv_sge gpu_flush_sge;
  // GPU flush buffer
  int gpu_flush;
};

class RXTracking {
 public:
  std::set<std::pair<UINT_CSN, void*>> ready_csn_;

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
    struct ucclRequest* ureq;
    struct wr_ex* wr_ex;
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

  std::pair<uint64_t, uint32_t> ack_rc_transmitted_chunks(
      void* subflow_context, RDMAContext* rdma_ctx, UINT_CSN csn, uint64_t now,
      uint32_t* flow_unacked_bytes, uint32_t* engine_outstanding_bytes);

  uint64_t ack_transmitted_chunks(void* subflow_context, RDMAContext* rdma_ctx,
                                  uint32_t num_acked_chunks, uint64_t t5,
                                  uint64_t t6, uint64_t remote_queueing_tsc,
                                  uint32_t* flow_unacked_bytes);

  inline void track_chunk(struct ucclRequest* ureq, struct wr_ex* wr_ex,
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
  SubUcclFlow* subflow;
  struct list_head ack_link;
};

class SubUcclFlow {
 public:
  SubUcclFlow() {}

  SubUcclFlow(uint32_t fid, double link_bandwidth)
      : fid_(fid),
        in_wheel_cnt_(0),
        txtracking(),
        rxtracking(),
        pcb(link_bandwidth) {
    INIT_LIST_HEAD(&ack.ack_link);
    ack.subflow = this;

    scoreboard_rtt_.resize(ucclParamPORT_ENTROPY());
  }

  ~SubUcclFlow() = default;

  // FlowID.
  uint32_t fid_;

  // Next path used in the ACK.
  uint16_t next_ack_path_;

  // Unacknowledged bytes.
  uint32_t unacked_bytes_ = 0;

  // # of chunks in the timing wheel.
  uint32_t in_wheel_cnt_;

  // Protocol Control Block.
  PCB pcb;

  // Used by EQDS.
  uint32_t backlog_bytes_ = 0;

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
  std::vector<double> scoreboard_rtt_;

  // Track outstanding RECV requests.
  // When a flow wants to receive message, it should allocate a request from
  // this pool.
  struct RecvRequest reqs_[kMaxReq];

  // Get an unused request, if no request is available, return nullptr.
  inline struct RecvRequest* alloc_recvreq(void) {
    for (int i = 0; i < kMaxReq; i++) {
      auto* req = &reqs_[i];
      if (req->type == RecvRequest::UNUSED) {
        return req;
      }
    }
    return nullptr;
  }

  // Get the ID of the request.
  inline uint64_t get_recvreq_id(struct RecvRequest* req) {
    return req - reqs_;
  }

  // Get the request by ID.
  inline struct RecvRequest* get_recvreq_by_id(int id) { return &reqs_[id]; }

  // Free the request.
  inline void free_recvreq(struct RecvRequest* req) {
    memset(req, 0, sizeof(struct RecvRequest));
  }

  inline void update_scoreboard_rtt(uint64_t newrtt_tsc, uint32_t qpidx) {
    scoreboard_rtt_[qpidx] = (1 - kPPEwmaAlpha) * scoreboard_rtt_[qpidx] +
                             kPPEwmaAlpha * to_usec(newrtt_tsc, freq_ghz);
  }
};

/**
 * @brief QPWrapper is a wrapper for ibv_qp with additional information for
 * implementing reliable data transfer.
 */
struct QPWrapper {
  struct ibv_qp* qp;
  // A counter for occasionally posting IBV_SEND_SIGNALED flag.
  uint32_t signal_cnt_ = 0;
};

/**
 * @brief UCCL SACK Packet Header for each QP.
 * Multiple SACKs are packed in a single packet transmitted through the Ctrl QP.
 */
struct __attribute__((packed)) UcclSackHdr {
  be16_t peer_id;  // Peer ID
  be16_t fid;      // Flow ID
  be16_t path;
  be16_t ackno;  // Sequence number to denote the packet counter in the flow.
  be16_t sack_bitmap_count;  // Length of the SACK bitmap [0-256].
  be16_t padding;
  be64_t remote_queueing;  // t_ack_sent (SW) - t_remote_nic_rx (HW)
  be64_t sack_bitmap[kSackBitmapSize /
                     PCB::kSackBitmapBucketSize];  // Bitmap of the
                                                   // SACKs received.
};

/**
 * @brief UCCL Pull Packet Header for each QP.
 */
struct __attribute__((packed)) UcclPullHdr {
  be16_t fid;
  be16_t pullno;
};

static size_t const kUcclSackHdrLen = sizeof(UcclSackHdr);
static_assert(kUcclSackHdrLen == 36, "UcclSackHdr size mismatch");
static_assert(CtrlChunkBuffPool::kPktSize >= kUcclSackHdrLen,
              "CtrlChunkBuffPool::PktSize must be larger than UcclSackHdr");

class UcclEngine;

struct RecvWRs {
  struct ibv_recv_wr recv_wrs[kPostRQThreshold];
  struct ibv_sge recv_sges[kPostRQThreshold];
  uint32_t post_rq_cnt = 0;
};

class SharedIOContext;

struct FactoryDevice {
  char ib_name[64];
  std::string local_ip_str;
  int numa_node;

  bool support_cq_ex;
  bool support_uc;
  struct ibv_context* context;
  struct ibv_device_attr dev_attr;
  struct ibv_port_attr port_attr;

  uint8_t ib_port_num;
  int gid_idx;
  bool is_roce;
  union ibv_gid gid;

  double link_bw;

  struct ibv_pd* pd;

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
  std::map<int, int> gpu_to_dev_idx_;
  int num_devices = 0;

 public:
  ~RDMAFactory() { devices_.clear(); }

  /**
   * @brief Initialize all eligible RDMA devices.
   */
  static int init_devs();

  static RDMAContext* CreateContext(TimerManager* rto,
                                    uint32_t* engine_unacked_bytes,
                                    eqds::EQDS* eqds, int dev,
                                    uint32_t engine_offset_,
                                    union CtrlMeta meta,
                                    SharedIOContext* io_ctx);

  static inline struct FactoryDevice* get_factory_dev(int dev) {
    DCHECK(dev >= 0 && dev < rdma_ctl->devices_.size());
    return &rdma_ctl->devices_[dev];
  }

  static inline bool is_roce(int dev) {
    DCHECK(dev >= 0 && dev < rdma_ctl->devices_.size());
    return rdma_ctl->devices_[dev].is_roce;
  }

  static inline int get_best_dev_idx(int gpu_idx) {
    auto it = rdma_ctl->gpu_to_dev_idx_.find(gpu_idx);
    CHECK(it != rdma_ctl->gpu_to_dev_idx_.end())
        << "Error: gpu_to_dev_idx_[" << gpu_idx << "] is not set.";
    return it->second;
  }

  std::string to_string(void) const;
};

static inline int modify_qp_rtr_gpuflush(struct ibv_qp* qp, int dev) {
  struct ibv_qp_attr attr;
  int attr_mask = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV | IBV_QP_DEST_QPN |
                  IBV_QP_RQ_PSN;

  memset(&attr, 0, sizeof(attr));

  auto factory_dev = RDMAFactory::get_factory_dev(dev);

  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = factory_dev->port_attr.active_mtu;
  if (RDMAFactory::is_roce(dev)) {
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid = factory_dev->gid;
    attr.ah_attr.grh.sgid_index = factory_dev->gid_idx;
    attr.ah_attr.grh.hop_limit = 0xff;
    attr.ah_attr.grh.traffic_class = ucclParamROCE_TRAFFIC_CLASS();
    attr.ah_attr.sl = ucclParamROCE_SERVICE_LEVEL();
  } else {
    attr.ah_attr.is_global = 0;
    attr.ah_attr.dlid = factory_dev->port_attr.lid;
    attr.ah_attr.sl = ucclParamIB_SERVICE_LEVEL();
  }

  attr.ah_attr.port_num = factory_dev->ib_port_num;
  attr.dest_qp_num = qp->qp_num;
  attr.rq_psn = BASE_PSN;

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

static inline int modify_qp_rtr(struct ibv_qp* qp, int dev,
                                struct RemoteRDMAContext* remote_ctx,
                                uint32_t remote_qpn) {
  struct ibv_qp_attr attr;
  int attr_mask = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV | IBV_QP_DEST_QPN |
                  IBV_QP_RQ_PSN;

  auto factory_dev = RDMAFactory::get_factory_dev(dev);

  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = factory_dev->port_attr.active_mtu;
  attr.ah_attr.port_num = factory_dev->ib_port_num;
  if (RDMAFactory::is_roce(dev)) {
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid = remote_ctx->remote_gid;
    attr.ah_attr.grh.sgid_index = factory_dev->gid_idx;
    attr.ah_attr.grh.hop_limit = 0xff;
    attr.ah_attr.grh.traffic_class = ucclParamROCE_TRAFFIC_CLASS();
    attr.ah_attr.sl = ucclParamROCE_SERVICE_LEVEL();
  } else {
    if (util_rdma_extract_local_subnet_prefix(
            factory_dev->gid.global.subnet_prefix) !=
        util_rdma_extract_local_subnet_prefix(
            remote_ctx->remote_gid.global.subnet_prefix)) {
      LOG(ERROR) << "Only support same subnet communication for now.";
    }
    attr.ah_attr.is_global = 0;
    attr.ah_attr.dlid = remote_ctx->remote_port_attr.lid;
    attr.ah_attr.sl = ucclParamIB_SERVICE_LEVEL();
  }
  attr.dest_qp_num = remote_qpn;
  attr.rq_psn = BASE_PSN;

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

static inline int modify_qp_rts(struct ibv_qp* qp, bool rc) {
  struct ibv_qp_attr attr;
  int attr_mask = IBV_QP_STATE | IBV_QP_SQ_PSN;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = BASE_PSN;

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

struct pair_hash {
  template <class T1, class T2>
  std::size_t operator()(std::pair<T1, T2> const& p) const {
    return std::hash<T1>()(p.first) ^ std::hash<T2>()(p.second);
  }
};

// Shared IO context for each UCCL engine.
class SharedIOContext {
 public:
  SharedIOContext(int dev) {
    support_cq_ex_ = RDMAFactory::get_factory_dev(dev)->support_cq_ex;
    rc_mode_ = ucclParamRCMode();
    auto support_uc = RDMAFactory::get_factory_dev(dev)->support_uc;
    auto ib_name = RDMAFactory::get_factory_dev(dev)->ib_name;
    if (rc_mode_) {
      if (support_uc) {
        printf(
            "Using RC on dev %s (per UCCL_RCMODE=1). Note it supports UC, so "
            "using RC may give less optimized/scalable performance.\n",
            ib_name);
      }
    } else {
      rc_mode_ = !support_uc;
    }
    bypass_pacing_ = ucclParamBypassPacing();
    auto context = RDMAFactory::get_factory_dev(dev)->context;
    auto pd = RDMAFactory::get_factory_dev(dev)->pd;
    auto port = RDMAFactory::get_factory_dev(dev)->ib_port_num;
    if (support_cq_ex_) {
      send_cq_ex_ = util_rdma_create_cq_ex(context, kCQSize);
      recv_cq_ex_ = util_rdma_create_cq_ex(context, kCQSize);
    } else {
      send_cq_ex_ = (struct ibv_cq_ex*)util_rdma_create_cq(context, kCQSize);
      recv_cq_ex_ = (struct ibv_cq_ex*)util_rdma_create_cq(context, kCQSize);
    }
    UCCL_INIT_CHECK(send_cq_ex_ != nullptr, "util_rdma_create_cq_ex failed");
    UCCL_INIT_CHECK(recv_cq_ex_ != nullptr, "util_rdma_create_cq_ex failed");

    if (support_cq_ex_) {
      int ret =
          util_rdma_modify_cq_attr(send_cq_ex_, kCQMODCount, kCQMODPeriod);
      UCCL_INIT_CHECK(ret == 0, "util_rdma_modify_cq_attr failed");
      ret = util_rdma_modify_cq_attr(recv_cq_ex_, kCQMODCount, kCQMODPeriod);
      UCCL_INIT_CHECK(ret == 0, "util_rdma_modify_cq_attr failed");
    }

    srq_ = util_rdma_create_srq(pd, kMaxSRQ, 1, 0);
    UCCL_INIT_CHECK(srq_ != nullptr, "util_rdma_create_srq failed");

    retr_mr_ = util_rdma_create_host_memory_mr(
        pd, kRetrChunkSize * RetrChunkBuffPool::kNumChunk);
    retr_hdr_mr_ = util_rdma_create_host_memory_mr(
        pd, RetrHdrBuffPool::kNumHdr * RetrHdrBuffPool::kHdrSize);

    // Initialize retransmission chunk and header buffer pool.
    retr_chunk_pool_.emplace(retr_mr_);
    retr_hdr_pool_.emplace(retr_hdr_mr_);

    cq_desc_mr_ = util_rdma_create_host_memory_mr(
        pd, CQEDescPool::kNumDesc * CQEDescPool::kDescSize);
    cq_desc_pool_.emplace(cq_desc_mr_);

    // Populate recv work requests to SRQ for consuming immediate data.
    inc_post_srq(kMaxSRQ);
    while (get_post_srq_cnt() > 0) {
      check_srq(true);
    }

    if (!rc_mode_) {
      // Create Ctrl QP, CQ, and MR.
      util_rdma_create_qp(
          context, &ctrl_qp_, IBV_QPT_UD, support_cq_ex_, true,
          (struct ibv_cq**)&ctrl_cq_ex_, false, kCQSize, pd, port, &ctrl_mr_,
          nullptr, CtrlChunkBuffPool::kChunkSize * CtrlChunkBuffPool::kNumChunk,
          kMaxCtrlWRs, kMaxCtrlWRs, 1, 1);

      struct ibv_qp_attr attr = {};
      attr.qp_state = IBV_QPS_RTR;
      UCCL_INIT_CHECK(ibv_modify_qp(ctrl_qp_, &attr, IBV_QP_STATE) == 0,
                      "ibv_modify_qp failed: ctrl qp rtr");

      memset(&attr, 0, sizeof(attr));
      attr.qp_state = IBV_QPS_RTS;
      attr.sq_psn = BASE_PSN;
      UCCL_INIT_CHECK(
          ibv_modify_qp(ctrl_qp_, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN) == 0,
          "ibv_modify_qp failed: ctrl qp rts");

      // Initialize Control packet buffer pool.
      ctrl_chunk_pool_.emplace(ctrl_mr_);

      // Populate recv work requests on Ctrl QP for consuming control packets.
      {
        for (int i = 0; i < kPostRQThreshold; i++) {
          ctrl_recv_wrs_.recv_sges[i].lkey = get_ctrl_chunk_lkey();
          ctrl_recv_wrs_.recv_sges[i].length = CtrlChunkBuffPool::kChunkSize;
          ctrl_recv_wrs_.recv_wrs[i].sg_list = &ctrl_recv_wrs_.recv_sges[i];
          ctrl_recv_wrs_.recv_wrs[i].num_sge = 1;
        }

        inc_post_ctrl_rq(kMaxCtrlWRs);
        while (get_post_ctrl_rq_cnt() > 0) {
          check_ctrl_rq(true);
        }
        for (int i = 0; i < kMaxAckWRs; i++) {
          memset(&tx_ack_wr_[i], 0, sizeof(tx_ack_wr_[i]));
          memset(&tx_ack_sge_[i], 0, sizeof(tx_ack_sge_[i]));
          tx_ack_wr_[i].sg_list = &tx_ack_sge_[i];
          tx_ack_wr_[i].num_sge = 1;
          tx_ack_wr_[i].opcode = IBV_WR_SEND_WITH_IMM;
          tx_ack_wr_[i].send_flags = IBV_SEND_SIGNALED;
        }
      }
    }
  }

  ~SharedIOContext() {
    ibv_destroy_cq(ibv_cq_ex_to_cq(send_cq_ex_));
    ibv_destroy_cq(ibv_cq_ex_to_cq(recv_cq_ex_));
    ibv_destroy_srq(srq_);
    ibv_dereg_mr(retr_mr_);
    ibv_dereg_mr(retr_hdr_mr_);
  }

  inline bool is_rc_mode() { return rc_mode_; }

  inline bool bypass_pacing() { return bypass_pacing_; }

  void flush_acks();

  // This should be extremely fast, probably even faster than C func pointer.
  int poll_ctrl_cq(void) {
    return support_cq_ex_ ? _poll_ctrl_cq_ex() : _poll_ctrl_cq_normal();
  }
  int uc_poll_send_cq(void) {
    return support_cq_ex_ ? _uc_poll_send_cq_ex() : _uc_poll_send_cq_normal();
  }
  int uc_poll_recv_cq(void) {
    return support_cq_ex_ ? _uc_poll_recv_cq_ex() : _uc_poll_recv_cq_normal();
  }
  int rc_poll_send_cq(void) {
    return support_cq_ex_ ? _rc_poll_send_cq_ex() : _rc_poll_send_cq_normal();
  }
  int rc_poll_recv_cq(void) {
    return support_cq_ex_ ? _rc_poll_recv_cq_ex() : _rc_poll_recv_cq_normal();
  }

  int _poll_ctrl_cq_normal(void);
  int _poll_ctrl_cq_ex(void);

  int _uc_poll_send_cq_normal(void);
  int _uc_poll_send_cq_ex(void);

  int _uc_poll_recv_cq_normal(void);
  int _uc_poll_recv_cq_ex(void);

  int _rc_poll_send_cq_normal(void);
  int _rc_poll_send_cq_ex(void);

  int _rc_poll_recv_cq_normal(void);
  int _rc_poll_recv_cq_ex(void);

  void check_srq(bool force);

  void check_ctrl_rq(bool force = false);

  inline uint32_t get_retr_hdr_lkey(void) { return retr_hdr_pool_->get_lkey(); }

  inline uint32_t get_retr_chunk_lkey(void) {
    return retr_chunk_pool_->get_lkey();
  }

  inline uint32_t get_ctrl_chunk_lkey(void) {
    return ctrl_chunk_pool_->get_lkey();
  }

  inline void push_cqe_desc(CQEDesc* desc) {
    uint64_t addr = (uint64_t)desc;
    cq_desc_pool_->free_buff(addr);
  }

  inline CQEDesc* pop_cqe_desc() {
    uint64_t addr;
    DCHECK(cq_desc_pool_->alloc_buff(&addr) == 0)
        << "Failed to allocate buffer for CQE descriptor";
    return reinterpret_cast<CQEDesc*>(addr);
  }

  inline void push_retr_hdr(uint64_t addr) { retr_hdr_pool_->free_buff(addr); }
  inline uint64_t pop_retr_hdr() {
    uint64_t addr;
    DCHECK(retr_hdr_pool_->alloc_buff(&addr) == 0)
        << "Failed to allocate buffer for retransmission header";
    return addr;
  }
  inline void push_retr_chunk(uint64_t addr) {
    retr_chunk_pool_->free_buff(addr);
  }
  inline uint64_t pop_retr_chunk() {
    uint64_t addr;
    DCHECK(retr_chunk_pool_->alloc_buff(&addr) == 0)
        << "Failed to allocate buffer for retransmission chunk";
    return addr;
  }

  inline void push_ctrl_chunk(uint64_t addr) {
    ctrl_chunk_pool_->free_buff(addr);
  }
  inline uint64_t pop_ctrl_chunk() {
    uint64_t addr;
    DCHECK(ctrl_chunk_pool_->alloc_buff(&addr) == 0)
        << "Failed to allocate buffer for control chunk";
    return addr;
  }

  inline RDMAContext* qpn_to_rdma_ctx(int qp_num) {
    return qpn_to_rdma_ctx_map_[qp_num];
  }

  inline void record_qpn_ctx_mapping(int qp_num, RDMAContext* ctx) {
    DCHECK(qpn_to_rdma_ctx_map_.find(qp_num) == qpn_to_rdma_ctx_map_.end())
        << "QP " << qp_num << " already exists";
    qpn_to_rdma_ctx_map_[qp_num] = ctx;
  }

  inline RDMAContext* find_rdma_ctx(uint32_t peer_id, uint32_t fid) {
    return fid_to_rdma_ctx_map_[std::make_pair(peer_id, fid)];
  }

  inline void record_sender_ctx_mapping(uint32_t peer_id, uint32_t fid,
                                        RDMAContext* ctx) {
    DCHECK(fid_to_rdma_ctx_map_.find(std::make_pair(peer_id, fid)) ==
           fid_to_rdma_ctx_map_.end())
        << "FID " << fid << " already exists";
    fid_to_rdma_ctx_map_[std::make_pair(peer_id, fid)] = ctx;
  }

  inline void inc_post_srq(void) { dp_recv_wrs_.post_rq_cnt++; }
  inline void inc_post_srq(int n) { dp_recv_wrs_.post_rq_cnt += n; }
  inline void dec_post_srq(void) { dp_recv_wrs_.post_rq_cnt--; }
  inline void dec_post_srq(int n) { dp_recv_wrs_.post_rq_cnt -= n; }
  inline int get_post_srq_cnt(void) { return dp_recv_wrs_.post_rq_cnt; }

  inline void inc_post_ctrl_rq(void) { ctrl_recv_wrs_.post_rq_cnt++; }
  inline void inc_post_ctrl_rq(int n) { ctrl_recv_wrs_.post_rq_cnt += n; }
  inline void dec_post_ctrl_rq(void) { ctrl_recv_wrs_.post_rq_cnt--; }
  inline void dec_post_ctrl_rq(int n) { ctrl_recv_wrs_.post_rq_cnt -= n; }
  inline int get_post_ctrl_rq_cnt(void) { return ctrl_recv_wrs_.post_rq_cnt; }

 private:
  // Whether the NIC supports CQ-EX. Mellanox NICs support CQ-EX.
  bool support_cq_ex_;

  bool rc_mode_;

  bool bypass_pacing_;

  struct ibv_qp* ctrl_qp_;
  struct ibv_cq_ex* ctrl_cq_ex_;
  struct ibv_mr* ctrl_mr_;

  // Shared CQ for all data path QPs.
  struct ibv_cq_ex* send_cq_ex_;
  struct ibv_cq_ex* recv_cq_ex_;

  // Shared SRQ for all data path QPs.
  struct ibv_srq* srq_;

  // Buffer pool for retransmission chunks.
  std::optional<RetrChunkBuffPool> retr_chunk_pool_;
  // Buffer pool for retransmission headers.
  std::optional<RetrHdrBuffPool> retr_hdr_pool_;
  // Buffer pool for CQE descriptors.
  std::optional<CQEDescPool> cq_desc_pool_;
  // Buffer pool for control chunks.
  std::optional<CtrlChunkBuffPool> ctrl_chunk_pool_;

  // Pre-allocated WQEs for consuming immediate data.
  struct RecvWRs dp_recv_wrs_;

  // Pre-allocated WQEs/SGEs for receiving ACKs.
  struct RecvWRs ctrl_recv_wrs_;

  // WQE for sending ACKs.
  struct ibv_send_wr tx_ack_wr_[kMaxAckWRs];
  struct ibv_sge tx_ack_sge_[kMaxAckWRs];
  uint32_t nr_tx_ack_wr_ = 0;
  uint32_t inflight_ctrl_wrs_ = 0;

  // Memory region for retransmission.
  struct ibv_mr* retr_mr_;
  struct ibv_mr* retr_hdr_mr_;
  struct ibv_mr* cq_desc_mr_;

  std::unordered_map<int, RDMAContext*> qpn_to_rdma_ctx_map_;

  std::unordered_map<std::pair<uint32_t, uint32_t>, RDMAContext*, pair_hash>
      fid_to_rdma_ctx_map_;

  friend class UcclRDMAEngine;
  friend class RDMAContext;
};

}  // namespace uccl

#endif