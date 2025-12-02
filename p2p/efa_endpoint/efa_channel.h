#pragma once
#include "define.h"
#include "rdma_context.h"
#include "seq_num.h"
#include "util/util.h"
#include <glog/logging.h>

class EFAChannel {
 public:
  explicit EFAChannel(std::shared_ptr<RdmaContext> ctx, uint32_t channel_id = 0)
      : ctx_(ctx),
        qp_(nullptr),
        cq_ex_(nullptr),
        ah_(nullptr),
        channel_id_(channel_id),
        local_meta_(std::make_shared<ChannelMetaData>()),
        remote_meta_(std::make_shared<ChannelMetaData>()) {
    // tracker_ = std::make_shared<AtomicBitmapPacketTracker>();
    initQP();
  }

  explicit EFAChannel(std::shared_ptr<RdmaContext> ctx,
                      ChannelMetaData const& remote_meta,
                      uint32_t channel_id = 0)
      : ctx_(ctx),
        qp_(nullptr),
        cq_ex_(nullptr),
        ah_(nullptr),
        channel_id_(channel_id),
        local_meta_(std::make_shared<ChannelMetaData>()),
        remote_meta_(std::make_shared<ChannelMetaData>(remote_meta)) {
    // tracker_ = std::make_shared<AtomicBitmapPacketTracker>();
    initQP();
    ah_ = ctx_->createAH(remote_meta_->gid);
    UCCL_LOG_EP << "EFAChannel connected to remote qpn=" << remote_meta.qpn;
  }

  // Delete copy constructor and copy assignment operator
  EFAChannel(const EFAChannel&) = delete;
  EFAChannel& operator=(const EFAChannel&) = delete;

  void connect(ChannelMetaData const& remote_meta) {
    remote_meta_ = std::make_shared<ChannelMetaData>(remote_meta);
    ah_ = ctx_->createAH(remote_meta_->gid);
    UCCL_LOG_EP << "EFAChannel connected to remote qpn=" << remote_meta.qpn;
  }

  // int64_t write_async(std::shared_ptr<EFASendRequest> req) {
  //   struct ibv_qp_ex* qpx = ibv_qp_to_qp_ex(qp_);
  //   ibv_wr_start(qpx);
  //   int32_t wr_id = tracker_->sendPacket();
  //   qpx->wr_id = wr_id;
  //   qpx->comp_mask = 0;
  //   qpx->wr_flags = IBV_SEND_SIGNALED;
  //   ibv_wr_rdma_write(qpx, req->getRemoteKey(), req->getRemoteAddress());

  //   struct ibv_sge sge {
  //     reinterpret_cast<uint64_t>(req->getLocalAddress()),
  //         reinterpret_cast<uint32_t>(req->getLocalLen()), req->getLocalKey()
  //   };
  //   ibv_wr_set_sge_list(qpx, 1, &sge);
  //   ibv_wr_set_ud_addr(qpx, ah_, remote_meta_->qpn, QKEY);

  //   if (ibv_wr_complete(qpx)) {
  //     LOG(ERROR) << "ibv_wr_complete failed in write_async";
  //   }

  //   return wr_id;
  // }

  // int read_async(std::shared_ptr<EFASendRequest> req) {
  //   struct ibv_qp_ex* qpx = ibv_qp_to_qp_ex(qp_);
  //   ibv_wr_start(qpx);
  //   int64_t wr_id = tracker_->sendPacket();
  //   qpx->wr_id = wr_id;
  //   qpx->comp_mask = 0;
  //   qpx->wr_flags = IBV_SEND_SIGNALED;
  //   ibv_wr_rdma_read(qpx, req->getRemoteKey(), req->getRemoteAddress());

  //   struct ibv_sge sge {
  //     reinterpret_cast<uint64_t>(req->getLocalAddress()), req->getLocalLen(),
  //         req->getLocalKey()
  //   };
  //   ibv_wr_set_sge_list(qpx, 1, &sge);
  //   ibv_wr_set_ud_addr(qpx, ah_, remote_meta_->qpn, QKEY);

  //   if (ibv_wr_complete(qpx)) {
  //     LOG(ERROR) << "ibv_wr_complete failed in read_async";
  //   }

  //   return wr_id;
  // }

  // int64_t send(std::shared_ptr<EFASendRequest> req){
  //   int64_t wr_id = tracker_->sendPacket();
  //   req->wr_id = wr_id;
  //   return send_with_wr_id(req);
  // }

  // int64_t get_wr_id() const {
  //   return tracker_->sendPacket();
  // }

  // Prepare SGE list for send request
  // Returns the number of SGE entries filled
  int prepareSGEList(struct ibv_sge* sge, std::shared_ptr<EFASendRequest> req) {
    uint32_t total_len = req->getLocalLen();
    uint64_t local_addr = req->getLocalAddress();
    uint32_t local_key = req->getLocalKey();
    sge[0].addr = local_addr;
    sge[0].length = total_len;
    sge[0].lkey = local_key;
    return 1;
  }

  int64_t send(std::shared_ptr<EFASendRequest> req) {
    auto* qpx = ibv_qp_to_qp_ex(qp_);
    ibv_wr_start(qpx);

    // int64_t wr_id = wr_id_counter_.fetch_add(1, std::memory_order_relaxed) +
    // 1;
    LOG(INFO) << *req;
    qpx->wr_id = req->wr_id;
    qpx->comp_mask = 0;
    // qpx->wr_flags = req->need_signaled ? IBV_SEND_SIGNALED : IBV_SEND_FENCE;
    qpx->wr_flags = IBV_SEND_SIGNALED;
    ibv_wr_rdma_write_imm(qpx, req->getRemoteKey(), req->getRemoteAddress(),
                          req->imm_data);

    struct ibv_sge sge[1];
    int num_sge = prepareSGEList(sge, req);
    ibv_wr_set_sge_list(qpx, num_sge, sge);
    ibv_wr_set_ud_addr(qpx, ah_, remote_meta_->qpn, kQKey);

    int ret = ibv_wr_complete(qpx);
    if (ret) {
      std::ostringstream sge_info;
      sge_info << "[";
      for (int i = 0; i < num_sge; ++i) {
        if (i > 0) sge_info << ", ";
        sge_info << "{addr:0x" << std::hex << sge[i].addr
                 << ", len:" << std::dec << sge[i].length << ", lkey:0x"
                 << std::hex << sge[i].lkey << std::dec << "}";
      }
      sge_info << "]";

      LOG(ERROR) << "ibv_wr_complete failed in send: " << ret << " "
                 << strerror(ret) << ", ah_=" << (void*)ah_
                 << ", remote_qpn=" << remote_meta_->qpn
                 << ", local_qpn=" << qp_->qp_num << ", wr_id=" << req->wr_id
                 << ", remote_key=" << req->getRemoteKey() << ", remote_addr=0x"
                 << std::hex << req->getRemoteAddress()
                 << ", local_key=" << req->getLocalKey()
                 << ", num_sge=" << num_sge << ", sge_list=" << sge_info.str()
                 << std::dec;
    }
    return req->wr_id;
  }

  int64_t recv(std::shared_ptr<EFARecvRequest> req) {
    struct ibv_sge sge = {
        .addr = (uintptr_t)req->getLocalAddress(),
        .length = (uint32_t)req->getLocalLen(),
        .lkey = req->getLocalKey(),
    };
    struct ibv_recv_wr wr = {0}, *bad_wr = nullptr;
    // int64_t wr_id = wr_id_counter_.fetch_add(1, std::memory_order_relaxed) +
    // 1;
    // int64_t wr_id = tracker_->sendPacket();
    int64_t wr_id = req->wr_id;
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    if (ibv_post_recv(qp_, &wr, &bad_wr)) {
      LOG(ERROR) << "ibv_post_recv failed: " << strerror(errno);
    }
    return wr_id;
  }

  bool poll_once(CQMeta& cq_data) {
    if (!cq_ex_) {
      LOG(INFO) << "poll_once - channel_id: " << channel_id_
                << ", cq_ex_ is null";
      return false;
    }
    struct ibv_poll_cq_attr poll_cq_attr = {.comp_mask = 0};
    ssize_t err = ibv_start_poll(cq_ex_, &poll_cq_attr);

    if (err) {
      // No completion available
      // LOG(INFO) << "poll_once - channel_id: " << channel_id_ << ", No
      // completion available, err: " << err;
      return false;
    }

    // Successfully polled one completion
    LOG(INFO) << "poll_once - channel_id: " << channel_id_
              << ", Successfully polled completion, wr_id: " << cq_ex_->wr_id
              << ", status: " << cq_ex_->status;

    if (cq_ex_->status != IBV_WC_SUCCESS) {
      LOG(WARNING) << "poll_once - channel_id: " << channel_id_
                   << ", CQ status error: "
                   << ibv_wc_status_str(cq_ex_->status);
      ibv_end_poll(cq_ex_);
      return false;
    }

    cq_data.wr_id = cq_ex_->wr_id;
    cq_data.op_code = ibv_wc_read_opcode(cq_ex_);
    cq_data.len = ibv_wc_read_byte_len(cq_ex_);
    if (cq_data.op_code == IBV_WC_RECV_RDMA_WITH_IMM) {
      cq_data.imm = ibv_wc_read_imm_data(cq_ex_);
    } else {
      cq_data.imm = 0;
    }

    LOG(INFO) << "poll_once - channel_id: " << channel_id_
              << ", Completion data: " << cq_data << "wr_id: " << cq_data.wr_id;

    // tracker_->acknowledge(cq_data.wr_id);
    ibv_end_poll(cq_ex_);

    return true;
  }

  // bool isAcknowledged(int32_t expected_wr) const {
  //   return tracker_->isAcknowledged(expected_wr);
  // }

  // void poll_cq(int32_t expected_wr) {
  //   if (tracker_->isAcknowledged(expected_wr)) {
  //     UCCL_LOG_EP << "tracker_->isAcknowledged(expected_wr): " <<
  //     expected_wr; return;
  //   }
  //   int poll_count = 0;

  //   struct ibv_poll_cq_attr poll_cq_attr = {.comp_mask = 0};
  //   ssize_t err = ibv_start_poll(cq_ex_, &poll_cq_attr);
  //   bool should_end_poll = !err;
  //   while (!err || !poll_count) {
  //     if (!err) {
  //       if (cq_ex_->status != IBV_WC_SUCCESS) {
  //         LOG(ERROR) << "poll_cq: " << ibv_wc_status_str(cq_ex_->status);
  //         return;
  //       }
  //       auto wr_id = cq_ex_->wr_id;
  //       int opcode = ibv_wc_read_opcode(cq_ex_);
  //       auto length = ibv_wc_read_byte_len(cq_ex_);
  //       tracker_->acknowledge(wr_id);
  //       if (wr_id == expected_wr || opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
  //         poll_count++;
  //         UCCL_LOG_EP << "wr_id == expected_wr:" << expected_wr;
  //         break;
  //       }
  //     }
  //     err = ibv_next_poll(cq_ex_);
  //   }

  //   if (should_end_poll) ibv_end_poll(cq_ex_);
  // }

  struct ibv_cq_ex* getCQ() const {
    return cq_ex_;
  }
  struct ibv_qp* getQP() const {
    return qp_;
  }

  // Get local metadata
  std::shared_ptr<ChannelMetaData> get_local_meta() const {
    return local_meta_;
  }

  // Get remote metadata
  std::shared_ptr<ChannelMetaData> get_remote_meta() const {
    return remote_meta_;
  }

  // Get RdmaContext
  inline const std::shared_ptr<RdmaContext> getContext() const { return ctx_; }

  inline const uint64_t getContextID() const { return ctx_->getContextID(); }

  inline uint32_t getChannelID() const { return channel_id_; }

  void printQP() {
    //     local_meta_->gid = ctx_->queryGid(kGidIndex);
    // local_meta_->qpn = qp_->qp_num;
    UCCL_LOG_EP << "local_meta_->qpn " << local_meta_->qpn;
    UCCL_LOG_EP << "remote_meta_->qpn " << remote_meta_->qpn;
  }

 private:
  std::shared_ptr<RdmaContext> ctx_;
  uint32_t channel_id_;

  struct ibv_cq_ex* cq_ex_;
  struct ibv_qp* qp_;
  struct ibv_ah* ah_;

  std::shared_ptr<ChannelMetaData> local_meta_;
  std::shared_ptr<ChannelMetaData> remote_meta_;

  std::shared_ptr<AtomicBitmapPacketTracker> tracker_;

  void initQP() {
    struct ibv_cq_init_attr_ex cq_attr = {0};
    cq_attr.cqe = 1024;
    cq_attr.wc_flags = IBV_WC_STANDARD_FLAGS;
    cq_attr.comp_mask = 0;

    cq_ex_ = ibv_create_cq_ex(ctx_->getCtx(), &cq_attr);
    assert(cq_ex_);

    struct ibv_qp_init_attr_ex qp_attr = {0};
    qp_attr.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
    qp_attr.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE |
                             IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM |
                             IBV_QP_EX_WITH_RDMA_READ;

    qp_attr.cap.max_send_wr = kMaxSendWr;
    qp_attr.cap.max_recv_wr = kMaxRecvWr;
    qp_attr.cap.max_send_sge = kMaxSendSeg;
    qp_attr.cap.max_recv_sge = kMaxRecvSeg;
    qp_attr.cap.max_inline_data = 0;

    qp_attr.send_cq = ibv_cq_ex_to_cq(cq_ex_);
    qp_attr.recv_cq = ibv_cq_ex_to_cq(cq_ex_);

    qp_attr.pd = ctx_->getPD();
    qp_attr.qp_context = ctx_->getCtx();
    qp_attr.sq_sig_all = 0;

    qp_attr.qp_type = IBV_QPT_DRIVER;

    struct efadv_qp_init_attr efa_attr = {};
    efa_attr.driver_qp_type = EFADV_QP_DRIVER_TYPE_SRD;
    efa_attr.sl = kEfaQpLowLatencyServiceLevel;
    efa_attr.flags = 0;
    // If set, Receive WRs will not be consumed for RDMA write with imm.
    efa_attr.flags |= EFADV_QP_FLAGS_UNSOLICITED_WRITE_RECV;

    qp_ = efadv_create_qp_ex(ctx_->getCtx(), &qp_attr, &efa_attr,
                             sizeof(efa_attr));
    assert(qp_);

    struct ibv_qp_attr attr = {};
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = kPortNum;
    attr.qkey = kQKey;
    attr.pkey_index = 0;
    assert(ibv_modify_qp(qp_, &attr,
                         IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                             IBV_QP_QKEY) == 0);

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    assert(ibv_modify_qp(qp_, &attr, IBV_QP_STATE) == 0);

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    // attr.rnr_retry = 10;
    // attr.min_rnr_timer = 10;
    attr.rnr_retry = kEfaRdmDefaultRnrRetry;
    assert(ibv_modify_qp(qp_, &attr,
                         IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_RNR_RETRY) == 0);

    local_meta_->gid = ctx_->queryGid(kGidIndex);
    local_meta_->qpn = qp_->qp_num;
  }
};
