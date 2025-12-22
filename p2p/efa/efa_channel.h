#pragma once
#include "define.h"
#include "rdma_context.h"
#include "seq_num.h"
#include "util/util.h"
#include <glog/logging.h>

class EFAChannel {
 public:
  enum class QPXType {
    WriteImm,
    Write,
    READ,
  };

  explicit EFAChannel(std::shared_ptr<RdmaContext> ctx, uint32_t channel_id = 0)
      : ctx_(ctx),
        qp_(nullptr),
        cq_ex_(nullptr),
        ah_(nullptr),
        channel_id_(channel_id),
        local_meta_(std::make_shared<ChannelMetaData>()),
        remote_meta_(std::make_shared<ChannelMetaData>()) {
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
    initQP();
    UCCL_LOG_EP << "EFAChannel connected to remote qpn=" << remote_meta.qpn;
  }

  EFAChannel(EFAChannel const&) = delete;
  EFAChannel& operator=(EFAChannel const&) = delete;

  void connect(ChannelMetaData const& remote_meta) {
    remote_meta_ = std::make_shared<ChannelMetaData>(remote_meta);
    ah_ = ctx_->createAH(remote_meta_->gid);
#ifdef UCCL_ENABLE_IBRC
    ibrcQP_rtr_rts();
#endif
    UCCL_LOG_EP << "EFAChannel connected to remote qpn=" << remote_meta.qpn;
  }

  int64_t submitRequest(std::shared_ptr<EFASendRequest> req) {
    return postRequest(req);
  }

  int64_t read(std::shared_ptr<EFASendRequest> req) {
    int ret = postRequest(req);
    if (ret != 0) {
      LOG(ERROR) << "Failed to post read request, wr_id=" << req->wr_id;
      return -1;
    }
    return req->wr_id;
  }

  int64_t send(std::shared_ptr<EFASendRequest> req) {
    int ret = postRequest(req);
    if (ret != 0) {
      LOG(ERROR) << "Failed to post send request, wr_id=" << req->wr_id;
      return -1;
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
    int64_t wr_id = req->wr_id;
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    if (ibv_post_recv(qp_, &wr, &bad_wr)) {
      LOG(ERROR) << "ibv_post_recv failed: " << strerror(errno);
    }
    return wr_id;
  }

#ifdef UCCL_ENABLE_IBRC
  bool poll_once(std::vector<CQMeta>& cq_datas) {
    if (!cq_ex_) {
      LOG(INFO) << "poll_once - channel_id: " << channel_id_
                << ", cq_ex_ is null";
      return false;
    }

    struct ibv_wc wcs[32];
    auto cq = ibv_cq_ex_to_cq(cq_ex_);
    int ret = ibv_poll_cq(cq, 32, wcs);

    if (ret <= 0) {
      return false;
    }

    for (int i = 0; i < ret; i++) {
      auto wc = &wcs[i];
      uint64_t wr_id = wc->wr_id;
      auto status = wc->status;
      if (unlikely(status != IBV_WC_SUCCESS)) {
        LOG(WARNING) << "poll_once - channel_id: " << channel_id_
                     << ", CQE error, wr_id=" << wr_id << ", status=" << status
                     << " (" << ibv_wc_status_str(status) << ")";
      } else {
        CQMeta cq_data{};
        cq_data.wr_id = wr_id;
        cq_data.op_code = wc->opcode;
        cq_data.len = wc->byte_len;

        if (cq_data.op_code == IBV_WC_RECV_RDMA_WITH_IMM) {
          cq_data.imm = wc->imm_data;
        } else {
          cq_data.imm = 0;
        }

        cq_datas.emplace_back(cq_data);
      }
    }

    return !cq_datas.empty();
  }
#else
  bool poll_once(std::vector<CQMeta>& cq_datas) {
    if (!cq_ex_) {
      LOG(INFO) << "poll_once - channel_id: " << channel_id_
                << ", cq_ex_ is null";
      return false;
    }

    struct ibv_poll_cq_attr attr = {};
    int ret = ibv_start_poll(cq_ex_, &attr);

    if (ret == ENOENT) {
      return false;
    }
    if (ret) {
      LOG(ERROR) << "poll_once - channel_id: " << channel_id_
                 << ", ibv_start_poll error: " << ret << " (" << strerror(ret)
                 << ")";
      return false;
    }

    do {
      uint64_t wr_id = cq_ex_->wr_id;
      auto status = cq_ex_->status;
      if (unlikely(status != IBV_WC_SUCCESS)) {
        LOG(WARNING) << "poll_once - channel_id: " << channel_id_
                     << ", CQE error, wr_id=" << wr_id << ", status=" << status
                     << " (" << ibv_wc_status_str(status) << ")";
      } else {
        CQMeta cq_data{};
        cq_data.wr_id = wr_id;
        cq_data.op_code = ibv_wc_read_opcode(cq_ex_);
        cq_data.len = ibv_wc_read_byte_len(cq_ex_);

        if (cq_data.op_code == IBV_WC_RECV_RDMA_WITH_IMM) {
          cq_data.imm = ibv_wc_read_imm_data(cq_ex_);
        } else {
          cq_data.imm = 0;
        }

        cq_datas.emplace_back(cq_data);
      }

      ret = ibv_next_poll(cq_ex_);
    } while (ret == 0);

    ibv_end_poll(cq_ex_);

    if (ret != ENOENT) {
      LOG(ERROR) << "poll_once - channel_id: " << channel_id_
                 << ", ibv_next_poll error: " << ret << " (" << strerror(ret)
                 << ")";
    }

    return !cq_datas.empty();
  }
#endif

  // Get local metadata
  std::shared_ptr<ChannelMetaData> get_local_meta() const {
    return local_meta_;
  }

  // Get remote metadata
  std::shared_ptr<ChannelMetaData> get_remote_meta() const {
    return remote_meta_;
  }

  // Get RdmaContext
  inline std::shared_ptr<RdmaContext> const getContext() const { return ctx_; }

  inline uint64_t const getContextID() const { return ctx_->getContextID(); }

  inline uint32_t getChannelID() const { return channel_id_; }

 private:
  std::shared_ptr<RdmaContext> ctx_;
  uint32_t channel_id_;

  struct ibv_cq_ex* cq_ex_;
  struct ibv_qp* qp_;
  struct ibv_ah* ah_;

  std::shared_ptr<ChannelMetaData> local_meta_;
  std::shared_ptr<ChannelMetaData> remote_meta_;

  std::shared_ptr<AtomicBitmapPacketTracker> tracker_;

  struct ibv_cq_ex* getCQ() const {
    return cq_ex_;
  }

  struct ibv_qp* getQP() const {
    return qp_;
  }

  // Post send request based on send_type
  // Returns 0 on success, error code on failure
  inline int postRequest(std::shared_ptr<EFASendRequest> req) {
    auto* qpx = ibv_qp_to_qp_ex(qp_);
    ibv_wr_start(qpx);
    LOG(INFO) << *req;
    qpx->wr_id = req->wr_id;
    qpx->comp_mask = 0;
    qpx->wr_flags = IBV_SEND_SIGNALED;

    if (req->send_type == SendType::Send) {
      ibv_wr_rdma_write_imm(qpx, req->getRemoteKey(), req->getRemoteAddress(),
                            req->imm_data);
    } else if (req->send_type == SendType::Write) {
      ibv_wr_rdma_write(qpx, req->getRemoteKey(), req->getRemoteAddress());
    } else if (req->send_type == SendType::Read) {
      ibv_wr_rdma_read(qpx, req->getRemoteKey(), req->getRemoteAddress());
    } else {
      LOG(ERROR) << "Unknown SendType in EFAChannel::postRequest";
      return -1;
    }

    struct ibv_sge sge[1];
    int num_sge = prepareSGEList(sge, req);
    // ibv_wr_set_sge_list(qpx, num_sge, sge);
    ibv_wr_set_sge(qpx, req->getLocalKey(), req->getLocalAddress(),
                   req->getLocalLen());

#ifndef UCCL_ENABLE_IBRC
    ibv_wr_set_ud_addr(qpx, ah_, remote_meta_->qpn, kQKey);
#endif

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

      LOG(ERROR) << "ibv_wr_complete failed in postRequest: " << ret << " "
                 << strerror(ret) << ", ah_=" << (void*)ah_
                 << ", remote_qpn=" << remote_meta_->qpn
                 << ", local_qpn=" << qp_->qp_num << ", wr_id=" << req->wr_id
                 << ", remote_key=" << req->getRemoteKey() << ", remote_addr=0x"
                 << std::hex << req->getRemoteAddress()
                 << ", local_key=" << req->getLocalKey()
                 << ", num_sge=" << num_sge << ", sge_list=" << sge_info.str()
                 << std::dec;
    }
    return ret;
  }

  void ibrcQP_rtr_rts() {
    int flags = 0;
    struct ibv_qp_attr attr = {};
    struct ibv_port_attr port_attr;
    assert(ibv_query_port(ctx_->getCtx(), kPortNum, &port_attr) == 0);

    // RTR
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = port_attr.active_mtu;
    attr.dest_qp_num = remote_meta_->qpn;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    // RoCE
    attr.ah_attr.is_global = 1;
    attr.ah_attr.port_num = 1;
    attr.ah_attr.sl = 135;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.grh.traffic_class = 3;
    attr.ah_attr.grh.hop_limit = 64;
    memcpy(&attr.ah_attr.grh.dgid, remote_meta_->gid.raw, 16);
    attr.ah_attr.grh.sgid_index = kGidIndex;
    flags = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
            IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER | IBV_QP_AV;
    assert(ibv_modify_qp(qp_, &attr, flags) == 0);

    // RTS
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = 0;
    attr.max_rd_atomic = 1;
    flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
            IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
    assert(ibv_modify_qp(qp_, &attr, flags) == 0);
  }

#ifdef UCCL_ENABLE_IBRC
  void initQP() {
    cq_ex_ = (struct ibv_cq_ex*)ibv_create_cq(ctx_->getCtx(), 1024, nullptr,
                                              nullptr, 0);
    assert(cq_ex_);

    struct ibv_qp_init_attr_ex qp_attr = {};
    memset(&qp_attr, 0, sizeof(qp_attr));
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

    qp_attr.qp_type = IBV_QPT_RC;
    qp_ = ibv_create_qp_ex(ctx_->getCtx(), &qp_attr);
    assert(qp_);

    struct ibv_qp_attr attr = {};
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = kPortNum;
    attr.pkey_index = 0;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                           IBV_ACCESS_REMOTE_WRITE;
    assert(ibv_modify_qp(qp_, &attr,
                         IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                             IBV_QP_ACCESS_FLAGS) == 0);

    local_meta_->gid = ctx_->queryGid(kGidIndex);
    local_meta_->qpn = qp_->qp_num;
  }
#else
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
    attr.rnr_retry = kEfaRdmDefaultRnrRetry;
    assert(ibv_modify_qp(qp_, &attr,
                         IBV_QP_STATE | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN) == 0);

    local_meta_->gid = ctx_->queryGid(kGidIndex);
    local_meta_->qpn = qp_->qp_num;
  }
#endif

  // Prepare SGE list for send request
  // Returns the number of SGE entries filled
  inline int prepareSGEList(struct ibv_sge* sge,
                            std::shared_ptr<EFASendRequest> req) {
    uint32_t total_len = req->getLocalLen();
    uint64_t local_addr = req->getLocalAddress();
    uint32_t local_key = req->getLocalKey();
    sge[0].addr = local_addr;
    sge[0].length = total_len;
    sge[0].lkey = local_key;
    return 1;
  }
};
