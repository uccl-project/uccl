#ifndef RDMA_CHANNEL_IMPL_IB_CC_INCLUDED
#define RDMA_CHANNEL_IMPL_IB_CC_INCLUDED

#include "rdma_channel_impl_ib.h"
#include <glog/logging.h>
#include <cstring>

inline void IBChannelImpl::initQP(std::shared_ptr<RdmaContext> ctx,
                                  struct ibv_cq_ex** cq_ex, struct ibv_qp** qp,
                                  ChannelMetaData* local_meta) {
  *cq_ex = (struct ibv_cq_ex*)ibv_create_cq(ctx->getCtx(), 1024, nullptr,
                                            nullptr, 0);
  assert(*cq_ex);

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
  qp_attr.cap.max_inline_data = getMaxInlineData();

  qp_attr.send_cq = ibv_cq_ex_to_cq(*cq_ex);
  qp_attr.recv_cq = ibv_cq_ex_to_cq(*cq_ex);

  qp_attr.pd = ctx->getPD();
  qp_attr.qp_context = ctx->getCtx();
  qp_attr.sq_sig_all = 0;

  qp_attr.qp_type = IBV_QPT_RC;
  *qp = ibv_create_qp_ex(ctx->getCtx(), &qp_attr);
  assert(*qp);

  struct ibv_qp_attr attr = {};
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_INIT;
  attr.port_num = kPortNum;
  attr.pkey_index = 0;
  attr.qp_access_flags =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
  assert(ibv_modify_qp(*qp, &attr,
                       IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                           IBV_QP_ACCESS_FLAGS) == 0);

  local_meta->gid = ctx->queryGid(getGidIndex());
  local_meta->qpn = (*qp)->qp_num;
}

inline void IBChannelImpl::connectQP(struct ibv_qp* qp,
                                     std::shared_ptr<RdmaContext> ctx,
                                     ChannelMetaData const& remote_meta,
                                     struct ibv_recv_wr* pre_alloc_recv_wrs,
                                     uint32_t kMaxRecvWr,
                                     uint32_t* pending_post_recv) {
  if (pre_alloc_recv_wrs && pending_post_recv) {
    ibrcQP_rtr_rts(qp, ctx, remote_meta, pre_alloc_recv_wrs, kMaxRecvWr,
                   *pending_post_recv);
  }
}

inline void IBChannelImpl::ibrcQP_rtr_rts(
    struct ibv_qp* qp, std::shared_ptr<RdmaContext> ctx,
    ChannelMetaData const& remote_meta, struct ibv_recv_wr* pre_alloc_recv_wrs,
    uint32_t kMaxRecvWr, uint32_t& pending_post_recv) {
  int flags = 0;
  struct ibv_qp_attr attr = {};
  struct ibv_port_attr port_attr;
  assert(ibv_query_port(ctx->getCtx(), kPortNum, &port_attr) == 0);

  // RTR
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = port_attr.active_mtu;
  attr.dest_qp_num = remote_meta.qpn;
  attr.rq_psn = 0;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;
  // RoCE
  // TODO: Infiniband
  attr.ah_attr.is_global = 1;
  attr.ah_attr.port_num = 1;
  attr.ah_attr.sl = 135;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.grh.traffic_class = 3;
  attr.ah_attr.grh.hop_limit = 64;
  memcpy(&attr.ah_attr.grh.dgid, remote_meta.gid.raw, 16);
  attr.ah_attr.grh.sgid_index = getGidIndex();
  flags = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
          IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER | IBV_QP_AV;
  assert(ibv_modify_qp(qp, &attr, flags) == 0);

  for (int i = 0; i < kMaxRecvWr; i++) {
    this->lazy_post_recv_wr(qp, kMaxRecvWr, pending_post_recv,
                            pre_alloc_recv_wrs, kMaxRecvWr);
  }

  // RTS
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.sq_psn = 0;
  attr.max_rd_atomic = 1;
  flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
          IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
  assert(ibv_modify_qp(qp, &attr, flags) == 0);
}

inline bool IBChannelImpl::poll_once(struct ibv_cq_ex* cq_ex,
                                     std::vector<CQMeta>& cq_datas,
                                     uint32_t channel_id) {
  if (!cq_ex) {
    LOG(INFO) << "poll_once - channel_id: " << channel_id << ", cq_ex_ is null";
    return false;
  }

  struct ibv_wc pre_alloc_wcs[kBatchPollCqe];
  auto cq = ibv_cq_ex_to_cq(cq_ex);
  int ret = ibv_poll_cq(cq, kBatchPollCqe, pre_alloc_wcs);

  if (ret <= 0) {
    return false;
  }

  for (int i = 0; i < ret; i++) {
    auto wc = &pre_alloc_wcs[i];
    uint64_t wr_id = wc->wr_id;
    auto status = wc->status;
    if (unlikely(status != IBV_WC_SUCCESS)) {
      LOG(WARNING) << "poll_once - channel_id: " << channel_id
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

inline void IBChannelImpl::lazy_post_recv_wr(
    struct ibv_qp* qp, uint32_t threshold, uint32_t& pending_post_recv,
    struct ibv_recv_wr* pre_alloc_recv_wrs, uint32_t kMaxRecvWr) {
  threshold = std::min(threshold, kMaxRecvWr);
  pending_post_recv++;
  while (pending_post_recv >= threshold) {
    struct ibv_recv_wr* bad_wr = nullptr;
    pre_alloc_recv_wrs[threshold - 1].next = nullptr;
    assert(ibv_post_recv(qp, pre_alloc_recv_wrs, &bad_wr) == 0);
    pre_alloc_recv_wrs[threshold - 1].next =
        (threshold == kMaxRecvWr) ? nullptr : &pre_alloc_recv_wrs[threshold];
    pending_post_recv -= threshold;
  }
}

inline void IBChannelImpl::setDstAddress(struct ibv_qp_ex* qpx,
                                         struct ibv_ah* ah,
                                         uint32_t remote_qpn) {
  // IB RC doesn't need UD address setup
  (void)qpx;
  (void)ah;
  (void)remote_qpn;
}

inline void IBChannelImpl::initPreAllocResources(
    struct ibv_recv_wr* pre_alloc_recv_wrs, uint32_t kMaxRecvWr) {
  for (int i = 0; i < kMaxRecvWr; i++) {
    pre_alloc_recv_wrs[i] = {};
    pre_alloc_recv_wrs[i].next =
        (i == kMaxRecvWr - 1) ? nullptr : &pre_alloc_recv_wrs[i + 1];
  }
}

#endif  // RDMA_CHANNEL_IMPL_IB_CC_INCLUDED
