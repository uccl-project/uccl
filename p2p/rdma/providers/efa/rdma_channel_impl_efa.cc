#ifndef RDMA_CHANNEL_IMPL_EFA_CC_INCLUDED
#define RDMA_CHANNEL_IMPL_EFA_CC_INCLUDED

#include "rdma_channel_impl_efa.h"
#include <glog/logging.h>
#include <cstring>
#include <errno.h>

#define GID_INDEX_EFA 0
#define LID_EFA 0
#define MAX_INLINE_DATA 0
#define SERVICE_LEVEL 8
#define QKEY 0x15695

#define RNR_RETRY 3

#define MAX_CQE 1024

inline void EFAChannelImpl::initQP(std::shared_ptr<RdmaContext> ctx,
                                   struct ibv_cq_ex** cq_ex, struct ibv_qp** qp,
                                   ChannelMetaData* local_meta) {
  struct ibv_cq_init_attr_ex cq_attr = {0};
  cq_attr.cqe = MAX_CQE;
  cq_attr.wc_flags = IBV_WC_STANDARD_FLAGS;
  cq_attr.comp_mask = 0;

  *cq_ex = ibv_create_cq_ex(ctx->getCtx(), &cq_attr);
  assert(*cq_ex);

  struct ibv_qp_init_attr_ex qp_attr = {0};
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

  qp_attr.qp_type = IBV_QPT_DRIVER;

  struct efadv_qp_init_attr efa_attr = {};
  efa_attr.driver_qp_type = EFADV_QP_DRIVER_TYPE_SRD;
  efa_attr.sl = get_sl_from_env(SERVICE_LEVEL);
  efa_attr.flags = 0;
  // If set, Receive WRs will not be consumed for RDMA write with imm.
  efa_attr.flags |= EFADV_QP_FLAGS_UNSOLICITED_WRITE_RECV;

  *qp =
      efadv_create_qp_ex(ctx->getCtx(), &qp_attr, &efa_attr, sizeof(efa_attr));

  assert(*qp);

  struct ibv_qp_attr attr = {};
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_INIT;
  attr.port_num = kPortNum;
  attr.qkey = QKEY;
  attr.pkey_index = 0;
  assert(ibv_modify_qp(*qp, &attr,
                       IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                           IBV_QP_QKEY) == 0);

  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  assert(ibv_modify_qp(*qp, &attr, IBV_QP_STATE) == 0);

  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.rnr_retry = RNR_RETRY;
  assert(ibv_modify_qp(*qp, &attr,
                       IBV_QP_STATE | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN) == 0);

  local_meta->gid = ctx->queryGid(GID_INDEX_EFA);
  local_meta->qpn = (*qp)->qp_num;
  local_meta->lid = LID_EFA;
}

inline void EFAChannelImpl::connectQP(struct ibv_qp* qp,
                                      std::shared_ptr<RdmaContext> ctx,
                                      ChannelMetaData const& remote_meta) {
  (void)qp;
  (void)ctx;
  (void)remote_meta;
}

inline bool EFAChannelImpl::poll_once(struct ibv_cq_ex* cq_ex,
                                      std::vector<CQMeta>& cq_datas,
                                      uint32_t channel_id,
                                      uint32_t& nb_post_recv) {
  nb_post_recv = 0;
  if (!cq_ex) {
    LOG(INFO) << "poll_once - channel_id: " << channel_id << ", cq_ex_ is null";
    return false;
  }

  struct ibv_poll_cq_attr attr = {};
  int ret = ibv_start_poll(cq_ex, &attr);

  if (ret == ENOENT) {
    return false;
  }
  if (ret) {
    LOG(ERROR) << "poll_once - channel_id: " << channel_id
               << ", ibv_start_poll error: " << ret << " (" << strerror(ret)
               << ")";
    return false;
  }

  do {
    uint64_t wr_id = cq_ex->wr_id;
    auto status = cq_ex->status;
    if (unlikely(status != IBV_WC_SUCCESS)) {
      LOG(WARNING) << "poll_once - channel_id: " << channel_id
                   << ", CQE error, wr_id=" << wr_id << ", status=" << status
                   << " (" << ibv_wc_status_str(status) << ")";
    } else {
      CQMeta cq_data{};
      cq_data.wr_id = wr_id;
      cq_data.op_code = ibv_wc_read_opcode(cq_ex);
      cq_data.len = ibv_wc_read_byte_len(cq_ex);

      if (cq_data.op_code == IBV_WC_RECV_RDMA_WITH_IMM) {
        cq_data.imm = ibv_wc_read_imm_data(cq_ex);
      } else {
        cq_data.imm = 0;
      }

      cq_datas.emplace_back(cq_data);
    }

    ret = ibv_next_poll(cq_ex);
  } while (ret == 0);

  ibv_end_poll(cq_ex);

  if (ret != ENOENT) {
    LOG(ERROR) << "poll_once - channel_id: " << channel_id
               << ", ibv_next_poll error: " << ret << " (" << strerror(ret)
               << ")";
  }

  return !cq_datas.empty();
}

inline void EFAChannelImpl::lazy_post_recv_wrs_n(struct ibv_qp* qp, uint32_t n,
                                                 bool force) {
  (void)qp;
  (void)n;
  (void)force;
}

inline void EFAChannelImpl::setDstAddress(struct ibv_qp_ex* qpx,
                                          struct ibv_ah* ah,
                                          uint32_t remote_qpn) {
  ibv_wr_set_ud_addr(qpx, ah, remote_qpn, QKEY);
}

inline void EFAChannelImpl::initPreAllocResources() {}

inline uint32_t EFAChannelImpl::getMaxInlineData() const {
  return MAX_INLINE_DATA;
}

#endif  // RDMA_CHANNEL_IMPL_EFA_CC_INCLUDED
