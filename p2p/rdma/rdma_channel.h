#pragma once
#include "define.h"
#include "rdma_channel_impl.h"
#include "rdma_context.h"
#include "seq_num.h"
#include "util/util.h"
#include <glog/logging.h>

#ifdef UCCL_P2P_USE_EFA
#include "providers/efa/rdma_channel_impl_efa.h"
#else
#include "providers/ib/rdma_channel_impl_ib.h"
#endif

// Factory function implementation (inline, defined after including impl
// headers)
inline std::unique_ptr<RDMAChannelImpl> createRDMAChannelImpl() {
#ifdef UCCL_P2P_USE_EFA
  return std::make_unique<EFAChannelImpl>();
#else
  return std::make_unique<IBChannelImpl>();
#endif
}

class RDMAChannel {
 public:
  explicit RDMAChannel(std::shared_ptr<RdmaContext> ctx,
                       uint32_t channel_id = 0)
      : ctx_(ctx),
        qp_(nullptr),
        cq_ex_(nullptr),
        ah_(nullptr),
        channel_id_(channel_id),
        local_meta_(std::make_shared<ChannelMetaData>()),
        remote_meta_(std::make_shared<ChannelMetaData>()),
        impl_(createRDMAChannelImpl()) {
    initQP();
  }

  explicit RDMAChannel(std::shared_ptr<RdmaContext> ctx,
                       ChannelMetaData const& remote_meta,
                       uint32_t channel_id = 0)
      : ctx_(ctx),
        qp_(nullptr),
        cq_ex_(nullptr),
        ah_(nullptr),
        channel_id_(channel_id),
        local_meta_(std::make_shared<ChannelMetaData>()),
        remote_meta_(std::make_shared<ChannelMetaData>(remote_meta)),
        impl_(createRDMAChannelImpl()) {
    initQP();
    establishChannel(remote_meta);
  }

  RDMAChannel(RDMAChannel const&) = delete;
  RDMAChannel& operator=(RDMAChannel const&) = delete;

  ~RDMAChannel() {
    if (qp_) ibv_destroy_qp(qp_);
    if (cq_ex_) ibv_destroy_cq(ibv_cq_ex_to_cq(cq_ex_));
  }

  void establishChannel(ChannelMetaData const& remote_meta) {
    remote_meta_ = std::make_shared<ChannelMetaData>(remote_meta);
#ifdef UCCL_P2P_USE_EFA
    ah_ = ctx_->createAH(remote_meta_->gid);
#endif
    impl_->connectQP(qp_, ctx_, *remote_meta_);
    UCCL_LOG_EP << "RDMAChannel connected to remote qpn=" << remote_meta.qpn;
  }

  int64_t submitRequest(std::shared_ptr<RDMASendRequest> req) {
    return postRequest(req);
  }

  int64_t read(std::shared_ptr<RDMASendRequest> req) {
    int ret = postRequest(req);
    if (ret != 0) {
      LOG(ERROR) << "Failed to post read request, wr_id=" << req->wr_id;
      return -1;
    }
    return req->wr_id;
  }

  int64_t send(std::shared_ptr<RDMASendRequest> req) {
    int ret = postRequest(req);
    if (ret != 0) {
      LOG(ERROR) << "Failed to post send request, wr_id=" << req->wr_id;
      return -1;
    }
    return req->wr_id;
  }

  int64_t recv(std::shared_ptr<RDMARecvRequest> req) {
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

  bool poll_once(std::vector<CQMeta>& cq_datas) {
    uint32_t nb_post_recv = 0;
    bool result = impl_->poll_once(cq_ex_, cq_datas, channel_id_, nb_post_recv);
    impl_->lazy_post_recv_wrs_n(qp_, nb_post_recv, false);
    return result;
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
  std::unique_ptr<RDMAChannelImpl> impl_;

  struct ibv_cq_ex* getCQ() const {
    return cq_ex_;
  }

  struct ibv_qp* getQP() const {
    return qp_;
  }

  // Post send request based on send_type
  // Returns 0 on success, error code on failure
  inline int __postRequest_ex(std::shared_ptr<RDMASendRequest> req) {
    auto* qpx = ibv_qp_to_qp_ex(qp_);
    ibv_wr_start(qpx);
    // LOG(INFO) << *req;
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
      LOG(ERROR) << "Unknown SendType in RDMAChannel::postRequest";
      return -1;
    }

    struct ibv_sge sge[1];
    int num_sge = prepareSGEList(sge, req);
    uint32_t max_inline = impl_->getMaxInlineData();
    if (req->getLocalLen() <= max_inline) {
      qpx->wr_flags |= IBV_SEND_INLINE;
      ibv_wr_set_inline_data(qpx, (void*)req->getLocalAddress(),
                             req->getLocalLen());
    } else {
      ibv_wr_set_sge_list(qpx, num_sge, sge);
    }

    impl_->setDstAddress(qpx, ah_, remote_meta_->qpn);

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

  inline int __postRequest(std::shared_ptr<RDMASendRequest> req) {
    struct ibv_send_wr wr;
    struct ibv_send_wr* bad_wr = nullptr;
    struct ibv_sge sge[1];

    memset(&wr, 0, sizeof(wr));
    // LOG(INFO) << *req;

    wr.wr_id = req->wr_id;
    wr.send_flags = IBV_SEND_SIGNALED;

    int num_sge = prepareSGEList(sge, req);
    wr.sg_list = sge;
    wr.num_sge = num_sge;

    if (req->send_type == SendType::Send) {
      wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
      wr.wr.rdma.remote_addr = req->getRemoteAddress();
      wr.wr.rdma.rkey = req->getRemoteKey();
      wr.imm_data = req->imm_data;
    } else if (req->send_type == SendType::Write) {
      wr.opcode = IBV_WR_RDMA_WRITE;
      wr.wr.rdma.remote_addr = req->getRemoteAddress();
      wr.wr.rdma.rkey = req->getRemoteKey();
    } else if (req->send_type == SendType::Read) {
      wr.opcode = IBV_WR_RDMA_READ;
      wr.wr.rdma.remote_addr = req->getRemoteAddress();
      wr.wr.rdma.rkey = req->getRemoteKey();
    } else {
      LOG(ERROR) << "Unknown SendType in RDMAChannel::postRequest";
      return -1;
    }

    uint32_t max_inline = impl_->getMaxInlineData();
    if (req->getLocalLen() <= max_inline) {
      wr.send_flags |= IBV_SEND_INLINE;
    }

    int ret = ibv_post_send(qp_, &wr, &bad_wr);

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

      LOG(ERROR) << "ibv_post_send failed: " << ret << " " << strerror(ret)
                 << ", ah_=" << (void*)ah_
                 << ", remote_qpn=" << remote_meta_->qpn
                 << ", local_qpn=" << qp_->qp_num << ", wr_id=" << req->wr_id
                 << ", remote_key=" << req->getRemoteKey() << ", remote_addr=0x"
                 << std::hex << req->getRemoteAddress()
                 << ", num_sge=" << num_sge << ", sge_list=" << sge_info.str();
    }
    return ret;
  }

  inline int postRequest(std::shared_ptr<RDMASendRequest> req) {
    if (ctx_->getVendorID() == 0x1dd8) {
      // Devices that don't support ibv_wr_xxx API
      return __postRequest(req);
    }

    return __postRequest_ex(req);
  }

  void initQP() {
    impl_->initQP(ctx_, &cq_ex_, &qp_, local_meta_.get());
    impl_->initPreAllocResources();
  }

  // Prepare SGE list for send request
  // Returns the number of SGE entries filled
  inline int prepareSGEList(struct ibv_sge* sge,
                            std::shared_ptr<RDMASendRequest> req) {
    uint32_t total_len = req->getLocalLen();
    uint64_t local_addr = req->getLocalAddress();
    uint32_t local_key = req->getLocalKey();
    sge[0].addr = local_addr;
    sge[0].length = total_len;
    sge[0].lkey = local_key;
    return 1;
  }
};
