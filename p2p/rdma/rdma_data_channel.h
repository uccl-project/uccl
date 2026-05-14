#pragma once
#include "define.h"
#include "include/transport_type.h"
#include "providers/efa/rdma_data_channel_impl_efa.h"
#include "providers/ib/rdma_data_channel_impl_ib.h"
#include "rdma_context.h"
#include "rdma_data_channel_impl.h"
#include "seq_num.h"
#include "util/debug.h"
#include "util/util.h"
#include <deque>
#include <mutex>

// When set on a thread, RDMADataChannel::submitRequest() accumulates send
// requests in a per-channel batch instead of immediately posting them. The
// thread must subsequently call flushBatch() (typically via
// NICEndpoint::flushAllSends()) to flush all pending work requests in a
// single doorbell per channel.
inline thread_local bool g_uccl_batch_post = false;

// Factory: select IB or EFA provider at runtime.
inline std::unique_ptr<RDMADataChannelImpl> createRDMADataChannelImpl() {
  if (uccl::is_efa_transport())
    return std::make_unique<EFAChannelImpl>();
  else
    return std::make_unique<IBChannelImpl>();
}

class RDMADataChannel {
 public:
  explicit RDMADataChannel(std::shared_ptr<RdmaContext> ctx,
                           uint32_t channel_id = 0)
      : ctx_(ctx),
        qp_(nullptr),
        cq_ex_(nullptr),
        ah_(nullptr),
        channel_id_(channel_id),
        local_meta_(std::make_shared<ChannelMetaData>()),
        remote_meta_(std::make_shared<ChannelMetaData>()),
        impl_(createRDMADataChannelImpl()) {
    initQP();
  }

  explicit RDMADataChannel(std::shared_ptr<RdmaContext> ctx,
                           ChannelMetaData const& remote_meta,
                           uint32_t channel_id = 0)
      : ctx_(ctx),
        qp_(nullptr),
        cq_ex_(nullptr),
        ah_(nullptr),
        channel_id_(channel_id),
        local_meta_(std::make_shared<ChannelMetaData>()),
        remote_meta_(std::make_shared<ChannelMetaData>(remote_meta)),
        impl_(createRDMADataChannelImpl()) {
    initQP();
    establishChannel(remote_meta);
  }

  RDMADataChannel(RDMADataChannel const&) = delete;
  RDMADataChannel& operator=(RDMADataChannel const&) = delete;

  ~RDMADataChannel() {
    if (qp_) ibv_destroy_qp(qp_);
    if (cq_ex_) ibv_destroy_cq(ibv_cq_ex_to_cq(cq_ex_));
  }

  void establishChannel(ChannelMetaData const& remote_meta) {
    remote_meta_ = std::make_shared<ChannelMetaData>(remote_meta);
    if (uccl::is_efa_transport()) {
      ah_ = ctx_->createAH(remote_meta_->gid);
    }
    impl_->connectQP(qp_, ctx_, *remote_meta_);
    UCCL_LOG_EP << "RDMADataChannel connected to remote qpn="
                << remote_meta.qpn;
  }

  int64_t submitRequest(std::shared_ptr<RDMASendRequest> req) {
    if (g_uccl_batch_post) {
      // Cap how many bytes we accumulate before forcing a doorbell. This
      // preserves pipelining for large iovs (each WR posts ~immediately)
      // while still amortizing the doorbell cost across many small iovs.
      static constexpr size_t kBatchFlushBytes = 32 * 1024;
      size_t req_size = req->getLocalLen();
      bool should_flush;
      {
        std::lock_guard<std::mutex> lock(batch_mu_);
        batch_.push_back(std::move(req));
        batch_bytes_ += req_size;
        should_flush = (batch_bytes_ >= kBatchFlushBytes);
      }
      if (should_flush) flushBatch();
      return 0;  // success; actual post deferred until flushBatch().
    }
    return postRequest(req);
  }

  // Flush all accumulated batched send requests in one doorbell. Safe to call
  // when nothing is batched (returns 0).
  int flushBatch() {
    std::vector<std::shared_ptr<RDMASendRequest>> local;
    {
      std::lock_guard<std::mutex> lock(batch_mu_);
      if (batch_.empty()) return 0;
      local.swap(batch_);
      batch_bytes_ = 0;
    }
    bool use_legacy = (ctx_->getVendorID() == 0x1dd8 ||  // Broadcom
                       ctx_->getVendorID() == 0x8086);   // Intel irdma
    if (use_legacy) {
      return __flushBatch_legacy(local);
    }
    return __flushBatch_ex(local);
  }

  // Given a CQE wr_id, return all wr_ids that should be acknowledged in the
  // tracker. With selective signaling, a single signaled CQE represents
  // completion of the WR with that wr_id plus all earlier unsignaled WRs
  // posted as part of the same batched flush. For CQEs that do not match
  // any pending batch (i.e. immediate-path WRs that signaled themselves),
  // only the original wr_id is returned.
  void expandCompletion(uint64_t cqe_wr_id, std::vector<uint64_t>& acks) {
    std::lock_guard<std::mutex> lock(pending_mu_);
    if (!pending_groups_.empty() &&
        pending_groups_.front().signal_wr_id == cqe_wr_id) {
      auto& group = pending_groups_.front();
      acks.insert(acks.end(), group.unsignaled_wr_ids.begin(),
                  group.unsignaled_wr_ids.end());
      acks.push_back(cqe_wr_id);
      pending_groups_.pop_front();
      return;
    }
    acks.push_back(cqe_wr_id);
  }

  int64_t read(std::shared_ptr<RDMASendRequest> req) {
    int ret = postRequest(req);
    if (ret != 0) {
      UCCL_LOG(ERROR) << "Failed to post read request, wr_id=" << req->wr_id;
      return -1;
    }
    return req->wr_id;
  }

  int64_t send(std::shared_ptr<RDMASendRequest> req) {
    int ret = postRequest(req);
    if (ret != 0) {
      UCCL_LOG(ERROR) << "Failed to post send request, wr_id=" << req->wr_id;
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
      UCCL_LOG(ERROR) << "ibv_post_recv failed: " << strerror(errno);
      return -1;
    }
    return wr_id;
  }

  bool pollOnce(std::vector<CQMeta>& cq_datas) {
    uint32_t nb_post_recv = 0;
    bool result = impl_->pollOnce(cq_ex_, cq_datas, channel_id_, nb_post_recv);
    impl_->lazyPostRecvWrsN(qp_, nb_post_recv, false);
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
  std::unique_ptr<RDMADataChannelImpl> impl_;

  // Per-channel buffer for batched doorbell posting. Protected by batch_mu_.
  std::vector<std::shared_ptr<RDMASendRequest>> batch_;
  size_t batch_bytes_ = 0;
  std::mutex batch_mu_;

  // Selective-signaling state: each batched flush emits one signaled CQE.
  // When that CQE arrives, all preceding unsignaled WRs in the same batch
  // are guaranteed (by the RC ordering rules) to have completed and can be
  // acknowledged together. Each entry corresponds to one batched flush, in
  // post order on this channel's QP.
  struct PendingSignalGroup {
    uint64_t signal_wr_id;
    std::vector<uint64_t> unsignaled_wr_ids;
  };
  std::deque<PendingSignalGroup> pending_groups_;
  std::mutex pending_mu_;

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
    // UCCL_LOG(INFO, UCCL_RDMA) << *req;
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
      UCCL_LOG(ERROR) << "Unknown SendType in RDMADataChannel::postRequest";
      return -1;
    }

    struct ibv_sge sge[1];
    int num_sge = prepareSGEList(sge, req);

    ibv_wr_set_sge_list(qpx, num_sge, sge);

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

      UCCL_LOG(ERROR) << "ibv_wr_complete failed in postRequest: " << ret << " "
                      << strerror(ret) << ", ah_=" << (void*)ah_
                      << ", remote_qpn=" << remote_meta_->qpn
                      << ", local_qpn=" << qp_->qp_num
                      << ", wr_id=" << req->wr_id
                      << ", remote_key=" << req->getRemoteKey()
                      << ", remote_addr=0x" << std::hex
                      << req->getRemoteAddress()
                      << ", local_key=" << req->getLocalKey()
                      << ", num_sge=" << num_sge
                      << ", sge_list=" << sge_info.str() << std::dec;
    }
    return ret;
  }

  inline int __postRequest(std::shared_ptr<RDMASendRequest> req) {
    struct ibv_send_wr wr;
    struct ibv_send_wr* bad_wr = nullptr;
    struct ibv_sge sge[1];

    memset(&wr, 0, sizeof(wr));
    // UCCL_LOG(INFO, UCCL_RDMA) << *req;

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
      UCCL_LOG(ERROR) << "Unknown SendType in RDMADataChannel::postRequest";
      return -1;
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

      UCCL_LOG(ERROR) << "ibv_post_send failed: " << ret << " " << strerror(ret)
                      << ", ah_=" << (void*)ah_
                      << ", remote_qpn=" << remote_meta_->qpn
                      << ", local_qpn=" << qp_->qp_num
                      << ", wr_id=" << req->wr_id
                      << ", remote_key=" << req->getRemoteKey()
                      << ", remote_addr=0x" << std::hex
                      << req->getRemoteAddress() << ", num_sge=" << num_sge
                      << ", sge_list=" << sge_info.str();
    }
    return ret;
  }

  inline int postRequest(std::shared_ptr<RDMASendRequest> req) {
    if (ctx_->getVendorID() == 0x1dd8 ||  // Broadcom
        ctx_->getVendorID() == 0x8086) {  // Intel irdma
      // These NICs don't support ibv_wr_* extended posting API.
      return __postRequest(req);
    }

    return __postRequest_ex(req);
  }

  // Post all requests in `batch` using a single ibv_wr_start/ibv_wr_complete
  // pair (one doorbell). Only the LAST WR is IBV_SEND_SIGNALED; the
  // preceding WRs rely on RC ordering and are acknowledged together when
  // the signaled CQE arrives. Returns 0 on success.
  inline int __flushBatch_ex(
      std::vector<std::shared_ptr<RDMASendRequest>> const& batch) {
    if (batch.empty()) return 0;
    // SGE storage must remain valid until ibv_wr_complete.
    std::vector<struct ibv_sge> sges(batch.size());
    size_t const last = batch.size() - 1;
    std::vector<uint64_t> unsignaled;
    unsignaled.reserve(last);
    auto* qpx = ibv_qp_to_qp_ex(qp_);
    ibv_wr_start(qpx);
    for (size_t i = 0; i < batch.size(); ++i) {
      auto const& req = batch[i];
      qpx->wr_id = req->wr_id;
      qpx->comp_mask = 0;
      qpx->wr_flags = (i == last) ? IBV_SEND_SIGNALED : 0;
      if (req->send_type == SendType::Send) {
        ibv_wr_rdma_write_imm(qpx, req->getRemoteKey(),
                              req->getRemoteAddress(), req->imm_data);
      } else if (req->send_type == SendType::Write) {
        ibv_wr_rdma_write(qpx, req->getRemoteKey(), req->getRemoteAddress());
      } else if (req->send_type == SendType::Read) {
        ibv_wr_rdma_read(qpx, req->getRemoteKey(), req->getRemoteAddress());
      } else {
        UCCL_LOG(ERROR) << "Unknown SendType in __flushBatch_ex";
        ibv_wr_abort(qpx);
        return -1;
      }
      int num_sge = prepareSGEList(&sges[i], req);
      ibv_wr_set_sge_list(qpx, num_sge, &sges[i]);
      impl_->setDstAddress(qpx, ah_, remote_meta_->qpn);
      if (i != last) unsignaled.push_back(static_cast<uint64_t>(req->wr_id));
    }
    int ret = ibv_wr_complete(qpx);
    if (ret) {
      UCCL_LOG(ERROR) << "ibv_wr_complete failed in __flushBatch_ex: " << ret
                      << " " << strerror(ret) << ", batch_size=" << batch.size()
                      << ", local_qpn=" << qp_->qp_num;
      return ret;
    }
    if (!unsignaled.empty()) {
      std::lock_guard<std::mutex> lock(pending_mu_);
      pending_groups_.push_back(
          {static_cast<uint64_t>(batch[last]->wr_id), std::move(unsignaled)});
    }
    return 0;
  }

  // Legacy verbs path: build a linked ibv_send_wr list and post once. Only
  // the last WR is IBV_SEND_SIGNALED.
  inline int __flushBatch_legacy(
      std::vector<std::shared_ptr<RDMASendRequest>> const& batch) {
    if (batch.empty()) return 0;
    size_t const last = batch.size() - 1;
    std::vector<struct ibv_send_wr> wrs(batch.size());
    std::vector<struct ibv_sge> sges(batch.size());
    std::vector<uint64_t> unsignaled;
    unsignaled.reserve(last);
    for (size_t i = 0; i < batch.size(); ++i) {
      auto const& req = batch[i];
      auto& wr = wrs[i];
      std::memset(&wr, 0, sizeof(wr));
      wr.wr_id = req->wr_id;
      wr.send_flags = (i == last) ? IBV_SEND_SIGNALED : 0;
      int num_sge = prepareSGEList(&sges[i], req);
      wr.sg_list = &sges[i];
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
        UCCL_LOG(ERROR) << "Unknown SendType in __flushBatch_legacy";
        return -1;
      }
      wr.next = (i + 1 < batch.size()) ? &wrs[i + 1] : nullptr;
      if (i != last) unsignaled.push_back(static_cast<uint64_t>(req->wr_id));
    }
    struct ibv_send_wr* bad_wr = nullptr;
    int ret = ibv_post_send(qp_, wrs.data(), &bad_wr);
    if (ret) {
      UCCL_LOG(ERROR) << "ibv_post_send failed in __flushBatch_legacy: " << ret
                      << " " << strerror(ret) << ", batch_size=" << batch.size()
                      << ", local_qpn=" << qp_->qp_num;
      return ret;
    }
    if (!unsignaled.empty()) {
      std::lock_guard<std::mutex> lock(pending_mu_);
      pending_groups_.push_back(
          {static_cast<uint64_t>(batch[last]->wr_id), std::move(unsignaled)});
    }
    return 0;
  }

  void initQP() {
    impl_->initPreAllocResources();
    impl_->initQP(ctx_, &cq_ex_, &qp_, local_meta_.get());
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
