#pragma once
#include "common.h"
#include "rdma_data_channel_impl.h"
#include "util/debug.h"

class IBDataChannelImpl : public RDMADataChannelImpl {
 public:
  IBDataChannelImpl() = default;
  ~IBDataChannelImpl() override { delete[] pre_alloc_recv_wrs_; }

  void initQP(std::shared_ptr<RdmaContext> ctx, struct ibv_cq_ex** cq_ex,
              struct ibv_qp** qp, ChannelMetaData* local_meta) override;

  void connectQP(struct ibv_qp* qp, std::shared_ptr<RdmaContext> ctx,
                 ChannelMetaData const& remote_meta) override;

  bool pollOnce(struct ibv_cq_ex* cq_ex, std::vector<CQMeta>& cq_datas,
                uint32_t channel_id, uint32_t& nb_post_recv) override;

  void lazyPostRecvWrsN(struct ibv_qp* qp, uint32_t n, bool force) override;

  void setDstAddress(struct ibv_qp_ex* qpx, struct ibv_ah* ah,
                     uint32_t remote_qpn) override;

  int postWrite(struct ibv_qp* qp, struct ibv_ah* ah, uint32_t remote_qpn,
                uint64_t wr_id, struct ibv_sge* sge, uint64_t remote_addr,
                uint32_t remote_rkey, bool signaled) override;

  void initPreAllocResources() override;

 private:
  void ibrcQP_rtr_rts(struct ibv_qp* qp, std::shared_ptr<RdmaContext> ctx,
                      ChannelMetaData const& remote_meta);
};
