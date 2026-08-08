#pragma once
#include "common.h"
#include "rdma_data_channel_impl.h"
#include "util/debug.h"

class IBDataChannelImpl : public RDMADataChannelImpl {
 public:
  IBDataChannelImpl() = default;
  ~IBDataChannelImpl() override { delete[] pre_alloc_recv_wrs_; }

  void init_qp(std::shared_ptr<RdmaContext> ctx, struct ibv_cq_ex** cq_ex,
               struct ibv_qp** qp, ChannelMetaData* local_meta) override;

  void connect_qp(struct ibv_qp* qp, std::shared_ptr<RdmaContext> ctx,
                  ChannelMetaData const& remote_meta) override;

  bool poll_once(struct ibv_cq_ex* cq_ex, std::vector<CQMeta>& cq_datas,
                 uint32_t channel_id, uint32_t& nb_post_recv) override;

  void lazy_post_recv_wrs_n(struct ibv_qp* qp, uint32_t n, bool force) override;

  void set_dst_address(struct ibv_qp_ex* qpx, struct ibv_ah* ah,
                       uint32_t remote_qpn) override;

  int post_write(struct ibv_qp* qp, struct ibv_ah* ah, uint32_t remote_qpn,
                 uint64_t wr_id, struct ibv_sge* sge, uint64_t remote_addr,
                 uint32_t remote_rkey, bool signaled) override;

  void init_pre_alloc_resources() override;

 private:
  void ibrc_qp_rtr_rts(struct ibv_qp* qp, std::shared_ptr<RdmaContext> ctx,
                       ChannelMetaData const& remote_meta);
};
