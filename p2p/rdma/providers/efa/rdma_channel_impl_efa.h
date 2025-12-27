#pragma once
#include "rdma/define.h"
#include "rdma/rdma_channel_impl.h"
#include <glog/logging.h>

class EFAChannelImpl : public RDMAChannelImpl {
 public:
  EFAChannelImpl() = default;
  ~EFAChannelImpl() override = default;

  void initQP(std::shared_ptr<RdmaContext> ctx, struct ibv_cq_ex** cq_ex,
              struct ibv_qp** qp, ChannelMetaData* local_meta) override;

  void connectQP(struct ibv_qp* qp, std::shared_ptr<RdmaContext> ctx,
                 ChannelMetaData const& remote_meta) override;

  bool poll_once(struct ibv_cq_ex* cq_ex, std::vector<CQMeta>& cq_datas,
                 uint32_t channel_id, uint32_t& nb_post_recv) override;

  void setDstAddress(struct ibv_qp_ex* qpx, struct ibv_ah* ah,
                     uint32_t remote_qpn) override;

  uint32_t getMaxInlineData() const override;

  void initPreAllocResources() override;
};

// Implementation (inline to avoid separate .cc file)
#include "rdma_channel_impl_efa.cc"
