#pragma once
#include "rdma/rdma_channel_impl.h"
#include "rdma/define.h"
#include <glog/logging.h>

#define GID_INDEX 0
#define MAX_INLINE_DATA 0

static constexpr int kEfaQpLowLatencyServiceLevel = 8;
static constexpr uint32_t kQKey = 0x15695;
static constexpr uint8_t kEfaRdmDefaultRnrRetry = 3;

class EFAChannelImpl : public RDMAChannelImpl {
 public:
  EFAChannelImpl() = default;
  ~EFAChannelImpl() override = default;

  void initQP(std::shared_ptr<RdmaContext> ctx,
              struct ibv_cq_ex** cq_ex,
              struct ibv_qp** qp,
              ChannelMetaData* local_meta) override;

  void connectQP(struct ibv_qp* qp,
                std::shared_ptr<RdmaContext> ctx,
                ChannelMetaData const& remote_meta,
                struct ibv_recv_wr* pre_alloc_recv_wrs = nullptr,
                uint32_t kMaxRecvWr = 0,
                uint32_t* pending_post_recv = nullptr) override;

  bool poll_once(struct ibv_cq_ex* cq_ex,
                 std::vector<CQMeta>& cq_datas,
                 uint32_t channel_id) override;

  void lazy_post_recv_wr(struct ibv_qp* qp,
                         uint32_t threshold,
                         uint32_t& pending_post_recv,
                         struct ibv_recv_wr* pre_alloc_recv_wrs,
                         uint32_t kMaxRecvWr) override;

  void setDstAddress(struct ibv_qp_ex* qpx,
                     struct ibv_ah* ah,
                     uint32_t remote_qpn) override;

  int getGidIndex() const override { return GID_INDEX; }
  uint32_t getMaxInlineData() const override { return MAX_INLINE_DATA; }

  void initPreAllocResources(struct ibv_recv_wr* pre_alloc_recv_wrs,
                             uint32_t kMaxRecvWr) override;
};

// Implementation (inline to avoid separate .cc file)
#include "rdma_channel_impl_efa.cc"

