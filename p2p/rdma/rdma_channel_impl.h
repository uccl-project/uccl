#pragma once
#include "define.h"
#include "rdma_context.h"
#include <vector>

// Forward declarations
struct ibv_qp;
struct ibv_qp_ex;
struct ibv_cq_ex;
struct ibv_ah;
struct ibv_sge;
struct ibv_recv_wr;
struct ibv_wc;

// Base class for RDMA channel implementations
class RDMAChannelImpl {
 public:
  virtual ~RDMAChannelImpl() = default;

  // Initialize QP and CQ
  virtual void initQP(std::shared_ptr<RdmaContext> ctx,
                      struct ibv_cq_ex** cq_ex, struct ibv_qp** qp,
                      ChannelMetaData* local_meta) = 0;

  // Connect QP to remote
  virtual void connectQP(struct ibv_qp* qp, std::shared_ptr<RdmaContext> ctx,
                         ChannelMetaData const& remote_meta) = 0;

  // Poll completion queue
  virtual bool poll_once(struct ibv_cq_ex* cq_ex, std::vector<CQMeta>& cq_datas,
                         uint32_t channel_id, uint32_t& nb_post_recv) = 0;

  // Post receive work request
  virtual void lazy_post_recv_wrs_n(struct ibv_qp* qp, uint32_t n,
                                    bool force) = 0;

  // Setup Destination address
  virtual void setDstAddress(struct ibv_qp_ex* qpx, struct ibv_ah* ah,
                             uint32_t remote_qpn) = 0;

  // Get max inline data size
  virtual uint32_t getMaxInlineData() const = 0;

  // Initialize pre-allocated resources
  virtual void initPreAllocResources() = 0;

 protected:
  struct ibv_recv_wr* pre_alloc_recv_wrs_;
  uint32_t pending_post_recv_;
};

// Forward declarations for implementations
#ifdef UCCL_P2P_USE_EFA
class EFAChannelImpl;
#else
class IBChannelImpl;
#endif

// Factory function declaration (implementation in rdma_channel.h after
// including impl headers)
std::unique_ptr<RDMAChannelImpl> createRDMAChannelImpl();
