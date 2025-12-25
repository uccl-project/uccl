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

// Base class for RDMA channel implementations (IB/EFA)
class RDMAChannelImpl {
 public:
  virtual ~RDMAChannelImpl() = default;

  // Initialize QP and CQ
  virtual void initQP(std::shared_ptr<RdmaContext> ctx,
                      struct ibv_cq_ex** cq_ex, struct ibv_qp** qp,
                      ChannelMetaData* local_meta) = 0;

  // Connect QP to remote
  virtual void connectQP(struct ibv_qp* qp, std::shared_ptr<RdmaContext> ctx,
                         ChannelMetaData const& remote_meta,
                         struct ibv_recv_wr* pre_alloc_recv_wrs = nullptr,
                         uint32_t kMaxRecvWr = 0,
                         uint32_t* pending_post_recv = nullptr) = 0;

  // Poll completion queue
  virtual bool poll_once(struct ibv_cq_ex* cq_ex, std::vector<CQMeta>& cq_datas,
                         uint32_t channel_id) = 0;

  // Post receive work request
  virtual void lazy_post_recv_wr(struct ibv_qp* qp, uint32_t threshold,
                                 uint32_t& pending_post_recv,
                                 struct ibv_recv_wr* pre_alloc_recv_wrs,
                                 uint32_t kMaxRecvWr) = 0;

  // Setup Destination address
  virtual void setDstAddress(struct ibv_qp_ex* qpx, struct ibv_ah* ah,
                             uint32_t remote_qpn) = 0;

  // Get GID index
  virtual int getGidIndex() const = 0;

  // Get max inline data size
  virtual uint32_t getMaxInlineData() const = 0;

  // Initialize pre-allocated resources (IB specific)
  virtual void initPreAllocResources(struct ibv_recv_wr* pre_alloc_recv_wrs,
                                     uint32_t kMaxRecvWr) = 0;
};

// Forward declarations for implementations
#ifdef UCCL_P2P_USE_IB
class IBChannelImpl;
#else
class EFAChannelImpl;
#endif

// Factory function declaration (implementation in rdma_channel.h after
// including impl headers)
std::unique_ptr<RDMAChannelImpl> createRDMAChannelImpl();
