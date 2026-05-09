#pragma once
#include "define.h"
#include "rdma_context.h"
#include <infiniband/verbs.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Forward declarations
struct ibv_qp;
struct ibv_qp_ex;
struct ibv_cq_ex;
struct ibv_ah;
struct ibv_sge;
struct ibv_recv_wr;
struct ibv_wc;

static inline int get_sl_from_env(int default_value) {
  static int sl = -1;
  if (sl == -1) {
    char const* env = getenv("UCCL_P2P_RDMA_SL");
    if (env) {
      sl = std::atoi(env);
      if (sl < 0 || sl > 15) {
        UCCL_LOG(ERROR)
            << "A valid service level must be between 0 and 15, use "
               "default value "
            << default_value;
        sl = default_value;
      }
    } else
      sl = default_value;
  }

  return sl;
}

static inline int get_tc_from_env(int default_value) {
  static int tc = -1;
  if (tc == -1) {
    char const* env = getenv("UCCL_P2P_RDMA_TC");
    if (env) {
      tc = std::atoi(env);
      if (tc < 0 || tc > 255) {
        UCCL_LOG(ERROR)
            << "A valid traffic class must be between 0 and 255, use "
               "default value "
            << default_value;
        tc = default_value;
      }
    } else
      tc = default_value;
  }
  return tc;
}

// Base class for RDMA channel implementations
class RDMADataChannelImpl {
 public:
  virtual ~RDMADataChannelImpl() = default;

  // Initialize QP and CQ
  virtual void initQP(std::shared_ptr<RdmaContext> ctx,
                      struct ibv_cq_ex** cq_ex, struct ibv_qp** qp,
                      ChannelMetaData* local_meta) = 0;

  // Connect QP to remote
  virtual void connectQP(struct ibv_qp* qp, std::shared_ptr<RdmaContext> ctx,
                         ChannelMetaData const& remote_meta) = 0;

  // Poll completion queue
  virtual bool pollOnce(struct ibv_cq_ex* cq_ex, std::vector<CQMeta>& cq_datas,
                        uint32_t channel_id, uint32_t& nb_post_recv) = 0;

  // Post receive work request
  virtual void lazyPostRecvWrsN(struct ibv_qp* qp, uint32_t n, bool force) = 0;

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
class EFAChannelImpl;
class IBChannelImpl;

// Factory function declaration (defined in rdma_data_channel.h after
// including provider headers)
std::unique_ptr<RDMADataChannelImpl> createRDMADataChannelImpl();
