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

static inline int get_gid_index_from_env(int default_value) {
  static int gid_index = -1;
  if (gid_index == -1) {
    char const* env = getenv("UCCL_P2P_RDMA_GID_INDEX");
    if (env)
      gid_index = std::atoi(env);
    else
      gid_index = default_value;
  }
  return gid_index;
}

static inline int get_sl_from_env(int default_value) {
  static int sl = -1;
  if (sl == -1) {
    char const* env = getenv("UCCL_P2P_RDMA_SL");
    if (env) {
      sl = std::atoi(env);
      if (sl < 0 || sl > 15) {
        LOG(ERROR) << "A valid service level must be between 0 and 15, use "
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
        LOG(ERROR) << "A valid traffic class must be between 0 and 255, use "
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
